"""Command line interface.

The pipeline is a sequence of commands rather than one script, because each stage
is slow and the later ones are iterated on far more than the earlier ones. Running
them in order reproduces every number in the paper:

    leakfree curate   --source chembl --input data/raw/chembl_kinase.parquet
    leakfree audit    --dataset chembl_3fam
    leakfree split    --dataset chembl_3fam --ladder
    leakfree audit    --dataset chembl_3fam --split leakfree
    leakfree train    --dataset chembl_3fam --split leakfree --model shift_robust
    leakfree evaluate --dataset chembl_3fam --split leakfree
    leakfree report   --dataset chembl_3fam

``leakfree demo`` runs the whole sequence on the synthetic calibration dataset in
about a minute, which is the fastest way to see what the pipeline produces and to
check an installation.

Built on ``argparse`` deliberately: the package should install and run with nothing
beyond its scientific dependencies.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from leakfree_dta import paths
from leakfree_dta._version import __version__
from leakfree_dta.logging import get_logger, setup_logging
from leakfree_dta.utils.io import read_table, write_json, write_table

logger = get_logger(__name__)


# --- helpers ----------------------------------------------------------------


def _dataset_path(dataset: str, *, aggregated: bool = True) -> Path:
    suffix = "aggregated" if aggregated else "replicates"
    return paths.PROCESSED / f"{dataset}.{suffix}.parquet"


def _load_dataset(dataset: str, *, aggregated: bool = True) -> pd.DataFrame:
    path = _dataset_path(dataset, aggregated=aggregated)
    if not path.exists():
        raise SystemExit(
            f"dataset not found: {path}\nRun `leakfree curate` first, or `leakfree demo` "
            f"to build the synthetic one."
        )
    return read_table(path)


def _split_dir(dataset: str) -> Path:
    return paths.SPLITS / dataset


# --- commands ---------------------------------------------------------------


def cmd_curate(args: argparse.Namespace) -> int:
    """Load a source, curate it, and write both the replicate and aggregated tables."""
    from leakfree_dta.data.curate import curate
    from leakfree_dta.data.sources import LOADERS

    if args.source not in LOADERS:
        raise SystemExit(f"unknown source {args.source!r}; available: {sorted(LOADERS)}")

    logger.info("loading %s from %s", args.source, args.input)
    raw = LOADERS[args.source](args.input)

    if args.assign_families:
        from leakfree_dta.prot.families import assign_families, load_overrides

        overrides = load_overrides(args.family_overrides)
        targets = raw.drop_duplicates(subset=["target_key"])[
            [column for column in ("target_key", "protein_class", "uniprot_keywords", "target_sequence") if column in raw.columns]
        ]
        assigned = assign_families(targets, overrides=overrides)
        raw = raw.merge(
            assigned[["target_key", "target_family", "family_rule", "family_confidence"]],
            on="target_key",
            how="left",
        )

    # The audit needs replicates; training needs them collapsed. Both are kept so
    # the noise floor is measured on the data the models actually see.
    replicates, log = curate(raw, aggregate=None, scope_families=args.scope_families)
    aggregated, _ = curate(raw, aggregate=args.aggregate, scope_families=args.scope_families)

    paths.ensure_dirs()
    write_table(replicates, _dataset_path(args.dataset, aggregated=False))
    write_table(aggregated, _dataset_path(args.dataset, aggregated=True))
    write_table(log.to_frame(), paths.TABLES / f"{args.dataset}.curation_log.csv")

    from leakfree_dta.data import schema

    summary = {
        "replicates": schema.summarize(replicates),
        "aggregated": schema.summarize(aggregated),
    }
    write_json(summary, paths.TABLES / f"{args.dataset}.dataset_summary.json")

    print(f"\ncurated `{args.dataset}`:")
    print(f"  with replicates : {len(replicates):,} rows -> {_dataset_path(args.dataset, aggregated=False)}")
    print(f"  aggregated      : {len(aggregated):,} rows -> {_dataset_path(args.dataset, aggregated=True)}")
    print(f"  curation log    : {paths.TABLES / f'{args.dataset}.curation_log.csv'}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Audit a dataset, or a particular split of it."""
    from leakfree_dta.audit.runner import (
        audit_dataset,
        audit_split_by_family,
        comparison_table,
        dominant_channel,
    )

    if args.split is None:
        df = _load_dataset(args.dataset, aggregated=False)
        report = audit_dataset(df, dataset=args.dataset, skip=args.skip)
        out = paths.AUDIT / args.dataset / "dataset"
        report.save(out)
        print("\n" + report.to_markdown())
        print(f"\nwritten to {out}")
        return 0 if not report.failures else 1

    from leakfree_dta.splits.manifest import apply_manifest, load_manifest

    df = _load_dataset(args.dataset, aggregated=True)
    manifest_path = _split_dir(args.dataset) / f"{args.split}.json"
    if not manifest_path.exists():
        raise SystemExit(f"no split manifest at {manifest_path}; run `leakfree split` first")

    assignment = apply_manifest(df, load_manifest(manifest_path), allow_partial=True)
    train, test = assignment.train(df), assignment.test(df)

    reports = audit_split_by_family(
        train, test, dataset=args.dataset, split=args.split, skip=args.skip
    )
    out = paths.AUDIT / args.dataset / args.split
    for label, report in reports.items():
        report.save(out / label)

    table = comparison_table(reports)
    if not table.empty:
        write_table(table, paths.TABLES / f"{args.dataset}.{args.split}.leakage_by_family.csv")

    print("\n" + reports["overall"].to_markdown())
    print("\n## Dominant leakage channel by family\n")
    for label, report in reports.items():
        axis, fraction = dominant_channel(report)
        print(f"  {label:12s} {axis:12s} {fraction:7.2%}")
    print(f"\nwritten to {out}")
    return 0 if not reports["overall"].failures else 1


def cmd_split(args: argparse.Namespace) -> int:
    """Build split manifests."""
    from leakfree_dta.splits.base import SplitSpec
    from leakfree_dta.splits.manifest import manifest_summary, save_manifest
    from leakfree_dta.splits.three_axis import SPLIT_LADDER, build_split

    df = _load_dataset(args.dataset, aggregated=True)
    directory = _split_dir(args.dataset)

    if args.ladder:
        specs = SPLIT_LADDER
    else:
        specs = (
            SplitSpec(
                name=args.name,
                ligand_axis=args.ligand_axis,
                protein_axis=args.protein_axis,
                temporal_axis=args.temporal_axis,
                test_fraction=args.test_fraction,
                valid_fraction=args.valid_fraction,
                protein_identity=args.protein_identity,
                temporal_cutoff_year=args.temporal_cutoff_year,
                seed=args.seed,
            ),
        )

    for spec in specs:
        try:
            assignment = build_split(df, spec)
        except ValueError as exc:
            logger.warning("split %r could not be built: %s", spec.name, exc)
            continue
        save_manifest(assignment, df, dataset=args.dataset, directory=directory)

    summary = manifest_summary(directory)
    if not summary.empty:
        write_table(summary, paths.TABLES / f"{args.dataset}.splits.csv")
        print("\n" + summary.to_string(index=False))
    print(f"\nmanifests in {directory}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Fit one model on one split and write its predictions."""
    from leakfree_dta.models.registry import available_models, build_model
    from leakfree_dta.splits.manifest import apply_manifest, load_manifest

    if args.model not in available_models():
        raise SystemExit(f"unknown model {args.model!r}; available: {available_models()}")

    df = _load_dataset(args.dataset, aggregated=True)
    manifest = load_manifest(_split_dir(args.dataset) / f"{args.split}.json")
    assignment = apply_manifest(df, manifest, allow_partial=True)

    train, valid, test = assignment.train(df), assignment.valid(df), assignment.test(df)
    if not len(train) or not len(test):
        raise SystemExit(f"split {args.split!r} has an empty fold: {assignment.counts()}")

    logger.info(
        "training %s on %s/%s (train=%d valid=%d test=%d)",
        args.model,
        args.dataset,
        args.split,
        len(train),
        len(valid),
        len(test),
    )

    model = build_model(args.model)
    fit_result = model.fit(train, valid if len(valid) else None)
    prediction = model.predict(test)

    from leakfree_dta.data import schema

    output = pd.DataFrame(
        {
            schema.ROW_KEY: test[schema.ROW_KEY].to_numpy(),
            schema.TARGET_KEY: test[schema.TARGET_KEY].to_numpy(),
            schema.LIGAND_KEY: test[schema.LIGAND_KEY].to_numpy(),
            schema.TARGET_FAMILY: test[schema.TARGET_FAMILY].to_numpy(),
            "y_true": model.targets(test),
            "y_pred": prediction.mean,
        }
    )
    for name in ("std", "lower", "upper"):
        value = getattr(prediction, name)
        if value is not None:
            output[f"y_{name}"] = value
    for group_column in ("ligand_group", "protein_group"):
        if group_column in test.columns:
            output[group_column] = test[group_column].to_numpy()

    run_dir = paths.RUNS / args.dataset / args.split / args.model
    write_table(output, run_dir / "predictions.parquet")
    write_json(
        {
            "model": model.describe(),
            "fit": {
                "n_train": fit_result.n_train,
                "n_valid": fit_result.n_valid,
                "best_epoch": fit_result.best_epoch,
                "notes": fit_result.notes,
            },
            "prediction_metadata": prediction.metadata,
            "split": manifest["spec"],
            "split_digest": manifest.get("spec_digest"),
        },
        run_dir / "run.json",
    )
    if fit_result.history:
        write_table(pd.DataFrame(fit_result.history), run_dir / "history.csv")

    from leakfree_dta.eval.metrics import evaluate

    metrics = evaluate(
        output["y_true"].to_numpy(),
        output["y_pred"].to_numpy(),
        y_std=output.get("y_std"),
        lower=output.get("y_lower"),
        upper=output.get("y_upper"),
    )
    write_json(metrics.to_dict(), run_dir / "metrics.json")

    print(f"\n{args.model} on {args.dataset}/{args.split}:")
    for key, value in metrics.to_dict().items():
        print(f"  {key:16s} {value}")
    print(f"\nwritten to {run_dir}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Compare every model trained on a split, with cluster-bootstrap intervals."""
    from leakfree_dta.eval.bootstrap import bootstrap_table, compare_models
    from leakfree_dta.eval.metrics import METRICS
    from leakfree_dta.eval.per_group import group_report
    from leakfree_dta.eval.risk_coverage import selective_prediction_summary

    split_dir = paths.RUNS / args.dataset / args.split
    if not split_dir.exists():
        raise SystemExit(f"no runs found at {split_dir}; run `leakfree train` first")

    runs = {
        path.parent.name: read_table(path) for path in sorted(split_dir.glob("*/predictions.parquet"))
    }
    if not runs:
        raise SystemExit(f"no predictions under {split_dir}")

    logger.info("evaluating %d models: %s", len(runs), sorted(runs))

    # Every model must have predicted the same rows, or the comparison is invalid.
    reference = next(iter(runs.values()))
    from leakfree_dta.data import schema

    reference_keys = set(reference[schema.ROW_KEY])
    for name, frame in runs.items():
        if set(frame[schema.ROW_KEY]) != reference_keys:
            raise SystemExit(
                f"model {name!r} predicted a different set of test rows; the models are "
                "not comparable. Re-run them on the same split."
            )

    aligned = {
        name: frame.set_index(schema.ROW_KEY).loc[list(reference[schema.ROW_KEY])]
        for name, frame in runs.items()
    }
    y_true = reference.set_index(schema.ROW_KEY).loc[list(reference[schema.ROW_KEY])]["y_true"].to_numpy()
    predictions = {name: frame["y_pred"].to_numpy() for name, frame in aligned.items()}

    # The bootstrap resamples whichever unit the dependence between rows runs
    # along. Protein clusters are the first choice; a split that constrains only
    # the ligand axis carries no protein_group, so ligand clusters come next, and
    # individual targets after that.
    #
    # Target *family* is deliberately not in this list. With three families it
    # gives a three-group bootstrap, whose interval is driven by which of three
    # blocks was drawn and is not an estimate of anything.
    group_column = next(
        (
            column
            for column in ("protein_group", "ligand_group", schema.TARGET_KEY)
            if column in reference.columns
        ),
        None,
    )
    groups = (
        reference.set_index(schema.ROW_KEY).loc[list(reference[schema.ROW_KEY])][group_column].to_numpy()
        if group_column
        else None
    )
    if groups is None:
        logger.warning("no grouping column; falling back to the row bootstrap, which is too narrow")
    else:
        n_groups = len(set(groups.tolist()))
        logger.info("cluster bootstrap over %d %s groups", n_groups, group_column)
        if n_groups < 20:
            logger.warning(
                "only %d bootstrap groups; the intervals in this table are unstable and "
                "should not be used to call small differences",
                n_groups,
            )

    table = bootstrap_table(
        y_true, predictions, METRICS, groups=groups, n_bootstrap=args.n_bootstrap
    )
    write_table(table, paths.TABLES / f"{args.dataset}.{args.split}.metrics.csv")

    # Paired comparisons against the strongest degenerate baseline. If a model
    # cannot beat `additive_bias`, it has not demonstrated interaction modelling.
    reference_model = args.against if args.against in predictions else None
    comparisons: list[dict[str, Any]] = []
    if reference_model:
        for name, y_pred in predictions.items():
            if name == reference_model:
                continue
            for metric_name in ("rmse", "pearson"):
                result = compare_models(
                    y_true,
                    y_pred,
                    predictions[reference_model],
                    METRICS[metric_name],
                    metric_name=metric_name,
                    groups=groups,
                    n_bootstrap=args.n_bootstrap,
                    higher_is_better=metric_name != "rmse",
                )
                comparisons.append({"model": name, "vs": reference_model, **result})
        write_table(
            pd.DataFrame(comparisons),
            paths.TABLES / f"{args.dataset}.{args.split}.comparisons.csv",
        )

    group_rows: list[pd.DataFrame] = []
    for name, frame in aligned.items():
        group_sets: dict[str, Any] = {}
        if schema.TARGET_FAMILY in frame.columns:
            group_sets["family"] = frame[schema.TARGET_FAMILY].to_numpy()
        if "protein_group" in frame.columns:
            group_sets["protein_cluster"] = frame["protein_group"].to_numpy()
        if group_sets:
            report = group_report(y_true, frame["y_pred"].to_numpy(), group_sets)
            report.insert(0, "model", name)
            group_rows.append(report)
    if group_rows:
        write_table(
            pd.concat(group_rows, ignore_index=True),
            paths.TABLES / f"{args.dataset}.{args.split}.per_group.csv",
        )

    selective: list[dict[str, Any]] = []
    for name, frame in aligned.items():
        if "y_std" not in frame.columns:
            continue
        summary = selective_prediction_summary(
            y_true, frame["y_pred"].to_numpy(), -frame["y_std"].to_numpy()
        )
        selective.append({"model": name, **summary})
    if selective:
        write_table(
            pd.DataFrame(selective),
            paths.TABLES / f"{args.dataset}.{args.split}.selective.csv",
        )

    pivot = table.pivot(index="model", columns="metric", values="estimate")
    print(f"\n{args.dataset} / {args.split} -- point estimates\n")
    print(pivot.to_string())
    print(f"\nintervals and per-group tables in {paths.TABLES}")
    if comparisons:
        significant = [row for row in comparisons if row["significant"]]
        print(
            f"\n{len(significant)}/{len(comparisons)} comparisons against "
            f"`{reference_model}` are significant at the 95% level "
            f"(cluster bootstrap over {table['n_units'].iloc[0]} groups)."
        )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Assemble the paper's tables and figures from what has been computed."""
    from leakfree_dta.report.paper import build_report

    written = build_report(args.dataset, splits=args.splits)
    print(f"\nreport assembled for `{args.dataset}`:")
    for path in written:
        print(f"  {path}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Run the whole pipeline on the synthetic calibration dataset.

    Also the installation check: if this completes, every dependency the core
    pipeline needs is present and working.
    """
    from leakfree_dta.audit.runner import audit_dataset, audit_split, dominant_channel
    from leakfree_dta.data.synthetic import make_synthetic_dataset
    from leakfree_dta.splits.manifest import save_manifest
    from leakfree_dta.splits.three_axis import SPLIT_LADDER, build_split

    paths.ensure_dirs()
    dataset = args.dataset

    print("1/5  building the synthetic dataset with planted leakage")
    df, truth = make_synthetic_dataset(seed=args.seed)
    write_table(df, _dataset_path(dataset, aggregated=False))

    from leakfree_dta.data.curate import aggregate_replicates

    aggregated = aggregate_replicates(df)
    write_table(aggregated, _dataset_path(dataset, aggregated=True))
    print(
        f"     {len(df):,} rows ({len(aggregated):,} after aggregation), "
        f"{truth.n_ligands} ligands, {truth.n_targets} targets"
    )
    print(f"     planted: {len(truth.conflicting_pair_keys)} conflicting pairs, "
          f"{len(truth.analog_ligand_keys)} analogue ligands, "
          f"{len(truth.paralogue_target_keys)} paralogue target pairs")
    print(f"     shortcut ceiling by construction: Pearson {truth.shortcut_ceiling:.3f}")

    print("\n2/5  auditing the dataset itself")
    dataset_report = audit_dataset(df, dataset=dataset)
    dataset_report.save(paths.AUDIT / dataset / "dataset")
    for result in dataset_report.results:
        print(f"     {result.name:28s} {result.fraction:7.2%}  {result.severity.value}")

    print("\n3/5  building the split ladder")
    ladder = [spec for spec in SPLIT_LADDER if spec.name in args.splits] if args.splits else list(SPLIT_LADDER)
    built: dict[str, Any] = {}
    for spec in ladder:
        try:
            assignment = build_split(aggregated, spec)
        except ValueError as exc:
            print(f"     {spec.name:20s} skipped: {exc}")
            continue
        if assignment.is_degenerate(min_rows=20):
            print(f"     {spec.name:20s} degenerate: {assignment.counts()}")
            continue
        save_manifest(assignment, aggregated, dataset=dataset, directory=_split_dir(dataset))
        built[spec.name] = assignment
        counts = assignment.counts()
        print(
            f"     {spec.name:20s} train={counts['train']:>5} valid={counts['valid']:>5} "
            f"test={counts['test']:>5} excluded={counts['excluded']:>5}"
        )

    print("\n4/5  auditing each split")
    rows: list[dict[str, Any]] = []
    for name, assignment in built.items():
        train, test = assignment.train(aggregated), assignment.test(aggregated)
        report = audit_split(
            train, test, dataset=dataset, split=name, skip=("protein_identity_overlap",) if args.fast else ()
        )
        report.save(paths.AUDIT / dataset / name)
        axis, fraction = dominant_channel(report)
        rows.append(
            {
                "split": name,
                "dominant_channel_fraction": round(fraction, 4),
                "n_train": len(train),
                "n_test": len(test),
                "recoverable_fraction": round(report.recoverable_fraction(), 4),
                "dominant_channel": axis,
                "worst_severity": report.worst_severity.value,
                **{
                    result.name: round(result.fraction, 4)
                    for result in report.results
                    if result.axis in ("pair", "ligand", "protein", "shortcut")
                },
            }
        )
    ladder_table = pd.DataFrame(rows)
    write_table(ladder_table, paths.TABLES / f"{dataset}.ladder.csv")
    print()
    print(
        ladder_table[["split", "n_train", "n_test", "recoverable_fraction", "dominant_channel"]].to_string(index=False)
    )

    print("\n5/5  training the baselines on the strictest usable split")
    strictest = rows[-1]["split"] if rows else None
    if strictest:
        from leakfree_dta.eval.metrics import evaluate
        from leakfree_dta.models.registry import build_model

        assignment = built[strictest]
        train, valid, test = (
            assignment.train(aggregated),
            assignment.valid(aggregated),
            assignment.test(aggregated),
        )
        results: list[dict[str, Any]] = []
        for name in args.models:
            try:
                model = build_model(name)
                model.fit(train, valid if len(valid) >= 10 else None)
                prediction = model.predict(test)
            except Exception as exc:  # a demo should report, not crash
                print(f"     {name:20s} failed: {type(exc).__name__}: {exc}")
                continue
            metrics = evaluate(model.targets(test), prediction.mean)
            results.append({"model": name, **metrics.to_dict()})
            print(
                f"     {name:20s} rmse={metrics.get('rmse'):.4f}  "
                f"pearson={metrics.get('pearson'):.4f}  spearman={metrics.get('spearman'):.4f}"
            )
        if results:
            write_table(pd.DataFrame(results), paths.TABLES / f"{dataset}.demo_models.csv")

    print(f"\ndone. artifacts under {paths.RESULTS} and {paths.SPLITS}")
    print(
        "\nThe point of the ladder table: leakage falls as constraints are added, and "
        "the models' scores fall with it. The gap between `random` and `leakfree` on "
        "the same data is the inflation this paper measures."
    )
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Print the environment, so a bug report can say what was running."""
    import platform

    from leakfree_dta.models.registry import available_models
    from leakfree_dta.prot.identity import identity_backend

    lines = [
        f"leakfree-dta      {__version__}",
        f"python            {platform.python_version()} ({platform.platform()})",
        f"project root      {paths.ROOT}",
        f"identity backend  {identity_backend()}",
        "",
        "dependencies:",
    ]
    for module in ("numpy", "pandas", "scipy", "sklearn", "rdkit", "torch", "xgboost"):
        try:
            imported = __import__(module)
            version = getattr(imported, "__version__", "?")
            extra = ""
            if module == "torch":
                extra = f" (cuda {'available' if imported.cuda.is_available() else 'unavailable'})"
            lines.append(f"  {module:12s} {version}{extra}")
        except ImportError:
            lines.append(f"  {module:12s} not installed")

    lines += ["", f"models available: {', '.join(available_models())}"]

    if identity_backend() != "mmseqs2":
        lines += [
            "",
            "NOTE: MMseqs2 is not on PATH, so protein clustering uses the approximate",
            "      k-mer + Smith-Waterman fallback. Install MMseqs2 before producing",
            "      numbers for publication.",
        ]

    print("\n".join(lines))
    return 0


# --- parser -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="leakfree",
        description="Leakage-audited benchmark and shift-robust baseline for drug-target affinity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"leakfree-dta {__version__}")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity (default: INFO)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # curate
    curate_parser = subparsers.add_parser("curate", help="load and curate a source into the canonical schema")
    curate_parser.add_argument("--source", required=True, help="chembl | bindingdb | davis | kiba | pdbbind")
    curate_parser.add_argument("--input", required=True, help="path to the raw file or directory")
    curate_parser.add_argument("--dataset", required=True, help="name to store the curated dataset under")
    curate_parser.add_argument("--aggregate", default="median", choices=["median", "mean", "first", "max"])
    curate_parser.add_argument("--no-scope-families", dest="scope_families", action="store_false")
    curate_parser.add_argument("--no-assign-families", dest="assign_families", action="store_false")
    curate_parser.add_argument(
        "--family-overrides",
        default=str(paths.CONFIGS / "data" / "target_families.csv"),
        help="curated target_key,target_family CSV",
    )
    curate_parser.set_defaults(func=cmd_curate, scope_families=True, assign_families=True)

    # audit
    audit_parser = subparsers.add_parser("audit", help="audit a dataset or one of its splits")
    audit_parser.add_argument("--dataset", required=True)
    audit_parser.add_argument("--split", default=None, help="omit to audit the dataset itself")
    audit_parser.add_argument("--skip", nargs="*", default=(), help="check names to skip")
    audit_parser.set_defaults(func=cmd_audit)

    # split
    split_parser = subparsers.add_parser("split", help="build split manifests")
    split_parser.add_argument("--dataset", required=True)
    split_parser.add_argument("--ladder", action="store_true", help="build every split in the paper's ladder")
    split_parser.add_argument("--name", default="custom")
    split_parser.add_argument("--ligand-axis", default="random")
    split_parser.add_argument("--protein-axis", default="random")
    split_parser.add_argument("--temporal-axis", default="none")
    split_parser.add_argument("--test-fraction", type=float, default=0.2)
    split_parser.add_argument("--valid-fraction", type=float, default=0.1)
    split_parser.add_argument("--protein-identity", type=float, default=0.30)
    split_parser.add_argument("--temporal-cutoff-year", type=int, default=None)
    split_parser.add_argument("--seed", type=int, default=0)
    split_parser.set_defaults(func=cmd_split)

    # train
    train_parser = subparsers.add_parser("train", help="fit one model on one split")
    train_parser.add_argument("--dataset", required=True)
    train_parser.add_argument("--split", required=True)
    train_parser.add_argument("--model", required=True)
    train_parser.set_defaults(func=cmd_train)

    # evaluate
    evaluate_parser = subparsers.add_parser("evaluate", help="compare every model trained on a split")
    evaluate_parser.add_argument("--dataset", required=True)
    evaluate_parser.add_argument("--split", required=True)
    evaluate_parser.add_argument("--n-bootstrap", type=int, default=1000)
    evaluate_parser.add_argument(
        "--against",
        default="additive_bias",
        help="baseline for the paired comparisons (default: additive_bias)",
    )
    evaluate_parser.set_defaults(func=cmd_evaluate)

    # report
    report_parser = subparsers.add_parser("report", help="assemble the paper's tables and figures")
    report_parser.add_argument("--dataset", required=True)
    report_parser.add_argument("--splits", nargs="*", default=None)
    report_parser.set_defaults(func=cmd_report)

    # demo
    demo_parser = subparsers.add_parser("demo", help="run the whole pipeline on synthetic data")
    demo_parser.add_argument("--dataset", default="synthetic")
    demo_parser.add_argument("--seed", type=int, default=0)
    demo_parser.add_argument("--splits", nargs="*", default=None, help="limit to these ladder splits")
    demo_parser.add_argument(
        "--models",
        nargs="*",
        default=["global_mean", "target_mean", "ligand_mean", "additive_bias", "ridge_ecfp"],
    )
    demo_parser.add_argument(
        "--fast", action="store_true", help="skip the slow pairwise sequence alignment check"
    )
    demo_parser.set_defaults(func=cmd_demo)

    # info
    info_parser = subparsers.add_parser("info", help="print the environment")
    info_parser.set_defaults(func=cmd_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
