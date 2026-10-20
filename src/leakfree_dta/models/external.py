"""Re-running published models, with provenance.

The paper's re-evaluation rests on running other people's code, and the credibility
of that rests entirely on being able to say exactly what was run. So each entry
here records the repository, the commit, the command, and -- importantly -- what was
*changed* to make it run, because something almost always has to be.

Two rules the paper commits to, enforced by the fields below:

* **Published code is run as published.** Only the data loading and the split are
  replaced; the architecture, the optimiser and the hyperparameters stay as the
  authors set them. Re-tuning someone else's model and reporting a lower number
  would be measuring our tuning effort, not their contribution.
* **Failures are reported, not dropped.** A model that will not run is an entry
  with ``status="failed"`` and a reason. Dropping it would bias the re-measured
  trend upward, since the models hardest to reproduce are disproportionately the
  older ones -- exactly the ones that anchor the left end of the progress curve.

The inclusion criteria are stated once, here, so they cannot be adjusted after
seeing the results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from leakfree_dta.logging import get_logger
from leakfree_dta.utils.io import write_table

logger = get_logger(__name__)

#: Fixed in advance. A model enters the re-evaluation if and only if all hold.
INCLUSION_CRITERIA = (
    "Source code publicly available under a licence permitting re-use.",
    "Trained from scratch in under 24 GPU-hours on one 8 GB GPU, or a published "
    "checkpoint is available.",
    "Reports a drug-target affinity regression result on Davis, KIBA, BindingDB or "
    "a ChEMBL extract in its original paper.",
    "Published between 2018 and 2026, so the progress curve spans the period in "
    "question.",
)

#: Stated in advance as well, so an inconvenient model cannot be excluded later.
EXCLUSION_CRITERIA = (
    "Requires a proprietary dependency or a licence we do not hold.",
    "Requires experimentally determined 3D complex structures for every pair, "
    "which the ChEMBL-derived benchmark does not have. Note this excludes models "
    "needing *measured* structures, not ones that predict their own: a co-folding "
    "model is in scope, on a subset, and its coverage_scope says so.",
    "No usable inference entry point after one working day of effort, documented "
    "in the attempt log.",
)


@dataclass
class ExternalModelSpec:
    """Everything needed to re-run one published model, and to cite it honestly."""

    name: str
    year: int
    citation: str
    repository: str
    commit: str | None = None
    entrypoint: str = ""
    environment: str = ""
    #: Fraction of the benchmark this model is run on, and why if not all of it.
    #: A model evaluated on a subset is comparable to the others only on that
    #: subset, so the figure has to carry the caveat rather than the caption.
    coverage_scope: str = "full benchmark"
    reported_metric: str = "ci"
    reported_value: float | None = None
    reported_benchmark: str = ""
    reported_split: str = ""
    status: str = "not_attempted"  # "ok" | "failed" | "not_attempted" | "excluded"
    failure_reason: str = ""
    adaptations: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["adaptations"] = "; ".join(self.adaptations)
        return payload


#: The models the paper re-runs, in publication order so the registry reads as
#: the timeline the progress curve plots. Values marked ``None`` are filled in as the
#: re-evaluation proceeds; the specs are committed first so the target list is
#: fixed before any result is known.
REGISTRY: tuple[ExternalModelSpec, ...] = (
    ExternalModelSpec(
        name="DeepDTA",
        year=2018,
        citation="Ozturk et al., Bioinformatics 2018",
        repository="https://github.com/hkmztrk/DeepDTA",
        entrypoint="python run_experiments.py",
        environment="CNN over SMILES and sequence character embeddings.",
        reported_metric="ci",
        reported_benchmark="davis",
        reported_split="random 5-fold",
        notes="The reference point for the whole literature. Reports CI, not Pearson, "
        "so the progress curve needs both metrics recomputed on our splits.",
    ),
    ExternalModelSpec(
        name="GraphDTA",
        year=2020,
        citation="Nguyen et al., Bioinformatics 2021",
        repository="https://github.com/thinng/GraphDTA",
        entrypoint="python training.py",
        environment="GNN over the molecular graph, CNN over the sequence.",
        reported_metric="mse",
        reported_benchmark="davis",
        reported_split="random",
    ),
    ExternalModelSpec(
        name="MolTrans",
        year=2021,
        citation="Huang et al., Bioinformatics 2021",
        repository="https://github.com/kexinhuang12345/MolTrans",
        entrypoint="python train.py",
        environment="Substructure-level transformer with an interaction map.",
        reported_metric="auroc",
        reported_benchmark="bindingdb",
        reported_split="random",
        notes="Reports classification metrics; the regression head has to be used "
        "for a comparable number, which is an adaptation to record.",
    ),
    ExternalModelSpec(
        name="PerceiverCPI",
        year=2022,
        citation="Nguyen et al., Briefings in Bioinformatics 2023",
        repository="https://github.com/dmis-lab/PerceiverCPI",
        entrypoint="python train.py",
        environment="Cross-attention between a molecular encoder and a protein encoder.",
        reported_metric="rmse",
        reported_benchmark="davis",
        reported_split="random and cold-start",
    ),
    ExternalModelSpec(
        name="ChemBERTa-2 + ESM-2",
        year=2023,
        citation="Ahmad et al., 2022; Lin et al., Science 2023",
        repository="https://huggingface.co/DeepChem/ChemBERTa-77M-MLM",
        entrypoint="scripts/run_foundation_baseline.py",
        environment="Frozen foundation-model embeddings with a trained interaction head.",
        reported_metric="pearson",
        reported_benchmark="various",
        reported_split="various",
        notes="Stands in for the 'pretrained embeddings' family. Fast to run, which "
        "makes it a useful control for whether scale alone closes the gap.",
    ),
    ExternalModelSpec(
        name="Boltz-2",
        year=2025,
        citation="Passaro et al., 2025",
        repository="https://github.com/jwohlwend/boltz",
        entrypoint="boltz predict --affinity",
        environment="Co-folding model with an affinity head; predicts structure and "
        "affinity together.",
        reported_metric="pearson",
        reported_benchmark="FEP and internal sets",
        reported_split="various",
        coverage_scope=(
            "a stratified subset of a few thousand pairs, not the full benchmark"
        ),
        adaptations=[
            "Run on a subset. Boltz-2 predicts a structure per complex, at minutes "
            "of GPU time each, so scoring the full three-family extract is several "
            "GPU-years. The subset is stratified over protein cluster and affinity "
            "range so the comparison is fair within it.",
        ],
        notes="Named in the project plan and kept here rather than dropped. It is "
        "the only model in the list from after 2023, so without it the progress "
        "curve has no modern anchor and the paper's central question goes "
        "unanswered at the end that matters most. The scores are produced by the "
        "companion project (conformal-cofolding), which needs them anyway, and "
        "imported here; the two papers make different claims from the same "
        "computation and cite each other.",
    ),
)


def registry_table() -> pd.DataFrame:
    """The re-evaluation's target list as a table, for the paper's supplement."""
    return pd.DataFrame([spec.to_dict() for spec in REGISTRY])


def write_attempt_log(specs: list[ExternalModelSpec], path: str | Path) -> Path:
    """Record what was attempted and what happened.

    Published alongside the results. Its purpose is to let a reader see that the
    models which did not make it into the re-measured curve were excluded for
    stated technical reasons, not for producing inconvenient numbers.
    """
    table = pd.DataFrame([spec.to_dict() for spec in specs])
    written = write_table(table, Path(path))

    by_status = table["status"].value_counts().to_dict() if len(table) else {}
    logger.info("attempt log written to %s: %s", written.name, by_status)
    if by_status.get("ok", 0) < 5:
        logger.warning(
            "only %d models re-ran successfully; the progress curve needs enough points "
            "to support a trend claim",
            by_status.get("ok", 0),
        )
    return written


def criteria_markdown() -> str:
    """The inclusion and exclusion criteria, rendered for the methods section."""
    lines = ["### Inclusion criteria", ""]
    lines += [f"{index}. {text}" for index, text in enumerate(INCLUSION_CRITERIA, start=1)]
    lines += ["", "### Exclusion criteria", ""]
    lines += [f"{index}. {text}" for index, text in enumerate(EXCLUSION_CRITERIA, start=1)]
    lines += [
        "",
        "These criteria were fixed before any model was re-run. Every model meeting "
        "them appears in the attempt log, whether or not it produced a usable result.",
        "",
    ]
    return "\n".join(lines)
