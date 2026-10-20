"""Figures.

Four figures carry the paper, and each one is a measurement rather than an
illustration:

1. **The leakage ladder.** Affected test fraction per channel, per split.
2. **The inflation curve.** Each model's score across the ladder.
3. **Coverage by group.** Mondrian versus global conformal calibration.
4. **Risk-coverage.** Selective prediction, which is what a screening group does.

Styling is deliberately plain -- no seaborn, no custom palette beyond what is
needed for legibility -- so the figures reproduce identically on any matplotlib
install and print correctly in greyscale.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on a cluster or in CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from leakfree_dta.logging import get_logger

logger = get_logger(__name__)

#: Colour-blind-safe and distinguishable in greyscale.
PALETTE = ("#0b5394", "#cc4125", "#38761d", "#a64d79", "#b45f06", "#45818e")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


#: Channels drawn in Figure 1, in order. The two ``pair``-axis channels come
#: first because they are the ones that make a test label recoverable by lookup;
#: the wider channels follow as context. Omitting
#: ``near_duplicate_pair_overlap`` would leave the figure without its headline.
LADDER_CHANNELS: tuple[str, ...] = (
    "pair_overlap",
    "near_duplicate_pair_overlap",
    "protein_identity_overlap",
    "ligand_identity_overlap",
    "near_duplicate_ligands",
    "scaffold_overlap",
)


def plot_leakage_ladder(
    table: pd.DataFrame, path: str | Path, *, channels: tuple[str, ...] = LADDER_CHANNELS
) -> Path | None:
    """Figure 1: affected test fraction per channel, as the split tightens."""
    present = [channel for channel in channels if channel in table.columns]
    if table.empty or not present:
        logger.warning("no leakage data to plot")
        return None

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(len(table))
    width = 0.8 / len(present)

    for index, channel in enumerate(present):
        values = pd.to_numeric(table[channel], errors="coerce").fillna(0.0).to_numpy() * 100
        # The two recoverable-by-lookup channels are drawn solid; the contextual
        # ones are hatched, so the distinction survives printing in greyscale.
        is_recoverable = channel in ("pair_overlap", "near_duplicate_pair_overlap")
        ax.bar(
            x + index * width - 0.4 + width / 2,
            values,
            width=width,
            label=channel.replace("_", " "),
            color=PALETTE[index % len(PALETTE)],
            hatch="" if is_recoverable else "///",
            edgecolor="white",
            linewidth=0.3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(table["split"], rotation=30, ha="right")
    ax.set_ylabel("test rows affected (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Leakage by channel, across the split ladder")
    # Below the axes: inside, the legend covers the tall bars that carry the point.
    ax.legend(
        frameon=False,
        fontsize=7,
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
    )
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def plot_inflation(table: pd.DataFrame, path: str | Path, *, metric: str = "pearson") -> Path | None:
    """Figure 2: each model's score across the ladder.

    The figure that answers the title. Lines that fall steeply were measuring
    leakage; lines that stay flat were measuring something real. Plotting the
    degenerate baselines on the same axes is what makes the comparison legible:
    a deep model whose line converges onto ``additive_bias`` at the strict end was
    never modelling interaction.
    """
    if table.empty or "model" not in table.columns:
        logger.warning("no inflation data to plot")
        return None

    split_columns = [
        column
        for column in table.columns
        if column != "model" and not column.startswith("drop_vs_random_")
    ]
    if not split_columns:
        return None

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(split_columns))

    for index, row in table.iterrows():
        values = pd.to_numeric(row[split_columns], errors="coerce").to_numpy(dtype=float)
        is_degenerate = row["model"] in {"global_mean", "target_mean", "ligand_mean", "additive_bias"}
        ax.plot(
            x,
            values,
            marker="o" if not is_degenerate else "s",
            linestyle="-" if not is_degenerate else "--",
            linewidth=1.6 if not is_degenerate else 1.2,
            label=row["model"],
            color=PALETTE[int(index) % len(PALETTE)],
            alpha=1.0 if not is_degenerate else 0.75,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(split_columns, rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} as the split tightens\n(dashed: predictors with no interaction term)")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def plot_group_coverage(
    coverage: pd.DataFrame, path: str | Path, *, target_coverage: float = 0.9
) -> Path | None:
    """Figure 3: per-group empirical coverage against the nominal level.

    The claim is not that overall coverage is correct -- global calibration gets
    that right too -- but that the *spread* across groups is narrower. So the
    figure shows every group, sorted, with the nominal level as a reference line.
    """
    groups = coverage[coverage["group"] != "__overall__"].copy()
    if groups.empty:
        return None

    groups = groups.sort_values("coverage").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.scatter(
        np.arange(len(groups)),
        groups["coverage"],
        s=np.clip(groups["n"] / max(groups["n"].max() / 60, 1), 6, 80),
        color=PALETTE[0],
        alpha=0.75,
        label="group (area proportional to size)",
    )
    ax.axhline(target_coverage, color=PALETTE[1], linestyle="--", linewidth=1.2, label=f"nominal {target_coverage:.0%}")
    overall = coverage[coverage["group"] == "__overall__"]
    if len(overall):
        ax.axhline(
            float(overall.iloc[0]["coverage"]),
            color=PALETTE[2],
            linestyle=":",
            linewidth=1.2,
            label="overall",
        )

    ax.set_xlabel("group, sorted by coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_ylim(0, 1.02)
    ax.set_title("Interval coverage per group")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def plot_risk_coverage(curves: dict[str, pd.DataFrame], path: str | Path) -> Path | None:
    """Figure 4: risk against coverage, one line per model."""
    if not curves:
        return None

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for index, (name, curve) in enumerate(curves.items()):
        if curve.empty:
            continue
        ax.plot(
            curve["coverage"],
            curve["risk"],
            marker="",
            linewidth=1.6,
            label=name,
            color=PALETTE[index % len(PALETTE)],
        )

    ax.set_xlabel("coverage (fraction of predictions retained)")
    ax.set_ylabel("RMSE on retained predictions")
    ax.set_title("Risk-coverage: does the uncertainty estimate help?")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def plot_progress_curve(table: pd.DataFrame, path: str | Path) -> Path | None:
    """The optional fifth figure: reported versus re-measured performance by year."""
    if table.empty or "year" not in table.columns:
        return None

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    reported = table.dropna(subset=["reported_value"])
    rerun = table.dropna(subset=["rerun_value"])

    ax.plot(
        reported["year"], reported["reported_value"], marker="o", color=PALETTE[0], label="as reported"
    )
    ax.plot(
        rerun["year"], rerun["rerun_value"], marker="s", color=PALETTE[1], label="re-measured, leakage-free"
    )

    if {"rerun_ci_lower", "rerun_ci_upper"} <= set(rerun.columns):
        ax.fill_between(
            rerun["year"],
            rerun["rerun_ci_lower"],
            rerun["rerun_ci_upper"],
            color=PALETTE[1],
            alpha=0.18,
            linewidth=0,
        )

    for row in table.itertuples(index=False):
        if getattr(row, "status", "ok") == "failed":
            ax.annotate(
                f"{row.model}\n(not reproducible)",
                (row.year, ax.get_ylim()[0]),
                fontsize=6,
                ha="center",
                color="grey",
            )

    ax.set_xlabel("publication year")
    ax.set_ylabel("Pearson correlation")
    ax.set_title("How much of the reported progress survives re-measurement?")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path)
    plt.close(fig)
    return Path(path)
