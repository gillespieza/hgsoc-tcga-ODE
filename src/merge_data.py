"""
merge_data.py — Merge clinical and RNA-seq data into the final HGSOC cohort table.

Joins clinical_clean.csv (which carries the BRCA1/2 mutation flag added by
prepare_brca1_2_mutation.py) with rna_clean.csv on PATIENT_ID.  The result
is the canonical hgsoc_tcga_merged.csv consumed by every downstream script.

After the merge a 2x3 cohort summary figure is produced covering the complete
attrition waterfall (raw -> OS filter -> deduplication -> RNA merge), the
follow-up distribution, event rate, whole-cohort KM curve, and the BRCA1/2
mutation split.  The figure is produced here rather than in prepare_clinical.py
because only at this step are all four attrition counts and the BRCA flag
simultaneously available.

Outputs
-------
- data/processed/hgsoc_tcga_merged.csv
- results/figures/fig_cohort_summary.png
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; prevents tkinter errors on Windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


# =================================================================
# Attrition counts
# =================================================================

def _reconstruct_clinical_counts() -> tuple[int, int, int]:
    """
    Recompute intermediate clinical attrition counts for the waterfall panel.

    Applies the same three filters as prepare_clinical.py to the raw
    cBioPortal file so each waterfall bar reflects a real filter step.
    Reads the raw file for counting only -- does not reproduce or overwrite
    clinical_clean.csv.

    NOTE: the filtering logic here is intentionally duplicated from
    prepare_clinical.py.  The alternative -- writing intermediate counts to
    a sidecar JSON file -- would create a hidden dependency between two
    scripts.  A small counting duplication is preferable to invisible coupling.

    Returns
    -------
    n_raw : int
        Patients in the raw cBioPortal file.
    n_after_os_filter : int
        Patients after removing missing or invalid survival records.
    n_after_dedup : int
        Patients after removing duplicate PATIENT_ID entries.
    """
    raw = pd.read_csv(
        ROOT / "data" / "raw" / "clinical" / "data_clinical_patient.txt",
        sep="\t",
        comment="#",
    )
    n_raw = len(raw)

    # Mirror prepare_clinical.py: drop missing/invalid OS, then deduplicate.
    filtered = raw.dropna(subset=["OS_MONTHS", "OS_STATUS"])
    filtered = filtered[filtered["OS_MONTHS"] > 0]
    n_after_os_filter = len(filtered)

    deduped = filtered.drop_duplicates(subset=["PATIENT_ID"], keep="first")
    n_after_dedup = len(deduped)

    return n_raw, n_after_os_filter, n_after_dedup


# =================================================================
# Cohort summary figure
# =================================================================

def plot_cohort_summary(
    merged: pd.DataFrame,
    n_raw: int,
    n_after_os_filter: int,
    n_after_dedup: int,
) -> None:
    """
    Produce a 2x3 panel cohort summary figure for the final merged cohort.

    Panel layout
    ------------
    [0, 0] Attrition waterfall -- raw -> OS filter -> dedup -> RNA merge
    [0, 1] OS months histogram -- events vs censored
    [0, 2] Event rate bar chart
    [1, 0] Kaplan-Meier curve -- full merged cohort
    [1, 1] BRCA1/2 mutation status bar chart
    [1, 2] Hidden (layout placeholder)

    The BRCA panel is placed here rather than in prepare_clinical.py because
    BRCA_MUTANT is appended by prepare_brca1_2_mutation.py, which runs after
    prepare_clinical.py.  It first appears in the final cohort at this step.

    Parameters
    ----------
    merged : pd.DataFrame
        Final merged cohort.  Must contain OS_MONTHS, OS_EVENT, BRCA_MUTANT.
    n_raw : int
        Patient count from the raw cBioPortal file.
    n_after_os_filter : int
        Patient count after OS validity filter in prepare_clinical.py.
    n_after_dedup : int
        Patient count after deduplication in prepare_clinical.py.
    """
    n_merged = len(merged)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        "HGSOC TCGA -- Clinical Cohort Summary",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    # -----------------------------------------------------------------
    # [0, 0] Attrition waterfall
    # -----------------------------------------------------------------
    ax = axes[0, 0]

    stages = [
        "Raw (loaded)",
        "After OS filter",
        "After deduplication",
        "After RNA merge",
    ]
    counts = [n_raw, n_after_os_filter, n_after_dedup, n_merged]
    # Blue gradient: lightest for raw, darkest for the retained final cohort.
    colors = ["#90CAF9", "#64B5F6", "#42A5F5", "#1976D2"]

    bars = ax.barh(stages, counts, color=colors, edgecolor="white", height=0.5)

    # Annotate each bar with the patient count and the drop from the
    # previous stage so the reader can see where attrition is largest.
    prev = None
    for bar, count in zip(bars, counts):
        drop_str = ""
        if prev is not None:
            drop = prev - count
            drop_str = f"  (-{drop})" if drop > 0 else ""
        ax.text(
            count + n_raw * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{count}{drop_str}",
            va="center", ha="left", fontsize=10,
        )
        prev = count

    ax.set_xlim(0, n_raw * 1.22)
    ax.set_xlabel("Patient count", fontsize=10)
    ax.set_title("Cohort attrition", fontsize=11)
    ax.invert_yaxis()  # largest group at top, consistent with CONSORT style
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # -----------------------------------------------------------------
    # [0, 1] OS months histogram -- events vs censored
    # -----------------------------------------------------------------
    ax = axes[0, 1]

    events   = merged.loc[merged["OS_EVENT"] == 1, "OS_MONTHS"]
    censored = merged.loc[merged["OS_EVENT"] == 0, "OS_MONTHS"]

    # Shared bin edges so the two distributions are directly comparable.
    bin_edges = np.linspace(
        merged["OS_MONTHS"].min(), merged["OS_MONTHS"].max(), 31
    )

    ax.hist(
        events, bins=bin_edges, color="#d32f2f", alpha=0.7,
        label=f"Deceased (n={len(events)})",
    )
    ax.hist(
        censored, bins=bin_edges, color="#1976D2", alpha=0.7,
        label=f"Censored (n={len(censored)})",
    )

    ax.set_xlabel("Overall survival (months)", fontsize=10)
    ax.set_ylabel("Patient count", fontsize=10)
    ax.set_title("Follow-up distribution", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # -----------------------------------------------------------------
    # [0, 2] Event rate bar chart
    # -----------------------------------------------------------------
    ax = axes[0, 2]

    n_deceased = int(merged["OS_EVENT"].sum())
    n_living   = n_merged - n_deceased

    bar_labels = ["Deceased", "Living / censored"]
    bar_counts = [n_deceased, n_living]
    bar_colors = ["#d32f2f", "#1976D2"]

    rects = ax.bar(
        bar_labels, bar_counts,
        color=bar_colors, edgecolor="white", width=0.45,
    )

    for rect, count in zip(rects, bar_counts):
        pct = count / n_merged * 100
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + n_merged * 0.01,
            f"{count}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_ylabel("Patient count", fontsize=10)
    ax.set_title("Event rate", fontsize=11)
    ax.set_ylim(0, max(bar_counts) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # -----------------------------------------------------------------
    # [1, 0] KM curve -- full merged cohort
    # -----------------------------------------------------------------
    ax = axes[1, 0]

    kmf = KaplanMeierFitter()
    kmf.fit(
        merged["OS_MONTHS"],
        merged["OS_EVENT"],
        label=f"Full cohort (n={n_merged})",
    )
    kmf.plot_survival_function(
        ax=ax, ci_show=True,
        color="#1976D2", ci_alpha=0.15, linewidth=2,
    )

    median_os = kmf.median_survival_time_
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.axvline(
        median_os, color="grey", linestyle="--", linewidth=0.9, alpha=0.7,
        label=f"Median OS = {median_os:.1f} mo",
    )

    ax.set_xlabel("Time (months)", fontsize=10)
    ax.set_ylabel("Survival probability", fontsize=10)
    ax.set_title("Overall survival -- full cohort", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # -----------------------------------------------------------------
    # [1, 1] BRCA1/2 mutation status
    #
    # BRCA_MUTANT is added by prepare_brca1_2_mutation.py before merge_data.py
    # runs and is carried into the merged cohort via clinical_clean.csv.
    # Showing the split here confirms the flag survived the join and gives
    # the reader context for the small mutant group (n~28) ahead of the
    # downstream Cox and boxplot analyses.
    # -----------------------------------------------------------------
    ax = axes[1, 1]

    n_mutant   = int(merged["BRCA_MUTANT"].sum())
    n_wildtype = n_merged - n_mutant

    brca_labels = ["Wild type", "Mutant (BRCA1/2)"]
    brca_counts = [n_wildtype, n_mutant]
    brca_colors = ["#1976D2", "#d32f2f"]

    rects = ax.bar(
        brca_labels, brca_counts,
        color=brca_colors, edgecolor="white", width=0.45,
    )

    for rect, count in zip(rects, brca_counts):
        pct = count / n_merged * 100
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + n_merged * 0.01,
            f"{count}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_ylabel("Patient count", fontsize=10)
    ax.set_title("BRCA1/2 mutation status", fontsize=11)
    ax.set_ylim(0, max(brca_counts) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # -----------------------------------------------------------------
    # [1, 2] Hide unused panel -- keeps the 2x3 grid uniform
    # -----------------------------------------------------------------
    axes[1, 2].set_visible(False)

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    plt.tight_layout()

    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "fig_cohort_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"[FILE] Saved: .results/figures/fig_cohort_summary.png")


# =================================================================
# Main
# =================================================================

def main() -> None:
    """
    Merge clinical and RNA-seq data into the final HGSOC cohort table.

    Steps
    -----
    1. Load clinical_clean.csv (carries BRCA_MUTANT) and rna_clean.csv.
    2. Validate that the two cohorts share at least one PATIENT_ID.
    3. Inner join on PATIENT_ID to define the final analysis cohort.
    4. Run post-merge quality checks (shape, event rate, missing values).
    5. Save hgsoc_tcga_merged.csv.
    6. Reconstruct clinical attrition counts and produce the cohort figure.
    """

    # -----------------------------------------------------------------
    # Load inputs
    # -----------------------------------------------------------------

    clinical = pd.read_csv(ROOT / "data" / "processed" / "clinical_clean.csv")
    expr     = pd.read_csv(ROOT / "data" / "processed" / "rna_clean.csv")

    # -----------------------------------------------------------------
    # Basic integrity checks
    # -----------------------------------------------------------------

    clinical_ids = set(clinical["PATIENT_ID"])
    expr_ids     = set(expr["PATIENT_ID"])
    overlap      = clinical_ids & expr_ids

    logger.info("Cohort sizes before merge:")
    logger.info(f"    Clinical   : {len(clinical_ids)}")
    logger.info(f"    RNA        : {len(expr_ids)}")
    logger.info(f"    Overlap    : {len(overlap)}")

    if len(overlap) == 0:
        raise ValueError(
            "No overlap between clinical and RNA cohorts. "
            "Check PATIENT_ID formatting."
        )

    # -----------------------------------------------------------------
    # Merge
    # -----------------------------------------------------------------

    merged = clinical.merge(expr, on="PATIENT_ID", how="inner")

    logger.summary(f"Merged dataset shape: {merged.shape}")

    if len(merged) != len(overlap):
        logger.warning(
            f"Unexpected merge size: "
            f"expected {len(overlap)}, got {len(merged)}."
        )

    # -----------------------------------------------------------------
    # Post-merge validation
    # -----------------------------------------------------------------

    if "OS_EVENT" in merged.columns:
        logger.info(
            f"Event rate in merged cohort: {merged['OS_EVENT'].mean():.2%}"
        )

    missing_total = merged.isna().sum().sum()
    if missing_total > 0:
        logger.warning(f"Missing values after merge: {missing_total}")

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    out_path = ROOT / "data" / "processed" / "hgsoc_tcga_merged.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    logger.success("[FILE] Saved: .data/processed/hgsoc_tcga_merged.csv")

    # -----------------------------------------------------------------
    # Cohort summary figure
    #
    # Intermediate clinical counts are reconstructed by re-reading and
    # filtering the raw file.  See _reconstruct_clinical_counts() for
    # the rationale behind this deliberate duplication.
    # -----------------------------------------------------------------

    n_raw, n_after_os_filter, n_after_dedup = _reconstruct_clinical_counts()
    plot_cohort_summary(merged, n_raw, n_after_os_filter, n_after_dedup)


if __name__ == "__main__":
    main()