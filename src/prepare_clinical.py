"""
prepare_clinical.py — Clean TCGA clinical data and produce a cohort summary figure.

Steps:
1. Load cBioPortal clinical patient table.
2. Remove missing or invalid survival records.
3. Deduplicate patient entries.
4. Convert survival status into a binary event variable (OS_EVENT).
5. Save cleaned dataset for downstream modelling.
6. Produce a 2x2 cohort summary figure:
   - Attrition waterfall (patient counts at each filter stage)
   - OS months histogram (events vs censored)
   - Event rate bar chart (deceased vs living)
   - Kaplan-Meier curve for the full cleaned cohort

Outputs
-------
- data/processed/clinical_clean.csv
- results/figures/fig_cohort_summary.png
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; required for headless pipeline runs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


# =================================================================
# Cohort summary figure
# =================================================================

def plot_cohort_summary(
    clinical_clean: pd.DataFrame,
    n_raw: int,
    n_after_os_filter: int,
    n_after_dedup: int,
) -> None:
    """
    Produce a 2x2 panel summarising the cleaned clinical cohort.

    Intended for inclusion in the Methods section of the report. The four
    panels collectively answer: how many patients remain after each filter,
    what does the follow-up distribution look like, what is the event rate,
    and what is the overall survival trajectory?

    Parameters
    ----------
    clinical_clean : pd.DataFrame
        Cleaned clinical table with OS_MONTHS, OS_EVENT, and PATIENT_ID.
    n_raw : int
        Patient count loaded from the raw cBioPortal file.
    n_after_os_filter : int
        Patient count after removing missing/invalid survival records.
    n_after_dedup : int
        Patient count after deduplication (equal to len(clinical_clean)).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "HGSOC TCGA — Clinical Cohort Summary",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    # -----------------------------------------------------------------
    # Top-left: attrition waterfall
    # -----------------------------------------------------------------
    ax = axes[0, 0]

    stages = [
        "Raw (loaded)",
        "After OS filter",
        "After deduplication",
    ]
    counts = [n_raw, n_after_os_filter, n_after_dedup]
    colors = ["#90CAF9", "#64B5F6", "#1976D2"]

    bars = ax.barh(
        stages, counts,
        color=colors, edgecolor="white", height=0.5,
    )

    # Annotate each bar with the exact patient count and the drop from
    # the previous stage so the reader can immediately see where attrition occurs.
    prev = None
    for bar, count in zip(bars, counts):
        drop_str = ""
        if prev is not None:
            drop = prev - count
            drop_str = f"  (−{drop})" if drop > 0 else ""
        ax.text(
            count + n_raw * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{count}{drop_str}",
            va="center", ha="left", fontsize=10,
        )
        prev = count

    ax.set_xlim(0, n_raw * 1.18)
    ax.set_xlabel("Patient count", fontsize=10)
    ax.set_title("Cohort attrition", fontsize=11)
    ax.invert_yaxis()  # largest stage at the top, matching standard CONSORT style
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # -----------------------------------------------------------------
    # Top-right: OS months histogram split by event/censored
    # -----------------------------------------------------------------
    ax = axes[0, 1]

    events   = clinical_clean.loc[clinical_clean["OS_EVENT"] == 1, "OS_MONTHS"]
    censored = clinical_clean.loc[clinical_clean["OS_EVENT"] == 0, "OS_MONTHS"]

    # Use a shared bin range so the two distributions are directly comparable.
    bin_edges = np.linspace(
        clinical_clean["OS_MONTHS"].min(),
        clinical_clean["OS_MONTHS"].max(),
        31,
    )

    ax.hist(
        events, bins=bin_edges,
        color="#d32f2f", alpha=0.7, label=f"Deceased (n={len(events)})",
    )
    ax.hist(
        censored, bins=bin_edges,
        color="#1976D2", alpha=0.7, label=f"Censored (n={len(censored)})",
    )

    ax.set_xlabel("Overall survival (months)", fontsize=10)
    ax.set_ylabel("Patient count", fontsize=10)
    ax.set_title("Follow-up distribution", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # -----------------------------------------------------------------
    # Bottom-left: event rate bar chart
    # -----------------------------------------------------------------
    ax = axes[1, 0]

    n_deceased = int(clinical_clean["OS_EVENT"].sum())
    n_living   = len(clinical_clean) - n_deceased

    bar_labels  = ["Deceased", "Living / censored"]
    bar_counts  = [n_deceased, n_living]
    bar_colors  = ["#d32f2f", "#1976D2"]

    rects = ax.bar(
        bar_labels, bar_counts,
        color=bar_colors, edgecolor="white", width=0.45,
    )

    # Annotate each bar with the count and percentage of the total cohort.
    total = len(clinical_clean)
    for rect, count in zip(rects, bar_counts):
        pct = count / total * 100
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + total * 0.01,
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
    # Bottom-right: Kaplan-Meier curve (full cohort, no stratification)
    # -----------------------------------------------------------------
    ax = axes[1, 1]

    kmf = KaplanMeierFitter()
    kmf.fit(
        clinical_clean["OS_MONTHS"],
        clinical_clean["OS_EVENT"],
        label=f"Full cohort (n={len(clinical_clean)})",
    )
    kmf.plot_survival_function(
        ax=ax, ci_show=True,
        color="#1976D2", ci_alpha=0.15, linewidth=2,
    )

    # Mark median survival with a dashed horizontal reference line so the
    # reader can read off the median OS without scanning the y-axis.
    median_os = kmf.median_survival_time_
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.axvline(
        median_os, color="grey",
        linestyle="--", linewidth=0.9, alpha=0.7,
        label=f"Median OS = {median_os:.1f} mo",
    )

    ax.set_xlabel("Time (months)", fontsize=10)
    ax.set_ylabel("Survival probability", fontsize=10)
    ax.set_title("Overall survival — full cohort", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    plt.tight_layout()

    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "fig_cohort_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"[FILE] Saved: fig_cohort_summary.png")


# =================================================================
# Main
# =================================================================

def main() -> None:
    """
    Prepare TCGA clinical data for survival analysis.

    Steps
    -----
    1. Load cBioPortal clinical patient table.
    2. Remove missing or invalid survival records.
    3. Deduplicate patient entries.
    4. Convert survival status into binary event variable (OS_EVENT).
    5. Save cleaned dataset for downstream modelling.
    6. Plot and save the 2x2 cohort summary figure.
    """

    # -----------------------------------------------------------------
    # Load raw clinical data
    # -----------------------------------------------------------------

    # cBioPortal files include metadata lines starting with "#"
    # that must be skipped for correct parsing.
    clinical = pd.read_csv(
        ROOT / "data" / "raw" / "clinical" / "data_clinical_patient.txt",
        sep="\t",
        comment="#",
    )

    n_raw = clinical.shape[0]
    logger.info(f"Loaded clinical data: {n_raw} rows")

    # -----------------------------------------------------------------
    # Basic survival filtering
    # -----------------------------------------------------------------

    # Survival analysis requires known follow-up time (OS_MONTHS) and
    # known event status (OS_STATUS); drop patients missing either.
    clinical_clean = clinical.dropna(subset=["OS_MONTHS", "OS_STATUS"])

    # OS_MONTHS <= 0 is not biologically meaningful for survival modelling.
    clinical_clean = clinical_clean[clinical_clean["OS_MONTHS"] > 0].copy()

    n_after_os_filter = len(clinical_clean)
    logger.info(
        f"After filtering missing/invalid survival data: "
        f"{n_after_os_filter} patients"
    )

    # -----------------------------------------------------------------
    # Deduplication
    # -----------------------------------------------------------------

    # TCGA tables occasionally contain duplicated patient IDs due to
    # multiple samples or annotation merges.
    clinical_clean = clinical_clean.drop_duplicates(
        subset=["PATIENT_ID"],
        keep="first",
    )

    n_after_dedup = len(clinical_clean)
    logger.info(f"After deduplication: {n_after_dedup} patients")

    # -----------------------------------------------------------------
    # Convert survival status to binary event variable
    # -----------------------------------------------------------------

    # OS_STATUS format:
    #   "0:LIVING"   -> censored (0)
    #   "1:DECEASED" -> event occurred (1)
    os_status_map = {
        "0:LIVING":   0,
        "1:DECEASED": 1,
    }

    clinical_clean["OS_EVENT"] = clinical_clean["OS_STATUS"].map(os_status_map)

    # -----------------------------------------------------------------
    # Validation check
    # -----------------------------------------------------------------

    # Unmapped values produce NaN in OS_EVENT, which would silently
    # corrupt downstream survival models.
    unmapped = clinical_clean["OS_EVENT"].isna().sum()
    if unmapped > 0:
        logger.warning(
            f"Found {unmapped} unmapped OS_STATUS values. "
            "These rows may be excluded downstream."
        )

    # -----------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------

    logger.info("-" * 50)
    logger.summary(
        f"SURVIVAL SUMMARY:\n"
        f"\t\t\t\t\t Patients          : {len(clinical_clean)}\n"
        f"\t\t\t\t\t Event rate        : {clinical_clean['OS_EVENT'].mean():.1%}\n"
        f"\t\t\t\t\t Median OS (mo)    : {clinical_clean['OS_MONTHS'].median():.1f}"
    )
    logger.info("-" * 50)

    # -----------------------------------------------------------------
    # Save processed output
    # -----------------------------------------------------------------

    out_path = ROOT / "data" / "processed" / "clinical_clean.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clinical_clean.to_csv(out_path, index=False)
    logger.success(f"[FILE] Saved: clinical_clean.csv")

    # -----------------------------------------------------------------
    # Cohort summary figure
    # -----------------------------------------------------------------

    plot_cohort_summary(
        clinical_clean,
        n_raw=n_raw,
        n_after_os_filter=n_after_os_filter,
        n_after_dedup=n_after_dedup,
    )


if __name__ == "__main__":
    main()