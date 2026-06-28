"""
kaplan_meier.py — Final Kaplan-Meier figure for the HR-DDR ODE AUC_X score.

Produces a single, clean, publication-ready KM figure using the PRIMARY
pre-specified tertile split on AUC_X (consistent with the primary analysis
in analyse_ode_survival.py).

Tertile split rationale
-----------------------
Splitting on pre-specified tertiles (T1 low / T2 mid / T3 high) is a valid
inferential test because the grouping is defined before examining survival
outcomes. The omnibus log-rank p-value from a three-group test is therefore
not inflated by data-driven cutpoint selection.

If the 33rd and 67th percentiles coincide (insufficient spread for a
three-group split — common for discrete scores), the function falls back to
a pre-specified median binary split. The fallback is still a valid
pre-specified test; a warning is logged.

This script loads survival_analysis_df.csv, which is produced by
run_ode_cohort.py and contains only patients with successful ODE
integrations.

Outputs
-------
- results/figures/fig_kaplan_meier_aucx_tertile.png
"""

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; required on Windows when
                       # parallel joblib workers are active in the pipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


# =================================================================
# Data loading
# =================================================================

def load_data() -> pd.DataFrame:
    """
    Load the survival analysis table produced by run_ode_cohort.py.

    Returns
    -------
    pd.DataFrame
        One row per patient with columns: PATIENT_ID, AUC_X, OS_MONTHS,
        OS_EVENT, BRCA_MUTANT, and log-transformed ODE scores.
    """
    path = ROOT / "data" / "processed" / "survival_analysis_df.csv"
    df = pd.read_csv(path)
    logger.info(f"Loaded survival analysis table: {len(df)} patients")
    return df


# =================================================================
# KM analysis
# =================================================================

def km_tertile_plot(df: pd.DataFrame) -> tuple[float, str]:
    """
    Stratify patients by pre-specified AUC_X tertiles and plot KM curves.

    Uses the raw AUC_X score (not log-transformed) for tertile boundaries,
    consistent with the primary analysis in analyse_ode_survival.py.
    Monotone transformations do not change tertile membership, but the raw
    score is cleaner and directly interpretable.

    Falls back to a median binary split if the 33rd and 67th percentiles
    coincide (score distribution too discrete for three-group stratification).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'AUC_X', 'OS_MONTHS', 'OS_EVENT'.

    Returns
    -------
    p_val : float
        Omnibus log-rank p-value across all groups.
    split_note : str
        Human-readable description of the split used.
    """
    df = df.copy().dropna(subset=["AUC_X", "OS_MONTHS", "OS_EVENT"])
    n_total = len(df)

    q33 = df["AUC_X"].quantile(1 / 3)
    q67 = df["AUC_X"].quantile(2 / 3)

    if q33 == q67:
        # Tertile boundaries are tied — score lacks sufficient spread for
        # a three-group split.  Fall back to a pre-specified median split.
        logger.warning(
            f"AUC_X tertile boundaries are tied "
            f"(33rd pct = 67th pct = {q33:.4g}). "
            "Falling back to median binary split."
        )
        median = df["AUC_X"].median()
        df["group"] = np.where(
            df["AUC_X"] <= median, "Low AUC_X (≤median)", "High AUC_X (>median)"
        )
        group_labels = ["Low AUC_X (≤median)", "High AUC_X (>median)"]
        colors = ["#1976D2", "#d32f2f"]
        split_note = "median binary split (tertile not possible — tied boundaries)"
    else:
        tertile_labels = ["T1 — Low AUC_X", "T2 — Mid AUC_X", "T3 — High AUC_X"]
        df["group"] = pd.cut(
            df["AUC_X"],
            bins=[-np.inf, q33, q67, np.inf],
            labels=tertile_labels,
        )
        group_labels = tertile_labels
        colors = ["#1976D2", "#F57C00", "#d32f2f"]
        split_note = "tertile split (pre-specified)"

    # -----------------------------------------------------------------
    # Omnibus log-rank test
    # -----------------------------------------------------------------
    res = multivariate_logrank_test(
        df["OS_MONTHS"],
        df["group"].astype(str),
        event_col=df["OS_EVENT"],
    )
    p_val = res.p_value

    logger.info(
        f"KM tertile split — n={n_total}, "
        f"split: {split_note}, "
        f"log-rank p={p_val:.4g}"
    )

    # -----------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    kmf = KaplanMeierFitter()

    for label, color in zip(group_labels, colors):
        grp = df[df["group"] == label]
        kmf.fit(
            grp["OS_MONTHS"], grp["OS_EVENT"],
            label=f"{label} (n={len(grp)})",
        )
        kmf.plot_survival_function(
            ax=ax, ci_show=True, color=color,
            ci_alpha=0.12, linewidth=2,
        )

        med = kmf.median_survival_time_
        logger.info(f"  {label}: n={len(grp)}, median OS={med:.1f} months")

    p_str = f"p = {p_val:.4g}" if p_val >= 1e-4 else "p < 0.0001"
    ax.text(
        0.03, 0.05,
        f"Log-rank {p_str}\n({split_note})",
        transform=ax.transAxes,
        ha="left", va="bottom", fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="#cccccc",
            alpha=0.9,
        ),
    )

    ax.set_xlabel("Time (months)", fontsize=11)
    ax.set_ylabel("Survival probability", fontsize=11)
    ax.set_title(
        "Kaplan-Meier Survival — Stratified by HR-DDR ODE Score (AUC_X)\n"
        f"HGSOC TCGA (n={n_total}), {split_note}",
        fontsize=11,
    )
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()

    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "fig_kaplan_meier_aucx_tertile.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")

    return p_val, split_note


# =================================================================
# Main
# =================================================================

def main() -> None:
    """
    Produce the final AUC_X Kaplan-Meier figure using the primary tertile split.

    Steps
    -----
    1. Load survival_analysis_df.csv.
    2. Stratify patients by pre-specified AUC_X tertiles (fallback: median split).
    3. Run omnibus log-rank test and plot KM curves with 95% CIs.
    4. Save figure to results/figures/.
    """
    warnings.filterwarnings("ignore", category=UserWarning)

    df = load_data()
    p_val, split_note = km_tertile_plot(df)

    logger.info(
        f"Final KM result: log-rank p={p_val:.4g} ({split_note})"
    )


if __name__ == "__main__":
    main()