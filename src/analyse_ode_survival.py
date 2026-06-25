"""
analyse_ode_survival.py — Threshold scans, Kaplan-Meier splits, and Cox
modelling for four ODE-derived survival scores.

Steps
-----
1. Load survival_analysis_df.csv; validate that pre-computed log columns
   are present.
2. Threshold scan for each score (EXPLORATORY): scan all binary cutpoints,
   record the one that minimises log-rank p.
3. PRIMARY KM: pre-specified tertile split for each score.  The omnibus
   log-rank p-value is valid for inference.
4. EXPLORATORY KM: binary split at the optimal scan cutpoint, labelled
   explicitly as exploratory (p-value not valid for inference).
5. Histogram of score distribution with tertile and scan boundaries marked.
6. Boxplot of each score by BRCA mutation status.
7. Univariate Cox proportional hazards model per score.
8. Multivariate Cox model adjusted for BRCA_MUTANT, with proportional
   hazards assumption check routed through logger.
9. Summary tables saved to data/processed/.

NOTE on threshold scanning:
    Selecting a cutpoint by minimising the log-rank p-value across all
    possible splits invalidates that p-value for inference — it is a
    data-driven optimisation, not a pre-specified test. The threshold scan
    is retained here as an exploratory diagnostic showing where signal
    concentrates in the score distribution. The PRIMARY KM analysis uses a
    pre-specified tertile split (T1/T2/T3), whose omnibus log-rank p-value
    IS valid for inference.

Outputs
-------
data/processed/:
    {score}_threshold_scan.csv        (× 4)
    threshold_scan_summary.csv
    univariate_cox_comparison.csv

results/figures/:
    fig_threshold_scan_{score}.png          (× 4)
    fig_km_{score}_tertile_primary.png      (× 4)
    fig_km_{score}_exploratory_cutoff.png   (× 4)
    fig_hist_{score}.png                    (× 4)
    fig_boxplot_{score}_brca.png            (× 4)
    fig_cox_hr_{score}_univariate.png       (× 4)
    fig_cox_hr_{score}_multivariate.png     (× 4)
    fig_forest_{score}_multivariate.png     (× 4)
"""

import contextlib
import io
import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; prevents tkinter threading
                       # crash when joblib workers are active on Windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)

# =================================================================
# Constants
# =================================================================

SCORE_COLS = ["AUC_X", "X_peak", "T_repair", "D_resid"]

# Expected HR direction for each score.
# AUC_X / X_peak: higher apoptotic commitment → better platinum response
#   → lower hazard (HR < 1).
# T_repair / D_resid: longer repair time / more residual damage → worse
#   outcome → higher hazard (HR > 1).
EXPECTED_DIRECTION: dict[str, str] = {
    "AUC_X":    "HR < 1",
    "X_peak":   "HR < 1",
    "T_repair": "HR > 1",
    "D_resid":  "HR > 1",
}


# =================================================================
# Helper: PRIMARY KM — pre-specified tertile split
# =================================================================

def km_tertile_split(
    df: pd.DataFrame,
    score_col: str,
    out_path: Path,
) -> float:
    """
    Fit and plot Kaplan-Meier curves using a pre-specified tertile split.

    The omnibus log-rank p-value is valid for inference because the grouping
    is defined before examining survival outcomes.

    If the 33rd and 67th percentiles coincide (common for T_repair, which
    takes a small number of discrete hour values), the function falls back
    to a pre-specified median binary split. The fallback is still a valid
    pre-specified test; a warning is logged.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain score_col, 'OS_MONTHS', and 'OS_EVENT'.
    score_col : str
        Name of the ODE score column to stratify on.
    out_path : Path
        Destination path for the saved figure.

    Returns
    -------
    float
        Omnibus log-rank p-value across all groups.
    """
    df = df.copy().dropna(subset=[score_col, "OS_MONTHS", "OS_EVENT"])

    q33 = df[score_col].quantile(1 / 3)
    q67 = df[score_col].quantile(2 / 3)

    if q33 == q67:
        # Tertile boundaries are identical — score has insufficient spread
        # for a three-group split (common for T_repair, a discrete hour value).
        # Fall back to a pre-specified median binary split.
        logger.warning(
            f"{score_col} tertile boundaries are tied "
            f"(33rd pct = 67th pct = {q33}). "
            "Falling back to median binary split."
        )
        median = df[score_col].median()
        df["tertile"] = np.where(
            df[score_col] <= median,
            "Low (≤median)",
            "High (>median)",
        )
        group_labels = ["Low (≤median)", "High (>median)"]
        colors     = ["#1976D2", "#d32f2f"]
        split_note = "median split (tertile not possible — tied boundaries)"
    else:
        tertile_labels = ["T1 (low)", "T2 (mid)", "T3 (high)"]
        df["tertile"] = pd.cut(
            df[score_col],
            bins=[-np.inf, q33, q67, np.inf],
            labels=tertile_labels,
        )
        group_labels = tertile_labels
        colors     = ["#1976D2", "#F57C00", "#d32f2f"]
        split_note = "tertile split, pre-specified"

    # Omnibus log-rank test across all groups.
    res = multivariate_logrank_test(
        df["OS_MONTHS"],
        df["tertile"].astype(str),
        event_col=df["OS_EVENT"],
    )
    p_val = res.p_value

    fig, ax = plt.subplots(figsize=(7, 5))
    kmf = KaplanMeierFitter()

    for label, color in zip(group_labels, colors):
        grp = df[df["tertile"] == label]
        kmf.fit(
            grp["OS_MONTHS"], grp["OS_EVENT"],
            label=f"{label} (n={len(grp)})",
        )
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color)

    ax.text(
        0.98, 0.05,
        f"log-rank p = {p_val:.3g}\n({split_note})",
        transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9,
    )
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability")
    ax.set_title(
        f"Kaplan-Meier by {score_col}\n"
        f"(PRIMARY analysis — {split_note})"
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")

    return p_val


# =================================================================
# Helper: threshold scan (EXPLORATORY)
# =================================================================

def _run_threshold_scan(
    analysis_df: pd.DataFrame,
    score_col: str,
    out_dir: Path,
) -> tuple[pd.DataFrame, float, float]:
    """
    Scan all binary cutpoints and identify the split that minimises log-rank p.

    NOTE: selecting the cutpoint by minimising log-rank p invalidates that
    p-value for inference. This result is retained as an exploratory
    diagnostic only.

    Parameters
    ----------
    analysis_df : pd.DataFrame
        Must contain score_col, 'OS_MONTHS', 'OS_EVENT'.
    score_col : str
        ODE score column to scan.
    out_dir : Path
        Directory for the threshold scan CSV output.

    Returns
    -------
    scan : pd.DataFrame
        Table of cutpoints and their log-rank p-values.
    best_cut : float
        Cutpoint that minimised log-rank p.
    best_p : float
        Minimum log-rank p (exploratory — not valid for inference).
    """
    x = np.sort(analysis_df[score_col].dropna().unique())
    cutpoints: list[float] = []
    pvals: list[float] = []

    for t in x[1:-1]:
        low  = analysis_df[score_col] <= t
        high = analysis_df[score_col] > t

        # Skip unstable splits with very small groups.
        if low.sum() < 10 or high.sum() < 10:
            continue

        res = logrank_test(
            analysis_df.loc[low,  "OS_MONTHS"],
            analysis_df.loc[high, "OS_MONTHS"],
            event_observed_A=analysis_df.loc[low,  "OS_EVENT"],
            event_observed_B=analysis_df.loc[high, "OS_EVENT"],
        )
        cutpoints.append(float(t))
        pvals.append(float(res.p_value))

    scan = pd.DataFrame({"cutpoint": cutpoints, "p_value": pvals})
    scan["neglog10p"] = -np.log10(scan["p_value"].clip(lower=1e-300))

    csv_path = out_dir / f"{score_col.lower()}_threshold_scan.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    scan.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")

    # NOTE: selecting the cutpoint by minimising log-rank p invalidates that
    # p-value for inference. This result is retained as exploratory only.
    best_idx = scan["p_value"].idxmin()
    best_cut = float(scan.loc[best_idx, "cutpoint"])
    best_p   = float(scan.loc[best_idx, "p_value"])

    logger.info(
        f"{score_col} — best cutoff (exploratory): {best_cut:.4g}, "
        f"scan p={best_p:.4g} (NOT for inference)"
    )
    return scan, best_cut, best_p


def _plot_threshold_scan(
    scan: pd.DataFrame,
    score_col: str,
    best_cut: float,
    best_p: float,
    fig_dir: Path,
) -> None:
    """
    Plot log-rank p-value vs cutpoint from the threshold scan.

    Parameters
    ----------
    scan : pd.DataFrame
        Output of _run_threshold_scan with 'cutpoint' and 'p_value' columns.
    score_col : str
        ODE score label for axis titles.
    best_cut : float
        Cutpoint that minimised p-value (marked with a vertical line).
    best_p : float
        Minimum p-value (annotated on the figure).
    fig_dir : Path
        Destination directory for the figure.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(scan["cutpoint"], scan["p_value"], color="black", linewidth=1)
    ax.scatter(scan["cutpoint"], scan["p_value"], s=12, color="black")
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1, label="p = 0.05")
    ax.axvline(best_cut, color="blue", linestyle=":", linewidth=1)
    ax.text(
        best_cut, best_p,
        f" best cutoff = {best_cut:.3g}",
        va="bottom", ha="left", fontsize=9,
    )
    ax.set_yscale("log")
    ax.set_xlabel(f"{score_col} cutpoint")
    ax.set_ylabel("Log-rank p-value")
    ax.set_title(
        f"Threshold scan for {score_col}\n"
        f"(EXPLORATORY — optimal p-value not valid for inference)"
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / f"fig_threshold_scan_{score_col.lower()}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# =================================================================
# Helper: EXPLORATORY KM — optimal cutpoint
# =================================================================

def _km_exploratory(
    analysis_df: pd.DataFrame,
    score_col: str,
    best_cut: float,
    fig_dir: Path,
) -> float:
    """
    Plot KM curves split at the data-driven optimal cutpoint (exploratory only).

    NOTE: the p-value from this analysis is inflated because the cutpoint was
    selected to minimise it. Do not use for inference.

    Parameters
    ----------
    analysis_df : pd.DataFrame
        Must contain score_col, 'OS_MONTHS', 'OS_EVENT'.
    score_col : str
        ODE score column to stratify on.
    best_cut : float
        Optimal cutpoint from _run_threshold_scan.
    fig_dir : Path
        Destination directory for the figure.

    Returns
    -------
    float
        Exploratory log-rank p-value (not valid for inference).
    """
    df = analysis_df.copy()
    df["_group"] = np.where(df[score_col] <= best_cut, "Low", "High")

    low_mask  = df["_group"] == "Low"
    high_mask = df["_group"] == "High"

    km_res = logrank_test(
        df.loc[low_mask,  "OS_MONTHS"],
        df.loc[high_mask, "OS_MONTHS"],
        event_observed_A=df.loc[low_mask,  "OS_EVENT"],
        event_observed_B=df.loc[high_mask, "OS_EVENT"],
    )

    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(7, 5))

    for group, grp in df.groupby("_group"):
        kmf.fit(
            grp["OS_MONTHS"], grp["OS_EVENT"],
            label=f"{group} (n={len(grp)})",
        )
        kmf.plot_survival_function(ax=ax, ci_show=True)

    ax.text(
        0.98, 0.05,
        f"log-rank p = {km_res.p_value:.3g}\n"
        f"(optimal cutpoint — EXPLORATORY)",
        transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9,
    )
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability")
    ax.set_title(
        f"Kaplan-Meier by {score_col} cutoff = {best_cut:.3g}\n"
        f"(EXPLORATORY — cutpoint selected post-hoc)"
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()

    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / f"fig_km_{score_col.lower()}_exploratory_cutoff.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")

    return float(km_res.p_value)


# =================================================================
# Helpers: distribution plots
# =================================================================

def _plot_histogram(
    analysis_df: pd.DataFrame,
    score_col: str,
    best_cut: float,
    fig_dir: Path,
) -> None:
    """
    Histogram of score distribution with tertile and scan boundaries marked.

    Parameters
    ----------
    analysis_df : pd.DataFrame
        Must contain score_col.
    score_col : str
        ODE score column to plot.
    best_cut : float
        Optimal cutpoint from threshold scan (shown as exploratory marker).
    fig_dir : Path
        Destination directory for the figure.
    """
    tertile_cuts = analysis_df[score_col].dropna().quantile([1 / 3, 2 / 3]).values

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(
        analysis_df[score_col].dropna(),
        bins=30, edgecolor="black", color="#90CAF9",
    )
    ax.axvline(
        best_cut, color="blue", linestyle=":", linewidth=1.5,
        label=f"Best cutoff (exploratory) = {best_cut:.3g}",
    )
    ax.axvline(
        tertile_cuts[0], color="black", linestyle="--", linewidth=1,
        label="Tertile boundaries (primary)",
    )
    ax.axvline(tertile_cuts[1], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(score_col)
    ax.set_ylabel("Count")
    ax.set_title(f"{score_col} distribution")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()

    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / f"fig_hist_{score_col.lower()}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


def _plot_boxplot(
    analysis_df: pd.DataFrame,
    score_col: str,
    fig_dir: Path,
) -> None:
    """
    Boxplot of ODE score stratified by BRCA mutation status.

    Parameters
    ----------
    analysis_df : pd.DataFrame
        Must contain score_col and 'BRCA_MUTANT'.
    score_col : str
        ODE score column to plot.
    fig_dir : Path
        Destination directory for the figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    analysis_df.boxplot(column=score_col, by="BRCA_MUTANT", ax=ax)
    ax.set_title(f"{score_col} by BRCA mutation status")
    fig.suptitle("")
    ax.set_xlabel("BRCA_MUTANT")
    ax.set_ylabel(score_col)
    fig.tight_layout()

    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / f"fig_boxplot_{score_col.lower()}_brca.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# =================================================================
# Helper: univariate Cox
# =================================================================

def _run_cox_univariate(
    analysis_df: pd.DataFrame,
    score_col: str,
    fig_dir: Path,
) -> dict:
    """
    Fit a univariate Cox proportional hazards model for one ODE score.

    Uses the pre-computed log-transformed score (log_<score_col>) from
    survival_analysis_df.csv rather than re-transforming the raw score.

    Parameters
    ----------
    analysis_df : pd.DataFrame
        Must contain f'log_{score_col}', 'OS_MONTHS', 'OS_EVENT'.
    score_col : str
        ODE score name (used for labelling and EXPECTED_DIRECTION lookup).
    fig_dir : Path
        Destination directory for the HR forest plot.

    Returns
    -------
    dict
        Summary row for the univariate Cox comparison table.
    """
    log_col = f"log_{score_col}"
    cox_df  = analysis_df[["OS_MONTHS", "OS_EVENT", log_col]].dropna().copy()

    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="OS_MONTHS", event_col="OS_EVENT")

    s       = cph.summary.loc[log_col]
    hr      = float(s["exp(coef)"])
    lo      = float(s["exp(coef) lower 95%"])
    hi      = float(s["exp(coef) upper 95%"])
    p       = float(s["p"])
    c_index = float(cph.concordance_index_)

    expected        = EXPECTED_DIRECTION[score_col]
    observed        = "HR < 1" if hr < 1 else "HR > 1"
    direction_match = "✓" if expected == observed else "✗ unexpected"

    logger.info(
        f"{score_col} univariate Cox — "
        f"HR={hr:.3g} [{lo:.3g}, {hi:.3g}], "
        f"p={p:.3g}, C-index={c_index:.4f}, {direction_match}"
    )

    # Single-variable HR plot.
    fig, ax = plt.subplots(figsize=(5, 2.2))
    ax.errorbar(
        hr, 0, xerr=[[hr - lo], [hi - hr]],
        fmt="o", color="black", capsize=4,
    )
    ax.axvline(1, color="red", linestyle="--", linewidth=1)
    ax.set_yticks([0])
    ax.set_yticklabels([log_col])
    ax.set_xlabel("Hazard ratio")
    ax.set_title(f"Univariate Cox model: {score_col}")
    fig.tight_layout()

    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / f"fig_cox_hr_{score_col.lower()}_univariate.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")

    return {
        "score":              score_col,
        "expected_direction": expected,
        "HR":                 hr,
        "CI_low":             lo,
        "CI_high":            hi,
        "p_value":            p,
        "C_index":            c_index,
        "direction_match":    direction_match,
    }


# =================================================================
# Helper: multivariate Cox
# =================================================================

def _run_cox_multivariate(
    analysis_df: pd.DataFrame,
    score_col: str,
    fig_dir: Path,
) -> None:
    """
    Fit a multivariate Cox model adjusted for BRCA_MUTANT and check PH assumption.

    BRCA_MUTANT is already present in survival_analysis_df.csv (added by
    run_ode_cohort.py), so no merge with the full expression matrix is needed.

    check_assumptions() writes to stdout via lifelines' internal print calls.
    We capture that output with contextlib.redirect_stdout and route it through
    logger.info to avoid bypassing pipeline.log.

    Parameters
    ----------
    analysis_df : pd.DataFrame
        Must contain f'log_{score_col}', 'OS_MONTHS', 'OS_EVENT', 'BRCA_MUTANT'.
    score_col : str
        ODE score name (used for labelling).
    fig_dir : Path
        Destination directory for HR and forest plots.
    """
    log_col   = f"log_{score_col}"
    cox_mv_df = analysis_df[
        ["OS_MONTHS", "OS_EVENT", log_col, "BRCA_MUTANT"]
    ].dropna().copy()

    cmvph = CoxPHFitter()
    cmvph.fit(cox_mv_df, duration_col="OS_MONTHS", event_col="OS_EVENT")

    logger.info(
        f"{score_col} multivariate Cox (adj. BRCA_MUTANT) — "
        f"C-index={cmvph.concordance_index_:.4f}"
    )

    # -----------------------------------------------------------------
    # PH assumption check.
    # lifelines.check_assumptions() uses print() internally. Redirect
    # stdout to a buffer and emit via logger so the output reaches
    # pipeline.log without violating the no-print rule.
    # -----------------------------------------------------------------
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cmvph.check_assumptions(
            cox_mv_df,
            p_value_threshold=0.05,
            show_plots=False,
            advice=True,
        )
    ph_text = buf.getvalue().strip()
    if ph_text:
        logger.info(f"PH assumption check ({score_col}):\n{ph_text}")

    # Multivariate HR plot for ODE score covariate only.
    s_mv  = cmvph.summary.loc[log_col]
    hr_mv = float(s_mv["exp(coef)"])
    lo_mv = float(s_mv["exp(coef) lower 95%"])
    hi_mv = float(s_mv["exp(coef) upper 95%"])

    fig, ax = plt.subplots(figsize=(5, 2.2))
    ax.errorbar(
        hr_mv, 0,
        xerr=[[hr_mv - lo_mv], [hi_mv - hr_mv]],
        fmt="o", color="black", capsize=4,
    )
    ax.axvline(1, color="red", linestyle="--", linewidth=1)
    ax.set_yticks([0])
    ax.set_yticklabels([log_col])
    ax.set_xlabel("Hazard ratio")
    ax.set_title(f"Multivariate Cox model: {score_col}")
    fig.tight_layout()

    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / f"fig_cox_hr_{score_col.lower()}_multivariate.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")

    # Full multivariate forest plot.
    summary = (
        cmvph.summary.copy()
        .reset_index()
        .rename(columns={"covariate": "variable"})
    )
    summary["HR"]   = summary["exp(coef)"]
    summary["low"]  = summary["exp(coef) lower 95%"]
    summary["high"] = summary["exp(coef) upper 95%"]
    summary = summary.sort_values("HR")

    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(summary))))
    y = range(len(summary))
    ax.errorbar(
        summary["HR"], y,
        xerr=[summary["HR"] - summary["low"], summary["high"] - summary["HR"]],
        fmt="o", color="black", capsize=3,
    )
    ax.axvline(1, color="red", linestyle="--", linewidth=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels(summary["variable"])
    ax.set_xscale("log")
    ax.set_xlabel("Hazard ratio (log scale)")
    ax.set_title(f"Multivariate Cox model: {score_col}")
    fig.tight_layout()

    out_path = fig_dir / f"fig_forest_{score_col.lower()}_multivariate.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# =================================================================
# Helper: save summary tables
# =================================================================

def _save_tables(
    scan_rows: list[dict],
    cox_rows: list[dict],
    out_dir: Path,
) -> None:
    """
    Assemble and save the threshold scan summary and univariate Cox tables.

    Parameters
    ----------
    scan_rows : list[dict]
        One dict per score with best cutoff, scan p, and tertile p.
    cox_rows : list[dict]
        One dict per score with Cox HR, CI, p-value, and C-index.
    out_dir : Path
        Destination directory for CSV outputs (data/processed/).
    """
    # -----------------------------------------------------------------
    # Threshold scan summary
    # -----------------------------------------------------------------
    scan_df   = pd.DataFrame(scan_rows)
    scan_path = out_dir / "threshold_scan_summary.csv"
    scan_path.parent.mkdir(parents=True, exist_ok=True)
    scan_df.to_csv(scan_path, index=False)
    logger.info(f"Saved: {scan_path}")

    logger.info(
        "Threshold scan summary\n"
        "  tertile_logrank_p = PRIMARY (pre-specified, valid for inference)\n"
        "  scan_best_p       = EXPLORATORY (optimal cutpoint, inflated)\n"
        + scan_df.to_string(index=False)
    )

    # -----------------------------------------------------------------
    # Univariate Cox comparison table
    # -----------------------------------------------------------------
    cox_df = pd.DataFrame(cox_rows)
    cox_df["95% CI"] = (
        "[" + cox_df["CI_low"].map(lambda x: f"{x:.3g}")
        + ", " + cox_df["CI_high"].map(lambda x: f"{x:.3g}") + "]"
    )

    cox_path = out_dir / "univariate_cox_comparison.csv"
    cox_df.to_csv(cox_path, index=False)
    logger.info(f"Saved: {cox_path}")

    display_cols = [
        "score", "expected_direction", "HR", "95% CI",
        "p_value", "C_index", "direction_match",
    ]
    display = cox_df[display_cols].copy()
    display["HR"]      = display["HR"].map(lambda x: f"{x:.3g}")
    display["p_value"] = display["p_value"].map(lambda x: f"{x:.3g}")
    display["C_index"] = display["C_index"].map(lambda x: f"{x:.3f}")

    logger.info(
        "Univariate Cox comparison table:\n"
        + display.to_string(index=False)
    )


# =================================================================
# Main
# =================================================================

def main() -> None:
    """
    Run the full ODE survival analysis workflow.

    Steps
    -----
    1. Load survival_analysis_df.csv; validate that pre-computed log columns
       are present (written by run_ode_cohort.py).
    2. For each ODE score: threshold scan, PRIMARY tertile KM, EXPLORATORY
       optimal-cutpoint KM, histogram, BRCA boxplot, univariate Cox,
       multivariate Cox with PH assumption check.
    3. Save threshold scan summary and univariate Cox comparison tables.
    """
    warnings.filterwarnings("ignore", category=UserWarning)

    out_dir = ROOT / "data" / "processed"
    fig_dir = ROOT / "results" / "figures"

    # -----------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------
    analysis_df = pd.read_csv(out_dir / "survival_analysis_df.csv")
    logger.info(f"Loaded survival analysis table: {len(analysis_df)} patients")

    # Validate pre-computed log columns (written by run_ode_cohort.py).
    missing_cols = [
        f"log_{c}" for c in SCORE_COLS
        if f"log_{c}" not in analysis_df.columns
    ]
    if missing_cols:
        raise ValueError(
            f"Log-transformed score columns missing from survival_analysis_df.csv: "
            f"{missing_cols}. Re-run run_ode_cohort.py."
        )

    # -----------------------------------------------------------------
    # Per-score analysis loop
    # -----------------------------------------------------------------
    scan_summary_rows: list[dict] = []
    cox_comparison_rows: list[dict] = []

    for score_col in SCORE_COLS:
        logger.info(f"{'=' * 60}")
        logger.info(f"Processing score: {score_col}")
        logger.info(f"{'=' * 60}")

        # Threshold scan (EXPLORATORY)
        scan, best_cut, best_p = _run_threshold_scan(
            analysis_df, score_col, out_dir
        )
        _plot_threshold_scan(scan, score_col, best_cut, best_p, fig_dir)

        # PRIMARY KM: pre-specified tertile split
        tertile_p = km_tertile_split(
            analysis_df,
            score_col,
            out_path=fig_dir / f"fig_km_{score_col.lower()}_tertile_primary.png",
        )
        logger.info(f"PRIMARY tertile log-rank p: {tertile_p:.4g}")

        # EXPLORATORY KM: best cutoff from threshold scan
        exploratory_p = _km_exploratory(
            analysis_df, score_col, best_cut, fig_dir
        )
        logger.info(
            f"EXPLORATORY KM p (optimal cutpoint, not for inference): "
            f"{exploratory_p:.4g}"
        )

        # Distribution plots
        _plot_histogram(analysis_df, score_col, best_cut, fig_dir)
        _plot_boxplot(analysis_df, score_col, fig_dir)

        # Univariate Cox
        cox_row = _run_cox_univariate(analysis_df, score_col, fig_dir)
        cox_comparison_rows.append(cox_row)

        # Multivariate Cox adjusted for BRCA_MUTANT
        _run_cox_multivariate(analysis_df, score_col, fig_dir)

        # Accumulate scan summary — tertile p is the primary result.
        scan_summary_rows.append({
            "score":             score_col,
            "best_cutoff":       best_cut,
            "scan_best_p":       best_p,         # exploratory only
            "tertile_logrank_p": tertile_p,       # primary, valid for inference
            "exploratory_km_p":  exploratory_p,   # same caveat as scan_best_p
        })

    # -----------------------------------------------------------------
    # Save summary tables
    # -----------------------------------------------------------------
    _save_tables(scan_summary_rows, cox_comparison_rows, out_dir)

    logger.info("analyse_ode_survival complete.")


if __name__ == "__main__":
    main()