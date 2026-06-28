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
    fig_forest_univariate_cox.png           (all four scores, combined)
    fig_forest_multivariate_cox.png         (all four scores × 2 covariates, combined)
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
    logger.success(f"[FILE] Saved: KM plot")

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
    logger.success(f"[FILE] Saved: {score_col.lower()}_threshold_scan.csv")

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
    logger.success(f"[FILE] Saved: fig_threshold_scan_{score_col.lower()}.png")


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
    logger.success(f"[FILE] Saved: fig_km_{score_col.lower()}_exploratory_cutoff.png")

    return float(km_res.p_value)


# =================================================================
# Helpers: distribution plots
# =================================================================

def _plot_combined_histogram(
    analysis_df: pd.DataFrame,
    score_cols: list[str],
    best_cuts: dict[str, float],
    fig_dir: Path,
) -> None:
    """
    Plot a 2x2 grid of cohort-wide distributions for all four ODE scores.

    Parameters
    ----------
    analysis_df : pd.DataFrame
        Must contain score columns.
    score_cols : list[str]
        List of score columns to plot.
    best_cuts : dict[str, float]
        Dictionary mapping score name to its optimal scan cutpoint.
    fig_dir : Path
        Destination directory for the figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for idx, score_col in enumerate(score_cols):
        ax = axes[idx]
        best_cut = best_cuts[score_col]
        tertile_cuts = (
            analysis_df[score_col].dropna().quantile([1 / 3, 2 / 3]).values
        )

        ax.hist(
            analysis_df[score_col].dropna(),
            bins=30,
            edgecolor="black",
            color="#90CAF9",
        )
        ax.axvline(
            best_cut,
            color="blue",
            linestyle=":",
            linewidth=1.5,
            label=f"Best cutoff (exp.) = {best_cut:.3g}",
        )
        ax.axvline(
            tertile_cuts[0],
            color="black",
            linestyle="--",
            linewidth=1,
            label="Tertile boundaries",
        )
        ax.axvline(tertile_cuts[1], color="black", linestyle="--", linewidth=1)

        ax.set_xlabel(score_col, fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"{score_col} distribution", fontsize=11)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.2)

    fig.suptitle(
        "Cohort-wide Distributions of ODE Scores",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "fig_hist_ode_scores.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"[FILE] Saved: fig_hist_ode_scores.png")


def _plot_combined_boxplot(
    analysis_df: pd.DataFrame,
    score_cols: list[str],
    fig_dir: Path,
) -> None:
    """
    Plot a 2x2 grid of boxplots of ODE scores stratified by BRCA mutation status.

    Parameters
    ----------
    analysis_df : pd.DataFrame
        Must contain score columns and 'BRCA_MUTANT'.
    score_cols : list[str]
        List of score columns to plot.
    fig_dir : Path
        Destination directory for the figure.
    """
    label_map = {0: "Wild type", 1: "Mutant"}
    fill_colours = ["#1976D2", "#d32f2f"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for idx, score_col in enumerate(score_cols):
        ax = axes[idx]
        group_data = [
            analysis_df.loc[
                analysis_df["BRCA_MUTANT"] == flag, score_col
            ].dropna().values
            for flag in (0, 1)
        ]

        bp = ax.boxplot(
            group_data,
            patch_artist=True,
            widths=0.5,
            medianprops={"color": "white", "linewidth": 2},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.2},
            flierprops={"marker": "o", "markersize": 4, "alpha": 0.5},
        )

        for patch, colour in zip(bp["boxes"], fill_colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.85)

        for flier, colour in zip(bp["fliers"], fill_colours):
            flier.set_markerfacecolor(colour)
            flier.set_markeredgecolor(colour)

        ax.set_xticks([1, 2])
        ax.set_xticklabels([label_map[0], label_map[1]], fontsize=10)
        ax.set_xlabel("BRCA mutation status", fontsize=10)
        ax.set_ylabel(score_col, fontsize=10)
        ax.set_title(f"{score_col} by BRCA status", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "ODE Scores by BRCA Mutation Status",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "fig_boxplot_ode_scores_brca.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"[FILE] Saved: fig_boxplot_ode_scores_brca.png")


# =================================================================
# Helper: univariate Cox
# =================================================================

def _run_cox_univariate(
    analysis_df: pd.DataFrame,
    score_col: str,
) -> dict:
    """
    Fit a univariate Cox proportional hazards model for one ODE score.

    Uses the pre-computed log-transformed score (log_<score_col>) from
    survival_analysis_df.csv rather than re-transforming the raw score.

    A post-fit validity check flags degenerate results where the solver
    produced an infinite HR or a NaN confidence interval — both indicate
    near-zero covariate variance causing Newton-Raphson divergence.

    The per-score figure is omitted intentionally; all four scores are
    combined into a single forest plot by _plot_cox_forest_univariate(),
    called once after the main analysis loop in main().

    Parameters
    ----------
    analysis_df : pd.DataFrame
        Must contain f'log_{score_col}', 'OS_MONTHS', 'OS_EVENT'.
    score_col : str
        ODE score name (used for labelling and EXPECTED_DIRECTION lookup).

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

    # -----------------------------------------------------------------
    # Post-fit validity check.
    #
    # An infinite HR or NaN confidence bound indicates that the covariate
    # had near-zero variance across the cohort, causing the Newton-Raphson
    # solver to diverge. This is a degenerate fit — direction_match must
    # not be inferred from a numerically invalid HR.
    # -----------------------------------------------------------------
    is_degenerate = not np.isfinite(hr) or not np.isfinite(lo) or not np.isfinite(hi)

    if is_degenerate:
        logger.warning(
            f"{score_col} univariate Cox — DEGENERATE FIT: "
            f"HR={hr}, CI=[{lo}, {hi}]. "
            "Covariate likely has near-zero variance across the cohort. "
            "direction_match set to 'degenerate'."
        )
        direction_match = "degenerate"
    else:
        expected        = EXPECTED_DIRECTION[score_col]
        observed        = "HR < 1" if hr < 1 else "HR > 1"
        direction_match = "✓" if expected == observed else "✗ unexpected"

        logger.info(
            f"{score_col} univariate Cox — "
            f"HR={hr:.3g} [{lo:.3g}, {hi:.3g}], "
            f"p={p:.3g}, C-index={c_index:.4f}, {direction_match}"
        )

    return {
        "score":              score_col,
        "expected_direction": EXPECTED_DIRECTION[score_col],
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
) -> dict:
    """
    Fit a multivariate Cox model adjusted for BRCA_MUTANT and check PH assumption.

    BRCA_MUTANT is already present in survival_analysis_df.csv (added by
    run_ode_cohort.py), so no merge with the full expression matrix is needed.

    check_assumptions() writes to stdout via lifelines' internal print calls.
    We capture that output with contextlib.redirect_stdout and route it through
    logger.info to avoid bypassing pipeline.log.

    Per-score figures are omitted intentionally. All four scores are combined
    into a single forest plot by _plot_cox_forest_multivariate(), called once
    after the main analysis loop.

    Parameters
    ----------
    analysis_df : pd.DataFrame
        Must contain f'log_{score_col}', 'OS_MONTHS', 'OS_EVENT', 'BRCA_MUTANT'.
    score_col : str
        ODE score name (used for labelling and row keying in the combined plot).

    Returns
    -------
    dict
        Keys: score, log_col, hr_score, lo_score, hi_score, p_score,
        hr_brca, lo_brca, hi_brca, p_brca, c_index.
        Used by _plot_cox_forest_multivariate() to build the combined figure.
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

    # -----------------------------------------------------------------
    # Extract HR and CI for both covariates.
    # -----------------------------------------------------------------
    s_score = cmvph.summary.loc[log_col]
    s_brca  = cmvph.summary.loc["BRCA_MUTANT"]

    return {
        "score":     score_col,
        "log_col":   log_col,
        # ODE score row
        "hr_score":  float(s_score["exp(coef)"]),
        "lo_score":  float(s_score["exp(coef) lower 95%"]),
        "hi_score":  float(s_score["exp(coef) upper 95%"]),
        "p_score":   float(s_score["p"]),
        # BRCA_MUTANT adjustment row
        "hr_brca":   float(s_brca["exp(coef)"]),
        "lo_brca":   float(s_brca["exp(coef) lower 95%"]),
        "hi_brca":   float(s_brca["exp(coef) upper 95%"]),
        "p_brca":    float(s_brca["p"]),
        "c_index":   float(cmvph.concordance_index_),
    }


# =================================================================
# Helper: combined univariate forest plot
# =================================================================

def _plot_cox_forest_univariate(
    cox_rows: list[dict],
    fig_dir: Path,
) -> None:
    """
    Forest plot of univariate Cox HRs for all four ODE scores in one figure.

    Each score occupies one row. The y-axis label shows the log-transformed
    covariate name, the expected HR direction, and the p-value so the reader
    can assess statistical and biological concordance at a glance.

    Parameters
    ----------
    cox_rows : list[dict]
        One dict per score from _run_cox_univariate, containing:
        score, HR, CI_low, CI_high, p_value, expected_direction,
        direction_match.
    fig_dir : Path
        Destination directory for the figure.
    """
    n = len(cox_rows)

    # Colour each row by observed direction:
    # protective (HR < 1) → blue; deleterious (HR > 1) → red.
    observed_colours = {True: "#1976D2", False: "#d32f2f"}

    fig, ax = plt.subplots(figsize=(8, 1.1 * n + 1.5))

    for i, row in enumerate(cox_rows):
        hr  = row["HR"]
        lo  = row["CI_low"]
        hi  = row["CI_high"]
        p   = row["p_value"]
        col = observed_colours[hr < 1]

        ax.errorbar(
            hr, i,
            xerr=[[hr - lo], [hi - hr]],
            fmt="o", color=col, capsize=4, markersize=7,
            linewidth=1.8,
        )

        # Annotate p-value to the right of each CI bar.
        p_str = f"p = {p:.3g}" if p >= 0.001 else "p < 0.001"
        ax.text(
            hi * 1.02, i, p_str,
            va="center", ha="left", fontsize=8, color="dimgrey",
        )

    ax.axvline(1, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Hazard ratio per SD (log scale)", fontsize=11)

    # y-axis: score name formatted as z-log(Score)
    y_labels = [
        f"z-log({r['score']})"
        for r in cox_rows
    ]
    ax.set_yticks(range(n))
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_ylim(-0.7, n - 0.3)

    ax.set_title(
        "Univariate Cox proportional hazards — ODE scores\n"
        "(adjusted for nothing; one covariate per model)",
        fontsize=11,
    )
    ax.grid(axis="x", alpha=0.25)

    # Legend: observed direction colour key.
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color="#1976D2", label="HR < 1 (protective)"),
        mpatches.Patch(color="#d32f2f", label="HR > 1 (deleterious)"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="lower left", framealpha=0.8)

    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "fig_forest_univariate_cox.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.success("[FILE] Saved: fig_forest_univariate_cox.png")


# =================================================================
# Helper: combined multivariate forest plot
# =================================================================

def _plot_cox_forest_multivariate(
    mv_rows: list[dict],
    fig_dir: Path,
) -> None:
    """
    Forest plot of multivariate Cox HRs for all four ODE scores in one figure.

    Each score contributes two rows (the ODE covariate and BRCA_MUTANT). A
    thin horizontal rule separates score groups so the reader can visually
    associate each ODE row with its BRCA adjustment row. The C-index for each
    model is annotated at the right margin of the ODE score row.

    Parameters
    ----------
    mv_rows : list[dict]
        One dict per score from _run_cox_multivariate, containing:
        score, log_col, hr_score, lo_score, hi_score, p_score,
        hr_brca, lo_brca, hi_brca, p_brca, c_index.
    fig_dir : Path
        Destination directory for the figure.
    """
    # Two rows per score: ODE score row then BRCA_MUTANT row.
    # Interleave with a half-unit gap between score groups.
    y_positions: list[float] = []
    y_labels:    list[str]   = []
    hrs:         list[float] = []
    los:         list[float] = []
    his:         list[float] = []
    colours:     list[str]   = []
    p_values:    list[float] = []
    c_annotations: list[tuple[float, float, str]] = []  # (x, y, text)

    # Colour convention: ODE score rows → score-specific; BRCA → grey.
    score_colours = {
        "AUC_X":    "#1976D2",
        "X_peak":   "#0288D1",
        "T_repair": "#d32f2f",
        "D_resid":  "#C62828",
    }
    brca_colour = "#757575"

    y_cursor = 0.0
    separator_ys: list[float] = []

    for row in mv_rows:
        score = row["score"]
        col   = score_colours.get(score, "black")

        # ODE score row
        y_positions.append(y_cursor)
        y_labels.append(f"z-log({score})")
        hrs.append(row["hr_score"])
        los.append(row["lo_score"])
        his.append(row["hi_score"])
        colours.append(col)
        p_values.append(row["p_score"])
        c_annotations.append((
            row["hi_score"],
            y_cursor,
            f"C={row['c_index']:.3f}",
        ))
        y_cursor += 1.0

        # BRCA_MUTANT adjustment row
        y_positions.append(y_cursor)
        y_labels.append("BRCA_MUTANT")
        hrs.append(row["hr_brca"])
        los.append(row["lo_brca"])
        his.append(row["hi_brca"])
        colours.append(brca_colour)
        p_values.append(row["p_brca"])
        y_cursor += 1.0

        # Record separator position between groups (skip after the last).
        separator_ys.append(y_cursor - 0.5)
        y_cursor += 0.5   # visual gap between score groups

    # Remove the trailing separator.
    if separator_ys:
        separator_ys.pop()

    n_rows = len(y_positions)
    fig, ax = plt.subplots(figsize=(9, 0.55 * n_rows + 2.0))

    for i in range(n_rows):
        hr  = hrs[i]
        lo  = los[i]
        hi  = his[i]
        col = colours[i]
        p   = p_values[i]

        ax.errorbar(
            hr, y_positions[i],
            xerr=[[hr - lo], [hi - hr]],
            fmt="o", color=col, capsize=4, markersize=6,
            linewidth=1.6,
        )

        p_str = f"p = {p:.3g}" if p >= 0.001 else "p < 0.001"
        ax.text(
            hi * 1.03, y_positions[i], p_str,
            va="center", ha="left", fontsize=7.5, color="dimgrey",
        )

    # C-index annotation on the ODE score row of each group.
    for x_hi, y_pos, label in c_annotations:
        ax.text(
            x_hi * 1.03, y_pos + 0.28, label,
            va="bottom", ha="left", fontsize=7, color="black", style="italic",
        )

    # Thin horizontal rules between score groups.
    for sep_y in separator_ys:
        ax.axhline(sep_y, color="lightgrey", linewidth=0.8, linestyle="-")

    ax.axvline(1, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Hazard ratio per SD (log scale)", fontsize=11)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_ylim(-0.6, max(y_positions) + 0.7)

    ax.set_title(
        "Multivariate Cox proportional hazards — ODE scores adjusted for BRCA mutation\n"
        "(two covariates per model: ODE score + BRCA_MUTANT)",
        fontsize=11,
    )
    ax.grid(axis="x", alpha=0.25)

    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color="#1976D2", label="AUC_X"),
        mpatches.Patch(color="#0288D1", label="X_peak"),
        mpatches.Patch(color="#d32f2f", label="T_repair"),
        mpatches.Patch(color="#C62828", label="D_resid"),
        mpatches.Patch(color="#757575", label="BRCA_MUTANT (adj.)"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="lower right", framealpha=0.8)

    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "fig_forest_multivariate_cox.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.success("[FILE] Saved: fig_forest_multivariate_cox.png")


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
    logger.success(f"[FILE] Saved: threshold_scan_summary.csv")

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
    logger.success(f"[FILE] Saved: univariate_cox_comparison.csv")

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
    3. After the per-score loop, draw two combined forest plots:
       - fig_forest_univariate_cox.png (all four scores, one row each)
       - fig_forest_multivariate_cox.png (all four scores × 2 covariates)
    4. Save threshold scan summary and univariate Cox comparison tables.
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
    mv_comparison_rows: list[dict] = []
    best_cuts: dict[str, float] = {}

    for score_col in SCORE_COLS:
        logger.info(f"Processing score: {score_col}")
        logger.info(f"{'-' * 60}")

        # Threshold scan (EXPLORATORY)
        scan, best_cut, best_p = _run_threshold_scan(
            analysis_df, score_col, out_dir
        )
        best_cuts[score_col] = best_cut
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

        # Univariate Cox — returns stats dict; combined plot drawn after loop
        cox_row = _run_cox_univariate(analysis_df, score_col)
        cox_comparison_rows.append(cox_row)

        # Multivariate Cox adjusted for BRCA_MUTANT — returns stats dict;
        # combined plot drawn after loop
        mv_row = _run_cox_multivariate(analysis_df, score_col)
        mv_comparison_rows.append(mv_row)

        # Accumulate scan summary — tertile p is the primary result.
        scan_summary_rows.append({
            "score":             score_col,
            "best_cutoff":       best_cut,
            "scan_best_p":       best_p,          # exploratory only
            "tertile_logrank_p": tertile_p,       # primary, valid for inference
            "exploratory_km_p":  exploratory_p,   # same caveat as scan_best_p
        })

    # -----------------------------------------------------------------
    # Combined distribution plots (drawn once, across all four scores)
    # -----------------------------------------------------------------
    _plot_combined_histogram(analysis_df, SCORE_COLS, best_cuts, fig_dir)
    _plot_combined_boxplot(analysis_df, SCORE_COLS, fig_dir)

    # -----------------------------------------------------------------
    # Combined Cox forest plots (drawn once, across all four scores)
    # -----------------------------------------------------------------
    _plot_cox_forest_univariate(cox_comparison_rows, fig_dir)
    _plot_cox_forest_multivariate(mv_comparison_rows, fig_dir)

    # -----------------------------------------------------------------
    # Save summary tables
    # -----------------------------------------------------------------
    _save_tables(scan_summary_rows, cox_comparison_rows, out_dir)


if __name__ == "__main__":
    main()