"""
analyse_ode_survival.py — threshold scans, Kaplan-Meier splits, and Cox modeling
for multiple ODE-derived survival scores.

This script:
1. Loads the merged HGSOC dataset and the precomputed survival analysis table.
2. Performs threshold scans for four ODE scores (EXPLORATORY only):
      - AUC_X
      - X_peak
      - T_repair
      - D_resid
3. PRIMARY KM analysis: pre-specified tertile split for each score.
4. EXPLORATORY KM analysis: best-cutoff split from threshold scan,
   labelled explicitly as exploratory (inflated p-value, not for inference).
5. Fits four separate univariate Cox models, one per ODE score.
6. Prints a comparison table:
      score | expected direction | HR | 95% CI | p-value | C-index | direction match
7. Fits multivariate Cox models adjusted for BRCA_MUTANT.
8. Checks proportional hazards assumptions for each multivariate model.
9. Saves all outputs to data/processed/.

NOTE on threshold scanning:
    Selecting a cutpoint by minimising the log-rank p-value across all possible
    splits invalidates that p-value for inference — it is a data-driven optimisation,
    not a pre-specified test. The threshold scan is retained here as an exploratory
    diagnostic showing where signal concentrates in the score distribution.
    The PRIMARY KM analysis uses a pre-specified tertile split (T1/T2/T3), whose
    omnibus log-rank p-value IS valid for inference.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
merged = pd.read_csv(OUT / "hgsoc_tcga_merged.csv")
analysis_df = pd.read_csv(OUT / "survival_analysis_df.csv")


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
score_cols = ["AUC_X", "X_peak", "T_repair", "D_resid"]

# Expected HR direction for each score.
# AUC_X / X_peak: higher apoptotic commitment → better platinum response → lower hazard (HR < 1).
# T_repair / D_resid: longer repair / more residual damage → worse outcome → higher hazard (HR > 1).
EXPECTED_DIRECTION = {
    "AUC_X":    "HR < 1",
    "X_peak":   "HR < 1",
    "T_repair": "HR > 1",
    "D_resid":  "HR > 1",
}

# Stores summary rows for each score.
scan_summary_rows = []
cox_comparison_rows = []


# ---------------------------------------------------------------------
# Helper: PRIMARY KM — pre-specified tertile split
# ---------------------------------------------------------------------
def km_tertile_split(df, score_col, out_path):
    """
    Fit and plot Kaplan-Meier curves by pre-specified tertile split (T1/T2/T3).

    The omnibus log-rank p-value from this function is valid for inference
    because the grouping was defined before seeing the survival data.

    If the 33rd and 67th percentiles coincide (many tied values, as can
    occur for T_repair which takes a small number of discrete hour values),
    the function falls back to a pre-specified median binary split and prints
    a warning. The fallback is still a valid pre-specified test.

    Parameters
    ----------
    df        : DataFrame containing score_col, OS_MONTHS, OS_EVENT
    score_col : str — name of the ODE score column
    out_path  : Path — where to save the figure

    Returns
    -------
    float — omnibus log-rank p-value
    """
    df = df.copy().dropna(subset=[score_col, "OS_MONTHS", "OS_EVENT"])

    q33 = df[score_col].quantile(1 / 3)
    q67 = df[score_col].quantile(2 / 3)

    if q33 == q67:
        # Tertile boundaries are identical — score has insufficient spread
        # for a three-group split (common for T_repair, a discrete hour value).
        # Fall back to a pre-specified median binary split.
        print(
            f"  WARNING: {score_col} tertile boundaries are tied "
            f"(33rd pct = 67th pct = {q33}). "
            f"Falling back to median binary split."
        )
        median = df[score_col].median()
        df["tertile"] = np.where(
            df[score_col] <= median,
            "Low (≤median)",
            "High (>median)",
        )
        group_labels = ["Low (≤median)", "High (>median)"]
        colors       = ["#1976D2", "#d32f2f"]
        split_note   = "median split (tertile not possible — tied boundaries)"
    else:
        tertile_labels = ["T1 (low)", "T2 (mid)", "T3 (high)"]
        df["tertile"] = pd.cut(
            df[score_col],
            bins=[-np.inf, q33, q67, np.inf],
            labels=tertile_labels,
        )
        group_labels = tertile_labels
        colors       = ["#1976D2", "#F57C00", "#d32f2f"]
        split_note   = "tertile split, pre-specified"

    fig, ax = plt.subplots(figsize=(7, 5))
    kmf = KaplanMeierFitter()

    for label, color in zip(group_labels, colors):
        grp = df[df["tertile"] == label]
        kmf.fit(grp["OS_MONTHS"], grp["OS_EVENT"], label=f"{label} (n={len(grp)})")
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color)

    # Omnibus log-rank test across all groups.
    res = multivariate_logrank_test(
        df["OS_MONTHS"],
        df["tertile"].astype(str),
        event_col=df["OS_EVENT"],
    )
    p_val = res.p_value

    ax.text(
        0.98, 0.05,
        f"log-rank p = {p_val:.3g}\n({split_note})",
        transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9,
    )
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability")
    ax.set_title(f"Kaplan-Meier by {score_col}\n(PRIMARY analysis — {split_note})")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return p_val


# ---------------------------------------------------------------------
# Loop over each ODE score
# ---------------------------------------------------------------------
for score_col in score_cols:
    print(f"\n{'=' * 70}")
    print(f"Processing score: {score_col}")
    print(f"{'=' * 70}")

    # -------------------------------------------------------------
    # Threshold scan (EXPLORATORY — p-value not valid for inference)
    # -------------------------------------------------------------
    x = np.sort(analysis_df[score_col].dropna().unique())

    cutpoints = []
    pvals = []

    for t in x[1:-1]:
        low  = analysis_df[score_col] <= t
        high = analysis_df[score_col] > t

        # Skip unstable splits with very small groups.
        if low.sum() < 10 or high.sum() < 10:
            continue

        res = logrank_test(
            analysis_df.loc[low, "OS_MONTHS"],
            analysis_df.loc[high, "OS_MONTHS"],
            event_observed_A=analysis_df.loc[low, "OS_EVENT"],
            event_observed_B=analysis_df.loc[high, "OS_EVENT"],
        )
        cutpoints.append(t)
        pvals.append(res.p_value)

    scan = pd.DataFrame({"cutpoint": cutpoints, "p_value": pvals})
    scan["neglog10p"] = -np.log10(scan["p_value"].clip(lower=1e-300))
    scan.to_csv(OUT / f"{score_col.lower()}_threshold_scan.csv", index=False)

    best_idx = scan["p_value"].idxmin()
    best_cut = scan.loc[best_idx, "cutpoint"]
    best_p   = scan.loc[best_idx, "p_value"]

    print(f"Best cutoff (exploratory): {best_cut:.4g}")
    print(f"Best scan p-value (exploratory, NOT for inference): {best_p:.4g}")

    # Threshold scan plot.
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
    fig.savefig(OUT / f"threshold_scan_{score_col.lower()}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------
    # PRIMARY KM: pre-specified tertile split
    # -------------------------------------------------------------
    tertile_p = km_tertile_split(
        analysis_df,
        score_col,
        out_path=OUT / f"km_{score_col.lower()}_tertile_primary.png",
    )
    print(f"PRIMARY tertile log-rank p-value: {tertile_p:.4g}")

    # -------------------------------------------------------------
    # EXPLORATORY KM: best-cutoff split from threshold scan
    # -------------------------------------------------------------
    group_col = f"{score_col}_group"
    analysis_df[group_col] = np.where(
        analysis_df[score_col] <= best_cut, "Low", "High"
    )

    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(7, 5))

    for group, grp in analysis_df.groupby(group_col):
        kmf.fit(
            grp["OS_MONTHS"], grp["OS_EVENT"],
            label=f"{group} (n={len(grp)})",
        )
        kmf.plot_survival_function(ax=ax, ci_show=True)

    low_mask  = analysis_df[group_col] == "Low"
    high_mask = analysis_df[group_col] == "High"

    km_res_exploratory = logrank_test(
        analysis_df.loc[low_mask,  "OS_MONTHS"],
        analysis_df.loc[high_mask, "OS_MONTHS"],
        event_observed_A=analysis_df.loc[low_mask,  "OS_EVENT"],
        event_observed_B=analysis_df.loc[high_mask, "OS_EVENT"],
    )

    ax.text(
        0.98, 0.05,
        f"log-rank p = {km_res_exploratory.p_value:.3g}\n"
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
    fig.savefig(OUT / f"km_{score_col.lower()}_exploratory_cutoff.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Exploratory KM p-value (optimal cutpoint): {km_res_exploratory.p_value:.4g}")

    # -------------------------------------------------------------
    # Histogram with tertile boundaries and best-cutoff marked
    # -------------------------------------------------------------
    tertile_cuts = analysis_df[score_col].dropna().quantile([1/3, 2/3]).values

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(analysis_df[score_col].dropna(), bins=30, edgecolor="black", color="#90CAF9")
    ax.axvline(best_cut,      color="blue",  linestyle=":",  linewidth=1.5,
               label=f"Best cutoff (exploratory) = {best_cut:.3g}")
    ax.axvline(tertile_cuts[0], color="black", linestyle="--", linewidth=1,
               label="Tertile boundaries (primary)")
    ax.axvline(tertile_cuts[1], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(score_col)
    ax.set_ylabel("Count")
    ax.set_title(f"{score_col} distribution")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / f"hist_{score_col.lower()}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------
    # Boxplot by BRCA mutation status
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    analysis_df.boxplot(column=score_col, by="BRCA_MUTANT", ax=ax)
    ax.set_title(f"{score_col} by BRCA mutation status")
    fig.suptitle("")
    ax.set_xlabel("BRCA_MUTANT")
    ax.set_ylabel(score_col)
    fig.tight_layout()
    fig.savefig(OUT / f"boxplot_{score_col.lower()}_by_brca.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Store scan summary — tertile p is the primary result.
    scan_summary_rows.append({
        "score":                score_col,
        "best_cutoff":          best_cut,
        "scan_best_p":          best_p,            # exploratory only
        "tertile_logrank_p":    tertile_p,          # primary, valid for inference
        "exploratory_km_p":     km_res_exploratory.p_value,  # same caveat as scan_best_p
    })

    # -------------------------------------------------------------
    # Univariate Cox model
    # -------------------------------------------------------------
    log_score_col = f"log_{score_col}"
    analysis_df[log_score_col] = np.log1p(analysis_df[score_col])

    cox_df = analysis_df[["OS_MONTHS", "OS_EVENT", log_score_col]].dropna().copy()
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="OS_MONTHS", event_col="OS_EVENT")

    s       = cph.summary.loc[log_score_col]
    hr      = s["exp(coef)"]
    lo      = s["exp(coef) lower 95%"]
    hi      = s["exp(coef) upper 95%"]
    p       = s["p"]
    c_index = cph.concordance_index_

    expected = EXPECTED_DIRECTION[score_col]
    observed = "HR < 1" if hr < 1 else "HR > 1"
    direction_match = "✓" if expected == observed else "✗ unexpected"

    cox_comparison_rows.append({
        "score":              score_col,
        "expected_direction": expected,
        "HR":                 hr,
        "CI_low":             lo,
        "CI_high":            hi,
        "p_value":            p,
        "C_index":            c_index,
        "direction_match":    direction_match,
    })

    print("\nUnivariate Cox summary:")
    print(cph.summary[["coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]])
    print(f"C-index: {c_index:.4f}")
    print(f"Expected direction: {expected} | Observed: {observed} | {direction_match}")

    # Univariate HR plot.
    fig, ax = plt.subplots(figsize=(5, 2.2))
    ax.errorbar(hr, 0, xerr=[[hr - lo], [hi - hr]], fmt="o", color="black", capsize=4)
    ax.axvline(1, color="red", linestyle="--", linewidth=1)
    ax.set_yticks([0])
    ax.set_yticklabels([log_score_col])
    ax.set_xlabel("Hazard ratio")
    ax.set_title(f"Univariate Cox model: {score_col}")
    fig.tight_layout()
    fig.savefig(OUT / f"cox_hr_{score_col.lower()}_univariate.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------
    # Multivariate Cox model adjusted for BRCA_MUTANT
    # -------------------------------------------------------------
    df_mv = merged.merge(
        analysis_df[["PATIENT_ID", score_col, log_score_col]],
        on="PATIENT_ID",
        how="inner",
    )

    cox_mv_df = df_mv[
        ["OS_MONTHS", "OS_EVENT", log_score_col, "BRCA_MUTANT"]
    ].dropna().copy()

    cmvph = CoxPHFitter()
    cmvph.fit(cox_mv_df, duration_col="OS_MONTHS", event_col="OS_EVENT")

    print("\nMultivariate Cox summary:")
    print(cmvph.summary[["coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]])
    print(f"Adjusted C-index: {cmvph.concordance_index_:.4f}")

    # PH assumption check — show_plots=False avoids interactive display issues
    # in terminal/headless environments. Text output is sufficient.
    print(f"\nChecking PH assumptions for {score_col} multivariate model...")
    cmvph.check_assumptions(
        cox_mv_df,
        p_value_threshold=0.05,
        show_plots=False,
        advice=True,
    )

    # Multivariate HR plot for ODE score only.
    s_mv  = cmvph.summary.loc[log_score_col]
    hr_mv = s_mv["exp(coef)"]
    lo_mv = s_mv["exp(coef) lower 95%"]
    hi_mv = s_mv["exp(coef) upper 95%"]

    fig, ax = plt.subplots(figsize=(5, 2.2))
    ax.errorbar(hr_mv, 0, xerr=[[hr_mv - lo_mv], [hi_mv - hr_mv]],
                fmt="o", color="black", capsize=4)
    ax.axvline(1, color="red", linestyle="--", linewidth=1)
    ax.set_yticks([0])
    ax.set_yticklabels([log_score_col])
    ax.set_xlabel("Hazard ratio")
    ax.set_title(f"Multivariate Cox model: {score_col}")
    fig.tight_layout()
    fig.savefig(OUT / f"cox_hr_{score_col.lower()}_multivariate.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Full multivariate forest plot.
    summary = cmvph.summary.copy()
    summary = summary.reset_index().rename(columns={"covariate": "variable"})
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
    fig.savefig(OUT / f"cox_forest_{score_col.lower()}_multivariate.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Save threshold scan summary table
# ---------------------------------------------------------------------
scan_summary_df = pd.DataFrame(scan_summary_rows)
scan_summary_df.to_csv(OUT / "threshold_scan_summary.csv", index=False)

print(f"\n{'=' * 70}")
print("Threshold scan summary")
print("  tertile_logrank_p = PRIMARY (pre-specified, valid for inference)")
print("  scan_best_p       = EXPLORATORY (optimal cutpoint, inflated)")
print(f"{'=' * 70}")
print(scan_summary_df.to_string(index=False))


# ---------------------------------------------------------------------
# Save and print univariate Cox comparison table
# ---------------------------------------------------------------------
cox_comparison_df = pd.DataFrame(cox_comparison_rows)

cox_comparison_df["95% CI"] = (
    "[" +
    cox_comparison_df["CI_low"].map(lambda x: f"{x:.3g}") +
    ", " +
    cox_comparison_df["CI_high"].map(lambda x: f"{x:.3g}") +
    "]"
)

cox_comparison_print = cox_comparison_df[[
    "score", "expected_direction", "HR", "95% CI",
    "p_value", "C_index", "direction_match"
]].copy()
cox_comparison_print["HR"]      = cox_comparison_print["HR"].map(lambda x: f"{x:.3g}")
cox_comparison_print["p_value"] = cox_comparison_print["p_value"].map(lambda x: f"{x:.3g}")
cox_comparison_print["C_index"] = cox_comparison_print["C_index"].map(lambda x: f"{x:.3f}")

cox_comparison_df.to_csv(OUT / "univariate_cox_comparison.csv", index=False)

print(f"\n{'=' * 70}")
print("Univariate Cox comparison table")
print(f"{'=' * 70}")
print(cox_comparison_print.to_string(index=False))

print("\nFinished successfully.")