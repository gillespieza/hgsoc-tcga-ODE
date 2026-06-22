"""
survival_analysis.py — threshold scans, Kaplan-Meier splits, and Cox modeling
for multiple ODE-derived survival scores.

This script:
1. Loads the merged HGSOC dataset and the precomputed survival analysis table.
2. Performs threshold scans for four ODE scores:
      - AUC_X
      - X_peak
      - T_repair
      - D_resid
3. Selects the best cutoff for each score (minimum log-rank p-value).
4. Splits patients into Low vs High groups using that cutoff.
5. Saves for each score:
      - threshold scan CSV
      - threshold scan plot
      - histogram
      - boxplot by BRCA mutation status
      - Kaplan-Meier curve by best cutoff
6. Fits four separate univariate Cox models, one per ODE score.
7. Prints a comparison table:
      score | HR | 95% CI | p-value | C-index
8. Fits multivariate Cox models adjusted for BRCA_MUTANT.
9. Checks proportional hazards assumptions for each multivariate model.
10. Saves all outputs to data/processed/.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test


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

# This will store the best cutoff and KM p-value for each score.
scan_summary_rows = []

# This will store the univariate Cox results for all four scores.
cox_comparison_rows = []


# ---------------------------------------------------------------------
# Loop over each ODE score
# ---------------------------------------------------------------------
for score_col in score_cols:
    print(f"\n{'=' * 70}")
    print(f"Processing score: {score_col}")
    print(f"{'=' * 70}")

    # -------------------------------------------------------------
    # Threshold scan
    # -------------------------------------------------------------
    x = np.sort(analysis_df[score_col].dropna().unique())

    cutpoints = []
    pvals = []

    for t in x[1:-1]:
        low = analysis_df[score_col] <= t
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

    scan = pd.DataFrame({
        "cutpoint": cutpoints,
        "p_value": pvals
    })

    scan["neglog10p"] = -np.log10(scan["p_value"].clip(lower=1e-300))
    scan.to_csv(OUT / f"{score_col.lower()}_threshold_scan.csv", index=False)

    best_idx = scan["p_value"].idxmin()
    best_cut = scan.loc[best_idx, "cutpoint"]
    best_p = scan.loc[best_idx, "p_value"]

    print(f"Best cutoff for {score_col}: {best_cut}")
    print(f"Best log-rank p-value: {best_p}")

    # Threshold scan plot.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(scan["cutpoint"], scan["p_value"], color="black", linewidth=1)
    ax.scatter(scan["cutpoint"], scan["p_value"], s=12, color="black")
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1, label="p = 0.05")
    ax.axvline(best_cut, color="blue", linestyle=":", linewidth=1)
    ax.text(
        best_cut,
        best_p,
        f" best cutoff = {best_cut:.3g}",
        va="bottom",
        ha="left",
        fontsize=9,
    )
    ax.set_yscale("log")
    ax.set_xlabel(f"{score_col} cutpoint")
    ax.set_ylabel("Log-rank p-value")
    ax.set_title(f"Threshold scan for {score_col}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / f"threshold_scan_{score_col.lower()}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------
    # Define Low vs High group using best cutoff
    # -------------------------------------------------------------
    group_col = f"{score_col}_group"
    analysis_df[group_col] = np.where(
        analysis_df[score_col] <= best_cut,
        "Low",
        "High",
    )

    # -------------------------------------------------------------
    # Histogram of score
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(analysis_df[score_col].dropna(), bins=30, edgecolor="black")
    ax.axvline(best_cut, color="blue", linestyle=":", linewidth=1)
    ax.set_xlabel(score_col)
    ax.set_ylabel("Count")
    ax.set_title(f"{score_col} distribution")
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

    # -------------------------------------------------------------
    # Kaplan-Meier curves by best cutoff
    # -------------------------------------------------------------
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(7, 5))

    for group, grp in analysis_df.groupby(group_col):
        kmf.fit(
            grp["OS_MONTHS"],
            grp["OS_EVENT"],
            label=f"{group} (n={len(grp)})",
        )
        kmf.plot_survival_function(ax=ax, ci_show=True)

    low = analysis_df[group_col] == "Low"
    high = analysis_df[group_col] == "High"

    km_res = logrank_test(
        analysis_df.loc[low, "OS_MONTHS"],
        analysis_df.loc[high, "OS_MONTHS"],
        event_observed_A=analysis_df.loc[low, "OS_EVENT"],
        event_observed_B=analysis_df.loc[high, "OS_EVENT"],
    )

    ax.text(
        0.98,
        0.05,
        f"log-rank p = {km_res.p_value:.3g}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability")
    ax.set_title(f"Kaplan-Meier by {score_col} cutoff = {best_cut:.3g}")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / f"km_{score_col.lower()}_best_cutoff.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"KM log-rank p-value for {score_col}: {km_res.p_value}")

    scan_summary_rows.append({
        "score": score_col,
        "best_cutoff": best_cut,
        "best_logrank_p": best_p,
        "km_logrank_p": km_res.p_value,
    })

    # -------------------------------------------------------------
    # Univariate Cox model for this score
    # -------------------------------------------------------------
    log_score_col = f"log_{score_col}"
    analysis_df[log_score_col] = np.log1p(analysis_df[score_col])

    cox_df = analysis_df[["OS_MONTHS", "OS_EVENT", log_score_col]].dropna().copy()
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="OS_MONTHS", event_col="OS_EVENT")

    s = cph.summary.loc[log_score_col]
    hr = s["exp(coef)"]
    lo = s["exp(coef) lower 95%"]
    hi = s["exp(coef) upper 95%"]
    p = s["p"]
    c_index = cph.concordance_index_

    cox_comparison_rows.append({
        "score": score_col,
        "HR": hr,
        "CI_low": lo,
        "CI_high": hi,
        "p_value": p,
        "C_index": c_index,
    })

    print("\nUnivariate Cox summary:")
    print(cph.summary[["coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]])
    print("C-index:", c_index)

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
    print("Adjusted C-index:", cmvph.concordance_index_)

    # PH assumption check.
    print(f"\nChecking PH assumptions for {score_col} multivariate model...")
    cmvph.check_assumptions(
        cox_mv_df,
        p_value_threshold=0.05,
        show_plots=True,
        advice=True,
    )

    # Multivariate HR plot for the ODE score only.
    s_mv = cmvph.summary.loc[log_score_col]
    hr_mv = s_mv["exp(coef)"]
    lo_mv = s_mv["exp(coef) lower 95%"]
    hi_mv = s_mv["exp(coef) upper 95%"]

    fig, ax = plt.subplots(figsize=(5, 2.2))
    ax.errorbar(hr_mv, 0, xerr=[[hr_mv - lo_mv], [hi_mv - hr_mv]], fmt="o", color="black", capsize=4)
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
    summary["HR"] = summary["exp(coef)"]
    summary["low"] = summary["exp(coef) lower 95%"]
    summary["high"] = summary["exp(coef) upper 95%"]
    summary = summary.sort_values("HR")

    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(summary))))
    y = range(len(summary))

    ax.errorbar(
        summary["HR"],
        y,
        xerr=[summary["HR"] - summary["low"], summary["high"] - summary["HR"]],
        fmt="o",
        color="black",
        capsize=3,
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

cox_comparison_print = cox_comparison_df[["score", "HR", "95% CI", "p_value", "C_index"]].copy()
cox_comparison_print["HR"] = cox_comparison_print["HR"].map(lambda x: f"{x:.3g}")
cox_comparison_print["p_value"] = cox_comparison_print["p_value"].map(lambda x: f"{x:.3g}")
cox_comparison_print["C_index"] = cox_comparison_print["C_index"].map(lambda x: f"{x:.3f}")

cox_comparison_df.to_csv(OUT / "univariate_cox_comparison.csv", index=False)

print(f"\n{'=' * 70}")
print("Univariate Cox comparison table")
print(f"{'=' * 70}")
print(cox_comparison_print.to_string(index=False))


print("\nFinished successfully.")