"""
kaplan_meier.py — Kaplan-Meier survival analysis stratified by ODE risk score.

Splits patients into High/Low AUC_X groups at median.
Computes log-rank test p-value.
Saves KM figure to results/figures/.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / 'results' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(ROOT / 'data/processed/survival_analysis_df.csv')
    return df


def stratify_and_plot(df):
    median_auc = df['log_AUC_X'].median()
    df = df.copy()
    df['risk_group'] = np.where(df['log_AUC_X'] >= median_auc, 'High AUC_X', 'Low AUC_X')

    high = df[df['risk_group'] == 'High AUC_X']
    low  = df[df['risk_group'] == 'Low AUC_X']

    # Log-rank test
    results = logrank_test(
        high['OS_MONTHS'], low['OS_MONTHS'],
        event_observed_A=high['OS_EVENT'],
        event_observed_B=low['OS_EVENT']
    )
    p_val = results.p_value

    # Median survival
    kmf_high = KaplanMeierFitter()
    kmf_low  = KaplanMeierFitter()
    kmf_high.fit(high['OS_MONTHS'], high['OS_EVENT'], label=f'High AUC_X (n={len(high)})')
    kmf_low.fit(low['OS_MONTHS'],  low['OS_EVENT'],  label=f'Low AUC_X (n={len(low)})')

    med_high = kmf_high.median_survival_time_
    med_low  = kmf_low.median_survival_time_

    print(f"Median AUC_X split: {median_auc:.4f}")
    print(f"High group: n={len(high)}, median OS={med_high:.1f} months")
    print(f"Low group:  n={len(low)},  median OS={med_low:.1f} months")
    print(f"Log-rank p-value: {p_val:.4f}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))

    kmf_high.plot_survival_function(
        ax=ax, ci_show=True, color='#d32f2f',
        ci_alpha=0.12, linewidth=2
    )
    kmf_low.plot_survival_function(
        ax=ax, ci_show=True, color='#1976D2',
        ci_alpha=0.12, linewidth=2
    )

    # Annotate p-value and medians
    p_str = f'p = {p_val:.4f}' if p_val >= 0.0001 else 'p < 0.0001'
    ax.text(0.97, 0.97,
            f'{p_str}\nMedian OS (High): {med_high:.1f} mo\nMedian OS (Low):  {med_low:.1f} mo',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cccccc', alpha=0.9))

    ax.set_xlabel('Time (months)', fontsize=11)
    ax.set_ylabel('Survival probability', fontsize=11)
    ax.set_title('Kaplan-Meier Survival — Stratified by HR-DDR ODE Score (AUC_X)\n'
                 'HGSOC TCGA (n=420), median split', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')

    plt.tight_layout()
    out = FIG_DIR / 'fig_kaplan_meier_aucx.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out}")

    return p_val, med_high, med_low


def main():
    df = load_data()
    print(f"Loaded: {len(df)} patients\n")
    p_val, med_high, med_low = stratify_and_plot(df)

    p_str = f'{p_val:.4f}' if p_val >= 0.0001 else '< 0.0001'
    print(f"Patients in the high AUC_X group had median OS of {med_high:.1f} months "
          f"vs {med_low:.1f} months in the low group (log-rank p={p_str}).")

if __name__ == "__main__":
    main()