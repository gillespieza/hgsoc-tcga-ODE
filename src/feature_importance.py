"""
feature_importance.py — Feature importance analysis for ML benchmark models.

A) Cox LASSO coefficients (full-data fit, best alpha = 0.001)
B) RSF permutation feature importances (full-data fit)

Both fit on full 420 patients — interpretation only, not prediction.
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / 'results' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

GENE_COLS = [
    'BRCA1', 'BRCA2', 'RAD51', 'PALB2', 'BRIP1',
    'ATM', 'ATR', 'CHEK1', 'CHEK2',
    'BCL2', 'BCL2L1', 'BAX', 'BAD', 'TP53'
]


def load_data():
    merged = pd.read_csv(ROOT / 'data/processed/hgsoc_tcga_merged.csv')
    survival = pd.read_csv(ROOT / 'data/processed/survival_analysis_df.csv')
    df = survival[['PATIENT_ID', 'OS_MONTHS', 'OS_EVENT']].merge(
        merged[['PATIENT_ID'] + GENE_COLS], on='PATIENT_ID', how='inner'
    )
    X = pd.DataFrame(df[GENE_COLS].values, columns=GENE_COLS, index=df['PATIENT_ID'])
    y = np.array(
        [(bool(e), t) for e, t in zip(df['OS_EVENT'], df['OS_MONTHS'])],
        dtype=[('event', bool), ('time', float)]
    )
    return X, y


def fit_cox_lasso_full(X, y, alpha=0.001):
    """Fit Cox LASSO on full data, return coefficients."""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('cox', CoxnetSurvivalAnalysis(
            l1_ratio=1.0, alphas=[alpha],
            fit_baseline_model=True, normalize=False, max_iter=10000
        ))
    ])
    pipe.fit(X.values, y)
    coefs = pipe.named_steps['cox'].coef_.flatten()
    return pd.Series(coefs, index=GENE_COLS).sort_values()


def fit_rsf_full(X, y):
    """Fit RSF on full data, return permutation importances."""
    rsf = RandomSurvivalForest(
        n_estimators=200, min_samples_leaf=25,
        max_features=0.5, n_jobs=-1, random_state=42
    )
    rsf.fit(X.values, y)

    result = permutation_importance(
        rsf, X.values, y,
        n_repeats=20,
        random_state=42,
        n_jobs=-1
    )
    importances = pd.Series(
        result.importances_mean, index=GENE_COLS
    ).sort_values(ascending=False)
    return importances, result.importances_std


def plot_cox_coefficients(coefs):
    """Figure 3 — Cox LASSO coefficients."""
    nonzero = coefs[coefs != 0]

    fig, ax = plt.subplots(figsize=(7, max(3, len(coefs) * 0.45)))
    colors = ['#d32f2f' if v > 0 else '#1976D2' for v in coefs.values]
    bars = ax.barh(range(len(coefs)), coefs.values, color=colors, edgecolor='white', height=0.6)
    ax.set_yticks(range(len(coefs)))
    ax.set_yticklabels(coefs.index, fontsize=10)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Cox LASSO coefficient (standardised)', fontsize=10)
    ax.set_title(f'Cox LASSO Feature Coefficients\n(alpha=0.001, full data fit)\n'
                 f'{len(nonzero)}/{len(coefs)} non-zero coefficients', fontsize=11)

    # Colour legend
    red_patch = plt.matplotlib.patches.Patch(color='#d32f2f', label='Higher risk (↑ hazard)')
    blue_patch = plt.matplotlib.patches.Patch(color='#1976D2', label='Protective (↓ hazard)')
    ax.legend(handles=[red_patch, blue_patch], fontsize=9, loc='lower right')

    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / 'fig_cox_lasso_coefficients.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")
    return out


def plot_rsf_importances(importances, stds):
    """Figure 4 — RSF permutation feature importances."""
    fig, ax = plt.subplots(figsize=(7, max(3, len(importances) * 0.45)))
    colors = ['#4CAF50' if v > 0 else '#9E9E9E' for v in importances.values]
    ax.barh(range(len(importances)), importances.values,
            xerr=stds[importances.index.map(lambda x: GENE_COLS.index(x))],
            color=colors, edgecolor='white', height=0.6,
            error_kw={'elinewidth': 1, 'capsize': 3})
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels(importances.index, fontsize=10)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Permutation importance (mean C-index drop)', fontsize=10)
    ax.set_title('RSF Permutation Feature Importances\n(n=20 repeats, full data fit)', fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / 'fig_rsf_feature_importances.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")
    return out

def main():
    X, y = load_data()
    print(f"Loaded: {X.shape[0]} patients, {X.shape[1]} genes\n")

    # --- Cox LASSO ---
    print("--- Cox LASSO Coefficients (alpha=0.001) ---")
    coefs = fit_cox_lasso_full(X, y, alpha=0.001)
    print(coefs.to_string())
    plot_cox_coefficients(coefs)

    # --- RSF ---
    print("\n--- RSF Permutation Importances (n=20 repeats) ---")
    importances, stds = fit_rsf_full(X, y)
    print(importances.to_string())
    plot_rsf_importances(importances, stds)

    # --- Save table ---
    feat_df = pd.DataFrame({
        'gene': GENE_COLS,
        'cox_lasso_coef': coefs.reindex(GENE_COLS).values,
        'rsf_importance_mean': importances.reindex(GENE_COLS).values,
        'rsf_importance_std': stds
    })
    out = ROOT / 'data/processed/feature_importance_table.csv'
    feat_df.to_csv(out, index=False)
    print(f"\nFeature table saved to {out}")

if __name__ == "__main__":
    main()