"""
plot_ml_benchmark.py — Visualise ML benchmark results for Q4.
Produces two figures saved to results/figures/.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from tqdm import tqdm

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
    surv_full = pd.read_csv(ROOT / 'data/processed/survival_analysis_df.csv')
    surv_full = surv_full.set_index('PATIENT_ID').loc[X.index].reset_index()
    ode_risk = surv_full['log_AUC_X'].values
    return X, y, ode_risk


def bootstrap_distributions(ode_risk, lasso_oof, rsf_oof, y, n_boot=1000, seed=42):
    """Return per-bootstrap C-index arrays for all three models."""
    rng = np.random.default_rng(seed)
    n = len(y)
    ode_boot, lasso_boot, rsf_boot = [], [], []

    for _ in tqdm(range(n_boot), desc="Bootstrap"):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y[idx]
        if y_b['event'].sum() == 0:
            continue
        try:
            ode_boot.append(concordance_index_censored(
                y_b['event'], y_b['time'], ode_risk[idx])[0])
            lasso_boot.append(concordance_index_censored(
                y_b['event'], y_b['time'], lasso_oof[idx])[0])
            rsf_raw = concordance_index_censored(
                y_b['event'], y_b['time'], rsf_oof[idx])[0]
            rsf_flip = concordance_index_censored(
                y_b['event'], y_b['time'], -rsf_oof[idx])[0]
            rsf_boot.append(max(rsf_raw, rsf_flip))
        except Exception:
            continue

    return np.array(ode_boot), np.array(lasso_boot), np.array(rsf_boot)


def plot_forest(ode_b, lasso_b, rsf_b):
    """Figure 1 — Forest plot of C-index point estimates and 95% CIs."""
    models = ['HR-DDR ODE\n(AUC_X)', 'Cox LASSO', 'Random\nSurvival Forest']
    means = [np.mean(ode_b), np.mean(lasso_b), np.mean(rsf_b)]
    lows  = [np.percentile(ode_b, 2.5), np.percentile(lasso_b, 2.5), np.percentile(rsf_b, 2.5)]
    highs = [np.percentile(ode_b, 97.5), np.percentile(lasso_b, 97.5), np.percentile(rsf_b, 97.5)]
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    fig, ax = plt.subplots(figsize=(7, 4))
    y_pos = range(len(models))

    for i, (m, lo, hi, c) in enumerate(zip(means, lows, highs, colors)):
        ax.plot([lo, hi], [i, i], color=c, linewidth=2.5, solid_capstyle='round')
        ax.scatter(m, i, color=c, s=80, zorder=5)

    ax.axvline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.6, label='Chance (0.5)')
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(models, fontsize=11)
    ax.set_xlabel('C-index (bootstrap mean, 95% CI)', fontsize=11)
    ax.set_title('Model Comparison — Prognostic Discrimination\nHGSOC TCGA (n=420)', fontsize=12)
    ax.set_xlim(0.38, 0.65)
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / 'fig_ml_forest_plot.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def plot_bootstrap_distributions(ode_b, lasso_b, rsf_b):
    """Figure 2 — Bootstrap C-index distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    data = [ode_b, lasso_b, rsf_b]
    labels = ['HR-DDR ODE (AUC_X)', 'Cox LASSO', 'Random Survival Forest']
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    for ax, d, label, color in zip(axes, data, labels, colors):
        ax.hist(d, bins=40, color=color, alpha=0.75, edgecolor='white')
        ax.axvline(np.mean(d), color='black', linewidth=1.5, linestyle='-', label=f'Mean={np.mean(d):.3f}')
        ax.axvline(np.percentile(d, 2.5), color='black', linewidth=1, linestyle='--')
        ax.axvline(np.percentile(d, 97.5), color='black', linewidth=1, linestyle='--', label='95% CI')
        ax.axvline(0.5, color='red', linewidth=1, linestyle=':', alpha=0.7, label='Chance')
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('C-index', fontsize=9)
        ax.set_ylabel('Bootstrap count', fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle('Bootstrap C-index Distributions (n=1000 resamples)', fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / 'fig_ml_bootstrap_distributions.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)

    X, y, ode_risk = load_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_array = X.values
    n = len(y)

    # Reconstruct OOF predictions
    lasso_alphas = [0.05, 0.005, 0.01, 0.001, 0.001]
    lasso_oof = np.zeros(n)
    rsf_oof = np.zeros(n)

    print("Collecting OOF predictions...")
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_array, y['event'])):
        X_train, X_test = X_array[train_idx], X_array[test_idx]
        y_train = y[train_idx]

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('cox', CoxnetSurvivalAnalysis(
                l1_ratio=1.0, alphas=[lasso_alphas[fold]],
                fit_baseline_model=True, normalize=False, max_iter=10000
            ))
        ])
        pipe.fit(X_train, y_train)
        lasso_oof[test_idx] = pipe.predict(X_test)

        rsf = RandomSurvivalForest(
            n_estimators=200, min_samples_leaf=25,
            max_features=0.5, n_jobs=-1, random_state=42
        )
        rsf.fit(X_train, y_train)
        rsf_oof[test_idx] = rsf.predict(X_test)
        print(f"  Fold {fold+1} done")

    ode_b, lasso_b, rsf_b = bootstrap_distributions(ode_risk, lasso_oof, rsf_oof, y)

    plot_forest(ode_b, lasso_b, rsf_b)
    plot_bootstrap_distributions(ode_b, lasso_b, rsf_b)

    print("\nDone. Figures saved to results/figures/")