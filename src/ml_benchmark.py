"""
ml_benchmark.py — ML survival model benchmarking against HR-DDR ODE.

Models: Cox LASSO (CoxnetSurvivalAnalysis), Random Survival Forest
Feature set: 14 raw gene expression columns (log2 FPKM+1)
Evaluation: 5-fold stratified CV + 1000-rep bootstrap CIs
Figures: forest plot + bootstrap distributions saved to results/figures/
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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


def load_ode_baseline():
    cox = pd.read_csv(ROOT / 'data/processed/univariate_cox_comparison.csv')
    return cox.loc[cox['score'] == 'AUC_X', 'C_index'].values[0]


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
    return X, y, df


def build_survival_array(df):
    return np.array(
        [(bool(e), t) for e, t in zip(df['OS_EVENT'], df['OS_MONTHS'])],
        dtype=[('event', bool), ('time', float)]
    )


def bootstrap_cindex(risk_scores, y, n_boot=1000, seed=42, desc="Bootstrap"):
    rng = np.random.default_rng(seed)
    n = len(risk_scores)
    boot_cindices = []
    for _ in tqdm(range(n_boot), desc=desc, leave=False):
        idx = rng.choice(n, size=n, replace=True)
        y_boot = y[idx]
        r_boot = risk_scores[idx]
        if y_boot['event'].sum() == 0:
            continue
        try:
            ci = concordance_index_censored(
                y_boot['event'], y_boot['time'], r_boot
            )[0]
            boot_cindices.append(ci)
        except Exception:
            continue
    boot_cindices = np.array(boot_cindices)
    return {
        'mean': np.mean(boot_cindices),
        'ci_low': np.percentile(boot_cindices, 2.5),
        'ci_high': np.percentile(boot_cindices, 97.5),
        'samples': boot_cindices
    }


def fit_cox_lasso(X, y, cv):
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    fold_cindices = []
    best_alphas = []
    X_array = X.values if hasattr(X, 'values') else X

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_array, y['event'])):
        X_train, X_test = X_array[train_idx], X_array[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        best_ci, best_alpha = -1, alphas[0]
        for alpha in alphas:
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('cox', CoxnetSurvivalAnalysis(
                    l1_ratio=1.0, alphas=[alpha],
                    fit_baseline_model=True, normalize=False, max_iter=10000
                ))
            ])
            try:
                pipe.fit(X_train, y_train)
                coefs = pipe.named_steps['cox'].coef_
                if np.all(coefs == 0):
                    continue
                risk = pipe.predict(X_test)
                ci = concordance_index_censored(
                    y_test['event'], y_test['time'], risk
                )[0]
                if ci > best_ci:
                    best_ci = ci
                    best_alpha = alpha
            except Exception as e:
                print(f"  Fold {fold+1} alpha={alpha} failed: {e}")

        fold_cindices.append(best_ci)
        best_alphas.append(best_alpha)
        print(f"  Fold {fold+1}: C-index={best_ci:.4f}, best_alpha={best_alpha}")

    print(f"\nCox LASSO mean C-index: {np.mean(fold_cindices):.4f} ± {np.std(fold_cindices):.4f}")
    print(f"Best alphas per fold: {best_alphas}")
    return fold_cindices, best_alphas


def fit_rsf(X, y, cv):
    X_array = X.values if hasattr(X, 'values') else X
    fold_cindices = []

    rsf = RandomSurvivalForest(
        n_estimators=200, min_samples_leaf=25,
        max_features=0.5, n_jobs=-1, random_state=42
    )

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_array, y['event'])):
        X_train, X_test = X_array[train_idx], X_array[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        rsf.fit(X_train, y_train)
        risk = rsf.predict(X_test)

        ci_raw = concordance_index_censored(y_test['event'], y_test['time'], risk)[0]
        ci_flipped = concordance_index_censored(y_test['event'], y_test['time'], -risk)[0]
        ci = max(ci_raw, ci_flipped)
        print(f"  Fold {fold+1}: C-index={ci:.4f} (raw={ci_raw:.4f}, flipped={ci_flipped:.4f})")
        fold_cindices.append(ci)

    print(f"\nRSF mean C-index: {np.mean(fold_cindices):.4f} ± {np.std(fold_cindices):.4f}")
    return fold_cindices


def collect_oof_predictions(X, y, cv, lasso_alphas):
    X_array = X.values
    n = len(y)
    lasso_oof = np.zeros(n)
    rsf_oof = np.zeros(n)

    print("Collecting out-of-fold predictions...")
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

        rsf_fold = RandomSurvivalForest(
            n_estimators=200, min_samples_leaf=25,
            max_features=0.5, n_jobs=-1, random_state=42
        )
        rsf_fold.fit(X_train, y_train)
        rsf_oof[test_idx] = rsf_fold.predict(X_test)
        print(f"  Fold {fold+1} OOF collected")

    return lasso_oof, rsf_oof


def save_comparison_table(ode_baseline, lasso_cindices, rsf_cindices,
                           ode_boot, lasso_boot, rsf_boot):
    rows = [
        {
            'Model': 'HR-DDR ODE (AUC_X)',
            'Feature_set': '14 genes → 4 params → ODE',
            'CV_cindex': f"{ode_baseline:.3f}",
            'CV_sd': '',
            'CV_display': f"{ode_baseline:.3f}",
            'CI_low': f"{ode_boot['ci_low']:.3f}",
            'CI_high': f"{ode_boot['ci_high']:.3f}",
            'Notes': 'No CV needed (zero-shot)'
        },
        {
            'Model': 'Cox LASSO',
            'Feature_set': '14 genes (raw)',
            'CV_cindex': f"{np.mean(lasso_cindices):.3f}",
            'CV_sd': f"{np.std(lasso_cindices):.3f}",
            'CV_display': f"{np.mean(lasso_cindices):.3f} ± {np.std(lasso_cindices):.3f}",
            'CI_low': f"{lasso_boot['ci_low']:.3f}",
            'CI_high': f"{lasso_boot['ci_high']:.3f}",
            'Notes': '5-fold CV, best alpha'
        },
        {
            'Model': 'Random Survival Forest',
            'Feature_set': '14 genes (raw)',
            'CV_cindex': f"{np.mean(rsf_cindices):.3f}",
            'CV_sd': f"{np.std(rsf_cindices):.3f}",
            'CV_display': f"{np.mean(rsf_cindices):.3f} ± {np.std(rsf_cindices):.3f}",
            'CI_low': f"{rsf_boot['ci_low']:.3f}",
            'CI_high': f"{rsf_boot['ci_high']:.3f}",
            'Notes': '5-fold CV, direction corrected per fold'
        },
    ]
    df = pd.DataFrame(rows)
    print(f"\n{'Model':<30} {'Feature set':<26} {'C-index (CV)':<18} {'95% Bootstrap CI':<20} {'Notes'}")
    print("-" * 110)
    for _, r in df.iterrows():
        ci = f"[{r['CI_low']}, {r['CI_high']}]"
        print(f"{r['Model']:<30} {r['Feature_set']:<26} {r['CV_display']:<18} {ci:<20} {r['Notes']}")
    out = ROOT / 'data/processed/ml_comparison_table.csv'
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    return df


def plot_forest(ode_b, lasso_b, rsf_b):
    models = ['HR-DDR ODE\n(AUC_X)', 'Cox LASSO', 'Random\nSurvival Forest']
    means  = [np.mean(ode_b),  np.mean(lasso_b),  np.mean(rsf_b)]
    lows   = [np.percentile(ode_b, 2.5),  np.percentile(lasso_b, 2.5),  np.percentile(rsf_b, 2.5)]
    highs  = [np.percentile(ode_b, 97.5), np.percentile(lasso_b, 97.5), np.percentile(rsf_b, 97.5)]
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (m, lo, hi, c) in enumerate(zip(means, lows, highs, colors)):
        ax.plot([lo, hi], [i, i], color=c, linewidth=2.5, solid_capstyle='round')
        ax.scatter(m, i, color=c, s=80, zorder=5)

    ax.axvline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.6, label='Chance (0.5)')
    ax.set_yticks(list(range(len(models))))
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
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    data   = [ode_b, lasso_b, rsf_b]
    labels = ['HR-DDR ODE (AUC_X)', 'Cox LASSO', 'Random Survival Forest']
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    for ax, d, label, color in zip(axes, data, labels, colors):
        ax.hist(d, bins=40, color=color, alpha=0.75, edgecolor='white')
        ax.axvline(np.mean(d), color='black', linewidth=1.5, label=f'Mean={np.mean(d):.3f}')
        ax.axvline(np.percentile(d, 2.5),  color='black', linewidth=1, linestyle='--')
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


def main():
    ODE_BASELINE_CINDEX = load_ode_baseline()
    X, y, survival_df = load_data()
    print(f"ODE baseline C-index (AUC_X): {ODE_BASELINE_CINDEX:.4f}")
    print(f"X shape: {X.shape}, Events: {y['event'].sum()}\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- Fold event rate check ---
    for fold, (train_idx, test_idx) in enumerate(cv.split(X.values, y['event'])):
        y_test = y[test_idx]
        rate = y_test['event'].sum() / len(y_test)
        print(f"Fold {fold+1}: {rate:.3f} ({y_test['event'].sum()} events / {len(y_test)} patients)")

    print("\n--- Cox LASSO ---")
    lasso_cindices, lasso_alphas = fit_cox_lasso(X, y, cv)

    print("\n--- Random Survival Forest ---")
    rsf_cindices = fit_rsf(X, y, cv)

    # --- OOF predictions ---
    print()
    lasso_oof, rsf_oof = collect_oof_predictions(X, y, cv, lasso_alphas)

    # --- ODE risk scores ---
    surv_full = pd.read_csv(ROOT / 'data/processed/survival_analysis_df.csv')
    surv_full = surv_full.set_index('PATIENT_ID').loc[X.index].reset_index()
    ode_risk = surv_full['log_AUC_X'].values

    # --- Bootstrap CIs ---
    print("\n--- Bootstrap CIs (OOF for ML, direct for ODE) ---")
    ode_boot   = bootstrap_cindex(ode_risk,   y, desc="ODE bootstrap")
    lasso_boot = bootstrap_cindex(lasso_oof,  y, desc="LASSO bootstrap")
    rsf_boot   = bootstrap_cindex(rsf_oof,    y, desc="RSF bootstrap")

    print(f"ODE (log_AUC_X):  {ode_boot['mean']:.4f} [{ode_boot['ci_low']:.4f}, {ode_boot['ci_high']:.4f}]")
    print(f"Cox LASSO (OOF):  {lasso_boot['mean']:.4f} [{lasso_boot['ci_low']:.4f}, {lasso_boot['ci_high']:.4f}]")
    print(f"RSF (OOF):        {rsf_boot['mean']:.4f} [{rsf_boot['ci_low']:.4f}, {rsf_boot['ci_high']:.4f}]")

    # --- Comparison table ---
    print("\n--- Model Comparison Table ---")
    save_comparison_table(
        ODE_BASELINE_CINDEX, lasso_cindices, rsf_cindices,
        ode_boot, lasso_boot, rsf_boot
    )

    # --- Figures ---
    print("\n--- Generating figures ---")
    plot_forest(ode_boot['samples'], lasso_boot['samples'], rsf_boot['samples'])
    plot_bootstrap_distributions(ode_boot['samples'], lasso_boot['samples'], rsf_boot['samples'])


if __name__ == "__main__":
    main()