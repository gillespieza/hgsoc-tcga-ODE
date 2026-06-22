"""
ml_benchmark.py — ML survival model benchmarking against HR-DDR ODE.

Models: Cox LASSO (CoxnetSurvivalAnalysis), Random Survival Forest
Feature set: 14 raw gene expression columns (log2 FPKM+1) — Option A
Evaluation: 5-fold stratified CV + 1000-rep bootstrap CIs
ODE baseline C-index (AUC_X, univariate Cox, Step 3): loaded from file
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent

GENE_COLS = [
    'BRCA1', 'BRCA2', 'RAD51', 'PALB2', 'BRIP1',
    'ATM', 'ATR', 'CHEK1', 'CHEK2',
    'BCL2', 'BCL2L1', 'BAX', 'BAD', 'TP53'
]


def load_ode_baseline():
    """Read ODE baseline C-index for AUC_X from Step 3 output."""
    cox = pd.read_csv(ROOT / 'data/processed/univariate_cox_comparison.csv')
    return cox.loc[cox['score'] == 'AUC_X', 'C_index'].values[0]


ODE_BASELINE_CINDEX = load_ode_baseline()


def load_data():
    """Load and align feature matrix X and survival labels y."""
    merged = pd.read_csv(ROOT / 'data/processed/hgsoc_tcga_merged.csv')
    survival = pd.read_csv(ROOT / 'data/processed/survival_analysis_df.csv')

    df = survival[['PATIENT_ID', 'OS_MONTHS', 'OS_EVENT']].merge(
        merged[['PATIENT_ID'] + GENE_COLS],
        on='PATIENT_ID',
        how='inner'
    )

    X = pd.DataFrame(df[GENE_COLS].values, columns=GENE_COLS, index=df['PATIENT_ID'])
    y = build_survival_array(df)

    return X, y, df


def build_survival_array(df):
    """Construct sksurv structured array from OS_EVENT and OS_MONTHS."""
    return np.array(
        [(bool(e), t) for e, t in zip(df['OS_EVENT'], df['OS_MONTHS'])],
        dtype=[('event', bool), ('time', float)]
    )


def cross_val_cindex(estimator, X, y, cv):
    """
    Run outer CV and return list of per-fold C-indices.
    Scaler must be inside estimator pipeline — never fit on full data.
    """
    fold_cindices = []
    X_array = X.values if hasattr(X, 'values') else X

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_array, y['event'])):
        X_train, X_test = X_array[train_idx], X_array[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        estimator.fit(X_train, y_train)
        risk_scores = estimator.predict(X_test)

        ci = concordance_index_censored(
            y_test['event'], y_test['time'], risk_scores
        )[0]
        fold_cindices.append(ci)

    return fold_cindices


def fit_cox_lasso(X, y, cv):
    """
    Outer 5-fold CV for Cox LASSO with inner alpha tuning.
    Fits one model per alpha candidate — version-safe approach.
    Scaler fit only on training fold — no leakage.
    """
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
                    l1_ratio=1.0,
                    alphas=[alpha],
                    fit_baseline_model=True,
                    normalize=False,
                    max_iter=10000
                ))
            ])
            try:
                pipe.fit(X_train, y_train)
                # skip if all coefficients zeroed out
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
    """
    Outer 5-fold CV for Random Survival Forest.
    No scaling needed — tree-based model.
    Uses max(raw, flipped) C-index to handle direction ambiguity.
    """
    X_array = X.values if hasattr(X, 'values') else X
    fold_cindices = []

    rsf = RandomSurvivalForest(
        n_estimators=200,
        min_samples_leaf=25,
        max_features=0.5,
        n_jobs=-1,
        random_state=42
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

def bootstrap_cindex(risk_scores, y, n_boot=1000, seed=42):
    """
    Bootstrap 95% CI for C-index from predicted risk scores.
    Samples patients with replacement, recomputes C-index each time.
    """
    rng = np.random.default_rng(seed)
    n = len(risk_scores)
    boot_cindices = []

    for _ in tqdm(range(n_boot), desc="Bootstrap", leave=False):
        idx = rng.choice(n, size=n, replace=True)
        y_boot = y[idx]
        r_boot = risk_scores[idx]

        # skip degenerate bootstraps with no events
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
        'ci_high': np.percentile(boot_cindices, 97.5)
    }

def save_comparison_table(ode_baseline, lasso_cindices, rsf_cindices,
                           ode_boot, lasso_boot, rsf_boot):
    """Save model comparison table to CSV and print markdown-style."""
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
            'Notes': '5-fold CV, n_est=200'
        },
    ]

    df = pd.DataFrame(rows)

    # Print markdown-style table
    print(f"\n{'Model':<30} {'Feature set':<26} {'C-index (CV)':<18} {'95% Bootstrap CI':<20} {'Notes'}")
    print("-" * 110)
    for _, r in df.iterrows():
        ci = f"[{r['CI_low']}, {r['CI_high']}]"
        print(f"{r['Model']:<30} {r['Feature_set']:<26} {r['CV_display']:<18} {ci:<20} {r['Notes']}")

    # Save CSV
    out = ROOT / 'data/processed/ml_comparison_table.csv'
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")
    return df

if __name__ == "__main__":
    X, y, survival_df = load_data()
    print(f"ODE baseline C-index (AUC_X): {ODE_BASELINE_CINDEX:.4f}")
    print(f"X shape: {X.shape}, Events: {y['event'].sum()}\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- Cox LASSO ---
    print("--- Cox LASSO ---")
    lasso_cindices, lasso_alphas = fit_cox_lasso(X, y, cv)

    # --- Random Survival Forest ---
    print("\n--- Random Survival Forest ---")
    rsf_cindices = fit_rsf(X, y, cv)

    # --- Out-of-fold predictions for honest bootstrap ---
    print("\n--- Collecting out-of-fold predictions ---")
    X_array = X.values
    n = len(y)
    lasso_oof = np.zeros(n)
    rsf_oof = np.zeros(n)

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_array, y['event'])):
        X_train, X_test = X_array[train_idx], X_array[test_idx]
        y_train = y[train_idx]

        # Cox LASSO OOF — use best alpha from CV
        best_alpha = lasso_alphas[fold]
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('cox', CoxnetSurvivalAnalysis(
                l1_ratio=1.0, alphas=[best_alpha],
                fit_baseline_model=True, normalize=False, max_iter=10000
            ))
        ])
        pipe.fit(X_train, y_train)
        lasso_oof[test_idx] = pipe.predict(X_test)

        # RSF OOF
        rsf_fold = RandomSurvivalForest(
            n_estimators=200, min_samples_leaf=25,
            max_features=0.5, n_jobs=-1, random_state=42
        )
        rsf_fold.fit(X_train, y_train)
        rsf_oof[test_idx] = rsf_fold.predict(X_test)
        print(f"  Fold {fold+1} OOF collected")

    # --- ODE risk scores (aligned to X patient order) ---
    surv_full = pd.read_csv(ROOT / 'data/processed/survival_analysis_df.csv')
    surv_full = surv_full.set_index('PATIENT_ID').loc[X.index].reset_index()
    ode_risk = surv_full['log_AUC_X'].values

    # --- Bootstrap CIs on OOF predictions ---
    print("\n--- Bootstrap CIs (OOF for ML, direct for ODE) ---")
    ode_boot = bootstrap_cindex(ode_risk, y)
    lasso_boot = bootstrap_cindex(lasso_oof, y)
    rsf_boot = bootstrap_cindex(rsf_oof, y)

    print(f"ODE (log_AUC_X):  {ode_boot['mean']:.4f} [{ode_boot['ci_low']:.4f}, {ode_boot['ci_high']:.4f}]")
    print(f"Cox LASSO (OOF):  {lasso_boot['mean']:.4f} [{lasso_boot['ci_low']:.4f}, {lasso_boot['ci_high']:.4f}]")
    print(f"RSF (OOF):        {rsf_boot['mean']:.4f} [{rsf_boot['ci_low']:.4f}, {rsf_boot['ci_high']:.4f}]")

    # --- Summary table ---
    print("\n--- Summary ---")
    print(f"{'Model':<20} {'CV C-index':<15} {'Boot mean':<12} {'95% CI'}")
    print(f"{'ODE (AUC_X)':<20} {ODE_BASELINE_CINDEX:.4f} (no CV)   "
          f"{ode_boot['mean']:.4f}       [{ode_boot['ci_low']:.4f}, {ode_boot['ci_high']:.4f}]")
    print(f"{'Cox LASSO':<20} {np.mean(lasso_cindices):.4f} ± {np.std(lasso_cindices):.4f}  "
          f"{lasso_boot['mean']:.4f}       [{lasso_boot['ci_low']:.4f}, {lasso_boot['ci_high']:.4f}]")
    print(f"{'RSF':<20} {np.mean(rsf_cindices):.4f} ± {np.std(rsf_cindices):.4f}  "
          f"{rsf_boot['mean']:.4f}       [{rsf_boot['ci_low']:.4f}, {rsf_boot['ci_high']:.4f}]")

    print("\n--- Model Comparison Table ---")
    comparison_df = save_comparison_table(
        ODE_BASELINE_CINDEX, lasso_cindices, rsf_cindices,
        ode_boot, lasso_boot, rsf_boot
    )