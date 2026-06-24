"""
feature_importance.py — Feature importance analysis for ML benchmark models.

Fits both ML models on the full cohort (interpretation only — not prediction)
and extracts:

A) Cox LASSO coefficients
   Alpha is selected by 5-fold CV on the full dataset so the coefficient
   magnitudes reflect a regularisation level supported by the data, not a
   hardcoded guess.

B) RSF permutation feature importances (20 repeats)
   Permutation importance is computed on the training data (see NOTE below).

NOTE on in-sample permutation importance:
    The RSF is fitted and scored on the same patients. This is standard
    practice for interpretation-only full-data fits but the importances are
    optimistic: they reflect what the model learned, not out-of-sample signal.
    Results should be read as "which genes the RSF used", not as validated
    predictors of held-out survival.

Both fits use the same 14-gene expression feature set as ml_benchmark.py.
All outputs are labelled interpretation-only to distinguish them from the
cross-validated performance estimates produced by ml_benchmark.py.

Outputs
-------
- data/processed/feature_importance_table.csv
- results/figures/fig_cox_lasso_coefficients.png
- results/figures/fig_rsf_feature_importances.png
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)

# =================================================================
# Constants
# =================================================================

GENE_COLS = [
    "BRCA1", "BRCA2", "RAD51", "PALB2", "BRIP1",
    "ATM", "ATR", "CHEK1", "CHEK2",
    "BCL2", "BCL2L1", "BAX", "BAD", "TP53",
]

# Candidate regularisation strengths for full-data CV alpha selection.
LASSO_ALPHAS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]


# =================================================================
# Data loading
# =================================================================

def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load and align the merged expression dataset with the survival table.

    Inner-joins on PATIENT_ID so the feature importance analysis operates
    on exactly the same cohort as ml_benchmark.py.

    Returns
    -------
    X : pd.DataFrame, shape (n_patients, n_genes)
        Log2(FPKM+1) expression values, indexed by PATIENT_ID.
    y : np.ndarray
        Structured array with fields ('event', bool) and ('time', float).
    """
    merged = pd.read_csv(
        ROOT / "data" / "processed" / "hgsoc_tcga_merged.csv"
    )
    survival = pd.read_csv(
        ROOT / "data" / "processed" / "survival_analysis_df.csv"
    )

    df = survival[["PATIENT_ID", "OS_MONTHS", "OS_EVENT"]].merge(
        merged[["PATIENT_ID"] + GENE_COLS],
        on="PATIENT_ID",
        how="inner",
    )

    logger.info(
        f"Loaded cohort for feature importance: "
        f"{len(df)} patients, {len(GENE_COLS)} genes"
    )

    X = pd.DataFrame(
        df[GENE_COLS].values,
        columns=GENE_COLS,
        index=df["PATIENT_ID"],
    )
    y = np.array(
        [
            (bool(e), float(t))
            for e, t in zip(df["OS_EVENT"], df["OS_MONTHS"])
        ],
        dtype=[("event", bool), ("time", float)],
    )
    return X, y


# =================================================================
# Model fitting
# =================================================================

def _select_alpha_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> float:
    """
    Select the best Cox LASSO alpha by stratified K-fold CV on the full dataset.

    Selecting alpha by CV rather than hardcoding ensures the regularisation
    level is supported by the data. The selected alpha governs coefficient
    sparsity and therefore which genes appear as non-zero in the
    interpretation plot.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_patients × n_genes).
    y : np.ndarray
        Structured survival array.
    n_splits : int
        Number of CV folds.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    float
        Alpha with the highest mean CV C-index across folds.
        Falls back to the smallest alpha if every fit fails.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    alpha_scores: dict[float, list[float]] = {a: [] for a in LASSO_ALPHAS}

    for alpha in LASSO_ALPHAS:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("cox", CoxnetSurvivalAnalysis(
                l1_ratio=1.0,
                alphas=[alpha],
                fit_baseline_model=True,
                normalize=False,
                max_iter=10000,
            )),
        ])

        for train_idx, val_idx in cv.split(X, y["event"]):
            try:
                pipe.fit(X[train_idx], y[train_idx])

                if np.all(pipe.named_steps["cox"].coef_ == 0):
                    continue

                risk = pipe.predict(X[val_idx])
                ci = concordance_index_censored(
                    y[val_idx]["event"], y[val_idx]["time"], risk
                )[0]
                alpha_scores[alpha].append(ci)

            except Exception:
                logger.warning(
                    f"CV fold failed for alpha={alpha}; skipping."
                )

    best_alpha = LASSO_ALPHAS[0]
    best_mean = -1.0

    for alpha, scores in alpha_scores.items():
        if scores and np.mean(scores) > best_mean:
            best_mean = np.mean(scores)
            best_alpha = alpha

    logger.info(
        f"CV alpha selection: best_alpha={best_alpha} "
        f"(mean C-index={best_mean:.4f})"
    )
    return best_alpha


def fit_cox_lasso_full(
    X: pd.DataFrame,
    y: np.ndarray,
) -> tuple[pd.Series, float]:
    """
    Fit Cox LASSO on the full dataset and return sorted coefficients.

    Alpha is chosen by 5-fold CV on the full dataset so the interpretation
    reflects a regularisation level supported by the data. This fit is for
    coefficient inspection only — predictive performance is reported by
    ml_benchmark.py using out-of-fold evaluation.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (n_patients × n_genes).
    y : np.ndarray
        Structured survival array with ('event', bool) and ('time', float).

    Returns
    -------
    coefs : pd.Series
        Standardised Cox LASSO coefficients, sorted ascending.
    alpha : float
        CV-selected regularisation strength used for the fit.
    """
    best_alpha = _select_alpha_cv(X.values, y)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("cox", CoxnetSurvivalAnalysis(
            l1_ratio=1.0,
            alphas=[best_alpha],
            fit_baseline_model=True,
            normalize=False,
            max_iter=10000,
        )),
    ])
    pipe.fit(X.values, y)

    coefs = pipe.named_steps["cox"].coef_.flatten()
    return pd.Series(coefs, index=GENE_COLS).sort_values(), best_alpha


def fit_rsf_full(
    X: pd.DataFrame,
    y: np.ndarray,
) -> tuple[pd.Series, np.ndarray]:
    """
    Fit RSF on the full dataset and return permutation feature importances.

    NOTE: permutation importance is evaluated on the same data used for
    fitting. This is an in-sample measure — it reflects what the model
    learned rather than out-of-sample signal. Results should be read as
    "which genes drove the RSF's risk ranking", not as validated predictors.
    See module docstring for further discussion.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (n_patients × n_genes).
    y : np.ndarray
        Structured survival array.

    Returns
    -------
    importances : pd.Series
        Mean permutation importance per gene, sorted descending.
    stds : np.ndarray
        Standard deviation of permutation importance across repeats,
        in GENE_COLS order (for alignment with importances via index mapping).
    """
    rsf = RandomSurvivalForest(
        n_estimators=200,
        min_samples_leaf=25,
        max_features=0.5,
        n_jobs=4,
        random_state=42,
    )
    rsf.fit(X.values, y)

    result = permutation_importance(
        rsf, X.values, y,
        n_repeats=20,
        random_state=42,
        n_jobs=4,
    )

    importances = pd.Series(
        result.importances_mean, index=GENE_COLS
    ).sort_values(ascending=False)

    return importances, result.importances_std


# =================================================================
# Figures
# =================================================================

def plot_cox_coefficients(coefs: pd.Series, alpha: float) -> None:
    """
    Horizontal bar chart of Cox LASSO coefficients sorted by magnitude.

    Red bars indicate positive coefficients (higher expression → higher
    hazard); blue bars indicate negative coefficients (protective). Zero
    coefficients are LASSO-shrunk to exactly zero and remain visible as
    absent bars.

    Parameters
    ----------
    coefs : pd.Series
        Standardised coefficients sorted ascending (from fit_cox_lasso_full).
    alpha : float
        CV-selected alpha shown in the figure title for transparency.
    """
    nonzero = coefs[coefs != 0]

    fig, ax = plt.subplots(figsize=(7, max(3, len(coefs) * 0.45)))
    colors = ["#d32f2f" if v > 0 else "#1976D2" for v in coefs.values]

    ax.barh(
        range(len(coefs)), coefs.values,
        color=colors, edgecolor="white", height=0.6,
    )
    ax.set_yticks(range(len(coefs)))
    ax.set_yticklabels(coefs.index, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Cox LASSO coefficient (standardised)", fontsize=10)
    ax.set_title(
        f"Cox LASSO Feature Coefficients\n"
        f"(alpha={alpha}, CV-selected, full-data fit — interpretation only)\n"
        f"{len(nonzero)}/{len(coefs)} non-zero coefficients",
        fontsize=11,
    )

    red_patch  = plt.matplotlib.patches.Patch(
        color="#d32f2f", label="Higher risk (↑ hazard)"
    )
    blue_patch = plt.matplotlib.patches.Patch(
        color="#1976D2", label="Protective (↓ hazard)"
    )
    ax.legend(handles=[red_patch, blue_patch], fontsize=9, loc="lower right")

    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "fig_cox_lasso_coefficients.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")


def plot_rsf_importances(
    importances: pd.Series,
    stds: np.ndarray,
) -> None:
    """
    Horizontal bar chart of RSF permutation feature importances with error bars.

    Error bars show ±1 SD across the 20 permutation repeats. Positive
    importances (green) indicate genes whose permutation reduced the model's
    C-index; near-zero or negative importances (grey) suggest the model did
    not rely on those genes.

    Parameters
    ----------
    importances : pd.Series
        Mean permutation importance per gene, sorted descending.
    stds : np.ndarray
        Per-gene standard deviation in GENE_COLS order; reindexed internally
        to match the sorted importance order.
    """
    fig, ax = plt.subplots(figsize=(7, max(3, len(importances) * 0.45)))
    colors = ["#4CAF50" if v > 0 else "#9E9E9E" for v in importances.values]

    # Reindex stds from GENE_COLS order to the sorted importance order.
    sorted_stds = stds[importances.index.map(lambda g: GENE_COLS.index(g))]

    ax.barh(
        range(len(importances)), importances.values,
        xerr=sorted_stds,
        color=colors, edgecolor="white", height=0.6,
        error_kw={"elinewidth": 1, "capsize": 3},
    )
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels(importances.index, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Permutation importance (mean C-index drop)", fontsize=10)
    ax.set_title(
        "RSF Permutation Feature Importances\n"
        "(n=20 repeats, full-data fit — interpretation only)",
        fontsize=11,
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "fig_rsf_feature_importances.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")


# =================================================================
# Main
# =================================================================

def main() -> None:
    """
    Run the full feature importance workflow.

    Steps
    -----
    1. Load the aligned expression and survival dataset.
    2. Fit Cox LASSO on the full cohort with CV-selected alpha; log and plot
       standardised coefficients.
    3. Fit RSF on the full cohort; compute and plot permutation importances.
    4. Assemble and save a combined gene-level summary table.
    """
    warnings.filterwarnings("ignore", category=UserWarning)

    # -----------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------
    X, y = load_data()

    # -----------------------------------------------------------------
    # Cox LASSO — full-data fit with CV alpha selection
    # -----------------------------------------------------------------
    logger.info('-' * 50)
    logger.info("Cox LASSO (CV alpha, full-data fit)")
    logger.info('-' * 50)
    coefs, best_alpha = fit_cox_lasso_full(X, y)
    logger.info(f"Cox LASSO coefficients (alpha={best_alpha}):\n{coefs.to_string()}")
    plot_cox_coefficients(coefs, best_alpha)

    # -----------------------------------------------------------------
    # RSF — full-data fit with permutation importance
    # -----------------------------------------------------------------
    logger.info("--- RSF permutation importances (n=20 repeats, full-data fit) ---")
    importances, stds = fit_rsf_full(X, y)
    logger.info(f"RSF importances:\n{importances.to_string()}")
    plot_rsf_importances(importances, stds)

    # -----------------------------------------------------------------
    # Save combined gene-level summary table
    # -----------------------------------------------------------------
    feat_df = pd.DataFrame({
        "gene":                 GENE_COLS,
        "cox_lasso_coef":       coefs.reindex(GENE_COLS).values,
        "rsf_importance_mean":  importances.reindex(GENE_COLS).values,
        "rsf_importance_std":   stds,
    })

    out_path = ROOT / "data" / "processed" / "feature_importance_table.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()