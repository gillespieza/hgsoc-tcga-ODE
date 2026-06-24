"""
ml_benchmark.py — ML survival model benchmarking against HR-DDR ODE.

Benchmarks two data-trained ML survival models against the zero-shot
HR-DDR ODE (AUC_X) on the HGSOC TCGA cohort.

Models
------
- Cox LASSO (CoxnetSurvivalAnalysis, L1 regularisation)
- Random Survival Forest (200 trees)

Feature set
-----------
14 log2(FPKM+1) gene expression columns (same genes used to parameterise
the ODE, plus TP53 as a negative control).

Evaluation
----------
- 5-fold stratified CV for ML models.
  Cox LASSO alpha is selected by inner 3-fold CV on the training fold
  only — the outer test fold never influences hyperparameter choice.
- 1000-rep bootstrap CIs on out-of-fold (OOF) predictions for ML models
  and on full-cohort scores for the ODE (zero-shot predictor; no training
  step means bootstrapping the full cohort is the correct procedure).
- Forest plot and bootstrap distribution figures saved to results/figures/.

NOTE on ODE vs ML bootstrap asymmetry:
    The ODE bootstrap uses the full cohort because no model was fitted
    (log_AUC_X scores are a pre-specified, parameter-free function of
    expression data). ML bootstrap uses OOF predictions so that the
    model never saw the test patients during training. Both procedures
    are internally valid, but the asymmetry should be noted when
    comparing confidence intervals across models.

Outputs
-------
- data/processed/ml_comparison_table.csv
- results/figures/fig_ml_forest_plot.png
- results/figures/fig_ml_bootstrap_distributions.png
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
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from tqdm import tqdm

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

# Candidate regularisation strengths evaluated during inner-CV alpha
# selection for Cox LASSO.
LASSO_ALPHAS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]


# =================================================================
# Data loading
# =================================================================

def load_ode_baseline() -> float:
    """
    Load the ODE AUC_X concordance index from the univariate Cox table.

    The C-index stored here comes from a univariate Cox model fitted to
    log(AUC_X + 1) on the full cohort in analyse_ode_survival.py. For a
    single-covariate Cox model the concordance of the linear predictor
    equals the concordance of the raw covariate, so this is equivalent
    to computing the C-index of log_AUC_X scores directly.

    Returns
    -------
    float
        Concordance index for log_AUC_X from the univariate Cox model.

    Raises
    ------
    ValueError
        If the AUC_X row is absent from the comparison table, indicating
        that analyse_ode_survival.py has not yet been run.
    """
    cox = pd.read_csv(
        ROOT / "data" / "processed" / "univariate_cox_comparison.csv"
    )
    aucx_rows = cox.loc[cox["score"] == "AUC_X", "C_index"]

    if aucx_rows.empty:
        raise ValueError(
            "AUC_X row not found in univariate_cox_comparison.csv. "
            "Run analyse_ode_survival.py before ml_benchmark.py."
        )

    return float(aucx_rows.values[0])


def load_data() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Load and align the merged expression dataset with the survival table.

    Inner-joins the survival analysis table (which contains only patients
    with successful ODE integrations) to the merged expression matrix so
    that the ML and ODE benchmarks operate on an identical cohort.

    Returns
    -------
    X : pd.DataFrame, shape (n_patients, n_genes)
        Log2(FPKM+1) expression values for GENE_COLS, indexed by PATIENT_ID.
    y : np.ndarray
        Structured array with fields ('event', bool) and ('time', float)
        for scikit-survival compatibility.
    df : pd.DataFrame
        Merged table of PATIENT_ID, OS_MONTHS, OS_EVENT, and GENE_COLS.
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
        f"Loaded aligned cohort: {len(df)} patients, "
        f"{len(GENE_COLS)} gene features"
    )

    X = pd.DataFrame(
        df[GENE_COLS].values,
        columns=GENE_COLS,
        index=df["PATIENT_ID"],
    )
    y = _build_survival_array(df)
    return X, y, df


def _build_survival_array(df: pd.DataFrame) -> np.ndarray:
    """
    Convert OS_EVENT / OS_MONTHS columns to a scikit-survival structured array.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'OS_EVENT' (int/bool) and 'OS_MONTHS' (float) columns.

    Returns
    -------
    np.ndarray
        Structured array with fields ('event', bool) and ('time', float).
    """
    return np.array(
        [
            (bool(e), float(t))
            for e, t in zip(df["OS_EVENT"], df["OS_MONTHS"])
        ],
        dtype=[("event", bool), ("time", float)],
    )


# =================================================================
# Bootstrap
# =================================================================

def bootstrap_cindex(
    risk_scores: np.ndarray,
    y: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    desc: str = "Bootstrap",
) -> dict:
    """
    Compute a bootstrap confidence interval for the concordance index.

    Parameters
    ----------
    risk_scores : np.ndarray
        Predicted risk scores; higher values must mean worse prognosis
        (i.e. the caller is responsible for sign convention).
    y : np.ndarray
        Structured survival array with ('event', bool) and ('time', float).
    n_boot : int
        Number of bootstrap resamples.
    seed : int
        Random seed for reproducibility.
    desc : str
        Label shown in the tqdm progress bar.

    Returns
    -------
    dict
        Keys: 'mean', 'ci_low', 'ci_high', 'samples'.
    """
    rng = np.random.default_rng(seed)
    n = len(risk_scores)
    boot_cindices: list[float] = []

    for _ in tqdm(range(n_boot), desc=desc, leave=False):
        idx = rng.choice(n, size=n, replace=True)
        y_boot = y[idx]
        r_boot = risk_scores[idx]

        # Concordance is undefined if the resample contains no events.
        if y_boot["event"].sum() == 0:
            continue

        try:
            ci = concordance_index_censored(
                y_boot["event"], y_boot["time"], r_boot
            )[0]
            boot_cindices.append(ci)
        except Exception:
            # NOTE: style guide mandates logger.exception, but a full
            # traceback for each of 1000 reps would flood pipeline.log.
            # logger.warning is used here as a documented deviation.
            logger.warning("Bootstrap resample failed; skipping iteration.")

    samples = np.array(boot_cindices)

    return {
        "mean":    float(np.mean(samples)),
        "ci_low":  float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
        "samples": samples,
    }


# =================================================================
# Cox LASSO
# =================================================================

def _select_alpha_inner_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_inner: int = 3,
    seed: int = 42,
) -> float:
    """
    Select the best Cox LASSO alpha by inner CV on the outer training fold.

    Selecting alpha by evaluating the outer test fold is data leakage:
    the reported C-index becomes optimistic because the test data influenced
    the model. This function selects alpha entirely within the outer training
    fold, keeping the outer test fold unseen during hyperparameter search.

    Parameters
    ----------
    X_train : np.ndarray
        Training features for the current outer fold.
    y_train : np.ndarray
        Training survival array for the current outer fold.
    n_inner : int
        Number of inner CV folds. Three is used rather than five to avoid
        making each inner training split uncomfortably small.
    seed : int
        Random seed for inner splits.

    Returns
    -------
    float
        Alpha with the highest mean C-index across inner folds.
        Falls back to the smallest alpha if every inner fit fails.
    """
    inner_cv = StratifiedKFold(
        n_splits=n_inner, shuffle=True, random_state=seed
    )
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

        for inner_train, inner_val in inner_cv.split(
            X_train, y_train["event"]
        ):
            try:
                pipe.fit(X_train[inner_train], y_train[inner_train])

                if np.all(pipe.named_steps["cox"].coef_ == 0):
                    # All coefficients zeroed — regularisation too strong;
                    # model is uninformative at this alpha.
                    continue

                risk = pipe.predict(X_train[inner_val])
                ci = concordance_index_censored(
                    y_train[inner_val]["event"],
                    y_train[inner_val]["time"],
                    risk,
                )[0]
                alpha_scores[alpha].append(ci)

            except Exception:
                logger.warning(
                    f"Inner-CV fit failed for alpha={alpha}; skipping."
                )

    best_alpha = LASSO_ALPHAS[0]
    best_mean = -1.0

    for alpha, scores in alpha_scores.items():
        if scores and np.mean(scores) > best_mean:
            best_mean = np.mean(scores)
            best_alpha = alpha

    return best_alpha


def fit_cox_lasso(
    X: pd.DataFrame,
    y: np.ndarray,
    cv: StratifiedKFold,
) -> tuple[list[float], list[float]]:
    """
    Evaluate Cox LASSO via 5-fold CV with inner-CV alpha selection.

    Alpha is selected by 3-fold inner CV on each outer training fold so
    the outer test fold is never used during hyperparameter search.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (n_patients × n_genes), indexed by PATIENT_ID.
    y : np.ndarray
        Structured survival array.
    cv : StratifiedKFold
        Outer cross-validation splitter.

    Returns
    -------
    fold_cindices : list[float]
        C-index on each outer test fold.
    best_alphas : list[float]
        Alpha chosen by inner CV for each outer fold.
    """
    X_array = X.values
    fold_cindices: list[float] = []
    best_alphas: list[float] = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X_array, y["event"])
    ):
        X_train, X_test = X_array[train_idx], X_array[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # -----------------------------------------------------------------
        # Select alpha on the training fold only to prevent leakage.
        # -----------------------------------------------------------------
        best_alpha = _select_alpha_inner_cv(X_train, y_train)

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

        try:
            pipe.fit(X_train, y_train)
            risk = pipe.predict(X_test)
            ci = concordance_index_censored(
                y_test["event"], y_test["time"], risk
            )[0]
        except Exception:
            logger.exception(
                f"Cox LASSO outer fold {fold + 1} failed; recording NaN."
            )
            ci = float("nan")

        fold_cindices.append(ci)
        best_alphas.append(best_alpha)

        logger.info(
            f"Cox LASSO fold {fold + 1}: "
            f"C-index={ci:.4f}, alpha={best_alpha}"
        )

    valid = [c for c in fold_cindices if not np.isnan(c)]
    logger.info(
        f"Cox LASSO mean C-index: "
        f"{np.mean(valid):.4f} ± {np.std(valid):.4f}"
    )
    logger.info(f"Best alphas per fold: {best_alphas}")
    return fold_cindices, best_alphas


# =================================================================
# Random Survival Forest
# =================================================================

def fit_rsf(
    X: pd.DataFrame,
    y: np.ndarray,
    cv: StratifiedKFold,
) -> list[float]:
    """
    Evaluate RSF via 5-fold CV.

    scikit-survival's RSF.predict() returns mean cumulative hazard values,
    where higher values correspond to worse prognosis (higher risk). No
    per-fold sign flip is applied; a single global direction check is
    performed on the assembled OOF array inside collect_oof_predictions.
    Applying per-fold flips independently would make some fold C-indices
    inconsistent with the OOF scores used in the bootstrap.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Structured survival array.
    cv : StratifiedKFold
        Outer cross-validation splitter.

    Returns
    -------
    list[float]
        Per-fold C-index values.
    """
    X_array = X.values
    fold_cindices: list[float] = []

    rsf = RandomSurvivalForest(
        n_estimators=200,
        min_samples_leaf=25,
        max_features=0.5,
        n_jobs=-1,
        random_state=42,
    )

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X_array, y["event"])
    ):
        X_train, X_test = X_array[train_idx], X_array[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        rsf.fit(X_train, y_train)
        risk = rsf.predict(X_test)

        ci = concordance_index_censored(
            y_test["event"], y_test["time"], risk
        )[0]

        logger.info(f"RSF fold {fold + 1}: C-index={ci:.4f}")
        fold_cindices.append(ci)

    valid = [c for c in fold_cindices if not np.isnan(c)]
    logger.info(
        f"RSF mean C-index: "
        f"{np.mean(valid):.4f} ± {np.std(valid):.4f}"
    )
    return fold_cindices


# =================================================================
# Out-of-fold prediction assembly
# =================================================================

def collect_oof_predictions(
    X: pd.DataFrame,
    y: np.ndarray,
    cv: StratifiedKFold,
    lasso_alphas: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assemble out-of-fold risk predictions for Cox LASSO and RSF.

    Each patient's prediction comes from a model trained on every other
    fold, so the model never saw that patient during training. These OOF
    predictions are used for bootstrap CI estimation instead of fold-level
    C-indices, giving a single coherent risk score per patient.

    A global RSF direction check is applied after assembly: if the OOF
    C-index is below 0.5, all RSF scores are negated. This is safer than
    per-fold flipping, which can produce OOF arrays with inconsistent sign
    conventions across patients.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (n_patients × n_genes).
    y : np.ndarray
        Structured survival array.
    cv : StratifiedKFold
        Must use the same split parameters as fit_cox_lasso / fit_rsf.
    lasso_alphas : list[float]
        Per-fold alphas selected by inner CV during fit_cox_lasso.

    Returns
    -------
    lasso_oof : np.ndarray
        OOF risk scores for Cox LASSO (higher = worse prognosis).
    rsf_oof : np.ndarray
        OOF risk scores for RSF after direction check.
    """
    X_array = X.values
    n = len(y)
    lasso_oof = np.zeros(n)
    rsf_oof = np.zeros(n)

    logger.info("Collecting out-of-fold predictions ...")

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X_array, y["event"])
    ):
        X_train, X_test = X_array[train_idx], X_array[test_idx]
        y_train = y[train_idx]

        # Cox LASSO
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("cox", CoxnetSurvivalAnalysis(
                l1_ratio=1.0,
                alphas=[lasso_alphas[fold]],
                fit_baseline_model=True,
                normalize=False,
                max_iter=10000,
            )),
        ])
        pipe.fit(X_train, y_train)
        lasso_oof[test_idx] = pipe.predict(X_test)

        # RSF
        rsf_fold = RandomSurvivalForest(
            n_estimators=200,
            min_samples_leaf=25,
            max_features=0.5,
            n_jobs=-1,
            random_state=42,
        )
        rsf_fold.fit(X_train, y_train)
        rsf_oof[test_idx] = rsf_fold.predict(X_test)

        logger.info(f"OOF fold {fold + 1} collected")

    # -----------------------------------------------------------------
    # Global RSF direction check.
    #
    # scikit-survival RSF.predict() returns cumulative hazard (higher =
    # worse). If the assembled OOF C-index is below 0.5 the predictions
    # are globally inverted. A single global flip preserves consistency
    # across all patients.
    # -----------------------------------------------------------------
    rsf_ci = concordance_index_censored(
        y["event"], y["time"], rsf_oof
    )[0]

    if rsf_ci < 0.5:
        rsf_oof = -rsf_oof
        logger.warning(
            f"RSF OOF C-index was {rsf_ci:.4f} (< 0.5); "
            "global sign flip applied to OOF scores."
        )

    return lasso_oof, rsf_oof


# =================================================================
# Results table
# =================================================================

def save_comparison_table(
    ode_baseline: float,
    lasso_cindices: list[float],
    rsf_cindices: list[float],
    ode_boot: dict,
    lasso_boot: dict,
    rsf_boot: dict,
) -> pd.DataFrame:
    """
    Assemble, log, and save the three-model performance comparison table.

    Parameters
    ----------
    ode_baseline : float
        Full-cohort concordance index for the ODE predictor.
    lasso_cindices : list[float]
        Per-fold C-indices from fit_cox_lasso (may contain NaN).
    rsf_cindices : list[float]
        Per-fold C-indices from fit_rsf.
    ode_boot, lasso_boot, rsf_boot : dict
        Bootstrap result dicts with keys 'mean', 'ci_low', 'ci_high'.

    Returns
    -------
    pd.DataFrame
        One row per model with CV C-index, SD, and bootstrap 95% CI.
    """
    valid_lasso = [c for c in lasso_cindices if not np.isnan(c)]
    valid_rsf   = [c for c in rsf_cindices if not np.isnan(c)]

    rows = [
        {
            "Model":      "HR-DDR ODE (AUC_X)",
            "Feature_set": "14 genes → 4 params → ODE",
            "CV_cindex":  f"{ode_baseline:.3f}",
            "CV_sd":      "",
            "CV_display": f"{ode_baseline:.3f}",
            "CI_low":     f"{ode_boot['ci_low']:.3f}",
            "CI_high":    f"{ode_boot['ci_high']:.3f}",
            "Notes":      "No CV needed (zero-shot predictor)",
        },
        {
            "Model":      "Cox LASSO",
            "Feature_set": "14 genes (raw)",
            "CV_cindex":  f"{np.mean(valid_lasso):.3f}",
            "CV_sd":      f"{np.std(valid_lasso):.3f}",
            "CV_display": (
                f"{np.mean(valid_lasso):.3f} "
                f"± {np.std(valid_lasso):.3f}"
            ),
            "CI_low":     f"{lasso_boot['ci_low']:.3f}",
            "CI_high":    f"{lasso_boot['ci_high']:.3f}",
            "Notes":      "5-fold CV, alpha via inner 3-fold CV",
        },
        {
            "Model":      "Random Survival Forest",
            "Feature_set": "14 genes (raw)",
            "CV_cindex":  f"{np.mean(valid_rsf):.3f}",
            "CV_sd":      f"{np.std(valid_rsf):.3f}",
            "CV_display": (
                f"{np.mean(valid_rsf):.3f} "
                f"± {np.std(valid_rsf):.3f}"
            ),
            "CI_low":     f"{rsf_boot['ci_low']:.3f}",
            "CI_high":    f"{rsf_boot['ci_high']:.3f}",
            "Notes":      "5-fold CV, global OOF direction check",
        },
    ]

    df = pd.DataFrame(rows)

    logger.info('-' * 50)
    logger.info("Model comparison table:")
    logger.info(
        f"{'MODEL':<30} {'FEATURE SET':<26} "
        f"{'C-INDEX (CV)':<18} {'95% BOOTSTRAP CI':<20} NOTES"
    )

    for _, r in df.iterrows():
        ci_str = f"[{r['CI_low']}, {r['CI_high']}]"
        logger.info(
            f"{r['Model']:<30} {r['Feature_set']:<26} "
            f"{r['CV_display']:<18} {ci_str:<20} {r['Notes']}"
        )

    out_path = ROOT / "data" / "processed" / "ml_comparison_table.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.success(f"[FILE] Saved: ./data/processed/ml_comparison_table.csv")

    return df


# =================================================================
# Figures
# =================================================================

def plot_forest(
    ode_samples: np.ndarray,
    lasso_samples: np.ndarray,
    rsf_samples: np.ndarray,
) -> None:
    """
    Forest plot of bootstrap C-index means and 95% CIs for all three models.

    Each model is a dot (bootstrap mean) with a horizontal bar spanning
    the 95% percentile bootstrap CI.

    Parameters
    ----------
    ode_samples, lasso_samples, rsf_samples : np.ndarray
        Raw 1000-rep bootstrap C-index samples from bootstrap_cindex.
    """
    models  = ["HR-DDR ODE\n(AUC_X)", "Cox LASSO", "Random\nSurvival Forest"]
    samples = [ode_samples, lasso_samples, rsf_samples]
    colors  = ["#2196F3", "#FF9800", "#4CAF50"]

    means = [np.mean(s) for s in samples]
    lows  = [np.percentile(s, 2.5) for s in samples]
    highs = [np.percentile(s, 97.5) for s in samples]

    fig, ax = plt.subplots(figsize=(7, 4))

    for i, (m, lo, hi, c) in enumerate(zip(means, lows, highs, colors)):
        ax.plot(
            [lo, hi], [i, i],
            color=c, linewidth=2.5, solid_capstyle="round",
        )
        ax.scatter(m, i, color=c, s=80, zorder=5)

    ax.axvline(
        0.5, color="black", linestyle="--",
        linewidth=1, alpha=0.6, label="Chance (0.5)",
    )
    ax.set_yticks(list(range(len(models))))
    ax.set_yticklabels(models, fontsize=11)
    ax.set_xlabel("C-index (bootstrap mean, 95% CI)", fontsize=11)
    ax.set_title(
        "Model Comparison — Prognostic Discrimination\n"
        "HGSOC TCGA (n=420)",
        fontsize=12,
    )
    ax.set_xlim(0.38, 0.65)
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "fig_ml_forest_plot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.success(f"[FILE] Saved: ./results/figures/fig_ml_forest_plot.png")


def plot_bootstrap_distributions(
    ode_samples: np.ndarray,
    lasso_samples: np.ndarray,
    rsf_samples: np.ndarray,
) -> None:
    """
    Histogram panel of bootstrap C-index distributions for all three models.

    Parameters
    ----------
    ode_samples, lasso_samples, rsf_samples : np.ndarray
        Raw 1000-rep bootstrap C-index samples from bootstrap_cindex.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)

    data   = [ode_samples, lasso_samples, rsf_samples]
    labels = ["HR-DDR ODE (AUC_X)", "Cox LASSO", "Random Survival Forest"]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for ax, d, label, color in zip(axes, data, labels, colors):
        ax.hist(d, bins=40, color=color, alpha=0.75, edgecolor="white")
        mean_val = np.mean(d)
        ax.axvline(
            mean_val, color="black", linewidth=1.5,
            label=f"Mean={mean_val:.3f}",
        )
        ax.axvline(
            np.percentile(d, 2.5),
            color="black", linewidth=1, linestyle="--",
        )
        ax.axvline(
            np.percentile(d, 97.5),
            color="black", linewidth=1, linestyle="--",
            label="95% CI",
        )
        ax.axvline(
            0.5, color="red", linewidth=1, linestyle=":", alpha=0.7,
            label="Chance",
        )
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("C-index", fontsize=9)
        ax.set_ylabel("Bootstrap count", fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle(
        "Bootstrap C-index Distributions (n=1000 resamples)",
        fontsize=12,
    )
    plt.tight_layout()

    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "fig_ml_bootstrap_distributions.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.success(f"[FILE] Saved: ./results/figures/fig_ml_bootstrap_distributions.png")


# =================================================================
# Main
# =================================================================

def main() -> None:
    """
    Run the full ML benchmarking workflow.

    Steps
    -----
    1. Load ODE baseline C-index and aligned expression / survival dataset.
    2. 5-fold CV for Cox LASSO (alpha via inner 3-fold CV) and RSF.
    3. Collect OOF predictions; apply global RSF direction check.
    4. 1000-rep bootstrap CIs for all three models.
    5. Save comparison table and two figures.
    """
    # Move warning suppression inside main() so it does not fire on import
    # and does not suppress warnings in other modules that import this one.
    warnings.filterwarnings("ignore", category=UserWarning)

    # -----------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------
    ode_baseline_cindex = load_ode_baseline()
    X, y, _ = load_data()

    logger.info(
        f"ODE baseline C-index (AUC_X): {ode_baseline_cindex:.4f}"
    )
    logger.info(
        f"Feature matrix: {X.shape}, events: {int(y['event'].sum())}"
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # -----------------------------------------------------------------
    # Report per-fold event rates as a data quality check.
    # Severe imbalance across folds would indicate stratification issues.
    # -----------------------------------------------------------------
    for fold, (_, test_idx) in enumerate(cv.split(X.values, y["event"])):
        y_test = y[test_idx]
        rate = y_test["event"].sum() / len(y_test)
        logger.info(
            f"Fold {fold + 1}: event rate={rate:.3f} "
            f"({int(y_test['event'].sum())} events / {len(y_test)} patients)"
        )

    # -----------------------------------------------------------------
    # Cross-validated model fitting
    # -----------------------------------------------------------------
    logger.info('-' * 50)
    logger.info("Cox LASSO (5-fold outer CV, inner 3-fold alpha)")
    lasso_cindices, lasso_alphas = fit_cox_lasso(X, y, cv)

    logger.info('-' * 50)
    logger.info("Random Survival Forest (5-fold CV)")
    rsf_cindices = fit_rsf(X, y, cv)

    # -----------------------------------------------------------------
    # Assemble OOF predictions for bootstrap CI estimation
    # -----------------------------------------------------------------
    lasso_oof, rsf_oof = collect_oof_predictions(X, y, cv, lasso_alphas)

    # -----------------------------------------------------------------
    # ODE risk scores from precomputed survival table.
    #
    # The ODE is a zero-shot predictor: no training was performed, so
    # all patients can be used for bootstrap without held-out splits.
    # Align to X.index to guarantee identical patient ordering.
    # -----------------------------------------------------------------
    surv_full = pd.read_csv(
        ROOT / "data" / "processed" / "survival_analysis_df.csv"
    )
    surv_full = (
        surv_full.set_index("PATIENT_ID")
        .loc[X.index]
        .reset_index()
    )
    ode_risk = surv_full["log_AUC_X"].values

    # -----------------------------------------------------------------
    # Bootstrap confidence intervals
    # -----------------------------------------------------------------
    logger.info('-' * 50)
    logger.info("Bootstrap CIs (1000 reps)")
    ode_boot   = bootstrap_cindex(ode_risk,  y, desc="ODE bootstrap")
    lasso_boot = bootstrap_cindex(lasso_oof, y, desc="LASSO bootstrap")
    rsf_boot   = bootstrap_cindex(rsf_oof,   y, desc="RSF bootstrap")

    logger.info(
        f"ODE   (log_AUC_X): {ode_boot['mean']:.4f} "
        f"[{ode_boot['ci_low']:.4f}, {ode_boot['ci_high']:.4f}]"
    )
    logger.info(
        f"LASSO (OOF):       {lasso_boot['mean']:.4f} "
        f"[{lasso_boot['ci_low']:.4f}, {lasso_boot['ci_high']:.4f}]"
    )
    logger.info(
        f"RSF   (OOF):       {rsf_boot['mean']:.4f} "
        f"[{rsf_boot['ci_low']:.4f}, {rsf_boot['ci_high']:.4f}]"
    )

    # -----------------------------------------------------------------
    # Save comparison table and figures
    # -----------------------------------------------------------------
    save_comparison_table(
        ode_baseline_cindex,
        lasso_cindices,
        rsf_cindices,
        ode_boot,
        lasso_boot,
        rsf_boot,
    )

    plot_forest(
        ode_boot["samples"],
        lasso_boot["samples"],
        rsf_boot["samples"],
    )
    plot_bootstrap_distributions(
        ode_boot["samples"],
        lasso_boot["samples"],
        rsf_boot["samples"],
    )


if __name__ == "__main__":
    main()