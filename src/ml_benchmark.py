"""
ml_benchmark.py — ML survival model benchmarking against HR-DDR ODE.

Benchmarks four data-trained ML survival models against the zero-shot
HR-DDR ODE (AUC_X) on the HGSOC TCGA cohort.

Models
------
14-gene feature set (ODE genes + TP53 negative control):
  - Cox LASSO (CoxnetSurvivalAnalysis, L1 regularisation)
  - Random Survival Forest (200 trees, max_features=0.5)

All-genes feature set (every gene in the raw FPKM file, no prior structure):
  - Cox LASSO (same architecture; alpha re-selected by inner CV)
  - Random Survival Forest (200 trees, max_features="sqrt")
    max_features="sqrt" replaces 0.5 for the all-genes set because with
    thousands of features, 0.5 would consider far too many candidates per
    split, increasing tree correlation and computation cost.

The all-genes arm is an unbiased, purely data-driven baseline that tests
whether curated pathway knowledge in the 14-gene set adds predictive value
beyond a genome-wide search with no prior biological structure.

Feature set
-----------
14-gene: log2(FPKM+1) for BRCA1/2, RAD51, PALB2, BRIP1, ATM, ATR,
  CHEK1/2, BCL2, BCL2L1, BAX, BAD, TP53.
All-genes: log2(FPKM+1) for every gene in data_mrna_seq_fpkm.txt with
  ≤20% missing values across patients, loaded at runtime from the raw
  expression file. Entrez IDs are used as column names (no symbol mapping).

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
from sklearn.feature_selection import VarianceThreshold
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

# Candidate regularisation strengths for the 14-gene arm.
# The smallest viable alpha here is 0.01: below that, inner folds have
# only ~220 patients and 14 features, so the penalty is light enough to
# be useful but not so permissive that coefficients explode.
LASSO_ALPHAS_14 = [0.01, 0.05, 0.1, 0.5, 1.0]

# Candidate regularisation strengths for the all-genes arm (~40k features,
# ~330 training patients per outer fold — severe p >> n).
#
# Why no small alphas here?
# With p >> n, even alpha=0.1 triggers an ArithmeticError ("weights too
# large") inside glmnet's Cython layer on inner folds of ~220 patients.
# glmnet internally extends the path below the smallest requested alpha via
# alpha_min_ratio; _select_alpha_inner_cv sets alpha_min_ratio=1.0 to pin
# the path to exactly these values with no downward extrapolation.
# The floor of 1.0 was validated empirically on this cohort.
LASSO_ALPHAS_ALL = [1.0, 2.0, 5.0, 10.0, 20.0]


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
    X : pd.DataFrame, shape (n_patients, 14)
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


def load_all_genes_data(patient_ids: pd.Index) -> pd.DataFrame:
    """
    Load all expression genes from the raw FPKM file for the benchmark cohort.

    Loads the raw FPKM file at runtime rather than from hgsoc_tcga_merged.csv,
    which contains only the 14 ODE genes. Applies the same preprocessing as
    prepare_RNA.py: log2(FPKM+1) and ≤20% missing gene filter. Missing values
    that survive the gene filter are imputed with per-gene medians before
    model fitting.

    No gene-symbol mapping is applied; Entrez IDs are used as column names.
    Only patients present in patient_ids are returned, in the same order, so
    the resulting matrix is row-aligned with the 14-gene benchmark cohort.

    Parameters
    ----------
    patient_ids : pd.Index
        PATIENT_IDs from the 14-gene cohort (X.index from load_data).
        Used to restrict and order the all-genes matrix identically.

    Returns
    -------
    pd.DataFrame
        shape (n_patients, n_all_genes), indexed by PATIENT_ID.

    Raises
    ------
    ValueError
        If no patients overlap between the raw expression file and
        patient_ids, indicating a PATIENT_ID formatting mismatch.
    """
    # -----------------------------------------------------------------
    # Load raw FPKM matrix
    # -----------------------------------------------------------------
    # Same loading as prepare_RNA.py: genes as rows, samples as columns.
    # index_col=0 makes the first column (Entrez Gene IDs) the row index.
    expr = pd.read_csv(
        ROOT / "data" / "raw" / "expression" / "data_mrna_seq_fpkm.txt",
        sep="\t",
        index_col=0,
    )

    # Drop any rows whose index is non-numeric — these are annotation
    # metadata rows (e.g. a Hugo_Symbol column that survived index_col=0).
    numeric_mask = pd.to_numeric(expr.index, errors="coerce").notna()
    n_annot = (~numeric_mask).sum()
    if n_annot > 0:
        logger.warning(
            f"Dropped {n_annot} non-numeric rows from raw expression "
            "matrix (likely annotation metadata)."
        )
    expr = expr[numeric_mask]

    # -----------------------------------------------------------------
    # Transpose and align sample IDs
    # -----------------------------------------------------------------
    expr = expr.T.copy()

    # TCGA barcodes are 28 characters; truncate to 12-char patient ID.
    # Consistent with the truncation in prepare_RNA.py.
    expr.index = expr.index.str[:12]
    expr.index.name = "PATIENT_ID"

    # Convert to float; any surviving non-numeric annotation columns
    # (e.g. a Hugo_Symbol column) become NaN and are removed below.
    expr = expr.apply(pd.to_numeric, errors="coerce")

    # -----------------------------------------------------------------
    # Deduplication: if a patient has multiple samples, keep the one
    # with the fewest missing values (consistent with prepare_RNA.py).
    # -----------------------------------------------------------------
    expr["_n_missing"] = expr.isna().sum(axis=1)
    expr = (
        expr.sort_values("_n_missing")
        .loc[~expr.index.duplicated(keep="first")]
        .drop(columns="_n_missing")
    )

    # -----------------------------------------------------------------
    # Log2(FPKM + 1) transform
    # -----------------------------------------------------------------
    # Avoids log(0); matches the transform applied in rna_clean.csv.
    expr = np.log2(expr.clip(lower=0) + 1)

    # -----------------------------------------------------------------
    # Filter genes with > 20% missing values across samples
    # -----------------------------------------------------------------
    missing_gene_frac = expr.isna().mean(axis=0)
    n_before = expr.shape[1]
    expr = expr.loc[:, missing_gene_frac <= 0.20].copy()
    n_after = expr.shape[1]

    logger.info(
        f"All-genes: retained {n_after} / {n_before} genes "
        f"(dropped {n_before - n_after} with >20% missing)"
    )

    # -----------------------------------------------------------------
    # Restrict to benchmark cohort and enforce patient ordering
    # -----------------------------------------------------------------
    overlap = expr.index.intersection(patient_ids)

    if len(overlap) == 0:
        raise ValueError(
            "No patient overlap between raw expression file and the "
            "benchmark cohort. Check PATIENT_ID formatting."
        )

    # Reindex aligns rows to patient_ids order exactly; patients absent
    # from the raw file would appear as all-NaN rows (unexpected).
    expr = expr.reindex(patient_ids)

    n_missing_patients = expr.isna().all(axis=1).sum()
    if n_missing_patients > 0:
        logger.warning(
            f"{n_missing_patients} benchmark patients have no all-genes "
            "expression data after reindex; rows contain NaN."
        )

    # -----------------------------------------------------------------
    # Median imputation for residual per-gene NaN values
    # -----------------------------------------------------------------
    # After the >20% missing filter, some genes may still have a small
    # number of NaN values. Impute with per-gene median to avoid
    # propagating NaN into model fitting.
    n_nan_cells = int(expr.isna().sum().sum())
    if n_nan_cells > 0:
        logger.info(
            f"All-genes: imputing {n_nan_cells} residual NaN cells "
            "with per-gene median."
        )
        expr = expr.fillna(expr.median())

    logger.info(
        f"All-genes feature matrix: "
        f"{expr.shape[0]} patients, {expr.shape[1]} genes"
    )

    # -----------------------------------------------------------------
    # Variance pre-filter (computational safeguard, not feature selection)
    # -----------------------------------------------------------------
    # With ~40k genes and ~400 patients this is a severe p >> n problem.
    # Genes with near-zero variance across all patients carry no information
    # and dramatically slow down coordinate descent. Removing them here is a
    # pure computational safeguard: no outcome data are involved, so there is
    # no leakage risk.
    #
    # NOTE: threshold=0.01 operates on the log2(FPKM+1) scale. A gene must
    # vary by at least ~0.1 log2-units across the cohort to be retained.
    # This is intentionally conservative — only truly flat genes are dropped.
    vt = VarianceThreshold(threshold=0.01)
    expr_filtered = pd.DataFrame(
        vt.fit_transform(expr),
        index=expr.index,
        columns=expr.columns[vt.get_support()],
    )

    n_dropped = expr.shape[1] - expr_filtered.shape[1]
    logger.info(
        f"All-genes variance filter (threshold=0.01): "
        f"retained {expr_filtered.shape[1]} / {expr.shape[1]} genes "
        f"(dropped {n_dropped} near-zero-variance genes)"
    )

    return expr_filtered


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
    alphas: list[float],
    n_inner: int = 3,
    seed: int = 42,
    progress_desc: str = "Alpha search",
) -> float:
    """
    Select the best Cox LASSO alpha by inner CV on the outer training fold.

    Selecting alpha by evaluating the outer test fold is data leakage:
    the reported C-index becomes optimistic because the test data influenced
    the model. This function selects alpha entirely within the outer training
    fold, keeping the outer test fold unseen during hyperparameter search.

    Strategy: fit the full regularisation path in a single CoxnetSurvivalAnalysis
    call per inner fold, then read off per-alpha coefficients and predictions.
    This is the intended use of the estimator — it uses warm starts internally
    (high alpha → low alpha), which keeps coefficients from growing uncontrollably
    and avoids the exp() overflow that occurs when fitting each alpha independently
    from scratch in the p >> n regime.

    Parameters
    ----------
    X_train : np.ndarray
        Training features for the current outer fold.
    y_train : np.ndarray
        Training survival array for the current outer fold.
    alphas : list[float]
        Candidate regularisation strengths to evaluate. Pass LASSO_ALPHAS_14
        for the 14-gene arm or LASSO_ALPHAS_ALL for the all-genes arm.
    n_inner : int
        Number of inner CV folds. Three is used rather than five to avoid
        making each inner training split uncomfortably small.
    seed : int
        Random seed for inner splits.
    progress_desc : str
        Label shown in the tqdm progress bar.

    Returns
    -------
    float
        Alpha with the highest mean C-index across inner folds.
        Falls back to the largest alpha if every inner fit fails, which
        errs on the side of stronger regularisation rather than overflow.
    """
    inner_cv = StratifiedKFold(
        n_splits=n_inner,
        shuffle=True,
        random_state=seed,
    )

    # Accumulate per-alpha C-index scores across all inner folds.
    alpha_scores: dict[float, list[float]] = {a: [] for a in alphas}

    # Pass the full alpha sequence to a single estimator so glmnet's
    # coordinate-descent warm starts across the path — numerically stable
    # and faster than one fit per alpha.
    #
    # alphas must be sorted descending so the path runs from most to least
    # regularised (standard glmnet convention).
    alphas_desc = sorted(alphas, reverse=True)

    for inner_train, inner_val in tqdm(
        inner_cv.split(X_train, y_train["event"]),
        total=n_inner,
        desc=progress_desc,
        leave=False,
    ):
        try:
            scaler = StandardScaler()
            X_scaled_train = scaler.fit_transform(X_train[inner_train])
            X_scaled_val   = scaler.transform(X_train[inner_val])

            cox_path = CoxnetSurvivalAnalysis(
                l1_ratio=1.0,
                alphas=alphas_desc,
                # alpha_min_ratio=1.0 pins the path exactly to the supplied
                # alphas with no downward extrapolation. Without this, glmnet
                # extends the path below the smallest alpha using an internal
                # ratio, which reaches numerically unsafe values in the p >> n
                # regime and raises ArithmeticError in the Cython solver.
                alpha_min_ratio=1.0,
                fit_baseline_model=True,
                normalize=False,
                max_iter=10000,
            )
            cox_path.fit(X_scaled_train, y_train[inner_train])

            # coef_ shape: (n_features, n_alphas) — one column per alpha.
            # Iterate columns in the same order as alphas_desc.
            for idx, alpha in enumerate(cox_path.alphas_):
                coef_col = cox_path.coef_[:, idx]

                # Skip alphas that shrink everything to zero — the model
                # has no discriminative power and predicts a constant risk.
                if np.all(coef_col == 0):
                    continue

                # Predict using only this alpha's coefficients to get a
                # per-alpha C-index without re-fitting.
                linear_pred = X_scaled_val @ coef_col
                ci = concordance_index_censored(
                    y_train[inner_val]["event"],
                    y_train[inner_val]["time"],
                    linear_pred,
                )[0]

                # Map the fitted alpha back to the caller's grid value.
                # cox_path.alphas_ may differ slightly from the requested
                # values due to internal scaling; match by closest value.
                nearest = min(alphas, key=lambda a: abs(a - alpha))
                alpha_scores[nearest].append(ci)

        except Exception:
            logger.exception(
                f"Inner-CV fold failed during alpha path search; skipping."
            )

    # Choose the alpha with the highest mean inner-fold C-index.
    # Fall back to the largest alpha (strongest regularisation) if every
    # fold failed — safer than falling back to the smallest.
    best_alpha = max(alphas)
    best_mean  = -1.0

    for alpha, scores in alpha_scores.items():
        if scores and np.mean(scores) > best_mean:
            best_mean  = np.mean(scores)
            best_alpha = alpha

    logger.info(
        f"{progress_desc} — best alpha={best_alpha} "
        f"(mean inner C-index={best_mean:.4f})"
    )
    return best_alpha


def fit_cox_lasso(
    X: pd.DataFrame,
    y: np.ndarray,
    cv: StratifiedKFold,
    alphas: list[float],
) -> tuple[list[float], list[float]]:
    """
    Evaluate Cox LASSO via 5-fold CV with inner-CV alpha selection.

    Alpha is selected by 3-fold inner CV on each outer training fold so
    the outer test fold is never used during hyperparameter search. The
    inner CV fits the full regularisation path in one CoxnetSurvivalAnalysis
    call per fold (warm starts), which is numerically stable and avoids the
    exp() overflow that occurs when fitting each alpha from scratch in the
    p >> n regime.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (n_patients × n_genes), indexed by PATIENT_ID.
    y : np.ndarray
        Structured survival array.
    cv : StratifiedKFold
        Outer cross-validation splitter.
    alphas : list[float]
        Candidate regularisation strengths forwarded to _select_alpha_inner_cv.
        Use LASSO_ALPHAS_14 for the 14-gene arm and LASSO_ALPHAS_ALL for the
        all-genes arm.

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
        tqdm(
            cv.split(X_array, y["event"]),
            total=cv.get_n_splits(),
            desc=f"Cox LASSO ({X.shape[1]} genes)",
        )
    ):
        X_train, X_test = X_array[train_idx], X_array[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # -----------------------------------------------------------------
        # Select alpha on the training fold only to prevent leakage.
        # -----------------------------------------------------------------
        best_alpha = _select_alpha_inner_cv(
            X_train,
            y_train,
            alphas=alphas,
            progress_desc=f"Fold {fold + 1} alpha search",
        )

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
    max_features: float | str = 0.5,
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
    max_features : float or str
        Fraction of features (float) or strategy string ('sqrt') to
        consider at each split. Default 0.5 suits the 14-gene set.
        Pass "sqrt" for the all-genes arm to limit tree correlation and
        computation when the feature space is thousands of genes wide.

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
        max_features=max_features,
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
    rsf_max_features: float | str = 0.5,
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
    rsf_max_features : float or str
        Passed to RandomSurvivalForest. Must match the value used in
        fit_rsf for the same feature set so the models are comparable.

    Returns
    -------
    lasso_oof : np.ndarray
        OOF risk scores for Cox LASSO (higher = worse prognosis).
    rsf_oof : np.ndarray
        OOF risk scores for RSF after global direction check.
    """
    X_array = X.values
    n = len(y)
    lasso_oof = np.zeros(n)
    rsf_oof = np.zeros(n)

    logger.info("Collecting out-of-fold predictions ...")

    for fold, (train_idx, test_idx) in enumerate(
        tqdm(
            cv.split(X_array, y["event"]),
            total=cv.get_n_splits(),
            desc="OOF predictions",
            leave=False,
        )
    ):
        X_train, X_test = X_array[train_idx], X_array[test_idx]
        y_train = y[train_idx]

        # -----------------------------------------------------------------
        # Cox LASSO — re-fit at the alpha chosen during fit_cox_lasso.
        # Uses a Pipeline so the scaler is fitted on X_train only, keeping
        # X_test unseen during normalisation (consistent with fit_cox_lasso).
        # -----------------------------------------------------------------
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
            max_features=rsf_max_features,
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
    lasso_all_cindices: list[float],
    rsf_all_cindices: list[float],
    ode_boot: dict,
    lasso_boot: dict,
    rsf_boot: dict,
    lasso_all_boot: dict,
    rsf_all_boot: dict,
    n_all_genes: int,
) -> pd.DataFrame:
    """
    Assemble, log, and save the five-model performance comparison table.

    Parameters
    ----------
    ode_baseline : float
        Full-cohort concordance index for the ODE predictor.
    lasso_cindices : list[float]
        Per-fold C-indices from fit_cox_lasso on 14-gene features.
    rsf_cindices : list[float]
        Per-fold C-indices from fit_rsf on 14-gene features.
    lasso_all_cindices : list[float]
        Per-fold C-indices from fit_cox_lasso on all-genes features.
    rsf_all_cindices : list[float]
        Per-fold C-indices from fit_rsf on all-genes features.
    ode_boot : dict
        Bootstrap result dict for the ODE predictor.
    lasso_boot : dict
        Bootstrap result dict for Cox LASSO (14-gene).
    rsf_boot : dict
        Bootstrap result dict for RSF (14-gene).
    lasso_all_boot : dict
        Bootstrap result dict for Cox LASSO (all-genes).
    rsf_all_boot : dict
        Bootstrap result dict for RSF (all-genes).
    n_all_genes : int
        Number of genes in the all-genes feature set; logged for traceability.

    Returns
    -------
    pd.DataFrame
        One row per model with CV C-index, SD, and bootstrap 95% CI.
    """
    valid_lasso     = [c for c in lasso_cindices     if not np.isnan(c)]
    valid_rsf       = [c for c in rsf_cindices       if not np.isnan(c)]
    valid_lasso_all = [c for c in lasso_all_cindices if not np.isnan(c)]
    valid_rsf_all   = [c for c in rsf_all_cindices   if not np.isnan(c)]

    all_genes_label = f"{n_all_genes} genes (raw)"

    rows = [
        {
            "Model":       "HR-DDR ODE (AUC_X)",
            "Feature_set": "14 genes → 4 params → ODE",
            "CV_cindex":   f"{ode_baseline:.3f}",
            "CV_sd":       "",
            "CV_display":  f"{ode_baseline:.3f}",
            "CI_low":      f"{ode_boot['ci_low']:.3f}",
            "CI_high":     f"{ode_boot['ci_high']:.3f}",
            "Notes":       "No CV needed (zero-shot predictor)",
        },
        {
            "Model":       "Cox LASSO (14-gene)",
            "Feature_set": "14 genes (raw)",
            "CV_cindex":   f"{np.mean(valid_lasso):.3f}",
            "CV_sd":       f"{np.std(valid_lasso):.3f}",
            "CV_display":  (
                f"{np.mean(valid_lasso):.3f} "
                f"± {np.std(valid_lasso):.3f}"
            ),
            "CI_low":      f"{lasso_boot['ci_low']:.3f}",
            "CI_high":     f"{lasso_boot['ci_high']:.3f}",
            "Notes":       "5-fold CV, alpha via inner 3-fold CV",
        },
        {
            "Model":       "RSF (14-gene)",
            "Feature_set": "14 genes (raw)",
            "CV_cindex":   f"{np.mean(valid_rsf):.3f}",
            "CV_sd":       f"{np.std(valid_rsf):.3f}",
            "CV_display":  (
                f"{np.mean(valid_rsf):.3f} "
                f"± {np.std(valid_rsf):.3f}"
            ),
            "CI_low":      f"{rsf_boot['ci_low']:.3f}",
            "CI_high":     f"{rsf_boot['ci_high']:.3f}",
            "Notes":       "5-fold CV, global OOF direction check",
        },
        {
            "Model":       "Cox LASSO (all-genes)",
            "Feature_set": all_genes_label,
            "CV_cindex":   f"{np.mean(valid_lasso_all):.3f}",
            "CV_sd":       f"{np.std(valid_lasso_all):.3f}",
            "CV_display":  (
                f"{np.mean(valid_lasso_all):.3f} "
                f"± {np.std(valid_lasso_all):.3f}"
            ),
            "CI_low":      f"{lasso_all_boot['ci_low']:.3f}",
            "CI_high":     f"{lasso_all_boot['ci_high']:.3f}",
            "Notes":       "5-fold CV, alpha via inner 3-fold CV",
        },
        {
            "Model":       "RSF (all-genes)",
            "Feature_set": all_genes_label,
            "CV_cindex":   f"{np.mean(valid_rsf_all):.3f}",
            "CV_sd":       f"{np.std(valid_rsf_all):.3f}",
            "CV_display":  (
                f"{np.mean(valid_rsf_all):.3f} "
                f"± {np.std(valid_rsf_all):.3f}"
            ),
            "CI_low":      f"{rsf_all_boot['ci_low']:.3f}",
            "CI_high":     f"{rsf_all_boot['ci_high']:.3f}",
            "Notes":       "5-fold CV, max_features=sqrt, OOF direction check",
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
    lasso_all_samples: np.ndarray,
    rsf_all_samples: np.ndarray,
) -> None:
    """
    Forest plot of bootstrap C-index means and 95% CIs for all five models.

    Each model is a dot (bootstrap mean) with a horizontal bar spanning
    the 95% percentile bootstrap CI. A dashed horizontal rule visually
    separates the ODE/14-gene arm from the all-genes arm.

    Parameters
    ----------
    ode_samples : np.ndarray
        Bootstrap C-index samples for the HR-DDR ODE.
    lasso_samples : np.ndarray
        Bootstrap C-index samples for Cox LASSO (14-gene).
    rsf_samples : np.ndarray
        Bootstrap C-index samples for RSF (14-gene).
    lasso_all_samples : np.ndarray
        Bootstrap C-index samples for Cox LASSO (all-genes).
    rsf_all_samples : np.ndarray
        Bootstrap C-index samples for RSF (all-genes).
    """
    models = [
        "HR-DDR ODE\n(AUC_X)",
        "Cox LASSO\n(14-gene)",
        "RSF\n(14-gene)",
        "Cox LASSO\n(all-genes)",
        "RSF\n(all-genes)",
    ]
    all_samples = [
        ode_samples, lasso_samples, rsf_samples,
        lasso_all_samples, rsf_all_samples,
    ]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0"]

    means = [np.mean(s)            for s in all_samples]
    lows  = [np.percentile(s, 2.5) for s in all_samples]
    highs = [np.percentile(s, 97.5) for s in all_samples]

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, (m, lo, hi, c) in enumerate(zip(means, lows, highs, colors)):
        ax.plot(
            [lo, hi], [i, i],
            color=c, linewidth=2.5, solid_capstyle="round",
        )
        ax.scatter(m, i, color=c, s=80, zorder=5)

    # Separator between ODE/14-gene models and the all-genes arm.
    ax.axhline(
        2.5, color="grey", linestyle=":", linewidth=0.8, alpha=0.7,
    )

    ax.axvline(
        0.5, color="black", linestyle="--",
        linewidth=1, alpha=0.6, label="Chance (0.5)",
    )
    ax.set_yticks(list(range(len(models))))
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlabel("C-index (bootstrap mean, 95% CI)", fontsize=11)
    ax.set_title(
        "Model Comparison — Prognostic Discrimination\n"
        "HGSOC TCGA (5-model benchmark)",
        fontsize=12,
    )
    ax.set_xlim(0.35, 0.70)
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
    lasso_all_samples: np.ndarray,
    rsf_all_samples: np.ndarray,
) -> None:
    """
    Histogram panel of bootstrap C-index distributions for all five models.

    Arranged as a 2×3 grid; the sixth panel is hidden to keep the layout
    uniform without a visible blank slot.

    Parameters
    ----------
    ode_samples : np.ndarray
        Bootstrap C-index samples for the HR-DDR ODE.
    lasso_samples : np.ndarray
        Bootstrap C-index samples for Cox LASSO (14-gene).
    rsf_samples : np.ndarray
        Bootstrap C-index samples for RSF (14-gene).
    lasso_all_samples : np.ndarray
        Bootstrap C-index samples for Cox LASSO (all-genes).
    rsf_all_samples : np.ndarray
        Bootstrap C-index samples for RSF (all-genes).
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    data = [
        ode_samples, lasso_samples, rsf_samples,
        lasso_all_samples, rsf_all_samples,
    ]
    labels = [
        "HR-DDR ODE (AUC_X)",
        "Cox LASSO (14-gene)",
        "RSF (14-gene)",
        "Cox LASSO (all-genes)",
        "RSF (all-genes)",
    ]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0"]

    # Flatten to a 1-D list; the sixth panel is hidden below.
    axes_flat = axes.flatten()

    for ax, d, label, color in zip(axes_flat[:5], data, labels, colors):
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

    # Hide the unused sixth panel.
    axes_flat[5].set_visible(False)

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
    logger.success(
        "[FILE] Saved: ./results/figures/fig_ml_bootstrap_distributions.png"
    )


# =================================================================
# Main
# =================================================================

def main() -> None:
    """
    Run the full ML benchmarking workflow.

    Steps
    -----
    1. Load ODE baseline C-index and aligned 14-gene expression / survival
       dataset.
    2. Load all-genes expression matrix from the raw FPKM file, restricted
       to the same patient cohort.
    3. 5-fold CV for Cox LASSO and RSF on both feature sets (14-gene and
       all-genes).
    4. Collect OOF predictions for each arm; apply global RSF direction
       checks independently.
    5. 1000-rep bootstrap CIs for all five models.
    6. Save five-model comparison table and two figures.
    """
    # Move warning suppression inside main() so it does not fire on import
    # and does not suppress warnings in other modules that import this one.
    warnings.filterwarnings("ignore", category=UserWarning)

    # -----------------------------------------------------------------
    # Load 14-gene data
    # -----------------------------------------------------------------
    ode_baseline_cindex = load_ode_baseline()
    X, y, _ = load_data()

    logger.info(
        f"ODE baseline C-index (AUC_X): {ode_baseline_cindex:.4f}"
    )
    logger.info(
        f"14-gene feature matrix: {X.shape}, "
        f"events: {int(y['event'].sum())}"
    )

    # -----------------------------------------------------------------
    # Load all-genes data aligned to the same patient cohort.
    #
    # X_all is loaded from the raw FPKM file at runtime so that no gene
    # pre-selection is applied — the feature set is the full transcriptome
    # minus genes with > 20% missing values.
    # -----------------------------------------------------------------
    X_all = load_all_genes_data(X.index)
    n_all_genes = X_all.shape[1]

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
    # 14-gene CV
    # -----------------------------------------------------------------
    logger.info('-' * 50)
    logger.info("14-gene arm: Cox LASSO (5-fold outer CV, inner 3-fold alpha)")
    lasso_cindices, lasso_alphas = fit_cox_lasso(X, y, cv, alphas=LASSO_ALPHAS_14)

    logger.info('-' * 50)
    logger.info("14-gene arm: RSF (5-fold CV, max_features=0.5)")
    rsf_cindices = fit_rsf(X, y, cv, max_features=0.5)

    # -----------------------------------------------------------------
    # All-genes CV
    # -----------------------------------------------------------------
    logger.info('-' * 50)
    logger.info(
        f"All-genes arm ({n_all_genes} genes): "
        "Cox LASSO (5-fold outer CV, inner 3-fold alpha)"
    )
    lasso_all_cindices, lasso_all_alphas = fit_cox_lasso(
        X_all, y, cv, alphas=LASSO_ALPHAS_ALL,
    )

    logger.info('-' * 50)
    logger.info(
        f"All-genes arm ({n_all_genes} genes): "
        "RSF (5-fold CV, max_features=sqrt)"
    )
    # max_features="sqrt" replaces 0.5 for the all-genes arm: with thousands
    # of features, 0.5 would sample far too many candidates per split,
    # increasing tree correlation and substantially inflating compute cost.
    rsf_all_cindices = fit_rsf(X_all, y, cv, max_features="sqrt")

    # -----------------------------------------------------------------
    # Assemble OOF predictions for bootstrap CI estimation
    # -----------------------------------------------------------------
    logger.info('-' * 50)
    logger.info("Collecting OOF predictions — 14-gene arm")
    lasso_oof, rsf_oof = collect_oof_predictions(
        X, y, cv, lasso_alphas, rsf_max_features=0.5,
    )

    logger.info('-' * 50)
    logger.info("Collecting OOF predictions — all-genes arm")
    lasso_all_oof, rsf_all_oof = collect_oof_predictions(
        X_all, y, cv, lasso_all_alphas, rsf_max_features="sqrt",
    )

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
    ode_boot       = bootstrap_cindex(ode_risk,      y, desc="ODE bootstrap")
    lasso_boot     = bootstrap_cindex(lasso_oof,     y, desc="LASSO-14 bootstrap")
    rsf_boot       = bootstrap_cindex(rsf_oof,       y, desc="RSF-14 bootstrap")
    lasso_all_boot = bootstrap_cindex(lasso_all_oof, y, desc="LASSO-all bootstrap")
    rsf_all_boot   = bootstrap_cindex(rsf_all_oof,   y, desc="RSF-all bootstrap")

    logger.info(
        f"ODE        (log_AUC_X): {ode_boot['mean']:.4f} "
        f"[{ode_boot['ci_low']:.4f}, {ode_boot['ci_high']:.4f}]"
    )
    logger.info(
        f"LASSO-14   (OOF):       {lasso_boot['mean']:.4f} "
        f"[{lasso_boot['ci_low']:.4f}, {lasso_boot['ci_high']:.4f}]"
    )
    logger.info(
        f"RSF-14     (OOF):       {rsf_boot['mean']:.4f} "
        f"[{rsf_boot['ci_low']:.4f}, {rsf_boot['ci_high']:.4f}]"
    )
    logger.info(
        f"LASSO-all  (OOF):       {lasso_all_boot['mean']:.4f} "
        f"[{lasso_all_boot['ci_low']:.4f}, {lasso_all_boot['ci_high']:.4f}]"
    )
    logger.info(
        f"RSF-all    (OOF):       {rsf_all_boot['mean']:.4f} "
        f"[{rsf_all_boot['ci_low']:.4f}, {rsf_all_boot['ci_high']:.4f}]"
    )

    # -----------------------------------------------------------------
    # Save comparison table and figures
    # -----------------------------------------------------------------
    save_comparison_table(
        ode_baseline_cindex,
        lasso_cindices,
        rsf_cindices,
        lasso_all_cindices,
        rsf_all_cindices,
        ode_boot,
        lasso_boot,
        rsf_boot,
        lasso_all_boot,
        rsf_all_boot,
        n_all_genes,
    )

    plot_forest(
        ode_boot["samples"],
        lasso_boot["samples"],
        rsf_boot["samples"],
        lasso_all_boot["samples"],
        rsf_all_boot["samples"],
    )
    plot_bootstrap_distributions(
        ode_boot["samples"],
        lasso_boot["samples"],
        rsf_boot["samples"],
        lasso_all_boot["samples"],
        rsf_all_boot["samples"],
    )


if __name__ == "__main__":
    main()