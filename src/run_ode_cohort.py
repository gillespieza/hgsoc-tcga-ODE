"""
run_ode_cohort.py — Simulate the HR-DDR ODE for all patients in the HGSOC cohort.

Workflow:
1. Load hgsoc_tcga_merged.csv (clinical + expression data, 427 patients).
2. Derive patient-specific ODE parameters via ode_model.compute_patient_params().
3. Simulate each patient with ode_model.simulate_patient().
4. Extract summary scores with ode_model.compute_ode_scores().
5. Save raw ODE outputs to ode_scores.csv (all patients, including failures).
6. Filter integration failures and save survival_analysis_df.csv (clean cohort only).

Scientific rationale:
- The HR-DDR (Homologous Recombination DNA Damage Response) ODE model
  represents a simplified mechanistic description of tumour DNA repair.
- Each patient's gene-expression profile is transformed into a unique set
  of model parameters, creating a personalised "virtual patient".
- The model is simulated independently for every patient to generate
  patient-specific repair dynamics.
- The resulting time-series are compressed into biologically interpretable
  summary scores that can be used in downstream survival analysis.
"""

from pathlib import Path
import logging
import pandas as pd
import numpy as np
import sys

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------

# __file__ is the location of this script.
#
# Example:
#   project/
#       src/
#           run_ode_cohort.py
#
# parent        -> src/
# parent.parent -> project root/
#
# Using pathlib.Path avoids hard-coded file paths and makes the code
# portable across Windows, Linux and macOS.
ROOT = Path(__file__).resolve().parent.parent

# Python searches directories listed in sys.path when importing modules.
# Adding src/ ensures ode_model.py can be imported regardless of where
# the script is launched from.
# Add src/ to Python's module search path.
#
# This allows:
#     import ode_model
#
# even when the script is launched from outside the src directory.
#
# In larger projects this is often handled through packaging or
# installation, but explicitly modifying sys.path keeps the project
# structure simple for development and teaching purposes.
sys.path.insert(0, str(ROOT / "src"))

import ode_model


def main():
    # -----------------------------------------------------------------
    # Load merged clinical + expression dataset
    # -----------------------------------------------------------------

    # Each row corresponds to one patient and contains:
    # - Clinical variables (survival time, event status, etc.)
    # - Gene-expression measurements used to personalise the ODE model
    merged = pd.read_csv(ROOT / "data" / "processed" / "hgsoc_tcga_merged.csv")

    logger.info(f"Loaded merged dataset: {len(merged)} patients")

    # -----------------------------------------------------------------
    # Convert expression data into patient-specific ODE parameters
    # -----------------------------------------------------------------

    # Gene-expression measurements cannot be used directly by the ODE.
    # Instead, selected biomarkers are transformed into biologically
    # meaningful model parameters (repair rate, signalling strength,
    # apoptotic response, etc.).
    #
    # Each row of params_df therefore represents one virtual patient
    # with a unique parameter set.
    params_df = ode_model.compute_patient_params(merged)

    logger.info(
        f"Parameter table created: "
        f"{params_df.shape[0]} patients, "
        f"{params_df.shape[1]} columns"
    )

    # Store one result dictionary per patient.
    results = []

    # -----------------------------------------------------------------
    # Simulate every patient
    # -----------------------------------------------------------------

    # We treat each patient as an independent virtual experiment.
    # The same biological model is simulated repeatedly, but with
    # different parameter values derived from each patient's tumour.
    #
    # iterrows() is convenient and readable for moderate-sized cohorts
    # (~hundreds of patients). For very large datasets, vectorised or
    # parallel approaches would typically be preferred.
    for _, row in params_df.iterrows():

        # Convert the current patient row into a Python dictionary.
        p = row.to_dict()

        # Simulate the HR-DDR ODE system for this patient.
        sim = ode_model.simulate_patient(p, ode_model.GLOBAL_PARAMS)

        # The full ODE solution contains many timepoints.
        # Survival models cannot easily use entire trajectories,
        # so we compress each simulation into a small set of
        # biologically interpretable summary features.
        #
        # Examples:
        #   AUC_X    = cumulative signalling burden
        #   X_peak   = maximum signalling intensity
        #   T_repair = time required to repair damage
        #   D_resid  = residual damage at the end of simulation
        scores = ode_model.compute_ode_scores(sim)

        # Add survival and clinical metadata needed for downstream
        # Kaplan-Meier and Cox proportional hazards analyses.
        scores.update(
            {
                "OS_MONTHS": p["OS_MONTHS"],
                "OS_EVENT": p["OS_EVENT"],
                "BRCA_MUTANT": p["BRCA_MUTANT"],
            }
        )

        results.append(scores)

    # -----------------------------------------------------------------
    # Assemble results table
    # -----------------------------------------------------------------

    scores_df = pd.DataFrame(results)

    # -----------------------------------------------------------------
    # Integration failure report
    # -----------------------------------------------------------------

    # Numerical ODE solvers occasionally fail to converge.
    # Rather than silently removing those patients, we record failures
    # explicitly for transparency and reproducibility.
    n_total = len(scores_df)
    n_failed = (~scores_df["success"]).sum()
    n_success = scores_df["success"].sum()

    logger.summary(
        f"ODE simulation complete: "
        f"{n_success}/{n_total} successful, "
        f"{n_failed} failed ({n_failed / n_total:.1%})"
    )

    if n_failed > 0:
        failed_ids = scores_df.loc[
            ~scores_df["success"], "PATIENT_ID"
        ].tolist()

        logger.warning(
            f"Integration failures ({n_failed}): {failed_ids}"
        )

    # -----------------------------------------------------------------
    # Save complete ODE output table
    # -----------------------------------------------------------------

    # This file contains every patient, including failures.
    #
    # The success column acts as a quality-control indicator,
    # allowing later inspection of unsuccessful simulations.
    scores_df.to_csv(
        ROOT / "data" / "processed" / "ode_scores.csv",
        index=False,
    )

    logger.success(
        f"[FILE] Saved: ./data/processed/ode_scores.csv "
        f"({n_total} rows, including failures)"
    )

    # -----------------------------------------------------------------
    # Build survival-analysis dataset
    # -----------------------------------------------------------------

    # Survival analysis assumes every covariate has a valid numerical
    # value. Failed ODE integrations generate undefined summary scores,
    # which can cause model-fitting errors or introduce bias if handled
    # inconsistently.
    #
    # We therefore retain failures in ode_scores.csv for transparency
    # but exclude them from the statistical analysis cohort.
    analysis_df = (
        scores_df[scores_df["success"]]
        .copy()
        .reset_index(drop=True)
    )

    if len(analysis_df) < len(scores_df):
        logger.warning(
            f"Excluded {len(scores_df) - len(analysis_df)} patients "
            f"from survival analysis due to ODE integration failure."
        )

    # -----------------------------------------------------------------
    # Feature engineering
    # -----------------------------------------------------------------

    # Many biological variables are right-skewed.
    # Applying a logarithmic transformation often improves
    # interpretability and statistical behaviour.
    #
    # log1p(x) = log(1 + x)
    #
    # Using log1p rather than log avoids problems when x = 0.
    for col in ["AUC_X", "X_peak", "T_repair", "D_resid"]:
        analysis_df[f"log_{col}"] = np.log1p(analysis_df[col])

    # -----------------------------------------------------------------
    # Save analysis-ready cohort
    # -----------------------------------------------------------------

    analysis_df.to_csv(
        ROOT / "data" / "processed" / "survival_analysis_df.csv",
        index=False,
    )

    logger.success(
        f"[FILE] Saved: ./data/processed/survival_analysis_df.csv "
        f"({len(analysis_df)} rows)"
    )

    # -----------------------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------------------

    # These summaries provide a quick validation that:
    # - patient counts are as expected
    # - survival data appear reasonable
    # - ODE-derived features fall within plausible ranges
    logger.info('-' * 50)
    logger.info(
        f"Sanity checks (survival_analysis_df):\n"
        f"\t\t\t\t\tShape          : {analysis_df.shape}\n"
        f"\t\t\t\t\tOS event rate  : {analysis_df['OS_EVENT'].mean():.1%}\n"
        f"\t\t\t\t\tMedian OS      : {analysis_df['OS_MONTHS'].median():.1f} months\n"
        f"\t\t\t\t\tAUC_X range    : {analysis_df['AUC_X'].min():.4f} – "
        f"{analysis_df['AUC_X'].max():.4f}\n"
        f"\t\t\t\t\tAUC_X median   : {analysis_df['AUC_X'].median():.4f}\n"
        f"\t\t\t\t\tBRCA mutant n  : {int(analysis_df['BRCA_MUTANT'].sum())}"
    )
    logger.info('-' * 50)


if __name__ == "__main__":
    main()