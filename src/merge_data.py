from pathlib import Path
import pandas as pd
import logging

# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


def main():
    """
    Merge clinical and RNA-seq data into a single TCGA cohort table.

    This step defines the final analysis cohort used for:
    - ODE simulations
    - survival modelling
    - machine learning benchmarks
    """

    # -----------------------------------------------------------------
    # Load inputs
    # -----------------------------------------------------------------

    clinical = pd.read_csv(
        ROOT / "data" / "processed" / "clinical_clean.csv"
    )

    expr = pd.read_csv(
        ROOT / "data" / "processed" / "rna_clean.csv"
    )

    # -----------------------------------------------------------------
    # Basic integrity checks
    # -----------------------------------------------------------------

    clinical_ids = set(clinical["PATIENT_ID"])
    expr_ids = set(expr["PATIENT_ID"])

    overlap = clinical_ids & expr_ids

    logger.info("Cohort sizes before merge:")
    logger.info(f"    Clinical   : {len(clinical_ids)}")
    logger.info(f"    RNA        : {len(expr_ids)}")
    logger.info(f"    Overlap    : {len(overlap)}")

    # Sanity check: ensure merge is meaningful
    if len(overlap) == 0:
        raise ValueError(
            "No overlap between clinical and RNA cohorts. "
            "Check PATIENT_ID formatting."
        )

    # -----------------------------------------------------------------
    # Perform merge
    # -----------------------------------------------------------------

    merged = clinical.merge(expr, on="PATIENT_ID", how="inner")

    # -----------------------------------------------------------------
    # Post-merge validation
    # -----------------------------------------------------------------

    logger.summary(f"Merged dataset shape: {merged.shape}")

    expected_n = len(overlap)

    if len(merged) != expected_n:
        logger.warning(
            f"Unexpected merge size:\n"
            f"  Expected (overlap): {expected_n}\n"
            f"  Actual merged     : {len(merged)}"
        )

    # Survival sanity check
    if "OS_EVENT" in merged.columns:
        logger.info(
            f"Event rate in merged cohort: {merged['OS_EVENT'].mean():.2%}"
        )

    # Check for missing values introduced by merge
    missing_total = merged.isna().sum().sum()

    if missing_total > 0:
        logger.warning(
            f"Missing values present after merge: {missing_total}"
        )

    # -----------------------------------------------------------------
    # Save final cohort
    # -----------------------------------------------------------------

    out_path = ROOT / "data" / "processed" / "hgsoc_tcga_merged.csv"
    merged.to_csv(out_path, index=False)

    logger.success(f"[FILE] Saved merged cohort to ./data/processed/hgsoc_tcga_merged.csv")


if __name__ == "__main__":
    main()