import pandas as pd
from pathlib import Path
import logging

# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------

# This script lives in src/, so we go up one level to reach project root.
ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


def main():
    """
    Prepare TCGA clinical data for survival analysis.

    Steps:
    1. Load cBioPortal clinical patient table
    2. Remove missing or invalid survival records
    3. Deduplicate patient entries
    4. Convert survival status into binary event variable
    5. Save cleaned dataset for downstream modelling
    """

    # -----------------------------------------------------------------
    # Load raw clinical data
    # -----------------------------------------------------------------

    # cBioPortal files include metadata lines starting with "#"
    # These must be ignored for correct parsing.
    clinical = pd.read_csv(
        ROOT / "data" / "raw" / "clinical" / "data_clinical_patient.txt",
        sep="\t",
        comment="#"
    )

    logger.info(f"Loaded clinical data: {clinical.shape[0]} rows")

    # -----------------------------------------------------------------
    # Basic survival filtering
    # -----------------------------------------------------------------

    # Survival analysis requires:
    # - known follow-up time (OS_MONTHS)
    # - known event status (OS_STATUS)
    # therefore drop patients with missing survival information.
    clinical_clean = clinical.dropna(subset=["OS_MONTHS", "OS_STATUS"])

    # Remove invalid follow-up times.
    # OS_MONTHS <= 0 is not biologically meaningful for survival modelling.
    clinical_clean = clinical_clean[clinical_clean["OS_MONTHS"] > 0].copy()

    logger.info(
        f"After filtering missing/invalid survival data: "
        f"{len(clinical_clean)} patients"
    )

    # -----------------------------------------------------------------
    # Deduplication
    # -----------------------------------------------------------------

    # TCGA tables occasionally contain duplicated patient IDs
    # due to multiple samples or annotation merges.
    clinical_clean = clinical_clean.drop_duplicates(
        subset=["PATIENT_ID"],
        keep="first"
    )

    logger.info(f"After deduplication: {len(clinical_clean)} patients")

    # -----------------------------------------------------------------
    # Convert survival status to binary event variable
    # -----------------------------------------------------------------

    # OS_STATUS format:
    #   "0:LIVING"     -> censored (0)
    #   "1:DECEASED"   -> event occurred (1)
    os_status_map = {
        "0:LIVING": 0,
        "1:DECEASED": 1
    }

    clinical_clean["OS_EVENT"] = clinical_clean["OS_STATUS"].map(os_status_map)

    # -----------------------------------------------------------------
    # Validation check (important!)
    # -----------------------------------------------------------------

    # If unmapped values exist, survival modelling becomes invalid.
    unmapped = clinical_clean["OS_EVENT"].isna().sum()

    if unmapped > 0:
        logger.warning(
            f"Found {unmapped} unmapped OS_STATUS values. "
            f"These rows may be excluded downstream."
        )

    # -----------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------

    logger.info('-' * 50)
    logger.summary(
        f"SURVIVAL SUMMARY:\n"
        f"\t\t\t\t\t Patients          : {len(clinical_clean)}\n"
        f"\t\t\t\t\t Event rate        : {clinical_clean['OS_EVENT'].mean():.1%}\n"
        f"\t\t\t\t\t Median OS (mo)    : {clinical_clean['OS_MONTHS'].median():.1f}"
    )
    logger.info('-' * 50)

    # -----------------------------------------------------------------
    # Save processed output
    # -----------------------------------------------------------------

    out_path = ROOT / "data" / "processed" / "clinical_clean.csv"
    clinical_clean.to_csv(out_path, index=False)

    logger.success(f"[FILE] Saved cleaned clinical data to ./data/processed/clinical_clean.csv")


if __name__ == "__main__":
    main()