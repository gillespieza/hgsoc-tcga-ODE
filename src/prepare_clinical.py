"""
prepare_clinical.py -- Clean TCGA clinical data for survival analysis.

Steps
-----
1. Load cBioPortal clinical patient table.
2. Remove missing or invalid survival records.
3. Deduplicate patient entries.
4. Convert survival status into a binary event variable (OS_EVENT).
5. Save cleaned dataset for downstream modelling.

The cohort summary figure (attrition waterfall, KM curve, BRCA split) is
produced by merge_data.py, which is the first step at which all attrition
counts and the BRCA_MUTANT flag are simultaneously available.

Outputs
-------
- data/processed/clinical_clean.csv
"""

import logging
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


# =================================================================
# Main
# =================================================================

def main() -> None:
    """
    Prepare TCGA clinical data for survival analysis.

    Steps
    -----
    1. Load cBioPortal clinical patient table.
    2. Remove missing or invalid survival records.
    3. Deduplicate patient entries.
    4. Convert survival status into binary event variable (OS_EVENT).
    5. Save cleaned dataset for downstream modelling.
    """

    # -----------------------------------------------------------------
    # Load raw clinical data
    # -----------------------------------------------------------------

    # cBioPortal files include metadata lines starting with "#"
    # that must be skipped for correct parsing.
    clinical = pd.read_csv(
        ROOT / "data" / "raw" / "clinical" / "data_clinical_patient.txt",
        sep="\t",
        comment="#",
    )

    n_raw = clinical.shape[0]
    logger.info(f"Loaded clinical data: {n_raw} rows")

    # -----------------------------------------------------------------
    # Basic survival filtering
    # -----------------------------------------------------------------

    # Survival analysis requires known follow-up time (OS_MONTHS) and
    # known event status (OS_STATUS); drop patients missing either.
    clinical_clean = clinical.dropna(subset=["OS_MONTHS", "OS_STATUS"])

    # OS_MONTHS <= 0 is not biologically meaningful for survival modelling.
    clinical_clean = clinical_clean[clinical_clean["OS_MONTHS"] > 0].copy()

    n_after_os_filter = len(clinical_clean)
    logger.info(
        f"After filtering missing/invalid survival data: "
        f"{n_after_os_filter} patients"
    )

    # -----------------------------------------------------------------
    # Deduplication
    # -----------------------------------------------------------------

    # TCGA tables occasionally contain duplicated patient IDs due to
    # multiple samples or annotation merges.
    clinical_clean = clinical_clean.drop_duplicates(
        subset=["PATIENT_ID"],
        keep="first",
    )

    n_after_dedup = len(clinical_clean)
    logger.info(f"After deduplication: {n_after_dedup} patients")

    # -----------------------------------------------------------------
    # Convert survival status to binary event variable
    # -----------------------------------------------------------------

    # OS_STATUS format:
    #   "0:LIVING"   -> censored (0)
    #   "1:DECEASED" -> event occurred (1)
    os_status_map = {
        "0:LIVING":   0,
        "1:DECEASED": 1,
    }

    clinical_clean["OS_EVENT"] = clinical_clean["OS_STATUS"].map(os_status_map)

    # -----------------------------------------------------------------
    # Validation check
    # -----------------------------------------------------------------

    # Unmapped values produce NaN in OS_EVENT, which would silently
    # corrupt downstream survival models.
    unmapped = clinical_clean["OS_EVENT"].isna().sum()
    if unmapped > 0:
        logger.warning(
            f"Found {unmapped} unmapped OS_STATUS values. "
            "These rows may be excluded downstream."
        )

    # -----------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------

    logger.info("-" * 50)
    logger.summary(
        f"SURVIVAL SUMMARY:\n"
        f"\t\t\t\t\t Patients          : {len(clinical_clean)}\n"
        f"\t\t\t\t\t Event rate        : "
        f"{clinical_clean['OS_EVENT'].mean():.1%}\n"
        f"\t\t\t\t\t Median OS (mo)    : "
        f"{clinical_clean['OS_MONTHS'].median():.1f}"
    )
    logger.info("-" * 50)

    # -----------------------------------------------------------------
    # Save processed output
    # -----------------------------------------------------------------

    out_path = ROOT / "data" / "processed" / "clinical_clean.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clinical_clean.to_csv(out_path, index=False)
    logger.success("[FILE] Saved: .data/processed/clinical_clean.csv")


if __name__ == "__main__":
    main()