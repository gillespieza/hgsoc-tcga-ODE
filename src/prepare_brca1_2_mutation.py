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
    Create BRCA1/2 mutation status for TCGA-OV cohort.

    Output:
    - Adds binary BRCA_MUTANT label to clinical dataset

    Definition:
    - 1 = at least one recorded mutation in BRCA1 or BRCA2
    - 0 = no detected mutations in these genes
    """

    # -----------------------------------------------------------------
    # Load mutation data
    # -----------------------------------------------------------------

    mutations = pd.read_csv(
        ROOT / "data" / "raw" / "mutation" / "data_mutations.txt",
        sep="\t",
        comment="#"
    )
    
    logger.info(f"Loaded mutation table: {mutations.shape}")

    # -----------------------------------------------------------------
    # Filter BRCA genes
    # -----------------------------------------------------------------

    brca_mut = mutations[
        mutations["Hugo_Symbol"].isin(["BRCA1", "BRCA2"])
    ].copy()

    # Convert sample → patient ID
    brca_patients = set(
        brca_mut["Tumor_Sample_Barcode"].str[:12]
    )

    logger.info(f"BRCA-mutant samples found: {len(brca_patients)}")

    # -----------------------------------------------------------------
    # Load clinical cohort
    # -----------------------------------------------------------------

    clinical_path = ROOT / "data" / "processed" / "clinical_clean.csv"
    clinical = pd.read_csv(clinical_path)

    clinical_ids = set(clinical["PATIENT_ID"])

    # -----------------------------------------------------------------
    # Sanity check: mutation coverage
    # -----------------------------------------------------------------

    overlap = len(clinical_ids & brca_patients)

    logger.info(
        f"Mutation overlap with clinical cohort: {overlap}"
    )

    missing_in_clinical = brca_patients - clinical_ids

    if missing_in_clinical:
        logger.warning(
            f"{len(missing_in_clinical)} BRCA-mutant samples "
            f"not found in clinical cohort (likely filtered earlier)"
        )

    # -----------------------------------------------------------------
    # Create binary label
    # -----------------------------------------------------------------

    clinical["BRCA_MUTANT"] = clinical["PATIENT_ID"].apply(
        lambda pid: 1 if pid in brca_patients else 0
    )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    n_mut = clinical["BRCA_MUTANT"].sum()
    n_wt = len(clinical) - n_mut

    logger.info('-' * 50)
    logger.summary(
        "BRCA status summary:\n"
        f"\t\t\t\t\t Mutant   : {n_mut}\n"
        f"\t\t\t\t\t Wildtype : {n_wt}"
    )
    logger.info('-' * 50)

    # -----------------------------------------------------------------
    # Save output (OVERWRITE WARNING)
    # -----------------------------------------------------------------

    out_path = ROOT / "data" / "processed" / "clinical_clean.csv"

    clinical.to_csv(out_path, index=False)

    logger.success(f"[FILE] Updated clinical file ./data/processed/clinical_clean.csv")


if __name__ == "__main__":
    main()