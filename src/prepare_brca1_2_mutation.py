from pathlib import Path
import pandas as pd

# set the root path of the project
ROOT = Path(__file__).resolve().parent.parent

def main():
    # Load the raw mutation table from cBioPortal.
    # comment="#" skips metadata lines at the top of the file.
    mutations = pd.read_csv(
        ROOT / "data" / "raw" / "mutation" / "data_mutations.txt",
        sep="\t",
        comment="#"
    )

    # Keep only BRCA1 and BRCA2 mutation records.
    brca_mut = mutations[mutations['Hugo_Symbol'].isin(['BRCA1', 'BRCA2'])]

    # Convert sample barcodes to 12-character patient IDs.
    # This collapses multiple samples from the same patient into one identifier.
    brca_patients = set(brca_mut['Tumor_Sample_Barcode'].str[:12]) # TCGA patient ID prefix

    # Load the cleaned clinical table from the previous step.
    clinical_clean = pd.read_csv(ROOT / "data" / "processed" / "clinical_clean.csv")

    # Create a binary mutation label:
    # 1 = patient has a BRCA1/2 mutation, 0 = no BRCA1/2 mutation detected.
    clinical_clean["BRCA_MUTANT"] = clinical_clean["PATIENT_ID"].apply(
        lambda pid: 1 if pid in brca_patients else 0
    )

    # Print a simple summary of the mutation counts.
    print(f"\nBRCA1/2 mutant patients: {clinical_clean['BRCA_MUTANT'].sum()}")
    print(f"\nBRCA1/2 wildtype patients: {(clinical_clean['BRCA_MUTANT'] == 0).sum()}\n")

    # Save the updated clinical table back to disk.
    clinical_clean.to_csv(ROOT / "data" / "processed" / "clinical_clean.csv", index=False)

if __name__ == "__main__":
    main()