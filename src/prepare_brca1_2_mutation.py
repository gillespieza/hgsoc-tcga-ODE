from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

def main():
    mutations = pd.read_csv(
        ROOT / "data" / "raw" / "mutation" / "data_mutations.txt",
        sep="\t",
        comment="#"
    )

    brca_mut = mutations[mutations['Hugo_Symbol'].isin(['BRCA1', 'BRCA2'])]

    brca_patients = set(brca_mut['Tumor_Sample_Barcode'].str[:12]) # TCGA patient ID prefix

    clinical_clean = pd.read_csv(ROOT / "data" / "processed" / "clinical_clean.csv")
    clinical_clean['BRCA_MUTANT'] = clinical_clean['PATIENT_ID'].apply(
        lambda pid: 1 if pid in brca_patients else 0
    )

    print(f"\nBRCA1/2 mutant patients: {clinical_clean['BRCA_MUTANT'].sum()}")
    print(f"\nBRCA1/2 wildtype patients: {(clinical_clean['BRCA_MUTANT'] == 0).sum()}\n")

    clinical_clean.to_csv(ROOT / "data" / "processed" / "clinical_clean.csv", index=False)

if __name__ == "__main__":
    main()