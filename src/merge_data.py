from pathlib import Path
import pandas as pd

# set the root path of the project
ROOT = Path(__file__).resolve().parent.parent

def main():
    # Load the cleaned clinical table.
    clinical_clean = pd.read_csv(ROOT / "data" / "processed" / "clinical_clean.csv")
    print(clinical_clean["PATIENT_ID"].is_unique)

    # Load the cleaned RNA expression table.
    expr_df = pd.read_csv(ROOT / "data" / "processed" / "rna_clean.csv")
    print(expr_df["PATIENT_ID"].is_unique)

    # Merge on patient ID.
    # "inner" keeps only patients present in both files.
    merged = clinical_clean.merge(expr_df, on="PATIENT_ID", how="inner")

    # Show basic sanity checks.
    print(f"\nFinal merged dataset shape: {merged.shape}")
    print(f"\nEvent rate: {merged['OS_EVENT'].mean():.2%}\n")

    # Save the final merged dataset for downstream analysis.
    merged.to_csv(ROOT / "data" / "processed" / "hgsoc_tcga_merged.csv", index=False)

if __name__ == "__main__":
    main()