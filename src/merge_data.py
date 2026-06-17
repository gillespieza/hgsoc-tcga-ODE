from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

def main():
    clinical_clean = pd.read_csv(ROOT / "data" / "processed" / "clinical_clean.csv")
    expr_df = pd.read_csv(ROOT / "data" / "processed" / "rna_clean.csv")

    merged = clinical_clean.merge(expr_df, on="PATIENT_ID", how="inner")

    print(f"\nFinal merged dataset shape: {merged.shape}")
    print(f"\nEvent rate: {merged['OS_EVENT'].mean():.2%}\n")
    #print(merged[['PATIENT_ID', 'OS_MONTHS', 'OS_EVENT']].head())

    merged.to_csv(ROOT / "data" / "processed" / "hgsoc_tcga_merged.csv", index=False)
    #print("Saved: data/processed/hgsoc_tcga_merged.csv")

if __name__ == "__main__":
    main()