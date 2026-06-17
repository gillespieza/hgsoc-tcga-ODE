from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

def main():
    clinical_clean = pd.read_csv(ROOT / "data" / "processed" / "clinical_clean.csv")
    expr_df = pd.read_csv(ROOT / "data" / "processed" / "expression" / "ode_genes_fpkm_subset.csv")

    merged = clinical_clean.merge(expr_df, on="PATIENT_ID", how="inner")

    print(f"Final merged dataset shape: {merged.shape}")
    print(f"Event rate: {merged['OS_EVENT'].mean():.2%}")
    print(merged[['PATIENT_ID', 'OS_MONTHS', 'OS_EVENT']].head())

    merged.to_csv(ROOT / "data" / "processed" / "tcga_ov_merged.csv", index=False)
    print("Saved: data/processed/tcga_ov_merged.csv")

if __name__ == "__main__":
    main()