import pandas as pd
from pathlib import Path

# set the root path of the project
ROOT = Path(__file__).resolve().parent.parent

def main():
    # cBioPortal clinical files contain metadata lines starting with "#".
    # The comment parameter tells pandas to ignore those lines.
    clinical = pd.read_csv(
        ROOT / "data" / "raw" / "clinical" / "data_clinical_patient.txt",
        sep="\t",
        comment="#"
    )

    # Keep only patients with non-missing overall survival information.
    clinical_clean = clinical.dropna(subset=['OS_MONTHS', 'OS_STATUS'])

    # Remove patients with non-positive survival time.
    # This avoids invalid or unusable follow-up values.
    clinical_clean = clinical_clean[clinical_clean['OS_MONTHS'] > 0].copy()

    # Remove duplicate patient rows if any exist.
    # keep="first" keeps the first occurrence of each PATIENT_ID.
    clinical_clean = clinical_clean.drop_duplicates(subset=["PATIENT_ID"], keep="first")

    print(f"\nPatients with complete OS data: {len(clinical_clean)}\n")

    # Convert OS_STATUS from text labels to binary event indicators.
    # 0 = alive at last follow-up, 1 = deceased.
    os_status_map = {
        '0:LIVING': 0,
        '1:DECEASED': 1
    }
    # Map the OS_STATUS to binary events
    clinical_clean['OS_EVENT'] = clinical_clean['OS_STATUS'].map(os_status_map)

    # Save the cleaned clinical table to a CSV for the merge step later.
    out_path = ROOT / "data" / "processed" / "clinical_clean.csv"
    clinical_clean.to_csv(out_path, index=False)
    
    #print("Saved: data/processed/clinical_clean.csv")

if __name__ == "__main__":
    main()