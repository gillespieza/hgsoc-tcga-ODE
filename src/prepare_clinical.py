# %%
import pandas as pd
from pathlib import Path

# set the root path of the project
ROOT = Path(__file__).resolve().parent.parent

# cBioPortal clinical files contain a metadata header (comments beginning with #) which must be skipped
clinical = pd.read_csv(
    ROOT / "data" / "raw" / "clinical" / "data_clinical_patient.txt",
    sep="\t",
    comment="#"
)

print("Shape:", clinical.shape, "\n")
print("Columns:", clinical.columns.tolist(), "\n")
print(clinical[['PATIENT_ID', 'OS_MONTHS', 'OS_STATUS']].head())

# %%
# Filter to patients with complete OS data
clinical_clean = clinical.dropna(subset=['OS_MONTHS', 'OS_STATUS'])
clinical_clean = clinical_clean[clinical_clean['OS_MONTHS'] > 0].copy()

print(f"Patients with complete OS data: {len(clinical_clean)}")

# Encode OS_STATUS as binary event
os_status_map = {
    '0:LIVING': 0,
    '1:DECEASED': 1
}
clinical_clean['OS_EVENT'] = clinical_clean['OS_STATUS'].map(os_status_map)

# Check event rate
print(f"OS event rate: {clinical_clean['OS_EVENT'].mean():.2%}")
print(clinical_clean[['PATIENT_ID', 'OS_MONTHS', 'OS_STATUS', 'OS_EVENT']].head(10))
# %%
