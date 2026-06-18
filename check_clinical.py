import pandas as pd

clinical = pd.read_csv(r'data\processed\clinical_clean.csv')

# Get BRCA_MUTANT breakdown
print("BRCA_MUTANT status in clinical data:")
print(clinical['BRCA_MUTANT'].value_counts())

# Find the patients we're using in the analysis
our_patients = ['TCGA-04-1331', 'TCGA-59-A5PD']

print("\nOur selected patients:")
for pid in our_patients:
    subset = clinical[clinical['PATIENT_ID'] == pid]
    if len(subset) > 0:
        print(f"  {pid}: BRCA_MUTANT = {subset['BRCA_MUTANT'].values[0]}")
    else:
        print(f"  {pid}: NOT FOUND")

# Check if the BRCA_MUTANT encoding might be reversed
print("\nSample BRCA_MUTANT values (first 10 patients):")
print(clinical[['PATIENT_ID', 'BRCA_MUTANT']].head(10))
