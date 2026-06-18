import pandas as pd
import sys
sys.path.insert(0, 'src')
import ode_model

merged = pd.read_csv(r'data\processed\hgsoc_tcga_merged.csv')
params_df = ode_model.compute_patient_params(merged)

# Check BRCA_MUTANT distribution
print("BRCA_MUTANT value counts:")
print(params_df['BRCA_MUTANT'].value_counts())

# Check BRCA_cap statistics
print("\nBRCA_cap statistics:")
print(f"  Min: {params_df['BRCA_cap'].min():.3f}")
print(f"  Max: {params_df['BRCA_cap'].max():.3f}")
print(f"  Median: {params_df['BRCA_cap'].median():.3f}")

# Check what patients we're selecting
low_brca_patients = params_df[params_df['BRCA_MUTANT'] == 1].sort_values('BRCA_cap')
high_brca_patients = params_df[params_df['BRCA_MUTANT'] == 0].sort_values('BRCA_cap', ascending=False)

print(f"\nLow BRCA patients (BRCA_MUTANT=1): {len(low_brca_patients)}")
if len(low_brca_patients) > 0:
    print(low_brca_patients[['PATIENT_ID', 'BRCA_cap']].head(3))

print(f"\nHigh BRCA patients (BRCA_MUTANT=0): {len(high_brca_patients)}")
if len(high_brca_patients) > 0:
    print(high_brca_patients[['PATIENT_ID', 'BRCA_cap']].head(3))

# Show what we're actually selecting
selected_low = params_df[params_df['BRCA_MUTANT'] == 1].iloc[0] if len(params_df[params_df['BRCA_MUTANT'] == 1]) > 0 else None
selected_high = params_df[params_df['BRCA_MUTANT'] == 0].nlargest(1, 'BRCA_cap').iloc[0] if len(params_df[params_df['BRCA_MUTANT'] == 0]) > 0 else None

print(f"\nSelected for analysis:")
print(f"  Low BRCA: PATIENT_ID={selected_low['PATIENT_ID']}, BRCA_cap={selected_low['BRCA_cap']:.3f}, BRCA_MUTANT={selected_low['BRCA_MUTANT']}")
print(f"  High BRCA: PATIENT_ID={selected_high['PATIENT_ID']}, BRCA_cap={selected_high['BRCA_cap']:.3f}, BRCA_MUTANT={selected_high['BRCA_MUTANT']}")
