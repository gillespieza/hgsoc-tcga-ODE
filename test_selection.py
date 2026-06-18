import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'src')
import ode_model

merged = pd.read_csv(r'data\processed\hgsoc_tcga_merged.csv')
params_df = ode_model.compute_patient_params(merged)

# NEW SELECTION: Use extremes
low_brca_candidates = params_df[params_df['BRCA_MUTANT'] == 1]
low_brca = low_brca_candidates.nsmallest(1, 'BRCA_cap').iloc[0]

high_brca_candidates = params_df[params_df['BRCA_MUTANT'] == 0]
high_brca = high_brca_candidates.nlargest(1, 'BRCA_cap').iloc[0]

print("=== PATIENT COMPARISON (WITH IMPROVED SELECTION) ===\n")
print(f"LOW BRCA (BRCA_MUTANT=1, lowest BRCA_cap):")
print(f"  PATIENT_ID: {low_brca['PATIENT_ID']}")
print(f"  BRCA_cap: {low_brca['BRCA_cap']:.3f}")
print(f"  ATM_tot: {low_brca['ATM_tot']:.3f}")
print(f"  CHK_tot: {low_brca['CHK_tot']:.3f}")
print(f"  BCL2_ratio: {low_brca['BCL2_ratio']:.3f}")

print(f"\nHIGH BRCA (BRCA_MUTANT=0, highest BRCA_cap):")
print(f"  PATIENT_ID: {high_brca['PATIENT_ID']}")
print(f"  BRCA_cap: {high_brca['BRCA_cap']:.3f}")
print(f"  ATM_tot: {high_brca['ATM_tot']:.3f}")
print(f"  CHK_tot: {high_brca['CHK_tot']:.3f}")
print(f"  BCL2_ratio: {high_brca['BCL2_ratio']:.3f}")

# Simulate
sim_low = ode_model.simulate_patient(low_brca.to_dict(), ode_model.GLOBAL_PARAMS)
sim_high = ode_model.simulate_patient(high_brca.to_dict(), ode_model.GLOBAL_PARAMS)

score_low = ode_model.compute_ode_scores(sim_low)
score_high = ode_model.compute_ode_scores(sim_high)

print("\n=== SIMULATIONS ===\n")

print(f"LOW BRCA:")
print(f"  D_max: {np.max(sim_low['D']):.4f}")
print(f"  A_max: {np.max(sim_low['A']):.4f}")
print(f"  C_max: {np.max(sim_low['C']):.4f}")
print(f"  X_max: {np.max(sim_low['X']):.4f}")
print(f"  AUC_X: {score_low['AUC_X']:.3f}")

print(f"\nHIGH BRCA:")
print(f"  D_max: {np.max(sim_high['D']):.4f}")
print(f"  A_max: {np.max(sim_high['A']):.4f}")
print(f"  C_max: {np.max(sim_high['C']):.4f}")
print(f"  X_max: {np.max(sim_high['X']):.4f}")
print(f"  AUC_X: {score_high['AUC_X']:.3f}")

print(f"\n=== RESULT ===")
if score_low['AUC_X'] > score_high['AUC_X']:
    print(f"✓ CORRECT: Low BRCA AUC_X ({score_low['AUC_X']:.3f}) >> High BRCA AUC_X ({score_high['AUC_X']:.3f})")
else:
    print(f"✗ INCORRECT: Low BRCA AUC_X ({score_low['AUC_X']:.3f}) <= High BRCA AUC_X ({score_high['AUC_X']:.3f})")
