import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'src')
import ode_model

merged = pd.read_csv(r'data\processed\hgsoc_tcga_merged.csv')
params_df = ode_model.compute_patient_params(merged)

low_brca = params_df[params_df['BRCA_MUTANT'] == 1].iloc[0]
high_brca = params_df[params_df['BRCA_MUTANT'] == 0].nlargest(1, 'BRCA_cap').iloc[0]

print("=== PATIENT COMPARISON ===\n")
print(f"LOW BRCA (BRCA_MUTANT=1):")
print(f"  BRCA_cap: {low_brca['BRCA_cap']:.3f}")
print(f"  ATM_tot: {low_brca['ATM_tot']:.3f}")
print(f"  CHK_tot: {low_brca['CHK_tot']:.3f}")
print(f"  BCL2_ratio: {low_brca['BCL2_ratio']:.3f}")

print(f"\nHIGH BRCA (BRCA_MUTANT=0):")
print(f"  BRCA_cap: {high_brca['BRCA_cap']:.3f}")
print(f"  ATM_tot: {high_brca['ATM_tot']:.3f}")
print(f"  CHK_tot: {high_brca['CHK_tot']:.3f}")
print(f"  BCL2_ratio: {high_brca['BCL2_ratio']:.3f}")

# Simulate and check intermediate states
print("\n=== SIMULATIONS ===\n")

sim_low = ode_model.simulate_patient(low_brca.to_dict(), ode_model.GLOBAL_PARAMS)
sim_high = ode_model.simulate_patient(high_brca.to_dict(), ode_model.GLOBAL_PARAMS)

score_low = ode_model.compute_ode_scores(sim_low)
score_high = ode_model.compute_ode_scores(sim_high)

print(f"LOW BRCA:")
print(f"  D_max: {np.max(sim_low['D']):.4f}")
print(f"  A_max: {np.max(sim_low['A']):.4f}")
print(f"  C_max: {np.max(sim_low['C']):.4f}")
print(f"  R_initial: {sim_low['R'][0]:.4f}")
print(f"  X_max: {np.max(sim_low['X']):.4f}")
print(f"  AUC_X: {score_low['AUC_X']:.3f}")

print(f"\nHIGH BRCA:")
print(f"  D_max: {np.max(sim_high['D']):.4f}")
print(f"  A_max: {np.max(sim_high['A']):.4f}")
print(f"  C_max: {np.max(sim_high['C']):.4f}")
print(f"  R_initial: {sim_high['R'][0]:.4f}")
print(f"  X_max: {np.max(sim_high['X']):.4f}")
print(f"  AUC_X: {score_high['AUC_X']:.3f}")

print(f"\n=== ANALYSIS ===")
print(f"Expected: Low BRCA AUC_X >> High BRCA AUC_X")
print(f"Actual:   Low BRCA AUC_X ({score_low['AUC_X']:.3f}) {'<' if score_low['AUC_X'] < score_high['AUC_X'] else '>'} High BRCA AUC_X ({score_high['AUC_X']:.3f})")
print(f"Status:   {'ERROR - FLIPPED!' if score_low['AUC_X'] < score_high['AUC_X'] else 'OK'}")
