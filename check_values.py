import pandas as pd
import sys
sys.path.insert(0, 'src')
import ode_model

merged = pd.read_csv(r'data\processed\hgsoc_tcga_merged.csv')
params_df = ode_model.compute_patient_params(merged)
low_brca = params_df[params_df['BRCA_MUTANT'] == 1].iloc[0]
high_brca = params_df[params_df['BRCA_MUTANT'] == 0].nlargest(1, 'BRCA_cap').iloc[0]

sim_low = ode_model.simulate_patient(low_brca.to_dict(), ode_model.GLOBAL_PARAMS)
sim_high = ode_model.simulate_patient(high_brca.to_dict(), ode_model.GLOBAL_PARAMS)

score_low = ode_model.compute_ode_scores(sim_low)
score_high = ode_model.compute_ode_scores(sim_high)

print(f"Low BRCA (BRCA_cap={low_brca['BRCA_cap']:.3f}): AUC_X = {score_low['AUC_X']:.3f}")
print(f"High BRCA (BRCA_cap={high_brca['BRCA_cap']:.3f}): AUC_X = {score_high['AUC_X']:.3f}")
print(f"\nRatio (Low/High): {score_low['AUC_X'] / (score_high['AUC_X'] + 0.001):.1f}x")

# Check parameters being used
print(f"\nCurrent k_x: {ode_model.GLOBAL_PARAMS['k_x']}")
print(f"Current d_x: {ode_model.GLOBAL_PARAMS['d_x']}")
