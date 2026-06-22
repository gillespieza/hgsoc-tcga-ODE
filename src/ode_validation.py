import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, 'src')
import ode_model

ROOT = Path(__file__).resolve().parent.parent

merged = pd.read_csv(ROOT / "data" / "processed" / "hgsoc_tcga_merged.csv")
params_df = ode_model.compute_patient_params(merged)

# Select two representative patients for validation:
# 1. Low BRCA_cap (BRCA-mutant, or lowest-quartile BRCA_cap) — expect high AUC_X
# 2. High BRCA_cap (BRCA-wildtype, highest-quartile BRCA_cap) — expect low AUC_X
low_brca = params_df[params_df['BRCA_MUTANT'] == 1].nsmallest(1, 'BRCA_cap').iloc[0]

high_brca = params_df[params_df['BRCA_MUTANT'] == 0].nlargest(1, 'BRCA_cap').iloc[0]

print(f"Low-BRCA patient  — BRCA_cap: {low_brca['BRCA_cap']:.3f}, "
      f"ATM_tot: {low_brca['ATM_tot']:.3f}, BCL2_ratio: {low_brca['BCL2_ratio']:.3f}")
print(f"High-BRCA patient — BRCA_cap: {high_brca['BRCA_cap']:.3f}, "
      f"ATM_tot: {high_brca['ATM_tot']:.3f}, BCL2_ratio: {high_brca['BCL2_ratio']:.3f}")

# Run simulations
sim_low  = ode_model.simulate_patient(low_brca.to_dict(),  ode_model.GLOBAL_PARAMS)
sim_high = ode_model.simulate_patient(high_brca.to_dict(), ode_model.GLOBAL_PARAMS)

score_low  = ode_model.compute_ode_scores(sim_low)
score_high = ode_model.compute_ode_scores(sim_high)

print(f"\nLow-BRCA  AUC_X: {score_low['AUC_X']:.3f}")
print(f"High-BRCA AUC_X: {score_high['AUC_X']:.3f}")

# Plot both patients on the same axes
fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), sharex=True)
labels = ['D — DNA damage', 'A — ATM/ATR*', 'C — CHK1/2* (Checkpoint kinetics)', 'R — HR complex (Repair capacity)', 'X — Apoptotic signal']
states = ['D', 'A', 'C', 'R', 'X']

for ax, key, label in zip(axes, states, labels):
    ax.plot(sim_low['t'], sim_low[key], color='#d62728', linewidth=2.0,
            label=f"Low BRCA_cap ({low_brca['PATIENT_ID']})")
    ax.plot(sim_high['t'], sim_high[key], color='#1f77b4', linewidth=2.0,
            label=f"High BRCA_cap ({high_brca['PATIENT_ID']})")
    ax.set_title(label, fontsize=9)
    ax.set_xlabel('Time (h)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, linewidth=0.6)

axes[0].set_ylabel('Signal level', fontsize=8)
axes[-1].legend(fontsize=7, frameon=False, loc='best')

plt.suptitle(
    f'HR-DDR ODE validation: two representative patients\n'
    f'Low-BRCA AUC_X = {score_low["AUC_X"]:.3f}   |   High-BRCA AUC_X = {score_high["AUC_X"]:.3f}',
    fontsize=11
)
plt.tight_layout()

plt.savefig(ROOT / "data" / "processed" / "ode_validation_trajectories.png", dpi=150, bbox_inches='tight')
plt.close(fig)

print("Saved: data/processed/ode_validation_trajectories.png")