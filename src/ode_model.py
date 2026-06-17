"""
ode_model.py — HR-DDR ODE model for HGSOC patient-specific survival prediction.

Five-variable ODE system:
    D(t) : DNA damage load (normalised DSBs)
    A(t) : Activated ATM/ATR (checkpoint kinase signal)
    C(t) : Activated CHK1/CHK2 (checkpoint effector signal)
    R(t) : Functional HR repair complex (BRCA1-BRCA2-PALB2-RAD51)
    X(t) : Apoptotic commitment signal

Patient-specific parameters derived analytically from zero-damage steady state.
Global kinetic parameters are literature-fixed (not data-calibrated).
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Global literature-fixed parameters ---
GLOBAL_PARAMS = {}        


# --- Patient parameter computation ---
def compute_patient_params(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive patient-specific ODE parameters from log2(FPKM+1) expression values.

    Goal: turn each patient’s expression profile into a small set of ODE parameters, without 
    fitting survival data. The key idea is that baseline DNA damage is zero, so the only 
    patient-specific baseline state we need is the HR repair capacity R_ss

    Approach:
    1. Back-transform log2(FPKM+1) → FPKM (linear scale)
    2. Compute composite parameters from gene groups
    3. Normalise each parameter to population median so that the
        median patient has parameter value = 1.0
        (global rate constants are then calibrated for this reference patient)

    Returns a DataFrame with columns:
        PATIENT_ID, BRCA_cap, ATM_tot, CHK_tot, BCL2_ratio, BRCA_MUTANT
    plus the original OS_MONTHS and OS_EVENT for convenience.
    """

    # copy the merged DataFrame to avoid modifying the original
    df = merged_df.copy()
    
    # Identify the gene columns used for parameterisation 
    gene_cols = [
        'BRCA1', 'BRCA2', 'RAD51', 'PALB2', 'BRIP1',       # Homologous recombination repair
        'ATM', 'ATR',                                      # DNA damage sensing/checkpoint kinases
        'CHEK1', 'CHEK2',                                  # Checkpoint effectors/signalling
        'BCL2', 'BCL2L1', 'BAX', 'BAD'                     # Apoptotic threshold
    ]        
    # TP53 excluded — negative control, not used in ODE

    # Step 1: Back-transform each gene from log2(FPKM+1) to FPKM using the inverse transformation.
    # log2(FPKM + 1) → FPKM = 2^value - 1
    fpkm = 2 ** df[gene_cols] - 1
    fpkm = fpkm.clip(lower=0)   # guard against floating-point negatives

    # Step 2: Build composite pathway features
    #
    # BRCA_cap: HR repair capacity
    #   min(BRCA1, BRCA2) captures the bottleneck in complex assembly —
    #   both proteins are required and the lesser-expressed one limits throughput.
    #   RAD51 is the strand-invasion effector loaded by the BRCA1-BRCA2 scaffold.
    #   PALB2 bridges BRCA1 and BRCA2; included as square-root modifier.
    BRCA_cap = (
        np.minimum(fpkm['BRCA1'], fpkm['BRCA2'])
        * fpkm['RAD51']
        * np.sqrt(fpkm['PALB2'] + 1)   # +1 avoids sqrt(0)
    )
    # BRCA_cap = mean(BRCA1, BRCA2, PALB2, RAD51) 
    # BRCA_cap = geometric_mean(BRCA1, BRCA2, PALB2, RAD51)

    # ATM_tot: total checkpoint kinase abundance
    #   ATM senses DSBs; ATR senses replication stress and ssDNA.
    #   Both are activated by platinum-induced lesions.
    ATM_tot = (fpkm['ATM'] + fpkm['ATR']) / 2

    # CHK_tot: total checkpoint effector abundance
    #   CHEK1 and CHEK2 are the direct substrates of ATR and ATM respectively.
    CHK_tot = (fpkm['CHEK1'] + fpkm['CHEK2']) / 2

    # BCL2_ratio: apoptotic resistance
    #   Pro-survival (BCL2, BCL2L1) over pro-apoptotic (BAX, BAD).
    #   Higher ratio → cells more resistant to apoptosis → slower X decay.
    #   Small epsilon prevents division by zero.
    BCL2_ratio = (fpkm['BCL2'] + fpkm['BCL2L1']) / (fpkm['BAX'] + fpkm['BAD'] + 1e-6)

    # Step 3: Normalise to population median
    #   After normalisation, the median patient has each parameter = 1.0.
    #   Global rate constants (GLOBAL_PARAMS) are calibrated for this reference.
    params = pd.DataFrame({
        'PATIENT_ID':   df['PATIENT_ID'].values,
        'OS_MONTHS':    df['OS_MONTHS'].values,
        'OS_EVENT':     df['OS_EVENT'].values,
        'BRCA_MUTANT':  df['BRCA_MUTANT'].values,
        'BRCA_cap':     BRCA_cap.values,
        'ATM_tot':      ATM_tot.values,
        'CHK_tot':      CHK_tot.values,
        'BCL2_ratio':   BCL2_ratio.values,
    })

    params['BRCA_cap']   /= params['BRCA_cap'].median()
    params['ATM_tot']    /= params['ATM_tot'].median()
    params['CHK_tot']    /= params['CHK_tot'].median()
    params['BCL2_ratio'] /= params['BCL2_ratio'].median()

    return params          


# --- ODE right-hand side ---
def hrddr_ode(t, y, patient_params: dict, global_params: dict) -> list:
    pass   

# --- Single patient simulation ---
def simulate_patient(patient_params: dict, global_params: dict) -> dict:
    pass             

# --- ODE score extraction ---
def compute_ode_scores(sim_result: dict) -> dict:
    pass   

def main():
    pass

if __name__ == "__main__":
    main()