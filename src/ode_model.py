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
GLOBAL_PARAMS = {
    # Drug perturbation input: φ(t) = D0 * exp(-t / tau_drug)
    # Models carboplatin delivery and clearance from tumour tissue.
    # Carboplatin plasma half-life ~2-6h; tau_drug set to 6h.
    'D0':        1.0,    # normalised initial damage magnitude (arbitrary units)
    'tau_drug':  6.0,    # h — drug decay time constant

    # ATM/ATR activation 
    # ATM activates within minutes; full activation plateau ~1-2h; 
    # deactivation by PP2A/WIP1 over ~2h after damage resolution.
    'k_a':  1.0,         # h⁻¹ — ATM/ATR activation rate per unit D and ATM_tot
    'd_a':  0.5,         # h⁻¹ — ATM/ATR deactivation rate (half-life ~2h)


    # CHK1/CHK2 activation
    # CHK2 phosphorylated by ATM within ~30 min; sustained for several hours.
    # CHK1 activated by ATR; deactivated by PP2A over ~3-5h.
    'k_c':  0.5,         # h⁻¹ — CHK activation rate per unit A and CHK_tot
    'd_c':  0.2,         # h⁻¹ — CHK deactivation rate (half-life ~5h)

    # DNA repair kinetics
    # HR repair of DSBs: 4-24h depending on complexity
    # NHEJ is faster (~1-4h) but error-prone; operates independently of BRCA capacity.
    'k_HR':    0.15,     # h⁻¹ — HR repair rate per unit R (repair complete in ~6-20h)
    'k_NHEJ':  0.30,     # h⁻¹ — NHEJ repair rate (faster, R-independent)

    # HR complex dynamics
    # k_load: CHK2 phosphorylates BRCA1 to facilitate HR loading, but sustained
    # checkpoint activity eventually depletes the available complex (exhaustion term).
    # d_r: constitutive HR complex turnover.
    'k_load':  0.20,     # h⁻¹ — CHK-driven HR complex depletion rate
    'd_r':     0.10,     # h⁻¹ — basal HR complex degradation rate

    # Apoptotic signal kinetics
    # X represents commitment to apoptosis (upstream of caspase activation).
    # k_x: rate at which sustained CHK activity, unchecked by HR, drives apoptosis.
    # d_x: base apoptotic signal decay (intrinsic anti-apoptotic buffering).
    # BCL2_ratio modulates d_x per patient (higher BCL2 → faster signal clearance).
    'k_x':  0.10,        # h⁻¹ — apoptotic signal generation rate per unit C
    'd_x':  0.02,        # h⁻¹ — base apoptotic signal decay rate (half-life ~35h)

    'k_suppress': 0.3,   # h⁻¹ — HR complex-mediated apoptotic signal degradation rate
    'K_hr': 1.0,         # # half-saturation constant

    'alpha': 0.1,
    'c_floor': 0.01,
    'x_floor': 0.001 
}        


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
    BRCA_cap = np.log1p(
        np.minimum(fpkm['BRCA1'], fpkm['BRCA2'])
        * fpkm['RAD51']
        * np.sqrt(fpkm['PALB2'] + 1)
    )
    # BRCA_cap = np.mean([fpkm['BRCA1'], fpkm['BRCA2'], fpkm['PALB2'], fpkm['RAD51']]) 
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
def hrddr_ode(t: float, y: list, patient_params: dict, global_params: dict) -> list:
    """
    Right-hand side of the HR-DDR ODE system.

    State vector y = [D, A, C, R, X]

    Parameters
    ----------
    t : float
        Current time (hours)
    y : list
        Current state [D, A, C, R, X]
    patient_params : dict
        Patient-specific parameters: BRCA_cap, ATM_tot, CHK_tot, BCL2_ratio
    global_params : dict
        Literature-fixed kinetic parameters (GLOBAL_PARAMS)

    Returns
    -------
    list of floats: [dD/dt, dA/dt, dC/dt, dR/dt, dX/dt]
    """
    D, A, C, R, X = y

    # Unpack patient-specific parameters
    BRCA_cap  = patient_params['BRCA_cap']
    ATM_tot   = patient_params['ATM_tot']
    CHK_tot   = patient_params['CHK_tot']
    BCL2_ratio = patient_params['BCL2_ratio']

    # Unpack global parameters
    D0       = global_params['D0']
    tau_drug = global_params['tau_drug']
    k_a      = global_params['k_a']
    d_a      = global_params['d_a']
    k_c      = global_params['k_c']
    d_c      = global_params['d_c']
    k_HR     = global_params['k_HR']
    k_NHEJ   = global_params['k_NHEJ']
    k_load   = global_params['k_load']
    d_r      = global_params['d_r']
    k_x      = global_params['k_x']
    d_x      = global_params['d_x']
    k_suppress = global_params['k_suppress']
    K_hr = global_params['K_hr'] 

    # Drug perturbation input: exponentially decaying damage bolus
    # φ(t) represents the rate of new DSB formation from circulating platinum
    phi = D0 * np.exp(-t / tau_drug)

    # Clamp state variables to non-negative (numerical safeguard)
    D = max(D, 0.0)
    A = max(A, 0.0)
    C = max(C, 0.0)
    R = max(R, 0.0)
    X = max(X, 0.0)

    # dD/dt: DNA damage dynamics: input_damage - repair_by_R
    # New damage arrives from drug (phi), repaired by HR (rate k_HR * R, proportional
    # to available HR complex) and NHEJ (rate k_NHEJ, R-independent).
    dD = phi - (k_HR * R + k_NHEJ) * D

    # dA/dt: ATM/ATR checkpoint kinase activation: activation_by_D - decay
    # Activated proportionally to damage D and total kinase abundance ATM_tot.
    # Deactivated by phosphatases (PP2A, WIP1) at rate d_a.
    dA = k_a * ATM_tot * D - d_a * A

    # dC/dt: CHK1/CHK2 checkpoint effector activation: activation_by_A - decay
    # Activated by ATM/ATR signal A, scaled by effector abundance CHK_tot.
    # Deactivated at rate d_c (PP2A-mediated dephosphorylation).
    dC = k_c * A * CHK_tot - d_c * C

    # dR/dt: HR repair complex dynamics: synthesis_from_BRCA_cap - basal_decay - damage_consumption
    # Constitutive loading at rate k_r * BRCA_cap (patient-specific synthesis).
    # Note: k_r is absorbed into BRCA_cap normalisation (k_r = d_r at steady state),
    # so R_ss = BRCA_cap after normalisation — the analytical zero-damage result.
    # Depleted by sustained checkpoint activity (k_load * C * R): CHK2-mediated
    # BRCA1 hyperphosphorylation eventually exhausts the repair complex.
    # Basal degradation at rate d_r.
    k_r = d_r   # ensures R_ss = BRCA_cap analytically at D=0
    dR = k_r * BRCA_cap - k_load * C * R - d_r * R

    # dX/dt: Apoptotic commitment signal: activation_by_C and BCL2_ratio - decay
    HR_suppression = 1.0 / (1.0 + R / K_hr)
    # dX = k_x * C * HR_suppression - d_x * BCL2_ratio * X
    dX = ( k_x*C - d_x*BCL2_ratio*X - k_suppress*R*X/(K_hr+R) )

    return [dD, dA, dC, dR, dX]

# --- Single patient simulation ---
def simulate_patient(patient_params: dict, global_params: dict) -> dict:
    """
    Simulate the HR-DDR ODE for a single patient.

    Initial conditions derived from zero-damage analytical steady state:
        D(0) = 0       (no damage before drug)
        A(0) = 0       (no ATM/ATR activation)
        C(0) = 0       (no CHK activation)
        R(0) = BRCA_cap  (HR complex at steady-state level — analytical result)
        X(0) = 0       (no apoptotic signal)

    Parameters
    ----------
    patient_params : dict  — BRCA_cap, ATM_tot, CHK_tot, BCL2_ratio, PATIENT_ID
    global_params  : dict  — GLOBAL_PARAMS

    Returns
    -------
    dict with keys: t, D, A, C, R, X, PATIENT_ID
    """
    # Initial conditions from zero-damage steady state
    R_ss = patient_params['BRCA_cap']   # analytical result: R_ss = BRCA_cap after normalisation
    y0 = [0.0, 0.0, 0.0, R_ss, 0.0]   # [D, A, C, R, X]

    t_span = (0.0, 120.0)              # 0 to 120 hours (5 days)
    t_eval = np.linspace(0, 120, 241)  # every 0.5h

    sol = solve_ivp(
        fun=hrddr_ode,
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        args=(patient_params, global_params),
        method='RK45',
        rtol=1e-6,
        atol=1e-9,
        dense_output=False,
    )

    if not sol.success:
        # Return NaN arrays if integration failed — flagged during score computation
        nan_arr = np.full(len(t_eval), np.nan)
        return {
            'PATIENT_ID': patient_params['PATIENT_ID'],
            't': t_eval, 'D': nan_arr, 'A': nan_arr,
            'C': nan_arr, 'R': nan_arr, 'X': nan_arr,
            'success': False
        }

    return {
        'PATIENT_ID': patient_params['PATIENT_ID'],
        't': sol.t,
        'D': sol.y[0],
        'A': sol.y[1],
        'C': sol.y[2],
        'R': sol.y[3],
        'X': sol.y[4],
        'success': True,
    }       

# --- ODE score extraction ---
def compute_ode_scores(sim_result: dict) -> dict:
    """
    Extract summary statistics from a single patient ODE simulation.

    Primary score: AUC_X — integral of apoptotic signal over 120h.
    Higher AUC_X → more apoptotic commitment → tumour more platinum-sensitive
    → expected better patient survival.

    Secondary scores retained for sensitivity analysis.
    """
    from scipy.integrate import trapezoid

    t = sim_result['t']
    X = sim_result['X']
    D = sim_result['D']

    AUC_X   = trapezoid(X, t)           # primary survival predictor
    X_peak  = float(np.max(X))          # peak apoptotic signal
    D_resid = float(D[-1])              # residual damage at t=120h

    # Time for D to fall to 10% of its peak value (repair speed proxy)
    D_peak  = float(np.max(D))
    thresh  = 0.1 * D_peak
    above   = np.where(D > thresh)[0]
    T_repair = float(t[above[-1]]) if len(above) > 0 else 120.0

    return {
        'PATIENT_ID': sim_result['PATIENT_ID'],
        'AUC_X':      float(AUC_X),
        'X_peak':     X_peak,
        'T_repair':   T_repair,
        'D_resid':    D_resid,
        'success':    sim_result['success'],
    }

def main():
    pass

if __name__ == "__main__":
    main()