"""
ode_model.py — HR-DDR ODE model for HGSOC patient-specific survival prediction.

Mechanistic structure:
    D(t) : DNA damage load
    A(t) : ATM/ATR checkpoint activation
    C(t) : CHK1/CHK2 signalling
    R(t) : HR repair capacity (BRCA axis)
    X(t) : apoptotic commitment

Key idea:
- Global kinetics are fixed (literature-informed)
- Patient variation enters via RNA-derived pathway compression
- Survival signal derived from apoptotic burden (AUC_X)
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ================================================================
# GLOBAL PARAMETERS (fixed kinetics)
# ================================================================
# Global kinetic parameters cannot be derived analytically and must be estimated.
# These are the same for every patient.
# Patient-to-patient differences come from expression-derived parameters,
# not from re-fitting these kinetic constants.
GLOBAL_PARAMS = {
    # Drug perturbation input: φ(t) = D0 * exp(-t / tau_drug)
    # Models carboplatin delivery and clearance from tumour tissue.
    # Carboplatin plasma half-life ~2-6h; tau_drug set to 6h.
    'D0':        1.0,    # Initial damage input magnitude (arbitrary normalized units)
    'tau_drug':  6.0,    # h — drug decay time constant

    # ATM/ATR activation and decay.
    # ATM activates within minutes; full activation plateau ~1-2h; 
    # deactivation by PP2A/WIP1 over ~2h after damage resolution.
    'k_a':  1.0,         # h⁻¹ — Rate at which damage activates ATM/ATR
    'd_a':  0.5,         # h⁻¹ — Deactivation rate of ATM/ATR (half-life ~2h)


    # CHK1/CHK2 activation and decay.
    # CHK2 phosphorylated by ATM within ~30 min; sustained for several hours.
    # CHK1 activated by ATR; deactivated by PP2A over ~3-5h.
    'k_c':  0.5,         # h⁻¹ — Rate at which ATM/ATR activates CHK1/CHK2
    'd_c':  0.2,         # h⁻¹ — Deactivation rate of CHK1/CHK2 (half-life ~5h)

    # DNA repair kinetics
    # HR repair of DSBs: 4-24h depending on complexity. HR depends on repair complex R, while NHEJ is modeled as always available
    # NHEJ is faster (~1-4h) but error-prone; operates independently of BRCA capacity.
    'k_HR':    0.15,     # h⁻¹ — HR repair rate per unit R (repair complete in ~6-20h)
    'k_NHEJ':  0.30,     # h⁻¹ — Baseline NHEJ repair rate (faster, R-independent)

    # HR complex dynamics
    # k_load: CHK2 phosphorylates BRCA1 to facilitate HR loading, but sustained
    # checkpoint activity eventually depletes the available complex (exhaustion term).
    # d_r: constitutive HR complex turnover.
    'k_load':  0.20,     # h⁻¹ — CHK-driven depletion/exhaustion of HR repair complex
    'd_r':     0.10,     # h⁻¹ — Basal turnover of HR repair complex

    # Apoptotic signal kinetics
    # X represents commitment to apoptosis (upstream of caspase activation).
    # k_x: rate at which sustained CHK activity, unchecked by HR, drives apoptosis.
    # d_x: base apoptotic signal decay (intrinsic anti-apoptotic buffering).
    # BCL2_ratio modulates d_x per patient (higher BCL2 → faster signal clearance).
    'k_x':  0.10,        # h⁻¹ — Rate at which checkpoint activity drives apoptosis
    'd_x':  0.02,        # h⁻¹ — Basal decay of apoptotic signal (half-life ~35h)

    'k_suppress': 0.3,   # h⁻¹ — HR-mediated suppression of apoptotic signal
    'K_hr': 1.0         # Half-saturation constant for HR suppression

}        


# ================================================================
# PATIENT PARAMETER CONSTRUCTION
# ================================================================
def compute_patient_params(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive patient-specific ODE parameters from log2(FPKM+1) expression values.

    Goal: turn each patient's RNA expression profile into a small set of ODE parameters, without 
    fitting survival data. The key idea is that baseline DNA damage is zero, so the only 
    patient-specific baseline state we need is the HR repair capacity R_ss

    Main idea:
    - The RNA data are stored as log2(FPKM + 1).
    - We convert them back to linear FPKM scale.
    - We combine related genes into pathway-level composite quantities.
    - We normalize each composite to the cohort median so the "typical"
      patient has value 1.0.

    Returns a DataFrame with columns:
        PATIENT_ID, OS_MONTHS, OS_EVENT, BRCA_MUTANT,
        BRCA_cap, ATM_tot, CHK_tot, BCL2_ratio
    """

    # Copy to avoid modifying the input DataFrame in place.
    df = merged_df.copy()
    
    # Gene columns actually used for parameterization.
    # TP53 is intentionally excluded because it is a negative control here.
    gene_cols = [
        'BRCA1', 'BRCA2', 'RAD51', 'PALB2', 'BRIP1',       # Homologous recombination repair
        'ATM', 'ATR',                                      # DNA damage sensing/checkpoint kinases
        'CHEK1', 'CHEK2',                                  # Checkpoint effectors/signalling
        'BCL2', 'BCL2L1', 'BAX', 'BAD'                     # Apoptotic threshold
    ]        

    # ------------------------------------------------------------
    # Safety check: ensure required genes exist
    # ------------------------------------------------------------
    missing_genes = [g for g in gene_cols if g not in df.columns]
    if missing_genes:
        raise ValueError(f"Missing gene columns in merged dataset: {missing_genes}")

    # ------------------------------------------------------------
    # Handle numerical stability issues
    # ------------------------------------------------------------
    expr = df[gene_cols].replace([np.inf, -np.inf], np.nan)

    if expr.isna().any().any():
        raise ValueError(
            "NaNs detected in expression matrix after merge. "
            "Check RNA preprocessing."
        )

    # ------------------------------------------------------------
    # Back-transform log2(FPKM+1)
    # ------------------------------------------------------------
    fpkm = 2 ** expr - 1

    # Guard against tiny negative values caused by floating-point precision.
    fpkm = fpkm.clip(lower=0)

    # ------------------------------------------------------------
    # Pathway compression (phenomenological, not mechanistic kinetics)
    # ------------------------------------------------------------

    # BRCA_cap: homologous recombination repair capacity.
    #
    # Why use min(BRCA1, BRCA2)?
    # Because both proteins are needed, and the lower one can act like a bottleneck/rate-limiting step.
    #
    # Why multiply by RAD51?
    # RAD51 is the strand-invasion effector loaded by the BRCA1-BRCA2 scaffold in HR repair.
    #
    # Why sqrt(PALB2 + 1)?
    # PALB2 supports BRCA1/2 function, but we include it as a softer modifier
    # rather than letting it dominate multiplicatively.
    #
    # np.log1p(...) means log(1 + value), which compresses very large values.

    BRCA_cap = np.log1p(
        np.minimum(fpkm['BRCA1'], fpkm['BRCA2'])
        * fpkm['RAD51']
        * np.sqrt(fpkm['PALB2'] + 1)
    )

    # ATM_tot: total upstream damage-sensing capacity.
    #   ATM senses DSBs; ATR senses replication stress and ssDNA.
    #   Both are activated by platinum-induced lesions.
    ATM_tot = (fpkm['ATM'] + fpkm['ATR']) / 2

    # CHK_tot: total downstream checkpoint effector capacity.
    #   CHEK1 and CHEK2 are the direct substrates of ATR and ATM respectively.
    CHK_tot = (fpkm['CHEK1'] + fpkm['CHEK2']) / 2

    # BCL2_ratio: balance of anti-apoptotic vs pro-apoptotic expression (i.e. apoptotic resistance).
    #    Pro-survival (BCL2, BCL2L1) over pro-apoptotic (BAX, BAD).
    # Small epsilon avoids division by zero.
    BCL2_ratio = (
        (fpkm["BCL2"] + fpkm["BCL2L1"]) /
        (fpkm["BAX"] + fpkm["BAD"] + 1e-6)
    )

    # Assemble the patient parameter table.
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

    # Normalize each parameter to the cohort median.
    # After this step, a median patient has parameter value ~1.0.
    for col in ["BRCA_cap", "ATM_tot", "CHK_tot", "BCL2_ratio"]:
        params[col] /= params[col].median()

    return params          



# ================================================================
# ODE right-hand side
# ================================================================
def hrddr_ode(t: float, y: list, patient_params: dict, global_params: dict) -> list:
    """
    Right-hand side of the HR-DDR ODE system. Compute the derivatives of the five ODE state variables at time t.

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

    # Unpack the state vector.
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

    # Drug-driven damage input.
    # At t = 0, phi is largest; then it decays exponentially.
    phi = D0 * np.exp(-t / tau_drug)

    # Numerical safety:
    # if the solver ever produces a tiny negative number, clamp it to zero so it doesn't explode
    D = max(D, 0.0)
    A = max(A, 0.0)
    C = max(C, 0.0)
    R = max(R, 0.0)
    X = max(X, 0.0)

    # dD/dt: DNA damage dynamics: input_damage - repair_by_R
    # Damage increases because of the drug input phi.
    # Damage decreases through HR repair (depends on R) and NHEJ repair.
    dD = phi - (k_HR * R + k_NHEJ) * D

    # dA/dt: ATM/ATR checkpoint kinase activation dynamics: activation_by_D - decay
    # More damage and more total ATM/ATR abundance lead to more activation.
    # Deactivated by phosphatases (PP2A, WIP1) at rate d_a.
    # `ATM_tot` is a patient-specific constant derived from mRNA
    dA = k_a * ATM_tot * D - d_a * A

    # dC/dt: CHK1/CHK2 checkpoint effector activation: activation_by_A - decay
    # Activated by ATM/ATR signal A, scaled by effector abundance CHK_tot.
    # Deactivated at rate d_c (PP2A-mediated dephosphorylation).
    # `CHK_tot` is a patient-specific constant derived from mRNA
    dC = k_c * A * CHK_tot - d_c * C


    # dR/dt: HR repair complex dynamics: synthesis_from_BRCA_cap - basal_decay - damage_consumption
    # We choose k_r = d_r so that, at zero damage steady state:
    # dR/dt = d_r * BRCA_cap - d_r * R = 0  ->  R_ss = BRCA_cap
    # Depleted by sustained checkpoint activity (k_load * C * R): CHK2-mediated
    # BRCA1 hyperphosphorylation eventually exhausts the repair complex.
    # Basal degradation at rate d_r.
    # `BRCA_cap` is a patient-specific constant derived from mRNA
    k_r = d_r   # ensures R_ss = BRCA_cap analytically at D=0
    dR = k_r * BRCA_cap - k_load * C * R - d_r * R

    # dX/dt: apoptotic commitment signal.
    #
    # X is increased by checkpoint stress (k_x * C).
    # X is decreased by:
    # 1. intrinsic buffering scaled by BCL2_ratio
    # 2. suppression from HR capacity R
    dX = (
        k_x * C
        - d_x * BCL2_ratio * X
        - k_suppress * R * X / (K_hr + R)
    )

    return [dD, dA, dC, dR, dX]

# ================================================================
# SINGLE PATIENT SIMULATION
# ================================================================
def simulate_patient(patient_params: dict, global_params: dict) -> dict:
    """
    Simulate the HR-DDR ODE for a single patient over 120 hours.
        - set initial conditions from zero-damage steady state
        - apply platinum perturbation input φ(t) = D0 * exp(-t / tau_drug)
        - integrate using `scipy.integrate.solve_ivp` with `method='RK45'
        - extract ODE summary scores for survival prediction

    Initial conditions derived from zero-damage analytical steady state:
        D(0) = 0         (no damage before drug)
        A(0) = 0         (no ATM/ATR activation: activation requires damage signal)
        C(0) = 0         (no CHK activation: activation requires ATM/ATR signal)
        R(0) = BRCA_cap  (HR complex at steady-state level — analytical result)
        X(0) = 0         (no apoptotic drive without checkpoint activation)

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

    # Simulate for 5 days, sampled every 0.5 hours.
    t_span = (0.0, 120.0)              # 0 to 120 hours (5 days)
    t_eval = np.linspace(0, 120, 241)  # every 0.5h

    # integrate using `scipy.integrate.solve_ivp` with `method='RK45' or LSODA
    sol = solve_ivp(
        hrddr_ode,
        t_span,
        y0,
        t_eval=t_eval,
        args=(patient_params, global_params),

        # method='RK45',
        method='LSODA', # RECOMMENDED solver for mixed-timescale systems

        rtol=1e-6,
        atol=1e-9,
        dense_output=False,
    )

    # If integration fails, return NaN arrays so downstream code can detect failure.
    if not sol.success:
        nan_arr = np.full(len(t_eval), np.nan)
        return {
            "PATIENT_ID": patient_params["PATIENT_ID"],
            "t": t_eval,
            "D": nan_arr,
            "A": nan_arr,
            "C": nan_arr,
            "R": nan_arr,
            "X": nan_arr,
            "success": False,
        }

    # If integration succeeds, return the time grid and all state trajectories.
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

# ================================================================
# ODE SCORING
# ================================================================
def compute_ode_scores(sim_result: dict) -> dict:
    """
    Reduce a full time-course simulation to a few summary numbers.

    Main score:
        AUC_X = area under the apoptotic signal curve over time

    Higher AUC_X → more apoptotic commitment → tumour more platinum-sensitive
    → expected better patient survival.

    Secondary scores retained for sensitivity analysis.
        X_peak   = maximum apoptotic signal
        T_repair = duration R stays significantly depleted below R_ss
        D_resid  = residual damage at final time

    T_repair definition
    -------------------
    R_ss = BRCA_cap is the analytical zero-damage steady state for R (derived
    in hrddr_ode: k_r = d_r, so dR/dt = 0 gives R_ss = BRCA_cap).

    T_repair is the last time point at which R has not yet recovered to within
    10% of that baseline:

        T_repair = max { t : R(t) < 0.9 * BRCA_cap }

    This is patient-specific because BRCA_cap varies per patient, and the
    depletion trajectory is driven by k_load * C * R, which couples ATM_tot
    and CHK_tot. A value of 0.0 means R was never meaningfully perturbed
    (high BRCA_cap, weak checkpoint activation).

    The previous definition — time D stays above 10% of D_peak — was almost
    entirely governed by the global tau_drug constant and carried negligible
    patient-specific information.
    """
    from scipy.integrate import trapezoid

    t = sim_result['t']
    X = sim_result['X']
    D = sim_result['D']
    R = sim_result['R']

    # BRCA_cap is the analytical R steady state at zero damage.
    # sim_result carries the patient's R trajectory; R[0] = BRCA_cap
    # by construction (initial condition set in simulate_patient).
    R_ss = float(R[0])

    # AUC of apoptosis signal: main model output and primary survival predictor.
    AUC_X = trapezoid(X, t)

    # Peak apoptosis signal.
    X_peak = float(np.max(X))

    # Residual damage at the end of the simulation.
    D_resid = float(D[-1])

    # T_repair: duration the HR complex remains significantly depleted.
    #
    # Threshold at 0.9 * R_ss: a 10% depletion from baseline is the smallest
    # perturbation considered biologically meaningful. Tighter thresholds
    # (e.g. 0.5) only fire for very low BRCA_cap patients; looser ones
    # (e.g. 0.95) risk firing from numerical noise.
    #
    # Fallback 0.0: if R never dips below the threshold the checkpoint never
    # meaningfully impaired the HR complex — a valid and informative result.
    depletion_thresh = 0.9 * R_ss
    depleted = np.where(R < depletion_thresh)[0]
    T_repair = float(t[depleted[-1]]) if len(depleted) > 0 else 0.0

    return {
        'PATIENT_ID': sim_result['PATIENT_ID'],
        'AUC_X':      float(AUC_X),
        'X_peak':     X_peak,
        'T_repair':   T_repair,
        'D_resid':    D_resid,
        'success':    sim_result['success'],
    }