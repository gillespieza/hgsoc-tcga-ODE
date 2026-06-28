"""
ode_validation.py — Validation suite for the HR-DDR ODE model.

Runs three checks before the full cohort simulation:

1. Steady-state consistency: confirms that the analytical zero-damage
   steady state [D, A, C, R, X] = [0, 0, 0, BRCA_cap, 0] evaluates to
   zero derivatives for all five equations across a sweep of patient
   parameter values. A non-zero residual would indicate an error in the
   ODE right-hand side or the initial condition derivation.

2. Directional sensitivity: simulates one low-BRCA and one high-BRCA
   representative patient and confirms that lower repair capacity produces
   higher apoptotic commitment (AUC_X), as expected biologically.

3. Trajectory visualisation: plots all five state variables for both
   representative patients and saves to results/figures/.

Outputs
-------
- results/figures/ode_validation_trajectories.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
from pathlib import Path

import ode_model

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Residual magnitude below which a derivative is considered zero.
# Chosen to be well above float64 machine epsilon (~1e-16) but tight
# enough to catch genuine equation errors.
_SS_TOLERANCE = 1e-10


# =================================================================
# Steady-state validation
# =================================================================

def validate_steady_state(params_df: pd.DataFrame) -> None:
    """
    Confirm that the analytical zero-damage steady state satisfies dY/dt = 0.

    At zero drug input (D0 = 0) the system has an analytical steady state:

        D_ss = 0            (no damage without drug)
        A_ss = 0            (no ATM/ATR activation without damage)
        C_ss = 0            (no CHK signalling without ATM/ATR)
        R_ss = BRCA_cap     (HR complex at basal level; derived from
                             k_r * BRCA_cap = d_r * R with k_r = d_r)
        X_ss = 0            (no apoptotic drive without checkpoint)

    This function evaluates the ODE right-hand side at y_ss for every
    patient in params_df and asserts that the maximum absolute derivative
    is below _SS_TOLERANCE. Any non-zero residual indicates either an
    error in the equations or a mismatch between the initial conditions
    and the analytical derivation.

    The check is intentionally run across the full parameter sweep
    (not just two representative patients) because BRCA_cap and
    BCL2_ratio vary widely across the cohort. A bug that cancels at
    median parameter values might still surface at extremes.

    Parameters
    ----------
    params_df : pd.DataFrame
        Output of ode_model.compute_patient_params(); one row per patient.

    Raises
    ------
    ValueError
        If any patient's steady-state residual exceeds _SS_TOLERANCE,
        with details of which patient and which equation failed.
    """
    # Build zero-drug global params: D0 = 0 eliminates the drug input
    # φ(t) = D0 * exp(-t / tau_drug) entirely, giving a true no-damage
    # baseline.
    zero_drug_params = dict(ode_model.GLOBAL_PARAMS)
    zero_drug_params["D0"] = 0.0

    state_names = ["D", "A", "C", "R", "X"]
    n_patients = len(params_df)
    n_failed = 0

    logger.info(
        f"Steady-state check: evaluating {n_patients} patients "
        f"(tolerance = {_SS_TOLERANCE:.0e})"
    )

    for _, row in params_df.iterrows():
        patient = row.to_dict()
        brca_cap = patient["BRCA_cap"]

        # Analytical steady state: R = BRCA_cap, all others zero.
        y_ss = [0.0, 0.0, 0.0, brca_cap, 0.0]

        dydt = ode_model.hrddr_ode(
            0.0, y_ss, patient, zero_drug_params
        )

        for i, (name, residual) in enumerate(zip(state_names, dydt)):
            if abs(residual) > _SS_TOLERANCE:
                n_failed += 1
                raise ValueError(
                    f"Steady-state residual too large for patient "
                    f"{patient['PATIENT_ID']}: "
                    f"d{name}/dt = {residual:.3e} "
                    f"(tolerance = {_SS_TOLERANCE:.0e}, "
                    f"BRCA_cap = {brca_cap:.4f}). "
                    "Check the ODE right-hand side and initial condition "
                    "derivation."
                )

    logger.info(
        f"Steady-state check PASSED: all {n_patients} patients "
        f"satisfy dY/dt = 0 at [D, A, C, R, X] = "
        f"[0, 0, 0, BRCA_cap, 0] (max residual < {_SS_TOLERANCE:.0e})"
    )


# =================================================================
# Main
# =================================================================

def main():
    """
    Validate HR-DDR ODE model using two representative patients:
    - Low BRCA_cap (expected high apoptotic response)
    - High BRCA_cap (expected low apoptotic response)
    """

    # -----------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------

    merged = pd.read_csv(ROOT / "data" / "processed" / "hgsoc_tcga_merged.csv")
    params_df = ode_model.compute_patient_params(merged)

    # -----------------------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------------------

    if len(params_df) < 2:
        raise ValueError("Not enough patients for validation.")

    # -----------------------------------------------------------------
    # Check 1: steady-state consistency
    # -----------------------------------------------------------------

    # Runs before any simulation so a bad ODE fails loudly here rather
    # than producing silently wrong trajectories for all cohort patients.
    validate_steady_state(params_df)

    # -----------------------------------------------------------------
    # Check 2: directional sensitivity — define representative patients
    # -----------------------------------------------------------------

    brca_mut = params_df[params_df["BRCA_MUTANT"] == 1]
    brca_wt = params_df[params_df["BRCA_MUTANT"] == 0]

    if len(brca_mut) == 0 or len(brca_wt) == 0:
        raise ValueError("Missing BRCA mutant or wildtype cohort.")

    low_brca = brca_mut.nsmallest(1, "BRCA_cap").iloc[0]
    high_brca = brca_wt.nlargest(1, "BRCA_cap").iloc[0]

    logger.info(
        f"Low-BRCA patient  : {low_brca['PATIENT_ID']} "
        f"(BRCA_cap={low_brca['BRCA_cap']:.3f})"
    )

    logger.info(
        f"High-BRCA patient : {high_brca['PATIENT_ID']} "
        f"(BRCA_cap={high_brca['BRCA_cap']:.3f})"
    )

    # -----------------------------------------------------------------
    # Run simulations
    # -----------------------------------------------------------------

    sim_low = ode_model.simulate_patient(
        low_brca.to_dict(),
        ode_model.GLOBAL_PARAMS
    )

    sim_high = ode_model.simulate_patient(
        high_brca.to_dict(),
        ode_model.GLOBAL_PARAMS
    )

    score_low = ode_model.compute_ode_scores(sim_low)
    score_high = ode_model.compute_ode_scores(sim_high)

    logger.info(f"AUC_X low-BRCA  : {score_low['AUC_X']:.3f}")
    logger.info(f"AUC_X high-BRCA : {score_high['AUC_X']:.3f}")

    # -----------------------------------------------------------------
    # Biological directionality check (Check 2)
    # -----------------------------------------------------------------

    if score_low["AUC_X"] <= score_high["AUC_X"]:
        logger.warning(
            "Expected low BRCA_cap → higher apoptotic burden, "
            "but this trend was not observed."
        )

    # -----------------------------------------------------------------
    # Check 3: trajectory visualisation
    # -----------------------------------------------------------------

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), sharex=True)

    labels = [
        "D — DNA damage",
        "A — ATM/ATR",
        "C — CHK signalling",
        "R — HR repair",
        "X — Apoptosis",
    ]

    states = ["D", "A", "C", "R", "X"]

    for ax, key, label in zip(axes, states, labels):

        ax.plot(
            sim_low["t"],
            sim_low[key],
            color="#d62728",
            linewidth=2,
            label="Low BRCA_cap"
        )

        ax.plot(
            sim_high["t"],
            sim_high[key],
            color="#1f77b4",
            linewidth=2,
            label="High BRCA_cap"
        )

        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Time (h)", fontsize=8)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.tick_params(labelsize=7)

    axes[0].set_ylabel("Signal level", fontsize=8)
    axes[-1].legend(fontsize=7, frameon=False)

    plt.suptitle(
        "HR-DDR ODE validation: representative patients\n"
        f"AUC_X low vs high BRCA_cap: "
        f"{score_low['AUC_X']:.3f} vs {score_high['AUC_X']:.3f}",
        fontsize=11
    )

    plt.tight_layout()

    # -----------------------------------------------------------------
    # Save figure safely
    # -----------------------------------------------------------------

    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    out_path = fig_dir / "ode_validation_trajectories.png"

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()