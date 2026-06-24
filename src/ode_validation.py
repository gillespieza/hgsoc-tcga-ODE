import pandas as pd
import numpy as np
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
    # Define representative patients
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
    # Biological directionality check
    # -----------------------------------------------------------------

    if score_low["AUC_X"] <= score_high["AUC_X"]:
        logger.warning(
            "Expected low BRCA_cap → higher apoptotic burden, "
            "but this trend was not observed."
        )

    # -----------------------------------------------------------------
    # Plot trajectories
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

    logger.success(f"[FILE] Saved figure: ./results/figures/ode_validation_trajectories.png")


if __name__ == "__main__":
    main()