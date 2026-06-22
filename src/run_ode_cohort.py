from pathlib import Path
import pandas as pd
import numpy as np
import sys

# Find the project root from this script's location.
ROOT = Path(__file__).resolve().parent.parent

# Add src/ so Python can import ode_model.py.
sys.path.insert(0, str(ROOT / "src"))

# Import the reusable ODE model module.
import ode_model


def main():
    # Load the merged dataset containing clinical data and gene expression.
    merged = pd.read_csv(ROOT / "data" / "processed" / "hgsoc_tcga_merged.csv")

    # Convert the merged table into patient-specific ODE parameters.
    params_df = ode_model.compute_patient_params(merged)

    # Store one result dictionary per patient.
    results = []

    # Loop over each patient row in the parameter table.
    for _, row in params_df.iterrows():
        # Convert the row into a normal Python dictionary for easier access.
        p = row.to_dict()

        # Simulate the ODE system for this patient.
        sim = ode_model.simulate_patient(p, ode_model.GLOBAL_PARAMS)

        # Extract summary scores from the full time-course output.
        scores = ode_model.compute_ode_scores(sim)

        # Add clinical metadata back onto the score dictionary.
        scores.update({
            "PATIENT_ID": p["PATIENT_ID"],
            "OS_MONTHS": p["OS_MONTHS"],
            "OS_EVENT": p["OS_EVENT"],
            "BRCA_MUTANT": p["BRCA_MUTANT"],
        })

        # Save this patient's result.
        results.append(scores)

    # Convert the list of dictionaries into a DataFrame.
    scores_df = pd.DataFrame(results)

    # Save raw ODE outputs.
    scores_df.to_csv(ROOT / "data" / "processed" / "ode_scores.csv", index=False)

    # Make a copy for survival analysis.
    analysis_df = scores_df.copy()

    # Add a log-transformed version of AUC_X.
    analysis_df["log_AUC_X"] = np.log1p(analysis_df["AUC_X"])

    # Save the survival-analysis-ready table.
    analysis_df.to_csv(ROOT / "data" / "processed" / "survival_analysis_df.csv", index=False)

    # Print simple sanity checks.
    # print(analysis_df.shape)
    # print(analysis_df["OS_EVENT"].mean())
    # print(analysis_df["OS_MONTHS"].median())


if __name__ == "__main__":
    main()