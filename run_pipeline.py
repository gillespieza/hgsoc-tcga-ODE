#!/usr/bin/env python
"""
Master pipeline script for HGSOC TCGA ODE survival modelling.

Purpose
-------
This script orchestrates the complete analysis workflow, starting from
raw clinical, mutation, and RNA-seq data and ending with survival,
machine-learning, and biomarker analyses.

The pipeline integrates:

- Clinical outcome data from TCGA ovarian cancer patients
- Tumour RNA-seq expression profiles
- BRCA1/2 mutation status
- A mechanistic HR-DDR ODE model
- Survival analysis methods
- Machine-learning survival models

The goal is to determine whether ODE-derived features of DNA damage
response dynamics are associated with overall survival and whether they
provide predictive value beyond standard molecular features.

Pipeline stages
---------------
1.  prepare_clinical.py          - clean clinical data
2.  prepare_RNA.py               - load RNA-seq data and keep ODE genes
3.  prepare_brca1_2_mutation.py  - build BRCA1/2 mutation labels
4.  merge_data.py                - merge clinical + expression data
5.  ode_validation.py            - validate ODE on representative patients
6.  run_ode_cohort.py            - simulate cohort and generate ODE scores
7.  analyse_ode_survival.py      - univariate Cox regression
8.  ml_benchmark.py              - Cox LASSO + RSF benchmark
9.  feature_importance.py        - model interpretation
10. kaplan_meier.py              - KM stratification and log-rank test
"""

import sys
import time
import logging
import os
from pathlib import Path

# ---------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------

# Several scientific Python libraries may use NumExpr internally
# for accelerated numerical computations.
#
# Explicitly limiting the thread count avoids occasional performance
# inconsistencies and warning messages on Windows systems.
os.environ["NUMEXPR_MAX_THREADS"] = "16"

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

# This file lives in the project root.
#
# Example:
#
# project_root/
# ├── run_pipeline.py
# ├── src/
# ├── data/
# └── results/
#
# ROOT is used throughout the project to build paths relative to the
# repository rather than the current working directory.
ROOT = Path(__file__).resolve().parent

# Python searches for modules in directories listed in sys.path.
#
# Adding src/ allows project modules to be imported directly:
#
#     import prepare_clinical
#
# without requiring installation as a formal Python package.
#
# This is a common pattern in research code where simplicity and
# transparency are often preferred over packaging infrastructure.
sys.path.insert(0, str(ROOT / "src"))

# ---------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------

log_file = ROOT / "pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Terminal colours
# ---------------------------------------------------------------------

RED = "\u001b[31m"
GREEN = "\u001b[32m"
YELLOW = "\u001b[33m"
RESET = "\u001b[0m"
CYAN = "\u001b[36m"


def success_msg(script_name: str, elapsed: float):
    return f"[OK] {script_name} completed ({elapsed:.2f}s)\n"


def fail_msg(script_name: str, elapsed: float):
    return f"[FAIL] {script_name} failed after {elapsed:.2f}s\n"


def run_script(script_name: str):
    """
    Import a module from src/ and execute its main() function.

    Most pipeline stages expose a main() entry point. This allows
    individual scripts to be run independently while also supporting
    orchestration from this master pipeline.

    Dynamic imports are used so that pipeline stages can be listed
    as strings and executed sequentially without hard-coded imports.
    """

    logger.info(f"{RESET}{'=' * 60}{RESET}")
    logger.info(f"{YELLOW}Running: {script_name}{RESET}")
    logger.info(f"{RESET}{'=' * 60}{RESET}")

    start = time.time()

    try:
        # Import module by name at runtime.
        #
        # Example:
        #     "prepare_clinical"
        #
        # becomes:
        #     import prepare_clinical
        module = __import__(script_name)

        # Execute the module's main() function when available.
        # This is preferred over relying on top-level code execution.
        if hasattr(module, "main"):
            module.main()
        else:
            __import__(script_name)

        elapsed = time.time() - start

        logger.info(f"{GREEN}{success_msg(script_name, elapsed)}{RESET}")

        return elapsed

    except Exception:
        elapsed = time.time() - start

        logger.exception(
            f"{script_name} failed after {elapsed:.2f}s"
        )

        raise


def main():
    """
    Run the complete preprocessing, modelling, and analysis workflow.
    """

    logger.info(f"{CYAN}{'=' * 60}{RESET}")
    logger.info(f"{CYAN}HGSOC TCGA ODE Survival Pipeline{RESET}")
    logger.info(f"{CYAN}{'=' * 60}{RESET}")

    total_start = time.time()

    # Pipeline order is critical because each stage produces outputs
    # consumed by downstream analyses.
    #
    # Data flow:
    #
    # Clinical data
    #      │
    # RNA-seq data
    #      │
    # BRCA mutation labels
    #      ▼
    # Integrated patient table
    #      ▼
    # Patient-specific ODE simulations
    #      ▼
    # ODE-derived biomarkers
    #      ▼
    # Survival and machine-learning analyses
    steps = [
        "prepare_clinical",
        "prepare_RNA",
        "prepare_brca1_2_mutation",
        "merge_data",

        # Imported by downstream scripts rather than run directly.
        # "ode_model",

        # Validate model behaviour on representative patients before
        # running the full cohort simulation.
        "ode_validation",

        # Simulate the entire cohort and generate ODE-derived features.
        "run_ode_cohort",

        # Test associations between ODE scores and survival outcomes
        # using univariate Cox proportional hazards models.
        "analyse_ode_survival",

        # Compare predictive performance of:
        # - Cox proportional hazards with LASSO regularisation
        # - Random Survival Forests (RSF)
        #
        # Bootstrap confidence intervals quantify uncertainty in
        # model performance estimates.
        "ml_benchmark",

        # Interpret model behaviour.
        #
        # Cox model:
        #     coefficient magnitude
        #
        # RSF:
        #     permutation importance
        "feature_importance",

        # Visualise survival differences between groups defined by
        # an ODE-derived biomarker threshold.
        #
        # Provides an interpretable complement to continuous hazard
        # ratios estimated by Cox regression.
        "kaplan_meier",
    ]

    for step in steps:
        run_script(step)

    total_elapsed = time.time() - total_start

    logger.info(f"{GREEN}{'=' * 60}{RESET}")
    logger.info(
        f"{GREEN}Pipeline complete ({total_elapsed:.2f}s total){RESET}"
    )
    logger.info(f"{GREEN}{'=' * 60}{RESET}")


if __name__ == "__main__":
    main()