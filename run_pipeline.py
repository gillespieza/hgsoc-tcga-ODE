#!/usr/bin/env python
"""
Master pipeline script for HGSOC TCGA ODE survival modeling.

This file runs the full preprocessing/model pipeline in order:
1.  prepare_clinical.py          - clean clinical data
2.  prepare_RNA.py               - load RNA-seq data and keep ODE genes
3.  prepare_brca1_2_mutation.py  - build BRCA1/2 mutation labels
4.  merge_data.py                - merge clinical + expression data
5.  ode_validation.py            - validate ODE on two representative patients
6.  run_ode_cohort.py            - simulate cohort and save ODE-derived scores
7.  analyse_ode_survival.py      - univariate Cox regression on ODE scores
8.  ml_benchmark.py              - Cox LASSO + RSF benchmark with bootstrap CIs
9.  feature_importance.py        - Cox LASSO coefficients + RSF permutation importances
10. kaplan_meier.py              - KM stratification by AUC_X, log-rank test
"""

import sys
import time
import logging
import os
from pathlib import Path

# Limit NUMEXPR threads to avoid performance or compatibility issues on Windows.
os.environ["NUMEXPR_MAX_THREADS"] = "16"

# Set the root path of the project.
ROOT = Path(__file__).resolve().parent

# Add src/ to path so we can import scripts from there.
sys.path.insert(0, str(ROOT / "src"))

# Configure logging with ASCII-only output. Log both to a file and to the console.
log_file = ROOT / "pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ANSI color codes for terminal output.
RED    = "\u001b[31m"
GREEN  = "\u001b[32m"
YELLOW = "\u001b[33m"
RESET  = "\u001b[0m"
CYAN   = "\u001b[36m"


def success_msg(script_name: str, elapsed: float):
    return f"[OK] {script_name} completed ({elapsed:.2f}s)\n"


def fail_msg(script_name: str, elapsed: float):
    return f"[FAIL] {script_name} failed after {elapsed:.2f}s\n"


def run_script(script_name: str):
    """
    Import a script from src/ and run its main() function if it exists.
    Falls back to top-level execution if no main() is defined.
    """
    logger.info(f"{RESET}{'='*60}{RESET}")
    logger.info(f"{YELLOW}Running: {script_name}{RESET}")
    logger.info(f"{RESET}{'='*60}{RESET}")

    start = time.time()

    try:
        module = __import__(script_name)

        if hasattr(module, "main"):
            module.main()
        else:
            __import__(script_name)

        elapsed = time.time() - start
        logger.info(f"{GREEN}{success_msg(script_name, elapsed)}{RESET}")
        return elapsed

    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"{RED}[FAIL]{RESET} {script_name} failed after {elapsed:.2f}s")
        logger.error(f"{RED}Error: {e}{RESET}")
        raise


def main():
    """
    Run all preprocessing, modelling, and analysis steps in order.
    """
    logger.info(f"{CYAN}{'='*60}{RESET}")
    logger.info(f"{CYAN}HGSOC TCGA ODE Survival Pipeline{RESET}")
    logger.info(f"{CYAN}{'='*60}{RESET}")

    total_start = time.time()

    # Steps must run in this order — later steps depend on earlier outputs.
    steps = [
        "prepare_clinical",
        "prepare_RNA",
        "prepare_brca1_2_mutation",
        "merge_data",
        #"ode_model",          # skipped — imported by downstream scripts
        "ode_validation",
        "run_ode_cohort",
        "analyse_ode_survival",
        "ml_benchmark",         # Cox LASSO + RSF + bootstrap CIs
        "feature_importance",   # Cox LASSO coefficients + RSF permutation importances
        "kaplan_meier",         # KM curves + log-rank test
    ]

    for step in steps:
        run_script(step)

    total_elapsed = time.time() - total_start

    logger.info(f"{GREEN}{'='*60}{RESET}")
    logger.info(f"{GREEN}Pipeline complete ({total_elapsed:.2f}s total){RESET}")
    logger.info(f"{GREEN}{'='*60}{RESET}")


if __name__ == "__main__":
    main()