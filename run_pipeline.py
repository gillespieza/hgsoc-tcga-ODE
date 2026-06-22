#!/usr/bin/env python
"""
Master pipeline script for HGSOC TCGA ODE survival modeling.

This file runs the full preprocessing/model pipeline in order:
1. prepare_clinical.py          - clean clinical data
2. prepare_RNA.py               - load RNA-seq data and keep ODE genes
3. prepare_brca1_2_mutation.py  - build BRCA1/2 mutation labels
4. merge_data.py                - merge clinical + expression data
5. run_ode_cohort.py            - simulate cohort and save ODE-derived scores
"""

import sys
import time
import logging
import os
from pathlib import Path

# Limit NUMEXPR threads to avoid performance or compatibility issues on Windows.
# NUMEXPR_MAX_THREADS=16 is a reasonable default for most modern CPUs (as of 2026).
os.environ["NUMEXPR_MAX_THREADS"] = "16"

# set the root path of the project
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

# Define ANSI color codes for terminal output.
RED    = "\u001b[31m"
GREEN  = "\u001b[32m"
YELLOW = "\u001b[33m"
RESET  = "\u001b[0m"
CYAN   = "\u001b[36m"

# Small helper for success messages.
def success_msg(script_name: str, elapsed: float):
    return f"[OK] {script_name} completed ({elapsed:.2f}s)\n"

# Small helper for failure messages.
def fail_msg(script_name: str, elapsed: float):
    return f"[FAIL] {script_name} failed after {elapsed:.2f}s\n"

def run_script(script_name: str):
    """
    Import a script from src/ and run its main() function if it exists.

    This lets us keep each pipeline step in a separate file, while still
    running everything from one master script.
    """

    # Start with some console output to indicate which script is running.    
    logger.info(f"{RESET}{'='*60}{RESET}")
    logger.info(f"{YELLOW}Running: {script_name}{RESET}")
    logger.info(f"{RESET}{'='*60}{RESET}")
    
    start = time.time()
    
    try:
        # Import the module by name, e.g. "prepare_clinical".
        module = __import__(script_name)

        # If the module defines main(), call it.
        if hasattr(module, "main"):
            module.main()
        else:
             # Fallback: just import the module so top-level code runs.
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
    Run all preprocessing and modeling steps in the correct order.
    """    

    # Start with some console output to indicate the pipeline is starting.
    logger.info(f"{CYAN}{'='*60}{RESET}")
    logger.info(f"{CYAN}HGSOC TCGA ODE Survival Pipeline{RESET}")
    logger.info(f"{CYAN}{'='*60}{RESET}")
    
    total_start = time.time()
    
    # Steps must run in this order because later files depend on earlier ones.
    steps = [
        "prepare_clinical",
        "prepare_RNA",
        "prepare_brca1_2_mutation",
        "merge_data",
        #"ode_model",
        #"ode_validation",
        "run_ode_cohort",
    ]
    
    # Run each step one by one.
    for step in steps:
        run_script(step)
    
    total_elapsed = time.time() - total_start
    
    # Final summary line.
    logger.info(f"{GREEN}{'='*60}{RESET}")
    logger.info(f"{GREEN}Pipeline complete ({total_elapsed:.2f}s total){RESET}")
    logger.info(f"{GREEN}{'='*60}{RESET}")

if __name__ == "__main__":
    main()