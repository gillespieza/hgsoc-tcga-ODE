#!/usr/bin/env python
"""
Master pipeline script for HGSOC TCGA ODE survival modeling.

Runs all preparation scripts in order:
1. prepare_clinical.py  - Load and clean clinical data
2. prepare_RNA.py       - Load RNA-seq, map Entrez→symbol, extract 14 ODE genes
3. merge_data.py        - Merge clinical + expression, save merged dataset
"""

import sys
import time
import logging
import os
os.environ["NUMEXPR_MAX_THREADS"] = "16"

import pandas as pd
from pathlib import Path

# set the root path of the project
ROOT = Path(__file__).resolve().parent

# Add src/ to path so we can import scripts from there
sys.path.insert(0, str(ROOT / "src"))

# Configure logging with ASCII-only output
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

# Define ANSI color codes
RED    = "\u001b[31m"
GREEN  = "\u001b[32m"
YELLOW = "\u001b[33m"
RESET  = "\u001b[0m"
CYAN   = "\u001b[36m"

# Use ASCII text instead of emoji for Windows compatibility
def success_msg(script_name: str, elapsed: float):
    return f"[OK] {script_name} completed ({elapsed:.2f}s)\n"

def fail_msg(script_name: str, elapsed: float):
    return f"[FAIL] {script_name} failed after {elapsed:.2f}s\n"

def run_script(script_name: str):
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
    logger.info(f"{CYAN}{'='*60}{RESET}")
    logger.info(f"{CYAN}HGSOC TCGA ODE Survival Pipeline{RESET}")
    logger.info(f"{CYAN}{'='*60}{RESET}")
    
    total_start = time.time()
    
    steps = [
        "prepare_clinical",
        "prepare_RNA",
        "prepare_brca1_2_mutation",
        "merge_data",
    ]
    
    for step in steps:
        run_script(step)
    
    total_elapsed = time.time() - total_start
    
    logger.info(f"{GREEN}{'='*60}{RESET}")
    logger.info(f"{GREEN}Pipeline complete ({total_elapsed:.2f}s total){RESET}")
    logger.info(f"{GREEN}{'='*60}{RESET}")

if __name__ == "__main__":
    main()