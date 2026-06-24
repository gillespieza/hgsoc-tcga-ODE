#!/usr/bin/env python
"""
Master pipeline orchestrator for HGSOC TCGA ODE survival modelling.

Orchestrates the complete analysis workflow, starting from raw clinical,
mutation, and RNA-seq data and ending with survival, machine-learning,
and biomarker analyses.

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

Note: ode_model is imported as a library by downstream scripts and is
not executed as a standalone pipeline stage.
"""

import sys
import time
import logging
import os
from pathlib import Path

from tqdm import tqdm
from colorama import just_fix_windows_console

# =================================================================
# Project paths
# =================================================================

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

# =================================================================
# Logging helpers
# =================================================================

SUCCESS = 25
logging.addLevelName(SUCCESS, "SUCCESS")

SUMMARY = 26
logging.addLevelName(SUMMARY, "SUMMARY")

def success(self, message, *args, **kwargs):
    if self.isEnabledFor(SUCCESS):
        self._log(SUCCESS, message, args, **kwargs)

def summary(self, message, *args, **kwargs):
    if self.isEnabledFor(SUMMARY):
        self._log(SUMMARY, message, args, **kwargs)

logging.Logger.success = success
logging.Logger.summary = summary

# =================================================================
# Logging setup
# =================================================================

logger = logging.getLogger(__name__)

log_file = ROOT / "pipeline.log"

class _ColorFormatter(logging.Formatter):
    """
    Formatter that prepends ANSI colour codes for terminal output only.

    Applied exclusively to the StreamHandler so that pipeline.log
    contains clean, ANSI-free text — readable in editors and CI logs.
    """

    logging.Logger.success = success

    _COLOURS = {
        SUCCESS: "\033[32m",
        SUMMARY: "\u001b[36m",
        logging.DEBUG:    "\u001b[36m",   # cyan
        logging.INFO:     "\u001b[0m",    # default terminal colour
        logging.WARNING:  "\u001b[33m",   # yellow
        logging.ERROR:    "\u001b[31m",   # red
        logging.CRITICAL: "\u001b[31m",   # red
    }
    _RESET = "\u001b[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Prepend the level-appropriate colour code and append a reset."""
        colour = self._COLOURS.get(record.levelno, self._RESET)
        return f"{colour}{super().format(record)}{self._RESET}"

class TqdmLoggingHandler(logging.Handler):
    """
    Logging handler compatible with tqdm progress bars.
    """

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)

def success_msg(script_name: str, elapsed: float) -> str:
    """Return a formatted completion message for a finished pipeline stage."""
    return f"[OK] {script_name} completed ({elapsed:.2f}s)"

def fail_msg(script_name: str, elapsed: float):
    return f"[FAIL] {script_name} failed after {elapsed:.2f}s\n"


def run_script(script_name: str) -> float:
    """
    Import a module from src/ and execute its main() function.

    Each pipeline stage exposes a main() entry point, allowing stages
    to be run independently or orchestrated from this master script.
    Dynamic imports are used so stages can be listed as strings and
    executed sequentially without hard-coded top-level imports.

    Parameters
    ----------
    script_name : str
        Module name (without .py) corresponding to a script in src/.

    Returns
    -------
    float
        Wall-clock time taken to complete the stage, in seconds.

    Raises
    ------
    Exception
        Re-raises any exception raised by the stage after logging it.
    """

    logger.info("=" * 60)
    logger.info(f"▶ Starting {script_name}")
    logger.info("=" * 60)

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
            # Stage has no main() — re-import triggers module-level code,
            # but this pattern is fragile. Stages should always define main().
            __import__(script_name)

        elapsed = time.time() - start

        logger.success(success_msg(script_name, elapsed))
        
        return elapsed

    except Exception:
        elapsed = time.time() - start

        logger.exception(
            f"{script_name} failed after {elapsed:.2f}s"
        )
        logger.exception(fail_msg(script_name, elapsed))

        raise
    
def main() -> None:
    """
    Run the complete preprocessing, modelling, and analysis workflow.

    Configures the environment and logging, then executes each pipeline
    stage in dependency order. Stages are imported dynamically at
    runtime and their main() functions called sequentially.
    """

    # -----------------------------------------------------------------
    # Environment configuration
    # -----------------------------------------------------------------

    # Several scientific Python libraries use NumExpr internally for
    # accelerated numerical computation. Bounding the thread count
    # avoids occasional performance warnings on Windows systems.
    os.environ["NUMEXPR_MAX_THREADS"] = "16"

    # -----------------------------------------------------------------
    # Add src/ to module search path
    # -----------------------------------------------------------------

    # run_script() imports stage modules by name at runtime.
    # Inserting src/ here ensures they are found regardless of which
    # directory the pipeline is launched from.
    sys.path.insert(0, str(ROOT / "src"))

    # -----------------------------------------------------------------
    # Logging configuration
    # -----------------------------------------------------------------

    just_fix_windows_console()
    
    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    # FileHandler receives plain text — no colour codes in the log file.
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(log_format))

    # StreamHandler receives colour-coded output for terminal readability.
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(_ColorFormatter(log_format))

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, stream_handler],
    )

    # -----------------------------------------------------------------
    # Pipeline execution
    # -----------------------------------------------------------------

    logger.info("=" * 60)
    logger.info("HGSOC TCGA ODE Survival Pipeline")
    logger.info("=" * 60)

    total_start = time.time()

    # Pipeline order is critical: each stage produces outputs consumed
    # by one or more downstream stages.
    #
    # Data flow:
    #
    #   Clinical data  ──┐
    #   RNA-seq data   ──┤
    #   BRCA mutations ──┘
    #                    ▼
    #          Integrated patient table
    #                    ▼
    #     Patient-specific ODE simulations
    #                    ▼
    #           ODE-derived biomarkers
    #                    ▼
    #  Survival and machine-learning analyses
    steps = [
        "prepare_clinical",
        "prepare_RNA",
        "prepare_brca1_2_mutation",
        "merge_data",

        # ode_model is imported as a library by downstream scripts;
        # it is not executed as a standalone pipeline stage.
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

        # Interpret model behaviour:
        #   Cox model — coefficient magnitude
        #   RSF       — permutation importance
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

    logger.info("=" * 60)
    logger.info(f"Pipeline complete ({total_elapsed:.2f}s total)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()