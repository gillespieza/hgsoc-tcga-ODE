---
title:
aliases: 
tags: 
created: 2026-06-23 10:45
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-06-23 11:21
---

# hgsoc-tcga-ODE — Coding & Comment Style Guide

All scripts added or modified in this project must follow these conventions. Consistency matters more than personal preference — deviations need a written reason in the PR or commit body.

---

## Core Principles

Code should prioritise **readability, reproducibility, maintainability, and transparency**. Assumptions must be made explicit and outputs must be traceable.

Future readers should be able to understand — without consulting external documentation — what the code does, why it exists, what assumptions were made, and where outputs are written.

---

## 1. Module Structure

Every script must follow this top-to-bottom layout, in this order:

```
module docstring
stdlib imports
third-party imports
local imports
── blank line ──
ROOT definition
logger definition
── blank line ──
constants / module-level config (if any)
── blank line ──
helper functions
main()
if __name__ == "__main__": main()
```

### 1.1 Module Docstring

Required on every file. One-liner title, then a short description of what the script does and what it outputs. For modelling scripts, include state variables.

```python
"""
ode_model.py — HR-DDR ODE model for HGSOC patient-specific survival prediction.

Mechanistic structure:
    D(t) : DNA damage load
    A(t) : ATM/ATR checkpoint activation
    ...

Key idea:
- Global kinetics are fixed (literature-informed)
- Patient variation enters via RNA-derived pathway compression
"""
```

### 1.2 Imports

Standard library first, third-party second, local imports third. No blank line between entries within a block; one blank line between blocks.

```python
# ✓
import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import ode_model

# ✗ — mixed order, missing separation
import numpy as np
import logging
import pandas as pd
from pathlib import Path
```

Never use wildcard imports:

```python
# ✗
from module import *

# ✓
from module import function_name
```

### 1.3 ROOT and Logger

Defined immediately after imports, before anything else.

```python
ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)
```

`run_pipeline.py` lives at the project root and uses `.parent` (one level up). All other scripts live in `src/` and use `.parent.parent` (two levels up). Never hardcode an absolute path. Never use `os.getcwd()`.

### 1.4 `sys.path` Manipulation

If a script needs to import from `src/`, derive the path from `ROOT`:

```python
# ✓
sys.path.insert(0, str(ROOT / "src"))

# ✗ — breaks if CWD is not the project root
sys.path.insert(0, 'src')
```

### 1.5 `main()` Wrapper — Mandatory

Every script must expose a `def main():` entry point. Module-level execution (top-level code that runs on import) is forbidden. `run_pipeline.py` discovers `main()` by name; scripts without it silently do nothing when called through the pipeline.

```python
# ✓
def main():
    merged = pd.read_csv(ROOT / "data" / "processed" / "hgsoc_tcga_merged.csv")
    ...

if __name__ == "__main__":
    main()

# ✗ — runs on import, breaks pipeline orchestration
merged = pd.read_csv(ROOT / "data" / "processed" / "hgsoc_tcga_merged.csv")
```

---

## 2. Logging

### 2.1 Never Use `print()`

All diagnostic output must go through the module logger so it flows into `pipeline.log` via the root logger configured in `run_pipeline.py`.

Setup (in `run_pipeline.py` or the entry point):

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
```

In every module:

```python
logger = logging.getLogger(__name__)

# ✓
logger.info(f"Loaded merged dataset: {len(merged)} patients")
logger.warning(f"Found {n} unmapped OS_STATUS values")

# ✗
print(f"Loaded merged dataset: {len(merged)} patients")
```

Commented-out `print` statements must be deleted before committing. The only acceptable exceptions are small exploratory notebooks used during active development and never committed to the main branch.

### 2.2 Log Levels

|Level|When to use|
|---|---|
|`logger.info`|Normal progress: rows loaded, files saved, counts|
|`logger.warning`|Recoverable anomaly: unmapped values, fallback triggered, unexpected size|
|`logger.error`|Unrecoverable but caught: use immediately before re-raising|
|`logger.exception`|Inside an `except` block — automatically appends the full traceback|
|`logger.debug`|Verbose detail for active development only; remove before committing|

### 2.3 What to Log

Log the start and end of every major operation, including counts. At minimum, always log:

- Dataset sizes (rows loaded, rows after filtering)
- Files loaded and files written
- Filtering and deduplication steps
- Quality-control metrics and warnings

```python
logger.info(f"Loaded clinical data: {clinical.shape[0]} rows")
# ... processing ...
logger.info(f"After filtering: {len(clinical_clean)} patients")
logger.info(f"Saved: {out_path}")
```

---

## 3. Comments

### 3.1 Explain _Why_, Not _What_

The code already says what it does. Comments exist to explain non-obvious decisions, biological rationale, and constraints that would otherwise require reading a paper to understand.

```python
# ✓ — explains the biological reasoning
# Use min(BRCA1, BRCA2) because both proteins are required for HR;
# the lower one acts as a rate-limiting bottleneck.
BRCA_cap = np.minimum(fpkm['BRCA1'], fpkm['BRCA2']) * fpkm['RAD51']

# ✗ — describes the code, not the reason
# Take the minimum of BRCA1 and BRCA2 and multiply by RAD51
BRCA_cap = np.minimum(fpkm['BRCA1'], fpkm['BRCA2']) * fpkm['RAD51']
```

Comments should document assumptions, domain rationale, and non-obvious implementation choices. They should never merely restate the code:

```python
# ✗
x = x + 1  # add one

# ✓
# Offset avoids log(0) when expression values are absent.
x = np.log1p(x)
```

### 3.2 Section Headers

Use 65-dash separators to divide logical sections within a function:

```python
    # -----------------------------------------------------------------
    # Load raw clinical data
    # -----------------------------------------------------------------
```

Module-level sections (constants, major function groups) use `=` separators:

```python
# =================================================================
# Global Parameters (fixed kinetics)
# =================================================================
```

Section titles use sentence case.

### 3.3 Inline Parameter Comments

For parameter dictionaries and magic numbers, add a trailing comment with units and biological justification on one line. Use ASCII unit notation (`h^-1`) to avoid encoding issues across editors and terminals:

```python
'k_a':     1.0,   # h^-1 — rate at which damage activates ATM/ATR
'tau_drug': 6.0,  # h    — carboplatin plasma half-life ~2-6 h; conservative upper bound
```

### 3.4 Known Caveats

Known methodological limitations go in a `NOTE:` comment at the point where the decision is made:

```python
# NOTE: selecting the cutpoint by minimising log-rank p invalidates that
# p-value for inference. This result is retained as exploratory only.
best_idx = scan["p_value"].idxmin()
```

---

## 4. Function Signatures and Docstrings

Type-annotate all public functions. Use bare built-in types (Python >= 3.9):

```python
# ✓
def compute_patient_params(merged_df: pd.DataFrame) -> pd.DataFrame: ...

def simulate_patient(patient_params: dict, global_params: dict) -> dict: ...

# ✗
def compute_patient_params(merged_df): ...
```

Every function requires a docstring. For short utility functions, a single summary line is sufficient. For modelling functions, document parameters, return values, and the biological or mathematical intent using NumPy style:

```python
def simulate_patient(patient_params: dict, global_params: dict) -> dict:
    """
    Simulate the HR-DDR ODE for a single patient over 120 hours.

    Initial conditions are derived from the zero-damage analytical steady
    state: R(0) = BRCA_cap, all other variables = 0.

    Parameters
    ----------
    patient_params : dict
        Per-patient values: BRCA_cap, ATM_tot, CHK_tot, BCL2_ratio, PATIENT_ID.
    global_params : dict
        Literature-fixed kinetic constants (GLOBAL_PARAMS).

    Returns
    -------
    dict
        Keys: t, D, A, C, R, X, PATIENT_ID, success.
    """
```

---

## 5. Error Handling

Use `logger.exception` inside `except` blocks — it includes the full traceback automatically and is sufficient on its own (no separate `logger.error` needed). Note that `as e` binding is not the problem; swallowing the exception is:

```python
# ✓
try:
    ...
except Exception:
    elapsed = time.time() - start
    logger.exception(f"{script_name} failed after {elapsed:.2f}s")
    raise

# ✗ — loses the traceback and swallows the exception
except Exception as e:
    logger.error(f"Failed: {e}")
```

Always re-raise after logging unless you have a specific, documented recovery path. Never swallow exceptions silently.

For validation failures — bad input shapes, missing columns, unexpected counts — raise `ValueError` with a message that names both expected and observed values:

```python
if len(overlap) == 0:
    raise ValueError(
        "No overlap between clinical and RNA cohorts. "
        "Check PATIENT_ID formatting."
    )
```

---

## 6. File Paths

Construct all paths relative to `ROOT` using the `/` operator (pathlib). Never use string concatenation. Never hardcode filenames that appear in multiple scripts — if a filename changes, it should break loudly.

```python
# ✓
out_path = ROOT / "data" / "processed" / "hgsoc_tcga_merged.csv"
merged.to_csv(out_path, index=False)
logger.info(f"Saved: {out_path}")

# ✗
merged.to_csv("data/processed/hgsoc_tcga_merged.csv", index=False)
```

### Canonical Processed Filenames

Update this table in the same PR that adds or renames a script.

|File|Owner script|
|---|---|
|`data/processed/clinical_clean.csv`|`prepare_clinical.py`|
|`data/processed/rna_clean.csv`|`prepare_RNA.py`|
|`data/processed/hgsoc_tcga_merged.csv`|`merge_data.py`|
|`data/processed/ode_scores.csv`|`run_ode_cohort.py`|
|`data/processed/survival_analysis_df.csv`|`run_ode_cohort.py`|
|`data/processed/univariate_cox_comparison.csv`|`analyse_ode_survival.py`|
|`data/processed/ml_comparison_table.csv`|`ml_benchmark.py`|
|`results/figures/`|all plotting scripts|

---

## 7. Output Files and Figures

Every `to_csv` and `savefig` call must be immediately followed by `logger.success(f"[FILE] Saved: {filename}")`.

Before writing any file, create its parent directory explicitly:

```python
out_path.parent.mkdir(parents=True, exist_ok=True)
```

Figures go to `results/figures/`. Processed data goes to `data/processed/`. No script writes to `data/raw/`.

Figure filenames follow the pattern `fig_{type}_{subject}.png`, where `type` is a short descriptor of the plot type and `subject` names what is plotted. Both parts are required and must be distinct:

```
fig_kaplan_meier_aucx.png
fig_forest_plot_ml_comparison.png
fig_lasso_coefficients_cox.png
```

Where practical, save intermediate processed inputs, merged datasets, and derived features. This improves reproducibility and makes debugging easier.

---

## 8. Naming Conventions

### Variables

Descriptive `snake_case`. Avoid single-letter or generic names except for mathematical temporaries:

```python
# ✓
patient_params
survival_analysis_df
gene_id_map

# ✗
x, tmp, data2
```

### Functions

Use verbs that describe what the function does:

```python
load_data()
clean_clinical_data()
simulate_patient()
save_results()
```

### Constants

Uppercase `snake_case`:

```python
ROOT
OUTPUT_DIR
GLOBAL_PARAMS
```

### Git Commit Scope

Use the script filename without `.py` as the scope. If two scripts share a natural scope name (e.g., both `pipeline.py` and `run_pipeline.py` could claim `pipeline`), use the full filename stem of the file being changed:

```
fix(run_ode_cohort): filter success flag before saving survival_analysis_df
refactor(ode_validation): wrap execution in main() and fix sys.path
docs(style_guide): consolidate naming and data handling sections
```

---

## 9. Data Handling

### Avoid In-Place Modification

Prefer explicit copies when transforming datasets to avoid unexpected side effects:

```python
df = df.copy()
```

### Validate Assumptions Explicitly

Fail early rather than producing silent downstream errors:

```python
if len(matched_genes) == 0:
    raise ValueError(
        "No matching genes found between RNA and clinical cohorts. "
        f"Expected > 0 overlapping PATIENT_IDs; got {len(matched_genes)}."
    )
```

### Report Quality-Control Metrics

Report row counts before and after every filtering or merging step, including duplicate removal and missing-value handling:

```python
logger.info(f"Loaded: {len(df)} rows")
logger.info(f"Dropped {n_dup} duplicate PATIENT_IDs")
logger.info(f"Dropped {n_missing} rows with missing OS_STATUS")
logger.info(f"Final cohort: {len(df_clean)} patients")
```

---

## 10. Formatting

**Line length:** target ≤ 88 characters. Use implicit line continuation for long log messages and function calls:

```python
logger.info(
    f"Final cohort: {n_patients} patients, "
    f"{n_genes} genes"
)
```

**Whitespace:** follow PEP 8. `x = y + z`, not `x=y+z`.

**Blank lines:** two blank lines between top-level functions; one blank line between logical blocks within a function. Avoid excessive vertical spacing.

---

## 11. Anti-Patterns

| Anti-pattern                          | Why it fails                              | Correct approach                        |
| ------------------------------------- | ----------------------------------------- | --------------------------------------- |
| `print(...)`                          | Bypasses logger; invisible in log file    | `logger.info(...)`                      |
| `sys.path.insert(0, 'src')`           | Breaks if CWD ≠ project root              | `sys.path.insert(0, str(ROOT / "src"))` |
| Module-level execution                | Runs on import; breaks pipeline           | Wrap everything in `def main():`        |
| Hardcoded or absolute paths           | Not portable across machines              | `ROOT / "..."`                          |
| `.parent.parent` in root scripts      | Resolves two levels above project root    | `.parent` only for scripts at root      |
| Committed `print` or debug statements | Noise; skips the log file                 | Delete before committing                |
| Dead utility functions                | Misleading; suggests code is used         | Remove or wire them up                  |
| `except ...: pass` or silent catch    | Swallows failures; impossible to diagnose | `logger.exception(...)` then `raise`    |
| `show_plots=True` in pipeline scripts | Blocks headless execution                 | `show_plots=False`                      |

---

## 12. Git Commit Messages

Conventional commits format:

```
<type>(<scope>): <short imperative summary>

- bullet explaining what changed and why
- second bullet if needed
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`. Scopes: the script filename stem (see section 8).

```
# ✓
fix(run_ode_cohort): filter success flag before saving survival_analysis_df

- Add boolean filter on scores_df["success"] before constructing analysis_df
- Prevents NaN ODE scores from propagating into Cox and KM analyses

refactor(ode_validation): wrap execution in main() and fix sys.path

- Replace module-level execution with def main() for pipeline compatibility
- Replace sys.path.insert(0, 'src') with ROOT-derived path

# ✗
fixed bug
update pipeline
```

---

## Reproducibility Checklist

Before committing any script, verify:

- [ ] Module docstring present and accurate
- [ ] All functions have docstrings (one-liner minimum)
- [ ] `logger` used throughout; no `print` calls remain
- [ ] Assumptions documented in comments at the point of decision
- [ ] All paths constructed with `ROOT / ...`
- [ ] Output directories created with `mkdir(parents=True, exist_ok=True)`
- [ ] Every file write followed by `logger.info(f"Saved: {out_path}")`
- [ ] QC metrics (row counts, drop counts) logged at each filtering step
- [ ] `ValueError` raised for critical validation failures
- [ ] `logger.exception` used inside `except` blocks; exceptions re-raised
- [ ] `def main():` entry point present; no module-level execution
- [ ] Canonical filename table updated if a script or output path changed
- [ ] No commented-out debug code