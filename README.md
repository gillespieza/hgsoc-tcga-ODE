# hgsoc-tcga-ODE


# Project Assumptions

## Cancer Cohort
- Cancer type: High-Grade Serous Ovarian Carcinoma (HGSOC)
- Cohort: HGSOC_TCGA_GDC
- Data Source: The Cancer Genome Atlas (TCGA) via the cBioPortal
- Data source code: hgsoc_tcga_gdc
- Data source link: https://www.cbioportal.org/study/summary?id=hgsoc_tcga_gdc

## ODE Prognostic Endpoint
- Endpoint: Apoptotic signal (Ap) Area Under the Curve (AUC)
- Rationale: Apoptotic commitment is more directly linked to chemotherapy-induced tumour cell death than residual DNA damage.

## Global Parameter Strategy
- Strategy: Literature-fixed rate constants
- Rationale: Reduces optimisation complexity and avoids fitting parameters without suitable experimental time-course data.

## Survival Endpoint
- Endpoint: Overall Survival (OS)
- Rationale: Larger event count, fewer missing values, and greater statistical power than PFS in TCGA-OV.

# Data Sources

## ⚠️ Why NOT TPM or z-scores?
🚫 TPM problem
- smooths out variability you actually need for:
    - DDR pathway activation
    - sensitivity heterogeneity
🚫 z-score problem (critical)
- removes: magnitude differences between patients
- ODE models need: real-valued state differences

👉 z-scores are for ML clustering, NOT mechanistic modelling

# Pipeline steps
- Patients with complete OS data: 583
- final cleaned set: 426
- final event rate: 62.21%

## BRCA1/2 mutations
- 30 mutations, non of them somatic. 
    - BRCA1/2 mutant patients: 28
    - BRCA1/2 wildtype patients: 555

## Step 1 — Data Acquisition (completed 2026-06-17)
- Dataset: hgsoc_tcga_gdc (cBioPortal)
- Raw patients with OS data: [607]
- Raw patients with sample data: [603]
- Final merged dataset: [425] patients × [43] columns
- OS event rate: [62.21%]
- BRCA1/2 mutant patients: [28]
- Output: data/processed/hgsoc_tcga_merged.csv



---
# HGSOC HR-DDR ODE Survival Model

Mechanistic ODE-based survival prediction for High-Grade Serous Ovarian Cancer (HGSOC)
using TCGA RNA-seq data. A five-variable HR-DDR ODE system is parameterised from
14 gene expression values and compared against Cox LASSO and Random Survival Forest.

---

## Results Summary

| Model | C-index | Log-rank p |
|---|---|---|
| HR-DDR ODE (AUC_X) — zero-shot | 0.533 | 0.0224 (median split) |
| Cox LASSO | 0.533 ± 0.035 | — |
| Random Survival Forest | 0.533 ± 0.018 | — |

**Key finding:** The mechanistic ODE matches data-trained ML models at C-index = 0.533
with no outcome data used for calibration. Kaplan-Meier stratification by AUC_X
achieves log-rank p = 0.0077 at the optimal cutoff.

---

## Repository Structure
```
hgsoc-tcga-ODE/
├── data/
│ ├── raw/ # TCGA downloads (not committed)
│ └── processed/ # Merged CSVs, ODE scores, figures
│ ├── hgsoc_tcga_merged.csv # 420 patients × 14 genes + clinical
│ ├── ode_scores.csv # AUC_X, X_peak, T_repair, D_resid
│ └── survival_analysis_df.csv
├── src/
│ ├── prepare_clinical.py
│ ├── prepare_RNA.py
│ ├── prepare_brca1_2_mutation.py
│ ├── merge_data.py
│ ├── ode_model.py
│ ├── cohort_simulation.py
│ ├── survival_analysis.py
│ ├── ml_comparison.py
│ └── kaplan_meier.py
├── results/
│ ├── figures/ # Final publication figures
│ └── report.md # Full project report
├── requirements.txt
└── README.md
```
## Pipeline
Step 1 → data acquisition & merging (merge_data.py)
Step 2 → ODE model definition (ode_model.py)
Step 3 → cohort simulation (420 patients) (cohort_simulation.py)
Step 4 → ODE score extraction
Step 5 → score distributions & BRCA boxplots
Step 6 → threshold scan & optimal KM cutoff
Step 7 → univariate/multivariate Cox
Step 8 → survival_analysis_df construction (survival_analysis.py)
Step 9 → ML comparison: Cox LASSO + RSF (ml_comparison.py)
Step 10 → Kaplan-Meier final figure (kaplan_meier.py)
Step 11 → report assembly
Step 12 → README (this file)


---

## Locked Decisions

| Decision | Value |
|---|---|
| ODE framework | HR-DDR: ATM/ATR → CHK1/2 → HR repair vs apoptosis |
| Primary output | AUC_X — area under apoptotic commitment curve |
| Parameter strategy | Literature-fixed, zero-shot (no outcome optimisation) |
| Survival endpoint | Overall Survival (OS) |
| Expression transform | log₂(FPKM + 1) |
| TP53 | Negative control only — not used in ODE |

---

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```