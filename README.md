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
- Output: data/processed/tcga_ov_merged.csv