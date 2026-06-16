# hgsoc-tcga-ODE


# Project Assumptions

## Cancer Cohort
- Cancer type: High-Grade Serous Ovarian Carcinoma (HGSOC)
- Cohort: TCGA-OV
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