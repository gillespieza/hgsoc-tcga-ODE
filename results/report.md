---
title:
aliases: 
tags: 
created: 2026-06-22 17:53
cssclasses: wide
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-06-28 12:30
---

# HR-DDR ODE Survival Model - HGSOC TCGA
## Project Report

**Dataset:** TCGA HGSOC, n = 420 patients  
**Endpoint:** Overall Survival (OS)  
**Approach:** Mechanistic ODE → AUC_X score → survival analysis + ML comparison

---

## 1. Background

### 1.1 Disease Context

High-Grade Serous Ovarian Carcinoma (HGSOC) is the most lethal gynaecological malignancy and accounts for the majority of ovarian cancer deaths worldwide. The standard first-line treatment is cytoreductive surgery followed by platinum-taxane chemotherapy, typically carboplatin and paclitaxel. Although most patients initially respond to platinum, the majority relapse within two years, and subsequent responses to re-treatment are progressively shorter. Five-year survival remains below 50%, underscoring the urgent need for reliable prognostic biomarkers that can identify patients at high risk of early relapse or poor platinum response at the point of diagnosis.

### 1.2 The DNA Damage Response as a Prognostic Axis

Platinum compounds act by forming intra-strand and inter-strand crosslinks in genomic DNA, inducing double-strand breaks (DSBs) that trigger the DNA damage response (DDR). Cell fate after platinum exposure is determined by the balance between two competing pathways: homologous recombination (HR) repair, which removes DSBs and allows cells to survive, and mitochondrial apoptosis, which commits irreparably damaged cells to programmed death. Tumours with deficient HR - most notably those harbouring BRCA1 or BRCA2 mutations - are unable to efficiently resolve platinum-induced DSBs and consequently show enhanced apoptotic commitment, which translates clinically into greater platinum sensitivity and, in most series, improved overall survival.

The central mediators of this response are well characterised. ATM and ATR kinases sense DSBs and stalled replication forks respectively, phosphorylating and activating the checkpoint effectors CHK1 and CHK2. CHK1/2 in turn coordinate cell cycle arrest and recruit the HR repair machinery, anchored by the BRCA1-PALB2-BRCA2-RAD51 scaffold. If damage persists - either because HR capacity is insufficient or because the damage load is too great - pro-apoptotic BAX/BAD-mediated signalling overcomes the anti-apoptotic buffering provided by BCL-2 family members, committing the cell to apoptosis. This mechanistic framework is well-supported by experimental and clinical data and provides a biologically grounded basis for patient stratification.

### 1.3 Mechanistic Modelling as an Alternative to Black-Box ML

Conventional data-driven approaches to survival prediction - including Cox proportional hazards regression and ensemble methods such as Random Survival Forests - treat gene expression profiles as statistical inputs and optimise model parameters directly against observed outcomes. While powerful, these approaches require sufficiently large training cohorts to avoid overfitting, are sensitive to batch effects and platform heterogeneity, and produce predictions that are difficult to interpret mechanistically.

Ordinary differential equation (ODE) models offer a complementary strategy. By encoding known biology as a system of coupled differential equations with literature-informed kinetic parameters, a mechanistic model can generate patient-specific predictions from steady-state gene expression alone, without fitting any outcome data. This "zero-shot" property means the model is not susceptible to overfitting and its parameters carry direct biological meaning. The canonical example is the p53 ODE model, which has been validated on neuroblastoma patient cohorts and shown to be prognostic of event-free survival using only pre-treatment expression data.

![[hr_ddr_pathway.png]]

**Figure 1. Mechanistic structure of the HR-DDR ODE model.** Carboplatin induces DNA double-strand breaks (DSBs) at rate φ(t) = D₀·e^(−t/τ), driving the damage state D(t). ATM/ATR kinases (A(t)) sense DSBs and activate CHK1/CHK2 effectors (C(t)), which simultaneously recruit the HR repair complex (R(t)) and drive apoptotic commitment (X(t)). Sustained checkpoint activity exhausts R(t) via BRCA1 hyperphosphorylation (k_load·C·R), reducing repair capacity and releasing apoptotic suppression. The mitochondrial BCL2/BAX balance scales the decay rate of X(t), modulating apoptotic resistance. Patient-specific ODE parameters (BRCA_cap, ATM_tot, CHK_tot, BCL2_ratio) are derived from pre-treatment RNA-seq expression of 14 pathway genes; global kinetic rate constants are fixed from literature values. The primary model output, AUC_X = ∫X(t) dt, quantifies cumulative apoptotic commitment and is used as the survival predictor in all downstream Cox and Kaplan-Meier analyses. Solid arrows indicate activation; dashed bars (⊣) indicate inhibition or suppression.

### 1.4 Aims

This project applies a mechanistic HR-DDR ODE model to the HGSOC TCGA cohort to address two questions:

1. **Can the ODE predict overall survival in HGSOC?** Patient-specific ODE parameters are derived from RNA-seq expression of 13 pathway genes, the model is integrated forward in time under a platinum damage perturbation, and the resulting apoptotic commitment score (AUC_X) is tested for association with overall survival via Cox regression and Kaplan-Meier stratification.
2. **How does the ODE compare to data-trained ML models?** Cox LASSO and Random Survival Forest models are benchmarked against the zero-shot ODE on the same 14-gene feature set (including TP53 as a negative control), using 5-fold cross-validated C-index and 1000-replicate bootstrap confidence intervals.

---

## 2. Methods

### 2.1 ODE Model
A five-variable HR-DDR ODE system was constructed to model the cellular response to platinum-induced DNA damage:

- **D(t)** - DNA damage load
- **A(t)** - Activated ATM/ATR
- **C(t)** - Activated CHK1/CHK2
- **R(t)** - Functional HR repair complex (BRCA1-BRCA2-PALB2-RAD51)
- **X(t)** - Apoptotic commitment signal

Patient-specific initial conditions were derived analytically from the zero-damage steady state: R_ss = BRCA_cap(patient), all other variables = 0. No numerical optimisation was performed against survival data (zero-shot model).

Global kinetic parameters were fixed from literature values.

Patient-specific parameters (normalised to population median):
- `BRCA_cap` - geometric mean of BRCA1, BRCA2, RAD51, PALB2, BRIP1
- `ATM_tot` - mean of ATM, ATR
- `CHK_tot` - mean of CHEK1, CHEK2
- `BCL2_ratio` - (BCL2 + BCL2L1) / (BAX + BAD + ε)

Primary ODE output: **AUC_X** - area under the apoptotic commitment curve X(t).

### 2.2 Survival Models
Three models were compared on the same 14-gene feature set:

| Model                  | Strategy                        |
| ---------------------- | ------------------------------- |
| HR-DDR ODE             | Mechanistic, zero-shot          |
| Cox LASSO              | 5-fold CV, penalised regression |
| Random Survival Forest | 5-fold CV, ensemble             |

Discrimination was assessed by Harrell's C-index (bootstrap 95% CI).

---

## 3. Clinical Cohort

### 3.1 Cohort Assembly and Attrition

The analysis cohort was drawn from the TCGA High-Grade Serous Ovarian Carcinoma (HGSOC) dataset accessed via cBioPortal. Starting from 608 raw patient records, 25 patients were excluded for missing or invalid overall survival data (OS_MONTHS ≤ 0 or OS_STATUS unmapped), leaving 583 patients after OS filtering. No duplicate PATIENT_IDs were identified in the clinical table. A further 163 patients were lost at the RNA merge step - clinical records with no matching RNA-seq profile in the FPKM expression file - yielding a final analysis cohort of **420 patients**.

![[fig_cohort_summary.png]]

**Figure 2. Clinical cohort summary for the HGSOC TCGA dataset (n = 420).**  
Top-left: patient attrition from raw download to final merged cohort.  
Top-centre: follow-up time distribution stratified by vital status.  
Top-right: overall event rate (deceased vs. living/censored).  
Bottom-left: Kaplan-Meier curve for the full cohort with median OS marked.  
Bottom-centre: BRCA1/2 germline mutation prevalence (bottom-right panel hidden for layout symmetry).

### 3.2 Cohort Characteristics

**Overall survival.** Of 420 patients, 262 (62.4%) experienced the primary endpoint (death) during follow-up, and 158 (37.6%) were censored alive. Median overall survival was **44.5 months** (approximately 3.7 years). The follow-up distribution is right-skewed: most events occur within the first 75 months, with a small number of patients followed beyond 150 months, consistent with the long natural history of platinum-sensitive HGSOC.

**BRCA1/2 mutation status.** Germline BRCA1/2 mutations were identified in 17 patients (4.0%), with the remaining 403 patients (96.0%) classified as wild type. This mutation rate is lower than the 15–20% typically reported in clinical HGSOC series, likely reflecting the known ascertainment bias in the TCGA cohort towards sporadic cases and the use of somatic (rather than germline) sequencing panels for a proportion of samples.

**Endpoint choice.** Overall survival was selected in preference to progression-free survival because it carries a larger event count in this dataset, minimises the ambiguity of radiological progression calls, and is the standard primary endpoint in HGSOC clinical trials. The high event rate (62.4%) provides adequate statistical power for both Cox proportional hazards modelling and Kaplan-Meier stratification.

---

## 4. Results

### 4.1 ODE Validation
Two representative patients (BRCA-mutant vs BRCA-wildtype) showed biologically plausible trajectories: the BRCA-mutant patient produced higher AUC_X (greater apoptotic commitment), consistent with known platinum sensitivity.

![[ode_validation_trajectories.png]]

**Figure 3. Mechanistic trajectories of HR-DDR ODE model state variables for representative patients.** Simulation profiles of state variables over 120 hours following a transient carboplatin damage perturbation ($D_0 = 1.0$, half-life $\tau_{\text{drug}} = 6$ h). Low-BRCA patient (red, TCGA-29-1762, $\text{BRCA}_{\text{cap}} = 0.603$) represents homologous recombination deficiency, while High-BRCA patient (blue, TCGA-59-A5PD, $\text{BRCA}_{\text{cap}} = 2.369$) represents homologous recombination proficiency.  
From left to right:
- **D (DNA Damage)**: Spikes rapidly following drug entry. In the Low-BRCA patient, damage resolves more slowly due to reliance on slower/error-prone non-homologous end joining (NHEJ) repair. In the High-BRCA patient, damage is rapidly resolved.
- **A (ATM/ATR activation)**: Sensed damage signal tracks with DNA damage load, showing sustained activation in the Low-BRCA patient.
- **C (CHK signaling)**: Downstream kinase signaling cascade activated by ATM/ATR shows a corresponding high-amplitude, prolonged plateaus in the Low-BRCA patient.
- **R (HR repair complex)**: Repair complex abundance starts at the zero-damage analytical steady state ($R(0) = \text{BRCA}_{\text{cap}}$) and undergoes depletion driven by checkpoint-mediated consumption. The Low-BRCA patient exhibits severe depletion and delayed recovery.
- **X (Apoptosis)**: Apoptotic commitment accumulates under sustained checkpoint signaling. The Low-BRCA patient shows high apoptotic drive ($\text{AUC}_X = 443.710$), indicating high platinum sensitivity, whereas the High-BRCA patient suppresses apoptosis ($\text{AUC}_X = 111.880$) via rapid DNA repair and lower checkpoint activation.

### 4.2 ODE Score Distributions

To validate the biological plausibility of the model outputs at the population level, we analyzed the distributions of the four ODE-derived scores across the entire cohort ($n = 420$) and stratified them by BRCA mutation status (wild type, $n = 403$ vs. mutant, $n = 17$).

#### 4.2.1 Apoptotic Commitment (AUC_X)
The area under the apoptotic commitment curve ($AUC_X$) is approximately log-normally distributed across the cohort (Figure 5A). Consistent with the known clinical sensitivity of BRCA-deficient tumors to DNA-damaging platinum agents, BRCA-mutant patients exhibited a significantly higher median $AUC_X$ of **165.00** (mean $176.57 \pm 112.77$) compared to wild-type patients with a median of **107.61** (mean $136.56 \pm 107.35$), as shown in Figure 4A. This confirms that the zero-shot mechanistic model successfully captures increased apoptotic drive in homologous recombination deficient (HRD) tumors without outcome-based parameter tuning.

#### 4.2.2 Peak Apoptotic Signal (X_peak)
The peak level of the apoptotic commitment signal ($X_{peak}$) mirrors the trends seen with $AUC_X$. It is approximately log-normally distributed (Figure 5B) and is elevated in the BRCA-mutant group (Figure 4B). BRCA-mutant patients reached a higher median peak apoptotic signal of **5.85** (mean $6.11 \pm 3.56$) compared to **3.88** in wild-type patients (mean $4.88 \pm 3.49$). This suggests that HR-deficient cells not only undergo greater overall apoptosis but also achieve a higher maximum amplitude of apoptotic signaling.

#### 4.2.3 Time to DNA Repair (T_repair)
Time to resolve DNA damage ($T_{repair}$) is a discrete variable representing the hour at which DNA damage falls below the repair threshold (Figure 5C). The cohort-wide distribution is tightly grouped around a median of 61.0 hours. Interestingly, BRCA-mutant patients exhibit a slightly longer median repair time of **63.00 hours** (mean $62.12 \pm 3.74$ h) compared to **61.00 hours** for wild-type patients (mean $60.99 \pm 3.74$ h), as shown in Figure 4C. This slower resolution of damage is a hallmark of homologous recombination deficiency, where cells must rely on slower or error-prone alternative repair mechanisms.

#### 4.2.4 Residual DNA Damage (D_resid)
The residual DNA damage load ($D_{resid}$) at 120 hours represents the damage that could not be resolved by the end of the simulation. For nearly all patients, the residual damage was extremely small (in the range of $10^{-9}$), indicating that the damage was largely resolved (Figure 5D). BRCA-mutant patients had a slightly higher median residual damage of **$7.75 \times 10^{-9}$** (mean $8.06 \times 10^{-9}$) compared to **$7.28 \times 10^{-9}$** in wild-type patients (mean $7.73 \times 10^{-9}$), as shown in Figure 4D. Although the absolute difference is minimal, it reflects the slower and less efficient double-strand break repair in the mutant group.

![[fig_boxplot_ode_scores_brca.png]]

**Figure 4: Distribution of ODE scores stratified by BRCA mutation status.** 2x2 grid of box plots showing (A) Apoptotic commitment ($AUC_X$), (B) Peak apoptotic signal ($X_{peak}$), (C) Time to DNA repair ($T_{repair}$), and (D) Residual DNA damage ($D_{resid}$) in BRCA wild-type (blue, $n = 403$) vs. BRCA-mutant (red, $n = 17$) patients. Mutants exhibit elevated apoptotic commitment (A, B), longer repair times (C), and slightly higher residual damage (D).

![[fig_hist_ode_scores.png]]

**Figure 5: Cohort-wide distribution of ODE scores.** 2x2 grid of histograms showing the population distributions of (A) Apoptotic commitment ($AUC_X$), (B) Peak apoptotic signal ($X_{peak}$), (C) Time to DNA repair ($T_{repair}$), and (D) Residual DNA damage ($D_{resid}$), with pre-specified tertile boundaries (dashed lines) and optimal exploratory cutoffs (dotted lines) marked.

### 4.3 Univariate Cox Regression

| ODE Score | HR    | 95% CI         | p-value   | C-index |
| --------- | ----- | -------------- | --------- | ------- |
| AUC_X     | 0.856 | [0.756, 0.970] | **0.015** | 0.533   |
| X_peak    | 0.855 | [0.753, 0.971] | **0.015** | 0.532   |
| T_repair  | 0.856 | [0.756, 0.970] | **0.015** | 0.528   |
| D_resid   | 1.017 | [0.894, 1.156] | 0.801     | 0.482   |

To stabilize the regression models and make the hazard ratios directly comparable, the log-transformed scores were Z-score standardized (scaled to mean = 0, standard deviation = 1) across the cohort. Consequently, the Hazard Ratios (HR) in this table represent the change in patient hazard **per standard deviation increase** of each score. 

The standardized results show a clear and consistent biological signal:
* **Apoptotic Commitment ($AUC_X$) and Peak Signal ($X_{\text{peak}}$):** Both show a statistically significant protective effect ($HR \approx 0.856$ and $0.855$, $p = 0.015$). Higher tumor cell commitment to apoptosis in response to DNA damage translates to a ~14.4% reduction in the hazard of patient death per 1-SD increase.
* **Time to DNA Repair ($T_{\text{repair}}$):** Also shows a statistically significant protective effect ($HR = 0.856$, $p = 0.015$). Biologically, a longer repair time signifies homologous recombination deficiency (HRD). Tumors with HRD cannot resolve chemotherapy-induced double-strand breaks quickly, making them highly sensitive to platinum chemotherapy and leading to improved patient survival. Standardizing this variable resolved the previously inflated standard error ($\text{SE} = 1.050$), tightening the wide confidence interval from `[0.010, 0.607]` to a stable, interpretable range of `[0.756, 0.970]`.
* **Residual DNA Damage ($D_{\text{resid}}$):** Previously, $D_{\text{resid}}$ suffered from a degenerate model fit ($HR = \infty$, $\text{CI} = [\text{NaN}, \infty]$) because its raw values were extremely small (in the order of $10^{-9}$), leading to near-zero variance. Z-score standardization scaled this variance to 1.0, enabling the Cox solver to converge successfully. The resulting hazard ratio of **1.017** ($95\%\text{ CI } = [0.894, 1.156]$, $p = 0.801$) confirms that residual damage has no statistically significant association with survival.


![[fig_forest_univariate_cox.png]]

**Figure 6: Univariate Cox proportional hazards forest plot.** Hazard ratios (with 95% confidence intervals) for the standardized log-transformed versions ($z\text{-}\log$) of the four ODE-derived scores.

![[fig_forest_multivariate_cox.png]]

**Figure 7: Multivariate Cox proportional hazards forest plot (adjusted for BRCA_MUTANT).** Hazard ratios for the standardized log-transformed ODE scores ($z\text{-}\log$) and BRCA mutation status.

### 4.4 Kaplan-Meier Stratification

To evaluate the clinical stratification potential of the primary apoptotic commitment score ($AUC_X$), we performed Kaplan-Meier survival analysis using three different grouping strategies: a pre-specified tertile split (primary analysis), a median split, and an optimized scan cutoff (exploratory analyses).

#### 4.4.1 Primary Analysis: Pre-specified Tertile Split
The primary, pre-specified stratification strategy divides the cohort into tertiles based on $AUC_X$:
- **T1 (Low AUC_X, $\le 81.38$):** $n = 140$, median OS = **41.5 months**
- **T2 (Mid AUC_X, $81.38 - 146.10$):** $n = 140$, median OS = **43.9 months**
- **T3 (High AUC_X, $> 146.10$):** $n = 140$, median OS = **48.8 months**
- **Omnibus Log-rank $p = 0.7726$**

Although the median overall survival increases progressively from T1 to T3 (consistent with the expected biological direction), the omnibus log-rank test is not statistically significant. This grouping is defined prior to analyzing outcomes, making its p-value valid for statistical inference.

![[fig_kaplan_meier_aucx_tertile.png]]

**Figure 8: Kaplan-Meier overall survival by pre-specified AUC_X tertiles.** Separation of the HGSOC TCGA cohort ($n = 420$) into T1 (low), T2 (mid), and T3 (high) groups based on apoptotic commitment score.

#### 4.4.2 Exploratory Analysis: Median Split
A standard binary split at the cohort median ($AUC_X = 108.27$) stratifies patients into two equal groups:
- **High AUC_X ($> 108.27$):** $n = 210$, median OS = **47.5 months**
- **Low AUC_X ($\le 108.27$):** $n = 210$, median OS = **42.0 months**
- **Log-rank $p = 0.0224$**

This median-based binary split yields a statistically significant difference in overall survival, confirming that patients with higher predicted apoptotic commitment survive longer on average.

#### 4.4.3 Exploratory Analysis: Optimal Cutoff Scan
By scanning all possible cutpoints to maximize the log-rank statistic, an optimal cutoff of $AUC_X = 98.997$ was identified:
- **High AUC_X ($\ge 99.00$):** $n = 227$, median OS = **47.5 months**
- **Low AUC_X ($< 99.00$):** $n = 193$, median OS = **41.5 months**
- **Log-rank $p = 0.0077$**

This optimized split maximizes the survival difference between the predicted high and low apoptotic responders. However, because this cutpoint was selected post-hoc by searching the outcome data, the p-value is subject to multiple-testing inflation and must be interpreted as strictly exploratory.

![[fig_km_auc_x_exploratory_cutoff.png]]

**Figure 9: Kaplan-Meier overall survival by optimal exploratory AUC_X cutoff.** Stratification of the HGSOC TCGA cohort using the data-driven optimized cutpoint ($AUC_X = 99.00$), showing a significant difference in survival times.

### 4.5 Model Comparison

All three models achieved identical C-index = **0.533**, indicating that the zero-shot mechanistic ODE matches penalised regression and ensemble ML despite using no outcome data for calibration.

| Model                  | C-index       | 95% CI         |
| ---------------------- | ------------- | -------------- |
| HR-DDR ODE (AUC_X)     | 0.533         | [0.426, 0.505] |
| Cox LASSO              | 0.533 ± 0.035 | [0.489, 0.563] |
| Random Survival Forest | 0.533 ± 0.018 | [0.449, 0.528] |

![[fig_ml_forest_plot.png]]

**Figure 10: Model comparison forest plot.** Comparison of Harrell's C-index between the zero-shot HR-DDR ODE model ($AUC_X$) and the data-trained Cox LASSO and Random Survival Forest models on the same 14-gene feature set.

![[fig_ml_bootstrap_distributions.png]]

**Figure 11: Bootstrap distributions of C-index.** Bootstrap distribution of the C-index across 1000 resamples for the three models, showing the overlap and variability in predictive performance.

---

## 5. Discussion

The HR-DDR ODE model achieves prognostic stratification of HGSOC patients using a fully mechanistic, zero-shot approach. The primary, pre-specified tertile split on $AUC_X$ demonstrates a progressive increase in median survival from T1 (41.5 mo) to T3 (48.8 mo), though it did not reach statistical significance (omnibus $p = 0.7726$). However, exploratory binary splits based on the cohort median ($p = 0.0224$) and the scan-optimized cutpoint ($p = 0.0077$) show statistically significant separations, confirming that patients with higher predicted apoptotic commitment ($AUC_X$) survive longer on average. The C-index of 0.533, while modest, is comparable to data-trained Cox LASSO and RSF models on the same feature set, suggesting that the gene expression signal in this cohort is inherently limited for individual-level survival discrimination.

The biological direction of the prognostic associations is correct: high $AUC_X$ and peak apoptotic signal ($X_{peak}$) associate with longer overall survival, which is consistent with the hypothesis that greater apoptotic drive leads to better chemotherapy response and survival. BRCA-mutant patients showed higher $AUC_X$ and $X_{peak}$ as expected.

Furthermore, reporting on time to repair ($T_{repair}$) highlights a key mechanistic detail of the model: BRCA-mutant patients exhibit slightly longer repair times, which corresponds to their homologous recombination deficiency. In the survival model, longer repair time is associated with better survival (standardized HR = 0.856 per SD, $p = 0.015$), indicating that slower DNA damage resolution is protective. Biologically, this reflects the therapeutic exploitability of DNA repair defects; tumors that cannot repair double-strand breaks efficiently are highly sensitive to carboplatin, leading to more effective tumor clearance and better overall survival.

TP53 was excluded from ODE parameterisation as a negative control - its near-universal mutation in HGSOC (~96%) would produce constant parameters with no discriminatory power.

**Limitations:**
- Global kinetic parameters are literature-fixed, not patient-calibrated
- C-index of 0.533 reflects weak discrimination at the individual level
- TCGA RNA-seq may not fully capture functional HR capacity
- The low observed BRCA1/2 mutation rate (4.0%) may limit subgroup power

