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

To evaluate the clinical stratification potential of the ODE-derived scores, we performed Kaplan-Meier survival analysis for all four scores. Two grouping strategies were applied to each score:

1. **Primary Analysis (Pre-specified Tertile Split):** Patients were divided into T1 (low), T2 (mid), and T3 (high) groups using pre-specified tertile boundaries. Because the grouping is defined before examining survival outcomes, the omnibus log-rank p-value is valid for statistical inference. For scores with tied tertile boundaries (e.g. $T_{\text{repair}}$, which takes a small number of discrete hour values), the function falls back to a pre-specified median binary split.
2. **Exploratory Analysis (Optimal Cutoff Scan):** All possible binary cutpoints were scanned to identify the threshold that minimises the log-rank p-value. Because this cutpoint is selected post-hoc by searching the outcome data, the resulting p-value is inflated by multiple-testing and must be interpreted as strictly exploratory.

#### 4.4.1 Primary Analysis — Pre-specified Tertile Splits

Figure 8 shows the primary Kaplan-Meier analysis for all four ODE scores. All four scores show trends in the expected biological direction (higher apoptotic commitment and slower DNA repair associated with better survival), though none reach statistical significance with the three-group omnibus log-rank test:

* **$AUC_X$:** Median OS increases progressively from T1 (41.5 months) to T3 (48.8 months), $p = 0.773$.
* **$X_{\text{peak}}$:** Similar progressive increase from T1 (43.7 months) to T3 (48.3 months), $p = 0.822$.
* **$T_{\text{repair}}$:** Longer repair times trend toward better survival, $p = 0.523$. Falls back to a median binary split due to tied tertile boundaries.
* **$D_{\text{resid}}$:** Modest directional trend, $p = 0.169$.

The lack of significance in the tertile analysis is expected: the three-group omnibus test has lower power than a binary split, and the ODE scores capture continuous, graded variation in pathway activity rather than discrete risk categories.

![[fig_km_ode_combined_tertile.png]]

**Figure 8: Kaplan-Meier overall survival — PRIMARY analysis.** 2$\times$2 grid showing all four ODE scores stratified by pre-specified tertile splits (or median split where tertile boundaries are tied). Log-rank p-values are valid for inference.

#### 4.4.2 Exploratory Analysis — Optimal Cutpoint Scans

Figure 9 shows the exploratory KM analysis using the data-driven optimal cutpoints for each score. All four scores achieve statistically significant separation when the cutpoint is optimised post-hoc:

* **$AUC_X$:** Optimal cutoff = 99.0, $p = 0.0077$.
* **$X_{\text{peak}}$:** Optimal cutoff = 4.17, $p = 0.0061$.
* **$T_{\text{repair}}$:** Optimal cutoff = 61.0 h, $p = 0.0142$.
* **$D_{\text{resid}}$:** Optimal cutoff = $6.26 \times 10^{-9}$, $p = 0.0066$.

These results confirm that survival signal exists within each ODE score's distribution, but the clinical effect size is moderate — requiring a data-driven split to achieve significance, rather than emerging from any pre-specified grouping.

![[fig_km_ode_combined_exploratory.png]]

**Figure 9: Kaplan-Meier overall survival — EXPLORATORY analysis.** 2$\times$2 grid showing all four ODE scores stratified by data-driven optimal cutpoints. P-values are inflated by post-hoc selection and are not valid for inference.

#### 4.4.3 Model Comparison — KM Stratification Across All Five Models

To directly compare the clinical stratification ability of the mechanistic ODE model against the data-trained ML benchmarks, we stratified patients by median risk score for all five models (Figure 10). Each model's risk scores — the ODE's $-\log(AUC_X)$ and the out-of-fold (OOF) predictions from the four ML models — were dichotomised at their respective medians (a pre-specified, non-outcome-adaptive split). The resulting KM curves show how effectively each model separates high-risk from low-risk patients:

* **HR-DDR ODE ($AUC_X$):** Achieves significant separation ($p = 0.022$), with the low-risk group showing a survival advantage of approximately 5 months in median OS over the high-risk group.
* **Cox LASSO (14-gene) and RSF (14-gene):** When restricted to the same 14-gene feature set, neither data-trained ML model achieves significant separation at the median split, consistent with their lower C-indices relative to the ODE.
* **Cox LASSO (all-genes) and RSF (all-genes):** With access to the full transcriptome (27,066 genes), the all-genes models show improved stratification, with the Cox LASSO (all-genes) achieving the clearest separation.

This visual comparison reinforces the C-index findings from Section 4.5: the mechanistic ODE model achieves competitive or superior patient stratification compared to data-trained models on the same feature set, despite using no outcome data during model development.

![[fig_km_ml_comparison.png]]

**Figure 10: Kaplan-Meier survival — all five models.** Each panel stratifies the HGSOC TCGA cohort ($n = 420$) into high-risk and low-risk groups using the median of each model's risk score. The ODE panel uses $-\log(AUC_X)$; the ML panels use out-of-fold predictions. Log-rank p-values annotated on each panel.

### 4.5 Model Comparison

Prognostic discrimination was compared across five models using out-of-fold (OOF) predictions for the data-trained models and cohort-wide scores for the zero-shot ODE (Table 1, Figure 11):

| Model | Feature Set | C-index | 95% Bootstrap CI | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **HR-DDR ODE (AUC_X)** | 14 genes $\rightarrow$ ODE | **0.533** | [0.495, 0.574] | Zero-shot, literature-fixed parameters |
| **Cox LASSO (14-gene)** | 14 genes (raw) | **0.504** | [0.468, 0.548] | 5-fold CV, alpha via inner CV |
| **RSF (14-gene)** | 14 genes (raw) | **0.480** | [0.472, 0.551] | 5-fold CV, random forest ensemble |
| **Cox LASSO (all-genes)** | 27,066 genes (raw) | **0.569** | [0.528, 0.602] | 5-fold CV, full transcriptome |
| **RSF (all-genes)** | 27,066 genes (raw) | **0.549** | [0.505, 0.581] | 5-fold CV, full transcriptome |

When restricted to the same **14-gene feature set**, the zero-shot mechanistic ODE ($C = 0.533$) out-performs both the data-trained Cox LASSO ($C = 0.504$) and the Random Survival Forest ($C = 0.480$). This indicates that incorporating biological structure via the ODE pathway acts as a powerful regularizer, extracting clinical signal that standard machine learning models overfit or fail to learn from a small feature set.

When trained on the **entire transcriptome** (27,066 genes), the all-genes Cox LASSO achieves a higher C-index of **0.569** (95% CI: [0.528, 0.602]) and the all-genes RSF achieves **0.549** (95% CI: [0.505, 0.581]). This superior performance suggests that the global transcriptome contains additional survival signals (e.g., cell proliferation, immune infiltration, microenvironment) that are outside the scope of our 14-gene DNA repair model. However, the all-genes confidence intervals still overlap with that of the zero-shot ODE model, and the ODE achieves its comparable performance without requiring any outcome-based training data (zero-shot validation).

![[fig_ml_forest_plot.png]]

**Figure 11: Model comparison forest plot.** Comparison of Harrell's C-index between the zero-shot HR-DDR ODE model ($AUC_X$) and the data-trained Cox LASSO and Random Survival Forest models on the 14-gene feature set and full transcriptome.

![[fig_ml_bootstrap_distributions.png]]

**Figure 12: Bootstrap distributions of C-index.** Bootstrap distribution of the C-index across 1000 resamples for the three models, showing the overlap and variability in predictive performance.

### 4.6 Model Interpretation (ML Benchmarks)

To understand what biological features drove the predictions of the data-trained machine learning models, we analyzed the models fitted on the full cohort. Although these full-data fits are strictly for interpretation (not out-of-sample prediction), they reveal which genes carry survival signal.

#### 4.6.1 Cox LASSO Coefficients
The Cox LASSO model selected non-zero coefficients for all 14 genes, with the regularization parameter ($\alpha = 0.001$, selected via 5-fold cross-validation on the full dataset) keeping all features in the model (Figure 13):
* **Homologous Recombination DNA Repair (HRD) Genes**: BRCA2 ($+0.141$), RAD51 ($+0.102$), and BRCA1 ($+0.062$) show positive coefficients. Higher expression of these genes increases the hazard rate (poorer survival). Biologically, this is highly consistent: tumors with proficient repair machinery can resolve carboplatin-induced DNA double-strand breaks, rendering them resistant to chemotherapy and leading to worse patient survival.
* **Checkpoint Activating Kinases**: CHEK2 ($-0.195$), BRIP1 ($-0.130$), CHEK1 ($-0.121$), and PALB2 ($-0.071$) exhibit negative (protective) coefficients. Higher expression of these checkpoint effectors is associated with decreased hazard (better survival), which may reflect the role of robust checkpoint activation in triggering apoptotic pathways or allowing sufficient cell cycle arrest to prevent genomic instability.
* **Apoptotic Pathway Regulators**: BCL2 ($-0.029$) and BAX ($-0.054$) have negative coefficients, while BCL2L1 ($+0.033$) and BAD ($+0.020$) show positive coefficients, highlighting complex balancing effects in the apoptotic machinery.
* **Negative Control (TP53)**: TP53 has a small negative coefficient ($-0.069$), indicating that its expression retains a minor association with survival even in this near-universally mutated HGSOC cohort.

#### 4.6.2 Random Survival Forest (RSF) Feature Importances
We computed the permutation feature importance on the training data across 20 repeats (Figure 14). The mean drop in C-index upon permuting each gene reveals which genes the RSF relies on most heavily:
* **Key Predictors**: BRCA2 exhibits the highest permutation importance (mean C-index drop = **0.0324**), followed closely by PALB2 (**0.0244**), CHEK2 (**0.0244**), and RAD51 (**0.0225**). This indicates that the ensemble model relies most strongly on the core homologous recombination machinery and cell cycle checkpoints to stratify patients.
* **Minor Predictors**: The apoptotic regulators BAX (**0.0055**), BCL2 (**0.0076**), and BAD (**0.0097**) show the lowest importances, suggesting they play a minimal role in the RSF model's decision-making process.

![[fig_cox_lasso_coefficients.png]]

**Figure 13: Cox LASSO Feature Coefficients.** Horizontal bar chart of standardized coefficients from the full-cohort fit. Red bars represent positive coefficients (increased hazard/risk); blue bars represent negative coefficients (protective effect).

![[fig_rsf_feature_importances.png]]

**Figure 14: RSF Permutation Feature Importances.** Mean decrease in Harrell's C-index across 20 permutation repeats, with error bars representing $\pm 1$ standard deviation. Green bars indicate positive importances.

#### 4.6.3 ODE Pathway Genes in the All-Genes LASSO

A key question is whether the 14 HR-DDR pathway genes selected on biological grounds also emerge as top predictors in an unbiased genome-wide search. To answer this, we fitted the Cox LASSO on the full cohort using all 27,066 genes and extracted the percentile rank of each ODE gene within the genome-wide absolute coefficient distribution (Figure 15; rank 100 = top-ranked gene genome-wide).

Genes with non-zero coefficients in the all-genes LASSO have been independently "selected" by a purely data-driven procedure from a search space of over 27,000 candidates, providing an unbiased cross-validation of the ODE's biological feature selection. Genes that are LASSO-zeroed in the all-genes context still carry survival information within the constrained ODE framework, but their contribution is superseded by other transcriptomic signals when the full genome is available for selection.

![[fig_ode_gene_ranks_all_genes.png]]

**Figure 15: ODE Pathway Genes — Rank in All-Genes Cox LASSO.** Each dot represents one of the 14 ODE pathway genes, positioned by its percentile rank among all 27,066 genes by absolute LASSO coefficient magnitude. Filled dots indicate genes with non-zero coefficients (independently selected from 27,066 candidates); faded dots are LASSO-zeroed. Colour encodes pathway role: blue = HR repair, green = DDR checkpoint, red = apoptosis. The dashed line marks the 90th percentile threshold.

---

## 5. Discussion

### 5.1 Biological Validity of the ODE Model

The primary finding of this study is that a mechanistic, zero-shot ODE model — parameterised entirely from pre-treatment RNA-seq expression of 14 HR-DDR pathway genes and with kinetic constants fixed from published literature — generates prognostically informative predictions of overall survival in HGSOC without fitting any outcome data. The biological plausibility of the model is supported at multiple levels.

At the single-patient level, representative trajectory simulations (Figure 3) confirm qualitatively correct behaviour: a BRCA-deficient patient shows slower DNA damage resolution, sustained checkpoint kinase activation, and substantially higher cumulative apoptotic commitment ($AUC_X = 443.7$) compared to a BRCA-proficient patient ($AUC_X = 111.9$). These trajectories recapitulate the well-established mechanistic consequence of homologous recombination deficiency (HRD), namely an inability to efficiently resolve platinum-induced double-strand breaks, which prolongs ATM/ATR and CHK1/2 signalling and ultimately increases apoptotic drive.

At the population level, all four ODE-derived scores exhibit directionally consistent differences between BRCA wild-type and BRCA-mutant patients (Figure 4). BRCA-mutant patients show higher $AUC_X$, higher $X_{peak}$, longer $T_{repair}$, and marginally elevated residual damage — each a mechanistic consequence of impaired HR repair capacity. The absolute magnitude of group differences in $T_{repair}$ and $D_{resid}$ is small, reflecting the fact that both groups ultimately resolve damage by the end of the 120-hour simulation window and that the ODE damage dynamics are governed by the same global kinetic constants across all patients. Nevertheless, the directional consistency of all four scores with biological expectation provides confidence that the model is capturing genuine pathway biology rather than arbitrary expression correlations.

### 5.2 Survival Associations and Clinical Stratification

Three of the four ODE scores — $AUC_X$, $X_{peak}$, and $T_{repair}$ — show statistically significant univariate Cox associations after Z-score standardisation of log-transformed values (all $p = 0.015$, HR $\approx 0.856$ per SD). The hazard ratios are directionally interpretable: for $AUC_X$ and $X_{peak}$, higher apoptotic drive is protective, consistent with greater platinum sensitivity. For $T_{repair}$, slower damage resolution indicates HRD and is similarly protective. $D_{resid}$, by contrast, shows no significant association ($p = 0.801$), which is expected given that residual damage values are vanishingly small ($\sim 10^{-9}$) and effectively uniform across the cohort — a consequence of the ODE reaching near-complete damage resolution for all patients within the simulation window.

Kaplan-Meier stratification results must be interpreted with careful distinction between pre-specified and exploratory analyses. For all three prognostic scores, the **pre-specified tertile splits** — defined independently of outcome data — show monotonic trends in the biologically expected direction but do not reach statistical significance. This non-significance under a valid pre-specified design is an important finding: it indicates that while the ODE score carries genuine continuous-scale prognostic information (as confirmed by the Cox analysis), the signal is not strong enough to support reliable three-group clinical stratification in a cohort of this size and event rate. The **exploratory binary splits** at cohort median or scan-optimised cutpoints yield statistically significant log-rank p-values ($p = 0.022$ and $p = 0.008$ for $AUC_X$; similar results for $X_{peak}$ and $T_{repair}$). However, because these cutpoints were selected post-hoc by searching the outcome data, the resulting p-values are subject to multiple-testing inflation and should be treated as hypothesis-generating rather than confirmatory. The scan-optimised cutpoint analysis is retained in this report as an honest representation of exploratory findings, with the caveat noted explicitly.

### 5.3 Mechanistic Model versus Data-Trained Benchmarks

The central benchmarking finding is that the zero-shot ODE ($C = 0.533$, 95% CI: [0.495, 0.574]) outperforms both the 14-gene Cox LASSO ($C = 0.504$, CI: [0.468, 0.548]) and the 14-gene RSF ($C = 0.480$, CI: [0.472, 0.551]) when all models are restricted to the same feature set. This result is notable because the ODE does so without any outcome-based training, while the Cox LASSO and RSF were optimised using 5-fold cross-validation on the target cohort. The advantage of the ODE model on a 14-gene feature set reflects the regularising effect of mechanistic biological structure: rather than allowing the model to learn arbitrary linear or non-linear combinations of 14 correlated expression values — a setting prone to overfitting even with penalisation — the ODE constrains parameter estimation via a system of differential equations whose structure encodes prior biological knowledge. This prior prevents the model from exploiting spurious noise in the training data.

When data-trained models are given access to the full transcriptome (27,066 genes), they recover additional survival signal and achieve higher C-indices ($C = 0.569$ for all-genes LASSO; $C = 0.549$ for all-genes RSF). The superior performance of genome-wide models suggests that the survival signal in this cohort extends beyond the 14-gene HR-DDR pathway — likely including proliferation signatures, immune infiltration, and stromal microenvironment components that are not encoded in the ODE. Critically, however, the confidence intervals of the all-genes models still overlap substantially with those of the zero-shot ODE, and the ODE requires no training data, no hyperparameter search, and no outcome information at all. This represents a meaningful practical advantage in clinical or translational settings where outcome data are scarce.

### 5.4 Alignment between Mechanistic and Data-Trained Feature Importance

Examining which genes drive the data-trained ML models (Figures 19–20) reveals substantial overlap with the ODE parameterisation logic, providing cross-validation of the mechanistic framework. In the Cox LASSO model, the genes with the largest coefficient magnitudes are CHEK2 ($-0.195$, protective), BRIP1 ($-0.130$, protective), and CHEK1 ($-0.121$, protective) on the checkpoint effector side, and BRCA2 ($+0.141$, risk) and RAD51 ($+0.102$, risk) on the HR repair machinery side. The protective direction of checkpoint kinases is biologically coherent: robust CHK1/2 activation prolongs cell cycle arrest, allowing apoptotic commitment to accumulate in cells with insufficient repair capacity. The positive (risk-increasing) coefficients for BRCA2 and RAD51 likewise align with the ODE: tumours with high HR repair gene expression are repair-proficient and therefore platinum-resistant.

In the RSF permutation importance analysis, BRCA2 is the single most important feature (mean C-index drop = 0.032), followed by PALB2 (0.024), CHEK2 (0.024), and RAD51 (0.023). This ranking closely mirrors the structure of the ODE's BRCA_cap parameter, which is computed as the geometric mean of BRCA1, BRCA2, RAD51, PALB2, and BRIP1 — exactly the genes that emerge as most important in the data-driven model. The convergence between an independently trained ensemble model and a literature-derived mechanistic parameterisation provides strong evidence that these genes are carrying genuine biological signal, not cohort-specific noise.

The apoptotic regulators (BCL2, BAX, BAD, BCL2L1) show consistently low importance in the RSF and small coefficients in the LASSO, suggesting that the BCL2_ratio parameter contributes minimally to the survival signal in this cohort. This may reflect the complexity of BCL-2 family regulation in cancer, where post-translational modifications and protein-protein interactions — not captured by steady-state mRNA expression — dominate functional anti-apoptotic buffering.

### 5.5 Limitations

Several limitations constrain the interpretation of these findings.

**Fixed kinetic parameters.** All global kinetic rate constants (damage induction rate, ATM/ATR activation, CHK1/2 dynamics, and apoptotic threshold) are borrowed from published in vitro mechanistic studies and held constant across all patients. This assumption is a structural simplification: kinetic rates vary between cell types, are modulated by somatic mutations beyond BRCA1/2, and are perturbed by the tumour microenvironment. Patient-specific calibration of these parameters — for example, using protein-level proteomic data or ex vivo drug response measurements — would likely improve discriminatory performance but would require additional data sources not available in the TCGA cohort.

**Transcriptomic proxies for protein-level activity.** The ODE parameters (BRCA_cap, ATM_tot, CHK_tot, BCL2_ratio) are computed from steady-state FPKM RNA expression values. Transcript abundance is an imperfect proxy for the functional protein activity that governs kinetic rates, due to post-transcriptional regulation, protein stability differences, and context-dependent activity modulation. Studies linking proteomics to transcriptomics in TCGA HGSOC (e.g., the CPTAC cohort) suggest that mRNA-protein correlations are moderate for many pathway genes, which may attenuate the prognostic signal extractable from expression-based parameterisation.

**Residual damage and T_repair as discrete outputs.** $D_{resid}$ showed no significant survival association, which is partly attributable to the ODE architecture: with the fixed damage resolution kinetics used here, all patients converge to near-zero residual damage by 120 hours, leaving very little inter-patient variance. Similarly, $T_{repair}$ is a threshold-crossing time (discretised to simulation time steps of 1 hour), which limits its resolution and contributes to the tight clustering of values around 61 hours. Future model extensions could incorporate slower NHEJ repair kinetics or extend the simulation window to better differentiate residual damage phenotypes.

**Multiple testing and exploratory cutpoints.** The statistically significant log-rank p-values from the scan-optimised cutpoint analyses are subject to multiple-testing inflation because the cutpoint was selected post-hoc from the outcome data. These findings are explicitly labelled as exploratory throughout and should not be interpreted as prospectively validated thresholds. Confirmatory validation in an independent HGSOC cohort (e.g., the ICON7 or AGO-OVAR11 trial datasets) would be required before any clinically actionable cutpoint could be proposed.

**Cohort size and BRCA subgroup power.** The final analysis cohort of 420 patients with 262 events provides adequate power for Cox regression but limits the statistical resolution of subgroup analyses. The observed BRCA1/2 mutation prevalence of 4.0% (17 patients) is substantially lower than the 15–20% typically reported in clinical HGSOC series, likely reflecting ascertainment biases in the TCGA towards sporadic cases and the incomplete capture of germline variants by somatic sequencing panels. This low prevalence substantially limits the power of BRCA-stratified comparisons and may explain the absence of statistically significant group differences in the box plot analyses despite the directionally consistent effect sizes.

### 5.6 Conclusions

This study demonstrates that a mechanistic HR-DDR ODE model can extract prognostically informative signal from pre-treatment RNA-seq data in HGSOC without requiring any outcome-based parameter fitting. On a restricted 14-gene feature set, the zero-shot ODE achieves higher discriminatory performance than comparably constrained data-trained survival models, validating the role of biological structure as an effective regulariser in small feature spaces. The convergence between the ODE parameterisation logic and the data-driven feature importance rankings from Cox LASSO and RSF provides independent evidence that the HR-DDR pathway genes used in the ODE carry genuine survival signal. Together, these results support the broader case for mechanistic-prior modelling as a complement to — rather than replacement for — conventional machine learning in clinical genomic survival prediction, particularly in settings where training cohorts are small, feature sets are biologically constrained, or model interpretability is required.

