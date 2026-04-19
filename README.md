# PRT564 Assessment 2 — ACNC Charity Register Regression Analysis

**Group 2 — Charles Darwin University — Semester 1 2026**
**Unit:** PRT564 Data Analytics and Visualisation

Regression analysis of the Australian Charities and Not-for-profits Commission (ACNC) Charity Register, investigating which operational and geographic factors predict charity governance capacity — with a particular focus on the Northern Territory.

---

## Research Question

Which operational and geographic factors most strongly predict the governance capacity of an Australian charity — measured by the number of responsible persons on its board — and do Northern Territory charities differ systematically from the national baseline?

**Target variable:** `Number_of_Responsible_Persons` (continuous)
**Method:** Multiple linear regression (OLS, Ridge, Lasso) with cross-validation and statistical testing

---

## Team

| Member | Role |
|--------|------|
| Van Hoi Dang | Data Acquisition Lead |
| Rudraksh Patel | Preprocessing Specialist |
| Harsh Rastogi | Analysis Lead |
| Rochak Bhusal | Visualisation & Reporting Lead |

---

## Datasets

Both datasets are publicly available from [data.gov.au](https://data.gov.au).

| Dataset | Records | Variables | Purpose |
|---------|---------|-----------|---------|
| ACNC Charity Register | 65,114 | 69 | Primary dataset — operational attributes, purposes, beneficiary groups, jurisdictions |
| ACNC 2023 Annual Information Statement (AIS) | — | — | Supplementary — financial fields (Total Revenue, Total Expenses, Net Assets, employee counts) |

**Join strategy:** Left join on Australian Business Number (ABN). Left join was chosen deliberately over inner join to preserve charities without AIS returns, since smaller and newer charities are systematically more likely to miss AIS filings. An inner join would have biased the sample toward larger, better-resourced organisations.

---

## Pipeline Overview

```
Raw Register (65,114)
    ↓
State standardisation       → 58,235 valid records
    ↓
Binary encoding (47 cols)   → Y/blank → 1/0
    ↓
Feature engineering         → +5 derived features
    ↓
Target cleaning             → Remove zero-target (303) + top 1% outliers
    ↓
AIS join on ABN (left)      → Financial variables added
    ↓
Final modelling sample      → 57,847 records, 27 features
    ↓
Train / test split (80/20)  → 46,277 train, 11,570 test
    ↓
StandardScaler              → Coefficients comparable in magnitude
    ↓
Model training              → OLS + Ridge + Lasso
    ↓
Evaluation                  → 10-fold CV + paired t-tests + residual diagnostics
```

---

## Engineered Features

| Feature | Description |
|---------|-------------|
| `Num_States_Operated` | Count of states/territories each charity operates in |
| `Num_Purposes` | Count of charitable purposes declared |
| `Num_Beneficiaries` | Count of beneficiary groups served |
| `Is_NT` | Binary flag for Northern Territory operation |
| `Size_Encoded` | Ordinal encoding of charity size (XS → XL) |

---

## Models Trained

| Model | R² | RMSE | MAE | 10-Fold CV R² |
|-------|------|--------|--------|----------------|
| OLS Linear Regression | 0.1067 | 2.7562 | 2.1638 | 0.1091 ± 0.010 |
| Ridge (α = 1.0) | 0.1067 | 2.7562 | 2.1638 | 0.1091 ± 0.010 |
| Lasso (α = 0.1) | 0.0946 | 2.7749 | 2.1963 | 0.0971 ± 0.009 |

**Why three models?** To triangulate rather than cherry-pick. OLS is the unconstrained baseline. Ridge (L2) hedges against multicollinearity. Lasso (L1) performs automatic feature selection. Running all three and comparing them statistically is methodologically stronger than selecting a single model.

---

## Statistical Tests

| Test | Statistic | p-value | Conclusion |
|------|-----------|---------|------------|
| Paired t-test: OLS vs Ridge residuals | t = −0.46 | 0.654 | Fail to reject H₀ — models indistinguishable |
| Paired t-test: OLS vs Lasso residuals | t = 9.24 | < 0.001 | Reject H₀ — OLS significantly better |
| Independent t-test: NT vs Non-NT charities | t = 5.008 | < 0.001 | NT charities have significantly more responsible persons (5.73 vs 5.11) |

---

## Model Diagnostics

- **Durbin–Watson:** 1.98 (no autocorrelation — residuals are independent)
- **Shapiro–Wilk:** p < 0.001 (non-normal residuals — expected given discrete target and large n)
- **Lasso feature selection:** 18 of 27 features shrunk to zero

---

## Top 5 Features (by |standardised coefficient|)

1. `Size_Encoded` — 0.8147 (dominant predictor — larger charities have larger boards)
2. `Advancing_Culture` — 0.1962
3. `Advancing_Security` — 0.1837
4. `Operates_in_QLD` — −0.1745 (Queensland charities run leaner boards on average)
5. `Advancing_Health` — 0.0972

---

## Key Finding

The register does not describe a uniform sector. Northern Territory charities — 1.5% of the national sector, but 76% serving Aboriginal and Torres Strait Islander communities — have statistically significantly larger governance boards than the national average, and this effect persists after controlling for charity size. This is not a compliance burden to be reduced; it reflects cultural governance structures that require broader community representation. A single national compliance framework systematically misfits remote and Indigenous-led organisations.

---

## How to Run

### Requirements

```
Python 3.9+
pandas
numpy
scikit-learn
scipy
statsmodels
matplotlib
seaborn
```

### Installation

```bash
pip install pandas numpy scikit-learn scipy statsmodels matplotlib seaborn
```

### Execution

```bash
python regression_analysis.py
```

The script outputs model coefficients, evaluation metrics, statistical test results, and saves all evaluation charts to the `charts/` directory.

---

## Repository Structure

```
.
├── regression_analysis.py          # Main regression pipeline
├── README.md                       # This file
├── data/
│   ├── acnc_register.csv           # Primary dataset (not committed — download from data.gov.au)
│   └── acnc_ais_2023.csv           # Supplementary dataset (not committed — download from data.gov.au)
└── charts/
    ├── eda_target_distribution.png
    ├── eda_correlation_heatmap.png
    ├── eda_boxplot_size.png
    ├── eda_scatter_beneficiaries.png
    ├── eda_nt_comparison.png
    ├── eval_residual_analysis.png
    ├── eval_model_comparison.png
    ├── eval_feature_importance.png
    ├── eval_actual_vs_predicted.png
    └── eval_cv_boxplot.png
```

---

## Limitations

- **R² of 0.107 is low.** The target is a discrete integer with high organisational variance not captured by the public register. Variables likely to explain more (charity age, founder relationships, constitutional structure, governance history) are not in the available data.
- **Linear models underfit discrete targets.** A count model (Poisson or negative binomial) would be a better distributional fit and is flagged as future work.
- **NT sample is small** (n = 867), which limits the statistical power of NT-specific sub-analyses.

---

## Future Work

1. Fit a Poisson or negative binomial regression to test whether the distributional mismatch changes the coefficient structure.
2. Run an `Is_NT × Size_Encoded` interaction model to isolate the NT effect from pure size effects.
3. Qualitative validation with Aboriginal-controlled organisations to test the cultural-governance interpretation before any policy recommendation is operationalised.

---

## AI Usage Reflection

This project used AI tools (primarily Claude) for code scaffolding, debugging preprocessing logic, and drafting documentation structure. All modelling decisions, statistical interpretation, stakeholder framing, and final writing were done by the team. Every AI-generated code snippet was reviewed line-by-line against the dataset behaviour before inclusion. The GitHub commit history reflects team authorship. AI was treated as a productivity tool, not an analytical authority.

---

## Data Sources and Attribution

- Australian Charities and Not-for-profits Commission (ACNC). *Charity Register*. data.gov.au. Available at: https://data.gov.au
- Australian Charities and Not-for-profits Commission (ACNC). *2023 Annual Information Statement*. data.gov.au. Available at: https://data.gov.au

Data used under the terms of the Creative Commons licences specified on data.gov.au.

---

## License

This repository is submitted as academic coursework for PRT564 at Charles Darwin University. Code is provided for assessment purposes.
