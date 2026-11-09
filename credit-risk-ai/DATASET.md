# German Credit dataset and preprocessing

## Source and inspected file

FairCredit uses the original categorical/symbolic `german.data` file from the [UCI Statlog German Credit dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data), authored by Hans Hofmann (1994), DOI [10.24432/C5NC77](https://doi.org/10.24432/C5NC77). UCI lists the dataset under CC BY 4.0; attribution is retained here. The acquired UCI source file is bundled at `data/german.data` so training and audits work offline.

The source file was inspected directly before preprocessing changes. It contains 1,000 rows, 21 whitespace-separated fields (20 predictors and the target), with no malformed-width rows. The UCI feature count is 20. The original source target values are `1=good` and `2=bad`; FairCredit normalizes them to `0=good` and `1=bad`.

| Target | Meaning | Rows | Share |
| --- | --- | ---: | ---: |
| 0 (source 1) | Good credit label | 700 | 70% |
| 1 (source 2) | Bad credit label | 300 | 30% |

There are zero missing cells in the acquired source file and zero exact duplicate records. The pipeline still imputes missing feature values so compatible user-supplied files can be processed. Rows without a valid target cannot be used for supervised fitting and are dropped; exact duplicates are detected and one copy is retained before splitting.

## Feature descriptions and source encodings

The source documentation describes 7 numeric and 13 categorical predictors. Numeric values and category codes are retained as provided; category strings are one-hot encoded by the fitted training pipeline.

| Source field | Type | Description / observed values |
| --- | --- | --- |
| `checking_status` | Categorical | Checking account: A11 below 0 DM; A12 0–200 DM; A13 200 DM or more or salary assignment; A14 no checking account. |
| `duration` | Numeric | Loan duration in months (4–72). |
| `credit_history` | Categorical | A30 no credit/all paid; A31 all at this bank paid; A32 existing credits paid duly; A33 past delay; A34 critical account/other credits. |
| `purpose` | Categorical | A40 new car; A41 used car; A42 furniture/equipment; A43 radio/TV; A44 domestic appliance; A45 repairs; A46 education; A47 vacation (defined in source documentation, absent here); A48 retraining; A49 business; A410 other. |
| `credit_amount` | Numeric | Requested credit amount in DM (250–18,424). |
| `savings_status` | Categorical | A61 under 100 DM; A62 100–500; A63 500–1,000; A64 at least 1,000; A65 unknown/no savings account. |
| `employment` | Categorical | A71 unemployed; A72 under 1 year; A73 1–4 years; A74 4–7 years; A75 at least 7 years. |
| `installment_rate` | Numeric | Installment rate as a percentage band of disposable income (1–4). |
| `personal_status` | Categorical, sensitive | Personal status and sex combined: observed A91 male divorced/separated; A92 female (source combines multiple marital states); A93 male single; A94 male married/widowed. The codebook also defines A95 female single, but it is absent from these records. |
| `other_debtors` | Categorical | Other debtors/guarantors: A101 none; A102 co-applicant; A103 guarantor. |
| `residence_since` | Numeric | Years at current residence (1–4). |
| `property` | Categorical | A121 real estate; A122 building society savings/life insurance; A123 car/other property; A124 unknown/no property. |
| `age` | Numeric, sensitive | Age in years (19–75). |
| `other_installment_plans` | Categorical | A141 bank; A142 stores; A143 none. |
| `housing` | Categorical | A151 rent; A152 own; A153 rent-free. |
| `existing_credits` | Numeric | Number of existing credits at this bank (1–4). |
| `job` | Categorical | A171 unemployed/unskilled non-resident; A172 unskilled resident; A173 skilled employee/official; A174 management/self-employed/highly qualified. |
| `num_dependents` | Numeric | People the applicant is liable to maintain (1–2). |
| `telephone` | Categorical | A191 none; A192 registered in applicant's name. |
| `foreign_worker` | Categorical, sensitive | A201 yes; A202 no. |
| `credit_risk` | Target | Source 1 good / 2 bad; normalized to 0 / 1. |

Observed categorical codes are: checking A11–A14; history A30–A34; purpose A40–A46, A48–A49, A410; savings A61–A65; employment A71–A75; personal status A91–A94; debtors A101–A103; property A121–A124; plans A141–A143; housing A151–A153; job A171–A174; telephone A191–A192; foreign-worker A201–A202. A47 and A95 have documented meanings but no examples in this file.

## Sensitive attributes and audit pathway

The source does not have a standalone sex column. `personal_status` combines sex and marital status, so it cannot support a clean, independent sex comparison. `age` is numeric and `foreign_worker` is binary in the source, but foreign-worker values are highly imbalanced (963 yes, 37 no). The app retains all three attributes in the data used for group audits, derives age bands for a separate comparison, and excludes them from all applicant-model predictors and the reweighing audit model's predictors.

Removing these columns does not remove proxy discrimination: financial and historical features can correlate with demographic attributes. The small foreign-worker group also makes estimates unstable. Group metrics are descriptive and not a legal or fairness certification.

## Preprocessing and leakage controls

1. Validate the expected source columns and normalize target labels so `1` always means the source's bad-credit class.
2. Detect exact duplicate rows and retain one copy. Inspect missing predictors; none occur in this bundled data. For uploaded data, missing numeric values receive training-set medians and categorical values receive training-set modes.
3. Make a stratified 80/20 train/test split with `random_state=42` before any fitted transformation.
4. Use a scikit-learn `ColumnTransformer` inside a `Pipeline`: median imputation and `StandardScaler` for numeric columns; most-frequent imputation and `OneHotEncoder(handle_unknown="ignore")` for categorical columns.
5. Fit the pipeline only on training rows. Scaling statistics, imputation values, and one-hot categories are therefore learned from training rows and then applied to test/applicant rows.

## Outlier investigation

An exploratory Tukey 1.5×IQR scan found 70 rows outside the duration fences (Q1=12, Q3=24 months), 72 outside the credit-amount fences (Q1=1,365.5 DM, Q3=3,972.25 DM), and 23 outside the age fences (Q1=27, Q3=42). These are not automatically erroneous values; amounts and durations have long right tails. Existing-credit and dependent counts also cross univariate IQR fences (6 and 155 rows respectively), because these are low-cardinality bounded counts and the dependent count is concentrated at 1. No rows are clipped or removed based on these flags. Numerical scaling occurs within the training pipeline.

## Feature engineering

The original notebook added three ratios. `credit_amount / age` and `duration / age` were removed: age-normalized borrowing and loan duration have no clear financial interpretation for this use. `credit_amount / duration` is retained as `principal_per_month_proxy`:

```text
principal_per_month_proxy = credit_amount / duration
```

It is the average requested principal per loan month, not the installment amount or an affordability measure: the data does not provide interest schedules, fees, or income. The feature is deterministically derived before splitting; it learns no statistics from held-out rows.

## Limitations

- The data is a small historical German sample from 1994 and may not represent current applicants, products, or populations.
- The binary target is a historical “good/bad” label, not a default probability, approval decision, or neutral ground truth. UCI documents an asymmetric cost matrix for this dataset; metric choice can change which errors appear most important.
- Important repayment-capacity variables such as income and a full debt-to-income measure are not supplied.
- Demographic coverage is limited and encoded in ways that prevent independent analysis of some attributes. Foreign-worker group counts are especially uneven.
- Model predictions do not establish causal explanations. Removing sensitive predictors does not remove correlated proxies.
