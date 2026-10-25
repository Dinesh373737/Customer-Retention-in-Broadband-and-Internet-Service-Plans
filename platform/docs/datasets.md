# Dataset sources, timing and licenses

## Iranian Churn

Source: [UCI Iranian Churn](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset), DOI [10.24432/C5JW3Z](https://doi.org/10.24432/C5JW3Z). License: CC BY 4.0. Attribution: Iranian Churn [Dataset] (2020), UCI Machine Learning Repository.

Actual downloaded CSV: **3,150 rows, 14 columns including Churn**. Names include repeated spaces: `Call  Failure`, `Subscription  Length`, `Charge  Amount`. There is no customer ID in this downloaded version. The source describes predictors aggregated over the first nine months and churn at month 12, leaving a three-month planning gap. `Status` means active/non-active; it is excluded conservatively despite the aggregate timing statement, rather than assuming a status measure is always safe.

`Charge  Amount` is described as ordered bands 0–9, but the actual file also contains 10. The example explicitly treats charge codes as unordered categories, with its pricing semantic mapping unconfirmed. No rows are silently removed to hide this inconsistency. Age/Age Group are excluded. Exact duplicates removed during training are disclosed in model metadata. This is compatibility validation on that historical sample, not current Iranian provider performance.

## Bank Customer Churn

Source: [Shantanu Dhakad's Bank Customer Churn Prediction dataset](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction). Kaggle's downloaded metadata reports **Other (specified in description)** licensing. Inspect the current source card before redistribution. Raw files are **not committed**.

Actual `Churn_Modelling.csv`: **10,000 rows, 14 columns**, target `Exited` with 0/1 labels. `CustomerId`, `RowNumber` and `Surname` are not predictors. The conservative benchmark also excludes age, gender, geography and salary. Six remaining business fields are explicitly configured. `IsActiveMember` supports an account-engagement review when evidence qualifies; it does not establish dissatisfaction, financial difficulty or a causal benefit of contact. Prediction horizon and observation dates are not established by this file.

## E-commerce Customer Churn

Source: [Ankit Verma's Ecommerce Customer Churn Analysis and Prediction dataset](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction). Downloaded metadata reports **CC BY-NC-SA 4.0**. This is an academic, noncommercial analysis; consult the source and license before reuse. Raw workbook files are **not committed**.

Actual workbook `E Commerce Dataset.xlsx` has `Data Dict` and `E Comm` sheets. The real customer table is **5,630 rows, 20 columns** on **E Comm**. Seven numerical columns contain missing values; trained pipelines impute them using training-only statistics. CustomerID is excluded. Gender, marital status and city tier are excluded in the conservative benchmark. SatisfactionScore is explicitly ordered 1–5. The data dictionary describes a complaint in the last month, app usage, order recency and past cashback; none supplies causal intervention outcomes or a verified prediction horizon.

## Acquisition and manual fallback

`platform/scripts/download_benchmarks.py` downloads only the named official UCI/Kaggle sources, validates archive types/size, selects safe CSV/XLSX filenames and records hashes. Network/authentication errors remain errors; no synthetic substitute is generated. Files go in ignored `platform/benchmarks/data/{iranian,banking,ecommerce}/` directories.

If public Kaggle download requires authentication in your environment, open the source link in your own account, download the archive and place its original file in the corresponding directory. Do not put credentials in source, browser forms controlled by this application or Git. The benchmark paths and worksheet are declared in `platform/benchmarks/configurations.py`. Uploading through Dataset Studio avoids editing Python files for ordinary analyses.

The original IBM CSV and documentation stay in the root `data/` and `docs/`. A separate generic IBM run uses its own selected predictors and artifacts; original case-study results are not relabelled as generic-platform results.
