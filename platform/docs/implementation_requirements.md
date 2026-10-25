# MASTER IMPLEMENTATION PROMPT — RETENTION LAB 2.0

## Role and objective

You are Astra, acting as a senior full-stack engineer, machine-learning engineer, data scientist and UI/UX architect.

Your task is to upgrade an existing customer-churn analytics project into a reusable, cross-industry Customer Retention Intelligence Platform.

**Important: This is an implementation task. Deliver working, tested code—not just architecture diagrams, explanations, mockups or a project plan.**

### Existing GitHub repository

https://github.com/Dinesh373737/Customer-Retention-in-Broadband-and-Internet-Service-Plans

Repository: `Dinesh373737/Customer-Retention-in-Broadband-and-Internet-Service-Plans`

Default branch: `main`

First inspect the actual repository, including its README, existing modules, dependencies, tests, saved artifacts and source code.

Create a separate feature branch for the upgrade. Do not directly replace the existing project.

---

# 1. CRITICAL REQUIREMENT: PRESERVE IBM TELCO

The existing project, Retention Lab, is a Streamlit-based Customer Churn Analysis and Retention Decision Support System developed for the IBM Telco Customer Churn dataset.

It already implements:

- Logistic Regression
- Random Forest
- XGBoost
- Equal-weight hybrid probability ensemble
- Statistical analysis
- Exploratory Data Analysis
- Model comparisons and evaluations
- Feature importance and explainability
- Customer risk exploration
- Retention recommendations
- Economic impact analysis
- CSV export
- Automated tests

Existing important modules include:

- `app.py`
- `retention/data.py`
- `retention/preprocessing.py`
- `retention/modeling.py`
- `retention/ensemble.py`
- `retention/evaluation.py`
- `retention/explainability.py`
- `retention/recommendations.py`
- `retention/economics.py`
- `retention/dashboard/`
- `tests/`

**Do not remove, rewrite, degrade or replace the existing IBM Telco system.**

Preserve its:

1. Original dataset and documentation.
2. Streamlit application.
3. Existing models and evaluation methodology.
4. Working graphs and dashboards.
5. Retention recommendations.
6. Economic analysis.
7. Existing tests and run instructions.
8. Existing analysis results and artifacts.

The original application must remain independently executable using its current documented commands.

It is acceptable to reuse ideas or extract independent utilities through new modules, but do not introduce breaking dependencies into the legacy application.

Do not move or overwrite the original dataset or trained artifacts.

The IBM implementation is now the preserved **IBM Telco Case Study**.

The new platform must coexist with it.

---

# 2. NEW PLATFORM OBJECTIVE

Build a generic customer-churn analysis platform that accepts different structured datasets with different:

- Numbers of columns
- Feature names
- Numeric attributes
- Categorical attributes
- Target column names
- Missing-value patterns
- Class distributions
- Business domains

The system should train dataset-specific machine-learning models and generate:

1. Data-quality reports.
2. Exploratory analysis.
3. Statistical insights.
4. Churn-risk predictions.
5. Model comparisons.
6. Explainability reports.
7. Segment-level risk analysis.
8. Individual customer risk drivers.
9. Evidence-based retention recommendations.
10. Economic and campaign analysis.
11. Downloadable results.

The platform must not assume that every dataset uses IBM's column names.

### Supported scope

Initially support structured, customer-level tabular datasets containing a binary churn or cancellation outcome.

Support CSV files and Excel workbooks (`.xlsx`) where feasible.

Do not claim support for arbitrary raw text, images, time-series event streams or datasets without a meaningful churn definition.

Distinguish three operating modes:

**Mode A — Historical labelled analysis**

The dataset contains customer features and historical churn outcomes.

The platform can train models, evaluate performance, analyse historical trends and generate evidence-linked recommendations.

**Mode B — Unlabelled batch scoring**

The dataset lacks churn outcomes but conforms to an existing trained model's saved schema.

The platform can produce churn-risk scores and applicable explanations.

**Mode C — Exploratory-only analysis**

The dataset has no churn labels and no compatible trained model.

The platform can inspect quality and provide descriptive analytics, but must not invent predictions or model performance.

---

# 3. RECOMMENDED SYSTEM ARCHITECTURE

Use a modular monolithic architecture.

### Frontend

- React
- TypeScript
- Vite
- Tailwind CSS
- shadcn/ui
- Lucide icons
- Recharts for standard analytics
- Plotly.js where advanced interactive statistical charts are appropriate
- TanStack Query for API data fetching and caching

### Backend

- Python
- FastAPI
- Pydantic
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SciPy
- Statsmodels
- SHAP
- Joblib

### Persistence

Use SQLite for:

- Uploaded dataset metadata
- Schema configurations
- Dataset fingerprints
- Analysis configurations
- Training-job status
- Model-version metadata
- Saved recommendations
- Economic scenario configurations

Store dataset files and trained models in dedicated local directories.

Use a lightweight local background-job mechanism for expensive training. Do not block HTTP requests until training finishes.

Do not introduce Docker, Redis, Celery, cloud infrastructure, paid APIs or microservices unless genuinely required.

### Required project organisation

Create a separate platform directory with a structure conceptually similar to:

```text
platform/
    backend/
        app/
            api/
            core/
            schemas/
            services/
                data_engine/
                analytics/
                modeling/
                explainability/
                recommendations/
                economics/
            persistence/
            workers/
        tests/
        requirements.txt

    frontend/
        src/
            components/
            pages/
            features/
            hooks/
            services/
            types/
            styles/
        package.json

    docs/
    README.md
```

Adapt the structure where necessary, but keep the responsibilities separate.

Do not duplicate large amounts of legacy code unnecessarily.

---

# 4. GENERIC DATASET ENGINE

This is one of the most important components.

Implement the following workflow:

### Step 1 — Dataset upload

Allow users to upload CSV or Excel files.

Display:

- Filename
- Row count
- Column count
- File size
- Detected column names
- Inferred data types
- Missing-value statistics
- Duplicate statistics
- Class balance when the target is identified

Enforce reasonable file-size limits and provide clear errors for corrupt or unsupported files.

Never execute user-uploaded files.

### Step 2 — Schema detection

Automatically suggest whether each column is:

- Numeric
- Categorical
- Binary
- Date/time
- Identifier
- Potential target
- Potentially sensitive
- Unsupported or ambiguous

Detect likely identifier columns and suspicious leakage variables.

Automatic inference must remain editable.

### Step 3 — User-confirmed feature mapping

Provide a configuration wizard.

Allow the user to select:

1. Target column.
2. Positive churn label.
3. Customer identifier column, if present.
4. Columns to exclude.
5. Feature data types.
6. Prediction horizon, if known.
7. Business domain.
8. Optional business meanings of important columns.

Example mappings:

```text
IBM Telco:
    Churn -> Target
    customerID -> Identifier

Banking:
    Exited -> Target
    CustomerId -> Identifier
    Surname -> Excluded

E-Commerce:
    Churn -> Target
    CustomerID -> Identifier
```

Never automatically assume that the last column is the target.

Never use target labels or identifiers as model predictors.

### Step 4 — Dynamic preprocessing

Construct preprocessing pipelines using the actual validated schema.

Requirements:

- Numeric missing-value imputation.
- Categorical missing-value handling.
- Numeric scaling when appropriate.
- Categorical encoding.
- Unknown-category handling.
- Reasonable high-cardinality safeguards.
- Explicit treatment of ordinal columns.
- Basic date feature handling only when meaningful.
- Detection of invalid, constant and suspicious columns.

Fit all learned preprocessing transformations using training data only.

Use Scikit-learn Pipeline and ColumnTransformer.

Store the exact preprocessing pipeline alongside each trained model.

### Step 5 — Data-quality and leakage checks

Detect:

- Missing or invalid targets.
- Duplicate records.
- Missing identifier values.
- Highly imbalanced classes.
- Near-constant features.
- Possible post-outcome features.
- Potential target proxies.
- Dataset sizes insufficient for reliable evaluation.
- Categories appearing only in validation or scoring data.

Warn users when a prediction horizon cannot be established.

For longitudinal datasets, use an appropriate temporal or group-aware split when supported.

Do not silently drop problematic columns or rows without reporting what happened.

---

# 5. GENERIC MACHINE-LEARNING ENGINE

Retain the original four-way comparison:

1. Logistic Regression.
2. Random Forest.
3. XGBoost.
4. Hybrid probability ensemble.

For each new dataset, train separate models.

Do not reuse the IBM model on unrelated schemas.

### Training methodology

Implement:

- Reproducible random seeds.
- Appropriate train/validation/test partitioning.
- Stratified splits where justified.
- Training-only preprocessing.
- Training-only hyperparameter tuning.
- Class-imbalance handling where needed.
- Probability calibration evaluation.
- Validation-based model selection.
- Held-out final evaluation.

For the hybrid:

\[
P_{\text{hybrid}} =
w_1P_{\text{LR}}+
w_2P_{\text{RF}}+
w_3P_{\text{XGB}}
\]

Use equal weights as an initial baseline.

If optimising weights, use validation data only.

Evaluate calibration before treating model outputs as churn probabilities.

### Required metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix
- Brier score
- Probability calibration plots

Handle metrics that become undefined for extremely small or single-class partitions.

Allow the user to choose the metric used to rank models.

Do not hardcode a preferred model.

The hybrid is not automatically considered superior.

Save model artifacts together with dataset fingerprints, feature mappings, target mappings and training metadata.

Show training progress, completion status and real errors in the interface.

Never display fabricated model metrics.

---

# 6. DYNAMIC EXPLORATORY DATA ANALYSIS

Build an EDA engine that generates relevant visualisations based on column types.

Do not hardcode IBM features into the generic dashboard.

For numeric columns, support:

- Distribution plots
- Histograms
- Box plots by churn outcome
- Missing-value analysis
- Correlation heatmaps
- Suitable feature-vs-target comparisons

For categorical columns, support:

- Category distributions
- Churn rate by category
- Category sample sizes
- Confidence intervals when appropriate
- Comparisons across selected customer segments

For selected feature pairs, support appropriate scatterplots, grouped summaries or relationship charts.

Handle high-cardinality features by grouping or limiting the displayed categories.

Provide filters and descriptive explanations.

Generate data-driven observations from computed statistics.

Do not force every possible chart onto the page.

Prioritise useful, interpretable visualisations.

---

# 7. EXPLAINABLE AI ENGINE

Provide both global and individual explanations.

### Global explanations

- Logistic Regression coefficients where meaningful.
- Odds ratios where statistically appropriate.
- Random Forest feature importance.
- XGBoost feature importance.
- SHAP global importance when supported.
- SHAP summary visualisations where supported.
- Feature-level statistical associations.

### Individual explanations

For a selected customer, show:

- Predicted churn probability.
- Risk category.
- Most influential features.
- Direction of their contribution where valid.
- Available customer-level feature values.
- Explanation limitations.

Ensure explanations correspond to the model actually used.

Provide safe fallbacks when SHAP is unavailable.

### Critical scientific requirement

Prediction explanations must never be described as established causal explanations.

Distinguish:

1. Predictive drivers.
2. Historical statistical associations.
3. Experimentally or causally supported intervention effects.

A model saying that contract type is associated with churn does not establish that changing the contract will prevent churn.

---

# 8. ADVANCED GENERIC RECOMMENDATION ENGINE

This is the key new feature.

The current IBM recommendation implementation relies on five telecom-specific segments.

Develop a separate generic recommendation framework.

It must work through the following stages.

### Stage A — Detect elevated-risk customers and segments

Use:

- Predicted churn probability.
- Global model explanations.
- Individual explanations.
- Historical segment churn rates.
- Segment sample sizes.
- Appropriate statistical evidence.

### Stage B — Map dataset features to business concepts

Create an editable business-semantic mapping system.

Examples:

| Feature names | Business concept |
|---|---|
| MonthlyCharges, SubscriptionFee, PlanPrice | Pricing |
| Tenure, SubscriptionLength, AccountAge | Customer lifecycle |
| SupportTickets, Complaints, TechSupport | Support experience |
| WeeklyLogins, UsageFrequency, LastActive | Customer engagement |
| Contract, RenewalType | Contract terms |
| SatisfactionScore, CustomerRating | Customer satisfaction |

Mappings must be based on column meanings, not only fuzzy similarity of names.

If a mapping is uncertain, request confirmation.

Unknown features may still participate in prediction, but should not generate invented business recommendations.

### Stage C — Recommendation knowledge base

Create a configurable catalogue containing:

- Business concept
- Supported industry
- Trigger conditions
- Suggested intervention
- Intended objective
- Supporting evidence requirements
- Actionability constraints
- Possible costs
- Important caveats

Support at least:

1. Telecommunications.
2. Banking.
3. E-commerce.
4. Generic subscription businesses.

Examples of possible recommendations:

**Telecommunications**

- Review plan suitability.
- Offer onboarding assistance.
- Investigate support requirements.
- Review contract alternatives.

**Banking**

- Review account engagement.
- Investigate service suitability.
- Offer an appropriate customer-support review.
- Consider targeted engagement.

**E-commerce**

- Investigate reported complaints.
- Improve engagement.
- Review customer experience.
- Consider suitable loyalty or re-engagement campaigns.

These must not be unconditional hardcoded outputs.

Only recommend an action when the relevant features, segment evidence and business context support considering it.

Avoid discriminatory recommendations based on immutable or protected customer characteristics.

### Stage D — Recommendation prioritisation

For each recommendation, display:

- Customer or segment identifier.
- Churn probability.
- Observed risk drivers.
- Proposed retention action.
- Why the action was suggested.
- Supporting statistical evidence.
- Sample size.
- Expected business objective.
- Important uncertainties.
- Estimated cost, when supplied.
- Evidence status.

Do not invent service failures, customer complaints, financial difficulties or personal preferences.

If evidence is insufficient, the system must say so.

In such cases, a recommendation to investigate or gather feedback is more appropriate than asserting a specific cause.

### Stage E — Optional what-if analysis

Support hypothetical changes to actionable features.

Show the model's predicted risk under the modified input.

Clearly state that changing an input and observing a lower model prediction does not prove the intervention will reduce actual churn.

Do not suggest impossible changes to immutable attributes.

Consider using DiCE for constrained counterfactual explanations if feasible.

---

# 9. CAUSAL EVIDENCE AND RETENTION EFFECTIVENESS

A major goal is helping companies understand why customers leave.

However, historical churn datasets generally cannot prove why specific customers churned.

The system must explicitly distinguish:

- Model-associated risk factors.
- Observational relationships.
- Verified causes or intervention effects.

Do not introduce an LLM that invents explanations or pretends to establish causality.

Start with deterministic, transparent recommendation rules.

If optional natural-language summaries are implemented, they must only summarise structured, computed evidence.

Future causal or uplift modelling may be supported when suitable intervention datasets exist.

Do not estimate actual discount effectiveness or intervention uplift from ordinary churn labels alone.

---

# 10. ECONOMIC IMPACT AND RETENTION OPTIMISATION

Preserve the ideas in the original economic calculator and implement them in the generic platform.

Allow users to configure:

- Campaign budget.
- Number of target customers.
- Cost per intervention.
- Customer contribution margin.
- Retention value period.
- Assumed incremental retention impact.
- Fixed campaign costs.

Calculate:

- Total campaign expenditure.
- Expected incremental retention.
- Estimated incremental contribution.
- Estimated net benefit.
- ROI.
- Break-even intervention success rate.
- Sensitivity to uncertain assumptions.

Use:

\[
\text{Expected Net Value}
=
\text{Incremental Retention Probability}
\times
\text{Customer Contribution}
-
\text{Intervention Cost}
\]

Distinguish model predictions from user-supplied assumptions.

Do not multiply churn probability by an uplift estimate if the uplift already represents an unconditional incremental retention probability.

Provide an optional budget-constrained recommendation ranking using explicitly declared assumptions.

Do not present hypothetical financial values as observed results.

---

# 11. COMPLETE REACT UI REDESIGN

Build a polished professional SaaS-style dashboard.

The interface must be visually appealing, modern, responsive and genuinely functional.

### Design direction

- Light theme by default.
- Restrained navy, blue and teal palette.
- Professional typography.
- Consistent spacing.
- Rounded but restrained cards.
- Attractive KPI cards.
- Responsive sidebar navigation.
- Interactive graphs.
- Professional data tables.
- Clear loading indicators.
- Useful empty states.
- Toast notifications.
- Accessible form controls.
- Search, sorting and filtering.
- Clear separation between actual observations, predictions and scenarios.

Avoid:

- Streamlit-style default layouts.
- Cluttered pages.
- Excessive chart density.
- Decorative animations with no purpose.
- Fake status indicators.
- Placeholder graphs.
- Hardcoded analytical results.
- Nonfunctional buttons.

### Required screens

**Page 1 — Workspace Selection**

Two entry points:

- IBM Telco Case Study.
- New Generic Dataset Analysis.

The IBM option must preserve access to the existing Streamlit application rather than silently replacing it.

The new platform may launch or link to the existing local Streamlit interface.

**Page 2 — Dataset Studio**

- Drag-and-drop upload.
- Data preview.
- Schema inference.
- Target selection.
- Positive-class configuration.
- Feature mapping.
- Leakage warnings.
- Data-quality analysis.

**Page 3 — Executive Analytics**

- Customer statistics.
- Churn distribution.
- Segment comparisons.
- Computed key findings.
- Dynamic charts.
- Dataset filters.

**Page 4 — Model Lab**

- Training configuration.
- Training progress.
- Four-model comparison.
- Confusion matrices.
- ROC and PR curves.
- Calibration analysis.
- Model selection.

**Page 5 — Explainability**

- Global importance.
- SHAP plots when available.
- Individual customer explanations.
- Explanatory caveats.

**Page 6 — Customer Intelligence**

- Filterable customer table.
- Risk probabilities.
- Risk categories.
- Customer detail panel.
- Individual risk drivers.
- Batch-scoring support.
- CSV export.

**Page 7 — Retention Strategy Centre**

- Suggested interventions.
- Evidence-linked rationales.
- Customer and segment recommendations.
- Cost estimates.
- Campaign ranking.
- Economic impact calculator.
- Sensitivity analysis.

**Page 8 — Reports and Methodology**

- Dataset information.
- Preprocessing decisions.
- Training methodology.
- Model metrics.
- Recommendation methodology.
- Economic assumptions.
- Limitations.
- Downloadable reports.

All controls and charts must connect to real backend functionality.

---

# 12. API DESIGN

Design versioned FastAPI endpoints.

Implement the functional equivalent of:

```text
POST /api/datasets/upload
GET  /api/datasets/{id}/profile
POST /api/datasets/{id}/configuration

POST /api/analyses/train
GET  /api/jobs/{id}
GET  /api/analyses/{id}/overview
GET  /api/analyses/{id}/eda
GET  /api/analyses/{id}/models
GET  /api/analyses/{id}/explanations

GET  /api/analyses/{id}/customers
GET  /api/analyses/{id}/recommendations

POST /api/analyses/{id}/batch-score
POST /api/analyses/{id}/what-if
POST /api/analyses/{id}/economics

GET  /api/analyses/{id}/export
```

Adjust endpoint names if necessary.

Use typed request and response schemas.

Handle invalid inputs and background-job failures cleanly.

Avoid unnecessary communication overhead and repeated model loading.

Ensure every response belongs to the correct dataset and model version.

---

# 13. EXTERNAL DATASETS FOR VALIDATION

Do not validate generalisation using only IBM Telco.

Implement and test the generic platform on the following datasets.

### Dataset 1 — Iranian Telecom Churn

Official source:

https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset

Approximately 3,150 customer records and 13 features.

Target: `Churn`

Features include:

- Call Failure
- Complains
- Subscription Length
- Charge Amount
- Seconds of Use
- Frequency of use
- Frequency of SMS
- Distinct Called Numbers
- Age Group
- Tariff Plan
- Status
- Customer Value

Test:

- A different telecom schema.
- Numeric and ordinal attributes.
- Dynamic preprocessing.
- Target selection.
- Non-IBM charts.
- Recommendations based on available features.
- Appropriate prediction timing.

Inspect potentially leakage-prone variables such as `Status` before approving them as predictors. Verify their timing and meaning rather than automatically including them.

### Dataset 2 — Bank Customer Churn

Source:

https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction

Approximately 10,000 customer records.

Target: `Exited`

Features typically include:

- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary

Test:

- A non-telecom dataset.
- Different target name.
- Categorical and numerical processing.
- Financial customer segments.
- Semantic feature mapping.
- Banking-specific recommendation generation.

Exclude identifiers and names from predictors.

Do not assume a customer's income, age or location establishes the cause of churn.

### Dataset 3 — E-Commerce Customer Churn

Source:

https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction

Approximately 5,630 customer records.

Target: `Churn`

Features typically include:

- Tenure
- PreferredLoginDevice
- CityTier
- WarehouseToHome
- PreferredPaymentMode
- HourSpendOnApp
- NumberOfDeviceRegistered
- PreferedOrderCat
- SatisfactionScore
- MaritalStatus
- Complain
- OrderAmountHikeFromlastYear
- CouponUsed
- OrderCount
- DaySinceLastOrder
- CashbackAmount

Test:

- E-commerce customer behaviour.
- Mixed categorical and numerical variables.
- Missing values.
- Dynamic visualisations.
- Engagement and customer-experience recommendations.
- Feature-semantic mapping.

The downloaded source may be an Excel workbook; support it or document an explicit validated conversion to CSV.

### Dataset acquisition rules

Inspect the downloaded files and verify their actual schemas. The descriptions above are guidance, not permission to assume the contents without inspection.

Respect licenses and attribution requirements.

Do not commit externally sourced datasets unless redistribution is allowed.

If Kaggle authentication is required, provide a clear download and placement procedure.

Never replace an unavailable external dataset with synthetic data while reporting that the external benchmark passed.

---

# 14. AUTOMATED TESTING AND VERIFICATION

Test the existing IBM application before and after implementation.

Its original tests must continue passing.

For the generic platform, add comprehensive tests.

### Dataset tests

- Different feature counts.
- Different feature names.
- Different target column names.
- Different positive-class encodings.
- Missing numeric values.
- Missing categorical values.
- Duplicate records.
- Unseen categories.
- Unsupported target labels.
- Empty files.
- Incorrect file formats.
- No churn labels.
- Identifiers accidentally selected as features.
- Suspicious target-leakage columns.

### ML tests

- Correct dataset-specific preprocessing.
- Train/test separation.
- No preprocessing leakage.
- Reproducible training.
- Valid probability range.
- Calibration outputs.
- Model comparison.
- Valid ensemble combination.
- Saved-model loading.
- Correct dataset-model matching.
- No reuse of incompatible models.

### Explainability tests

- Valid explanation dimensions.
- Correct feature-name mapping.
- Correct customer-model association.
- Consistent interpretation of the selected model.
- Graceful fallback when SHAP is unavailable.

### Recommendation tests

- Correct feature-semantic mapping.
- Applicable actions for supported business concepts.
- No action invented for an unknown feature.
- No unsupported causal assertions.
- Correct segment evidence.
- Appropriate handling of small segments.
- No automatic recommendation solely from protected characteristics.
- Correct economic assumptions and formulas.

### Frontend and API tests

- Dataset upload.
- Schema configuration.
- Training progress.
- Analysis navigation.
- Filtering.
- Customer detail selection.
- Batch scoring.
- What-if analysis.
- Recommendations.
- Data export.
- Empty and failure states.

Use appropriate unit tests and at least a small number of genuine end-to-end tests.

Do not consider the project verified merely because it starts successfully.

### Required benchmark report

Produce a table for every tested dataset:

| Dataset | Rows | Features | Target | Selected model | ROC-AUC | PR-AUC | F1 | Recommendation status |
|---|---|---|---|---|---|---|---|---|

Populate the table using actual execution results only.

Include dataset-specific failures, limitations and any unsupported features.

Clearly distinguish successful runs from tests that could not execute.

---

# 15. DEVELOPMENT AND EXECUTION STRATEGY

Implement the project incrementally.

### Phase 1 — Repository inspection

Read the complete existing implementation.

Identify reusable Python code and hardcoded IBM-specific assumptions.

Run existing tests and record baseline results.

### Phase 2 — Generic backend

Implement:

- Data upload.
- Dataset profiling.
- Schema mapping.
- Dynamic preprocessing.
- Model training.
- Model persistence.
- Evaluation.

Verify using the non-IBM datasets before beginning extensive UI work.

### Phase 3 — Explainability and recommendations

Implement:

- Dataset-specific explanations.
- Semantic feature mapping.
- Evidence-based recommendation rules.
- Recommendation catalogue.
- Economic evaluation.

### Phase 4 — React frontend

Create the complete new dashboard.

Connect every page to the working backend.

Use actual data from API responses.

### Phase 5 — End-to-end verification

Run the complete workflows.

Fix failures.

Verify legacy compatibility.

Perform a final UI check at desktop and mobile widths.

### Phase 6 — Documentation

Provide:

- Complete setup guide.
- Windows PowerShell instructions.
- Separate legacy and generic-platform launch commands.
- Dataset download instructions.
- API documentation.
- New architecture overview.
- Schema mapping guide.
- Explanation and recommendation limitations.
- Testing instructions.
- Actual benchmark results.

Provide a convenient local launcher if practical.

Do not require users to manually edit Python source code to switch datasets.

---

# 16. NON-NEGOTIABLE ENGINEERING RULES

1. Do not break the existing IBM Telco application.
2. Do not replace actual ML predictions with sample values.
3. Do not fabricate metrics, graphs or evidence.
4. Do not train new datasets using an incompatible IBM-trained model.
5. Do not assume feature names or feature counts are fixed.
6. Do not use identifiers or outcomes as predictors.
7. Do not leak held-out test data into tuning or preprocessing.
8. Do not present predictive associations as proven causes.
9. Do not claim a recommended intervention will prevent churn without suitable evidence.
10. Do not automatically recommend discounts to every high-risk customer.
11. Do not infer customer complaints or dissatisfaction from unrelated attributes.
12. Do not calculate actual campaign profit from nonexistent intervention-outcome data.
13. Do not hardcode dataset-specific analytical values into the UI.
14. Do not add unnecessary infrastructure.
15. Do not silently catch failures and display success.
16. Do not remove existing tests or lower quality checks to obtain a passing build.
17. Keep the project practical enough to run on a typical college student's laptop.
18. Ensure all downloaded datasets and saved model artifacts have clear provenance.
19. Preserve the original repository's working code and history.
20. Prioritise functional correctness over visual polish when trade-offs arise, but deliver both.

---

# 17. FINAL ACCEPTANCE CRITERIA

The implementation is complete only when:

- The original IBM Telco Streamlit application still works.
- The new React application starts locally.
- The FastAPI backend starts locally.
- A user can upload an unfamiliar supported customer dataset.
- The system detects its columns.
- The user can configure the target and predictor roles.
- The system creates a valid dataset-specific preprocessing pipeline.
- Logistic Regression, Random Forest and XGBoost can be trained.
- The hybrid ensemble is genuinely evaluated.
- The application displays actual comparison metrics.
- EDA charts adapt to the uploaded features.
- Churn predictions come from fitted models.
- Global and individual explanations work or provide transparent fallbacks.
- The recommendation engine uses available evidence and semantic mappings.
- Unsupported recommendation claims are avoided.
- Economic calculations use explicit assumptions.
- Export functions produce valid outputs.
- Tests demonstrate compatibility with the required benchmark datasets.
- Existing IBM functionality remains intact.
- README instructions allow a fresh installation and reproducible demonstration.

## Required final response

At completion, provide:

1. A concise summary of what was implemented.
2. The final project structure.
3. What was reused versus newly developed.
4. Confirmation of legacy IBM compatibility, supported by actual tests.
5. Dataset-by-dataset benchmark results.
6. Recommendation-engine verification results.
7. Backend and frontend launch commands.
8. Remaining limitations.
9. Git branch and commit references.

If a dataset is inaccessible, a dependency fails, or a feature cannot be verified, disclose that explicitly.

Do not claim complete success unless the functionality has actually been tested.

**Begin by inspecting the GitHub repository and running the existing tests. Then implement the generic platform phase by phase. Do not stop after producing a plan.**
