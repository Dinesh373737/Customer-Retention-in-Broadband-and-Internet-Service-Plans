# MASTER PROMPT: Customer Retention in Broadband and Internet Service Plans

## 1. Project Overview

Build a complete, working, interactive web application for a college Principles of Management and Economics (PoME) project titled:

**Customer Retention in Broadband and Internet Service Plans**

The application must function as a Customer Churn Analysis and Retention Decision Support System for a broadband or telecom company.

The objective is not merely to predict whether customers will churn. The system should help the company:

1. Understand customer churn patterns.
2. Identify customer segments with elevated churn risk.
3. Predict churn using a hybrid machine-learning approach.
4. Explain why the models flag customers as potential churners.
5. Recommend possible retention strategies based on observed data patterns.
6. Evaluate the potential economic benefits and costs of retention campaigns.
7. Present all findings through a professional, interactive dashboard.

This is an academic prototype using historical data. Do not misrepresent it as a production system connected to a live telecom provider.

---

## 2. Existing Dataset and Data Source

Use the IBM Telco Customer Churn dataset.

Official IBM repository:
https://github.com/IBM/telco-customer-churn-on-icp4d

Alternative dataset source:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The commonly used version contains approximately 7,043 customer records and 21 columns. Verify the actual dataset file and its schema before implementation.

The target variable is `Churn`.

Important available features include:

- `customerID`
- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`
- `tenure`
- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`
- `MonthlyCharges`
- `TotalCharges`

Requirements:

- Allow the user to upload the dataset or load it from a configurable local file.
- Include a fallback sample-data option only when the actual dataset file is unavailable, and label that data clearly.
- Do not silently generate synthetic data and present it as real customer data.
- Remove `customerID` from model predictors.
- Convert numeric fields correctly, especially `TotalCharges`, which can contain blank or whitespace values.
- Check missing values, duplicate records, inconsistent categories and target labels.
- Show a data-quality summary.
- Handle the class distribution appropriately.
- Document all preprocessing decisions.
- Do not assume every dataset version has the same schema or record count.

The findings must be described as findings from the IBM sample dataset. Do not claim they represent every broadband provider or that the dataset contains live customer activity.

---

## 3. Technology Stack

Use a practical Python-based architecture suitable for local development.

Recommended stack:

- Frontend and interactive dashboard: Streamlit
- Data processing: Pandas and NumPy
- Visualisation: Plotly
- Machine learning: Scikit-learn
- Third model: XGBoost
- Statistical analysis: SciPy and Statsmodels
- Model explanations: SHAP, where appropriate
- Persistence: joblib for trained models and preprocessing pipelines
- Optional local storage: SQLite for saved analyses or settings if persistence is required

Do not introduce unnecessary cloud services, Docker, complex microservices or paid infrastructure.

The application must run locally on a standard laptop.

Create a clear project structure with separate modules for:

- Data loading and validation
- Data preprocessing
- Exploratory data analysis
- Statistical analysis
- Model training and evaluation
- Hybrid ensemble
- Explainability
- Retention recommendations
- Economic analysis
- Dashboard pages
- Tests and configuration

Provide a `requirements.txt`, README and step-by-step setup and execution instructions.

---

## 4. Preserve the Existing Customer-Centric Workflow

Maintain the following workflow from the previous version. Improve its technical implementation without changing its core stages.

### Stage 1: Customer Uses Broadband or Internet Services

A customer subscribes to a broadband or telecom plan and uses services such as internet, phone or related subscription services.

In the real-world workflow, the company already has account and subscription information.

In our prototype, represent this stage using historical customer records from the IBM dataset.

### Stage 2: System Collects Customer Data

The system processes available information such as:

- Customer tenure
- Contract type
- Internet service category
- Monthly charges
- Total charges
- Payment method
- Technical support subscription
- Online security and other subscribed services

Show that the system uses existing account information rather than requiring customers to manually enter all these fields.

Only use variables available in the dataset or supplied by the user. Do not pretend that we have real-time network telemetry, customer satisfaction scores, complaint logs or internet-speed measurements.

### Stage 3: System Predicts Churn Risk

The system processes the cleaned customer data using trained machine-learning models.

For each eligible customer, calculate the probability of churn.

Display:

- Predicted churn probability
- Risk category
- Key factors contributing to the prediction
- A clear explanation of the prediction

### Stage 4: System Classifies Customer Risk

Classify customers into configurable risk categories.

Use a sensible initial threshold and allow adjustment based on validation results and business priorities.

A simple initial policy may be:

- Low risk: predicted probability below 30%
- Medium risk: 30% to below 60%
- High risk: 60% or higher

These are illustrative starting thresholds, not statistically established boundaries. Make them configurable, document them and explain how the classification changes when the threshold is adjusted.

Never confuse a high predicted probability with certainty that a customer will leave.

### Stage 5: System Suggests Personalised Retention Actions

For high-risk customers, display possible actions based on the observed patterns and the available data.

Examples:

- Review whether a suitable longer-term contract offer could be appropriate.
- Evaluate the pricing and service plan offered to the customer.
- Consider onboarding support for newer subscribers if analysis supports that recommendation.
- Consider technical-support outreach where the observed customer segment and available evidence make it relevant.
- Prioritise further customer engagement for selected at-risk groups.

Recommendations must have a visible rationale linked to observed features or segment-level results.

Do not assume that a customer with no technical support subscription has experienced a technical problem.

Do not automatically apply discounts or claim that an action will prevent churn.

Where several explanations are possible, communicate uncertainty.

### Stage 6: Customer Response

In a real telecom environment, a customer may accept an offer, change a plan, contact support or continue with the existing subscription.

The IBM dataset does not contain the outcomes of retention offers administered by our proposed system.

Therefore, represent customer response as a conceptual stage in the customer journey. Do not invent actual responses or claim that our system has contacted customers.

Where useful, provide a clearly labelled simulated demonstration of possible customer responses, separate from actual dataset results.

### Stage 7: Ongoing Monitoring and Learning

In a real deployment, new customer data and observed outcomes would be used to monitor model performance and periodically retrain it.

For our prototype:

- Allow users to re-run analysis on uploaded data.
- Provide a model-evaluation view.
- Explain how performance could be monitored after deployment.
- Show how new labelled data could support retraining.

Do not claim continuous monitoring or automatic learning from live customers unless it is genuinely implemented and connected to an appropriate data source.

Preserve this seven-stage workflow in the application and include a customer journey diagram explaining it.

---

## 5. Hybrid Machine-Learning Approach

Use all three requested models:

1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier

Implement them as individual models and as a hybrid ensemble where appropriate.

### A. Logistic Regression

Use Logistic Regression as the interpretable statistical and predictive baseline.

Requirements:

- Use an appropriate preprocessing pipeline.
- Encode categorical features.
- Scale numeric features where appropriate.
- Display coefficients and odds ratios where meaningful.
- Explain how model coefficients relate to the direction of predicted churn odds.
- Do not interpret an association as proof of causation.

### B. Random Forest

Use Random Forest to capture nonlinear relationships and feature interactions.

Requirements:

- Use a reproducible random seed.
- Tune a reasonable subset of hyperparameters using training data only.
- Display evaluation metrics and feature importance.
- Use suitable explanatory methods where appropriate.

### C. XGBoost

Use XGBoost as a gradient-boosted tree model.

Requirements:

- Configure the classifier for binary churn prediction.
- Handle class imbalance when justified by the training distribution.
- Tune a reasonable set of hyperparameters without excessive computation.
- Display evaluation metrics and feature explanations.
- Include understandable error handling if the package is not installed.

### D. Build an Actual Hybrid Ensemble

Do not simply train three models and call that a hybrid model.

Build an additional ensemble that combines the three models' predicted probabilities.

Use soft voting or a carefully designed weighted probability ensemble as the initial approach.

For example:

P(hybrid churn) = w1 × P(Logistic Regression) + w2 × P(Random Forest) + w3 × P(XGBoost)

where the non-negative weights sum to one.

Start with equal weights as a baseline. Any learned weights must be selected using training or validation data, never the test set.

If probabilities are poorly calibrated, evaluate calibration before combining them.

Compare the hybrid model with all three individual models to establish whether ensembling actually improves performance.

The application must be able to report that the hybrid is not the best model if the test results show otherwise. Never hardcode or fabricate a claim that the ensemble performs best.

### E. Data Leakage Prevention

Follow these rules strictly:

- Make a stratified train/test split.
- Reserve the test set for final evaluation.
- Fit imputers, encoders, scalers and other preprocessing only on the training data.
- Use pipelines wherever feasible.
- Perform hyperparameter selection and threshold selection using training/validation data or training-only cross-validation.
- Never use test-set information to train the models, select ensemble weights or choose the preferred model.
- Exclude `customerID` from predictors.
- Keep the target variable separate from input features.
- Explain the importance of leakage prevention in the README and project report.

Save reproducible model artifacts and the exact preprocessing pipelines required for inference.

---

## 6. Model Evaluation Dashboard

Create a dedicated Model Comparison page.

Compare:

- Logistic Regression
- Random Forest
- XGBoost
- Hybrid Ensemble

Display the following metrics where applicable:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix

Explain the practical meaning of the metrics.

In this project:

- Precision helps assess how many flagged customers actually churn in the historical test data.
- Recall measures how many actual churners the model identifies.
- F1-score summarises the balance between precision and recall.
- ROC-AUC measures ranking ability across thresholds.
- PR-AUC is particularly informative when positive and negative class frequencies are unequal.

Use the same held-out test set for the final comparison.

Show a table and comparison charts to make the differences clear.

Allow the user to choose a metric for ranking the models, but display all key metrics regardless of that choice.

Include a short plain-language interpretation explaining the trade-offs. Avoid claiming that a high score guarantees a profitable retention campaign.

---

## 7. Required Company Dashboard and Important Graphs

Build a polished, professional, company-facing dashboard with a clear layout, consistent styling and interactive controls.

Use Plotly for interactive charts wherever appropriate.

Every graph must have meaningful labels, readable legends, tooltips and clear interpretations. Avoid decorative charts that do not add analytical value.

### Page 1: Executive Overview

Show KPI cards for:

- Total customer records
- Number of churned customers
- Overall churn rate
- Number of non-churned customers
- Average tenure
- Average monthly charges

Use data-driven values. Never hardcode numbers that should be calculated from the uploaded file.

Required graphs:

1. Overall churn distribution.
2. Churn rate by contract type.
3. Churn rate by internet service category.
4. Churn rate by tenure group.
5. Monthly charges distribution, separated by churn status.

Place a short insight section below the charts. The insights must be generated from computed results, not predetermined assumptions.

### Page 2: Exploratory Data Analysis

Include:

1. Churn rate by contract type.
2. Churn rate by payment method.
3. Churn rate by InternetService.
4. Churn rate by TechSupport subscription.
5. Churn rate by OnlineSecurity subscription.
6. Churn rate by tenure bands.
7. MonthlyCharges box plot by churn status.
8. TotalCharges distribution by churn status.
9. Tenure versus MonthlyCharges scatterplot, coloured by churn.
10. Numeric-feature correlation heatmap where statistically appropriate.
11. Customer segment comparison chart.
12. Churn rate across SeniorCitizen, Partner and Dependents categories where relevant.

For tenure bands, use clearly defined, reproducible intervals and explain them.

For every categorical comparison, show both the churn rate and the sample size where practical. Avoid misleading comparisons between very small groups.

### Page 3: Model Comparison and Performance

Include:

1. Grouped or aligned bar chart comparing model metrics.
2. ROC curves for all four models.
3. Precision-recall curves for all four models.
4. Confusion matrices.
5. Threshold-versus-precision-and-recall chart.
6. Actual-versus-predicted class summary.
7. Predicted probability distribution for churners and non-churners in the held-out test set.

Use the same test records and appropriate probability scores for comparisons.

### Page 4: Feature Importance and Explainability

Include:

1. Logistic Regression coefficient chart, with a clear explanation of the direction and meaning.
2. Random Forest feature-importance chart.
3. XGBoost feature-importance chart.
4. SHAP global importance plot where feasible.
5. SHAP beeswarm or another suitable summary plot.
6. Individual-customer explanation showing the factors contributing to an individual score.

Be careful that feature importance is not interpreted as causality.

Ensure explanation functionality works with the model and preprocessing pipeline actually used by the application. If SHAP is unavailable or fails, provide a reliable fallback such as model feature importance and logistic regression coefficients. Clearly distinguish the fallback from SHAP.

### Page 5: Customer Risk Explorer

Create a filterable table of customer records showing:

- Anonymised customer reference or customer ID for navigation
- Tenure
- Contract type
- Internet service category
- Monthly charges
- Technical support subscription
- Actual churn label, if present in a historical analysis
- Predicted churn probability
- Risk category

Do not expose personally identifiable information. The dataset's customerID may be used to distinguish records in the dashboard but must not be used as a predictor.

Include:

- Filters for risk category.
- Filters for contract type.
- Filters for InternetService.
- Filters for tenure ranges.
- Sorting by predicted churn probability.
- A detail panel for a selected record.
- A way to export the filtered results as CSV.

For historical customers, clearly distinguish actual labels from predicted outcomes. In a new-customer scoring mode, actual churn labels may be unavailable.

### Page 6: Retention Recommendation Centre

Create a recommendation module connected to the exploratory findings and model outputs.

For a selected customer or segment, display:

- Predicted churn probability
- Risk category
- Key explanatory features
- Suggested next action
- Why the action is suggested
- Evidence supporting the recommendation
- Important caveats

Include a segment-level recommendation table with these columns:

Customer segment | Observed pattern | Possible action | Expected objective | Evidence and limitations

Examples of possible actions include reviewing contract suitability, investigating pricing options, improving onboarding and considering proactive technical-support outreach.

Generate recommendations from actual analysis results and transparent rules. Never fabricate customer complaints, preferences or reasons for cancellation.

Do not automatically offer a discount just because a customer has a high churn probability.

### Page 7: Economic Impact and Cost-Benefit Analysis

This page is important because the project is for Principles of Management and Economics.

Build an interactive economic scenario calculator.

Allow the user to enter:

- Number of customers to target or the proportion of customers targeted
- Cost per retention contact or offer
- Expected intervention success rate
- Estimated incremental contribution per retained customer over a stated period
- Any additional campaign costs
- Optional customer-segment value assumptions

Calculate and display:

- Number of customers targeted
- Total campaign cost
- Estimated number of additional customers retained
- Estimated incremental contribution
- Estimated net benefit
- Estimated return on investment
- Break-even success rate or break-even retained-customer value

Use formulas with clearly defined units and periods.

For example:

Campaign cost = Number targeted × Cost per targeted customer + Additional campaign costs

Estimated additional retained customers = Number targeted × Assumed incremental intervention success rate

Estimated net benefit = Estimated incremental contribution − Campaign cost

ROI = Estimated net benefit / Campaign cost × 100%, provided campaign cost is greater than zero.

Avoid counting customers who would have stayed anyway as customers retained because of the campaign. The scenario calculator must represent incremental retention rather than simply the predicted churn count.

All hypothetical assumptions must be editable and explicitly labelled.

Show sensitivity analysis by varying campaign cost, intervention success and retained-customer value. Add a line chart or heatmap of estimated net benefit under different assumptions.

Do not present hypothetical financial figures as real financial results. The IBM dataset does not provide retention campaign costs or causal intervention success rates.

### Page 8: Methodology and Limitations

Include a dedicated section explaining:

- Dataset source and schema
- Data-quality checks
- Feature engineering and preprocessing
- Model-training methodology
- Hybrid ensemble method
- Evaluation metrics
- Selected threshold and rationale
- Explanation methods
- Economic assumptions
- Dataset limitations
- Generalisability limitations
- Difference between predictive association and causal evidence

Make the section accessible to a professor who may not have an advanced machine-learning background.

---

## 8. Additional Useful Features

Include the following features if feasible without sacrificing correctness.

### A. Batch Customer Scoring

Allow a company analyst to upload a CSV containing customer records with the required input columns.

Validate the schema, run the saved preprocessing and model pipeline, and produce churn probabilities, risk categories and available explanations.

Allow the scored file to be downloaded.

Do not require a churn label in scoring mode.

### B. What-If Analysis

Let the user experiment with hypothetical changes to selected inputs, such as contract type or monthly charges, and observe how predicted risk changes.

Clearly label the result as a model-based scenario, not proof that making the change will prevent churn.

Make sure that modified values are valid combinations and that the system does not accidentally change immutable customer attributes.

### C. Business Filters

Provide global filters for contract type, internet service, tenure and other relevant categories. Ensure charts, summaries and tables update consistently when filters are changed.

Do not let dashboard filters change the held-out test set or alter reported model metrics in an invalid way.

### D. Downloadable Reports

Allow users to export:

- Filtered customer-risk records
- Model comparison metrics
- Key analytical findings
- Segment-level retention recommendations
- Economic scenario results

Include a concise report view summarising the selected dataset, model, threshold and assumptions.

### E. Model and Data Status

Show which dataset is currently loaded, when the current analysis was generated, which model was selected and whether the model artifacts are available.

Use actual system information. Do not display fake live status indicators or pretend to fetch real-time broadband data.

---

## 9. Design Requirements

The interface should look like a professional customer analytics and business intelligence platform.

Design direction:

- Clean and modern dashboard.
- Light background with restrained blue and teal accents.
- Consistent typography and spacing.
- Clear KPI cards.
- Interactive Plotly charts.
- Sidebar navigation.
- Professional tables with useful filters.
- Clear empty states and error messages.
- Responsive layout where possible.

Avoid clutter, unnecessary animations and decorative visual elements.

Use plain business language. Include tooltips or short help text for technical terms such as ROC-AUC, recall, probability and SHAP.

The dashboard must distinguish actual dataset results, model predictions and hypothetical business scenarios through explicit labels.

---

## 10. Statistical Analysis Requirements

Alongside machine learning, implement suitable statistical analysis to give the project a strong economics and research foundation.

Include:

- Descriptive statistics for important numeric variables.
- Group-level churn rates.
- Appropriate tests for categorical feature associations, such as chi-square tests where assumptions are satisfied.
- Appropriate comparisons of numeric features between churn groups.
- Effect sizes and confidence intervals where appropriate.
- Logistic regression odds ratios and confidence intervals, with explanations of the modelling assumptions.

Handle multiple testing thoughtfully if many feature tests are conducted.

Do not interpret statistical significance alone as proof of business importance.

Avoid claiming causation from this observational dataset.

---

## 11. Engineering and Reliability Requirements

The project must be a genuinely functional application, not just a UI mock-up.

Requirements:

- All charts must use actual computed data.
- Models must be trained or loaded from legitimate saved artifacts.
- Prediction probabilities must come from the fitted model pipelines.
- Dashboard controls must work.
- Filters must change the appropriate outputs.
- CSV export must produce valid files.
- Missing data and schema errors must be handled gracefully.
- Repeated training should be reproducible where possible.
- Expensive computation should be cached appropriately without returning stale results for a different dataset.
- Never silently substitute random or hardcoded values if model training fails.
- Do not hide exceptions that are necessary for debugging.
- Use type-appropriate preprocessing and reliable probability handling.
- Include basic unit tests for preprocessing, training, inference, evaluation and the economic formulas.

Add validation checks for data leakage and ensure no target labels or identifiers leak into feature matrices.

Do not overbuild. Focus on correct analysis, useful decisions and a coherent end-to-end demonstration.

---

## 12. Expected Final Deliverables

Produce:

1. A complete working application.
2. A documented project folder structure.
3. All necessary source files.
4. A requirements file.
5. A README with setup and launch instructions.
6. A data dictionary and explanation of preprocessing.
7. A trained-model persistence and loading mechanism.
8. Model-comparison results generated from the actual dataset.
9. Interactive dashboard graphs.
10. Explainability outputs.
11. A customer risk explorer with CSV export.
12. A retention recommendation module.
13. An economic scenario calculator.
14. A customer-centric workflow diagram.
15. Basic automated tests.
16. A concise project report outline covering objectives, literature motivation, methodology, results, economics, limitations and future work.

Do not invent final accuracy, ROC-AUC, recall or financial benefit values before running the pipeline.

---

## 13. Implementation Plan

Build and verify the application in the following order:

**Phase 1 — Inspect and validate the data**
- Confirm the actual file and column schema.
- Implement cleaning, preprocessing and data-quality reports.
- Generate exploratory analysis from actual records.

**Phase 2 — Train and compare models**
- Implement Logistic Regression, Random Forest and XGBoost.
- Establish fair evaluation with a held-out test set.
- Implement and evaluate the hybrid ensemble.
- Save model artifacts.

**Phase 3 — Implement explainability and recommendations**
- Generate global and individual explanations.
- Develop transparent recommendation rules.
- Connect recommendations to observed findings.

**Phase 4 — Build the dashboard**
- Create the executive dashboard and EDA pages.
- Add model comparison and explainability.
- Add risk explorer, filters and CSV export.
- Implement the economic scenario calculator.

**Phase 5 — Test the full system**
- Verify the entire data-to-prediction workflow.
- Test missing or invalid input data.
- Check for data leakage.
- Verify each chart, prediction and economic calculation.
- Test that the app runs locally from the README instructions.

If a required feature cannot be implemented reliably, explain the limitation and provide a clearly labelled fallback rather than simulating a successful implementation.

---

## 14. Final Acceptance Criteria

The project is complete only when:

- The application launches locally.
- The actual IBM dataset can be loaded and validated.
- Logistic Regression, Random Forest and XGBoost train and produce valid predictions.
- A genuine hybrid probability ensemble is trained or assembled and evaluated.
- All models are compared fairly on the same held-out test set.
- Metrics and graphs are calculated from real model outputs.
- The customer risk explorer works.
- Explanations are available with a reliable fallback.
- Recommendations are grounded in the observed data.
- Economic calculations use transparent, editable assumptions.
- The complete seven-stage customer journey is represented.
- The project states clearly what is historical analysis, what is prediction and what is hypothetical.
- The README allows another student to install, run and verify the application.

**Start by inspecting the existing project files and available dataset before writing code. Then implement the project in phases, testing each phase before proceeding. Preserve working functionality when making changes and do not replace real functionality with a static demonstration.**