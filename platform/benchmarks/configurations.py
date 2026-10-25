"""Explicit benchmark mappings reviewed against the downloaded source schemas.

These mappings are editable examples, not universal semantic inference. They do
not alter the preserved IBM case study. External data files remain untracked.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def semantic(column, concept, meaning, actionable=False):
    return {"column": column, "concept": concept, "meaning": meaning,
            "confirmed": True, "actionable": actionable, "protected": False}


def configuration(name, df):
    if name == "iranian":
        target, identifier, domain = "Churn", None, "telecommunications"
        excluded = ["Status", "Age", "Age Group"]
        horizon = "Source: attributes aggregated during first 9 months; outcome at month 12; 3-month planning gap. Status excluded conservatively."
        categorical = ["Complains", "Tariff Plan", "Charge  Amount"]
        # UCI describes bands 0..9, but the actual file also contains code 10.
        # Keep codes unordered and withhold business actions until clarified.
        ordinal = {}
        semantics = [semantic("Complains", "support", "Recorded binary complaint indicator: 1 means a complaint, 0 means no recorded complaint."),
                     semantic("Subscription  Length", "lifecycle", "Total months of subscription recorded in the pre-outcome observation window."),
                     semantic("Charge  Amount", "pricing", "UCI documents bands 0..9, but code 10 is also present; unordered codes are retained pending clarification.") | {"confirmed": False},
                     semantic("Frequency of use", "engagement", "Total number of calls in the pre-outcome observation window."),
                     semantic("Tariff Plan", "contract", "Plan type: 1 pay-as-you-go, 2 contractual; no preference or intervention response is recorded.")]
    elif name == "banking":
        target, identifier, domain = "Exited", "CustomerId", "banking"
        excluded = ["RowNumber", "Surname", "Age", "Gender", "Geography", "EstimatedSalary"]
        horizon = None
        categorical, ordinal = ["HasCrCard", "IsActiveMember"], {}
        semantics = [semantic("IsActiveMember", "engagement", "Recorded account activity indicator, 1 active and 0 inactive; not a complaint or satisfaction measure."),
                     semantic("Tenure", "lifecycle", "Recorded length of the banking relationship; no verified intervention outcomes or prediction horizon are supplied.")]
    elif name == "ecommerce":
        target, identifier, domain = "Churn", "CustomerID", "ecommerce"
        excluded = ["Gender", "MaritalStatus", "CityTier"]
        horizon = None
        categorical = ["PreferredLoginDevice", "PreferredPaymentMode", "PreferedOrderCat", "Complain"]
        ordinal = {"SatisfactionScore": [str(i) for i in range(1, 6)]}
        semantics = [semantic("Complain", "support", "Recorded complaint indicator; a recorded complaint does not establish the cause of churn."),
                     semantic("SatisfactionScore", "satisfaction", "Recorded customer satisfaction score on the source's five-level scale."),
                     semantic("DaySinceLastOrder", "engagement", "Recorded number of days since last order; observed recency rather than an intervention effect."),
                     semantic("HourSpendOnApp", "engagement", "Recorded hours spent in the application; no causal effect of increasing usage is established."),
                     semantic("Tenure", "lifecycle", "Recorded customer tenure as documented in the workbook data dictionary.")]
    elif name == "ibm":
        target, identifier, domain = "Churn", "customerID", "telecommunications"
        excluded = ["gender", "SeniorCitizen", "Partner", "Dependents"]
        horizon = None
        categorical = [column for column in df if column not in [target, identifier, *excluded, "tenure", "MonthlyCharges", "TotalCharges"]]
        ordinal = {}
        semantics = [semantic("MonthlyCharges", "pricing", "Current subscribed monthly charges in source monetary units; excludes a claim about affordability.", True),
                     semantic("tenure", "lifecycle", "Number of months subscribed to the company."),
                     semantic("Contract", "contract", "Current recorded contract term; no preference or treatment response is known.", True),
                     semantic("TechSupport", "support", "Technical-support subscription category; absence of a subscription is not evidence of a support incident.")]
    else:
        raise ValueError("Unknown benchmark name.")
    features = []
    for column in df:
        if column in [target, identifier, *excluded]:
            continue
        if column in ordinal:
            features.append({"column": column, "kind": "ordinal", "ordinal_order": ordinal[column]})
        else:
            features.append({"column": column, "kind": "categorical" if column in categorical else "numeric"})
    return {"target_column": target, "positive_label": "Yes" if name == "ibm" else "1",
            "customer_id": identifier, "excluded_columns": excluded,
            "features": features, "business_domain": domain,
            "prediction_horizon": horizon, "semantics": semantics, "leakage_approved": {}}


DATASETS = {
    "iranian": {"path": ROOT / "benchmarks/data/iranian/Customer Churn.csv", "sheet": None},
    "banking": {"path": ROOT / "benchmarks/data/banking/Churn_Modelling.csv", "sheet": None},
    "ecommerce": {"path": ROOT / "benchmarks/data/ecommerce/E Commerce Dataset.xlsx", "sheet": "E Comm"},
    "ibm": {"path": ROOT.parent / "data/Telco-Customer-Churn.csv", "sheet": None},
}
