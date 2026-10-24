# Generated historical analysis

Dataset: data/Telco-Customer-Churn.csv. Records: 7,043.
Generated: 2026-10-23T13:52:21.243456+00:00. Fingerprint: 663439c063b78e467836a119a38cd6a921823dabba3b3cb78176ab54023a4f91.

## Observed findings

- In this selected IBM sample, 1,869 of 7,043 customers (26.5%) are labelled as churned.
- Among contract groups with at least 30 records, Month-to-month has the highest observed churn rate (42.7%; n=3,875), compared with Two year (2.8%; n=1,695).
- Among internet-service groups with at least 30 records, Fiber optic has the highest observed churn rate (41.9%; n=3,096), compared with No (7.4%; n=1,526).
- Among tenure groups with at least 30 records, 0–12 months has the highest observed churn rate (47.4%; n=2,186), compared with 61+ months (6.6%; n=1,407).
- Average monthly charges are 74.44 for churned customers and 61.27 for non-churned customers, in the dataset's source currency units.
- These are historical associations within the selected sample; they do not establish why an individual left or whether an offer would change their decision.

## Held-out model comparison

```text
              model  accuracy  precision   recall       f1  roc_auc   pr_auc    brier
Logistic Regression  0.805536   0.657233 0.558824 0.604046 0.841486 0.635307 0.138215
      Random Forest  0.799858   0.659722 0.508021 0.574018 0.845320 0.654117 0.135119
            XGBoost  0.801987   0.656766 0.532086 0.587888 0.846190 0.661325 0.135297
    Hybrid Ensemble  0.804826   0.668942 0.524064 0.587706 0.847260 0.656088 0.134975
```

Preferred model selected on validation PR-AUC: **XGBoost**.
Test results are final descriptive comparisons, not a basis for changing the fitted model or ensemble weights.

See training_metadata.json for actual splits, tuning, calibration and ensemble policy.
No retention interventions or financial outcomes were observed. Predictions are associations in historical sample data.
