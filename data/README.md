# Historical IBM Telco sample data

`Telco-Customer-Churn.csv` was copied from the user's existing local Downloads file. Its verified schema has 21 columns and 7,043 customer records. The IBM project identifies this sample dataset at [the official repository](https://github.com/IBM/telco-customer-churn-on-icp4d/tree/master/data). The original download history of the user's local file is unavailable; `provenance.json` records its content hash and observed schema.

`sample_telco.csv` contains 240 **actual** records sampled without replacement from that CSV with seed 42. It is used only when the configured full dataset cannot be found, and the dashboard displays a warning. It is not synthetic, does not have the same distribution necessarily, and is not a substitute for full-data model results. The default artifact is tied to the exact full-data fingerprint and cannot be applied as that sample's evaluation artifact.

Source records describe a historical sample; they do not identify current subscribers or represent live broadband usage. `customerID` is a pseudonymous navigation identifier and never a predictor. No additional personal information is collected. Monthly/total charges use source monetary units; they do not establish campaign cost or contribution margin. See `docs/data_dictionary.md`.
