# Schema and business-meaning guide

The target is selected by the analyst, never by last-column position. A historical analysis requires exactly two non-missing canonical classes and an explicit positive churn label. Boolean labels become true/false; integral numeric labels become strings such as 0/1. Unsupported or missing labels are errors, not silently discarded rows.

- **Numeric:** finite values with missing entries allowed. Median imputation and scaling are fitted on training data.
- **Categorical/binary:** nominal levels, missing handling and training-only encoding. Previously unseen nominal scoring categories use the saved encoder and are reported. High-cardinality raw text/IDs must be excluded; infrequent-category grouping has bounded output size.
- **Ordinal:** provide the entire meaningful order, e.g. Low, Medium, High. A source codebook must justify the order. Unknown ordinal levels are rejected instead of assigned arbitrary positions.
- **Date/time:** provide a fixed reference known at prediction time. Dates become elapsed days before that reference. Invalid dates, later dates or outcome-derived timing are not accepted.
- **Identifier/excluded:** customer references and names never become predictors. Direct personal identifiers are blocked. Sensitive/protected predictors have explicit fairness warnings and cannot trigger interventions or what-if changes. Platform output names (`row_index`, `customer_reference`, `churn_probability`, `risk_category`, `actual_outcome`, `actual_churn`) and the internal `__` prefix must be renamed if proposed as predictors.

Constants, entirely missing fields, target copies and suspicious deterministic low-cardinality proxies are rejected with a correction message. Possible post-outcome fields such as Status need exclusion or an explicit timing justification; a justification does not bypass an exact target-proxy block. Duplicate rows are reported and removed only before historical training; batch row cardinality is preserved. Repeated conflicting customer IDs are not accepted for independent-customer evaluation.

Semantic concepts are optional: pricing, lifecycle, support, engagement, contract, satisfaction and loyalty. A mapping needs the actual meaning, units and timing, plus explicit confirmation. A column called Support can mean a subscription, complaint count or contact indicator; those meanings are not interchangeable. Unknown/unconfirmed features can participate in prediction while producing no business recommendation. Industry-specific actions are considered only if observed evidence qualifies; they are never proof of causation.

An actionable checkbox means a hypothetical edit is operationally meaningful. It cannot authorize changing identity, outcomes, protected attributes or a customer's age/lifecycle. Numeric edits must remain in supported observed bounds; categories must be valid. The result is a model sensitivity, not an estimated retention treatment effect.

See the benchmark configurations for real, explicit mappings and documented exclusions. Ordinary users configure these choices in Dataset Studio; they do not need to edit Python source to switch datasets.
