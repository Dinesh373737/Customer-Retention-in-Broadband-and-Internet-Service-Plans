"""Incremental retention scenarios; no causal uplift is inferred from risk scores."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


def campaign_scenario(inputs: dict) -> dict:
    targeted = inputs.get("targeted", 100)
    cost = float(inputs.get("intervention_cost", 15))
    uplift = float(inputs.get("incremental_impact", .1))
    contribution = float(inputs.get("contribution", 240))
    fixed = float(inputs.get("fixed_cost", 0))
    budget = inputs.get("budget")
    if any(not math.isfinite(float(value)) for value in [targeted, cost, uplift, contribution, fixed]):
        raise ValueError("Economic assumptions must be finite numbers.")
    if targeted < 0 or int(targeted) != targeted or min(cost, contribution, fixed) < 0 or not 0 <= uplift <= 1:
        raise ValueError("Target count/costs must be non-negative; incremental impact must be between zero and one.")
    if budget is not None and (not math.isfinite(float(budget)) or budget < 0):
        raise ValueError("Campaign budget must be a finite non-negative amount.")
    requested = int(targeted)
    if budget is not None:
        targeted = 0 if budget < fixed else min(requested, math.floor((budget - fixed) / cost) if cost else requested)
    targeted = int(targeted)
    # A campaign unable to fund its fixed cost is not launched and spends zero.
    launched = budget is None or budget >= fixed
    expenditure = targeted * cost + fixed if launched else 0.
    retained = targeted * uplift
    incremental_contribution = retained * contribution
    net = incremental_contribution - expenditure
    if not all(math.isfinite(value) for value in [expenditure, retained, incremental_contribution, net]):
        raise ValueError("Scenario arithmetic exceeds supported numeric limits.")
    return {**inputs, "requested_targeted": requested, "targeted": targeted, "launched": launched,
            "total_expenditure": expenditure, "campaign_cost": expenditure,
            "incremental_retained": retained, "incremental_contribution": incremental_contribution,
            "net_benefit": net, "roi_pct": net / expenditure * 100 if expenditure > 0 else None,
            "break_even_impact": expenditure / (targeted * contribution) if targeted * contribution else None,
            "expected_net_value_per_customer": uplift * contribution - cost,
            "budget_remaining": budget - expenditure if budget is not None else None,
            "assumption_status": "hypothetical_unconditional_incremental_impact",
            "caveats": ["Incremental impact is a user assumption, not estimated from churn labels.",
                        "Churn probability is not multiplied by unconditional uplift.",
                        "Contribution is margin after service costs over the stated period, before campaign costs."]}


def economic_analysis(inputs: dict, scored: pd.DataFrame | None = None):
    scenario = campaign_scenario(inputs)
    sensitivity = []
    for multiplier in [.5, 1., 1.5]:
        for rate in np.linspace(0, min(1., max(.3, inputs.get("incremental_impact", .1) * 2)), 13):
            for value_multiplier in [.5, 1., 1.5]:
                changed = {**inputs, "intervention_cost": inputs.get("intervention_cost", 15) * multiplier,
                           "incremental_impact": float(rate), "contribution": inputs.get("contribution", 240) * value_multiplier}
                run = campaign_scenario(changed)
                sensitivity.append({"intervention_cost": changed["intervention_cost"], "incremental_impact": float(rate),
                                    "contribution": changed["contribution"], "cost_multiplier": multiplier,
                                    "value_multiplier": value_multiplier, "targeted": run["targeted"], "net_benefit": run["net_benefit"]})
    ranking = []
    if scored is not None and len(scored) and scenario["expected_net_value_per_customer"] > 0:
        ordered = scored.sort_values("churn_probability", ascending=False, na_position="last") if "churn_probability" in scored else scored
        for position, (_, row) in enumerate(ordered.head(min(scenario["targeted"], 1000)).iterrows(), 1):
            ranking.append({"rank": position, "customer_reference": str(row.get("customer_reference", row.get("row_index"))),
                            "row_index": int(row["row_index"]), "churn_probability": row.get("churn_probability"),
                            "expected_net_value": scenario["expected_net_value_per_customer"],
                            "estimated_cost": inputs.get("intervention_cost", 15),
                            "ranking_basis": "Common assumed contribution and unconditional uplift; predicted risk breaks equal-value ties only."})
    return {"scenario": scenario, "sensitivity": sensitivity, "ranking": ranking,
            "ranking_total_eligible": len(scored) if scored is not None else 0,
            "ranking_display_limit": 1000,
            "assumptions": ["All costs, margins, impact and the value period are user-supplied hypothetical assumptions.",
                            "Homogeneous per-customer contribution, unconditional impact and intervention cost are assumed.",
                            "Budget constrains the target count after fixed cost; no campaign is launched when fixed cost exceeds budget.",
                            "No causal retention effect or actual campaign profit is observed."]}
