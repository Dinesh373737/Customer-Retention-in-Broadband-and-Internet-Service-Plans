"""Incremental campaign scenarios; these inputs are assumptions, not estimates."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


def economic_scenario(
    targeted: int, contact_cost: float, success_rate: float, contribution: float,
    additional_cost: float = 0, period: str = "12 months", currency: str = "USD",
) -> dict:
    """Success is incremental retention uplift (0..1), not baseline survival.

    Contribution is the incremental contribution margin over ``period`` per
    additional retained customer, after service costs but before campaign costs.
    """
    values = [targeted, contact_cost, success_rate, contribution, additional_cost]
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("Scenario inputs must be finite numbers.")
    if targeted < 0 or int(targeted) != targeted:
        raise ValueError("The number targeted must be a non-negative integer.")
    if min(contact_cost, contribution, additional_cost) < 0:
        raise ValueError("Costs and retained-customer contribution must be non-negative.")
    if not 0 <= success_rate <= 1:
        raise ValueError("Incremental intervention success must be between 0 and 1.")
    targeted = int(targeted)
    cost = targeted * contact_cost + additional_cost
    retained = targeted * success_rate
    gain = retained * contribution
    if not all(math.isfinite(value) for value in [cost, retained, gain, gain - cost]):
        raise ValueError("Scenario arithmetic exceeds supported finite numeric limits; reduce the assumptions.")
    return {
        "targeted": targeted, "contact_cost": float(contact_cost),
        "success_rate": float(success_rate), "contribution": float(contribution),
        "additional_cost": float(additional_cost), "period": period, "currency": currency,
        "campaign_cost": cost, "retained": retained,
        "incremental_contribution": gain, "net_benefit": gain - cost,
        "roi_pct": (gain - cost) / cost * 100 if cost > 0 else None,
        "break_even_success_rate": cost / (targeted * contribution)
        if targeted * contribution > 0 else None,
        "break_even_value": cost / retained if retained > 0 else None,
        "assumption_label": "Hypothetical incremental retention scenario; no observed campaign outcomes",
    }


def sensitivity_table(
    targeted: int, contact_cost: float, success_rate: float, contribution: float,
    additional_cost: float = 0, period: str = "12 months", currency: str = "USD",
) -> pd.DataFrame:
    """Vary all three assumptions; no probabilities come from historical Churn."""
    economic_scenario(targeted, contact_cost, success_rate, contribution,
                      additional_cost, period, currency)
    scenarios = []
    rates = sorted(set(np.clip([success_rate * .5, success_rate, success_rate * 1.5], 0, 1)))
    for cost_factor in [.5, 1., 1.5]:
        for value_factor in [.5, 1., 1.5]:
            for rate in rates:
                item = economic_scenario(targeted, contact_cost * cost_factor, float(rate),
                                         contribution * value_factor, additional_cost, period, currency)
                item.update(cost_multiplier=cost_factor, value_multiplier=value_factor)
                scenarios.append(item)
    return pd.DataFrame(scenarios)
