"""
Action recommendation engine for the churn model.

WHY THIS EXISTS
----------------
A churn probability by itself isn't a decision. This module converts
`churn_probability` + basic customer economics into:
  1. A risk tier (High / Medium / Low)
  2. A specific recommended action for that tier
  3. The expected DOLLAR VALUE of taking that action

The point of #3 is to make the recommendation defensible. "Offer a 25%
discount to high-risk customers" is a guess. "Offer a 25% discount because
the expected value of doing so is +$X per customer, given a Y% chance the
offer prevents churn" is an analysis. This module computes the latter.

ASSUMPTIONS (clearly flagged — replace with real historical data when available)
----------------------------------------------------------------------------
This is a portfolio project trained on the public IBM Telco dataset, which
has no intervention-outcome history (i.e. we don't know what actually
happens when a "20% discount" is offered to a real customer). So three
constants below are informed estimates, not fitted parameters:

  - GROSS_MARGIN_PCT: assumed share of MonthlyCharges that is margin
    (telecom is typically a high-margin, low-marginal-cost business).
  - RETENTION_SUCCESS_RATE: assumed probability that the recommended
    action actually prevents the predicted churn, by tier. High-risk
    customers are harder to win back (already decided to leave) than
    medium-risk ones (still undecided), which is why success rate is
    *lower* for High than Medium below.
  - REMAINING_LIFETIME_MONTHS_IF_RETAINED: assumed additional months a
    customer stays if successfully retained, by tier.

In production, all three would be estimated from a labeled history of
past interventions and their outcomes (see the data/interventions.csv
scaffold and the /metrics endpoint aggregation this module feeds).
"""

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Tunable business assumptions — isolated here so they're easy to find,
# question, and replace with real fitted values later.
# ---------------------------------------------------------------------------

RISK_THRESHOLDS = {
    "High": 0.70,
    "Medium": 0.40,
    # anything below Medium threshold falls into Low
}

GROSS_MARGIN_PCT = 0.60  # assumed 60% margin on MonthlyCharges

RETENTION_SUCCESS_RATE = {
    "High": 0.30,    # already decided to leave — hardest to reverse
    "Medium": 0.45,  # still undecided — most winnable segment
    "Low": 0.70,     # not actually at risk — "success" just means status quo holds
}

REMAINING_LIFETIME_MONTHS_IF_RETAINED = {
    "High": 12,
    "Medium": 18,
    "Low": 24,
}

# Intervention cost model: (discount_pct_of_monthly_bill, offer_duration_months, flat_overhead)
# flat_overhead approximates staff time (e.g. a retention call, account manager time)
INTERVENTION_COST_MODEL = {
    "High": {"discount_pct": 0.25, "duration_months": 3, "flat_overhead": 50.0},
    "Medium": {"discount_pct": 0.125, "duration_months": 3, "flat_overhead": 15.0},
    "Low": {"discount_pct": 0.0, "duration_months": 0, "flat_overhead": 2.0},
}

ACTION_TEXT = {
    "High": {
        "headline": "URGENT — Immediate Action Required",
        "steps": [
            "Offer 20-30% discount on current plan",
            "Assign dedicated account manager",
            "Offer free upgrade to higher-tier plan for 3 months",
            "Schedule retention call within 24 hours",
        ],
    },
    "Medium": {
        "headline": "MONITOR — Proactive Outreach Recommended",
        "steps": [
            "Send personalized check-in email",
            "Offer loyalty discount (10-15%)",
            "Suggest contract upgrade from month-to-month",
            "Add free value (extra data, premium channel trial)",
        ],
    },
    "Low": {
        "headline": "STABLE — Maintain Relationship",
        "steps": [
            "Continue standard engagement",
            "Include in loyalty rewards program",
            "Send satisfaction survey quarterly",
        ],
    },
}


@dataclass
class ActionRecommendation:
    risk_tier: str
    headline: str
    recommended_steps: List[str]
    estimated_value_at_risk: float
    estimated_intervention_cost: float
    assumed_success_rate: float
    expected_value_of_action: float
    risk_factors: List[str] = field(default_factory=list)
    assumptions_note: str = (
        "Dollar figures use documented assumptions (gross margin, retention "
        "success rate, remaining lifetime) — see recommendations.py docstring. "
        "Not fitted to real intervention outcomes; treat as illustrative."
    )


def _risk_tier(probability: float) -> str:
    if probability >= RISK_THRESHOLDS["High"]:
        return "High"
    if probability >= RISK_THRESHOLDS["Medium"]:
        return "Medium"
    return "Low"


def _detect_risk_factors(tenure: int, contract: str, internet: str,
                          payment: str, tech_support: str) -> List[str]:
    """
    Simple, transparent threshold rules based on known churn drivers in the
    Telco dataset's own EDA (short tenure, month-to-month contracts, fiber
    optic, electronic check, no tech support all correlate with churn).

    NOTE: these are hand-picked thresholds, not derived from the trained
    model's feature importances. A natural next iteration is to replace
    this function with SHAP values or model.feature_importances_ so the
    explanation is generated from the model itself rather than asserted
    separately from it.
    """
    factors = []
    if tenure < 12:
        factors.append("Very short tenure — customer hasn't committed")
    if contract == "Month-to-month":
        factors.append("Month-to-month contract — easy to leave")
    if internet == "Fiber optic":
        factors.append("Fiber optic users have higher churn rates")
    if payment == "Electronic check":
        factors.append("Electronic check payment — correlated with churn")
    if tech_support == "No":
        factors.append("No tech support — less engagement")
    if not factors:
        factors.append("No major risk factors detected")
    return factors


def recommend_action(
    probability: float,
    monthly_charges: float,
    tenure: int,
    contract: str,
    internet: str,
    payment: str,
    tech_support: str,
) -> ActionRecommendation:
    """
    Turn a churn probability + basic customer economics into a costed,
    tiered action recommendation.

    expected_value_of_action = (success_rate * value_at_risk) - intervention_cost

    A positive expected value means the recommended action is worth taking
    on average, given the stated assumptions. A negative value is a signal
    the assumptions (or the action itself) need revisiting for that segment
    — the function still returns a recommendation, but the number makes the
    trade-off visible rather than hiding it behind a fixed if/elif rule.
    """
    tier = _risk_tier(probability)

    monthly_margin = monthly_charges * GROSS_MARGIN_PCT
    value_at_risk = (
        probability
        * monthly_margin
        * REMAINING_LIFETIME_MONTHS_IF_RETAINED[tier]
    )

    cost_model = INTERVENTION_COST_MODEL[tier]
    intervention_cost = (
        cost_model["discount_pct"] * monthly_charges * cost_model["duration_months"]
        + cost_model["flat_overhead"]
    )

    success_rate = RETENTION_SUCCESS_RATE[tier]
    expected_value = (success_rate * value_at_risk) - intervention_cost

    action = ACTION_TEXT[tier]
    risk_factors = _detect_risk_factors(tenure, contract, internet, payment, tech_support)

    return ActionRecommendation(
        risk_tier=tier,
        headline=action["headline"],
        recommended_steps=action["steps"],
        estimated_value_at_risk=round(value_at_risk, 2),
        estimated_intervention_cost=round(intervention_cost, 2),
        assumed_success_rate=success_rate,
        expected_value_of_action=round(expected_value, 2),
        risk_factors=risk_factors,
    )