from typing import List
from pydantic import BaseModel


class CustomerData(BaseModel):
    """Raw customer data — same columns as the CSV (minus customerID and Churn)."""
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class ActionRecommendationResponse(BaseModel):
    """
    Costed, tiered action recommendation derived from the churn probability.
    See src/api/recommendations.py for the full methodology and assumptions.
    """
    risk_tier: str
    headline: str
    recommended_steps: List[str]
    estimated_value_at_risk: float
    estimated_intervention_cost: float
    assumed_success_rate: float
    expected_value_of_action: float
    risk_factors: List[str]
    assumptions_note: str


class PredictionResponse(BaseModel):
    """API response."""
    customer_will_churn: bool
    churn_probability: float
    recommendation: ActionRecommendationResponse