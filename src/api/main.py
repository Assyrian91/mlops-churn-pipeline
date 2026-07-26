import pandas as pd
import joblib
import logging
from datetime import datetime
from fastapi import FastAPI
from src.api.schemas import CustomerData, PredictionResponse, ActionRecommendationResponse
from src.api.recommendations import recommend_action

# File logging for audit trail
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler("predictions.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Churn Prediction API", version="1.1")

# Load model once at startup
model = joblib.load("models/churn_model.joblib")
logger.info("Model loaded successfully")

# Simple in-memory metrics
prediction_count = 0
churn_count = 0
cumulative_value_at_risk = 0.0
cumulative_expected_value_of_actions = 0.0

# These must match training exactly
BINARY_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
MULTI_CAT_COLS = [
    "gender", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod"
]


def transform_input(customer: CustomerData) -> pd.DataFrame:
    """Convert raw customer data to model-ready features."""
    data = customer.model_dump()

    for col in BINARY_COLS:
        data[col] = 1 if data[col] == "Yes" else 0

    df = pd.DataFrame([data])
    df = pd.get_dummies(df, columns=MULTI_CAT_COLS, drop_first=True, dtype=int)

    expected_features = model.feature_names_in_
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_features]
    return df


@app.get("/")
def home():
    return {"message": "Churn Prediction API is running", "docs": "/docs"}


@app.get("/health")
def health_check():
    """Check if API and model are healthy."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/metrics")
def get_metrics():
    """Return prediction metrics, including cumulative estimated business value."""
    return {
        "total_predictions": prediction_count,
        "churn_predicted": churn_count,
        "no_churn_predicted": prediction_count - churn_count,
        "churn_rate": round(churn_count / prediction_count, 4) if prediction_count > 0 else 0,
        "cumulative_value_at_risk": round(cumulative_value_at_risk, 2),
        "cumulative_expected_value_of_recommended_actions": round(cumulative_expected_value_of_actions, 2),
        "note": "Dollar figures are illustrative — based on documented assumptions in "
                "src/api/recommendations.py, not fitted to real intervention outcomes."
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    """Predict churn for a single customer and return a costed action recommendation."""
    global prediction_count, churn_count
    global cumulative_value_at_risk, cumulative_expected_value_of_actions

    df = transform_input(customer)
    probability = model.predict_proba(df)[0][1]
    will_churn = bool(probability >= 0.5)

    recommendation = recommend_action(
        probability=probability,
        monthly_charges=customer.MonthlyCharges,
        tenure=customer.tenure,
        contract=customer.Contract,
        internet=customer.InternetService,
        payment=customer.PaymentMethod,
        tech_support=customer.TechSupport,
    )

    # Update metrics
    prediction_count += 1
    if will_churn:
        churn_count += 1
    cumulative_value_at_risk += recommendation.estimated_value_at_risk
    cumulative_expected_value_of_actions += recommendation.expected_value_of_action

    # Log to file
    logger.info(
        f"PREDICTION #{prediction_count} | churn={will_churn} | prob={probability:.4f} | "
        f"tier={recommendation.risk_tier} | expected_value=${recommendation.expected_value_of_action:.2f} | "
        f"tenure={customer.tenure} | contract={customer.Contract}"
    )

    return PredictionResponse(
        customer_will_churn=will_churn,
        churn_probability=round(probability, 4),
        recommendation=ActionRecommendationResponse(
            risk_tier=recommendation.risk_tier,
            headline=recommendation.headline,
            recommended_steps=recommendation.recommended_steps,
            estimated_value_at_risk=recommendation.estimated_value_at_risk,
            estimated_intervention_cost=recommendation.estimated_intervention_cost,
            assumed_success_rate=recommendation.assumed_success_rate,
            expected_value_of_action=recommendation.expected_value_of_action,
            risk_factors=recommendation.risk_factors,
            assumptions_note=recommendation.assumptions_note,
        ),
    )