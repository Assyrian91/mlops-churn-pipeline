import pandas as pd
import joblib
import logging
from fastapi import FastAPI
from src.api.schemas import CustomerData, PredictionResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Churn Prediction API", version="1.0")

# Load model once at startup
model = joblib.load("models/churn_model.joblib")
logger.info("Model loaded successfully")

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

    # Encode binary columns
    for col in BINARY_COLS:
        data[col] = 1 if data[col] == "Yes" else 0

    # Create DataFrame with all columns
    df = pd.DataFrame([data])

    # One-hot encode (must match training: drop_first=True)
    df = pd.get_dummies(df, columns=MULTI_CAT_COLS, drop_first=True, dtype=int)

    # Ensure all 30 feature columns exist (fill missing with 0)
    expected_features = model.feature_names_in_
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    # Keep only the columns the model expects, in the right order
    df = df[expected_features]

    return df


@app.get("/")
def home():
    return {"message": "Churn Prediction API is running", "docs": "/docs"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    """Predict churn for a single customer."""
    df = transform_input(customer)
    probability = model.predict_proba(df)[0][1]
    will_churn = bool(probability >= 0.5)

    logger.info(f"Prediction: churn={will_churn}, probability={probability:.4f}")

    return PredictionResponse(
        customer_will_churn=will_churn,
        churn_probability=round(probability, 4)
    )