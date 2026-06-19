import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import pandas as pd
import mlflow
import mlflow.sklearn
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Use local file tracking instead of UI server
# MLflow will create an "mlruns" folder with all experiment data
mlflow.set_tracking_uri("file:///C:/Users/HB Laptop Store/Desktop/mlops-churn-pipeline/mlruns")
mlflow.set_experiment("churn-prediction")


def load_features(path: str = "data/processed/churn_features.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info(f"Loaded features: {df.shape}")
    return df


def split_data(df: pd.DataFrame, target: str = "Churn", test_size: float = 0.2):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, params: dict):
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    logger.info(f"Model trained with params: {params}")
    return model


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1_score": f1_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
    }
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    return metrics


def run_experiment():
    df = load_features()
    X_train, X_test, y_train, y_test = split_data(df)

    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
    }

    with mlflow.start_run(run_name="baseline_random_forest"):
        mlflow.log_params(params)

        model = train_model(X_train, y_train, params)
        metrics = evaluate(model, X_test, y_test)

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        logger.info("MLflow run logged to mlruns/ folder")

    return model


if __name__ == "__main__":
    model = run_experiment()