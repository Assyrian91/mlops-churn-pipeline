import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
import joblib
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("file:///C:/Users/HB Laptop Store/Desktop/mlops-churn-pipeline/mlruns")


def load_latest_model():
    """Load the most recent model from MLflow."""
    client = mlflow.tracking.MlflowClient()
    experiment = client.search_experiments()[0]
    runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
    run_id = runs[0].info.run_id
    logger.info(f"Loading model from run: {run_id}")

    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    return model


def save_model(model, path: str = "models/churn_model.joblib"):
    """Save model as joblib file."""
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


if __name__ == "__main__":
    model = load_latest_model()
    save_model(model)