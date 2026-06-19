import pandas as pd
import yaml
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FeatureEngineering:
    """Transforms raw cleaned data into model-ready features."""

    def __init__(self, config_path: str = "configs/ingestion_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.target = self.config["schema"]["target_column"]
        self.id_col = self.config["schema"]["id_column"]
        self.numeric_cols = self.config["schema"]["numeric_columns"]
        self.cat_cols = self.config["schema"]["categorical_columns"]
        self.features_path = Path("data/processed/churn_features.csv")
        logger.info("FeatureEngineering initialized")

    def drop_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop customerID — not useful for modeling."""
        if self.id_col in df.columns:
            df = df.drop(columns=[self.id_col])
            logger.info(f"Dropped column: {self.id_col}")
        return df

    def encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert Churn: Yes -> 1, No -> 0."""
        df[self.target] = df[self.target].map({"Yes": 1, "No": 0})
        logger.info(f"Encoded target: {self.target} -> 0/1")
        return df

    def encode_binary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert Yes/No columns to 1/0."""
        binary_cols = []
        for col in self.cat_cols:
            unique_vals = set(df[col].dropna().unique())
            if unique_vals <= {"Yes", "No"}:
                df[col] = df[col].map({"Yes": 1, "No": 0})
                binary_cols.append(col)
        logger.info(f"Binary encoded {len(binary_cols)} columns: {binary_cols}")
        return df

    def one_hot_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode remaining categorical columns."""
        remaining_cats = [
            col for col in self.cat_cols
            if df[col].dtype == "object"
        ]
        if remaining_cats:
            df = pd.get_dummies(df, columns=remaining_cats, drop_first=True, dtype=int)
            logger.info(f"One-hot encoded {len(remaining_cats)} columns: {remaining_cats}")
        else:
            logger.info("No columns left to one-hot encode")
        return df

    def save_features(self, df: pd.DataFrame) -> None:
        """Save feature matrix."""
        df.to_csv(self.features_path, index=False)
        logger.info(f"Features saved to {self.features_path} ({df.shape[1]} columns)")

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full feature engineering pipeline."""
        df = self.drop_id(df)
        df = self.encode_target(df)
        df = self.encode_binary(df)
        df = self.one_hot_encode(df)
        self.save_features(df)
        return df


if __name__ == "__main__":
    from src.validation.validate import DataValidation
    from src.data_ingestion.ingest import DataIngestion

    df = DataIngestion().run()
    df = DataValidation().run(df)
    df_features = FeatureEngineering().run(df)

    print(f"Shape: {df_features.shape}")
    print(f"Columns: {list(df_features.columns)}")
    print(f"\nTarget distribution:\n{df_features['Churn'].value_counts()}")