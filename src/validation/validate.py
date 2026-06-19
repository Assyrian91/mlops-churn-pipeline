import pandas as pd
import yaml
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataValidation:
    """Validates and cleans the dataset after ingestion."""

    def __init__(self, config_path: str = "configs/ingestion_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.processed_path = Path(self.config["data_source"]["processed_path"])
        self.target = self.config["schema"]["target_column"]
        self.numeric_cols = self.config["schema"]["numeric_columns"]
        logger.info("DataValidation initialized")

    def fix_total_charges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert TotalCharges to numeric, fill blanks with 0."""
        before = df["TotalCharges"].dtype
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        blank_count = df["TotalCharges"].isna().sum()
        df["TotalCharges"] = df["TotalCharges"].fillna(0)
        after = df["TotalCharges"].dtype
        logger.info(f"TotalCharges: {before} -> {after}, filled {blank_count} blanks with 0")
        return df

    def validate_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure Churn only contains Yes/No."""
        valid = {"Yes", "No"}
        actual = set(df[self.target].unique())
        invalid = actual - valid
        if invalid:
            raise ValueError(f"Unexpected values in {self.target}: {invalid}")
        logger.info(f"Target '{self.target}' valid: {valid}")
        return df

    def validate_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure numeric columns are actually numeric type."""
        for col in self.numeric_cols:
            if df[col].dtype not in ["float64", "int64"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                logger.warning(f"Converted {col} to numeric")
        logger.info(f"All numeric columns verified: {self.numeric_cols}")
        return df

    def save_clean_data(self, df: pd.DataFrame) -> None:
        """Save cleaned data to processed folder."""
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.processed_path, index=False)
        logger.info(f"Clean data saved to {self.processed_path}")

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full validation pipeline."""
        df = self.fix_total_charges(df)
        df = self.validate_numeric(df)
        df = self.validate_target(df)
        self.save_clean_data(df)
        return df


if __name__ == "__main__":
    from src.data_ingestion.ingest import DataIngestion

    ingestor = DataIngestion()
    df = ingestor.run()

    validator = DataValidation()
    df_clean = validator.run(df)

    print(df_clean.dtypes)
    print(f"\nFinal shape: {df_clean.shape}")