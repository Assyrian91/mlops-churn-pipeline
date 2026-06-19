import pandas as pd
import yaml
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataIngestion:
    """Loads raw CSV and performs basic sanity checks."""

    def __init__(self, config_path: str = "configs/ingestion_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.raw_path = Path(self.config["data_source"]["raw_path"])
        logger.info(f"Initialized with raw_path={self.raw_path}")

    def load_data(self) -> pd.DataFrame:
        if not self.raw_path.exists():
            raise FileNotFoundError(f"Data not found at {self.raw_path}")
        df = pd.read_csv(self.raw_path)
        logger.info(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")
        return df

    def basic_checks(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        if before - after > 0:
            logger.warning(f"Dropped {before - after} duplicate rows")
        else:
            logger.info("No duplicates found")

        nulls = df.isnull().sum()
        blank_total = (df["TotalCharges"].astype(str).str.strip() == "").sum()
        logger.info(f"Null values: {nulls[nulls > 0].to_dict() if nulls.sum() > 0 else 'None'}")
        logger.info(f"Blank strings in TotalCharges: {blank_total}")

        return df

    def run(self) -> pd.DataFrame:
        df = self.load_data()
        df = self.basic_checks(df)
        return df


if __name__ == "__main__":
    ingestor = DataIngestion()
    df = ingestor.run()
    print(df.head(3).to_string())
    print(f"\nFinal shape: {df.shape}")