import pandas as pd
from src.data_ingestion.ingest import DataIngestion
from src.validation.validate import DataValidation
from src.feature_engineering.features import FeatureEngineering


def test_ingestion_loads_data():
    """Test that ingestion loads 7043 rows."""
    ingestor = DataIngestion()
    df = ingestor.run()
    assert df.shape[0] == 7043
    assert df.shape[1] == 21


def test_validation_fixes_total_charges():
    """Test that TotalCharges becomes float64 with no blanks."""
    ingestor = DataIngestion()
    df = ingestor.run()
    validator = DataValidation()
    df = validator.run(df)
    assert df["TotalCharges"].dtype == "float64"
    assert df["TotalCharges"].isna().sum() == 0


def test_feature_engineering_creates_features():
    """Test that features are all numeric and customerID is dropped."""
    ingestor = DataIngestion()
    df = ingestor.run()
    validator = DataValidation()
    df = validator.run(df)
    fe = FeatureEngineering()
    df_features = fe.run(df)
    assert "customerID" not in df_features.columns
    assert df_features["Churn"].isin([0, 1]).all()
    assert df_features.shape[1] == 31