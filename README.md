🔄 End-to-End MLOps Churn Prediction Pipeline
A production-grade MLOps pipeline that predicts customer churn for a telecom company, built from scratch with full automation.

🎯 What It Does
Takes raw customer data → cleans it → engineers features → trains a model → serves predictions via API → all automated.

🏗️ Architecture
Raw Data → Ingestion → Validation → Feature Engineering → Model Training (MLflow)
↓
FastAPI Service
↓
Docker → Hugging Face Cloud
↓
/health /metrics /predict


## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.14 | Core language |
| Pandas / Scikit-learn | Data processing & modeling |
| MLflow | Experiment tracking |
| FastAPI | REST API service |
| Docker | Containerization |
| GitHub Actions | CI/CD automation |
| Hugging Face Spaces | Cloud deployment |

## 📁 Project Structure

mlops-churn-pipeline/
├── configs/ # YAML configuration files
├── data/
│ ├── raw/ # Raw dataset (gitignored)
│ └── processed/ # Cleaned & feature data (gitignored)
├── src/
│ ├── data_ingestion/ # Load & validate raw data
│ ├── validation/ # Fix blanks, enforce schema
│ ├── feature_engineering/ # Encode categorical, one-hot encoding
│ ├── models/ # Train Random Forest with MLflow
│ ├── api/ # FastAPI service + schemas
│ └── utils/ # Shared utilities
├── tests/ # Pytest pipeline tests
├── scripts/ # Model export scripts
├── models/ # Saved model artifacts (gitignored)
├── .github/workflows/ # CI/CD pipeline
├── Dockerfile # Container definition
└── requirements.txt # Dependencies


## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/Assyrian91/mlops-churn-pipeline.git
cd mlops-churn-pipeline
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

2. Download Dataset
Download from Kaggle and place in data/raw/Telco-Customer-Churn.csv

3. Run the Pipeline
# Ingestion + Validation + Feature Engineering
python -m src.feature_engineering.features

# Train model with MLflow tracking
python -m src.models.train

# Export model for deployment
python -m scripts.save_model

4. Run API Locally

uvicorn src.api.main:app --reload --port 8000

5. Run with Docker

docker build -t churn-api:latest .
docker run -d -p 8000:8000 --name churn-container churn-api:latest

📡 API Endpoints
Endpoint
Method
Description
/	GET	API info
/health	GET	Health check
/metrics	GET	Prediction statistics
/predict	POST	Predict customer churn
/docs	GET	Swagger UI documentation

Example Prediction

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
  }'

Response:

{
  "customer_will_churn": true,
  "churn_probability": 0.5857
}

🧪 Run Tests

pytest tests/test_pipeline.py -v

📊 Model Performance (Baseline)
Metric
Score
Accuracy	80.84%
F1 Score	0.5946
Precision	0.6781
Recall	0.5294

🌐 Live Demo
API: https://assyriana-churn-api.hf.space

Swagger Docs: https://assyriana-churn-api.hf.space/docs

⚙️ CI/CD
GitHub Actions automatically runs on every push:

✅ Run pipeline tests
✅ Build Docker image
CI Pipeline


📝 License
MIT