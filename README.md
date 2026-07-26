# Telecom Customer Churn Predictor
### By Khoshaba Odesho

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)

Turning raw customer data into proactive retention strategies using modern MLOps tools.

---

## 💡 The Business Problem

Telecom companies lose an average of $1,200 – $2,400 per year for every customer who leaves. For a mid-sized company with 7,000 customers, a mere 1% reduction in churn saves $84,000+ annually.

Traditionally, retention teams react after a customer cancels. This application changes the approach from reactive to proactive, giving frontline teams the data they need to intervene before a customer leaves.

## 🚀 The Solution

A cloud-hosted web application where non-technical Customer Success Managers can input a customer's profile and instantly receive:

- A churn probability score (High / Medium / Low risk).
- Personalized retention actions (e.g., "Offer 20% discount", "Schedule retention call within 24hrs").
- Key risk factors explaining why the customer is at risk.

No coding required. Just data in, strategy out.

## 🛠️ Tools & Technologies Used

| Tool | Role in This Project |
|------|----------------------|
| **Python** | Core programming language for data processing, cleaning, and application logic |
| **Scikit-Learn** | Random Forest classification model to calculate churn probability based on customer features |
| **MLflow** | Experiment tracking to log model parameters, metrics, and save artifacts for reproducibility |
| **FastAPI** | High-performance REST API to serve predictions at scale for developer integration |
| **Streamlit** | Frontend dashboard allowing non-technical business users to interact with the model |
| **Docker** | Containerization to ensure the application runs identically on any machine or cloud environment |
| **GitHub Actions** | CI/CD pipeline to automatically run tests and build Docker images on every code push |
| **Hugging Face Spaces** | Cloud hosting platform to make both the API and Dashboard accessible globally via URL |

## 🏗️ Architecture

```
Raw Data → Ingestion → Validation → Feature Engineering → Model Training (MLflow)
                                                                  ↓
                                                          FastAPI Service
                                                                  ↓
                                                    Docker → Hugging Face Cloud
                                                                  ↓
                                              /health | /metrics | /predict
                                                                  +
                                                Streamlit Business Dashboard
```

## 🌐 Live Applications

| Application | URL | Target User |
|-------------|-----|--------------|
| **Business Dashboard** | [Streamlit App](https://mlops-churn-pipeline-by-khoshaba.streamlit.app/) | Retention Managers, Sales Teams |
| **Backend API** | [FastAPI Docs](https://assyriana-churn-api.hf.space/docs) | Developers, System Integrations |
| **System Health** | [Health Endpoint](https://assyriana-churn-api.hf.space/health) | IT Operations, Monitoring |

## 👥 Who Benefits From This?

- **Customer Success Managers:** Get instant, data-backed retention strategies without waiting on data science teams.
- **Sales Teams:** Identify accounts that need contract upgrades before they churn.
- **Executives:** Monitor overall churn risk across the customer base via the `/metrics` endpoint.

## 📁 Project Structure

```
mlops-churn-pipeline/
├── assets/                     # Branding assets (logo)
├── configs/                    # YAML configuration files
├── data/
│   ├── raw/                    # Raw dataset (gitignored)
│   └── processed/               # Cleaned & feature data (gitignored)
├── src/
│   ├── data_ingestion/          # Load & validate raw data
│   ├── validation/              # Fix blanks, enforce schema
│   ├── feature_engineering/     # Encode categorical, one-hot encoding
│   ├── models/                  # Train Random Forest with MLflow
│   ├── api/                     # FastAPI service + Streamlit dashboard
│   └── utils/                   # Shared utilities
├── tests/                       # Pytest pipeline tests
├── scripts/                     # Model export scripts
├── models/                      # Saved model artifacts (gitignored)
├── .github/workflows/           # CI/CD pipeline
├── Dockerfile                   # Container definition
└── requirements.txt             # Dependencies
```

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/Assyrian91/mlops-churn-pipeline.git
cd mlops-churn-pipeline
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Download Dataset

Download from Kaggle and place in `data/raw/Telco-Customer-Churn.csv`

### 3. Run the Pipeline

```bash
# Ingestion + Validation + Feature Engineering
python -m src.feature_engineering.features

# Train model with MLflow tracking
python -m src.models.train

# Export model for deployment
python -m scripts.save_model
```

### 4. Run API Locally

```bash
uvicorn src.api.main:app --reload --port 8000
```

### 5. Run Dashboard Locally

```bash
streamlit run src/api/app.py
```

### 6. Run with Docker

```bash
docker build -t churn-api:latest .
docker run -d -p 8000:8000 --name churn-container churn-api:latest
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/metrics` | GET | Prediction statistics |
| `/predict` | POST | Predict customer churn |
| `/docs` | GET | Swagger UI documentation |

### Example Prediction

```bash
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
```

## 🧪 Run Tests

```bash
pytest tests/test_pipeline.py -v
```

## 📊 Model Performance

Trained using a Random Forest Classifier on the IBM Telecom Dataset (7,043 records):

| Metric | Score |
|--------|-------|
| Accuracy | 80.84% |
| F1 Score | 0.5946 |
| Precision | 0.6781 |
| Recall | 0.5294 |

## ⚙️ CI/CD

GitHub Actions automatically runs on every push:

- ✅ Run pipeline tests
- ✅ Build Docker image

## 📝 License

MIT

<div align="center">
<p>Built by <strong>Khoshaba Odesho</strong> using modern MLOps practices.</p>
</div>
