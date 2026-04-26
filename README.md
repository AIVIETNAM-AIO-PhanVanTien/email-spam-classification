# End-to-End Email Spam Classification

## 1. Overview

This project builds an end-to-end machine learning system for **binary email spam classification**.

The model classifies an email into:

| Label | Meaning |
|---|---|
| `0` | Ham / Not Spam |
| `1` | Spam |

The project includes:

- Local Medallion data pipeline: Bronze, Silver, Gold
- TF-IDF + Logistic Regression model
- MLflow experiment tracking
- FastAPI backend
- Streamlit frontend
- Prediction logging
- Docker-based deployment
- QA testing

---

## 2. Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Data Processing | Pandas, Parquet |
| ML | Scikit-learn |
| Feature Engineering | TF-IDF |
| Model | Logistic Regression |
| Experiment Tracking | MLflow |
| Backend | FastAPI |
| Frontend | Streamlit |
| Testing | Pytest |
| Deployment | Docker, Docker Compose |

---

## 3. Project Structure

```text
email-spam-classification/
│
├── data/
│   ├── raw/
│   ├── bronze/
│   ├── silver/
│   ├── gold/
│   └── test_samples/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_logistic_regression.ipynb
│   ├── 03_baseline_naive_bayes.ipynb
│   ├── 04_baseline_knn.ipynb
│   └── 05_model_comparison.ipynb
│
├── src/
│   ├── bronze_ingest.py
│   ├── silver_transform.py
│   ├── gold_build.py
│   ├── text_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── tune.py
│   └── predict.py
│
├── app/
│   ├── api.py
│   └── streamlit_app.py
│
├── models/
├── reports/
├── logs/
├── tests/
├── docs/
│
├── README.md
├── requirements.txt
├── .env.example
├── Dockerfile
└── docker-compose.yml
```

---

## 4. Environment Setup

### 4.1 Clone Repository

```bash
git clone https://github.com/<your-username>/email-spam-classification.git
cd email-spam-classification
```

### 4.2 Create Virtual Environment

For macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

For Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 4.3 Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4.4 Install Dependencies

```bash
pip install -r requirements.txt
```

### 4.5 Create Environment File

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

For Windows PowerShell:

```powershell
copy .env.example .env
```

Example `.env`:

```env
MODEL_PATH=models/best_spam_classifier.pkl
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
API_HOST=0.0.0.0
API_PORT=8000
STREAMLIT_PORT=8501
PREDICTION_LOG_PATH=logs/predictions.csv
API_URL=http://127.0.0.1:8000/predict
```

---

## 5. Dataset

The project uses a spam/ham email dataset.

Recommended dataset:

```text
190K+ Spam/Ham Email Dataset for Classification
```

All datasets must be normalized to this schema:

```csv
text,label
"Congratulations, you won a free prize",1
"Hi team, please review the meeting note",0
```

Label mapping:

| Original Label | Encoded Label |
|---|---|
| `ham` / `not_spam` | `0` |
| `spam` | `1` |

Place the raw dataset here:

```text
data/raw/spam.csv
```

---

## 6. Run Data Pipeline

### 6.1 Bronze Layer

```bash
python src/bronze_ingest.py
```

Output:

```text
data/bronze/spam_raw.csv
```

### 6.2 Silver Layer

```bash
python src/silver_transform.py
```

Output:

```text
data/silver/spam_clean.parquet
```

### 6.3 Gold Layer

```bash
python src/gold_build.py
```

Output:

```text
data/gold/train.parquet
data/gold/test.parquet
```

---

## 7. Train Model

### 7.1 Start MLflow UI

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

### 7.2 Run Training Pipeline

```bash
python src/train.py
```

Outputs:

```text
models/best_spam_classifier.pkl
models/model_metadata.json
reports/classification_report.txt
reports/confusion_matrix.txt
```

---

## 8. Run Backend API

Start FastAPI:

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

### Predict Endpoint

```http
POST /predict
```

Request:

```json
{
  "subject": "Congratulations",
  "body": "You won a free iPhone. Click here now."
}
```

Response:

```json
{
  "label": "spam",
  "spam_probability": 0.94
}
```

---

## 9. Run Streamlit App

Start frontend:

```bash
streamlit run app/streamlit_app.py --server.port 8501
```

Open:

```text
http://127.0.0.1:8501
```

---

## 10. Run Tests

```bash
pytest
```

Run specific tests:

```bash
pytest tests/test_predict.py
pytest tests/test_api.py
```

---

## 11. Run with Docker Compose

Build and run services:

```bash
docker compose up -d --build
```

Open:

| Service | URL |
|---|---|
| FastAPI | `http://localhost:8000/docs` |
| Streamlit | `http://localhost:8501` |

Stop services:

```bash
docker compose down
```

---

## 12. Git Workflow

Branch strategy:

```text
main
develop
feature/*
fix/*
docs/*
deploy/*
```

Commit convention:

```text
feat: add new feature
fix: fix bug
docs: update documentation
refactor: improve code
test: add tests
chore: update config
deploy: update deployment files
```

Example:

```bash
git commit -m "feat: add logistic regression training pipeline"
```

---

## 13. Team Roles

| Role | Main Responsibility |
|---|---|
| Tech Lead | Architecture, code review, integration, deployment |
| QA / Reviewer | Test plan, acceptance criteria, quality review |
| AI Engineer — Data | Dataset, EDA, Bronze/Silver/Gold pipeline |
| AI Engineer — Training Pipeline | Training, MLflow, tuning, model artifact |
| AI Engineer — Model Serving | FastAPI, Streamlit, prediction logging |

---

## 14. Definition of Done

The project is completed when:

- Data pipeline creates Bronze, Silver, and Gold layers
- Model training pipeline runs successfully
- MLflow logs experiments
- Best model artifact is saved
- FastAPI `/predict` works
- Streamlit UI works
- Prediction logs are created
- Tests pass
- Docker Compose runs the full app
- Documentation is updated

---

## 15. Future Improvements

- Gmail API integration
- Phishing URL detection
- Model drift monitoring
- Dashboard for prediction logs
- Hugging Face Spaces deployment
- EC2 / Azure App Service deployment
- GitHub Actions CI/CD
