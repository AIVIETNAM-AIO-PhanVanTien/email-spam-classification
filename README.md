
# End-to-End Email Spam Classification

## 1. Project Overview

This project builds an end-to-end machine learning system for **binary email spam classification**.

The system classifies an email into one of two classes:

| Label | Meaning |
|---|---|
| `0` | Ham / Not Spam |
| `1` | Spam |

The project covers the full AI/ML lifecycle:

- Local Medallion data pipeline: Bronze, Silver, Gold
- Exploratory Data Analysis
- Baseline model experiments
- TF-IDF feature extraction
- Logistic Regression model training
- MLflow experiment tracking
- Model evaluation and selection
- FastAPI inference backend
- Streamlit frontend demo
- Prediction logging
- QA testing
- Docker-based deployment

---

## 2. Problem Statement

Email spam is a common problem where users receive unwanted, suspicious, promotional, or fraudulent messages.

The goal of this project is to build a machine learning system that can automatically classify whether an email is **spam** or **not spam** based on its textual content.

The system is designed as a practical AI engineering project, not only a notebook experiment.

---

## 3. Business Objective

The main objective is to create a lightweight, reproducible, and deployable spam classification system.

The final system should allow a user to input:

- Email subject
- Email body

Then the system returns:

- Predicted label: `spam` or `not_spam`
- Spam probability score

Example response:

```json
{
  "label": "spam",
  "spam_probability": 0.94
}

---

## 4. Project Scope

### In Scope

* Collect spam/ham email dataset
* Clean and validate data
* Build local Medallion Architecture
* Train binary classification model
* Track experiments with MLflow
* Serve model with FastAPI
* Build Streamlit demo UI
* Save prediction logs
* Test data, model, API, and UI
* Run application with Docker Compose

### Out of Scope

* Real Gmail integration
* Real-time inbox scanning
* User authentication
* Attachment scanning
* Multi-class email categorization
* BERT/LLM fine-tuning
* Large-scale production monitoring
* Kubernetes deployment

---

## 5. Tech Stack

| Layer               | Technology                   |
| ------------------- | ---------------------------- |
| Language            | Python                       |
| Data Processing     | Pandas, Parquet              |
| Data Architecture   | Local Medallion Architecture |
| Machine Learning    | Scikit-learn                 |
| Feature Engineering | TF-IDF                       |
| Main Model          | Logistic Regression          |
| Experiment Tracking | MLflow                       |
| Backend API         | FastAPI                      |
| Frontend UI         | Streamlit                    |
| Model Serialization | Joblib                       |
| Testing             | Pytest                       |
| Deployment          | Docker, Docker Compose       |
| Version Control     | Git, GitHub                  |

---

## 6. System Architecture

The system contains three main parts:

### 6.1 Offline Training Pipeline

```text
Raw Dataset
   ↓
Bronze Layer
   ↓
Silver Layer
   ↓
Gold Train/Test Dataset
   ↓
Text Preprocessing
   ↓
TF-IDF Feature Engineering
   ↓
Logistic Regression Training
   ↓
Model Evaluation
   ↓
MLflow Tracking
   ↓
Best Model Artifact
```

### 6.2 Online Inference Pipeline

```text
User
   ↓
Streamlit UI
   ↓
FastAPI /predict
   ↓
Load Best Model
   ↓
Preprocess Input
   ↓
Predict Spam Probability
   ↓
Return Result
```

### 6.3 Storage and Monitoring

```text
FastAPI Prediction
   ↓
Prediction Logging
   ↓
logs/predictions.csv
```

---

## 7. Team Roles

| Role                            | Owner           | Main Responsibility                                     |
| ------------------------------- | --------------- | ------------------------------------------------------- |
| Tech Lead                       | Phan Văn Tiến   | Architecture, review code, integration, deployment      |
| QA / Reviewer                   | Lê Dũng         | Test plan, acceptance criteria, quality review          |
| AI Engineer — Data              | Tien Doan       | Dataset, EDA, Bronze/Silver/Gold data pipeline          |
| AI Engineer — Training Pipeline | Võ Ngọc Gia Bảo | Training pipeline, MLflow, GridSearchCV, model artifact |
| AI Engineer — Model Serving     | Tâm Anh Phan    | Inference, FastAPI, Streamlit, prediction logging       |

---

## 8. Dataset

The project uses a labeled spam/ham email dataset.

Recommended primary dataset:

```text
190K+ Spam/Ham Email Dataset for Classification
```

The dataset is normalized into a common schema:

```csv
text,label
"Congratulations, you won a free prize",1
"Hi team, please review the meeting note",0
```

Label mapping:

| Original Label     | Encoded Label |
| ------------------ | ------------- |
| `ham` / `not_spam` | `0`           |
| `spam`             | `1`           |

---

## 9. Project Structure

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
│   ├── __init__.py
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
│   ├── best_spam_classifier.pkl
│   └── model_metadata.json
│
├── reports/
│   ├── data_report.md
│   ├── model_report.md
│   ├── api_ui_report.md
│   └── final_test_report.md
│
├── logs/
│   └── predictions.csv
│
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_predict.py
│   └── test_api.py
│
├── docs/
│   ├── fsd.md
│   ├── tsd.md
│   ├── architecture.md
│   ├── coding_and_git_convention.md
│   ├── test_plan.md
│   └── deployment_guide.md
│
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── Dockerfile
└── docker-compose.yml
```

---

## 10. Local Setup

### 10.1 Clone Repository

```bash
git clone https://github.com/<your-username>/email-spam-classification.git
cd email-spam-classification
```

### 10.2 Create Virtual Environment

```bash
python -m venv .venv
```

Activate environment:

```bash
source .venv/bin/activate
```

For Windows:

```bash
.venv\Scripts\activate
```

### 10.3 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 11. Environment Variables

Create a `.env` file from `.env.example`.

Example:

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

## 12. Data Pipeline

The project uses a local Medallion Architecture.

### 12.1 Bronze Layer

Bronze stores raw data with minimal modification.

```bash
python src/bronze_ingest.py
```

Output:

```text
data/bronze/spam_raw.csv
```

### 12.2 Silver Layer

Silver contains cleaned and standardized data.

```bash
python src/silver_transform.py
```

Output:

```text
data/silver/spam_clean.parquet
```

Required schema:

```text
text
label
```

### 12.3 Gold Layer

Gold contains model-ready train/test datasets.

```bash
python src/gold_build.py
```

Output:

```text
data/gold/train.parquet
data/gold/test.parquet
```

---

## 13. Model Training

### 13.1 Start MLflow UI

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

### 13.2 Train Model

```bash
python src/train.py
```

The training pipeline will:

* Load Gold train/test data
* Apply text preprocessing
* Convert text to TF-IDF features
* Train Logistic Regression model
* Evaluate model
* Log parameters and metrics to MLflow
* Save the best model artifact

Output:

```text
models/best_spam_classifier.pkl
models/model_metadata.json
reports/classification_report.txt
reports/confusion_matrix.txt
```

---

## 14. Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Classification Report

Main selection metric:

```text
F1-score
```

Important business consideration:

```text
False positives should be minimized because ham emails are legitimate messages and should not be incorrectly classified as spam.
```

---

## 15. Run FastAPI Backend

Start the API server:

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

### 15.1 Health Check

Endpoint:

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### 15.2 Predict Endpoint

Endpoint:

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

## 16. Run Streamlit Frontend

Start Streamlit:

```bash
streamlit run app/streamlit_app.py --server.port 8501
```

Open:

```text
http://127.0.0.1:8501
```

The Streamlit UI allows users to:

* Enter email subject
* Enter email body
* Select example emails
* Click Predict
* View prediction label
* View spam probability

---

## 17. Prediction Logging

Each successful prediction is logged to:

```text
logs/predictions.csv
```

Log schema:

```csv
timestamp,subject,body,prediction,spam_probability
```

Example:

```csv
2026-04-24T20:00:00,Congratulations,You won a free prize,spam,0.94
```

---

## 18. Run Tests

Install test dependencies:

```bash
pip install pytest
```

Run all tests:

```bash
pytest
```

Run specific tests:

```bash
pytest tests/test_predict.py
pytest tests/test_api.py
```

---

## 19. Docker Deployment

### 19.1 Build and Run

```bash
docker compose up -d --build
```

### 19.2 Services

| Service   | URL                          |
| --------- | ---------------------------- |
| FastAPI   | `http://localhost:8000/docs` |
| Streamlit | `http://localhost:8501`      |

### 19.3 Stop Services

```bash
docker compose down
```

---

## 20. Sprint Plan

| Sprint   | Timeline      | Goal                                                |
| -------- | ------------- | --------------------------------------------------- |
| Sprint 1 | 23/04 - 29/04 | Setup, architecture, EDA, QA plan, baseline demo    |
| Sprint 2 | 30/04 - 06/05 | Medallion data pipeline and baseline experiments    |
| Sprint 3 | 07/05 - 13/05 | Training pipeline, MLflow, best model, API skeleton |
| Sprint 4 | 14/05 - 20/05 | FastAPI, Streamlit, logging, integration testing    |
| Sprint 5 | 21/05 - 25/05 | Docker, deployment, final QA, documentation, demo   |

---

## 21. Functional Deliverables

| Deliverable                             | Owner                           |
| --------------------------------------- | ------------------------------- |
| GitHub repository and project structure | Tech Lead                       |
| FSD and TSD                             | Tech Lead                       |
| Dataset and EDA                         | AI Engineer — Data              |
| Bronze/Silver/Gold data pipeline        | AI Engineer — Data              |
| Baseline model notebooks                | AI Engineer — Training Pipeline |
| Training pipeline with MLflow           | AI Engineer — Training Pipeline |
| Best model artifact                     | AI Engineer — Training Pipeline |
| FastAPI backend                         | AI Engineer — Model Serving     |
| Streamlit frontend                      | AI Engineer — Model Serving     |
| Prediction logging                      | AI Engineer — Model Serving     |
| Test plan and final QA report           | QA / Reviewer                   |
| Docker deployment                       | Tech Lead                       |

---

## 22. Definition of Done

The project is considered complete when:

* Raw dataset is collected
* Bronze, Silver, and Gold data layers are created
* Baseline demo runs successfully
* Training pipeline runs from Gold dataset
* MLflow logs model experiments
* Best model artifact is saved
* FastAPI `/predict` endpoint works
* Streamlit UI works
* Prediction logs are saved
* QA test report is completed
* Docker Compose runs the full application
* README and documentation are complete

---

## 23. Git Workflow

### Branch Strategy

```text
main
develop
feature/*
fix/*
docs/*
deploy/*
```

### Example Branches

```text
feature/data-medallion-pipeline
feature/baseline-model
feature/mlflow-training
feature/fastapi-serving
feature/streamlit-ui
feature/prediction-logging
deploy/docker-compose
docs/fsd-tsd
```

### Commit Convention

```text
feat: add new feature
fix: fix bug
docs: update documentation
refactor: improve code structure
test: add or update tests
chore: update configs or dependencies
deploy: update deployment files
```

Example:

```bash
git commit -m "feat: add logistic regression training pipeline"
```

---

## 24. QA Scope

QA verifies:

* Data schema
* Label mapping
* Data quality
* Model loading
* Prediction output
* API response format
* UI behavior
* Prediction logging
* Docker run
* Final demo readiness

QA outputs:

```text
docs/test_plan.md
tests/test_predict.py
tests/test_api.py
reports/final_test_report.md
```

---

## 25. Future Improvements

Possible future improvements:

* Gmail API integration
* Phishing URL detection
* Email attachment scanning
* Model drift monitoring
* Dashboard for prediction logs
* Advanced model: Linear SVM, XGBoost, DistilBERT
* Cloud deployment on EC2, Azure App Service, or Hugging Face Spaces
* CI/CD with GitHub Actions

---

## 26. License

This project is created for educational purposes as part of an AI engineering team project.
