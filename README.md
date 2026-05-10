# End-to-End Email Spam Classification (MLOps demo)

Binary email-spam classifier (`0 = ham`, `1 = spam`) built as an MVP MLOps
system: synthetic monthly partitions → Medallion data lake → MLflow-tracked
training → Model Registry promotion → FastAPI/Streamlit serving with hot
reload, all orchestrated by Airflow.

> **Project status (10/May/2026)**
>
> | Sprint | Window | Status | Backlog |
> |---|---|---|---|
> | Sprint 1 | 21/Apr → 09/May/2026 | ✅ Done — setup, Medallion ingest, baseline TF-IDF + LR, Gmail draft | [Jira.csv](docs/Jira.csv) (rows for `ACW3 Sprint 1`) |
> | **Sprint 2** | **10/May → 24/May/2026** | 🚧 In progress — drift, C/C, Airflow DAGs, FastAPI/Streamlit, Gmail poller, EC2 + CI/CD | [docs/Sprint_2_Tickets.md](docs/Sprint_2_Tickets.md) — Epic `ACW3-61 [EPIC2] MLOPS PIPELINE & DEPLOYMENT` |
>
> Project chỉ chạy **2 sprint** (không chia thêm sprint nhỏ). Đầu mối tài liệu: [docs/INDEX.md](docs/INDEX.md).

> **Two parallel tracks** in this repo:
>
> 1. **Production pipeline** (this README): Medallion + Airflow + MLflow + drift +
>    champion/challenger. Trained via `src/pipelines/initial_load.py`.
> 2. **Baseline POC**: model trained from
>    [notebooks/baseline/Email_Classification_V2.ipynb](notebooks/baseline/Email_Classification_V2.ipynb),
>    deployed to a single EC2 with FastAPI + Streamlit (no Airflow / MLflow).
>    See [docs/DEPLOY_BASELINE_EC2.md](docs/DEPLOY_BASELINE_EC2.md) and
>    [docs/SETUP_GMAIL_API.md](docs/SETUP_GMAIL_API.md) (Gmail integration).
>    Serving uses [src/notebook_preprocessing.py](src/notebook_preprocessing.py)
>    instead of the canonical [src/text_preprocessing.py](src/text_preprocessing.py)
>    — controlled by `PREPROCESS_PIPELINE=notebook` env var on `src/predict.py`.

## 1. Architecture

```
Raw 193k email CSV
        │
        ▼  (synthetic timestamps + concept drift)
data/raw/by_month/YYYY-MM.csv
        │
        ▼  bronze_ingest
data/bronze/by_month/month_partition=YYYY-MM/data.parquet
        │
        ▼  silver_transform  (cleaning + canonical preprocess_text)
data/silver/by_month/month_partition=YYYY-MM/data.parquet
        │
        ▼  gold_build (window of months)
data/gold/runs/<run_name>/{train,test}.parquet
        │
        ▼
┌──────────── Airflow ────────────┐
│ initial_full_load   (one-shot)  │
│ monthly_email_ml_pipeline (@m)  │
│   • monthly_run.py              │
│   • drift check vs ref window   │
│   • retrain if drift_score ≥ τ  │
│   • MLflow log + register +     │
│     promote Production          │
│   • POST /admin/reload-model    │
└──────────────┬──────────────────┘
               ▼
   ┌──── MLflow Registry ────┐
   │  email-spam-classifier  │
   │  v1, v2, …              │
   │  alias / stage = prod   │
   └──────────┬──────────────┘
              ▼
   ┌────── FastAPI ────────┐         ┌──── Streamlit ────┐
   │ GET  /health          │ ◀────── │ subject + body    │
   │ POST /predict         │         │ → POST /predict   │
   │ POST /admin/reload    │         └───────────────────┘
   └──────────┬────────────┘
              ▼
       logs/predictions.csv
```

GitHub Actions does **CI/CD only** (test + deploy code/DAGs). Airflow owns
all training orchestration.

## 2. Tech Stack

| Layer | Tech |
|---|---|
| Language | Python 3.11 |
| Data | pandas, pyarrow, parquet, Hive partitioning |
| ML | scikit-learn (`Pipeline(TfidfVectorizer + LogisticRegression)`) |
| Tracking & registry | MLflow |
| Orchestration | Airflow |
| Serving | FastAPI + Uvicorn |
| UI | Streamlit |
| Tests | Pytest + FastAPI TestClient |
| Deployment | Docker, docker-compose, systemd on EC2 |
| CI/CD | GitHub Actions (`ci.yml`, `deploy.yml`) |

## 3. Repository Layout

```
email-spam-classification/
├── app/                            # Serving stack (FastAPI + Streamlit + Gmail poller)
│   ├── api.py                      # FastAPI service
│   ├── streamlit_app.py            # Streamlit UI
│   ├── Dockerfile                  # serving image (python 3.11-slim)
│   └── docker-compose.yml          # api + streamlit + gmail-poller
├── airflow/                        # Local-dev Airflow 3.x stack (production = systemd on EC2)
│   ├── Dockerfile                  # extends apache/airflow:3.0.4 + project requirements
│   ├── docker-compose.yml          # postgres + apiserver + scheduler + dag-processor
│   └── .env.example                # AIRFLOW_UID, FERNET_KEY, JWT_SECRET, …
├── dags/                           # Airflow DAGs (mounted by both stacks)
│   ├── initial_full_load_dag.py
│   └── monthly_email_ml_pipeline.py
├── data/
│   ├── raw/                        # raw CSV + raw/by_month/
│   ├── bronze/by_month/
│   ├── silver/by_month/
│   └── gold/runs/<name>/
├── docs/                           # FSD, sprint plan, conventions, advisor PDFs
├── models/                         # local pkl + metadata.json (offline fallback)
├── notebooks/
│   ├── 01_eda.ipynb
│   └── baseline/Email_Classification_V2.ipynb
├── reports/monthly_runs/<month>/   # drift_report.json + decision.json + champion_challenger.json
├── scripts/
│   ├── run_monthly_training.sh     # manual fallback (Airflow is the prod trigger)
│   ├── gmail_oauth_bootstrap.py    # OAuth flow → app/secrets/gmail_token.json
│   └── systemd/                    # mlflow, fastapi, airflow-{webserver,scheduler}
├── src/
│   ├── etl/
│   │   ├── raw_partition.py        # synthetic monthly split + drift injection
│   │   ├── bronze_ingest.py
│   │   ├── silver_transform.py
│   │   └── gold_build.py
│   ├── monitoring/
│   │   └── drift_detector.py       # chi-square label drift + KS+PSI feature drift
│   ├── pipelines/
│   │   ├── initial_load.py         # bootstrap champion v1
│   │   └── monthly_run.py          # ingest + drift + champion/challenger per month
│   ├── serving/
│   │   ├── gmail_client.py         # OAuth + Gmail API wrapper
│   │   └── gmail_poller.py         # 30s loop → /predict → label AI_SPAM/AI_HAM
│   ├── text_preprocessing.py       # canonical preprocess used at train AND inference
│   ├── train.py                    # thin CLI wrapper → pipelines.initial_load
│   ├── evaluate.py
│   ├── tune.py                     # GridSearchCV over the unified Pipeline
│   └── predict.py                  # canonical inference (MLflow Registry → pkl fallback)
├── tests/                          # preprocessing, ETL, predict, API, drift, monthly_run
├── .github/workflows/
│   ├── ci.yml                      # lint + pytest + docker build
│   └── deploy.yml                  # rsync code+dags to EC2 on push to main
├── pyproject.toml                  # black/isort/pytest config
├── .pre-commit-config.yaml         # black + isort + flake8 hooks
├── .flake8                         # max-line=120, ignore E203/W503
├── .env.example
└── requirements.txt
```

## 4. Local quick-start

> **Python 3.11 only.** A 3.12+ venv will not match the Dockerfile or CI matrix.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pre-commit install   # optional: enable formatting hooks before each commit
cp .env.example .env

# 1) Partition raw CSV by month + inject drift
python -m src.etl.raw_partition --raw-input data/raw/spam_Emails_data.csv

# 2) Bronze + Silver for everything
python -m src.etl.bronze_ingest --all
python -m src.etl.silver_transform --all

# 3) Train champion v1 on the first 6 months → MLflow
python -m src.pipelines.initial_load --start-month 2025-05 --end-month 2025-10

# 4) Serve
uvicorn app.api:app --reload --port 8000
streamlit run app/streamlit_app.py --server.port 8501
```

Test:
```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
     -d '{"subject":"Free iPhone","body":"Click here now to claim"}'
```

## 5. Monthly loop (Airflow on EC2)

In the Airflow UI, trigger `monthly_email_ml_pipeline` with config:

```json
{"month": "2025-11", "ref_start": "2025-05", "ref_end": "2025-10"}
```

The DAG runs `src.pipelines.monthly_run` end-to-end:
ingest → drift check → (if `drift_score ≥ DRIFT_SCORE_THRESHOLD`) train challenger
→ champion/challenger comparison on the new month's gold test set → promote
challenger only if `challenger.f1 ≥ champion.f1 + PROMOTION_F1_DELTA` (default 0.0)
→ `POST /admin/reload-model`. On rejection, the local `models/best_spam_classifier.pkl`
is restored from a pre-train backup so `/predict` fallback keeps using the champion.

Per-month artefacts persisted under `reports/monthly_runs/<month>/`:
`drift_report.json`, `decision.json`, `champion_challenger.json`.

Repeat with `2025-12`, `2026-01`, … to replay a year of production in minutes.

## 6. Model contract

| Field | Value |
|---|---|
| Pipeline | `Pipeline(TfidfVectorizer(sublinear_tf=True, ngram=(1,2), min_df=2) + LogisticRegression)` |
| Training input | `processed_text` from silver layer (single source of truth via `src/text_preprocessing.py`) |
| Threshold selection | smallest `t` s.t. `precision(t) ≥ TARGET_PRECISION` (default 0.99 — minimize false positives, FSD §11.2) |
| Primary metric | F1 + Average Precision (AUC-PR); accuracy is **not** a selection metric |
| Registry | `email-spam-classifier`, alias/stage `Production` |

## 7. API contract

```
POST /predict
{ "subject": "Congratulations", "body": "You won..." }
→ { "label": "spam", "spam_probability": 0.94, "threshold": 0.99 }

GET  /health
POST /admin/reload-model      # called at the end of every monthly DAG run
```

Predictions append to `logs/predictions.csv`:
`timestamp, subject, body, prediction, spam_probability`.

## 8. Tests

```bash
pytest -q
```
Covers `src/text_preprocessing.py`, the full ETL pipeline (toy dataset),
`src/predict.py` (with a tiny in-memory model), and the FastAPI endpoints.

## 9. Docker

The serving stack and the Airflow stack are **two independent compose files**.
Production EC2 still uses systemd for Airflow + MLflow; Docker is for local dev.

### Serving stack (FastAPI + Streamlit + Gmail poller)

```bash
docker compose -f app/docker-compose.yml up -d --build
# add the gmail poller (requires app/secrets/gmail_token.json from OAuth bootstrap)
docker compose -f app/docker-compose.yml --profile gmail up -d
```

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |

### Airflow 3.x local stack

```bash
cp airflow/.env.example airflow/.env
# fill in AIRFLOW_FERNET_KEY, AIRFLOW_SECRET_KEY, AIRFLOW_JWT_SECRET
# (one-liners to generate are in .env.example)

# NOTE: docker compose looks for .env in the cwd, not next to the compose file.
# Always pass --env-file, or cd into airflow/ first.
docker compose --env-file airflow/.env -f airflow/docker-compose.yml up airflow-init
docker compose --env-file airflow/.env -f airflow/docker-compose.yml up -d
```

| Service | URL |
|---|---|
| Airflow UI | http://localhost:8080 (admin / admin) |

The Airflow image extends `apache/airflow:3.0.4-python3.11` with this project's
`requirements.txt` so DAG tasks can run `python -m src.pipelines.monthly_run`
directly without an external venv.

## 10. Roles

| Role | Scope |
|---|---|
| Tech Lead / Solution Architect | Architecture, PR review, Airflow/Docker integration, deploy, demo |
| QA / QC Engineer | Acceptance criteria, test cases, issue log, final QA report |
| AI Data Engineer | Dataset, EDA, Medallion pipeline, Dataset Quality & Hygiene |
| AI Pipeline / MLOps Engineer | Training workflow, MLflow registry, Airflow DAGs, monthly report |
| AI Model Serving Engineer | `predict.py`, FastAPI, Streamlit, prediction logging, model version |

## 11. Definition of Done

- Medallion pipeline produces bronze, silver, gold for at least one month window
- `initial_full_load` DAG trains champion v1 and promotes it to MLflow Production
- `monthly_email_ml_pipeline` DAG runs end-to-end for at least one new month
- FastAPI `/predict` returns the contract response and `/admin/reload-model` works
- Streamlit UI predicts via the API and shows label + probability
- `logs/predictions.csv` is populated
- `pytest` is green (43 tests)
- `docker compose -f app/docker-compose.yml up` runs api + streamlit
- `docker compose -f airflow/docker-compose.yml up` runs Airflow 3.x + MLflow
- README reflects current state

## 12. Gmail near-real-time filter (online flow)

Two flows live side-by-side:

```
Offline                          Online
───────                          ───────
Airflow monthly DAG              Gmail poller (30s loop)
  → train + promote              → /predict (FastAPI)
  → MLflow Production            → apply Gmail label AI_SPAM / AI_HAM
       │                         → logs/gmail_predictions.csv
       └─────────► FastAPI ◄──────────┘
```

The poller is a separate container ([app/gmail_poller.py](app/gmail_poller.py))
that uses the same FastAPI `/predict` as the UI — so it always picks up the
latest Production model after `/admin/reload-model` is called.

### One-time OAuth bootstrap

1. Google Cloud Console → enable Gmail API → create OAuth 2.0 client of type
   **Desktop app** → download the JSON to `app/secrets/gmail_credentials.json`.
2. From your dev machine:
   ```bash
   python scripts/gmail_oauth_bootstrap.py
   ```
   A browser opens; grant the requested scopes. Output: `app/secrets/gmail_token.json`.
3. The `app/secrets/` folder is gitignored. For EC2/docker, copy `gmail_token.json`
   to the host and let the compose volume mount it read-only.

### Run the poller (Docker)

```bash
docker compose --profile gmail up -d gmail-poller
docker compose logs -f gmail-poller
```

The first tick calls `users.getProfile` to pin a starting `historyId` —
**no backfill of the existing inbox**. From then on, only new messages are
labeled. State is persisted to `logs/gmail_state.json` so restarts resume.

### Run locally (no Docker)

```bash
export GMAIL_TOKEN_JSON=app/secrets/gmail_token.json
export PYTHONPATH=$PWD
export API_URL=http://127.0.0.1:8000/predict
python -m app.gmail_poller
```

### Production-like extension (Sprint 5)

Replace polling with **Gmail Push Notifications** via Google Cloud Pub/Sub
(`users.watch` → Pub/Sub topic → push subscription → public HTTPS webhook
on FastAPI). The classifier + label-application logic stays the same — only
the *trigger* changes from a 30s tick to an event.

## 13. Documentation map

Tất cả tài liệu sống ở [docs/](docs/). Đầu mối: [docs/INDEX.md](docs/INDEX.md).

| File | Mục đích |
|---|---|
| [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md) | Single-doc brief đầy đủ (architecture, data, ML, drift, C/C, DAG, serving, contracts) |
| [docs/Sprint_Planning_Email_Spam_Classification.pdf](docs/Sprint_Planning_Email_Spam_Classification.pdf) | Sprint plan tổng quan (Sprint 1 + 2) |
| [docs/Sprint_2_Tickets.md](docs/Sprint_2_Tickets.md) | Backlog Sprint 2 (Epic `ACW3-61` + 30 ticket con) |
| [docs/Jira.csv](docs/Jira.csv) | Export Jira (Sprint 1 tasks done) |
| [docs/FSD_Functional_Tasks_Email_Spam_Classification.pdf](docs/FSD_Functional_Tasks_Email_Spam_Classification.pdf) | Functional Spec (FSD §11.2 threshold, §4.2 gold contract, …) |
| [docs/DEPLOY_BASELINE_EC2.md](docs/DEPLOY_BASELINE_EC2.md) | Deploy baseline FastAPI + Streamlit lên EC2 (systemd) |
| [docs/SETUP_GMAIL_API.md](docs/SETUP_GMAIL_API.md) | Setup Gmail API + OAuth bootstrap |
| [docs/SETUP_CLOUDFRONT.md](docs/SETUP_CLOUDFRONT.md) | CloudFront HTTPS + custom domain + EC2 lockdown |
| [docs/QA_Test_Plan_Sprint2.md](docs/QA_Test_Plan_Sprint2.md) | QA test plan Sprint 2 (≥ 40 cases) |
| [docs/QA_Final_Report.md](docs/QA_Final_Report.md) | QA sign-off cuối Sprint 2 (skeleton — fill khi close sprint) |
| [docs/Risk_Register.md](docs/Risk_Register.md) | Risk register sống chung 2 sprint |
| [docs/Final_Demo.md](docs/Final_Demo.md) | Outline buổi demo cuối (skeleton — fill 23-24/May) |
| [docs/coding_and_git_convention.md](docs/coding_and_git_convention.md) | Coding + Git workflow convention |
| [docs/advisor_*.pdf](docs/) | Advisor materials per role (Tech Lead, QA, Data, Pipeline, Model) — **read-only** |

## 14. Future Improvements (post Warm Up — không thuộc Sprint 1/2)

- Shadow scoring: log predictions of challenger alongside champion before promotion
- Gmail Pub/Sub push subscription replacing polling
- Phishing-URL feature engineering on top of TF-IDF
- Hugging Face Spaces / EC2 / Azure App Service deployment
- Drift dashboard fed from `reports/monthly_runs/<month>/drift_report.json`
