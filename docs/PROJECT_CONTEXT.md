# Project Context: `email-spam-classification`

> Single-document brief for an LLM (or new engineer) to understand the entire
> project end-to-end. Paste this into a chat to enable accurate Q&A about the
> codebase, runtime behavior, contracts, and deployment.
>
> **Last updated**: 10/May/2026 (start of Sprint 2)

---

## 0. Sprint timeline (AIO Conquer Warm Up 3 — 2 sprint duy nhất)

| Sprint | Window | Epic | Status | Output |
|---|---|---|---|---|
| Sprint 1 | 21/Apr → 09/May/2026 | `ACW3-13 [EPIC1] WARMUP` | ✅ Done | Repo setup, role boundary, convention, Medallion ETL skeleton, EDA, baseline TF-IDF + LR, draft Gmail API, weekly sync ×1 |
| **Sprint 2** | **10/May → 24/May/2026** | `ACW3-61 [EPIC2] MLOPS PIPELINE & DEPLOYMENT` | 🚧 Planned | Drift detector, Champion/Challenger, Airflow 3.x DAGs, FastAPI + Streamlit, Gmail poller, 2 Docker stacks, EC2 + systemd, CI/CD, ≥ 40 tests, final demo |

Backlog đầy đủ Sprint 2: [Sprint_2_Tickets.md](Sprint_2_Tickets.md) (30 ticket con + 10 sync/retro).

> Project chỉ chạy **đúng 2 sprint**. Các hạng mục trong §19 (Known gaps) **không** thuộc Warm Up.

---

## 1. What this project is

End-to-end MLOps demo for **binary email spam classification** (`0 = ham`,
`1 = spam`). Built as an MVP MLOps system with:

- Synthetic monthly data partitions (12 months: 2025-05 → 2026-04)
  with injected concept drift starting **2025-11**.
- Medallion data lake (Raw → Bronze → Silver → Gold).
- MLflow-tracked training, model registry, champion/challenger promotion.
- FastAPI + Streamlit serving with hot-reload after every retrain.
- Airflow 3.x orchestration of the monthly retrain loop.
- Optional Gmail near-real-time polling worker (30s loop) for live demo.
- Two independent Docker stacks for local dev; systemd-on-EC2 for prod.

**Model**: a single `sklearn.Pipeline = TfidfVectorizer + LogisticRegression`.
No deep learning, no XGBoost (intentionally removed to keep the image lean).

---

## 2. Repository layout

```
email-spam-classification/
├── app/                              # Serving Docker stack
│   ├── api.py                        # FastAPI service (132 lines)
│   ├── streamlit_app.py              # Streamlit UI (75 lines)
│   ├── Dockerfile                    # python:3.11-slim + project deps
│   └── docker-compose.yml            # api + streamlit + gmail-poller (opt-in profile)
├── airflow/                          # Orchestration Docker stack (Airflow 3.x)
│   ├── Dockerfile                    # extends apache/airflow:3.0.4-python3.11
│   ├── docker-compose.yml            # postgres + airflow-init + apiserver +
│   │                                 # scheduler + dag-processor + mlflow
│   ├── .env.example                  # FERNET_KEY, JWT_SECRET, SECRET_KEY templates
│   └── simple_auth_manager_passwords.json.generated  # admin:admin (gitignored)
├── dags/                             # Airflow DAGs
│   ├── initial_full_load_dag.py      # one-shot bootstrap
│   └── monthly_email_ml_pipeline.py  # multi-task: bronze → silver → drift → C/C → reload
├── data/
│   ├── raw/by_month/YYYY-MM.csv
│   ├── bronze/by_month/month_partition=YYYY-MM/data.parquet
│   ├── silver/by_month/month_partition=YYYY-MM/{data.parquet, quality_report.json}
│   └── gold/runs/<run_name>/{train.parquet, test.parquet, metadata.json}
├── docs/                             # FSD, Sprint Plan, conventions, advisor PDFs
├── models/                           # local pkl + metadata.json (offline fallback)
├── mlruns/                           # MLflow file store (host-mounted from container)
├── notebooks/                        # 01_eda.ipynb + baseline/Email_Classification_V2.ipynb
├── reports/
│   ├── monthly_runs/<month>/         # drift_report.json + decision.json + champion_challenger.json
│   └── silver_quality/<month>.json
├── scripts/
│   ├── run_monthly_training.sh       # manual fallback (Airflow is the prod trigger)
│   ├── gmail_oauth_bootstrap.py      # OAuth flow → app/secrets/gmail_token.json
│   └── systemd/                      # mlflow + fastapi + airflow-{webserver,scheduler}
├── src/
│   ├── etl/
│   │   ├── raw_partition.py          # synthetic monthly split + drift injection
│   │   ├── bronze_ingest.py          # CSV → parquet + ingestion metadata
│   │   ├── silver_transform.py       # validation + canonical preprocess + quality report
│   │   └── gold_build.py             # silver → train/test parquet (stratified split)
│   ├── monitoring/
│   │   └── drift_detector.py         # chi-square label drift + KS+PSI feature drift
│   ├── pipelines/
│   │   ├── initial_load.py           # bootstrap champion v1 → MLflow Production
│   │   └── monthly_run.py            # ingest + drift + champion/challenger
│   ├── serving/
│   │   ├── gmail_client.py           # OAuth + Gmail API wrapper
│   │   └── gmail_poller.py           # 30s loop → /predict → label AI_SPAM/AI_HAM
│   ├── text_preprocessing.py         # canonical preprocess (train + inference)
│   ├── predict.py                    # MLflow Registry → pkl fallback
│   ├── train.py                      # thin CLI wrapper → pipelines.initial_load
│   ├── evaluate.py                   # CLI: evaluate a saved pkl on a parquet
│   └── tune.py                       # GridSearchCV over the unified Pipeline
├── tests/                            # 43 tests covering preprocessing, ETL,
│   │                                 # drift, monthly_run, predict, API, gmail
│   ├── test_data_pipeline.py
│   ├── test_drift_detector.py
│   ├── test_monthly_run_integration.py
│   ├── test_predict.py
│   ├── test_text_preprocessing.py
│   ├── test_api.py
│   └── test_gmail_poller.py
├── .github/workflows/
│   ├── ci.yml                        # flake8 (strict) + pytest + docker build
│   └── deploy.yml                    # rsync code+dags to EC2 on push to main
├── pyproject.toml                    # black/isort/pytest config
├── .pre-commit-config.yaml           # black + isort + flake8 hooks
├── .flake8                           # max-line=120, ignore E203/W503
├── requirements.txt
└── README.md
```

---

## 3. Architecture overview

```
Raw 193k email CSV
        │
        ▼  raw_partition  (synthetic timestamps + concept drift after 2025-11)
data/raw/by_month/YYYY-MM.csv
        │
        ▼  bronze_ingest
data/bronze/by_month/month_partition=YYYY-MM/data.parquet
        │
        ▼  silver_transform  (schema + label + null + dup + empty validation)
data/silver/by_month/month_partition=YYYY-MM/{data.parquet, quality_report.json}
        │
        ▼  gold_build (stratified train/test split for a window of months)
data/gold/runs/<run_name>/{train,test}.parquet
        │
        ▼
┌──────────── Airflow 3.x ─────────────────┐
│ initial_full_load   (one-shot)           │  →  MLflow champion v1
│ monthly_email_ml_pipeline (@monthly)     │
│   bronze → silver → drift_check          │
│   → branch_retrain (BranchPythonOperator)│
│   → [champion_challenger | skip_retrain] │
│   → reload_fastapi                       │
└──────────────┬───────────────────────────┘
               ▼
   ┌──── MLflow Registry ────┐
   │  email-spam-classifier  │
   │  v1, v2, …              │
   │  alias = production     │
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

**Online flow (opt-in)**: `app/gmail_poller.py` runs a 30s loop,
calls `/predict`, applies Gmail labels `AI_SPAM` / `AI_HAM`, logs to
`logs/gmail_predictions.csv`. Same FastAPI as the UI → always picks up the
latest model after `/admin/reload-model`.

---

## 4. Data layer details (Medallion)

### 4.1 Raw
- One CSV per month: `data/raw/by_month/YYYY-MM.csv`
- Schema: `text`, `label`, `received_at`
- Generated by `src/etl/raw_partition.py` from the original 193k-row CSV
  (synthetic timestamps + 5% concept drift injected for spam after 2025-11
  via the marker token `aigenphishalert`).

### 4.2 Bronze
- One parquet per month: `data/bronze/by_month/month_partition=YYYY-MM/data.parquet`
- Adds: `ingestion_timestamp`, `source_file`. **No transformation**.
- Idempotent — re-running for the same month overwrites cleanly.

### 4.3 Silver — clean, validated, conformed
- One parquet per month + quality report.
- **Required columns**: `["text", "label"]` (asserted; missing → `SilverValidationError`).
- **Drop reasons** (each counted in `quality_report.json`):
  - `null_text`
  - `duplicate_text` (within the month)
  - `invalid_label` (must map via `LABEL_MAPPING = {"ham","Ham","not_spam","spam","Spam"}`)
  - `empty_processed_text` (preprocessing produced `""`)
- Adds: `cleaned_text`, `processed_text` (from canonical
  `src/text_preprocessing.py`).
- Quality report persisted twice:
  - Beside partition: `data/silver/by_month/month_partition=YYYY-MM/quality_report.json`
  - Aggregate: `reports/silver_quality/YYYY-MM.json`

Quality report schema:
```json
{
  "month": "2025-11",
  "rows_in": 16234,
  "rows_out": 15998,
  "dropped": {
    "null_text": 12,
    "duplicate_text": 38,
    "invalid_label": 7,
    "empty_processed_text": 179
  },
  "label_distribution": {"ham": 10500, "spam": 5498},
  "valid_labels": ["Ham", "Spam", "ham", "not_spam", "spam"],
  "generated_at": "..."
}
```

### 4.4 Gold — model-ready
- `data/gold/runs/<run_name>/{train.parquet, test.parquet, metadata.json}`
- Stratified `train_test_split` (default `test_size=0.2`, `random_state=42`).
- Encodes label to `label_encoded` (0/1).
- Drops rows with empty `processed_text` (defensive — silver should already).
- Per FSD §4.2, gold contains `text + label` (+ optional helpers like
  `processed_text`). **No TF-IDF here** — TF-IDF is fit inside the training
  Pipeline at train time to prevent train/serve skew.

---

## 5. ML layer

### 5.1 Pipeline (canonical, single .pkl)

```python
Pipeline([
    ("tfidf",      TfidfVectorizer(max_features=50000,
                                   ngram_range=(1, 2),
                                   sublinear_tf=True,
                                   min_df=2,
                                   strip_accents="unicode")),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000,
                                      random_state=42, n_jobs=-1)),
])
```

### 5.2 Training input
- `processed_text` from the silver layer (single source of truth via
  `src/text_preprocessing.py`).

### 5.3 Threshold selection (FSD §11.2)
- Smallest `t` such that `precision(t) ≥ TARGET_PRECISION` (default `0.99`)
  on the validation set's PR curve. Implemented in
  `pipelines.initial_load.select_threshold`.
- **Goal**: minimize false positives (legitimate emails marked as spam).

### 5.4 Metrics tracked in MLflow
- `f1`, `precision`, `recall`, `ap_score` (Average Precision = AUC-PR)
- After threshold pick: `f1_at_threshold`, `precision_at_threshold`,
  `recall_at_threshold`
- Artifacts: `classification_report.txt`, `confusion_matrix.png`,
  `pr_curve.png`, `metadata.json`

### 5.5 MLflow Model Registry
- Registered model name: `email-spam-classifier`
- Promotion mechanism (in priority order — both supported for compat):
  1. Alias `production` (MLflow ≥ 2.9 — `set_registered_model_alias`)
  2. Stage `Production` (legacy — `transition_model_version_stage`)

---

## 6. Drift detection (`src/monitoring/drift_detector.py`)

Three families of checks:

### 6.1 Label drift — chi-square
- On `P(spam)` distribution between reference and new windows.
- `drifted = True` if `p_value < 0.05`.

### 6.2 Feature drift — KS-test + PSI
- Fit a small TF-IDF (`max_features=100`) on the reference texts.
- For each top feature: KS-test + PSI on the column means in both windows.
- A feature is "drifted" if `KS p_value < 0.05 AND PSI > 0.1`.
- PSI rule of thumb: `< 0.1 = no shift`, `< 0.25 = moderate`, `≥ 0.25 = significant`.

### 6.3 Aggregate score
```
feature_score = min(mean_psi / 0.25, 1.0)
label_score   = 1.0 if label_drift.drifted else 0.0
drift_score   = max(0.7 * feature_score + 0.3 * label_score, 0.0)
```
- `drift_score ∈ [0, 1]`
- Default threshold: `DRIFT_SCORE_THRESHOLD = 0.25` (env-overridable).

### 6.4 Output
`detect_drift()` returns:
```json
{
  "drift_score": 0.42,
  "label_drift":   { "chi2": ..., "p_value": ..., "drifted": true,  "ref_dist": {...}, "new_dist": {...} },
  "feature_drift": { "n_features": ..., "ks_drifted_features": ..., "mean_psi": ..., "max_psi": ..., "drifted_top_features": [...] },
  "ref_size": ...,
  "new_size": ...
}
```
Persisted to `reports/monthly_runs/<month>/drift_report.json`.

---

## 7. Champion / Challenger (`src/pipelines/monthly_run.py`)

Triggered when `drift_score ≥ DRIFT_SCORE_THRESHOLD`.

```
1. Backup local pkl + metadata.json   →  models/.backup_before_challenger/
2. Train challenger via initial_load(promote_to_production=False)
       → MLflow logs new version, no Production promotion
3. Load test set:
       data/gold/runs/initial_<ref_start>_to_<month>/test.parquet
4. Evaluate challenger on test set
5. Load champion from MLflow (alias "production" or stage "Production")
6. If no champion exists → auto-promote challenger
7. Else evaluate champion on the SAME test set (fair comparison)
8. Compare F1:
     if challenger.f1 ≥ champion.f1 + PROMOTION_F1_DELTA:
         promote challenger
     else:
         reject; restore local pkl from backup
9. Persist comparison → reports/monthly_runs/<month>/champion_challenger.json
```

- `PROMOTION_F1_DELTA` defaults to `0.0` (env-overridable).
- The local `best_spam_classifier.pkl` matters because `predict.py`
  falls back to it if MLflow is unreachable. Restoring it on rejection
  prevents the fallback from drifting away from the champion.

---

## 8. Orchestration — Airflow 3.x DAGs

### 8.1 `initial_full_load` (`dags/initial_full_load_dag.py`)
Manual trigger only. One-shot bootstrap.

```
partition_raw → bronze_all → silver_all → train_champion_v1 → reload_fastapi
```

### 8.2 `monthly_email_ml_pipeline` (`dags/monthly_email_ml_pipeline.py`)
- `schedule="@monthly"`, `catchup=False`, `max_active_runs=1`
- 7 discrete tasks (multi-task — each step retryable independently):

```
bronze_ingest (BashOperator)        — python -m src.etl.bronze_ingest --month {{ MONTH_TPL }}
        ↓
silver_transform (BashOperator)     — python -m src.etl.silver_transform --month {{ MONTH_TPL }}
        ↓
drift_check (PythonOperator)        — calls compute_drift_for_month(...); returns decision via XCom
        ↓
branch_retrain (BranchPythonOperator) — pulls XCom; routes to "champion_challenger" or "skip_retrain"
        ↓                                 ↓
champion_challenger (PythonOp)    skip_retrain (EmptyOperator)
        ↓                                 ↓
        └──────────┬──────────────────────┘
                   ↓
reload_fastapi (BashOperator, TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
                                          → curl POST /admin/reload-model
```

#### DAG Run Conf
```json
{"month": "2025-11", "ref_start": "2025-05", "ref_end": "2025-10", "promotion_delta": 0.0}
```

### 8.3 Env-driven DAG portability
DAGs work identically on EC2 systemd and inside Docker via:
- `EMAIL_SPAM_PROJECT_DIR` — code root
- `EMAIL_SPAM_VENV_ACT` — `source .../activate` on EC2; `true` (noop) in Docker
- `MLFLOW_TRACKING_URI` — `http://127.0.0.1:5000` on EC2; `http://mlflow:5000` in Docker
- `EMAIL_SPAM_API_RELOAD_URL` — `http://127.0.0.1:8000/admin/reload-model` on EC2;
  `http://host.docker.internal:8000/admin/reload-model` in Docker

---

## 9. Serving

### 9.1 FastAPI (`app/api.py`, port 8000)
| Endpoint | Behavior |
|---|---|
| `GET /health`            | `{"status": "ok", "model_loaded": bool, "model_version": "..."}` |
| `POST /predict`          | Body `{"subject", "body"}` → `{"label", "spam_probability", "threshold"}`. Appends to `logs/predictions.csv`. |
| `POST /admin/reload-model` | Re-loads pipeline from MLflow Registry alias `production` or local pkl. Idempotent. |

### 9.2 Predict resolution (`src/predict.py`)
1. **MLflow Registry** — `mlflow.sklearn.load_model("models:/email-spam-classifier@production")`
2. **Local pickle fallback** — `models/best_spam_classifier.pkl` (path via `MODEL_PATH` env)
3. Threshold default loaded from `models/model_metadata.json` (`MODEL_METADATA_PATH` env);
   can be overridden per-request.

### 9.3 Streamlit (`app/streamlit_app.py`, port 8501)
- Form for subject + body, calls `API_URL` (default `http://127.0.0.1:8000/predict`)
- Two preset examples (phishing spam / work meeting ham)
- Renders label + probability + threshold

### 9.4 Gmail poller (`app/gmail_poller.py`, opt-in profile)
- 30s polling loop using Gmail History API (`users.history.list`)
- First tick pins a `historyId` from `users.getProfile` → **no inbox backfill**
- For each new message: extract subject + body → `POST /predict` → apply
  Gmail label `AI_SPAM` or `AI_HAM` → log to `logs/gmail_predictions.csv`
- State persisted to `logs/gmail_state.json` so restarts resume.
- OAuth flow bootstrapped once via `scripts/gmail_oauth_bootstrap.py` on the
  dev machine; token JSON copied to host and mounted read-only.

---

## 10. Local dev — Docker

Two **independent** compose stacks. Cross-stack communication via
`host.docker.internal`.

### 10.1 Serving stack (`app/docker-compose.yml`)
| Service | Port | Notes |
|---|---|---|
| `api`           | 8000 | FastAPI; mounts `../models:ro` and `../logs` |
| `streamlit`     | 8501 | Streamlit UI; depends_on api |
| `gmail-poller`  | —    | profile `gmail`; mounts `../secrets:ro` |

```bash
docker compose -f app/docker-compose.yml up -d --build
docker compose -f app/docker-compose.yml --profile gmail up -d
```

### 10.2 Orchestration + MLflow stack (`airflow/docker-compose.yml`)
| Service | Port | Notes |
|---|---|---|
| `postgres`              | (internal) | Airflow metadata DB |
| `airflow-init`          | —          | One-shot `airflow db migrate` |
| `airflow-apiserver`     | 8080       | Airflow 3.x web + REST (replaces webserver) |
| `airflow-scheduler`     | (internal) | LocalExecutor |
| `airflow-dag-processor` | (internal) | DAG parser (separated in 3.x) |
| `mlflow`                | **5001** → 5000 | macOS port 5000 reserved by AirPlay → published on 5001. SQLite backend, artifacts mounted from `../mlruns`. |

```bash
cp airflow/.env.example airflow/.env   # then fill FERNET_KEY, JWT_SECRET, SECRET_KEY
docker compose --env-file airflow/.env -f airflow/docker-compose.yml up airflow-init
docker compose --env-file airflow/.env -f airflow/docker-compose.yml up -d
```

**Note**: docker compose loads `.env` from the **CWD**, not next to the
compose file. Always pass `--env-file airflow/.env` or `cd airflow/` first.

### 10.3 Airflow 3.x admin login
- Auth manager: `SimpleAuthManager` (default in 3.0; FAB moved to a provider).
- The `airflow users` CLI was **removed** in 3.0.
- Users defined by JSON file:
  `airflow/simple_auth_manager_passwords.json.generated` →
  `/opt/airflow/simple_auth_manager_passwords.json.generated` (mount, gitignored).
- Default content: `{"admin": "admin"}`. Edit + `down && up` to change.

---

## 11. Production — EC2 + systemd

`scripts/systemd/`:
- `mlflow.service` — `mlflow server --backend-store-uri sqlite:///mlflow.db`
- `email-spam-api.service` — `uvicorn app.api:app --host 0.0.0.0 --port 8000`
- `airflow-webserver.service` — `airflow webserver --port 8080` (or `apiserver` on 3.x)
- `airflow-scheduler.service` — `airflow scheduler`

`.github/workflows/deploy.yml` SSHs into EC2 and:
1. `git pull --ff-only origin main`
2. `pip install -r requirements.txt`
3. `rsync -a --delete dags/ $AIRFLOW_HOME/dags/email_spam/`
4. `systemctl restart email-spam-api airflow-{scheduler,webserver}`

**Secrets**: `EC2_HOST`, `EC2_USER`, `EC2_SSH_KEY` configured in repo settings.

---

## 12. CI/CD

### 12.1 `.github/workflows/ci.yml`
- Triggers: push to `main`/`develop`, PRs to same.
- Python 3.11 matrix.
- Steps: install deps → download NLTK assets → **flake8 (strict, no
  `continue-on-error`)** → pytest → docker build (only on push to `main`).

### 12.2 `.github/workflows/deploy.yml`
- Triggers: push to `main` touching `src/**`, `app/**`, `dags/**`,
  `airflow/**`, `scripts/**`, `requirements.txt`.
- Action: SSH-deploy code + DAGs + restart services on EC2.

---

## 13. Tech stack

| Layer | Tech |
|---|---|
| Language          | Python 3.11 (pinned: venv, Dockerfile, CI, `pyproject.toml`) |
| Data              | pandas, pyarrow, parquet, Hive-style partitioning |
| ML                | scikit-learn (`Pipeline(TfidfVectorizer + LogisticRegression)`) |
| Tracking & registry | MLflow 3.x |
| Orchestration     | Airflow 3.0.4 (LocalExecutor, SimpleAuthManager) |
| Serving           | FastAPI + Uvicorn |
| UI                | Streamlit |
| Gmail integration | google-api-python-client, google-auth, google-auth-oauthlib |
| Tests             | Pytest + FastAPI TestClient (httpx) |
| Quality           | black, isort, flake8, pre-commit |
| Deployment        | Docker, docker-compose, systemd on EC2 |
| CI/CD             | GitHub Actions |
| Drift             | scipy (chi-square + KS), custom PSI |

`requirements.txt` is **lean** — `xgboost`, `duckdb`, `python-dotenv`,
`PyYAML` were audited out as unused.

---

## 14. Configuration / env vars

| Env var | Default | Used by |
|---|---|---|
| `MODEL_PATH`                 | `/app/models/best_spam_classifier.pkl` | FastAPI predict.py |
| `MODEL_METADATA_PATH`        | `/app/models/model_metadata.json`      | FastAPI predict.py |
| `PREDICTION_LOG_PATH`        | `/app/logs/predictions.csv`            | FastAPI |
| `MLFLOW_TRACKING_URI`        | `http://mlflow:5000` (airflow) / `http://host.docker.internal:5001` (app) | Airflow + FastAPI |
| `DRIFT_SCORE_THRESHOLD`      | `0.25`                                 | monthly_run.py |
| `PROMOTION_F1_DELTA`         | `0.0`                                  | monthly_run.py |
| `EMAIL_SPAM_PROJECT_DIR`     | `/home/ubuntu/email-spam-classification` (EC2) / `/opt/airflow` (Docker) | DAGs |
| `EMAIL_SPAM_VENV_ACT`        | `source $EMAIL_SPAM_PROJECT_DIR/.venv/bin/activate` (EC2) / `true` (Docker) | DAGs |
| `EMAIL_SPAM_API_RELOAD_URL`  | `http://127.0.0.1:8000/admin/reload-model` (EC2) / `http://host.docker.internal:8000/admin/reload-model` (Docker) | DAGs |
| `GMAIL_TOKEN_JSON`           | `/app/secrets/gmail_token.json`        | gmail_poller.py |
| `GMAIL_STATE_PATH`           | `/app/logs/gmail_state.json`           | gmail_poller.py |
| `GMAIL_PREDICTION_LOG`       | `/app/logs/gmail_predictions.csv`      | gmail_poller.py |
| `GMAIL_POLL_INTERVAL_SECONDS`| `30`                                   | gmail_poller.py |
| `GMAIL_LABEL_SPAM`           | `AI_SPAM`                              | gmail_poller.py |
| `GMAIL_LABEL_HAM`            | `AI_HAM`                               | gmail_poller.py |
| `AIRFLOW_FERNET_KEY` / `AIRFLOW_JWT_SECRET` / `AIRFLOW_SECRET_KEY` | (must be set in `airflow/.env`) | Airflow |

---

## 15. Contracts

### 15.1 API contract (`POST /predict`)
**Request**:
```json
{"subject": "Congratulations, you won...", "body": "Click here to claim..."}
```
**Response**:
```json
{"label": "spam", "spam_probability": 0.94, "threshold": 0.99}
```
- `label ∈ {"spam", "not_spam"}`
- `spam_probability ∈ [0, 1]`
- `threshold` = the threshold the model was selected with (default 0.99)

### 15.2 DAG trigger conf (`monthly_email_ml_pipeline`)
```json
{"month": "2025-11", "ref_start": "2025-05", "ref_end": "2025-10", "promotion_delta": 0.0}
```

### 15.3 Model contract
| Field | Value |
|---|---|
| Pipeline           | `TfidfVectorizer(sublinear_tf, ngram=(1,2), min_df=2) + LogisticRegression` |
| Training input     | `processed_text` (silver layer) |
| Threshold          | smallest `t` s.t. `precision(t) ≥ TARGET_PRECISION` (default 0.99) |
| Primary metric     | F1 + Average Precision (AUC-PR). Accuracy is **not** a selection metric. |
| Registry model     | `email-spam-classifier`, alias `production` |
| Promotion gate     | `challenger.f1 ≥ champion.f1 + PROMOTION_F1_DELTA` |

---

## 16. Tests (43 total, all green)

| Test file | Coverage |
|---|---|
| `test_text_preprocessing.py`     | clean_text + preprocess_text + tokenize_and_lemmatize |
| `test_data_pipeline.py`          | raw_partition + drift injection + full ETL E2E + silver validation (schema, label, dups, empty, quality report) |
| `test_drift_detector.py`         | chi-square label drift + PSI + feature drift + aggregate score range |
| `test_predict.py`                | predict_spam contract + threshold override + empty input |
| `test_api.py`                    | FastAPI `/health` + `/predict` + `/admin/reload-model` via TestClient |
| `test_gmail_poller.py`           | Gmail polling loop + label application + state persistence |
| `test_monthly_run_integration.py`| ETL → drift → champion/challenger E2E (with mocks) + helper functions (`ingest_month`, `compute_drift_for_month`, `_evaluate_on_test`, `_backup/restore_local_artifacts`, alias `run_champion_challenger`) |

Run: `pytest -q`

---

## 17. Quality / dev tooling

- `pyproject.toml` — black (line-length 120), isort (profile=black), pytest config
- `.flake8` — max-line=120, ignore E203/W503, exclude .venv/data/models/etc.
- `.pre-commit-config.yaml` — black, isort, flake8 (with flake8-bugbear),
  trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files,
  detect-private-key
- Install: `pre-commit install` after `pip install -r requirements.txt`

---

## 18. Recent decisions / context

- **Champion/Challenger** added (was originally direct retrain+promote).
  Challenger evaluation uses the same gold test set as champion for fairness.
  Local pkl is backed up + restored on rejection.
- **Silver validation** added (was minimal — only `dropna(subset=["text"])`).
  Now full schema + label + null + dup + empty checks with reconciled
  drop counts in a quality report.
- **Multi-task DAG** for `monthly_email_ml_pipeline` (was 1 BashOperator
  calling the whole script). Each step is now its own Airflow task.
- **Airflow 3.x specifics** addressed:
  - `airflow users` CLI removed → SimpleAuthManager + JSON-file passwords.
  - apiserver replaces webserver; dag-processor is a separate component.
  - JWT secret required for REST API (`AIRFLOW__API_AUTH__JWT_SECRET`).
- **macOS port 5000** is reserved by AirPlay Receiver → MLflow exposed on 5001
  on host (still 5000 inside Docker network).
- **NLTK** was missing from `requirements.txt` despite being imported by
  `text_preprocessing.py` — added.
- **Removed unused deps**: `xgboost`, `duckdb`, `python-dotenv`, `PyYAML` —
  saved ~140 MB image size, cleaner CI.

---

## 19. Known gaps / Future improvements

> ⚠️ **Out of scope cho Sprint 1/2** — đây là post-Warm Up backlog, KHÔNG thuộc EPIC2.

- Shadow scoring (log challenger predictions alongside champion before promotion)
- Gmail Pub/Sub push subscription replacing polling
- Phishing-URL feature engineering on top of TF-IDF
- Drift dashboard fed from `reports/monthly_runs/<month>/drift_report.json`
- Production migration: SQLite → Postgres for both Airflow metadata and MLflow
  backend store
- `/metrics` Prometheus endpoint on FastAPI (currently only `/health`)

---

## 20. Quickstart commands

```bash
# 1. Recreate venv (Python 3.11)
rm -rf .venv && python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pre-commit install

# 2. Bootstrap data + champion v1 (only needed once on a fresh checkout)
python -m src.etl.raw_partition --raw-input data/raw/spam_Emails_data.csv
python -m src.etl.bronze_ingest --all
python -m src.etl.silver_transform --all
python -m src.pipelines.initial_load --start-month 2025-05 --end-month 2025-10

# 3. Run tests + lint
pytest -q
flake8 src app tests dags

# 4. Local dev — Docker
docker compose -f app/docker-compose.yml up -d --build
docker compose --env-file airflow/.env -f airflow/docker-compose.yml up airflow-init
docker compose --env-file airflow/.env -f airflow/docker-compose.yml up -d

# 5. Smoke
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' \
     -d '{"subject":"Free iPhone","body":"Click here now"}'
open http://localhost:8501   # Streamlit
open http://localhost:8080   # Airflow (admin/admin)
open http://localhost:5001   # MLflow

# 6. Trigger monthly retrain (in Airflow UI)
#    DAG: monthly_email_ml_pipeline
#    Conf: {"month": "2025-11", "ref_start": "2025-05", "ref_end": "2025-10"}
```
