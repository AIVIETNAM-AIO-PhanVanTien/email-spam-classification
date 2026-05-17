"""
monthly_email_ml_pipeline — per-month retrain loop.

Multi-task DAG (each step retryable independently). Default schedule is monthly
on the 1st @ 02:00 but the realistic trigger is manual with explicit conf:

    {"month": "2025-11", "ref_start": "2025-05", "ref_end": "2025-10", "promotion_delta": 0.0}

Topology when drift triggers retrain:

    bronze ──► silver ──► drift_check ──► branch_retrain ──┬─► skip_retrain ─────────────────────────────────────────────────────────────────────────┐
                                                            └─► gold_build_month ─► train_lr ─► train_nb ─► train_rf ─► train_xgb ─► challenge_and_promote ─┘
                                                                                                                                                              ▼
                                                                                                                                                       reload_fastapi

The 4 train tasks run sequentially so peak RAM holds only one model at a time —
the all-in-one `--model auto` path OOMs on a 7.6 GB Docker host with ~100k×30k
sparse features. `challenge_and_promote` reads the candidates dir, picks the
winner, and atomic-swaps it against the current champion if F1 wins.
"""
from __future__ import annotations

import os
from datetime import datetime

from airflow.models.dag import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.trigger_rule import TriggerRule

PROJECT_DIR = os.getenv("EMAIL_SPAM_PROJECT_DIR", "/opt/airflow")
VENV_ACT = os.getenv("EMAIL_SPAM_VENV_ACT", "true")
API_RELOAD_URL = os.getenv(
    "EMAIL_SPAM_API_RELOAD_URL", "http://host.docker.internal:8000/admin/reload-model"
)
DRIFT_THRESHOLD = float(os.getenv("DRIFT_SCORE_THRESHOLD", "0.25"))
PROMOTION_DELTA_DEFAULT = float(os.getenv("PROMOTION_F1_DELTA", "0.0"))

# Jinja templates resolved by Airflow at task render time. `dag_run.conf` is the
# canonical place per FSD; we fall back to the logical date's prior month if no
# conf is provided so the DAG also works on its schedule.
MONTH_TPL = "{{ (dag_run.conf.get('month') if dag_run and dag_run.conf else (data_interval_start | ds_format('%Y-%m-%d', '%Y-%m'))) }}"


def _drift_check(**context) -> dict:
    """Run the drift detector and push the score for branch_retrain."""
    from src.monitoring.drift_detector import compute_drift_for_month

    conf = (context.get("dag_run").conf or {}) if context.get("dag_run") else {}
    month = conf.get("month") or context["data_interval_start"].strftime("%Y-%m")
    ref_start = conf["ref_start"]
    ref_end = conf["ref_end"]

    print(f"[drift_check] month={month} ref=[{ref_start}..{ref_end}]")
    report = compute_drift_for_month(month, ref_start, ref_end)

    ti = context["ti"]
    ti.xcom_push(key="drift_score", value=report["drift_score"])
    ti.xcom_push(key="month", value=month)
    return {"month": month, "drift_score": report["drift_score"]}


def _branch_retrain(**context) -> str:
    """Route to gold_build_month when drift_score >= DRIFT_SCORE_THRESHOLD; else skip."""
    ti = context["ti"]
    drift_score = float(ti.xcom_pull(task_ids="drift_check", key="drift_score") or 0.0)
    threshold = float(
        ((context.get("dag_run").conf or {}) if context.get("dag_run") else {}).get(
            "drift_threshold", DRIFT_THRESHOLD
        )
    )
    if drift_score >= threshold:
        print(f"[branch_retrain] drift {drift_score:.4f} >= {threshold} → gold_build_month")
        return "gold_build_month"
    print(f"[branch_retrain] drift {drift_score:.4f} < {threshold} → skip_retrain")
    return "skip_retrain"


def _challenge_and_promote(**context) -> dict:
    """Pick winner from candidates, compare vs current champion, atomic-swap if it wins."""
    from src.pipelines.monthly_run import champion_challenger_from_candidates

    ti = context["ti"]
    month = ti.xcom_pull(task_ids="drift_check", key="month")
    conf = (context.get("dag_run").conf or {}) if context.get("dag_run") else {}
    promotion_delta = float(conf.get("promotion_delta", PROMOTION_DELTA_DEFAULT))

    decision = champion_challenger_from_candidates(month=month, promotion_delta=promotion_delta)
    ti.xcom_push(key="promoted", value=bool(decision["promoted"]))
    return decision


with DAG(
    dag_id="monthly_email_ml_pipeline",
    description="Bronze → Silver → drift → (gold + 4 candidates + challenge) → reload FastAPI",
    start_date=datetime(2025, 5, 1),
    schedule="@monthly",
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "email-spam"],
) as dag:

    bronze = BashOperator(
        task_id="bronze_ingest",
        bash_command=(
            f"cd {PROJECT_DIR} && {VENV_ACT} && "
            f"python -m src.etl.bronze_ingest --month {MONTH_TPL}"
        ),
    )

    silver = BashOperator(
        task_id="silver_transform",
        bash_command=(
            f"cd {PROJECT_DIR} && {VENV_ACT} && "
            f"python -m src.etl.silver_transform --month {MONTH_TPL}"
        ),
    )

    drift = PythonOperator(
        task_id="drift_check",
        python_callable=_drift_check,
    )

    branch = BranchPythonOperator(
        task_id="branch_retrain",
        python_callable=_branch_retrain,
    )

    skip = EmptyOperator(task_id="skip_retrain")

    gold_month = BashOperator(
        task_id="gold_build_month",
        bash_command=(
            f"cd {PROJECT_DIR} && {VENV_ACT} && "
            "python -m src.etl.gold_build "
            "--month {{ ti.xcom_pull(task_ids='drift_check', key='month') }}"
        ),
    )

    # 4 candidates trained as separate subprocesses — each task releases its
    # process (and RAM) before the next starts.
    train_tasks = {
        name: BashOperator(
            task_id=f"train_{name}",
            bash_command=(
                f"cd {PROJECT_DIR} && {VENV_ACT} && "
                "python -m src.pipelines.train --mode candidate "
                f"--model {name} "
                "--snapshot {{ ti.xcom_pull(task_ids='drift_check', key='month') }}"
            ),
        )
        for name in ("lr", "nb", "rf", "xgb")
    }

    # ALL_DONE: still try to promote even if one candidate (typically xgb) OOM'd
    # — challenge_and_promote raises if zero candidates exist for the snapshot.
    challenge = PythonOperator(
        task_id="challenge_and_promote",
        trigger_rule=TriggerRule.ALL_DONE,
        python_callable=_challenge_and_promote,
    )

    reload_api = BashOperator(
        task_id="reload_fastapi",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
        bash_command=(
            f'curl -fsS -X POST "{API_RELOAD_URL}" '
            '|| echo "[reload_fastapi] WARN: FastAPI unreachable; champion pkl unchanged on disk."'
        ),
    )

    bronze >> silver >> drift >> branch

    # Skip branch → reload (NONE_FAILED_MIN_ONE_SUCCESS handles the skipped sibling).
    branch >> skip >> reload_api

    # Retrain branch: gold → train_lr → train_nb → train_rf → train_xgb → challenge → reload
    branch >> gold_month
    prev = gold_month
    for name in ("lr", "nb", "rf", "xgb"):
        prev >> train_tasks[name]
        prev = train_tasks[name]
    prev >> challenge >> reload_api
