"""
initial_full_load — one-shot bootstrap DAG.

Run-on-demand only (no schedule). Walks every silver month present on disk,
ingests + transforms anything still missing, builds the gold snapshot pinned
by `--holdout-month` (default = the latest available silver month), then
trains champion v1 → models/best_spam_classifier.pkl, and finally pokes
FastAPI to reload.

Trigger from the Airflow UI with optional conf:
    { "holdout_month": "2026-04" }

If `holdout_month` is omitted the DAG resolves it at runtime to the
latest silver partition found under data/silver/.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from airflow.models.dag import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

PROJECT_DIR = os.getenv("EMAIL_SPAM_PROJECT_DIR", "/opt/airflow")
VENV_ACT = os.getenv("EMAIL_SPAM_VENV_ACT", "true")
API_RELOAD_URL = os.getenv(
    "EMAIL_SPAM_API_RELOAD_URL", "http://host.docker.internal:8000/admin/reload-model"
)

SILVER_DIR = Path(PROJECT_DIR) / "data" / "silver"
RAW_BY_MONTH = Path(PROJECT_DIR) / "data" / "raw" / "by_month"


def _discover_months(**context) -> list[str]:
    """Discover all raw partitions on disk; push the holdout month for downstream tasks.

    `holdout_month` precedence:
        1. dag_run.conf["holdout_month"] if set
        2. latest silver partition found on disk
        3. latest raw partition found on disk
    """
    raw_months = sorted(
        p.name.removeprefix("emails_").removesuffix(".csv")
        for p in RAW_BY_MONTH.glob("emails_*.csv")
    )
    silver_months = sorted(p.name.split("=")[1] for p in SILVER_DIR.glob("month_partition=*"))

    conf = (context.get("dag_run").conf or {}) if context.get("dag_run") else {}
    holdout = conf.get("holdout_month") or (silver_months[-1] if silver_months else None) or (
        raw_months[-1] if raw_months else None
    )
    if not holdout:
        raise RuntimeError(
            f"No raw or silver partitions found under {RAW_BY_MONTH} / {SILVER_DIR}. "
            "Run `python -m src.utils.split_raw` first."
        )
    if not raw_months:
        raise RuntimeError(f"No raw partitions found under {RAW_BY_MONTH}.")

    print(f"[initial_full_load] raw months: {raw_months}")
    print(f"[initial_full_load] silver months on disk: {silver_months}")
    print(f"[initial_full_load] holdout month resolved: {holdout}")

    ti = context["ti"]
    ti.xcom_push(key="months", value=raw_months)
    ti.xcom_push(key="holdout_month", value=holdout)
    return raw_months


def _bronze_silver_all(**context) -> int:
    """Idempotently bronze + silver every raw month. Skips months already done."""
    from src.etl import bronze_ingest, silver_transform

    ti = context["ti"]
    months: list[str] = ti.xcom_pull(task_ids="discover_months", key="months") or []
    if not months:
        raise RuntimeError("`discover_months` did not push any months.")

    for m in months:
        print(f"[bronze_silver_all] processing {m}")
        bronze_ingest.ingest(m)
        silver_transform.process(m)
    return len(months)


with DAG(
    dag_id="initial_full_load",
    description="One-shot bootstrap: split → bronze → silver → gold → champion v1 → reload",
    start_date=datetime(2024, 11, 1),
    schedule=None,  # manual trigger only
    catchup=False,
    max_active_runs=1,
    tags=["bootstrap", "email-spam"],
) as dag:

    discover = PythonOperator(
        task_id="discover_months",
        python_callable=_discover_months,
    )

    bronze_silver = PythonOperator(
        task_id="bronze_silver_all",
        python_callable=_bronze_silver_all,
    )

    gold = BashOperator(
        task_id="gold_build",
        bash_command=(
            f"cd {PROJECT_DIR} && {VENV_ACT} && "
            "python -m src.etl.gold_build "
            "--month {{ ti.xcom_pull(task_ids='discover_months', key='holdout_month') }}"
        ),
    )

    # Per-model train tasks (sequential — keeps peak RAM bounded to one model
    # at a time; the all-in-one --model auto path OOMs at ~6GB on the LocalExecutor).
    train_tasks = {
        name: BashOperator(
            task_id=f"train_{name}",
            bash_command=(
                f"cd {PROJECT_DIR} && {VENV_ACT} && "
                "python -m src.pipelines.train --mode candidate "
                f"--model {name} "
                "--snapshot {{ ti.xcom_pull(task_ids='discover_months', key='holdout_month') }}"
            ),
        )
        for name in ("lr", "nb", "rf", "xgb")
    }

    # ALL_DONE: pick from whichever candidates succeeded; pick_winner itself
    # raises if zero candidates exist, so a total failure still surfaces.
    pick_winner = BashOperator(
        task_id="pick_winner",
        trigger_rule=TriggerRule.ALL_DONE,
        bash_command=(
            f"cd {PROJECT_DIR} && {VENV_ACT} && "
            "python -m src.pipelines.train --mode pick "
            "--snapshot {{ ti.xcom_pull(task_ids='discover_months', key='holdout_month') }}"
        ),
    )

    reload_api = BashOperator(
        task_id="reload_fastapi",
        bash_command=(
            f'curl -fsS -X POST "{API_RELOAD_URL}" '
            "|| echo \"[reload_fastapi] WARN: FastAPI unreachable; champion pkl was still written.\""
        ),
    )

    # Chain: gold → train_lr → train_nb → train_rf → train_xgb → pick_winner → reload_api
    discover >> bronze_silver >> gold
    prev = gold
    for name in ("lr", "nb", "rf", "xgb"):
        prev >> train_tasks[name]
        prev = train_tasks[name]
    prev >> pick_winner >> reload_api
