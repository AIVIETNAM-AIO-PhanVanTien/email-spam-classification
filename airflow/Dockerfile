# Extends apache/airflow:3.0.4-python3.11 with the project's runtime deps so
# DAG tasks can run `python -m src.pipelines.monthly_run` directly — no separate
# venv inside the container, no PythonVirtualenvOperator gymnastics.
#
# Build context = repo root (one level up from this Dockerfile).
#   docker compose --env-file airflow/.env -f airflow/docker-compose.yml build

FROM apache/airflow:3.0.4-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

ENV PYTHONPATH=/opt/airflow
