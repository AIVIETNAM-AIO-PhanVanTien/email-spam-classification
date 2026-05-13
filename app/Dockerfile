# Serving stack image: shared by FastAPI, Streamlit, and Gmail poller.
# Build context = repo root (one level up from this Dockerfile).
#   docker compose -f app/docker-compose.yml build
#   docker build -f app/Dockerfile -t email-spam-classifier:latest .

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serving.txt ./
RUN pip install --upgrade pip && pip install -r requirements-serving.txt

# Code: src/utils for TextCleaner, app/ for serving entrypoints
COPY src/__init__.py    ./src/__init__.py
COPY src/utils/         ./src/utils/
COPY app/               ./app/

# Model + gold artifacts (snapshot resolved from models/train.json at runtime)
COPY models/            ./models/
COPY data/gold/         ./data/gold/

ENV PYTHONPATH=/app

EXPOSE 8000 8501

# Default = API. docker-compose overrides command for streamlit / gmail-poller.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
