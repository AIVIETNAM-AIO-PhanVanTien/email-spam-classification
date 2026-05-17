"""
Near-real-time Gmail filter — polling worker.

Loop:
    1. On first run, call users.getProfile to fetch the current historyId,
       persist it, and skip (don't backfill the entire mailbox).
    2. Each tick: list messages added since last_history_id.
    3. For each new message: fetch subject+body, POST /predict, apply
       AI_SPAM / AI_HAM via Gmail API, append to logs/gmail_predictions.csv.
    4. Persist the latest historyId, sleep GMAIL_POLL_INTERVAL_SECONDS.

Run from the project root:
    python -m app.gmail_poller

Env (see .env.example):
    GMAIL_CREDENTIALS_JSON  — only needed for OAuth bootstrap
    GMAIL_TOKEN_JSON        — output of bootstrap; what the poller reads
    GMAIL_POLL_INTERVAL_SECONDS (default 30)
    API_URL                 — FastAPI predict endpoint
    GMAIL_STATE_PATH        — where to persist last_history_id
    GMAIL_PREDICTION_LOG    — where to append predictions
    GMAIL_LABEL_SPAM, GMAIL_LABEL_HAM
"""
from __future__ import annotations

import csv
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from app.gmail_client import GmailClient, GmailMessage

_BASE = Path(__file__).resolve().parent.parent

DEFAULT_TOKEN_PATH = _BASE / "app" / "secrets" / "gmail_token.json"
DEFAULT_STATE_PATH = _BASE / "logs" / "gmail_state.json"
DEFAULT_LOG_PATH = _BASE / "logs" / "gmail_predictions.csv"
DEFAULT_API_URL = "http://127.0.0.1:8000/predict"

LOG_FIELDS = [
    "timestamp", "message_id", "subject", "snippet",
    "prediction", "spam_probability", "applied_label",
]

_running = True


def _stop(_sig, _frame):
    global _running
    _running = False
    print("[gmail_poller] received stop signal, finishing current tick…")


signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)


def _load_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def _append_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def predict_via_api(api_url: str, subject: str, body: str, timeout: float = 10.0) -> dict:
    resp = requests.post(api_url, json={"subject": subject, "body": body}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _process_message(client: GmailClient, msg: GmailMessage,
                     api_url: str, log_path: Path) -> None:
    try:
        pred = predict_via_api(api_url, msg.subject, msg.body)
    except requests.RequestException as e:
        print(f"[gmail_poller] /predict failed for {msg.id}: {e}")
        return

    label = pred["label"]
    try:
        client.apply_prediction_label(msg.id, label)
        applied = "AI_SPAM" if label == "spam" else "AI_HAM"
    except Exception as e:  # noqa: BLE001
        print(f"[gmail_poller] failed to apply label to {msg.id}: {e}")
        applied = "ERROR"

    _append_log(log_path, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message_id": msg.id,
        "subject": (msg.subject or "")[:200],
        "snippet": (msg.snippet or "")[:200],
        "prediction": label,
        "spam_probability": f"{pred['spam_probability']:.6f}",
        "applied_label": applied,
    })
    print(f"[gmail_poller] {msg.id}: {label} p={pred['spam_probability']:.3f} → {applied}")


def run() -> int:
    token_path = Path(os.getenv("GMAIL_TOKEN_JSON", str(DEFAULT_TOKEN_PATH)))
    state_path = Path(os.getenv("GMAIL_STATE_PATH", str(DEFAULT_STATE_PATH)))
    log_path = Path(os.getenv("GMAIL_PREDICTION_LOG", str(DEFAULT_LOG_PATH)))
    api_url = os.getenv("API_URL", DEFAULT_API_URL)
    interval = int(os.getenv("GMAIL_POLL_INTERVAL_SECONDS", "30"))

    print(f"[gmail_poller] token={token_path} api={api_url} every {interval}s")
    client = GmailClient(token_path)

    state = _load_state(state_path)
    last_history_id = state.get("last_history_id")
    if not last_history_id:
        last_history_id = client.current_history_id()
        _save_state(state_path, {"last_history_id": last_history_id})
        print(f"[gmail_poller] bootstrapped last_history_id={last_history_id} "
              "(future messages only — no backfill)")

    while _running:
        try:
            new_ids, latest = client.list_new_message_ids(last_history_id)
            for mid in new_ids:
                try:
                    msg = client.fetch_message(mid)
                except Exception as fetch_err:  # noqa: BLE001
                    # 404 = message was deleted/moved between history.list and
                    # messages.get (typically Gmail's auto-spam filter trashing
                    # the email). Skip the message but don't block state advance,
                    # otherwise the same dead id keeps being retried forever.
                    status = getattr(fetch_err, "status_code", None) or getattr(
                        getattr(fetch_err, "resp", None), "status", None
                    )
                    print(f"[gmail_poller] fetch {mid} failed (status={status}): {fetch_err}")
                    continue
                _process_message(client, msg, api_url, log_path)

            if latest != last_history_id:
                last_history_id = latest
                _save_state(state_path, {"last_history_id": last_history_id})
        except Exception as e:  # noqa: BLE001
            # Don't crash the worker on transient API errors; back off and retry.
            print(f"[gmail_poller] tick error: {e}")

        # Sleep in 1s slices so SIGTERM reacts quickly.
        for _ in range(interval):
            if not _running:
                break
            time.sleep(1)

    print("[gmail_poller] exited cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(run())
