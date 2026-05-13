"""
Streamlit UI for the email spam classifier.

Two tabs:
    1. "Predict" — manual subject+body form, calls FastAPI /predict.
    2. "Gmail feed" — live view of the Gmail poller's predictions
       (reads logs/gmail_predictions.csv on disk).
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

_BASE = Path(__file__).resolve().parent.parent

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
HEALTH_URL = os.getenv("HEALTH_URL", API_URL.replace("/predict", "/health"))
GMAIL_PREDICTION_LOG = Path(
    os.getenv("GMAIL_PREDICTION_LOG", str(_BASE / "logs" / "gmail_predictions.csv"))
)
PREDICTION_LOG = Path(
    os.getenv("PREDICTION_LOG_PATH", str(_BASE / "logs" / "predictions.csv"))
)

EXAMPLES = {
    "Spam example — phishing": {
        "subject": "Congratulations, you won an iPhone!",
        "body": "Click here now to claim your free prize before it expires. "
                "Limited offer, act fast! http://bit.ly/win-prize",
    },
    "Ham example — work meeting": {
        "subject": "Tomorrow's Q3 review meeting",
        "body": "Hi team, please find attached the agenda for our Q3 review. "
                "We start at 10am in room 4B. Let me know if you have items to add.",
    },
}


st.set_page_config(page_title="Email Spam Classifier", page_icon=":email:", layout="wide")
st.title("Email Spam Classifier")

# Show current champion info from /health so users see which model is live.
try:
    health = requests.get(HEALTH_URL, timeout=5).json()
    if health.get("model_loaded"):
        st.caption(
            f"Champion = **{health.get('model_type', '?')}** "
            f"(winner=`{health.get('winner')}`, snapshot=`{health.get('snapshot')}`, "
            f"threshold={health.get('threshold', 0):.4f}) — FastAPI at `{API_URL}`"
        )
    else:
        st.warning(f"API at {API_URL} is up but no model loaded.")
except requests.RequestException as e:
    st.error(f"Cannot reach API health at {HEALTH_URL}: {e}")

tab_predict, tab_gmail = st.tabs(["🔮 Predict", "📬 Gmail feed"])


# ------------------------------------------------------------------ Predict tab
with tab_predict:
    with st.sidebar:
        st.header("Examples")
        chosen = st.radio("Load an example", ["(none)"] + list(EXAMPLES.keys()))

    default_subject = ""
    default_body = ""
    if chosen != "(none)":
        default_subject = EXAMPLES[chosen]["subject"]
        default_body = EXAMPLES[chosen]["body"]

    col_in, col_out = st.columns([3, 2])

    with col_in:
        subject = st.text_input("Subject", value=default_subject)
        body = st.text_area("Body", value=default_body, height=200)
        predict_clicked = st.button("Predict", type="primary")

    with col_out:
        if predict_clicked:
            if not subject.strip() and not body.strip():
                st.error("Subject or body must not be empty.")
            else:
                try:
                    resp = requests.post(
                        API_URL,
                        json={"subject": subject, "body": body},
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        label = data["label"]
                        proba = data["spam_probability"]
                        threshold = data["threshold"]

                        if label == "spam":
                            st.error(
                                f"Predicted: **SPAM**  "
                                f"(p={proba:.3f}, threshold={threshold:.3f})"
                            )
                        else:
                            st.success(
                                f"Predicted: **NOT SPAM**  "
                                f"(p={proba:.3f}, threshold={threshold:.3f})"
                            )

                        st.progress(min(max(proba, 0.0), 1.0))
                        with st.expander("Raw response"):
                            st.json(data)
                    else:
                        detail = resp.json().get("detail", resp.text)
                        st.error(f"API error {resp.status_code}: {detail}")
                except requests.RequestException as e:
                    st.error(f"Failed to reach API at {API_URL}: {e}")

    st.markdown("---")
    st.caption(f"Manual predictions logged at `{PREDICTION_LOG}`.")


# ---------------------------------------------------------------- Gmail tab
with tab_gmail:
    st.subheader("Live Gmail predictions")
    st.caption(
        f"Reading `{GMAIL_PREDICTION_LOG}` (auto-refresh on rerun). "
        "The Gmail poller writes 1 row per email it classifies."
    )

    if st.button("🔄 Refresh"):
        st.rerun()

    if not GMAIL_PREDICTION_LOG.exists():
        st.info("No Gmail predictions yet. Send an email to the test inbox and wait ~30 seconds.")
        st.code(f"Expected file: {GMAIL_PREDICTION_LOG}")
    else:
        try:
            df = pd.read_csv(GMAIL_PREDICTION_LOG)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()

        if df.empty:
            st.info("File exists but is empty. Send an email and wait ~30 seconds.")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)

            n_total = len(df)
            n_spam = int((df["prediction"] == "spam").sum())
            n_ham = n_total - n_spam
            today = datetime.utcnow().date()
            n_today = int((df["timestamp"].dt.date == today).sum())

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total scored", n_total)
            c2.metric("Spam", n_spam)
            c3.metric("Ham", n_ham)
            c4.metric("Today (UTC)", n_today)

            st.markdown("**Recent predictions**")
            display = df[[
                "timestamp", "subject", "snippet",
                "prediction", "spam_probability", "applied_label",
            ]].head(50).copy()
            display["spam_probability"] = display["spam_probability"].astype(float).round(3)
            display["timestamp"] = display["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "timestamp": st.column_config.TextColumn("Time (UTC)", width="small"),
                    "subject": st.column_config.TextColumn("Subject", width="medium"),
                    "snippet": st.column_config.TextColumn("Snippet", width="large"),
                    "prediction": st.column_config.TextColumn("Label", width="small"),
                    "spam_probability": st.column_config.NumberColumn("p(spam)", format="%.3f"),
                    "applied_label": st.column_config.TextColumn("Applied", width="small"),
                },
            )

            with st.expander("Raw CSV preview"):
                st.dataframe(df, use_container_width=True, hide_index=True)
