"""
Canonical inference module.

Composes the three artifacts produced by the training pipeline:
    1. Classifier         — models/best_spam_classifier.pkl  (RF/LR/NB/XGB)
    2. TF-IDF vectorizer  — data/gold/snapshot=<S>/artifacts/tfidf_vectorizer.pkl
    3. Numeric scaler     — data/gold/snapshot=<S>/artifacts/numeric_scaler.pkl

Snapshot <S> is read from models/train.json (the same file the trainer writes
its run summary to). When monthly_run.py promotes a new champion it rewrites
train.json and best_spam_classifier.pkl atomically; /admin/reload-model on
the API picks both up.

Train/serve parity:
    - Text cleaning uses src.utils.text_preprocessing.TextCleaner.aggressive_clean,
      same instance Silver fits on.
    - Numeric features are recomputed using the exact formulas in
      src/etl/silver_transform.py::run_text_features. If those drift, this
      module must follow.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from src.utils.text_preprocessing import TextCleaner

_BASE = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL_PATH = _BASE / "models" / "best_spam_classifier.pkl"
_DEFAULT_TRAIN_META_PATH = _BASE / "models" / "train.json"
_DEFAULT_GOLD_DIR = _BASE / "data" / "gold"

_DEFAULT_THRESHOLD = 0.5

_lock = threading.Lock()
_state: dict[str, Any] = {}


def _resolve_path(env_var: str, default: Path) -> Path:
    p = os.getenv(env_var)
    return Path(p) if p else default


def load_model(force_reload: bool = False) -> dict[str, Any]:
    """Load classifier + vectorizer + scaler + train metadata. Thread-safe, cached.

    Returns the cache dict so callers can read threshold / feature_subset / etc.
    """
    if _state and not force_reload:
        return _state

    with _lock:
        if _state and not force_reload:
            return _state

        model_path = _resolve_path("MODEL_PATH", _DEFAULT_MODEL_PATH)
        meta_path = _resolve_path("MODEL_METADATA_PATH", _DEFAULT_TRAIN_META_PATH)
        gold_dir = _resolve_path("GOLD_DIR", _DEFAULT_GOLD_DIR)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Classifier not found at {model_path}. "
                "Run `python -m src.pipelines.train --snapshot <YYYY-MM>` first."
            )
        if not meta_path.exists():
            raise FileNotFoundError(
                f"train.json not found at {meta_path}. "
                "Re-run the trainer — it writes models/train.json alongside the pkl."
            )

        meta = json.loads(meta_path.read_text())
        snapshot = meta.get("snapshot")
        if not snapshot:
            raise ValueError(f"`snapshot` field missing in {meta_path}")

        artifact_dir = gold_dir / f"snapshot={snapshot}" / "artifacts"
        vec_path = artifact_dir / "tfidf_vectorizer.pkl"
        scaler_path = artifact_dir / "numeric_scaler.pkl"
        tfidf_meta_path = artifact_dir / "tfidf_metadata.json"

        for p in (vec_path, scaler_path, tfidf_meta_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Gold artifact missing: {p}. "
                    f"Rebuild the snapshot: `python -m src.etl.gold_build --month {snapshot}`."
                )

        tfidf_meta = json.loads(tfidf_meta_path.read_text())
        numeric_features = tfidf_meta["numeric_features"]

        _state.clear()
        _state.update({
            "classifier": joblib.load(model_path),
            "vectorizer": joblib.load(vec_path),
            "scaler": joblib.load(scaler_path),
            "numeric_features": numeric_features,
            "threshold": float(meta.get("threshold", _DEFAULT_THRESHOLD)),
            "feature_subset": meta.get("feature_subset", "full"),
            "snapshot": snapshot,
            "model_type": meta.get("model_type"),
            "winner": meta.get("winner"),
            "trained_at": meta.get("trained_at"),
            "model_path": str(model_path),
        })
    return _state


def reload_model() -> dict[str, Any]:
    """Force re-load (used by POST /admin/reload-model after a promotion)."""
    state = load_model(force_reload=True)
    return {
        "snapshot": state["snapshot"],
        "model_type": state["model_type"],
        "winner": state["winner"],
        "trained_at": state["trained_at"],
        "threshold": state["threshold"],
        "feature_subset": state["feature_subset"],
    }


def _build_input_text(subject: str | None, body: str | None) -> str:
    parts = [p for p in (subject, body) if p]
    return " ".join(parts).strip()


def _compute_numeric_features(body_raw: str, body_clean: str) -> dict[str, float]:
    """Recompute the 4 numeric features that gold_build kept.

    Must match src/etl/silver_transform.py::run_text_features exactly:
        log_chars         = log1p(len(body_clean)),                round 4
        avg_word_length   = len(body_clean) / max(word_count, 1),  round 2
        unique_word_ratio = unique_words / max(word_count, 1),     round 4
        exclaim_count     = count('!') in body_RAW (not body_clean)
    """
    tokens = body_clean.split()
    char_count = len(body_clean)
    word_count = len(tokens)
    unique_words = len(set(tokens))

    denom = word_count if word_count > 0 else 1
    return {
        "log_chars": round(float(np.log1p(char_count)), 4),
        "avg_word_length": round(char_count / denom, 2),
        "unique_word_ratio": round(unique_words / denom, 4),
        "exclaim_count": body_raw.count("!"),
    }


def predict_spam(
    subject: str = "",
    body: str = "",
    threshold: float | None = None,
) -> dict[str, Any]:
    """Score one (subject, body) pair.

    Returns:
        {
            "label": "spam" | "not_spam",
            "spam_probability": float in [0, 1],
            "threshold": float,
        }
    """
    raw = _build_input_text(subject, body)
    if not raw:
        raise ValueError("Subject or body must not be empty.")

    state = load_model()

    body_clean = TextCleaner(pd.Series([raw])).aggressive_clean().get().iloc[0]
    if not body_clean:
        return {
            "label": "not_spam",
            "spam_probability": 0.0,
            "threshold": float(threshold or state["threshold"]),
            "note": "input contained no usable tokens after cleaning",
        }

    X_text = state["vectorizer"].transform([body_clean])

    if state["feature_subset"] == "tfidf_only":
        X = X_text
    else:
        feats = _compute_numeric_features(raw, body_clean)
        # DataFrame (not ndarray) so StandardScaler sees the same column names
        # it was fitted on — avoids sklearn's "X does not have valid feature names" warning.
        numeric_row = pd.DataFrame(
            [[feats[name] for name in state["numeric_features"]]],
            columns=state["numeric_features"],
            dtype=np.float64,
        )
        X_num = state["scaler"].transform(numeric_row)
        X = hstack([X_text, csr_matrix(X_num)])

    proba = float(state["classifier"].predict_proba(X)[0, 1])
    th = float(threshold if threshold is not None else state["threshold"])

    return {
        "label": "spam" if proba >= th else "not_spam",
        "spam_probability": proba,
        "threshold": th,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score one email from the CLI.")
    parser.add_argument("--subject", default="")
    parser.add_argument("--body", default="")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    result = predict_spam(args.subject, args.body, threshold=args.threshold)
    print(json.dumps(result, indent=2))
