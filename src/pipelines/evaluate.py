import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix, hstack, load_npz
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

SILVER_DIR = Path("data/silver")
GOLD_DIR = Path("data/gold")
MODEL_DIR = Path("models")

NUMERIC_FEATURES = [
    "log_chars",
    "avg_word_length",
    "unique_word_ratio",
    "exclaim_count",
]

# ── Gold snapshot loader ──────────────────────────────────────────────────────

def load_split(snapshot: str, split: str):
    """
    Load a Gold split (features + label) from data/gold/snapshot=<m>/full_load/.
    Returns: (X_sparse, y_ndarray).
    """
    base = GOLD_DIR / f"snapshot={snapshot}" / "full_load"
    X = load_npz(base / f"{split}_X.npz")
    y = pq.read_table(base / f"{split}.parquet", columns=["label"]).to_pandas()["label"].values
    return X, y


# ── Threshold & Evaluation Metrics ────────────────────────────────────────────

def pick_threshold(y_true, y_prob, target_precision: float):
    """
    Find the smallest threshold where precision >= target_precision.
    Returns: (threshold, precision_at_t, recall_at_t, hit_target)
    """
    p, r, t = precision_recall_curve(y_true, y_prob)
    t = np.append(t, 1.0)
    mask = p >= target_precision
    if not mask.any():
        idx = int(np.argmax(p))
        return float(t[idx]), float(p[idx]), float(r[idx]), False
    idx = int(np.where(mask)[0][0])
    return float(t[idx]), float(p[idx]), float(r[idx]), True


def evaluate(y_true, y_prob, threshold: float) -> dict:
    """Calculate metrics at a fixed threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

# ── Standalone CLI Logic ──────────────────────────────────────────────────────

def _resolve_threshold(arg_threshold: float | None) -> float:
    if arg_threshold is not None:
        return arg_threshold
    meta_path = MODEL_DIR / "train.json"
    if not meta_path.exists():
        raise FileNotFoundError("No train.json found. Please pass --threshold directly.")
    meta = json.loads(meta_path.read_text())
    return float(meta["threshold"])

def load_test_data(test_start_month: str):
    """Load evaluation data (e.g. >= April 2026) directly from Silver."""
    print(f"Loading test data (>= {test_start_month}) from {SILVER_DIR}...")
    frames = []
    for p in SILVER_DIR.glob("month_partition=*"):
        month = p.name.split("=")[1]
        if month >= test_start_month:
            parquet_path = p / "data_silver.parquet"
            if parquet_path.exists():
                df = pq.read_table(parquet_path).to_pandas()
                df["month_partition"] = month
                frames.append(df)
    
    if not frames:
        raise ValueError(f"No test data found for months >= {test_start_month}")
    return pd.concat(frames, ignore_index=True)

def main():
    parser = argparse.ArgumentParser(description="Evaluate ML Models on holdout test set")
    parser.add_argument("--test-start", default="2026-04", help="Min month for evaluation (e.g., 2026-04)")
    parser.add_argument("--threshold", type=float, default=None, help="Override threshold")
    args = parser.parse_args()

    model_path = MODEL_DIR / "best_spam_classifier.pkl"
    meta = json.loads((MODEL_DIR / "train.json").read_text())
    snapshot = meta["snapshot"]
    vec_path = GOLD_DIR / f"snapshot={snapshot}" / "artifacts" / "tfidf_vectorizer.pkl"
    scaler_path = GOLD_DIR / f"snapshot={snapshot}" / "artifacts" / "numeric_scaler.pkl"

    if not all(p.exists() for p in [model_path, vec_path, scaler_path]):
        raise FileNotFoundError("Missing model artifacts. Run train.py first.")

    print("[1/3] Loading Model, Vectorizer, and Scaler...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    scaler = joblib.load(scaler_path)
    
    threshold = _resolve_threshold(args.threshold)
    print(f"  threshold = {threshold:.4f}")

    print("[2/3] Loading Test Data...")
    df_test = load_test_data(args.test_start)
    y_test = df_test["label"].values

    print("  Transforming Text & Numeric Features...")
    X_test_text = vectorizer.transform(df_test["body_clean"])
    X_test_num = scaler.transform(df_test[NUMERIC_FEATURES])
    X_test = hstack([X_test_text, csr_matrix(X_test_num)])

    # Handle NB requirement natively
    if type(model).__name__ == "MultinomialNB":
        X_test = X_test[:, :len(vectorizer.vocabulary_)]

    print("[3/3] Predicting & Evaluating...")
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, y_prob, threshold)

    summary = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
    print(f"\nRESULTS (>={args.test_start}): {summary}")
    print(f"Confusion Matrix: {metrics['confusion_matrix']}  [[TN,FP],[FN,TP]]")

    report = {
        "evaluated_at": datetime.now(UTC).isoformat(),
        "test_start_month": args.test_start,
        "threshold": threshold,
        "metrics": metrics,
        "n_rows": int(X_test.shape[0]),
    }
    out_path = MODEL_DIR / "evaluate.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved to -> {out_path}")


if __name__ == "__main__":
    main()