import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.pipelines.evaluate import evaluate, pick_threshold

SILVER_DIR = Path("data/silver")
MODEL_DIR = Path("models")

NUMERIC_FEATURES = [
    "log_chars",
    "avg_word_length",
    "unique_word_ratio",
    "exclaim_count",
]

TFIDF_CONFIG = {
    "max_features": 30_000,
    "ngram_range": (1, 2),
    "min_df": 5,
    "max_df": 0.95,
    "sublinear_tf": True,
}


def load_data_by_time(train_end_month: str, test_start_month: str):
    """
    Load data from Silver layer and split based on time logic:
    - Train/Val: <= train_end_month (March backwards)
    - Test: >= test_start_month (April forward)
    """
    print(f"Loading data from {SILVER_DIR}...")
    frames_trainval = []
    frames_test = []

    for p in SILVER_DIR.glob("month_partition=*"):
        month = p.name.split("=")[1]
        parquet_path = p / "data_silver.parquet"
        if parquet_path.exists():
            df = pq.read_table(parquet_path).to_pandas()
            df["month_partition"] = month
            if month <= train_end_month:
                frames_trainval.append(df)
            elif month >= test_start_month:
                frames_test.append(df)

    if not frames_trainval:
        raise ValueError("No data found for Train/Val split.")
    if not frames_test:
        raise ValueError("No data found for Test split.")

    df_trainval = pd.concat(frames_trainval, ignore_index=True)
    df_test = pd.concat(frames_test, ignore_index=True)

    # Split trainval into train and val (85/15)
    df_train, df_val = train_test_split(
        df_trainval, test_size=0.15, stratify=df_trainval["label"], random_state=42
    )

    print(f"Train: {len(df_train):,} rows")
    print(f"Val:   {len(df_val):,} rows")
    print(f"Test:  {len(df_test):,} rows")

    return df_train, df_val, df_test


def build_features(df_train, df_val, df_test):
    """Fit TF-IDF and Scaler on train, transform val and test."""
    print("\nFitting TF-IDF on train set...")
    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    X_train_text = vectorizer.fit_transform(df_train["body_clean"])
    X_val_text = vectorizer.transform(df_val["body_clean"])
    X_test_text = vectorizer.transform(df_test["body_clean"])

    print("Fitting StandardScaler on numeric features...")
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(df_train[NUMERIC_FEATURES])
    X_val_num = scaler.transform(df_val[NUMERIC_FEATURES])
    X_test_num = scaler.transform(df_test[NUMERIC_FEATURES])

    X_train = hstack([X_train_text, csr_matrix(X_train_num)])
    X_val = hstack([X_val_text, csr_matrix(X_val_num)])
    X_test = hstack([X_test_text, csr_matrix(X_test_num)])

    return vectorizer, scaler, X_train, X_val, X_test


def build_candidates():
    """Return dictionary of ML candidates."""
    return {
        "lr": LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42),
        "nb": MultinomialNB(),
        "rf": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        "xgb": XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1),
    }


def maybe_slice_features(name: str, X, n_tfidf: int):
    """NB requires non-negative inputs. Slice only TF-IDF for NB."""
    if name == "nb":
        return X[:, :n_tfidf]
    return X


def train_one(
    name, model, X_train, y_train, X_val, y_val, X_test, y_test, target_precision, n_tfidf
):
    print(f"\n── [{name.upper()}] fitting ─────────────────────────────")
    Xt_train = maybe_slice_features(name, X_train, n_tfidf)
    Xt_val = maybe_slice_features(name, X_val, n_tfidf)
    Xt_test = maybe_slice_features(name, X_test, n_tfidf)

    if name == "nb":
        print(f"  [NB] using TF-IDF only: {Xt_train.shape} (ignoring numeric due to negative values)")

    model.fit(Xt_train, y_train)
    val_prob = model.predict_proba(Xt_val)[:, 1]
    test_prob = model.predict_proba(Xt_test)[:, 1]

    threshold, p_at_t, r_at_t, hit = pick_threshold(y_val, val_prob, target_precision)
    status = "HIT" if hit else f"MISS (max P = {p_at_t:.4f})"
    print(f"  threshold={threshold:.4f}  val P={p_at_t:.4f}  R={r_at_t:.4f}  | {status}")

    val_metrics = evaluate(y_val, val_prob, threshold)
    test_metrics = evaluate(y_test, test_prob, threshold)

    summary = lambda m: {k: round(v, 4) for k, v in m.items() if k != "confusion_matrix"}
    print(f"  VAL  : {summary(val_metrics)}")
    print(f"  TEST : {summary(test_metrics)}")

    return {
        "name": name,
        "model": model,
        "threshold": threshold,
        "hit_target": hit,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "sklearn_params": model.get_params(),
        "feature_subset": "tfidf_only" if name == "nb" else "full",
    }


def pick_winner(results, target_precision):
    """Pick model with highest precision. Tiebreak by recall. Fallback to AP."""
    hit_models = {n: r for n, r in results.items() if r["hit_target"]}
    if hit_models:
        winner = max(
            hit_models,
            key=lambda n: (
                hit_models[n]["val_metrics"]["precision"],
                hit_models[n]["val_metrics"]["recall"],
            ),
        )
        reason = (
            f"highest precision ({hit_models[winner]['val_metrics']['precision']:.4f}) "
            f"with recall ({hit_models[winner]['val_metrics']['recall']:.4f})"
        )
    else:
        winner = max(results, key=lambda n: results[n]["val_metrics"]["average_precision"])
        reason = f"no model hit target P. Fallback to highest AP(val) ({results[winner]['val_metrics']['average_precision']:.4f})"
    return winner, reason


def main():
    parser = argparse.ArgumentParser(description="Train Baseline Models")
    parser.add_argument("--train-end", default="2026-03", help="Max month for train/val (e.g., 2026-03)")
    parser.add_argument("--test-start", default="2026-04", help="Min month for test (e.g., 2026-04)")
    parser.add_argument("--target-precision", type=float, default=0.99, help="Target Precision")
    args = parser.parse_args()

    print("[1/5] Loading & Splitting Data")
    df_train, df_val, df_test = load_data_by_time(args.train_end, args.test_start)

    y_train = df_train["label"].values
    y_val = df_val["label"].values
    y_test = df_test["label"].values

    print("[2/5] Building Features")
    vectorizer, scaler, X_train, X_val, X_test = build_features(df_train, df_val, df_test)
    n_tfidf = len(vectorizer.vocabulary_)

    print("[3/5] Training Candidates")
    candidates = build_candidates()
    results = {}
    for name, model in candidates.items():
        results[name] = train_one(
            name, model, X_train, y_train, X_val, y_val, X_test, y_test,
            args.target_precision, n_tfidf
        )

    print("\n[4/5] Picking Winner")
    winner_name, reason = pick_winner(results, args.target_precision)
    winner = results[winner_name]
    print(f"  WINNER = {winner_name.upper()} — {reason}")

    print("[5/5] Saving Artifacts")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save the winner model, vectorizer, and scaler
    joblib.dump(winner["model"], MODEL_DIR / "best_model.pkl")
    joblib.dump(vectorizer, MODEL_DIR / "vectorizer.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    def serializable(r):
        return {k: v for k, v in r.items() if k != "model"}

    meta = {
        "trained_at": datetime.now(UTC).isoformat(),
        "train_end": args.train_end,
        "test_start": args.test_start,
        "target_precision": args.target_precision,
        "winner": winner_name,
        "winner_reason": reason,
        "threshold": winner["threshold"],
        "val_metrics": winner["val_metrics"],
        "test_metrics": winner["test_metrics"],
        "candidates": {n: serializable(r) for n, r in results.items()},
        "n_tfidf_features": n_tfidf,
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
    }
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print("  Models and metadata successfully saved in `models/` directory.")


if __name__ == "__main__":
    main()
