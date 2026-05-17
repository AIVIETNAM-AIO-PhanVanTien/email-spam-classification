import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

from src.pipelines.evaluate import evaluate, load_split, pick_threshold

GOLD_DIR = Path("data/gold")
MODEL_DIR = Path("models")
CANDIDATES_DIR = MODEL_DIR / "candidates"


def build_candidates(model_choice: str) -> dict:
    catalog = {
        "lr": LogisticRegression(
            max_iter=1000, solver="lbfgs", random_state=42,
        ),
        "nb": MultinomialNB(),
        "rf": RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=42,
        ),
        "xgb": XGBClassifier(
            eval_metric="logloss", random_state=42,
            # n_jobs=-1 + tree_method=hist OOMs on 6 GB containers when X is
            # ~100k × 30k sparse. Bound workers + use small histograms.
            n_jobs=2, tree_method="hist", max_bin=128,
        ),
    }
    if model_choice == "auto":
        return catalog
    return {model_choice: catalog[model_choice]}


def get_tfidf_vocab_size(snapshot: str) -> int:
    meta_path = GOLD_DIR / f"snapshot={snapshot}" / "artifacts" / "tfidf_metadata.json"
    return int(json.loads(meta_path.read_text())["vocab_size"])


def maybe_slice_features(name: str, X, n_tfidf: int):
    # MultinomialNB requires non-negative input; the scaled numeric block can be negative.
    if name == "nb":
        return X[:, :n_tfidf]
    return X


def train_one(
    name: str, model, X_train, y_train, X_val, y_val, X_test, y_test,
    target_precision: float, n_tfidf: int,
) -> dict:
    print(f"\n── [{name.upper()}] fitting ─────────────────────────────")
    Xt_train = maybe_slice_features(name, X_train, n_tfidf)
    Xt_val = maybe_slice_features(name, X_val, n_tfidf)
    Xt_test = maybe_slice_features(name, X_test, n_tfidf)
    if name == "nb":
        print(f"  [NB] tfidf-only: {Xt_train.shape} (dropping numeric block: contains negatives)")

    model.fit(Xt_train, y_train)
    val_prob = model.predict_proba(Xt_val)[:, 1]
    test_prob = model.predict_proba(Xt_test)[:, 1]

    threshold, p_at_t, r_at_t, hit = pick_threshold(
        y_val, val_prob, target_precision
    )
    status = "HIT" if hit else f"MISS (max P observed = {p_at_t:.4f})"
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


def pick_winner(results: dict, target_precision: float):
    # All hit models already satisfy P >= target at their chosen threshold,
    # so recall is the meaningful tiebreaker. Fall back to AP when none hit.
    hit_models = {n: r for n, r in results.items() if r["hit_target"]}
    if hit_models:
        winner = max(
            hit_models, key=lambda n: hit_models[n]["val_metrics"]["recall"]
        )
        reason = (
            f"precision >= {target_precision} with highest recall@threshold "
            f"({hit_models[winner]['val_metrics']['recall']:.4f})"
        )
    else:
        winner = max(
            results, key=lambda n: results[n]["val_metrics"]["average_precision"]
        )
        reason = (
            f"no model hit P>={target_precision}; fallback to highest "
            f"AP(val) ({results[winner]['val_metrics']['average_precision']:.4f})"
        )
    return winner, reason


def main(snapshot: str, target_precision: float, model_choice: str):
    print(f"[1/4] Loading snapshot={snapshot}")
    X_train, y_train = load_split(snapshot, "train")
    X_val, y_val = load_split(snapshot, "val")
    X_test, y_test = load_split(snapshot, "test")
    n_tfidf = get_tfidf_vocab_size(snapshot)
    print(f"  train={X_train.shape}  val={X_val.shape}  test={X_test.shape}  "
          f"(tfidf_vocab={n_tfidf})")
    print(f"  spam_ratio  train={y_train.mean():.4f}  val={y_val.mean():.4f}  "
          f"test={y_test.mean():.4f}")

    print(f"[2/4] Training candidates: {model_choice}")
    candidates = build_candidates(model_choice)
    results = {
        name: train_one(
            name, model, X_train, y_train, X_val, y_val, X_test, y_test,
            target_precision, n_tfidf,
        )
        for name, model in candidates.items()
    }

    print("\n[3/4] Picking winner")
    winner_name, reason = pick_winner(results, target_precision)
    winner = results[winner_name]
    print(f"  WINNER = {winner_name.upper()} — {reason}")

    print("[4/4] Saving model + metadata")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "best_spam_classifier.pkl"
    joblib.dump(winner["model"], model_path)

    def serializable(r):
        return {k: v for k, v in r.items() if k != "model"}

    meta = {
        "snapshot": snapshot,
        "trained_at": datetime.now(UTC).isoformat(),
        "target_precision": target_precision,
        "model_choice_flag": model_choice,
        "winner": winner_name,
        "winner_reason": reason,
        "model_type": type(winner["model"]).__name__,
        "threshold": winner["threshold"],
        "threshold_hit_target": winner["hit_target"],
        "feature_subset": winner["feature_subset"],
        "val_metrics": winner["val_metrics"],
        "test_metrics": winner["test_metrics"],
        "candidates": {n: serializable(r) for n, r in results.items()},
        "n_features": int(X_train.shape[1]),
        "n_tfidf_features": n_tfidf,
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "gold_artifacts": {
            "vectorizer": f"data/gold/snapshot={snapshot}/artifacts/tfidf_vectorizer.pkl",
            "scaler": f"data/gold/snapshot={snapshot}/artifacts/numeric_scaler.pkl",
        },
    }
    meta_path = MODEL_DIR / "train.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"  model    → {model_path}")
    print(f"  metadata → {meta_path}")
    return meta


def train_single_candidate(snapshot: str, name: str, target_precision: float) -> dict:
    """Train ONE model and save it to models/candidates/<name>.{pkl,json}.

    Used by per-model Airflow tasks to keep peak memory bounded — only one
    classifier is held in memory per task, and the heavier ones (rf/xgb) get
    isolated retries when they OOM.
    """
    print(f"[1/3] Loading snapshot={snapshot}")
    X_train, y_train = load_split(snapshot, "train")
    X_val, y_val = load_split(snapshot, "val")
    X_test, y_test = load_split(snapshot, "test")
    n_tfidf = get_tfidf_vocab_size(snapshot)
    print(f"  train={X_train.shape}  val={X_val.shape}  test={X_test.shape}  (tfidf_vocab={n_tfidf})")

    print(f"[2/3] Training {name}")
    model = build_candidates(name)[name]
    result = train_one(
        name, model, X_train, y_train, X_val, y_val, X_test, y_test,
        target_precision, n_tfidf,
    )

    print(f"[3/3] Saving candidate {name}")
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    pkl_path = CANDIDATES_DIR / f"{name}.pkl"
    json_path = CANDIDATES_DIR / f"{name}.json"
    joblib.dump(result["model"], pkl_path)

    meta = {
        "name": name,
        "snapshot": snapshot,
        "trained_at": datetime.now(UTC).isoformat(),
        "target_precision": target_precision,
        "model_type": type(result["model"]).__name__,
        "threshold": result["threshold"],
        "hit_target": result["hit_target"],
        "feature_subset": result["feature_subset"],
        "val_metrics": result["val_metrics"],
        "test_metrics": result["test_metrics"],
        "n_features": int(X_train.shape[1]),
        "n_tfidf_features": n_tfidf,
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "pkl_path": str(pkl_path),
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  pkl  → {pkl_path}\n  meta → {json_path}")
    return meta


def pick_winner_from_candidates(snapshot: str, target_precision: float) -> dict:
    """Read all candidate JSONs, run pick_winner() on metadata, promote winner pkl atomically.

    Avoids loading any model into memory — we only need the val_metrics dict
    plus the pkl path to copy. Output:
        models/best_spam_classifier.pkl     ← copy of winner candidate pkl
        models/train.json                   ← aggregated metadata
    """
    if not CANDIDATES_DIR.exists():
        raise FileNotFoundError(f"No candidates dir: {CANDIDATES_DIR}")

    metas: dict[str, dict] = {}
    for json_path in sorted(CANDIDATES_DIR.glob("*.json")):
        meta = json.loads(json_path.read_text())
        if meta.get("snapshot") != snapshot:
            print(f"  [skip] {json_path.name} snapshot={meta.get('snapshot')} != {snapshot}")
            continue
        metas[meta["name"]] = meta

    if not metas:
        raise RuntimeError(
            f"No candidate JSONs for snapshot={snapshot} in {CANDIDATES_DIR}. "
            "Run the per-model train tasks first."
        )

    results = {
        name: {
            "val_metrics": m["val_metrics"],
            "hit_target": m["hit_target"],
        }
        for name, m in metas.items()
    }
    winner_name, reason = pick_winner(results, target_precision)
    winner_meta = metas[winner_name]
    winner_pkl = Path(winner_meta["pkl_path"])
    print(f"  WINNER = {winner_name.upper()} — {reason}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    champion_pkl = MODEL_DIR / "best_spam_classifier.pkl"
    # Atomic-replace: copy to a sibling tmp then os.replace so FastAPI never
    # observes a half-written pkl.
    tmp = champion_pkl.with_suffix(".pkl.promoting")
    shutil.copy2(winner_pkl, tmp)
    tmp.replace(champion_pkl)

    out = {
        "snapshot": snapshot,
        "trained_at": datetime.now(UTC).isoformat(),
        "target_precision": target_precision,
        "model_choice_flag": "auto_split",
        "winner": winner_name,
        "winner_reason": reason,
        "model_type": winner_meta["model_type"],
        "threshold": winner_meta["threshold"],
        "threshold_hit_target": winner_meta["hit_target"],
        "feature_subset": winner_meta["feature_subset"],
        "val_metrics": winner_meta["val_metrics"],
        "test_metrics": winner_meta["test_metrics"],
        "candidates": {n: {k: v for k, v in m.items() if k != "pkl_path"} for n, m in metas.items()},
        "n_features": winner_meta["n_features"],
        "n_tfidf_features": winner_meta["n_tfidf_features"],
        "n_train": winner_meta["n_train"],
        "n_val": winner_meta["n_val"],
        "n_test": winner_meta["n_test"],
        "gold_artifacts": {
            "vectorizer": f"data/gold/snapshot={snapshot}/artifacts/tfidf_vectorizer.pkl",
            "scaler": f"data/gold/snapshot={snapshot}/artifacts/numeric_scaler.pkl",
        },
    }
    with open(MODEL_DIR / "train.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  champion → {champion_pkl}\n  train.json updated")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train one or all candidates on a Gold snapshot."
    )
    parser.add_argument(
        "--snapshot", required=True,
        help="Holdout month snapshot name (YYYY-MM, e.g. 2026-04)"
    )
    parser.add_argument(
        "--target-precision", type=float, default=0.99,
        help="Minimum precision for threshold selection (default 0.99, FSD §11.2)"
    )
    parser.add_argument(
        "--model", choices=["auto", "lr", "nb", "rf", "xgb"], default="auto",
        help="auto = train all 4 in one process (heavy); otherwise train a single model."
    )
    parser.add_argument(
        "--mode", choices=["winner", "candidate", "pick"], default="winner",
        help=(
            "winner: legacy single-process train → best_spam_classifier.pkl (default). "
            "candidate: train --model and save to models/candidates/<name>.{pkl,json}. "
            "pick: read models/candidates/*.json, pick winner, write best_spam_classifier.pkl."
        ),
    )
    args = parser.parse_args()

    if args.mode == "winner":
        main(args.snapshot, args.target_precision, args.model)
    elif args.mode == "candidate":
        if args.model == "auto":
            parser.error("--mode candidate requires --model {lr|nb|rf|xgb}")
        train_single_candidate(args.snapshot, args.model, args.target_precision)
    else:  # pick
        pick_winner_from_candidates(args.snapshot, args.target_precision)