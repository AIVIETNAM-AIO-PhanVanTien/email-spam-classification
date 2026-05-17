"""
Monthly run — orchestrates the per-month retrain loop.

Pipeline:
    1. ingest_month         — bronze_ingest + silver_transform for the new month
    2. compute_drift        — drift detector vs the reference window
    3. champion_challenger  — only if drift_score >= DRIFT_SCORE_THRESHOLD:
         a. rebuild gold snapshot ≤ month (auto-discover)
         b. train challenger via src.pipelines.train (writes to a tmp pkl)
         c. evaluate champion + challenger on the SAME gold test set
         d. atomic promote if challenger.f1 >= champion.f1 + PROMOTION_F1_DELTA
            (backup current champion → models/versions/v{N}.pkl,
             os.replace(challenger_tmp, best_spam_classifier.pkl))
    4. reload_fastapi       — POST /admin/reload-model (handled by the DAG)

This module exposes Python helpers (used by Airflow DAGs) AND a CLI entry-point
for manual runs / debugging:

    python -m src.pipelines.monthly_run \
        --month 2025-11 --ref-start 2025-05 --ref-end 2025-10
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from src.etl import bronze_ingest, gold_build, silver_transform
from src.monitoring.drift_detector import compute_drift_for_month
from src.pipelines import train as train_module
from src.pipelines.evaluate import evaluate, load_split

MODEL_DIR = Path("models")
VERSIONS_DIR = MODEL_DIR / "versions"
DECISIONS_DIR = MODEL_DIR / "decisions"
REPORTS_DIR = Path("reports/monthly_runs")

CHAMPION_PKL = MODEL_DIR / "best_spam_classifier.pkl"
CHAMPION_META = MODEL_DIR / "train.json"

DEFAULT_DRIFT_THRESHOLD = float(os.getenv("DRIFT_SCORE_THRESHOLD", "0.25"))
DEFAULT_PROMOTION_DELTA = float(os.getenv("PROMOTION_F1_DELTA", "0.0"))
DEFAULT_TARGET_PRECISION = float(os.getenv("TARGET_PRECISION", "0.99"))


# ── Step 1: per-month ingest ─────────────────────────────────────────────────


def ingest_month(month: str) -> dict:
    """Run bronze + silver for `month` (both idempotent — skip if already done)."""
    bronze_ingest.ingest(month)
    silver_transform.process(month)
    return {"month": month, "status": "ingested"}


# ── Step 2: drift (delegated) ────────────────────────────────────────────────
# Re-exported here so DAG code can `from src.pipelines.monthly_run import compute_drift_for_month`.
__all__ = [
    "ingest_month",
    "compute_drift_for_month",
    "run_champion_challenger",
    "champion_challenger_from_candidates",
    "monthly_run",
]


# ── Step 3: champion / challenger ────────────────────────────────────────────


def _next_version_number() -> int:
    """Highest existing v{N}.pkl, plus 1. Starts at 1 on a clean install."""
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    nums = []
    for p in VERSIONS_DIR.glob("v*.pkl"):
        m = re.match(r"v(\d+)\.pkl$", p.name)
        if m:
            nums.append(int(m.group(1)))
    return max(nums, default=0) + 1


def _evaluate_on_test(model, snapshot: str, threshold: float, n_tfidf: int) -> dict:
    """Score `model` on data/gold/snapshot=<snapshot>/full_load/test.{parquet,npz}."""
    X_test, y_test = load_split(snapshot, "test")

    if type(model).__name__ == "MultinomialNB":
        X_test = X_test[:, :n_tfidf]

    y_prob = model.predict_proba(X_test)[:, 1]
    return evaluate(y_test, y_prob, threshold)


def _load_champion_state() -> dict[str, Any] | None:
    """Returns {classifier, threshold, snapshot, feature_subset, ...} or None if no champion yet."""
    if not CHAMPION_PKL.exists() or not CHAMPION_META.exists():
        return None
    meta = json.loads(CHAMPION_META.read_text())
    return {
        "classifier": joblib.load(CHAMPION_PKL),
        "threshold": float(meta.get("threshold", 0.5)),
        "snapshot": meta["snapshot"],
        "feature_subset": meta.get("feature_subset", "full"),
        "winner": meta.get("winner"),
        "trained_at": meta.get("trained_at"),
        "model_type": meta.get("model_type"),
    }


def _backup_local_artifacts() -> dict[str, Path | None]:
    """Snapshot the current champion pkl + train.json to `.bak` siblings.

    Trainer overwrites both unconditionally; we restore them on rejection so
    the champion file on disk is byte-identical to before the run.
    """
    backups: dict[str, Path | None] = {"pkl": None, "meta": None}
    if CHAMPION_PKL.exists():
        pkl_bak = CHAMPION_PKL.with_suffix(".pkl.bak")
        shutil.copy2(CHAMPION_PKL, pkl_bak)
        backups["pkl"] = pkl_bak
    if CHAMPION_META.exists():
        meta_bak = CHAMPION_META.with_suffix(".json.bak")
        shutil.copy2(CHAMPION_META, meta_bak)
        backups["meta"] = meta_bak
    return backups


def _restore_local_artifacts(backups: dict[str, Path | None]) -> None:
    if backups.get("pkl") and backups["pkl"].exists():
        os.replace(backups["pkl"], CHAMPION_PKL)
    if backups.get("meta") and backups["meta"].exists():
        os.replace(backups["meta"], CHAMPION_META)


def _clear_backups(backups: dict[str, Path | None]) -> None:
    for p in backups.values():
        if p and p.exists():
            p.unlink()


def _ensure_gold_snapshot(month: str) -> None:
    """Build data/gold/snapshot=<month>/ if missing. Idempotent skip handled by gold_build."""
    snap = Path(f"data/gold/snapshot={month}")
    if (snap / "full_load" / "train.parquet").exists():
        return
    print(f"[monthly_run] Building gold snapshot for {month}")
    gold_build.build(month)


def _write_decision(month: str, decision: dict) -> None:
    DECISIONS_DIR.mkdir(parents=True, exist_ok=True)
    with (DECISIONS_DIR / f"{month}.json").open("w") as f:
        json.dump(decision, f, indent=2, default=str)

    out_dir = REPORTS_DIR / month
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "decision.json").open("w") as f:
        json.dump(decision, f, indent=2, default=str)


def _write_comparison(month: str, comparison: dict) -> None:
    out_dir = REPORTS_DIR / month
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "champion_challenger.json").open("w") as f:
        json.dump(comparison, f, indent=2, default=str)


def _compare_and_promote(
    month: str,
    train_meta: dict,
    champion_before: dict | None,
    backups: dict[str, Path | None],
    promotion_delta: float,
) -> dict:
    """Compare the in-place challenger (CHAMPION_PKL post-train) against the backed-up
    champion. Atomic promote or restore.

    Pre-conditions:
        - CHAMPION_PKL on disk = freshly-trained challenger pkl
        - CHAMPION_META on disk = challenger train.json
        - `backups` holds the previous champion files (or None entries if first run)
        - `champion_before` is the previous champion loaded into memory (or None)
        - `train_meta` is the dict returned by train.main / pick_winner_from_candidates
    """
    challenger_path_tmp = CHAMPION_PKL.with_suffix(".pkl.challenger")
    shutil.copy2(CHAMPION_PKL, challenger_path_tmp)

    challenger_model = joblib.load(challenger_path_tmp)
    challenger_threshold = float(train_meta["threshold"])
    challenger_snapshot = train_meta["snapshot"]
    n_tfidf = train_module.get_tfidf_vocab_size(challenger_snapshot)

    challenger_metrics = _evaluate_on_test(
        challenger_model, challenger_snapshot, challenger_threshold, n_tfidf
    )

    if champion_before is not None:
        # The champion was trained on its own snapshot; we still evaluate it on the
        # *new* month's gold test so the comparison is fair.
        champion_metrics = _evaluate_on_test(
            champion_before["classifier"],
            challenger_snapshot,
            champion_before["threshold"],
            n_tfidf,
        )
    else:
        champion_metrics = None

    if champion_before is None:
        promote = True
        reason = "no champion yet — auto-promote challenger"
    else:
        delta = challenger_metrics["f1"] - champion_metrics["f1"]
        if delta >= promotion_delta:
            promote = True
            reason = (
                f"challenger.f1={challenger_metrics['f1']:.4f} >= "
                f"champion.f1={champion_metrics['f1']:.4f} + delta={promotion_delta}"
            )
        else:
            promote = False
            reason = (
                f"challenger.f1={challenger_metrics['f1']:.4f} < "
                f"champion.f1={champion_metrics['f1']:.4f} + delta={promotion_delta} (rejected)"
            )

    comparison = {
        "month": month,
        "evaluated_on": f"data/gold/snapshot={challenger_snapshot}/full_load/test",
        "promotion_delta": promotion_delta,
        "champion": (
            {
                "snapshot": champion_before["snapshot"],
                "model_type": champion_before["model_type"],
                "winner": champion_before["winner"],
                "threshold": champion_before["threshold"],
                "metrics": {k: v for k, v in champion_metrics.items() if k != "confusion_matrix"},
                "confusion_matrix": champion_metrics["confusion_matrix"],
            }
            if champion_before is not None
            else None
        ),
        "challenger": {
            "snapshot": challenger_snapshot,
            "model_type": type(challenger_model).__name__,
            "winner": train_meta.get("winner"),
            "threshold": challenger_threshold,
            "metrics": {k: v for k, v in challenger_metrics.items() if k != "confusion_matrix"},
            "confusion_matrix": challenger_metrics["confusion_matrix"],
        },
    }
    _write_comparison(month, comparison)

    if promote:
        version = _next_version_number()
        if backups.get("pkl") is not None:
            shutil.copy2(backups["pkl"], VERSIONS_DIR / f"v{version}.pkl")
        # CHAMPION_PKL already holds the challenger bytes — the .replace below is
        # just an atomicity guarantee against concurrent readers.
        os.replace(challenger_path_tmp, CHAMPION_PKL)
        _clear_backups(backups)
        decision: dict[str, Any] = {
            "month": month,
            "promoted": True,
            "reason": reason,
            "new_version": f"v{version}",
            "champion_pkl_sha": _short_sha(CHAMPION_PKL),
            "decided_at": datetime.now(UTC).isoformat(),
        }
    else:
        if challenger_path_tmp.exists():
            challenger_path_tmp.unlink()
        _restore_local_artifacts(backups)
        decision = {
            "month": month,
            "promoted": False,
            "reason": reason,
            "decided_at": datetime.now(UTC).isoformat(),
        }

    _write_decision(month, decision)
    print(f"[monthly_run] decision={decision['promoted']} reason='{decision['reason']}'")
    return decision


def run_champion_challenger(
    month: str,
    promotion_delta: float = DEFAULT_PROMOTION_DELTA,
    target_precision: float = DEFAULT_TARGET_PRECISION,
) -> dict:
    """Single-process: build gold → train all 4 in one process → compare/promote.

    Heavy on RAM (≈6 GB on this dataset). The Airflow path prefers
    `champion_challenger_from_candidates` so the 4 candidates train as separate tasks.
    """
    _ensure_gold_snapshot(month)
    champion_before = _load_champion_state()
    backups = _backup_local_artifacts()

    try:
        train_meta = train_module.main(
            snapshot=month, target_precision=target_precision, model_choice="auto",
        )
    except Exception:
        _restore_local_artifacts(backups)
        raise

    return _compare_and_promote(month, train_meta, champion_before, backups, promotion_delta)


def champion_challenger_from_candidates(
    month: str,
    promotion_delta: float = DEFAULT_PROMOTION_DELTA,
    target_precision: float = DEFAULT_TARGET_PRECISION,
) -> dict:
    """DAG path: assumes 4 candidates already exist in models/candidates/.

    Picks winner from candidates (writes new pkl + train.json), then compares vs
    the previous champion on the new gold test set, atomic-swaps if it wins.
    Gold snapshot must already exist — call gold_build as a separate task upstream.
    """
    champion_before = _load_champion_state()
    backups = _backup_local_artifacts()

    try:
        train_meta = train_module.pick_winner_from_candidates(month, target_precision)
    except Exception:
        _restore_local_artifacts(backups)
        raise

    return _compare_and_promote(month, train_meta, champion_before, backups, promotion_delta)


def _short_sha(path: Path, n: int = 8) -> str:
    """First-n hex digits of the SHA-256 of `path`. Used as a model "version" tag."""
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


# ── Step 4: reload (deferred to the DAG via curl) ────────────────────────────


def reload_fastapi(api_url: str | None = None) -> bool:
    """Best-effort POST /admin/reload-model. Returns True on 200, False otherwise.

    Airflow normally does this with a BashOperator (`curl`); this is a Python helper
    for the standalone CLI flow.
    """
    import urllib.error
    import urllib.request

    url = api_url or os.getenv(
        "EMAIL_SPAM_API_RELOAD_URL", "http://127.0.0.1:8000/admin/reload-model"
    )
    try:
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"[monthly_run] reload-model HTTP {resp.status}")
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"[monthly_run] WARN reload-model failed: {e}")
        return False


# ── Orchestrator ─────────────────────────────────────────────────────────────


def monthly_run(
    month: str,
    ref_start: str,
    ref_end: str,
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
    promotion_delta: float = DEFAULT_PROMOTION_DELTA,
    target_precision: float = DEFAULT_TARGET_PRECISION,
    reload: bool = True,
) -> dict:
    """End-to-end monthly loop callable from the CLI or a single DAG task."""
    ingest_month(month)
    drift_report = compute_drift_for_month(month, ref_start, ref_end)

    outcome: dict[str, Any] = {
        "month": month,
        "ref_start": ref_start,
        "ref_end": ref_end,
        "drift_threshold": drift_threshold,
        "drift_score": drift_report["drift_score"],
        "trained_challenger": False,
        "promoted": False,
        "reason": "drift below threshold — skip retrain",
    }

    if drift_report["drift_score"] >= drift_threshold:
        decision = run_champion_challenger(
            month, promotion_delta=promotion_delta, target_precision=target_precision
        )
        outcome["trained_challenger"] = True
        outcome["promoted"] = decision["promoted"]
        outcome["reason"] = decision["reason"]
        if reload and decision["promoted"]:
            outcome["reloaded"] = reload_fastapi()
    else:
        # Still persist a decision so /<month>/ always has a complete trail
        _write_decision(
            month,
            {
                "month": month,
                "promoted": False,
                "reason": outcome["reason"],
                "drift_score": drift_report["drift_score"],
                "drift_threshold": drift_threshold,
                "decided_at": datetime.now(UTC).isoformat(),
            },
        )

    print(f"[monthly_run] outcome: {json.dumps(outcome, default=str)}")
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the monthly retrain loop end-to-end.")
    parser.add_argument("--month", required=True, help="New month (YYYY-MM)")
    parser.add_argument("--ref-start", required=True, help="First reference month (YYYY-MM)")
    parser.add_argument("--ref-end", required=True, help="Last reference month (YYYY-MM)")
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=DEFAULT_DRIFT_THRESHOLD,
        help=f"Default {DEFAULT_DRIFT_THRESHOLD} (env DRIFT_SCORE_THRESHOLD)",
    )
    parser.add_argument(
        "--promotion-delta",
        type=float,
        default=DEFAULT_PROMOTION_DELTA,
        help=f"Default {DEFAULT_PROMOTION_DELTA} (env PROMOTION_F1_DELTA)",
    )
    parser.add_argument(
        "--target-precision", type=float, default=DEFAULT_TARGET_PRECISION
    )
    parser.add_argument("--no-reload", action="store_true", help="Skip POST /admin/reload-model")
    args = parser.parse_args()

    monthly_run(
        month=args.month,
        ref_start=args.ref_start,
        ref_end=args.ref_end,
        drift_threshold=args.drift_threshold,
        promotion_delta=args.promotion_delta,
        target_precision=args.target_precision,
        reload=not args.no_reload,
    )
