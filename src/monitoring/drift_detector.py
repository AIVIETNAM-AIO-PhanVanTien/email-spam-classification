"""
Drift detector — compares a "reference window" (older months) against a
"new window" (the month just ingested) and returns an aggregate `drift_score`.

Three signals are combined:
    1. Label drift   — chi-square on P(spam) distribution
    2. Feature drift — KS + PSI on top TF-IDF token means
    3. Aggregate    — drift_score = 0.7 * feature_score + 0.3 * label_score

The monthly DAG branches on `drift_score >= DRIFT_SCORE_THRESHOLD` (default 0.25)
to decide whether to train a challenger.

Loads Silver partitions directly (the cheapest source-of-truth for both label +
cleaned text). Gold is not required here — drift is run *before* gold rebuild.

CLI:
    python -m src.monitoring.drift_detector \
        --month 2025-11 --ref-start 2025-05 --ref-end 2025-10
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer

SILVER_DIR = Path("data/silver")
REPORTS_DIR = Path("reports/monthly_runs")

DEFAULT_TOP_K_FEATURES = 100
PSI_BIN_COUNT = 10
PSI_EPSILON = 1e-6
KS_PVALUE_CUTOFF = 0.05
PSI_DRIFT_CUTOFF = 0.1
PSI_FEATURE_NORM = 0.25  # PSI >= 0.25 considered fully drifted → feature_score=1


def _load_silver_months(months: list[str]) -> pd.DataFrame:
    frames = []
    for month in months:
        path = SILVER_DIR / f"month_partition={month}" / "data_silver.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Silver partition missing: {path}")
        df = pq.read_table(path, columns=["body_clean", "label"]).to_pandas()
        df["month_partition"] = month
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _months_in_range(start: str, end: str) -> list[str]:
    """Return sorted list of YYYY-MM strings (inclusive) within Silver partitions."""
    months = []
    for p in SILVER_DIR.glob("month_partition=*"):
        m = p.name.split("=")[1]
        if start <= m <= end and (p / "data_silver.parquet").exists():
            months.append(m)
    if not months:
        raise FileNotFoundError(
            f"No silver partitions found in [{start}, {end}] under {SILVER_DIR}"
        )
    return sorted(months)


def chi_square_label_drift(ref_labels: np.ndarray, new_labels: np.ndarray) -> dict:
    """Chi-square test on the 2x2 contingency table (ham/spam) between windows."""
    ref_counts = np.array([(ref_labels == 0).sum(), (ref_labels == 1).sum()])
    new_counts = np.array([(new_labels == 0).sum(), (new_labels == 1).sum()])
    contingency = np.vstack([ref_counts, new_counts])

    chi2, p_value, _, _ = stats.chi2_contingency(contingency)

    ref_total = max(int(ref_counts.sum()), 1)
    new_total = max(int(new_counts.sum()), 1)
    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "drifted": bool(p_value < KS_PVALUE_CUTOFF),
        "ref_dist": {
            "ham": float(ref_counts[0] / ref_total),
            "spam": float(ref_counts[1] / ref_total),
        },
        "new_dist": {
            "ham": float(new_counts[0] / new_total),
            "spam": float(new_counts[1] / new_total),
        },
        "ref_n": int(ref_counts.sum()),
        "new_n": int(new_counts.sum()),
    }


def population_stability_index(
    ref_values: np.ndarray, new_values: np.ndarray, n_bins: int = PSI_BIN_COUNT
) -> float:
    """PSI on a single numeric column. Bins fitted on `ref_values`.

    PSI = sum((p_new - p_ref) * log(p_new / p_ref)) across bins.
    Returns a non-negative float; 0 = identical, >0.25 considered significant.
    """
    ref = np.asarray(ref_values, dtype=float)
    new = np.asarray(new_values, dtype=float)
    if ref.size == 0 or new.size == 0:
        return 0.0

    # Quantile bins on ref; fall back to a tiny epsilon span if ref is constant.
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if edges.size < 2:
        edges = np.array([ref.min() - PSI_EPSILON, ref.max() + PSI_EPSILON])
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_hist, _ = np.histogram(ref, bins=edges)
    new_hist, _ = np.histogram(new, bins=edges)

    ref_p = ref_hist / max(ref_hist.sum(), 1)
    new_p = new_hist / max(new_hist.sum(), 1)

    ref_p = np.where(ref_p == 0, PSI_EPSILON, ref_p)
    new_p = np.where(new_p == 0, PSI_EPSILON, new_p)

    return float(np.sum((new_p - ref_p) * np.log(new_p / ref_p)))


def feature_drift(
    ref_texts: pd.Series, new_texts: pd.Series, top_k: int = DEFAULT_TOP_K_FEATURES
) -> dict:
    """KS + PSI per top-k TF-IDF feature; vectorizer is fit on `ref_texts` only."""
    vectorizer = TfidfVectorizer(max_features=top_k, sublinear_tf=True, min_df=2)
    ref_matrix = vectorizer.fit_transform(ref_texts.fillna(""))
    new_matrix = vectorizer.transform(new_texts.fillna(""))
    feature_names = vectorizer.get_feature_names_out()

    drifted_features = []
    psi_values = []
    for i, name in enumerate(feature_names):
        ref_col = np.asarray(ref_matrix[:, i].todense()).ravel()
        new_col = np.asarray(new_matrix[:, i].todense()).ravel()

        # KS-test on the per-document TF-IDF weight distribution
        try:
            ks_stat, ks_p = stats.ks_2samp(ref_col, new_col)
        except ValueError:
            ks_stat, ks_p = 0.0, 1.0

        psi = population_stability_index(ref_col, new_col)
        psi_values.append(psi)

        if ks_p < KS_PVALUE_CUTOFF and psi > PSI_DRIFT_CUTOFF:
            drifted_features.append(
                {
                    "token": str(name),
                    "ks_stat": float(ks_stat),
                    "ks_p_value": float(ks_p),
                    "psi": float(psi),
                }
            )

    psi_arr = np.asarray(psi_values) if psi_values else np.array([0.0])
    drifted_features.sort(key=lambda d: d["psi"], reverse=True)

    return {
        "n_features": int(len(feature_names)),
        "ks_drifted_features": int(len(drifted_features)),
        "mean_psi": float(psi_arr.mean()),
        "max_psi": float(psi_arr.max()),
        "drifted_top_features": drifted_features[:20],
    }


def aggregate_drift_score(label: dict, feature: dict) -> float:
    """Combine label + feature drift into a single score in [0, 1]."""
    feature_score = min(feature["mean_psi"] / PSI_FEATURE_NORM, 1.0)
    label_score = 1.0 if label["drifted"] else 0.0
    return float(max(0.7 * feature_score + 0.3 * label_score, 0.0))


def detect_drift(
    ref_df: pd.DataFrame, new_df: pd.DataFrame, top_k: int = DEFAULT_TOP_K_FEATURES
) -> dict:
    """Run all three checks; returns the dict persisted to drift_report.json."""
    label = chi_square_label_drift(ref_df["label"].values, new_df["label"].values)
    feature = feature_drift(ref_df["body_clean"], new_df["body_clean"], top_k=top_k)
    return {
        "computed_at": datetime.now(UTC).isoformat(),
        "ref_size": int(len(ref_df)),
        "new_size": int(len(new_df)),
        "label_drift": label,
        "feature_drift": feature,
        "drift_score": aggregate_drift_score(label, feature),
    }


def compute_drift_for_month(
    month: str,
    ref_start: str,
    ref_end: str,
    top_k: int = DEFAULT_TOP_K_FEATURES,
) -> dict:
    """End-to-end helper used by the Airflow DAG and the CLI.

    Loads silver, computes drift, persists `reports/monthly_runs/<month>/drift_report.json`.
    """
    ref_months = _months_in_range(ref_start, ref_end)
    if month in ref_months:
        raise ValueError(
            f"`month` ({month}) overlaps the reference window [{ref_start},{ref_end}]"
        )

    ref_df = _load_silver_months(ref_months)
    new_df = _load_silver_months([month])

    report = detect_drift(ref_df, new_df, top_k=top_k)
    report.update(
        {"month": month, "ref_start": ref_start, "ref_end": ref_end, "ref_months": ref_months}
    )

    out_dir = REPORTS_DIR / month
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "drift_report.json").open("w") as f:
        json.dump(report, f, indent=2, default=str)

    print(
        f"[drift] month={month} ref=[{ref_start}..{ref_end}] "
        f"drift_score={report['drift_score']:.4f} "
        f"label_drifted={report['label_drift']['drifted']} "
        f"ks_features={report['feature_drift']['ks_drifted_features']}"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute drift for a month vs a reference window.")
    parser.add_argument("--month", required=True, help="New month (YYYY-MM)")
    parser.add_argument("--ref-start", required=True, help="First reference month (YYYY-MM)")
    parser.add_argument("--ref-end", required=True, help="Last reference month (YYYY-MM)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K_FEATURES)
    args = parser.parse_args()
    compute_drift_for_month(args.month, args.ref_start, args.ref_end, top_k=args.top_k)
