"""
Silver layer — monthly processing pipeline.
Bronze partition → clean → feature extract → quality check → Silver partition.

EDA findings đã áp dụng (01_eda.ipynb):
    - Dataset plain text: bỏ run_header_parsing, run_html_features, run_url_features
    - Phát hiện 'escapenumber' obfuscation → xử lý trong TextCleaner
    - Ngưỡng drop short text: nâng từ 20 lên 50 chars
    - Outlier max = 11.5M chars → truncate body_clean > 100,000 chars
    - Label balanced (~50/50) → không cần sampling strategy ở Gold

Pipeline:
    ① Text cleaning       — normalize, lowercase, fix noise, remove escapenumber
    ② Text features       — char/word/ratio features, spam keywords, has_escapenumber
    ③ Quality check       — truncate quá dài, drop quá ngắn, drop null label
    ④ Write parquet       — lưu Silver partition + quality report

Chạy mỗi tháng bởi monthly_run.py:
    py -m src.data.silver_transform --month YYYY-MM (Ví dụ: 2025-05)
"""
import re
import json
import argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime

from src.data.text_preprocessing import TextCleaner
from src.data.data_quality_check import TextDataQuality

BRONZE_DIR = Path("data/bronze")
SILVER_DIR = Path("data/silver")

# ── Quality thresholds ────────────────────────────────────────────────────────
# Xác nhận từ EDA:
#   % < 50 chars = 2.2%  → emails ngắn < 50 chars gần như không có thông tin
#   max = 11.5M chars    → truncate để tránh làm chậm pipeline
MIN_CHAR_COUNT = 50
MAX_CHAR_COUNT = 100_000

# Spam keywords — đếm trên body_clean đã lowercase
SPAM_KEYWORDS = [
    "free", "win", "winner", "cash", "prize", "offer",
    "discount", "cheap", "deal", "limited", "urgent",
    "congratulations", "unsubscribe", "credit", "loan",
    "viagra", "casino", "lottery", "guarantee", "investment",
    "click here", "act now", "order now", "special promotion",
]


# ── Step 1: Text Cleaning ─────────────────────────────────────────────────────
def run_text_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch body email thô → body_clean.
    Dataset là plain text → không cần strip headers/HTML/URL.
    TextCleaner.aggressive_clean():
        normalize unicode → lowercase → remove linebreaks
        → fix repeated chars → remove escapenumber
        → remove special chars → clean spaces → remove short words
    """
    df["body_clean"] = TextCleaner(df["body"]).aggressive_clean().get()
    return df


# ── Step 2: Text Feature Engineering ─────────────────────────────────────────
def run_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính features từ body_clean (sau khi đã clean).

    Nguồn tính feature:
        body_raw   — punctuation/style features (!, ?, digit, upper)
                     vì body_clean đã normalize những thứ này
        body_clean — length, vocabulary, spam keyword features
    """
    body_raw   = df["body"].astype(str)
    body_clean = df["body_clean"]

    # ── Length features ───────────────────────────────────────────────────────
    df["char_count"]     = body_clean.str.len().astype("int32")
    df["word_count"]     = body_clean.str.split().str.len().fillna(0).astype("int32")
    df["n_unique_words"] = body_clean.str.split().apply(
        lambda x: len(set(x)) if isinstance(x, list) else 0
    ).astype("int32")
    df["avg_word_length"] = (
        df["char_count"] / df["word_count"].replace(0, 1)
    ).round(2).astype("float32")

    # ── Log-scale length ──────────────────────────────────────────────────────
    # Ổn định hơn cho model vì char_count có phân phối lệch phải mạnh (std=47k)
    df["log_chars"] = np.log1p(df["char_count"]).round(4).astype("float32")
    df["log_words"] = np.log1p(df["word_count"]).round(4).astype("float32")

    # ── Vocabulary richness ───────────────────────────────────────────────────
    df["unique_word_ratio"] = (
        df["n_unique_words"] / df["word_count"].replace(0, 1)
    ).round(4).astype("float32")
    df["repetition_ratio"] = (
        1 - df["unique_word_ratio"]
    ).round(4).astype("float32")
    df["complexity"] = (
        df["unique_word_ratio"] * df["avg_word_length"]
    ).round(4).astype("float32")
    df["info_density"] = (
        df["n_unique_words"] / df["char_count"].replace(0, 1)
    ).round(6).astype("float32")

    # ── Punctuation / style (từ body gốc) ────────────────────────────────────
    df["exclaim_count"]  = body_raw.str.count(r"!").astype("int16")
    df["question_count"] = body_raw.str.count(r"\?").astype("int16")
    df["digit_ratio"]    = (
        body_raw.str.count(r"\d") / body_raw.str.len().replace(0, 1)
    ).round(4).astype("float32")
    df["upper_ratio"]    = (
        body_raw.str.count(r"[A-Z]") / body_raw.str.len().replace(0, 1)
    ).round(4).astype("float32")

    # ── Spam signals ──────────────────────────────────────────────────────────
    pattern = "|".join(rf"\b{re.escape(kw)}\b" for kw in SPAM_KEYWORDS)
    df["spam_keyword_count"] = body_clean.str.count(pattern).astype("int16")

    # EDA phát hiện: escapenumber là obfuscation pattern đặc trưng của spam
    # Đếm trên body_raw
    df["has_escapenumber"] = body_raw.str.contains(
        r"escapenumber", case=False, na=False
    ).astype("int8")

    return df


# ── Step 3: Quality Check ─────────────────────────────────────────────────────
def run_quality_check(df: pd.DataFrame, month: str) -> tuple[pd.DataFrame, dict]:
    """
    3a. Truncate body_clean quá dài (> 100k chars)
        EDA: max = 11.5M chars → outlier làm chậm toàn pipeline
        Truncate thay vì drop để giữ label + phần text còn lại có nghĩa

    3b. Drop rows không đủ chất lượng:
        - body_clean < 50 chars  → EDA: 2.2% rows, gần như không có thông tin
        - label null             → không train/evaluate được

    3c. Chạy TextDataQuality → quality report đầy đủ

    3d. Drift flag: spam ratio < 10% hoặc > 90% so với baseline
    """
    initial_count = len(df)

    # Truncate trước khi tính quality metrics
    mask_too_long  = df["body_clean"].str.len() > MAX_CHAR_COUNT
    truncated_count = int(mask_too_long.sum())
    df.loc[mask_too_long, "body_clean"] = (
        df.loc[mask_too_long, "body_clean"].str[:MAX_CHAR_COUNT]
    )

    # Quality check trên body_clean đã truncate, TRƯỚC khi drop
    dq = TextDataQuality(df, "body_clean")
    quality_metrics = dq.run_all_checks(label_col="label")

    # Drop
    mask_short = df["body_clean"].str.len() < MIN_CHAR_COUNT
    mask_null  = df["label"].isna()
    dropped_short = int(mask_short.sum())
    dropped_null  = int(mask_null.sum())
    df = df[~mask_short & ~mask_null].reset_index(drop=True)

    label_dist = quality_metrics.get("label_distribution", {})
    report = {
        "month":              month,
        "initial_rows":       initial_count,
        "final_rows":         len(df),
        "truncated_rows":     truncated_count,
        "dropped_short_body": dropped_short,
        "dropped_null_label": dropped_null,
        "total_dropped":      initial_count - len(df),
        "processed_at":       datetime.utcnow().isoformat(),
        "quality_metrics":    quality_metrics,
        "drift_flag":         label_dist.get("drift_flag", False),
    }
    return df, report


# ── Step 4: Write Silver ──────────────────────────────────────────────────────
SILVER_SCHEMA = pa.schema([
    # ── Identity ──────────────────────────────────────────────────────────────
    pa.field("email_id",           pa.string()),
    pa.field("body_clean",         pa.string()),
    pa.field("label",              pa.int8()),
    pa.field("received_at",        pa.timestamp("us")),

    # ── Length features ───────────────────────────────────────────────────────
    pa.field("char_count",         pa.int32()),
    pa.field("word_count",         pa.int32()),
    pa.field("n_unique_words",     pa.int32()),
    pa.field("avg_word_length",    pa.float32()),
    pa.field("log_chars",          pa.float32()),
    pa.field("log_words",          pa.float32()),

    # ── Vocabulary richness ───────────────────────────────────────────────────
    pa.field("unique_word_ratio",  pa.float32()),
    pa.field("repetition_ratio",   pa.float32()),
    pa.field("complexity",         pa.float32()),
    pa.field("info_density",       pa.float32()),

    # ── Punctuation / style ───────────────────────────────────────────────────
    pa.field("exclaim_count",      pa.int16()),
    pa.field("question_count",     pa.int16()),
    pa.field("digit_ratio",        pa.float32()),
    pa.field("upper_ratio",        pa.float32()),

    # ── Spam signals ──────────────────────────────────────────────────────────
    pa.field("spam_keyword_count", pa.int16()),
    pa.field("has_escapenumber",   pa.int8()),
])


def write_silver_partition(df: pd.DataFrame, month: str):
    out_dir = SILVER_DIR / f"month_partition={month}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cols  = [f.name for f in SILVER_SCHEMA]
    table = pa.Table.from_pandas(df[cols], schema=SILVER_SCHEMA)
    pq.write_table(table, out_dir / "data_silver.parquet", compression="snappy")


def write_quality_report(report: dict, month: str):
    report_dir = SILVER_DIR / f"month_partition={month}"
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(report_dir / "quality_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Append vào quality log tổng — bỏ quality_metrics để file nhỏ gọn
    log_row  = {k: v for k, v in report.items() if k != "quality_metrics"}
    log_path = SILVER_DIR / "_quality_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(log_row, default=str) + "\n")


# ── Main pipeline ─────────────────────────────────────────────────────────────
def process(month: str):
    # Idempotent guard
    out_path = SILVER_DIR / f"month_partition={month}" / "data_silver.parquet"
    if out_path.exists():
        print(f"[SKIP] Silver {month} đã tồn tại.")
        return

    # Load Bronze
    bronze_path = BRONZE_DIR / f"month_partition={month}"
    if not bronze_path.exists():
        raise FileNotFoundError(f"Bronze partition không tồn tại: {bronze_path}")

    df = pq.read_table(bronze_path).to_pandas()
    print(f"[Silver] Loaded Bronze {month}: {len(df):,} rows")

    # ── Pipeline ──────────────────────────────────────────────────────────────
    df = run_text_cleaning(df)
    print(f"  1. Text cleaning done")

    df = run_text_features(df)
    print(f"  2. Text feature engineering done  ({len(df.columns)} cols total)")

    df, report = run_quality_check(df, month)
    print(f"  3. Quality check done")
    if report["truncated_rows"]:
        print(f"     truncated: {report['truncated_rows']} emails > {MAX_CHAR_COUNT:,} chars")
    print(f"     dropped:   {report['dropped_short_body']} short body "
          f"+ {report['dropped_null_label']} null label "
          f"= {report['total_dropped']} rows")

    write_silver_partition(df, month)
    write_quality_report(report, month)

    label_dist = report["quality_metrics"].get("label_distribution", {})
    spam_ratio = label_dist.get("spam_ratio", None)
    drift_flag = "⚠  DRIFT FLAG" if report["drift_flag"] else "OK"

    print(f"  4. Written → {out_path}")
    print(
        f"[{drift_flag}] Silver {month}: {report['final_rows']:,} rows"
        + (f" | spam={spam_ratio:.1%}" if isinstance(spam_ratio, float) else "")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", required=True, help="YYYY-MM")
    args = parser.parse_args()
    process(args.month)