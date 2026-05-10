"""
Chạy mỗi tháng bởi monthly_run.py
Chỉ ingest đúng 1 file: data/raw/emails_{month}.csv → Bronze partition

Ingest từng tháng:
    py -m src.etl.bronze_ingest --month YYYY-MM (Ví dụ: 2025-05)

"""
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime, timedelta, UTC
from dateutil.relativedelta import relativedelta
import argparse

BRONZE_DIR     = Path("data/bronze")
RAW_DIR        = Path("data/raw/by_month")
ROWS_PER_MONTH = 10_000
BASE_MONTH     = datetime(2024, 11, 1)
SEED           = 42

def inject_received_at(df: pd.DataFrame, month: str) -> pd.DataFrame:
    """Inject timestamp ngẫu nhiên trong đúng tháng được chỉ định."""
    rng        = np.random.default_rng(SEED)
    base       = datetime.strptime(month, "%Y-%m")
    day_jitter = rng.integers(0, 28, size=len(df))
    hr_jitter  = rng.integers(0, 24, size=len(df))
    mn_jitter  = rng.integers(0, 60, size=len(df))

    df["received_at"] = [
        base + timedelta(days=int(d), hours=int(h), minutes=int(m))
        for d, h, m in zip(day_jitter, hr_jitter, mn_jitter)
    ]
    df["month_partition"] = month
    return df

def already_ingested(month: str) -> bool:
    """Kiểm tra Bronze partition đã tồn tại chưa — idempotent guard."""
    path = BRONZE_DIR / f"month_partition={month}" / "data.parquet"
    return path.exists()

def ingest(month: str):
    # Guard: skip nếu đã ingest rồi
    if already_ingested(month):
        print(f"[SKIP] Bronze partition {month} đã tồn tại.")
        return

    # Load raw file by month
    raw_path = RAW_DIR / f"emails_{month}.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file không tồn tại: {raw_path}")

    df = pd.read_csv(raw_path)
    df.columns = df.columns.str.strip().str.lower()
    df["email_id"] = [
        f"email_{month.replace('-','')}_{i:05d}" for i in range(len(df))
    ]

    # Convert the label: ham:0, spam:1, other: NaN.
    df["label"] = df["raw_label"].str.strip().str.lower().map({"ham": 0, "spam": 1})

    # Inject timestamps
    df = inject_received_at(df, month)

    # Write Bronze partition
    out_dir = BRONZE_DIR / f"month_partition={month}"
    out_dir.mkdir(parents=True, exist_ok=True)

    schema = pa.schema([
        pa.field("email_id",    pa.string()),
        pa.field("body",        pa.string()),
        pa.field("label",       pa.int8()),
        pa.field("raw_label",   pa.string()),
        pa.field("received_at", pa.timestamp("us")),
    ])
    table = pa.Table.from_pandas(
        df[["email_id", "body", "label", "raw_label", "received_at"]],
        schema=schema,
    )
    pq.write_table(table, out_dir / "data.parquet", compression="snappy")

    # Append ingestion log
    log_path = BRONZE_DIR / "_ingestion_log.csv"
    log_row  = pd.DataFrame([{
        "month_partition":  month,
        "row_count":        len(df),
        "spam_count":       int(df["label"].sum()),
        "ham_count":        int((df["label"] == 0).sum()),
        "min_received_at":  df["received_at"].min(),
        "max_received_at":  df["received_at"].max(),
        "ingested_at":      datetime.now(UTC),
        "source_file":      raw_path.name,
    }])
    header = not log_path.exists()
    log_row.to_csv(log_path, mode="a", header=header, index=False)

    print(f"[OK] Ingested {month}: {len(df):,} rows → {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", required=True, help="YYYY-MM")
    args = parser.parse_args()
    ingest(args.month)