"""
Split emails.csv gốc → data/raw/emails_YYYY-MM.csv
Phần dư < 50% ROWS_PER_MONTH sẽ gộp vào tháng cuối.

Chạy: py -m src.utils.split_raw

"""
import pandas as pd
from pathlib import Path

RAW_CSV        = Path("data/raw/spam_Emails_data.csv")
RAW_DIR        = Path("data/raw/by_month")
ROWS_PER_MONTH = 10_000
BASE_YEAR      = 2024
BASE_MONTH     = 11

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower()
    if "text" in df.columns and "label" in df.columns:
        df = df.rename(columns={"text": "body", "label": "raw_label"})
    elif "message" in df.columns and "category" in df.columns:
        df = df.rename(columns={"message": "body", "category": "raw_label"})
    else:
        raise ValueError(f"Unknown columns: {df.columns.tolist()}")
    return df[["body", "raw_label"]]

def month_label(offset: int) -> str:
    total_months = BASE_MONTH - 1 + offset
    year  = BASE_YEAR + total_months // 12
    month = total_months % 12 + 1
    return f"{year}-{month:02d}"

def build_chunks(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    chunks = []

    for i, start in enumerate(range(0, len(df), ROWS_PER_MONTH)):
        chunk = df.iloc[start : start + ROWS_PER_MONTH].copy()

        # Nếu chunk cuối < 50% ROWS_PER_MONTH → gộp vào tháng trước
        if len(chunk) < ROWS_PER_MONTH // 2 and chunks:
            last_month, last_chunk = chunks[-1]
            merged = pd.concat([last_chunk, chunk], ignore_index=True)
            chunks[-1] = (last_month, merged)
            print(
                f"  ↩ Dư {len(chunk):,} rows → gộp vào {last_month} "
                f"(tổng {len(merged):,} rows)"
            )
        else:
            chunks.append((month_label(i), chunk))

    return chunks

if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_CSV)
    df = normalize_columns(df)
    print(f"Loaded {len(df):,} rows\n")

    chunks = build_chunks(df)

    for month, chunk in chunks:
        out = RAW_DIR / f"emails_{month}.csv"
        chunk.to_csv(out, index=False)
        flag = "← tháng cuối (có dư)" if len(chunk) != ROWS_PER_MONTH else ""
        print(f"  ✓ {out.name}  ({len(chunk):,} rows)  {flag}")

    total_written = sum(len(c) for _, c in chunks)
    print(f"\nDone — {len(chunks)} files | {total_written:,} rows written to {RAW_DIR}")