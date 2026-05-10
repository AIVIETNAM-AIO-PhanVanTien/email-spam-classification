"""
Gold layer — full load pipeline (monthly-capable).
Auto-discover Silver partitions → feature selection → TF-IDF → train/val/test split → Gold.

Chạy:
    py -m src.etl.gold_build --month YYYY-MM

Ví dụ:
    py -m src.etl.gold_build --month 2025-09
    → Auto-discover tất cả Silver partitions ≤ 2025-09
    → Holdout test = 2025-09 (tháng cuối)
    → Train/val = tất cả tháng trước đó

EDA Silver findings áp dụng:
    Feature selection — bỏ 9 features redundant/vô nghĩa
    Giữ lại 4 numeric features: log_chars, avg_word_length, unique_word_ratio, exclaim_count
    TF-IDF trên body_clean: max_features=30,000  ngram_range=(1,2)  sublinear_tf=True
    Label ~50/50 → không cần class weighting

"""
import argparse
import joblib
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime, UTC
from scipy.sparse import csr_matrix, hstack, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SILVER_DIR   = Path("data/silver")
GOLD_DIR     = Path("data/gold")

# ── Config ────────────────────────────────────────────────────────────────────
VAL_SIZE        = 0.15         # 15% của trainval → val set
RANDOM_STATE    = 42

# ── Feature selection (kết quả EDA Silver) ───────────────────────────────────
# Bỏ: corr > 0.86, MI thấp, distribution không phân tách được
DROP_FEATURES = [
    "repetition_ratio",     # corr = -1.00 với unique_word_ratio
    "info_density",         # corr >  0.92 với complexity, unique_word_ratio
    "complexity",           # corr =  0.86 với repetition_ratio
    "spam_keyword_count",   # corr =  0.99 với char_count, word_count
    "log_words",            # corr =  0.99 với log_chars
    "has_escapenumber",     # MI = 0.003, ham (0.74) > spam (0.67) → noise
    "question_count",       # distribution giống nhau 2 class
    "digit_ratio",          # distribution giống nhau 2 class
    "upper_ratio",          # distribution giống nhau 2 class
]

# Giữ lại — đã xác nhận có separation qua EDA
NUMERIC_FEATURES = [
    "log_chars",            # đại diện độ dài (thay cho char_count, word_count, log_words)
    "avg_word_length",      # spam tập trung 7-8, ham rải rộng hơn
    "unique_word_ratio",    # đại diện vocabulary richness group
    "exclaim_count",        # punctuation signal
]

# TF-IDF config
TFIDF_CONFIG = {
    "max_features": 30_000,
    "ngram_range":  (1, 2),   # unigram + bigram: bắt được "click here", "act now"
    "min_df":       5,        # bỏ token xuất hiện < 5 docs
    "max_df":       0.95,     # bỏ token xuất hiện > 95% docs
    "sublinear_tf": True,     # log(tf) — giảm ảnh hưởng email rất dài
}


# ── Step 1: Load Silver ───────────────────────────────────────────────────────
def load_silver(months: list[str]) -> pd.DataFrame:
    """
    Load Silver partitions theo danh sách tháng chỉ định.
    Dùng pyarrow dataset để đọc hiệu quả, không load toàn bộ Silver.
    """
    frames = []
    for month in months:
        path = SILVER_DIR / f"month_partition={month}" / "data_silver.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Silver partition không tồn tại: {path}")
        df = pq.read_table(path).to_pandas()
        df["month_partition"] = month
        frames.append(df)
        print(f"  Loaded {month}: {len(df):,} rows")

    df = pd.concat(frames, ignore_index=True)
    print(f"  Total: {len(df):,} rows | {df['label'].mean():.1%} spam\n")
    return df


def discover_silver_months(up_to: str) -> list[str]:
    """
    Scan data/silver/ cho tất cả month partitions có data_silver.parquet,
    lọc chỉ giữ các tháng <= up_to.
    Returns sorted list of month strings (YYYY-MM).
    """
    months = []
    for p in SILVER_DIR.glob("month_partition=*"):
        month = p.name.split("=")[1]
        if month <= up_to and (p / "data_silver.parquet").exists():
            months.append(month)
    if not months:
        raise FileNotFoundError(
            f"Không tìm thấy Silver partition nào ≤ {up_to} trong {SILVER_DIR}"
        )
    return sorted(months)


# ── Step 2: Feature selection ─────────────────────────────────────────────────
def apply_feature_selection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bỏ các features đã xác định không có giá trị qua EDA Silver.
    Log ra để audit — quan trọng khi review lại sau này.

    Fail-fast nếu silver có NaN trong NUMERIC_FEATURES — silver phải clean,
    gold không impute âm thầm.
    """
    existing_drops = [f for f in DROP_FEATURES if f in df.columns]
    df = df.drop(columns=existing_drops)

    # Schema gate — silver phải đảm bảo no NaN
    nulls = df[NUMERIC_FEATURES].isnull().sum()
    if nulls.any():
        raise ValueError(
            f"Silver có NaN trong numeric features (silver phải clean trước, "
            f"không impute ở gold): {nulls[nulls > 0].to_dict()}"
        )

    print(f"  Dropped {len(existing_drops)} features: {existing_drops}")
    print(f"  Numeric features kept: {NUMERIC_FEATURES}")
    print(f"  Remaining cols: {len(df.columns)}\n")
    return df


# ── Step 3: Train / Val / Test split ─────────────────────────────────────────
def split_dataset(df: pd.DataFrame, holdout_month: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Tách theo thời gian trước, sau đó random split trainval.

    Lý do tách theo thời gian thay vì random:
        - Realistic hơn: model train trên quá khứ, test trên tương lai
        - Tránh data leakage: email tháng holdout không lẫn vào train
        - Holdout month = tháng cuối (truyền qua --month)

    Sau đó random split trainval → train (85%) / val (15%)
    để val không bị bias theo thứ tự thời gian trong các tháng đầu.

    Cuối cùng assert email_id no-leak giữa 3 split để catch bug nếu
    sau này split logic thay đổi (vd random split nhầm).
    """
    df_test    = df[df["month_partition"] == holdout_month].copy()
    df_trainval= df[df["month_partition"] != holdout_month].copy()

    df_train, df_val = train_test_split(
        df_trainval,
        test_size=VAL_SIZE,
        stratify=df_trainval["label"],
        random_state=RANDOM_STATE,
    )

    # Assert email_id không leak giữa 3 split
    train_ids = set(df_train["email_id"])
    val_ids   = set(df_val["email_id"])
    test_ids  = set(df_test["email_id"])
    assert not train_ids & val_ids,  "email_id leak giữa train ↔ val"
    assert not train_ids & test_ids, "email_id leak giữa train ↔ test"
    assert not val_ids & test_ids,   "email_id leak giữa val ↔ test"

    print(f"  Train:    {len(df_train):,} rows | spam={df_train['label'].mean():.1%}")
    print(f"  Val:      {len(df_val):,} rows | spam={df_val['label'].mean():.1%}")
    print(f"  Test:     {len(df_test):,} rows | spam={df_test['label'].mean():.1%} "
          f"(holdout month={holdout_month})\n")

    return df_train, df_val, df_test


# ── Step 4: TF-IDF + numeric scaling ──────────────────────────────────────────
def build_features(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
) -> tuple:
    """
    Fit TF-IDF + StandardScaler trên train set → transform val và test.

    Quan trọng:
        - Chỉ fit trên train — không được nhìn thấy val/test
        - Save vectorizer + scaler pkl → monthly pipeline dùng transform only
        - StandardScaler cho numeric features TRƯỚC khi hstack với TF-IDF —
          tránh `exclaim_count=20` dominate hàng nghìn TF-IDF token (~0.05).
        - Kết hợp TF-IDF sparse + numeric scaled (hstack)

    Output shape:
        X_train: (n_train, 30000 + 4)  — sparse matrix
        X_val:   (n_val,   30000 + 4)
        X_test:  (n_test,  30000 + 4)
    """
    print("  Fitting TF-IDF on train set...")
    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    X_train_text = vectorizer.fit_transform(df_train["body_clean"])
    X_val_text   = vectorizer.transform(df_val["body_clean"])
    X_test_text  = vectorizer.transform(df_test["body_clean"])

    print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,} tokens")
    print(f"  X_train_text shape: {X_train_text.shape}")

    # Scale numeric features (apply_feature_selection đã đảm bảo no NaN)
    print("  Fitting StandardScaler on numeric features...")
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(df_train[NUMERIC_FEATURES])
    X_val_num   = scaler.transform(df_val[NUMERIC_FEATURES])
    X_test_num  = scaler.transform(df_test[NUMERIC_FEATURES])

    # Combine TF-IDF (sparse) + numeric scaled
    X_train = hstack([X_train_text, csr_matrix(X_train_num)])
    X_val   = hstack([X_val_text,   csr_matrix(X_val_num)])
    X_test  = hstack([X_test_text,  csr_matrix(X_test_num)])

    print(f"  Final X_train shape: {X_train.shape}")
    print(f"  Final X_val shape:   {X_val.shape}")
    print(f"  Final X_test shape:  {X_test.shape}\n")

    return vectorizer, scaler, X_train, X_val, X_test


# ── Step 5: Save artifacts ────────────────────────────────────────────────────
def save_artifacts(vectorizer: TfidfVectorizer, scaler: StandardScaler,
                   months: list[str], holdout_month: str,
                   snapshot_dir: Path):
    """
    Save TF-IDF vectorizer + numeric scaler vào snapshot directory.
    Monthly pipeline dùng transform only từ artifacts của snapshot tương ứng.
    """
    artifact_dir = snapshot_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    vec_path = artifact_dir / "tfidf_vectorizer.pkl"
    joblib.dump(vectorizer, vec_path)
    print(f"  Vectorizer saved → {vec_path}")

    scaler_path = artifact_dir / "numeric_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved     → {scaler_path}")

    meta = {
        "fitted_at":        datetime.now(UTC).isoformat(),
        "train_months":     [m for m in months if m != holdout_month],
        "holdout_month":    holdout_month,
        "vocab_size":       len(vectorizer.vocabulary_),
        "tfidf_config":     {k: str(v) for k, v in TFIDF_CONFIG.items()},
        "numeric_features": NUMERIC_FEATURES,
        "scaler_mean":      scaler.mean_.tolist(),
        "scaler_scale":     scaler.scale_.tolist(),
        "dropped_features": DROP_FEATURES,
    }
    meta_path = artifact_dir / "tfidf_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved   → {meta_path}\n")


# ── Step 6: Write Gold parquet ────────────────────────────────────────────────
def write_gold_split(
    df:     pd.DataFrame,
    X,                          # sparse matrix
    split:  str,                # "train" | "val" | "test"
    snapshot_dir: Path,
):
    """
    Lưu Gold split vào snapshot_dir/full_load/.
    Metadata + numeric (raw) ở Parquet, TF-IDF sparse ở `.npz`.
    """
    full_load_dir = snapshot_dir / "full_load"
    full_load_dir.mkdir(parents=True, exist_ok=True)

    meta_cols = ["email_id", "label", "month_partition"] + NUMERIC_FEATURES
    df_out = df[meta_cols].reset_index(drop=True)
    pq.write_table(
        pa.Table.from_pandas(df_out),
        full_load_dir / f"{split}.parquet",
        compression="snappy",
    )

    save_npz(str(full_load_dir / f"{split}_X.npz"), X)

    print(f"  Gold {split} saved → {full_load_dir / split}.parquet + {split}_X.npz")


def write_gold_build_log(df_train, df_val, df_test, vectorizer, scaler,
                         months, holdout_month, snapshot_dir: Path):
    """Ghi log tổng kết gold build để audit."""
    log = {
        "built_at":          datetime.now(UTC).isoformat(),
        "months":            months,
        "holdout_month":     holdout_month,
        "train_rows":        len(df_train),
        "val_rows":          len(df_val),
        "test_rows":         len(df_test),
        "train_spam_ratio":  round(float(df_train["label"].mean()), 4),
        "val_spam_ratio":    round(float(df_val["label"].mean()), 4),
        "test_spam_ratio":   round(float(df_test["label"].mean()), 4),
        "tfidf_vocab_size":  len(vectorizer.vocabulary_),
        "numeric_features":  NUMERIC_FEATURES,
        "scaler_mean":       scaler.mean_.tolist(),
        "scaler_scale":      scaler.scale_.tolist(),
        "dropped_features":  DROP_FEATURES,
        "tfidf_config":      {k: str(v) for k, v in TFIDF_CONFIG.items()},
    }
    log_path = snapshot_dir / "full_load" / "_build_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Build log saved → {log_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def build(month: str):
    """
    Full load gold build tới tháng chỉ định.
    Auto-discover silver partitions ≤ month, dùng month làm holdout test.
    Output: data/gold/snapshot=YYYY-MM/ (self-contained, không ghi đè tháng khác).
    """
    holdout_month = month
    all_months = discover_silver_months(month)

    if holdout_month not in all_months:
        raise ValueError(
            f"Silver partition cho tháng {holdout_month} không tồn tại. "
            f"Chạy bronze + silver trước."
        )

    if len(all_months) < 2:
        raise ValueError(
            f"Cần ít nhất 2 tháng silver để split train/test "
            f"(hiện có {len(all_months)}). Chạy thêm bronze + silver."
        )

    # Snapshot directory cho tháng này
    snapshot_dir = GOLD_DIR / f"snapshot={holdout_month}"
    if snapshot_dir.exists():
        import shutil
        shutil.rmtree(snapshot_dir)
        print(f"[INFO] Xóa snapshot cũ {snapshot_dir.name} để rebuild.\n")

    print("=" * 60)
    print(f"GOLD BUILD — Snapshot {holdout_month}")
    print(f"Months: {all_months[0]} → {all_months[-1]} ({len(all_months)} tháng)")
    print(f"Holdout test: {holdout_month}")
    print("=" * 60)

    # ── Pipeline ──────────────────────────────────────────────────────────────
    print("\n[1/6] Loading Silver partitions...")
    df = load_silver(all_months)

    print("[2/6] Applying feature selection...")
    df = apply_feature_selection(df)

    print("[3/6] Splitting train / val / test...")
    df_train, df_val, df_test = split_dataset(df, holdout_month)

    print("[4/6] Building TF-IDF + scaler...")
    vectorizer, scaler, X_train, X_val, X_test = build_features(
        df_train, df_val, df_test
    )

    print("[5/6] Saving artifacts...")
    save_artifacts(vectorizer, scaler, all_months, holdout_month, snapshot_dir)

    print("[6/6] Writing Gold partitions...")
    write_gold_split(df_train, X_train, "train", snapshot_dir)
    write_gold_split(df_val,   X_val,   "val",   snapshot_dir)
    write_gold_split(df_test,  X_test,  "test",  snapshot_dir)
    write_gold_build_log(
        df_train, df_val, df_test, vectorizer, scaler,
        all_months, holdout_month, snapshot_dir
    )

    print(f"\n[DONE] Gold snapshot={holdout_month} build complete.")
    print(f"  → {snapshot_dir}/")
    print(f"     full_load/train.parquet + train_X.npz")
    print(f"     full_load/val.parquet   + val_X.npz")
    print(f"     full_load/test.parquet  + test_X.npz")
    print(f"     full_load/_build_log.json")
    print(f"     artifacts/tfidf_vectorizer.pkl")
    print(f"     artifacts/numeric_scaler.pkl")
    print(f"     artifacts/tfidf_metadata.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gold full load — build train/val/test từ Silver partitions"
    )
    parser.add_argument(
        "--month", required=True,
        help="Tháng holdout/test (YYYY-MM). Gold sẽ load tất cả silver ≤ tháng này."
    )
    args = parser.parse_args()
    build(args.month)