"""
Gold layer — initial full load pipeline.
Silver partitions (2025-05 → 2025-10) → feature selection → TF-IDF → train/val/test split → Gold parquet.

EDA Silver findings áp dụng:
    Feature selection — bỏ 9 features redundant/vô nghĩa:
        repetition_ratio    corr = -1.00 với unique_word_ratio
        info_density        corr >  0.92 với complexity, unique_word_ratio
        complexity          corr =  0.86 với repetition_ratio
        spam_keyword_count  corr =  0.99 với char_count, word_count
        log_words           corr =  0.99 với log_chars
        has_escapenumber    MI   =  0.003, ham > spam → noise
        question_count      distribution giống nhau 2 class
        digit_ratio         distribution giống nhau 2 class
        upper_ratio         distribution giống nhau 2 class

    Giữ lại 4 numeric features có separation tốt nhất:
        log_chars           đại diện độ dài email
        avg_word_length     có separation spam vs ham
        unique_word_ratio   đại diện vocabulary richness group
        exclaim_count       punctuation signal

    TF-IDF trên body_clean:
        max_features=30,000  ngram_range=(1,2)  sublinear_tf=True
        Fit 1 lần trên train set → save pkl → transform only cho tháng mới

    Test set = tháng 2026-03 (holdout theo thời gian — realistic hơn random split)
    Label ~50/50 → không cần class weighting

Initial full load:
    py -m src.etl.gold_build

"""
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
ARTIFACT_DIR = GOLD_DIR / "artifacts"

# ── Config ────────────────────────────────────────────────────────────────────
INITIAL_MONTHS  = [
    "2025-05", "2025-06", "2025-07",
    "2025-08", "2025-09", "2025-10",
    "2025-11", "2025-12", "2026-01",
    "2026-02", "2026-03"
]
HOLDOUT_MONTH   = "2026-03"    # test set — tách theo thời gian
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
def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Tách theo thời gian trước, sau đó random split trainval.

    Lý do tách theo thời gian thay vì random:
        - Realistic hơn: model train trên quá khứ, test trên tương lai
        - Tránh data leakage: email tháng holdout không lẫn vào train
        - Holdout month = 2026-03 (tháng cuối initial load)

    Sau đó random split trainval → train (85%) / val (15%)
    để val không bị bias theo thứ tự thời gian trong các tháng đầu.

    Cuối cùng assert email_id no-leak giữa 3 split để catch bug nếu
    sau này split logic thay đổi (vd random split nhầm).
    """
    df_test    = df[df["month_partition"] == HOLDOUT_MONTH].copy()
    df_trainval= df[df["month_partition"] != HOLDOUT_MONTH].copy()

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
          f"(holdout month={HOLDOUT_MONTH})\n")

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
def save_artifacts(vectorizer: TfidfVectorizer, scaler: StandardScaler):
    """
    Save TF-IDF vectorizer + numeric scaler để monthly pipeline dùng transform only.
    Không bao giờ fit lại trên data mới — để vector space nhất quán giữa
    champion và challenger.
    """
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    vec_path = ARTIFACT_DIR / "tfidf_vectorizer.pkl"
    joblib.dump(vectorizer, vec_path)
    print(f"  Vectorizer saved → {vec_path}")

    scaler_path = ARTIFACT_DIR / "numeric_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved     → {scaler_path}")

    # Save metadata để audit
    meta = {
        "fitted_at":        datetime.now(UTC).isoformat(),
        "train_months":     [m for m in INITIAL_MONTHS if m != HOLDOUT_MONTH],
        "vocab_size":       len(vectorizer.vocabulary_),
        "tfidf_config":     {k: str(v) for k, v in TFIDF_CONFIG.items()},
        "numeric_features": NUMERIC_FEATURES,
        "scaler_mean":      scaler.mean_.tolist(),
        "scaler_scale":     scaler.scale_.tolist(),
        "dropped_features": DROP_FEATURES,
    }
    meta_path = ARTIFACT_DIR / "tfidf_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved   → {meta_path}\n")


# ── Step 6: Write Gold parquet ────────────────────────────────────────────────
def write_gold_split(
    df:     pd.DataFrame,
    X,                          # sparse matrix
    split:  str,                # "train" | "val" | "test"
):
    """
    Lưu Gold split: metadata + numeric features ở Parquet, TF-IDF sparse
    riêng ở `.npz` (scipy `save_npz`).

    Tránh ghi dense 30k cột vào parquet — sẽ phồng lên ~6GB toàn 0
    (TF-IDF với max_features=30k × 50k rows × dense = ~6GB; sparse npz
    chỉ vài chục MB vì 99% là 0).

    Label đã có trong parquet → KHÔNG lưu `_y.npy` để tránh duplicate.
    Khi train: load label từ `parquet["label"]`.
    """
    out_dir = GOLD_DIR / "full_load"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parquet: email_id + label + month + numeric (đã scale chưa? — KHÔNG, parquet
    # giữ raw numeric cho audit; X.npz mới là numeric đã scale)
    meta_cols = ["email_id", "label", "month_partition"] + NUMERIC_FEATURES
    df_out = df[meta_cols].reset_index(drop=True)
    pq.write_table(
        pa.Table.from_pandas(df_out),
        out_dir / f"{split}.parquet",
        compression="snappy",
    )

    # Sparse matrix (TF-IDF + numeric scaled)
    save_npz(str(out_dir / f"{split}_X.npz"), X)

    print(f"  Gold {split} saved → {out_dir / split}.parquet + {split}_X.npz")


def write_gold_build_log(df_train, df_val, df_test, vectorizer, scaler):
    """Ghi log tổng kết gold build để audit."""
    log = {
        "built_at":          datetime.now(UTC).isoformat(),
        "initial_months":    INITIAL_MONTHS,
        "holdout_month":     HOLDOUT_MONTH,
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
    log_path = GOLD_DIR / "full_load" / "_build_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Build log saved → {log_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def build():
    # Idempotent guard
    if (GOLD_DIR / "full_load" / "train.parquet").exists():
        print("[SKIP] Gold full_load đã tồn tại. Xóa thư mục để build lại.")
        return

    print("=" * 60)
    print("GOLD BUILD — Initial Full Load")
    print(f"Months: {INITIAL_MONTHS[0]} → {INITIAL_MONTHS[-1]}")
    print(f"Holdout test: {HOLDOUT_MONTH}")
    print("=" * 60)

    # ── Pipeline ──────────────────────────────────────────────────────────────
    print("\n[1/6] Loading Silver partitions...")
    df = load_silver(INITIAL_MONTHS)

    print("[2/6] Applying feature selection...")
    df = apply_feature_selection(df)

    print("[3/6] Splitting train / val / test...")
    df_train, df_val, df_test = split_dataset(df)

    print("[4/6] Building TF-IDF + scaler...")
    vectorizer, scaler, X_train, X_val, X_test = build_features(df_train, df_val, df_test)

    print("[5/6] Saving artifacts...")
    save_artifacts(vectorizer, scaler)

    print("[6/6] Writing Gold partitions...")
    write_gold_split(df_train, X_train, "train")
    write_gold_split(df_val,   X_val,   "val")
    write_gold_split(df_test,  X_test,  "test")
    write_gold_build_log(df_train, df_val, df_test, vectorizer, scaler)

    print("\n[DONE] Gold full_load build complete.")
    print(f"  → data/gold/full_load/")
    print(f"     train.parquet + train_X.npz")
    print(f"     val.parquet   + val_X.npz")
    print(f"     test.parquet  + test_X.npz")
    print(f"  → data/gold/artifacts/")
    print(f"     tfidf_vectorizer.pkl")
    print(f"     numeric_scaler.pkl")
    print(f"     tfidf_metadata.json")


if __name__ == "__main__":
    build()