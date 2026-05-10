# Pipeline Runbook — Bronze / Silver / Gold

> Hướng dẫn chạy data pipeline từ raw CSV → gold features sẵn sàng cho training.
>
> **Pipeline**: `Raw CSV → split → Bronze → Silver → Gold (full_load)`
> **Idempotent**: chạy lại không phá data cũ — mỗi script tự skip nếu output đã tồn tại.
> **Quick run**: scroll xuống §6 cho script chạy hết 1 lệnh.

---

## 1. Prerequisites (1 lần duy nhất)

### 1.1 Python 3.11 + venv

```bash
cd email-spam-classification
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
```

> **KHÔNG dùng Python 3.12+** — sẽ không match Dockerfile / CI matrix.

### 1.2 NLTK assets (cho `text_preprocessing`)

```bash
python -m nltk.downloader stopwords wordnet punkt_tab averaged_perceptron_tagger_eng omw-1.4
```

### 1.3 Đặt source CSV vào đúng chỗ

```bash
mkdir -p data/raw
# Copy / move file CSV gốc vào:
cp /path/to/spam_Emails_data.csv data/raw/
```

File phải có path: `data/raw/spam_Emails_data.csv` (hardcoded trong [`src/data/split_raw.py:11`](../src/data/split_raw.py#L11)).

Schema CSV: 2 cột `text + label` (hoặc `message + category` — sẽ tự rename).

---

## 2. Step 1 — Split raw CSV thành monthly partitions

**Output**: `data/raw/by_month/emails_YYYY-MM.csv` (11 files, mỗi tháng ~10k rows)

```bash
python -m src.data.split_raw
```

**Kỳ vọng output**:

```
Loaded 100,000 rows

  ✓ emails_2025-05.csv  (10,000 rows)
  ✓ emails_2025-06.csv  (10,000 rows)
  ...
  ✓ emails_2026-03.csv  (10,000 rows)

Done — 11 files | 100,000 rows written to data/raw/by_month
```

> Phần dư < 50% `ROWS_PER_MONTH` (5000) sẽ tự gộp vào tháng cuối. Tham số ở [`src/data/split_raw.py:13-15`](../src/data/split_raw.py#L13).

**Verify**:

```bash
ls data/raw/by_month/
# emails_2025-05.csv  emails_2025-09.csv  emails_2026-01.csv
# emails_2025-06.csv  emails_2025-10.csv  emails_2026-02.csv
# emails_2025-07.csv  emails_2025-11.csv  emails_2026-03.csv
# emails_2025-08.csv  emails_2025-12.csv
```

---

## 3. Step 2 — Bronze ingest (per-month)

**Output**: `data/bronze/month_partition=YYYY-MM/data.parquet` (raw + ingestion metadata)

Bronze chỉ ingest 1 tháng/lần. Chạy loop cho tất cả 11 tháng:

```bash
for month in 2025-05 2025-06 2025-07 2025-08 2025-09 2025-10 \
             2025-11 2025-12 2026-01 2026-02 2026-03; do
    python -m src.etl.bronze_ingest --month "$month"
done
```

**Kỳ vọng output mỗi tháng**:

```
Ingesting 2025-05 from data/raw/by_month/emails_2025-05.csv
  Rows: 10,000
  ✓ Wrote data/bronze/month_partition=2025-05/data.parquet
  ✓ Logged to data/bronze/_ingestion_log.csv
```

**Idempotent**: nếu đã ingest tháng đó, sẽ skip:

```
[SKIP] 2025-05 đã ingest. Xoá data/bronze/month_partition=2025-05/ để re-ingest.
```

**Verify**:

```bash
ls data/bronze/
# _ingestion_log.csv          month_partition=2025-09/    month_partition=2026-01/
# month_partition=2025-05/    month_partition=2025-10/    month_partition=2026-02/
# month_partition=2025-06/    month_partition=2025-11/    month_partition=2026-03/
# month_partition=2025-07/    month_partition=2025-12/
# month_partition=2025-08/

cat data/bronze/_ingestion_log.csv
# month,rows,ingested_at,source_file
# 2025-05,10000,2026-05-10T...,emails_2025-05.csv
# ...
```

---

## 4. Step 3 — Silver transform (per-month)

**Output**: `data/silver/month_partition=YYYY-MM/{data_silver.parquet, _quality.json}`

Silver clean text + tính 9 numeric features + quality check. Per-month, idempotent.

```bash
for month in 2025-05 2025-06 2025-07 2025-08 2025-09 2025-10 \
             2025-11 2025-12 2026-01 2026-02 2026-03; do
    python -m src.etl.silver_transform --month "$month"
done
```

**Kỳ vọng output mỗi tháng**:

```
Processing 2025-05
  Step 1: text cleaning...
  Step 2: text features (char_count, log_chars, exclaim_count, ...)
  Step 3: quality check
    rows_in:  10000
    rows_out: 9824
    dropped: {null_text: 12, duplicate_text: 38, invalid_label: 7, empty_processed_text: 119}
  ✓ Wrote data/silver/month_partition=2025-05/data_silver.parquet
  ✓ Quality report → data/silver/month_partition=2025-05/_quality.json
```

**Verify quality reports**:

```bash
# Tổng rows in/out qua các tháng
for f in data/silver/month_partition=*/{_quality,_quality_report}.json 2>/dev/null; do
    python -c "import json; d=json.load(open('$f')); print(f\"{d['month']}: {d['rows_in']} → {d['rows_out']}\")"
done
```

---

## 5. Step 4 — Gold full_load (one-shot, all months at once)

**Output**:
- `data/gold/full_load/{train,val,test}.parquet` (metadata + numeric features raw)
- `data/gold/full_load/{train,val,test}_X.npz` (TF-IDF sparse + numeric scaled, hstacked)
- `data/gold/full_load/_build_log.json`
- `data/gold/artifacts/{tfidf_vectorizer,numeric_scaler}.pkl` + `tfidf_metadata.json`

Gold gộp 11 tháng silver → fit TF-IDF + StandardScaler 1 lần → split train/val/test:

```bash
python -m src.etl.gold_build
```

**Kỳ vọng output**:

```
============================================================
GOLD BUILD — Initial Full Load
Months: 2025-05 → 2026-03
Holdout test: 2026-03
============================================================

[1/6] Loading Silver partitions...
  Loaded 2025-05: 9,824 rows
  Loaded 2025-06: 9,810 rows
  ...
  Total: 107,500 rows | 49.8% spam

[2/6] Applying feature selection...
  Dropped 9 features: ['repetition_ratio', 'info_density', ...]
  Numeric features kept: ['log_chars', 'avg_word_length', 'unique_word_ratio', 'exclaim_count']
  Remaining cols: 7

[3/6] Splitting train / val / test...
  Train:    83,096 rows | spam=49.8%
  Val:      14,664 rows | spam=49.8%
  Test:      9,740 rows | spam=49.7% (holdout month=2026-03)

[4/6] Building TF-IDF + scaler...
  Fitting TF-IDF on train set...
  Vocabulary size: 30,000 tokens
  X_train_text shape: (83096, 30000)
  Fitting StandardScaler on numeric features...
  Final X_train shape: (83096, 30004)
  Final X_val shape:   (14664, 30004)
  Final X_test shape:  (9740, 30004)

[5/6] Saving artifacts...
  Vectorizer saved → data/gold/artifacts/tfidf_vectorizer.pkl
  Scaler saved     → data/gold/artifacts/numeric_scaler.pkl
  Metadata saved   → data/gold/artifacts/tfidf_metadata.json

[6/6] Writing Gold partitions...
  Gold train saved → data/gold/full_load/train.parquet + train_X.npz
  Gold val saved   → data/gold/full_load/val.parquet   + val_X.npz
  Gold test saved  → data/gold/full_load/test.parquet  + test_X.npz
  Build log saved  → data/gold/full_load/_build_log.json

[DONE] Gold full_load build complete.
```

**Idempotent**: rerun sẽ skip:

```
[SKIP] Gold full_load đã tồn tại. Xóa thư mục để build lại.
```

**Force rebuild**: `rm -rf data/gold/full_load/ data/gold/artifacts/ && python -m src.etl.gold_build`

**Verify artifacts**:

```bash
# Build log
cat data/gold/full_load/_build_log.json | python -m json.tool | head -20

# Artifacts size
ls -lh data/gold/artifacts/
# tfidf_vectorizer.pkl    ~5-10 MB
# numeric_scaler.pkl      ~1 KB
# tfidf_metadata.json     ~few KB

# Sparse matrix size (compressed)
ls -lh data/gold/full_load/*.npz
# train_X.npz   ~30-50 MB
# val_X.npz     ~5-10 MB
# test_X.npz    ~5-10 MB
```

---

## 6. Quick run — chạy hết 1 lệnh

Save vào `scripts/run_data_pipeline.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

MONTHS=(2025-05 2025-06 2025-07 2025-08 2025-09 2025-10
        2025-11 2025-12 2026-01 2026-02 2026-03)

echo "=== Step 1: Split raw CSV ==="
python -m src.data.split_raw

echo
echo "=== Step 2: Bronze ingest (11 months) ==="
for m in "${MONTHS[@]}"; do
    python -m src.etl.bronze_ingest --month "$m"
done

echo
echo "=== Step 3: Silver transform (11 months) ==="
for m in "${MONTHS[@]}"; do
    python -m src.etl.silver_transform --month "$m"
done

echo
echo "=== Step 4: Gold full_load ==="
python -m src.etl.gold_build

echo
echo "=== DONE ==="
```

Chạy:

```bash
chmod +x scripts/run_data_pipeline.sh
./scripts/run_data_pipeline.sh
```

---

## 7. Layer overview (cheat sheet)

| Layer | Granularity | Path | Idempotent | TF-IDF? |
|---|---|---|---|---|
| Raw | per-month CSV | `data/raw/by_month/emails_YYYY-MM.csv` | ✅ split lại không phá | ❌ |
| Bronze | per-month parquet | `data/bronze/month_partition=YYYY-MM/data.parquet` | ✅ skip nếu đã ingest | ❌ |
| Silver | per-month parquet | `data/silver/month_partition=YYYY-MM/data_silver.parquet` | ✅ skip nếu đã transform | ❌ |
| **Gold full_load** | **one-shot, all months** | `data/gold/full_load/*` + `data/gold/artifacts/*` | ✅ skip nếu `train.parquet` tồn tại | ✅ **fit + save pkl** |

---

## 8. Troubleshooting

| Triệu chứng | Nguyên nhân | Fix |
|---|---|---|
| `FileNotFoundError: data/raw/spam_Emails_data.csv` | Chưa đặt source CSV | Copy file vào `data/raw/spam_Emails_data.csv` |
| `ValueError: Unknown columns: [...]` | CSV schema lạ (không phải `text+label` hoặc `message+category`) | Rename columns trước, hoặc sửa [`split_raw.py:17-25`](../src/data/split_raw.py#L17-L25) |
| Bronze raise `FileNotFoundError: emails_YYYY-MM.csv` | Step 1 chưa chạy hoặc tháng nằm ngoài range | Chạy `split_raw` trước, check tháng có trong list MONTHS |
| Silver raise `[SKIP] 2025-05 đã transform` | Idempotent guard | Đúng behavior. Force = `rm -rf data/silver/month_partition=YYYY-MM/` |
| Gold raise `ValueError: Silver có NaN trong numeric features` | Silver không clean hết NaN | Check `data/silver/month_partition=*/_quality.json`, có thể silver bug. KHÔNG impute âm thầm ở gold. |
| Gold raise `AssertionError: email_id leak giữa train ↔ val` | Bug ở `split_dataset` | Check logic split — train/val/test không được trùng `email_id` |
| `nltk LookupError` | Chưa download NLTK assets | Chạy lại §1.2 |
| `TfidfVectorizer.fit_transform` chậm > 5 phút | Data quá lớn / vocab khổng lồ | Giảm `max_features` hoặc `min_df` cao hơn trong [`gold_build.py:86-92`](../src/etl/gold_build.py#L86-L92) |

---

## 9. Tham chiếu

- Code review gold: [Gold_Build_Review.md](Gold_Build_Review.md)
- Schema silver pinned: [`src/etl/silver_transform.py:187-218`](../src/etl/silver_transform.py#L187-L218)
- Gold build implementation: [`src/etl/gold_build.py`](../src/etl/gold_build.py)
- Sprint backlog liên quan: ticket `ACW3-75` (raw_partition / split), `ACW3-76` (silver), `ACW3-77` (gold) — cross-ref `email-spam-classification_old/docs/Sprint_2_Tickets.md`

---

## 10. Next steps (sau khi data pipeline xong)

1. **`monthly_feature.py`** — transform-only mỗi tháng mới (xem [Gold_Build_Review §4](Gold_Build_Review.md))
2. **`initial_load.py`** — train champion v1 trên gold/full_load → register lên MLflow
3. **Airflow DAG `initial_full_load`** — wrap §6 script này vào DAG (BashOperator), chạy local trước rồi mới EC2
