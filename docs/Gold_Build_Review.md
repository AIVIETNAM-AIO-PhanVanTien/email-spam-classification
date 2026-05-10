# Gold Build — Code Review

> **Reviewing**: [`src/etl/gold_build.py`](../src/etl/gold_build.py)
> **Date**: 10/May/2026
> **Reviewer**: Claude (peer review trước khi merge `main`)
> **Verdict**: ✅ 70-80% OK — design level tốt, **1 issue critical** cần fix trước khi train, còn lại là cleanups.
>
> Liên quan ticket Sprint 2: `ACW3-77` (`gold_build.py`), `ACW3-78` (quality dashboard).
> Cross-ref: `PROJECT_CONTEXT.md §4.4`, `Sprint_2_Tickets.md §5.3` (sẽ được copy sang `docs/` cùng các doc khác).

---

## 1. Verdict at a glance

| Khía cạnh | Đánh giá |
|---|---|
| **Design level** | ✅ Solid — EDA-driven feature selection, hybrid TF-IDF + numeric, 3-way split, time-based holdout |
| **Storage strategy** | ✅ Đúng — sparse `.npz` cho TF-IDF, parquet cho metadata + numeric (tránh dense parquet anti-pattern) |
| **Train/serve consistency** | ✅ Vectorizer fit-once → save pkl → monthly chỉ transform |
| **Idempotency + audit log** | ✅ Có guard + build_log JSON |
| **Critical correctness** | 🔴 1 issue — numeric features không scale trước khi hstack với TF-IDF |
| **Cleanups (comment/dead code/deprecated API)** | 🟡 6 issues nhỏ |
| **Sẵn sàng cho monthly pipeline** | 🟢 Cấu trúc support, nhưng cần thêm scaler artifact + schema contract |

---

## 2. ✅ Phần làm tốt

| # | Điểm | Vì sao tốt |
|---|---|---|
| 1 | Bỏ 9 features dựa trên correlation/MI/distribution, log rõ lý do | EDA-driven, có audit trail; không phải "thử nghiệm linh tinh" |
| 2 | Hybrid: TF-IDF + 4 numeric features (`log_chars`, `avg_word_length`, `unique_word_ratio`, `exclaim_count`) | Bổ sung tín hiệu metadata cho text |
| 3 | `HOLDOUT_MONTH = "2026-03"` — split theo thời gian | Realistic, tránh leakage tương lai vào train |
| 4 | 3-way split (train/val/test) | Val cho threshold tuning, test chỉ dùng 1 lần cuối |
| 5 | TF-IDF `fit_transform` chỉ trên train, val/test transform | Không leak |
| 6 | Sparse `.npz` cho TF-IDF, parquet cho metadata | Đúng strategy; nếu dense parquet 30k cột × 50k rows ≈ vài GB toàn 0 |
| 7 | Save vectorizer pkl + metadata JSON | Monthly pipeline transform-only, vocab khoá lại — fair champion/challenger |
| 8 | Idempotent guard `if (...).exists(): return` | Tránh rebuild ngoài ý muốn |
| 9 | Build log JSON: months, rows, spam ratios, vocab size, configs | Audit cuối |
| 10 | Comment tiếng Việt giải thích quyết định | Tốt cho team review |

---

## 3. ⚠️ Issues cần fix

### 🔴 Critical (block training)

#### 3.1 Numeric features chưa scale trước khi hstack với TF-IDF

**Problem**: TF-IDF với `sublinear_tf=True` cho values ~ [0, 1]. Numeric features có scale rất khác:
- `log_chars` ~ [0, 15]
- `exclaim_count` ~ [0, 50+]
- `avg_word_length` ~ [3, 10]
- `unique_word_ratio` ~ [0, 1]

Khi `hstack` rồi đưa vào Logistic Regression / Linear SVM, `exclaim_count=20` sẽ dominate hàng nghìn TF-IDF tokens với weight ~ 0.05. Model bị bias về numeric features.

**Location**: [`gold_build.py:191-199`](../src/etl/gold_build.py#L191-L199)

**Fix**:

```python
from sklearn.preprocessing import StandardScaler

def build_tfidf(df_train, df_val, df_test):
    # ... TF-IDF như cũ ...

    # Scale numeric features
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(df_train[NUMERIC_FEATURES].fillna(0))
    X_val_num   = scaler.transform(df_val[NUMERIC_FEATURES].fillna(0))
    X_test_num  = scaler.transform(df_test[NUMERIC_FEATURES].fillna(0))

    X_train = hstack([X_train_text, csr_matrix(X_train_num)])
    X_val   = hstack([X_val_text,   csr_matrix(X_val_num)])
    X_test  = hstack([X_test_text,  csr_matrix(X_test_num)])

    return vectorizer, scaler, X_train, X_val, X_test


def save_artifacts(vectorizer, scaler):
    joblib.dump(vectorizer, ARTIFACT_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(scaler,     ARTIFACT_DIR / "numeric_scaler.pkl")
    # metadata json gộp cả 2
```

> **Alternative**: dùng `sklearn.compose.ColumnTransformer` để wrap cả `TfidfVectorizer` + `StandardScaler` thành 1 pipeline. Cleaner, save 1 pkl thay vì 2. Nhưng đổi nhiều — ưu tiên fix nhanh trước.

---

### 🟡 Medium

#### 3.2 Comment sai trong `split_dataset` (line 139)

```
- Holdout month = 2025-10 (tháng cuối initial load)
```
Code thực ra dùng `HOLDOUT_MONTH = "2026-03"`. Sửa comment cho khớp constant.

**Fix**: đổi `2025-10` → `2026-03` trong docstring.

#### 3.3 Comment trong `write_gold_split` mâu thuẫn code (line 242-247)

```python
"""
Lưu dạng dense vì sparse Parquet cần thư viện đặc biệt.
...
Note: TF-IDF 30k features × 50k rows = ~6GB nếu dense.
→ Chỉ lưu numeric + metadata, X sparse lưu riêng bằng scipy.
"""
```

Câu mở nói "lưu dense" nhưng kết luận đúng là "sparse riêng". Sửa câu mở.

**Fix**: thay câu đầu bằng *"Lưu metadata + numeric ở Parquet, TF-IDF sparse riêng ở `.npz` (scipy `save_npz`). Tránh ghi dense 30k cột vào parquet — sẽ phồng lên ~6GB toàn 0."*

#### 3.4 `fillna(0)` silent — mask bug silver layer

**Location**: [`gold_build.py:191-193`](../src/etl/gold_build.py#L191-L193)

Nếu silver có null trong 4 numeric features, gold âm thầm impute 0 → mất tín hiệu, lỗi silver bị ẩn.

**Fix** (fail-fast):

```python
nulls = df[NUMERIC_FEATURES].isnull().sum()
if nulls.any():
    raise ValueError(
        f"Silver có NaN trong numeric features (silver phải clean trước): "
        f"{nulls[nulls > 0].to_dict()}"
    )
```

Đặt trong `apply_feature_selection()` ngay sau `df.drop(...)` để fail sớm.

#### 3.5 `datetime.utcnow()` deprecated từ Python 3.12+

**Location**: [`gold_build.py:222, 272`](../src/etl/gold_build.py#L222)

**Fix**:

```python
from datetime import datetime, UTC
# ...
"fitted_at": datetime.now(UTC).isoformat(),
```

---

### 🟢 Low (cleanups)

#### 3.6 Dead code

- [Line 36](../src/etl/gold_build.py#L36): `import argparse` không dùng. Xoá.
- [Line 264](../src/etl/gold_build.py#L264): `np.save(..._y.npy)` — nhưng label đã có trong parquet (`meta_cols` chứa `"label"`). Duplicate. Chọn 1 chỗ:
  - **Option A**: bỏ `_y.npy`, đọc từ `parquet["label"]` khi train.
  - **Option B**: bỏ `label` khỏi parquet, giữ `_y.npy` cho sklearn-friendly.
  - **Recommend**: option A (parquet self-contained, không lệ thuộc thứ tự file).

#### 3.7 Thiếu assert email_id không leak giữa splits

Cuối `split_dataset`:

```python
train_ids = set(df_train["email_id"])
val_ids   = set(df_val["email_id"])
test_ids  = set(df_test["email_id"])
assert not train_ids & val_ids,  "email_id leak train ↔ val"
assert not train_ids & test_ids, "email_id leak train ↔ test"
assert not val_ids & test_ids,   "email_id leak val ↔ test"
```

Test set là tháng riêng (2026-03) nên không leak với train, nhưng nếu sau này `split_dataset` đổi (vd random split nhầm) thì assert sẽ catch.

#### 3.8 `from scipy.sparse import csr_matrix` import chui giữa hàm

**Location**: [`gold_build.py:196`](../src/etl/gold_build.py#L196)

Move lên top-level imports cùng `hstack`, `save_npz`.

---

## 4. 💡 Plan cho `monthly_feature.py` (sắp build)

Khi build pipeline tháng, lưu ý 4 điểm để khớp với gold full_load hiện tại:

### 4.1 Output structure

```
data/gold/
├── full_load/                              ← initial (đã có)
│   ├── train.parquet + train_X.npz + train_y.npy
│   ├── val.parquet   + val_X.npz   + val_y.npy
│   ├── test.parquet  + test_X.npz  + test_y.npy
│   └── _build_log.json
├── monthly/                                ← thêm mới
│   ├── 2026-04/
│   │   ├── data.parquet                    ← metadata + numeric (KHÔNG split, là 1 batch)
│   │   ├── X.npz                           ← TF-IDF transform-only
│   │   ├── y.npy                           ← (optional, chọn 1 với data.parquet)
│   │   └── _transform_log.json             ← OOV ratio, n_rows, transform_at
│   ├── 2026-05/
│   └── ...
└── artifacts/
    ├── tfidf_vectorizer.pkl                ← share, fitted lần initial
    ├── numeric_scaler.pkl                  ← thêm khi fix 🔴 3.1
    └── tfidf_metadata.json
```

### 4.2 Schema contract silver ↔ gold

Tạo `src/etl/_schema.py`:

```python
# src/etl/_schema.py
REQUIRED_SILVER_COLS = {
    "email_id":          "string",
    "label":             "int8",
    "body_clean":        "string",
    "log_chars":         "float32",
    "avg_word_length":   "float32",
    "unique_word_ratio": "float32",
    "exclaim_count":     "int16",
}

def assert_silver_schema(df):
    missing = set(REQUIRED_SILVER_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Silver thiếu cols: {missing}")
```

Cả `gold_build.py` (initial) và `monthly_feature.py` đều call `assert_silver_schema(df)` ngay sau load — fail-fast khi silver thay đổi schema.

### 4.3 Quyết định: test set cố định hay rolling?

| Strategy | Pros | Cons |
|---|---|---|
| **Test fixed = 2026-03 mãi mãi** | Champion/Challenger so trên cùng baseline → fair | Test stale, không phản ánh data hiện tại |
| **Test rolling = month mới nhất** | Test phản ánh tương lai gần | C/C bias do test khác nhau |

**Recommend**: **giữ test cố định ở 2026-03** cho fair C/C comparison. Thêm 1 metric phụ "F1 on latest month" trong monthly_run.py để monitor concept drift (không dùng để promote).

> Cập nhật vào `PROJECT_CONTEXT.md §4.4` khi quyết định.

### 4.4 Vocab OOV monitoring

Khi monthly có token mới chưa trong vectorizer fitted → `vectorizer.transform()` tự ignore (TF-IDF default — OK). Nhưng nếu OOV ratio cao quá ngưỡng, là signal phải retrain full vectorizer.

```python
# Trong monthly_feature.py
def compute_oov_ratio(vectorizer, texts):
    """% token trong texts không thuộc vocabulary đã fit."""
    vocab = set(vectorizer.vocabulary_.keys())
    total, oov = 0, 0
    for text in texts:
        tokens = vectorizer.build_analyzer()(text)
        total += len(tokens)
        oov   += sum(1 for t in tokens if t not in vocab)
    return oov / max(total, 1)


# Log + alert
oov = compute_oov_ratio(vectorizer, df_month["body_clean"])
if oov > 0.20:
    print(f"⚠️  OOV ratio {oov:.1%} > 20% — cân nhắc retrain full vectorizer")
log["oov_ratio"] = round(oov, 4)
```

---

## 5. Action items checklist

> Tick khi xong; mỗi action map sang ticket Sprint 2 hoặc PR thẳng.

### Trước khi train model (block by 🔴)

- [ ] **3.1** — Thêm `StandardScaler` cho numeric features, save `numeric_scaler.pkl` (`ACW3-77`)
- [ ] Update `tfidf_metadata.json` để chứa cả numeric scaler info
- [ ] Test E2E: load gold → train LR → đảm bảo coef của numeric features không dominate

### Cleanups (làm trước khi merge main)

- [ ] **3.2** — Sửa comment `Holdout month` thành `2026-03`
- [ ] **3.3** — Sửa comment `write_gold_split` để khớp logic sparse-on-side
- [ ] **3.4** — `fillna(0)` → raise `ValueError` nếu có NaN
- [ ] **3.5** — `datetime.utcnow()` → `datetime.now(UTC)`
- [ ] **3.6** — Xoá `import argparse` + decide A/B cho `_y.npy`
- [ ] **3.7** — Thêm 3 assert email_id no-leak giữa splits
- [ ] **3.8** — Move `csr_matrix` import lên top-level

### Cho monthly_feature.py (next sprint task)

- [ ] **4.1** — Tạo skeleton `src/etl/monthly_feature.py` theo cấu trúc đề xuất
- [ ] **4.2** — Tạo `src/etl/_schema.py` + dùng ở cả gold_build và monthly_feature
- [ ] **4.3** — Quyết định test strategy + cập nhật PROJECT_CONTEXT §4.4
- [ ] **4.4** — Implement `compute_oov_ratio()` + log vào `_transform_log.json`

### Documentation

- [ ] Update `PROJECT_CONTEXT.md §4.4 Gold layer` (sau khi copy sang) với:
  - Hybrid TF-IDF + 4 numeric features
  - Storage strategy (sparse npz + parquet metadata)
  - Test set strategy quyết định ở 4.3
- [ ] Add row vào `INDEX.md` cho file review này (sau khi copy INDEX.md sang)

---

## 6. References

- Source under review: [`src/etl/gold_build.py`](../src/etl/gold_build.py)
- Silver schema: [`src/etl/silver_transform.py`](../src/etl/silver_transform.py) (lines 187-218)
- Sprint backlog: `Sprint_2_Tickets.md `ACW3-77`, `ACW3-78`` (sẽ copy sang)
- QA test plan: `QA_Test_Plan_Sprint2.md §3.2 DATA-09/10` — sẽ thêm cases cho gold-specific assertions (sẽ copy sang)
