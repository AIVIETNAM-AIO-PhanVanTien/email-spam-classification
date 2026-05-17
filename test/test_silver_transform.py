import json

import pandas as pd
import pyarrow.parquet as pq

from src.etl import silver_transform


# Tiêu chí: Bước Silver làm sạch nội dung email và sinh đầy đủ đặc trưng văn bản quan trọng.
def test_run_text_cleaning_and_features(bronze_like_df):
    df = silver_transform.run_text_cleaning(bronze_like_df.copy())
    df = silver_transform.run_text_features(df)

    assert "body_clean" in df.columns
    assert df.loc[0, "body_clean"] == "free cash prize click here escapenumber now"
    assert df.loc[0, "spam_keyword_count"] >= 3
    assert df.loc[0, "has_escapenumber"] == 1
    assert df.loc[0, "exclaim_count"] == 3


# Tiêu chí: Kiểm tra chất lượng Silver cắt nội dung quá dài và loại bỏ bản ghi không đạt chuẩn.
def test_run_quality_check_truncates_and_drops_rows(monkeypatch):
    df = pd.DataFrame(
        {
            "body_clean": ["good text that is long enough", "short", "x" * 30],
            "label": [1, 0, None],
        }
    )
    monkeypatch.setattr(silver_transform, "MIN_CHAR_COUNT", 10)
    monkeypatch.setattr(silver_transform, "MAX_CHAR_COUNT", 20)

    cleaned, report = silver_transform.run_quality_check(df, "2025-02")

    assert len(cleaned) == 1
    assert cleaned.loc[0, "body_clean"] == "good text that is lo"
    assert report["truncated_rows"] == 2
    assert report["dropped_short_body"] == 1
    assert report["dropped_null_label"] == 1


# Tiêu chí: Silver ghi đúng partition parquet, quality report và quality log phục vụ truy vết.
def test_write_silver_partition_and_quality_report(tmp_path, monkeypatch, bronze_like_df):
    silver_dir = tmp_path / "silver"
    monkeypatch.setattr(silver_transform, "SILVER_DIR", silver_dir)
    df = silver_transform.run_text_features(
        silver_transform.run_text_cleaning(bronze_like_df.copy())
    )

    silver_transform.write_silver_partition(df, "2025-02")
    silver_transform.write_quality_report({"month": "2025-02", "quality_metrics": {}}, "2025-02")

    partition_dir = silver_dir / "month_partition=2025-02"
    assert (partition_dir / "data_silver.parquet").exists()
    assert (partition_dir / "quality_report.json").exists()
    assert (silver_dir / "_quality_log.jsonl").exists()
    assert len(pq.read_table(partition_dir / "data_silver.parquet").to_pandas()) == 2
    assert json.loads((partition_dir / "quality_report.json").read_text())["month"] == "2025-02"
