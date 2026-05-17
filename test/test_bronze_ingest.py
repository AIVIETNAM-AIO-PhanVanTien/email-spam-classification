from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.etl import bronze_ingest


# Tiêu chí: Thời gian nhận email được sinh ổn định và luôn nằm đúng tháng dữ liệu.
def test_inject_received_at_is_deterministic_and_in_month():
    df = pd.DataFrame({"body": ["a", "b"], "raw_label": ["ham", "spam"]})

    first = bronze_ingest.inject_received_at(df.copy(), "2025-02")
    second = bronze_ingest.inject_received_at(df.copy(), "2025-02")

    assert first["received_at"].tolist() == second["received_at"].tolist()
    assert first["received_at"].dt.strftime("%Y-%m").eq("2025-02").all()
    assert first["month_partition"].eq("2025-02").all()


# Tiêu chí: Bước Bronze tạo đúng partition, mã email, nhãn chuẩn hóa và log ingest.
def test_ingest_writes_bronze_partition_and_log(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw" / "by_month"
    bronze_dir = tmp_path / "bronze"
    raw_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "body": ["hello", "win cash"],
            "raw_label": ["ham", "spam"],
        }
    ).to_csv(raw_dir / "emails_2025-02.csv", index=False)
    monkeypatch.setattr(bronze_ingest, "RAW_DIR", raw_dir)
    monkeypatch.setattr(bronze_ingest, "BRONZE_DIR", bronze_dir)

    bronze_ingest.ingest("2025-02")

    out_path = bronze_dir / "month_partition=2025-02" / "data.parquet"
    assert out_path.exists()
    written = pq.read_table(out_path).to_pandas()
    assert written["email_id"].tolist() == ["email_202502_00000", "email_202502_00001"]
    assert written["label"].tolist() == [0, 1]
    assert (bronze_dir / "_ingestion_log.csv").exists()


# Tiêu chí: Bước ingest không ghi đè dữ liệu Bronze khi partition đã tồn tại.
def test_ingest_is_idempotent_when_partition_exists(tmp_path, monkeypatch):
    bronze_dir = tmp_path / "bronze"
    out_dir = bronze_dir / "month_partition=2025-02"
    out_dir.mkdir(parents=True)
    (out_dir / "data.parquet").write_bytes(b"already here")
    monkeypatch.setattr(bronze_ingest, "BRONZE_DIR", bronze_dir)
    monkeypatch.setattr(bronze_ingest, "RAW_DIR", Path("does-not-exist"))

    bronze_ingest.ingest("2025-02")

    assert (out_dir / "data.parquet").read_bytes() == b"already here"


# Tiêu chí: Bronze báo lỗi rõ ràng khi file raw theo tháng chưa được chuẩn bị.
def test_ingest_raises_when_raw_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(bronze_ingest, "BRONZE_DIR", tmp_path / "bronze")
    monkeypatch.setattr(bronze_ingest, "RAW_DIR", tmp_path / "raw")

    with pytest.raises(FileNotFoundError, match="Raw file"):
        bronze_ingest.ingest("2025-03")
