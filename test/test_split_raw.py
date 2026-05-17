import pandas as pd
import pytest

from src.utils.split_raw import ROWS_PER_MONTH, build_chunks, month_label, normalize_columns


# Tiêu chí: Dữ liệu raw từ các schema phổ biến được chuẩn hóa về cột body và raw_label.
def test_normalize_columns_supports_known_raw_schemas():
    text_label_df = pd.DataFrame({" text ": ["body"], " Label ": ["spam"]})
    message_category_df = pd.DataFrame({"message": ["body"], "category": ["ham"]})

    assert normalize_columns(text_label_df).columns.tolist() == ["body", "raw_label"]
    assert normalize_columns(message_category_df).columns.tolist() == ["body", "raw_label"]


# Tiêu chí: Schema raw không hợp lệ bị chặn sớm để tránh đưa dữ liệu sai vào pipeline.
def test_normalize_columns_rejects_unknown_schema():
    df = pd.DataFrame({"subject": ["x"], "kind": ["spam"]})

    with pytest.raises(ValueError, match="Unknown columns"):
        normalize_columns(df)


# Tiêu chí: Nhãn tháng dữ liệu tự động chuyển năm chính xác khi vượt qua tháng 12.
def test_month_label_rolls_over_year_boundary():
    assert month_label(0) == "2024-11"
    assert month_label(2) == "2025-01"


# Tiêu chí: Phần dữ liệu cuối quá nhỏ được gộp hợp lý để không tạo partition thiếu ổn định.
def test_build_chunks_merges_small_final_remainder():
    rows = ROWS_PER_MONTH + ROWS_PER_MONTH // 2 - 1
    df = pd.DataFrame({"body": range(rows), "raw_label": ["ham"] * rows})

    chunks = build_chunks(df)

    assert len(chunks) == 1
    assert chunks[0][0] == "2024-11"
    assert len(chunks[0][1]) == rows


# Tiêu chí: Phần dữ liệu cuối bằng đúng 50 phần trăm kích thước tháng được giữ thành partition riêng.
def test_build_chunks_keeps_half_sized_final_chunk():
    rows = ROWS_PER_MONTH + ROWS_PER_MONTH // 2
    df = pd.DataFrame({"body": range(rows), "raw_label": ["ham"] * rows})

    chunks = build_chunks(df)

    assert len(chunks) == 2
    assert chunks[0][0] == "2024-11"
    assert chunks[1][0] == "2024-12"
    assert len(chunks[1][1]) == ROWS_PER_MONTH // 2
