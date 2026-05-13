import pandas as pd
import pytest

from src.utils.split_raw import ROWS_PER_MONTH, build_chunks, month_label, normalize_columns


def test_normalize_columns_supports_known_raw_schemas():
    text_label_df = pd.DataFrame({" text ": ["body"], " Label ": ["spam"]})
    message_category_df = pd.DataFrame({"message": ["body"], "category": ["ham"]})

    assert normalize_columns(text_label_df).columns.tolist() == ["body", "raw_label"]
    assert normalize_columns(message_category_df).columns.tolist() == ["body", "raw_label"]


def test_normalize_columns_rejects_unknown_schema():
    df = pd.DataFrame({"subject": ["x"], "kind": ["spam"]})

    with pytest.raises(ValueError, match="Unknown columns"):
        normalize_columns(df)


def test_month_label_rolls_over_year_boundary():
    assert month_label(0) == "2024-11"
    assert month_label(2) == "2025-01"


def test_build_chunks_merges_small_final_remainder():
    rows = ROWS_PER_MONTH + ROWS_PER_MONTH // 2 - 1
    df = pd.DataFrame({"body": range(rows), "raw_label": ["ham"] * rows})

    chunks = build_chunks(df)

    assert len(chunks) == 1
    assert chunks[0][0] == "2024-11"
    assert len(chunks[0][1]) == rows
