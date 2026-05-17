import pandas as pd

from src.utils.data_quality_check import TextDataQuality


# Tiêu chí: Báo cáo chất lượng dữ liệu bao phủ đầy đủ completeness, uniqueness, pattern và label distribution.
def test_run_all_checks_returns_core_quality_sections():
    df = pd.DataFrame(
        {
            "body_clean": ["hello world", "hello world", "", None, "BUY 123!!!"],
            "label": [0, 0, 1, 1, 1],
        }
    )

    report = TextDataQuality(df, "body_clean").run_all_checks()

    assert report["completeness"]["total_rows"] == 5
    assert report["completeness"]["null_count"] == 1
    assert report["uniqueness"]["duplicate_rate"] == 0.2
    assert report["pattern"]["special_char_rate"] == 0.2
    assert report["label_distribution"]["spam_count"] == 3


# Tiêu chí: Bảng tổng hợp chất lượng chỉ giữ các chỉ số phẳng, phù hợp để đưa vào báo cáo.
def test_summary_df_flattens_non_nested_metrics():
    df = pd.DataFrame({"body_clean": ["alpha", "beta"], "label": [0, 1]})

    summary = TextDataQuality(df, "body_clean").summary_df()

    assert {"category", "metric", "value"} <= set(summary.columns)
    assert "top_duplicates" not in set(summary["metric"])


# Tiêu chí: Chỉ số completeness nhận diện đúng chuỗi rỗng, khoảng trắng và placeholder.
def test_completeness_flags_empty_whitespace_and_placeholders():
    df = pd.DataFrame({"body_clean": ["", "   ", "unknown", None]})

    report = TextDataQuality(df, "body_clean").check_completeness()

    assert report["total_rows"] == 4
    assert report["null_count"] == 1
    assert report["empty_string_rate"] == 0.25
    assert report["whitespace_rate"] == 0.5
    assert report["placeholder_rate"] == 0.5


# Tiêu chí: Phân phối nhãn được gắn drift_flag khi tỷ lệ spam quá lệch.
def test_label_distribution_flags_extreme_spam_ratio():
    df = pd.DataFrame({"body_clean": ["a"] * 10, "label": [1] * 10})

    report = TextDataQuality(df, "body_clean").check_label_distribution()

    assert report["spam_ratio"] == 1.0
    assert report["spam_count"] == 10
    assert report["ham_count"] == 0
    assert report["drift_flag"] is True
