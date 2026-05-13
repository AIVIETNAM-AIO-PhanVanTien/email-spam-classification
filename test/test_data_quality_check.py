import pandas as pd

from src.utils.data_quality_check import TextDataQuality


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


def test_summary_df_flattens_non_nested_metrics():
    df = pd.DataFrame({"body_clean": ["alpha", "beta"], "label": [0, 1]})

    summary = TextDataQuality(df, "body_clean").summary_df()

    assert {"category", "metric", "value"} <= set(summary.columns)
    assert "top_duplicates" not in set(summary["metric"])
