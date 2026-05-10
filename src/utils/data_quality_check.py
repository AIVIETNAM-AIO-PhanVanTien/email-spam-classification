"""
Text data quality checker
Dùng cho Silver layer: tạo quality_report.json mỗi tháng.
"""
from __future__ import annotations
from typing import Any
import pandas as pd


class TextDataQuality:
    def __init__(self, df: pd.DataFrame, text_col: str):
        self.df  = df.copy()
        self.col = text_col
        self.series = self.df[self.col].astype(str)

    # Completeness
    def check_completeness(self) -> dict[str, Any]:
        s = self.df[self.col]
        return {
            "total_rows":        int(len(s)),
            "null_count":        int(s.isna().sum()),
            "null_rate":         round(float(s.isna().mean()), 4),
            "empty_string_rate": round(float((s == "").mean()), 4),
            "whitespace_rate":   round(
                float(s.astype(str).str.strip().eq("").mean()), 4
            ),
            "placeholder_rate":  round(
                float(
                    s.astype(str)
                     .str.lower()
                     .isin(["na", "n/a", "none", "null", "unknown"])
                     .mean()
                ),
                4,
            ),
        }

    # Consistency
    def check_consistency(self) -> dict[str, Any]:
        s = self.series
        return {
            "case_inconsistency_rate":       round(float((s != s.str.lower()).mean()), 4),
            "leading_trailing_spaces_rate":  round(float((s != s.str.strip()).mean()), 4),
            "multiple_spaces_rate":          round(
                float(s.str.contains(r"\s{2,}", regex=True).mean()), 4
            ),
        }

    # Uniqueness
    def check_uniqueness(self) -> dict[str, Any]:
        s = self.series
        return {
            "n_unique":        int(s.nunique()),
            "unique_rate":     round(float(s.nunique() / len(s)), 4),
            "duplicate_rate":  round(float(s.duplicated().mean()), 4),
            "top_duplicates":  s.value_counts().head(5).to_dict(),
        }

    # Length Analysis
    def check_length(self) -> dict[str, Any]:
        length = self.series.str.len()
        return {
            "min_length":          int(length.min()),
            "max_length":          int(length.max()),
            "mean_length":         round(float(length.mean()), 2),
            "short_text_rate_lt3": round(float((length < 3).mean()), 4),
            "long_text_rate_gt100":round(float((length > 100).mean()), 4),
        }

    # Pattern Issues
    def check_pattern(self) -> dict[str, Any]:
        s = self.series
        return {
            "numeric_only_rate":  round(float(s.str.match(r"^\d+$").mean()), 4),
            "special_char_rate":  round(
                float(s.str.contains(r"[^a-zA-Z0-9\s]", regex=True).mean()), 4
            ),
            "contains_digit_rate":round(float(s.str.contains(r"\d").mean()), 4),
        }

    # Semantic
    def check_semantic(self) -> dict[str, Any]:
        s = self.series
        return {
            "single_word_rate":          round(
                float((s.str.split().str.len() == 1).mean()), 4
            ),
            "very_low_information_rate": round(float((s.str.len() < 2).mean()), 4),
        }

    # Label distribution
    def check_label_distribution(
        self, label_col: str = "label"
    ) -> dict[str, Any] | None:
        if label_col not in self.df.columns:
            return None
        spam_ratio = float(self.df[label_col].mean())
        return {
            "spam_ratio":  round(spam_ratio, 4),
            "spam_count":  int(self.df[label_col].sum()),
            "ham_count":   int((self.df[label_col] == 0).sum()),
            # Flag nếu tỉ lệ bất thường → có thể data drift
            "drift_flag":  spam_ratio < 0.10 or spam_ratio > 0.90,
        }

    # Run all
    def run_all_checks(self, label_col: str = "label") -> dict[str, Any]:
        result = {
            "completeness":  self.check_completeness(),
            "consistency":   self.check_consistency(),
            "uniqueness":    self.check_uniqueness(),
            "length":        self.check_length(),
            "pattern":       self.check_pattern(),
            "semantic":      self.check_semantic(),
        }
        label_dist = self.check_label_distribution(label_col)
        if label_dist:
            result["label_distribution"] = label_dist
        return result

    def summary_df(self) -> pd.DataFrame:
        """Trả về bảng tổng hợp dạng long — để log hoặc display."""
        rows = []
        for category, metrics in self.run_all_checks().items():
            for k, v in metrics.items():
                if not isinstance(v, dict):   # bỏ top_duplicates dict lồng
                    rows.append({"category": category, "metric": k, "value": v})
        return pd.DataFrame(rows)