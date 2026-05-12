import json

import pandas as pd
import pytest

from src.etl import gold_build


def _gold_ready_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "email_id": [f"email_{i}" for i in range(8)],
            "body_clean": [
                "ham meeting schedule notes",
                "spam free cash prize",
                "ham project update notes",
                "spam urgent cash offer",
                "ham lunch agenda notes",
                "spam winner free deal",
                "ham invoice review notes",
                "spam casino cash prize",
            ],
            "label": [0, 1, 0, 1, 0, 1, 0, 1],
            "month_partition": ["2025-01"] * 4 + ["2025-02"] * 4,
            "log_chars": [3.1, 3.2, 3.0, 3.3, 3.2, 3.4, 3.1, 3.5],
            "avg_word_length": [5.0, 4.5, 5.2, 4.2, 5.1, 4.7, 5.3, 4.4],
            "unique_word_ratio": [1.0] * 8,
            "exclaim_count": [0, 2, 0, 1, 0, 3, 0, 4],
            "repetition_ratio": [0.0] * 8,
            "info_density": [0.1] * 8,
            "complexity": [5.0] * 8,
            "spam_keyword_count": [0, 3, 0, 2, 0, 3, 0, 3],
            "log_words": [1.5] * 8,
            "has_escapenumber": [0] * 8,
            "question_count": [0] * 8,
            "digit_ratio": [0.0] * 8,
            "upper_ratio": [0.0] * 8,
        }
    )


def test_apply_feature_selection_drops_configured_features():
    selected = gold_build.apply_feature_selection(_gold_ready_df())

    assert "repetition_ratio" not in selected.columns
    assert set(gold_build.NUMERIC_FEATURES) <= set(selected.columns)


def test_apply_feature_selection_fails_fast_on_numeric_nulls():
    df = _gold_ready_df()
    df.loc[0, "log_chars"] = None

    with pytest.raises(ValueError, match="NaN"):
        gold_build.apply_feature_selection(df)


def test_split_dataset_uses_holdout_month_without_id_leak(monkeypatch):
    monkeypatch.setattr(gold_build, "VAL_SIZE", 0.5)

    train, val, test = gold_build.split_dataset(_gold_ready_df(), "2025-02")

    assert set(test["month_partition"]) == {"2025-02"}
    assert set(train["email_id"]).isdisjoint(val["email_id"])
    assert set(train["email_id"]).isdisjoint(test["email_id"])
    assert set(val["email_id"]).isdisjoint(test["email_id"])


def test_build_features_fits_on_train_and_transforms_other_splits(monkeypatch):
    monkeypatch.setattr(
        gold_build,
        "TFIDF_CONFIG",
        {"max_features": 20, "ngram_range": (1, 1), "min_df": 1, "max_df": 1.0},
    )
    df = _gold_ready_df()
    train = df.iloc[:4]
    val = df.iloc[4:6]
    test = df.iloc[6:]

    vectorizer, scaler, x_train, x_val, x_test = gold_build.build_features(train, val, test)

    assert len(vectorizer.vocabulary_) > 0
    assert scaler.mean_.shape[0] == len(gold_build.NUMERIC_FEATURES)
    assert x_train.shape[0] == 4
    assert x_val.shape[0] == 2
    assert x_test.shape[0] == 2


def test_save_artifacts_writes_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(
        gold_build,
        "TFIDF_CONFIG",
        {"max_features": 20, "ngram_range": (1, 1), "min_df": 1, "max_df": 1.0},
    )
    df = _gold_ready_df()
    vectorizer, scaler, *_ = gold_build.build_features(
        df.iloc[:4],
        df.iloc[4:6],
        df.iloc[6:],
    )

    gold_build.save_artifacts(vectorizer, scaler, ["2025-01", "2025-02"], "2025-02", tmp_path)

    meta_path = tmp_path / "artifacts" / "tfidf_metadata.json"
    assert meta_path.exists()
    assert json.loads(meta_path.read_text())["holdout_month"] == "2025-02"
