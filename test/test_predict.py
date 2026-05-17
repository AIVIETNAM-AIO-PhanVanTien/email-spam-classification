import pytest

from app.predict import _build_input_text, _compute_numeric_features, predict_spam


# Tiêu chí: Subject và body được ghép thành văn bản đầu vào gọn sạch, bỏ qua phần rỗng.
def test_build_input_text_joins_non_empty_parts():
    assert _build_input_text("Hello", "Team") == "Hello Team"
    assert _build_input_text("", "Only body") == "Only body"
    assert _build_input_text(None, "Only body") == "Only body"


# Tiêu chí: Đặc trưng số tại inference khớp đúng công thức đã dùng trong pipeline huấn luyện.
def test_compute_numeric_features_matches_expected_formulas():
    features = _compute_numeric_features("Hi team!!!", "hi team")

    assert features["log_chars"] == pytest.approx(2.0794)
    assert features["avg_word_length"] == 3.5
    assert features["unique_word_ratio"] == 1.0
    assert features["exclaim_count"] == 3


# Tiêu chí: Hàm dự đoán từ chối email rỗng để tránh sinh kết quả không có cơ sở dữ liệu.
def test_predict_spam_rejects_empty_subject_and_body():
    with pytest.raises(ValueError, match="must not be empty"):
        predict_spam("", "")


# Tiêu chí: Hàm dự đoán trả nhãn not_spam khi nội dung bị làm sạch thành rỗng.
def test_predict_spam_returns_not_spam_when_cleaned_text_is_empty(monkeypatch):
    monkeypatch.setattr(
        "app.predict.load_model",
        lambda: {"threshold": 0.6},
    )

    result = predict_spam("!!!", "", threshold=None)

    assert result["label"] == "not_spam"
    assert result["spam_probability"] == 0.0
    assert result["threshold"] == 0.6
    assert "no usable tokens" in result["note"]
