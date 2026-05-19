import csv

import pytest

pytest.importorskip("fastapi")

from app import api


# Tiêu chí: API health phản hồi trạng thái degraded rõ ràng khi mô hình chưa sẵn sàng.
def test_health_returns_degraded_when_model_missing(monkeypatch):
    def missing_model():
        raise FileNotFoundError("missing model")

    monkeypatch.setattr(api, "load_model", missing_model)

    assert api.health() == {"status": "degraded", "model_loaded": False}


# Tiêu chí: API health hiển thị đầy đủ metadata khi mô hình đã được nạp thành công.
def test_health_returns_model_metadata(monkeypatch):
    monkeypatch.setattr(
        api,
        "load_model",
        lambda: {
            "model_type": "LogisticRegression",
            "winner": "lr",
            "snapshot": "2026-04",
            "trained_at": "2026-05-01T00:00:00+00:00",
            "threshold": 0.7,
            "feature_subset": "full",
        },
    )

    response = api.health()

    assert response["status"] == "ok"
    assert response["model_loaded"] is True
    assert response["threshold"] == 0.7


# Tiêu chí: Log dự đoán được ghi đúng header, định dạng xác suất và chuẩn hóa ký tự xuống dòng.
def test_append_prediction_log_writes_header_and_sanitizes_text(tmp_path, monkeypatch):
    log_path = tmp_path / "predictions.csv"
    monkeypatch.setattr(api, "PREDICTION_LOG_PATH", log_path)

    api._append_prediction_log("Hello\nthere", "Body\rtext", "spam", 0.9876543)

    with log_path.open(newline="") as f:
        rows = list(csv.reader(f))

    assert rows[0] == api._LOG_FIELDS
    assert rows[1][1] == "Hello there"
    assert rows[1][2] == "Body text"
    assert rows[1][3:] == ["spam", "0.987654"]


# Tiêu chí: Endpoint predict chặn request không có subject và body trước khi gọi mô hình.
def test_predict_endpoint_rejects_blank_payload():
    with pytest.raises(api.HTTPException) as exc:
        api.predict(api.EmailInput(subject="   ", body=""))

    assert exc.value.status_code == 400
    assert "must not be empty" in exc.value.detail


# Tiêu chí: Endpoint predict trả về response đúng schema và ghi log khi mô hình dự đoán thành công.
def test_predict_endpoint_returns_response_and_logs(monkeypatch):
    logged = []
    monkeypatch.setattr(
        api,
        "predict_spam",
        lambda subject, body, threshold=None: {
            "label": "spam",
            "spam_probability": 0.91,
            "threshold": threshold if threshold is not None else 0.5,
        },
    )
    monkeypatch.setattr(api, "_append_prediction_log", lambda *args: logged.append(args))

    response = api.predict(api.EmailInput(subject="Sale", body="Win now", threshold=0.8))

    assert response.label == "spam"
    assert response.spam_probability == 0.91
    assert response.threshold == 0.8
    assert logged == [("Sale", "Win now", "spam", 0.91)]


# Tiêu chí: Endpoint predict trả lỗi 503 rõ ràng khi artifact mô hình phục vụ chưa tồn tại.
def test_predict_endpoint_maps_missing_model_to_503(monkeypatch):
    def missing_model(*_args, **_kwargs):
        raise FileNotFoundError("model missing")

    monkeypatch.setattr(api, "predict_spam", missing_model)

    with pytest.raises(api.HTTPException) as exc:
        api.predict(api.EmailInput(subject="Sale", body="Win now"))

    assert exc.value.status_code == 503
    assert "model missing" in exc.value.detail
