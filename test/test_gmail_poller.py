import csv

import pytest

pytest.importorskip("google.auth")
pytest.importorskip("googleapiclient.discovery")

from app import gmail_poller


# Tiêu chí: Poller khởi tạo state rỗng an toàn khi chưa có file lịch sử Gmail.
def test_load_state_returns_empty_dict_for_missing_file(tmp_path):
    assert gmail_poller._load_state(tmp_path / "missing.json") == {}


# Tiêu chí: last_history_id được lưu và đọc lại chính xác để tránh xử lý trùng email.
def test_save_and_load_state_round_trips_json(tmp_path):
    state_path = tmp_path / "gmail_state.json"

    gmail_poller._save_state(state_path, {"last_history_id": "123"})

    assert gmail_poller._load_state(state_path) == {"last_history_id": "123"}


# Tiêu chí: Log dự đoán Gmail có đầy đủ header và giữ đúng thứ tự cột theo thiết kế.
def test_append_log_writes_header_and_row(tmp_path):
    log_path = tmp_path / "gmail_predictions.csv"
    row = {
        "timestamp": "2026-05-17T00:00:00+00:00",
        "message_id": "m1",
        "subject": "Hello",
        "snippet": "Short",
        "prediction": "not_spam",
        "spam_probability": "0.100000",
        "applied_label": "AI_HAM",
    }

    gmail_poller._append_log(log_path, row)

    with log_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows == [row]


# Tiêu chí: Poller gửi đúng subject/body tới API và nhận lại kết quả dự đoán dạng JSON.
def test_predict_via_api_posts_subject_and_body(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"label": "spam", "spam_probability": 0.9, "threshold": 0.5}

    calls = []

    def fake_post(url, json, timeout):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr(gmail_poller.requests, "post", fake_post)

    result = gmail_poller.predict_via_api("http://api/predict", "Sub", "Body", timeout=3)

    assert result["label"] == "spam"
    assert calls == [
        {
            "url": "http://api/predict",
            "json": {"subject": "Sub", "body": "Body"},
            "timeout": 3,
        }
    ]
