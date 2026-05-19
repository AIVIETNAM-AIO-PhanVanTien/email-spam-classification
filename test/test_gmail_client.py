import base64

import pytest

pytest.importorskip("googleapiclient.discovery")

from app.gmail_client import _decode_b64, _extract_body, _iter_parts


def _encoded(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("utf-8")


# Tiêu chí: Nội dung email mã hóa base64 URL-safe được giải mã chính xác.
def test_decode_b64_returns_text():
    assert _decode_b64(_encoded("Xin chao")) == "Xin chao"


# Tiêu chí: Trích xuất body ưu tiên bản text/plain để giữ nội dung email sạch và dễ phân loại.
def test_extract_body_prefers_plain_text_over_html():
    payload = {
        "parts": [
            {"mimeType": "text/html", "body": {"data": _encoded("<b>Hello</b>")}},
            {"mimeType": "text/plain", "body": {"data": _encoded("Hello plain")}},
        ]
    }

    assert _extract_body(payload) == "Hello plain"


# Tiêu chí: Bộ duyệt MIME parts đọc được cả cấu trúc email có nhiều tầng lồng nhau.
def test_iter_parts_walks_nested_payloads():
    nested = {"mimeType": "text/plain", "body": {"data": _encoded("Nested")}}
    payload = {"mimeType": "multipart/mixed", "parts": [{"parts": [nested]}]}

    parts = list(_iter_parts(payload))

    assert nested in parts


# Tiêu chí: Trích xuất body trả về chuỗi rỗng an toàn khi payload email không có nội dung.
def test_extract_body_returns_empty_string_for_missing_payload():
    assert _extract_body({}) == ""
    assert _extract_body({"parts": []}) == ""


# Tiêu chí: Decode base64 lỗi không làm vỡ poller và trả về chuỗi rỗng an toàn.
def test_decode_b64_returns_empty_string_for_invalid_data():
    assert _decode_b64("abc") == ""
