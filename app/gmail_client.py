"""
Thin wrapper around the Gmail REST API for the polling worker.

Responsibilities:
    - OAuth user-credentials flow (loads/refreshes token.json).
    - Ensure AI_SPAM / AI_HAM labels exist (create on first run, cache IDs).
    - List new messages since a stored historyId (incremental, polling-friendly).
    - Fetch subject + body for a message.
    - Modify message labels.

We deliberately keep this surface minimal — the poller orchestrates calls,
this module is just the HTTP boundary.
"""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.labels",
]

LABEL_SPAM = os.getenv("GMAIL_LABEL_SPAM", "AI_SPAM")
LABEL_HAM = os.getenv("GMAIL_LABEL_HAM", "AI_HAM")


@dataclass
class GmailMessage:
    id: str
    history_id: str
    subject: str
    body: str
    snippet: str


def _load_creds(token_path: Path) -> Credentials:
    if not token_path.exists():
        raise FileNotFoundError(
            f"Gmail token.json not found at {token_path}. "
            "Run `python scripts/gmail_oauth_bootstrap.py` once to generate it."
        )
    creds = Credentials.from_authorized_user_file(str(token_path), GMAIL_SCOPES)
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            token_path.write_text(creds.to_json())
        else:
            raise RuntimeError(
                "Gmail credentials invalid and not refreshable. "
                "Re-run the OAuth bootstrap to regenerate token.json."
            )
    return creds


class GmailClient:
    """Resource-holder. Reuse one instance across poller ticks."""

    def __init__(self, token_path: Path):
        self._creds = _load_creds(token_path)
        self._service = build("gmail", "v1", credentials=self._creds, cache_discovery=False)
        self._label_cache: dict[str, str] = {}

    # ---------- Labels -----------------------------------------------------

    def ensure_label(self, name: str) -> str:
        """Return the label ID for `name`, creating it if missing. Cached."""
        if name in self._label_cache:
            return self._label_cache[name]

        existing = self._service.users().labels().list(userId="me").execute().get("labels", [])
        for lbl in existing:
            if lbl["name"] == name:
                self._label_cache[name] = lbl["id"]
                return lbl["id"]

        created = self._service.users().labels().create(
            userId="me",
            body={
                "name": name,
                "labelListVisibility": "labelShow",
                "messageListVisibility": "show",
            },
        ).execute()
        self._label_cache[name] = created["id"]
        return created["id"]

    def apply_prediction_label(self, message_id: str, label: str) -> None:
        """label ∈ {'spam','not_spam'} — applies AI_SPAM or AI_HAM."""
        target = LABEL_SPAM if label == "spam" else LABEL_HAM
        label_id = self.ensure_label(target)
        self._service.users().messages().modify(
            userId="me", id=message_id,
            body={"addLabelIds": [label_id]},
        ).execute()

    # ---------- History / message fetch -----------------------------------

    def current_history_id(self) -> str:
        return self._service.users().getProfile(userId="me").execute()["historyId"]

    def list_new_message_ids(self, start_history_id: str) -> tuple[list[str], str]:
        """Return (new_message_ids, latest_history_id). Empty list if nothing new."""
        ids: list[str] = []
        latest = start_history_id
        page_token = None

        while True:
            resp = self._service.users().history().list(
                userId="me",
                startHistoryId=start_history_id,
                historyTypes=["messageAdded"],
                pageToken=page_token,
            ).execute()
            for entry in resp.get("history", []):
                latest = entry.get("id", latest)
                for added in entry.get("messagesAdded", []):
                    msg = added.get("message", {})
                    if msg.get("id"):
                        ids.append(msg["id"])
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        # De-dupe (same message can appear in multiple history entries)
        seen = set()
        unique_ids = [m for m in ids if not (m in seen or seen.add(m))]
        return unique_ids, latest

    def fetch_message(self, message_id: str) -> GmailMessage:
        msg = self._service.users().messages().get(
            userId="me", id=message_id, format="full",
        ).execute()
        headers = {h["name"].lower(): h["value"] for h in msg.get("payload", {}).get("headers", [])}
        subject = headers.get("subject", "")
        body = _extract_body(msg.get("payload", {}))
        return GmailMessage(
            id=message_id,
            history_id=str(msg.get("historyId", "")),
            subject=subject,
            body=body,
            snippet=msg.get("snippet", ""),
        )


def _extract_body(payload: dict) -> str:
    """Walk the MIME parts and return the first text/plain (or text/html) body."""
    if not payload:
        return ""
    if payload.get("body", {}).get("data"):
        return _decode_b64(payload["body"]["data"])

    plain, html = "", ""
    for part in _iter_parts(payload):
        mime = part.get("mimeType", "")
        data = part.get("body", {}).get("data")
        if not data:
            continue
        if mime == "text/plain" and not plain:
            plain = _decode_b64(data)
        elif mime == "text/html" and not html:
            html = _decode_b64(data)
    return plain or html


def _iter_parts(payload: dict) -> Iterable[dict]:
    yield payload
    for p in payload.get("parts", []) or []:
        yield from _iter_parts(p)


def _decode_b64(data: str) -> str:
    try:
        return base64.urlsafe_b64decode(data.encode("utf-8")).decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return ""
