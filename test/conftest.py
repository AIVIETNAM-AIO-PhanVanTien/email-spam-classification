from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest


@pytest.fixture
def bronze_like_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "email_id": ["email_202401_00000", "email_202401_00001"],
            "body": [
                "FREEEEE cash prize!!! Click here escapenumber now.",
                "Hello team, the project meeting is confirmed for tomorrow.",
            ],
            "raw_label": ["spam", "ham"],
            "label": [1, 0],
            "received_at": [
                datetime(2024, 1, 2, 8, 30),
                datetime(2024, 1, 3, 9, 45),
            ],
        }
    )


@pytest.fixture
def long_body() -> str:
    return " ".join(["useful"] * 20)
