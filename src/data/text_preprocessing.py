"""
Text cleaning utility — dùng cho Silver layer.

EDA findings (01_eda.ipynb trên Bronze data):
- Dataset là plain text hoàn toàn: 0% HTML, 0% URL, 0% email headers
  → bỏ strip_email_headers, replace_html, replace_urls
- Phát hiện obfuscation pattern 'escapenumber' trong spam
  → thêm normalize_escapenumber()
- Outlier max length = 11.5M chars → truncate ở Gold/Quality check
- Ngưỡng drop short text nâng từ 20 lên 50 chars (xử lý ở quality check)

Nguyên tắc: TextCleaner chỉ làm sạch text, không quyết định drop/keep rows.
Drop logic thuộc về run_quality_check() trong silver_transform.py.
"""
import re
import unicodedata
import pandas as pd


class TextCleaner:
    def __init__(self, series: pd.Series):
        self.s = series.astype(str)

    # ── 1. Normalize unicode ──────────────────────────────────────────────────
    # Chuẩn hóa full-width chars và loại bỏ ký tự non-ASCII
    # ví dụ: ＦＲＥＥl (full-width) → FREE, café → cafe
    def normalize_unicode(self):
        self.s = self.s.apply(
            lambda x: unicodedata.normalize("NFKD", x)
                                 .encode("ascii", "ignore")
                                 .decode("ascii")
        )
        return self

    # ── 2. Lowercase ──────────────────────────────────────────────────────────
    def to_lower(self):
        self.s = self.s.str.lower()
        return self

    # ── 3. Remove newlines & tabs ─────────────────────────────────────────────
    # Gộp toàn bộ email thành 1 dòng để xử lý nhất quán
    def remove_linebreaks(self):
        self.s = self.s.str.replace(r"[\n\r\t]+", " ", regex=True)
        return self

    # ── 4. Fix repeated characters ────────────────────────────────────────────
    # Spam hay dùng kỹ thuật lặp ký tự để bypass filter
    # viiiiagra → viigra, FREEEEE → freee (sau lowercase)
    # Giữ lại 2 ký tự thay vì 1 để không mất nghĩa từ (ví dụ: "good" vs "goood")
    def fix_repeated_chars(self):
        self.s = self.s.str.replace(r"(.)\1{2,}", r"\1\1", regex=True)
        return self

    # ── 5. Normalize escapenumber obfuscation ───────────────────────────────────
    def normalize_escapenumber(self):
        # Normalize tất cả biến thể về 1 token chuẩn
        # cescapenumber, bescapenumber, 3escapenumber... → escapenumber
        self.s = self.s.str.replace(
            r"\b[a-z0-9]{0,4}escapenumber\b", "escapenumber", regex=True
        )
        return self

    # ── 6. Remove special characters ─────────────────────────────────────────
    # Chỉ giữ chữ cái, số, khoảng trắng
    def remove_special_chars(self):
        self.s = self.s.str.replace(r"[^a-z0-9\s]", " ", regex=True)
        return self

    # ── 7. Clean extra spaces ─────────────────────────────────────────────────
    def clean_spaces(self):
        self.s = self.s.str.strip().str.replace(r"\s+", " ", regex=True)
        return self

    # ── 8. Remove very short tokens ───────────────────────────────────────────
    # Bỏ token < min_len ký tự (noise: ký tự đơn lẻ, 2-letter gibberish)
    # EDA xác nhận: top raw tokens có nhiều single chars như b, e, u, c, r, k, z
    def remove_short_words(self, min_len: int = 3):
        self.s = self.s.apply(
            lambda x: " ".join(w for w in x.split() if len(w) >= min_len)
        )
        return self

    # ── Full pipeline ─────────────────────────────────────────────────────────
    def aggressive_clean(self):
        """
        Pipeline cho dataset plain text (không có HTML/URL/headers).

        Thứ tự quan trọng:
            1. normalize_unicode    — chuẩn hóa encoding trước tất cả
            2. to_lower             — lowercase trước khi fix patterns
            3. remove_linebreaks    — gộp thành 1 dòng
            4. fix_repeated_chars   — sau lowercase mới catch được hết
            5. normalize_escapenumber  — sau lowercase để match đúng pattern
            6. remove_special_chars — sau escapenumber để không vỡ pattern
            7. clean_spaces         — normalize spaces sau khi đã xóa chars
            8. remove_short_words   — bước cuối, sau khi text đã ổn định
        """
        return (
            self.normalize_unicode()
                .to_lower()
                .remove_linebreaks()
                .fix_repeated_chars()
                .normalize_escapenumber()
                .remove_special_chars()
                .clean_spaces()
                .remove_short_words()
        )

    def get(self) -> pd.Series:
        return self.s