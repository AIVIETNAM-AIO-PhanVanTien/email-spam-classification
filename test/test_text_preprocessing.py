import pandas as pd

from src.utils.text_preprocessing import TextCleaner


# Tiêu chí: Bộ làm sạch văn bản loại bỏ nhiễu spam phổ biến nhưng vẫn giữ token có giá trị.
def test_aggressive_clean_normalizes_common_spam_noise():
    series = pd.Series(["FREEEEE\tcaf\u00e9!!! cescapenumber a an winner"])

    cleaned = TextCleaner(series).aggressive_clean().get()

    assert cleaned.iloc[0] == "free cafe escapenumber winner"


# Tiêu chí: Bộ lọc từ ngắn tuân thủ ngưỡng tối thiểu được cấu hình.
def test_remove_short_words_keeps_configurable_minimum():
    series = pd.Series(["a an cat deal"])

    cleaned = TextCleaner(series).remove_short_words(min_len=4).get()

    assert cleaned.iloc[0] == "deal"


# Tiêu chí: Chuẩn hóa unicode chuyển ký tự có dấu và full-width về dạng ASCII ổn định.
def test_normalize_unicode_converts_accents_and_full_width_chars():
    series = pd.Series(["ＦＲＥＥ café"])

    cleaned = TextCleaner(series).normalize_unicode().get()

    assert cleaned.iloc[0] == "FREE cafe"


# Tiêu chí: Chuẩn hóa escapenumber gom các biến thể obfuscation về một token thống nhất.
def test_normalize_escapenumber_collapses_obfuscated_variants():
    series = pd.Series(["cescapenumber 3escapenumber abcdescapenumber"])

    cleaned = TextCleaner(series).to_lower().normalize_escapenumber().get()

    assert cleaned.iloc[0] == "escapenumber escapenumber escapenumber"
