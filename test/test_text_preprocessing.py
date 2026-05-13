import pandas as pd

from src.utils.text_preprocessing import TextCleaner


def test_aggressive_clean_normalizes_common_spam_noise():
    series = pd.Series(["FREEEEE\tcaf\u00e9!!! cescapenumber a an winner"])

    cleaned = TextCleaner(series).aggressive_clean().get()

    assert cleaned.iloc[0] == "free cafe escapenumber winner"


def test_remove_short_words_keeps_configurable_minimum():
    series = pd.Series(["a an cat deal"])

    cleaned = TextCleaner(series).remove_short_words(min_len=4).get()

    assert cleaned.iloc[0] == "deal"
