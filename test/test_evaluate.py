import numpy as np
import pytest

from src.pipelines.evaluate import evaluate, pick_threshold


# Tiêu chí: Ngưỡng phân loại được chọn tại điểm đầu tiên đạt precision mục tiêu.
def test_pick_threshold_returns_first_threshold_hitting_target_precision():
    threshold, precision, recall, hit = pick_threshold(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.8, 0.9]),
        target_precision=0.9,
    )

    assert threshold == pytest.approx(0.8)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)
    assert hit is True


# Tiêu chí: Bước evaluate tính đúng các chỉ số phân loại cốt lõi tại một ngưỡng cố định.
def test_evaluate_returns_metrics_at_fixed_threshold():
    metrics = evaluate(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.7, 0.8, 0.9]),
        threshold=0.75,
    )

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


# Tiêu chí: Chọn threshold fallback an toàn khi không điểm nào đạt precision mục tiêu.
def test_pick_threshold_falls_back_when_target_precision_is_not_hit():
    threshold, precision, recall, hit = pick_threshold(
        np.array([1, 0, 0, 0]),
        np.array([0.1, 0.2, 0.3, 0.4]),
        target_precision=1.1,
    )

    assert hit is False
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.0)
    assert threshold == pytest.approx(1.0)
