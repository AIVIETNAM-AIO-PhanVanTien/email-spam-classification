import numpy as np
import pytest

from src.monitoring.drift_detector import (
    aggregate_drift_score,
    chi_square_label_drift,
    population_stability_index,
)


# Tiêu chí: PSI bằng 0 khi phân phối dữ liệu mới giống hoàn toàn dữ liệu tham chiếu.
def test_population_stability_index_is_zero_for_identical_values():
    values = np.array([0.0, 0.1, 0.2, 0.3, 0.4])

    assert population_stability_index(values, values) == 0.0


# Tiêu chí: PSI tăng lên khi phân phối dữ liệu mới có dấu hiệu lệch khỏi dữ liệu tham chiếu.
def test_population_stability_index_detects_distribution_shift():
    ref = np.array([0.0, 0.0, 0.1, 0.1, 0.2])
    new = np.array([0.8, 0.9, 0.9, 1.0, 1.0])

    assert population_stability_index(ref, new) > 0.0


# Tiêu chí: Kiểm định label drift báo cáo đúng tỷ lệ ham/spam và kích thước từng cửa sổ dữ liệu.
def test_chi_square_label_drift_reports_label_distributions():
    report = chi_square_label_drift(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 1]),
    )

    assert report["ref_dist"] == {"ham": 0.5, "spam": 0.5}
    assert report["new_dist"] == {"ham": 0.25, "spam": 0.75}
    assert report["ref_n"] == 4
    assert report["new_n"] == 4


# Tiêu chí: Điểm drift tổng hợp kết hợp đúng tín hiệu nhãn và đặc trưng theo trọng số thiết kế.
def test_aggregate_drift_score_combines_label_and_feature_scores():
    score = aggregate_drift_score(
        {"drifted": True},
        {"mean_psi": 0.125},
    )

    assert score == pytest.approx(0.65)
