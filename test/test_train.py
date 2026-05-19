import numpy as np
import pytest
from scipy.sparse import csr_matrix

pytest.importorskip("xgboost")

from src.pipelines.train import maybe_slice_features, pick_winner


# Tiêu chí: MultinomialNB chỉ nhận phần TF-IDF không âm để tránh lỗi do feature số đã scale.
def test_maybe_slice_features_drops_numeric_block_for_nb():
    X = csr_matrix(np.arange(12).reshape(3, 4))

    sliced = maybe_slice_features("nb", X, n_tfidf=2)

    assert sliced.shape == (3, 2)
    assert sliced.toarray().tolist() == [[0, 1], [4, 5], [8, 9]]


# Tiêu chí: Các mô hình không phải MultinomialNB giữ nguyên toàn bộ đặc trưng huấn luyện.
def test_maybe_slice_features_keeps_all_features_for_non_nb():
    X = csr_matrix(np.arange(12).reshape(3, 4))

    assert maybe_slice_features("lr", X, n_tfidf=2) is X


# Tiêu chí: Chọn champion ưu tiên mô hình đạt precision mục tiêu và có recall validation cao nhất.
def test_pick_winner_prefers_hit_model_with_highest_recall():
    results = {
        "lr": {"hit_target": True, "val_metrics": {"recall": 0.7, "average_precision": 0.8}},
        "rf": {"hit_target": True, "val_metrics": {"recall": 0.9, "average_precision": 0.7}},
        "nb": {"hit_target": False, "val_metrics": {"recall": 1.0, "average_precision": 0.99}},
    }

    winner, reason = pick_winner(results, target_precision=0.99)

    assert winner == "rf"
    assert "highest recall" in reason


# Tiêu chí: Khi không mô hình nào đạt precision mục tiêu, champion fallback theo average precision cao nhất.
def test_pick_winner_falls_back_to_highest_average_precision_when_no_model_hits():
    results = {
        "lr": {"hit_target": False, "val_metrics": {"recall": 0.8, "average_precision": 0.72}},
        "rf": {"hit_target": False, "val_metrics": {"recall": 0.6, "average_precision": 0.88}},
        "nb": {"hit_target": False, "val_metrics": {"recall": 0.9, "average_precision": 0.81}},
    }

    winner, reason = pick_winner(results, target_precision=0.99)

    assert winner == "rf"
    assert "fallback" in reason
