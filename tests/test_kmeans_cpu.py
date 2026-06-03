import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cpu_backend.kmeans import KMeansCPU


def test_kmeans_cpu_initializes_with_expected_defaults():
    model = KMeansCPU()

    assert model.n_clusters == 3
    assert model.max_iter == 300
    assert model.tol == 1e-4
    assert model.random_state is None
    assert model.centroids is None
    assert model.labels is None
    assert model.n_iters_ == 0


def test_fit_generates_labels_for_simple_clusters():
    samples = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.2],
            [9.8, 10.0],
            [10.2, 9.9],
        ]
    )
    model = KMeansCPU(n_clusters=2, random_state=0)

    result = model.fit(samples)

    assert result is model
    assert model.labels.shape == (samples.shape[0],)
    assert set(model.labels.tolist()) == {0, 1}
    assert model.centroids.shape == (2, samples.shape[1])
    assert model.n_iters_ > 0


def test_predict_uses_fitted_centroids_for_new_samples():
    samples = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [5.0, 5.0],
            [5.1, 5.2],
        ]
    )
    model = KMeansCPU(n_clusters=2, random_state=1).fit(samples)

    labels = model.predict(np.array([[0.1, 0.1], [5.2, 5.1]]))

    assert labels.shape == (2,)
    assert labels[0] != labels[1]


def test_fit_predict_returns_labels_for_tiny_dataset():
    samples = np.array([[2.0, 2.0], [8.0, 8.0]])
    model = KMeansCPU(n_clusters=2, random_state=42)

    labels = model.fit_predict(samples)

    assert labels.shape == (2,)
    assert set(labels.tolist()) == {0, 1}


def test_predict_before_fit_raises_clear_error():
    model = KMeansCPU(n_clusters=2)

    with pytest.raises(ValueError, match="has not been fitted"):
        model.predict(np.array([[1.0, 1.0]]))
