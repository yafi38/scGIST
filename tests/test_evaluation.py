import numpy as np
import pytest
from sklearn.datasets import make_blobs

# Imported under an alias: the function is named `test_classifier`, which
# would otherwise collide with pytest's own test-discovery convention.
from scgist.evaluation import test_classifier as evaluate_classifier


def _make_signal_and_noise_dataset(n_samples: int = 60, n_informative: int = 2, n_noise: int = 4):
    X_informative, y = make_blobs(n_samples=n_samples, centers=3, n_features=n_informative,
                                   cluster_std=0.5, random_state=0)
    rng = np.random.RandomState(0)
    X_noise = rng.normal(size=(n_samples, n_noise))
    X = np.hstack([X_informative, X_noise]).astype(np.float32)

    informative_markers = list(range(n_informative))
    noise_markers = list(range(n_informative, n_informative + n_noise))
    return X, y, informative_markers, noise_markers


def test_markers_argument_restricts_classifier_to_selected_columns():
    X, y, informative_markers, noise_markers = _make_signal_and_noise_dataset()

    accuracy_on_informative, _ = evaluate_classifier(X=X, y=y, markers=informative_markers)
    accuracy_on_noise, _ = evaluate_classifier(X=X, y=y, markers=noise_markers)

    assert accuracy_on_informative > accuracy_on_noise


@pytest.mark.parametrize("kwargs", [{}, {"X": np.zeros((4, 2))}])
def test_requires_either_adata_or_x_and_y(kwargs):
    with pytest.raises(ValueError):
        evaluate_classifier(**kwargs)
