import anndata as ad
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from scgist.model import scGIST


def _make_synthetic_adata(n_samples: int = 60, n_features: int = 6, n_classes: int = 3) -> ad.AnnData:
    X, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=n_features,
                       cluster_std=0.5, random_state=0)
    obs = pd.DataFrame({"cell_type": [f"class_{c}" for c in y]}, index=[str(i) for i in range(n_samples)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_features)])
    return ad.AnnData(X=X.astype(np.float32), obs=obs, var=var)


@pytest.fixture(scope="module")
def synthetic_adata() -> ad.AnnData:
    return _make_synthetic_adata()


@pytest.fixture(scope="module")
def trained_gist(synthetic_adata: ad.AnnData) -> scGIST:
    import tensorflow as tf
    tf.keras.utils.set_random_seed(0)

    n_features = synthetic_adata.n_vars
    n_classes = synthetic_adata.obs["cell_type"].nunique()

    gist = scGIST()
    gist.create_model(n_features, n_classes, panel_size=3)
    gist.compile_model()
    gist.train_model(synthetic_adata, "cell_type", epochs=2, verbose=0)
    return gist


def test_create_model_builds_expected_output_shape_and_layer_names():
    gist = scGIST()
    gist.create_model(n_features=10, n_classes=4, panel_size=3)

    assert gist.model is not None
    assert gist.model.output_shape == (None, 4)
    assert gist.model.get_layer("weighted_layer") is not None


def test_compile_model_requires_create_model_first():
    gist = scGIST()
    with pytest.raises(ValueError):
        gist.compile_model()


def test_train_model_requires_compile_model_first():
    gist = scGIST()
    gist.create_model(n_features=5, n_classes=2)
    with pytest.raises(ValueError):
        gist.train_model(X=np.zeros((4, 5)), y=[0, 1, 0, 1])


def test_train_model_requires_data():
    gist = scGIST()
    gist.create_model(n_features=5, n_classes=2)
    gist.compile_model()
    with pytest.raises(ValueError):
        gist.train_model()


def test_get_markers_indices_requires_trained_model():
    gist = scGIST()
    with pytest.raises(ValueError):
        gist.get_markers_indices()


def test_end_to_end_training_produces_requested_panel_size(trained_gist, synthetic_adata):
    markers = trained_gist.get_markers_names(synthetic_adata)

    assert len(markers) == 3
    assert all(isinstance(name, str) for name in markers)


def test_get_markers_indices_return_types_match_return_weights_flag(trained_gist):
    indices = trained_gist.get_markers_indices()
    assert isinstance(indices, list)
    assert all(isinstance(i, int) for i in indices)

    indices_with_weights, weights = trained_gist.get_markers_indices(return_weights=True)
    assert indices_with_weights == indices
    assert isinstance(weights, np.ndarray)
    assert weights.shape[0] == len(indices)
