import numpy as np
import pytest
import tensorflow as tf

from scgist.layers import FeatureRegularizer, OneToOneLayer


@pytest.mark.parametrize("strict,expected", [(True, 2.0), (False, 0.0)])
def test_panel_size_penalty_respects_strict_flag(strict, expected):
    # x is already 0/1 (sum=1), so only the panel-size term contributes.
    # strict penalizes the shortfall from panel_size=3; non-strict only
    # penalizes overshoot, so a shortfall costs it nothing.
    x = tf.constant([1., 0., 0., 0., 0.])
    reg = FeatureRegularizer(l1=1.0, panel_size=3, alpha=1.0, strict=strict)
    assert float(reg(x)) == pytest.approx(expected)


def test_priority_score_penalizes_unselected_prioritized_genes():
    x = tf.zeros(5)
    with_priority = FeatureRegularizer(l1=1.0, priority_score=[1., 2., 3., 4., 5.], beta=1.0)
    without_priority = FeatureRegularizer(l1=1.0, beta=1.0)

    assert float(with_priority(x)) == pytest.approx(15.0)
    assert float(without_priority(x)) == pytest.approx(0.0)


def test_pairs_penalizes_genes_not_taken_together():
    x = tf.constant([1., 1., 0.])
    pairs = [[1., 0.], [1., 0.], [0., 1.]]
    with_pairs = FeatureRegularizer(l1=1.0, pairs=pairs, gamma=1.0)
    without_pairs = FeatureRegularizer(l1=1.0, gamma=1.0)

    assert float(with_pairs(x)) == pytest.approx(2.0)
    assert float(without_pairs(x)) == pytest.approx(0.0)


def test_get_config_roundtrips_l1():
    reg = FeatureRegularizer(l1=0.05)
    assert reg.get_config()["l1"] == pytest.approx(0.05)


def test_one_to_one_layer_default_kernel_multiplies_elementwise():
    layer = OneToOneLayer()
    inputs = tf.constant([[1., 2., 3., 4.]])

    output = layer(inputs)

    # default initializer is Constant(0.5)
    np.testing.assert_allclose(output.numpy(), [[0.5, 1.0, 1.5, 2.0]])
    assert layer.kernel.shape == (4,)


def test_one_to_one_layer_get_config_includes_initializer_and_regularizer():
    layer = OneToOneLayer()
    layer.build(tf.TensorShape([None, 3]))

    config = layer.get_config()

    assert "kernel_initializer" in config
    assert "kernel_regularizer" in config
