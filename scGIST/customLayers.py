import tensorflow as tf
from tensorflow.keras import backend, initializers, regularizers
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import Regularizer


class FeatureRegularizer(Regularizer):
    """A regularization layer that applies a Feature Selector regularization penalty.

    Args:
        l1: regularization factor.
        panel_size: number of genes to be included in the panel
        priority_score: priority values of genes
        pairs: pairs of genes that should be included in the panel together
        alpha: strictness of the panel size
        beta: priority coefficient
        gamma: likeliness to take pairs together
        strict: when True, the model will select exactly the same amount of genes specified in the panel size.
                when False, the model will select less than or equal to the amount of genes specified.
    """

    def __init__(self, l1=0.1, panel_size=None, priority_score=None, pairs=None, alpha=0.5, beta=0.5, gamma=0.5,
                 strict=True, **kwargs):  # pylint: disable=redefined-outer-name
        l1 = kwargs.pop('l', l1)  # Backwards compatibility
        if kwargs:
            raise TypeError(f'Argument(s) not recognized: {kwargs}')

        l1 = 0.01 if l1 is None else l1

        self.l1 = backend.cast_to_floatx(l1)
        self.alpha = backend.cast_to_floatx(alpha)
        self.beta = backend.cast_to_floatx(beta)
        self.gamma = backend.cast_to_floatx(gamma)

        self.n_features = tf.constant(panel_size, dtype=tf.dtypes.float32) if panel_size else None
        if priority_score is not None:
            self.s = tf.constant(priority_score, dtype=tf.dtypes.float32)
        else:
            self.s = None

        if pairs is not None:
            self.pairs = tf.constant(pairs, dtype=tf.dtypes.float32)
        else:
            self.pairs = None

        self.strict = strict

    def __call__(self, x):
        abs_x = tf.abs(x)
        regularization = tf.constant(0., dtype=x.dtype)

        # force weights to be either 0 or 1
        regularization += tf.reduce_sum(abs_x * tf.abs(x - 1))

        # force total sum to be either equal, or less than or equal to the gene panel size
        if self.n_features:
            if self.strict:
                regularization += tf.abs(tf.reduce_sum(abs_x) - self.n_features) * self.alpha
            else:
                regularization += tf.maximum(tf.reduce_sum(abs_x) - self.n_features, 0) * self.alpha

        # take prioritized genes
        if self.s is not None:
            regularization += tf.reduce_sum(tf.multiply(1 - tf.minimum(x, 1), self.s)) * self.beta

        # take pairs of gene together
        if self.pairs is not None:
            regularization += tf.reduce_sum(tf.abs(tf.tensordot(abs_x, self.pairs, axes=1))) * self.gamma

        return self.l1 * regularization

    def get_config(self):
        return {'l1': float(self.l1)}


class OneToOneLayer(Layer):
    def __init__(self,
                 kernel_initializer=None,
                 kernel_regularizer=None,
                 **kwargs):
        super(OneToOneLayer, self).__init__(**kwargs)

        if kernel_initializer:
            self.kernel_initializer = initializers.get(kernel_initializer)
        else:
            self.kernel_initializer = Constant(0.5)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel = None

    def build(self, input_shape):
        self.kernel = self.add_weight('kernel',
                                      shape=(int(input_shape[-1]),),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True,
                                      dtype=self.dtype)
        self.built = True

    def call(self, inputs):
        return tf.multiply(inputs, self.kernel)

    def get_config(self):
        config = super(OneToOneLayer, self).get_config()
        config.update({
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        })
        return config
