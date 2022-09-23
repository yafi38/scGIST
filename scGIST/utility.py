import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras import initializers, regularizers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


class ThresholdedLinear(Layer):
    """Thresholded Linear Activation.
    It follows:
    ```
      f(x) = x for abs(x) > theta
      f(x) = 0 otherwise`
    ```
    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.
    Output shape:
      Same shape as the input.
    Args:
      theta: Float >= 0. Threshold location of activation.
    """

    def __init__(self, theta=1.0, **kwargs):
        super(ThresholdedLinear, self).__init__(**kwargs)
        if theta is None:
            raise ValueError(
                'Theta of a Thresholded Linear layer cannot be None, expecting a float.'
                f' Received: {theta}')
        if theta < 0:
            raise ValueError('The theta value of a Thresholded Linear layer '
                             f'should be >=0. Received: {theta}')
        self.supports_masking = True
        self.theta = backend.cast_to_floatx(theta)

    def call(self, inputs):
        theta = tf.cast(self.theta, inputs.dtype)
        return inputs * tf.cast(tf.greater(tf.abs(inputs), theta), inputs.dtype)

    def get_config(self):
        config = {'theta': float(self.theta)}
        base_config = super(ThresholdedLinear, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


# TODO: change documentation
class FS(Regularizer):
    """A regularizer that applies a Feature Selector regularization penalty.
    Attributes:
        l1: Float; FS regularization factor.
    """

    def __init__(self, l1=0.1, n_features=None, s=None, pairs=None, alpha=0.5, beta=0.5, gamma=0.5,
                 **kwargs):  # pylint: disable=redefined-outer-name
        l1 = kwargs.pop('l', l1)  # Backwards compatibility
        if kwargs:
            raise TypeError(f'Argument(s) not recognized: {kwargs}')

        l1 = 0.01 if l1 is None else l1

        self.l1 = backend.cast_to_floatx(l1)
        self.alpha = backend.cast_to_floatx(alpha)
        self.beta = backend.cast_to_floatx(beta)
        self.gamma = backend.cast_to_floatx(gamma)

        self.n_features = tf.constant(n_features, dtype=tf.dtypes.float32) if n_features else None
        if s is not None:
            self.s = tf.constant(s, dtype=tf.dtypes.float32)
        else:
            self.s = None

        if pairs is not None:
            self.pairs = tf.constant(pairs, dtype=tf.dtypes.float32)
            # self.n_pairs = tf.constant(pairs.shape[1], dtype=tf.dtypes.float32)
        else:
            self.pairs = None
            # self.n_pairs = None

    def __call__(self, x):
        abs_x = tf.abs(x)
        # sum_x = tf.reduce_sum(abs_x)
        regularization = tf.constant(0., dtype=x.dtype)

        regularization += tf.reduce_sum(abs_x * tf.abs(x - 1))
        # if total sum is greater than number of features
        if self.n_features:
            regularization += tf.abs(tf.reduce_sum(abs_x) - self.n_features) * 1.5
            # regularization += tf.maximum(tf.reduce_sum(abs_x) - self.n_features, 0) * self.alpha

        # if weight of weighted feature is less than 1
        if self.s is not None:
            # regularization += tf.reduce_sum(tf.multiply(tf.abs(abs_x - 1), self.s)) * self.alpha
            regularization += tf.reduce_sum(tf.multiply(1 - tf.minimum(x, 1), self.s)) * self.beta

        # if pairs are given
        if self.pairs is not None:
            regularization += tf.reduce_sum(tf.abs(tf.tensordot(abs_x, self.pairs, axes=1))) * self.gamma

        return self.l1 * regularization

    def get_config(self):
        return {'l1': float(self.l1)}


class WeightedLayer(Layer):
    def __init__(self,
                 kernel_initializer='ones',
                 kernel_regularizer=None,
                 **kwargs):
        super(WeightedLayer, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

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
        config = super(WeightedLayer, self).get_config()
        config.update({
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        })
        return config


###############################################################################
def plot_confusion_matrix(y, y_pred, labels=None, title=None, save_path=None):
    if labels is None:
        cm = pd.DataFrame(confusion_matrix(y, y_pred))
    else:
        cm = pd.DataFrame(confusion_matrix(y, y_pred, normalize='true'), index=labels, columns=labels) * 100

    sns.set_theme()
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, vmin=0, vmax=100, annot=True, annot_kws={'fontsize': 10}, fmt='.0f', cmap="viridis", square=True)
    # sns.heatmap(cm, vmin=0, vmax=100, cmap="viridis", cbar=False, square=True)

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(size=8, rotation=45, ha='right', rotation_mode='default')
    plt.yticks(size=8)

    if title is not None:
        plt.title(title)
    if save_path is not None:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()


def plot_weights(weights):
    plt.bar(np.arange(weights.shape[0]), weights)
    plt.grid()
    plt.show()


def plot_history(history):
    sns.set_theme()
    # sns.despine(offset=5, trim=False)
    f, axs = plt.subplots(1, 2, figsize=(20, 8))

    # sns.set(font_scale=1.5)
    # sns.set_style("white")

    axs[0].semilogy(history.history['loss'])
    axs[0].semilogy(history.history['val_loss'])
    axs[0].set_title('Loss')
    axs[0].set_ylabel('loss')
    axs[0].set_xlabel('epoch')
    axs[0].legend(['Train Set', 'Validation Set'], loc='upper right')

    axs[1].semilogy(history.history['accuracy'])
    axs[1].semilogy(history.history['val_accuracy'])
    axs[1].set_title('Accuracy')
    axs[1].set_ylabel('accuracy')
    axs[1].set_xlabel('epoch')
    axs[0].legend(['Train Set', 'Validation Set'], loc='upper right')

    plt.tight_layout()
    plt.show(block=False)
    # plt.close("all")


def test_classifier(X, y, markers=None, labels=None, clf=None, title=None, verbose=1, save_path=None):
    """
    Test performance of the markers using a classifier
    :param X: data
    :param y: labels
    :param markers: selected markers or None. If None, run the classifier on whole dataset
    :param clf: a classifier
    :return: accuracy of the classifier
    """
    if clf is None:
        clf = KNeighborsClassifier()

    if markers is not None:
        X = X[:, markers]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=33, stratify=y
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    if verbose == 1:
        if title is None:
            title = 'Classifier Confusion Matrix'
        plot_confusion_matrix(y_test, y_pred, labels=labels, title=title, save_path=save_path)

    # accuracy = clf.score(X_test, y_test) * 100
    accuracy = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred, average="macro")
    # print('Accuracy: %.2f' % accuracy)
    return accuracy, f1
