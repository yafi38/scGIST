import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Model, backend, initializers, regularizers
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Layer, Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import Regularizer, l2
from tensorflow.keras.utils import to_categorical

from scGIST.utility import *


class scGIST:
    def __init__(self):
        self.model = None
        self.panel_size = None
        self.strict = True

    def create_model(self, n_features, n_classes, panel_size=None, weights=None, pairs=None,
                     alpha=0.5, beta=0.5, gamma=0.5, strict=True):
        """
        Creates and compiles a DNN model
        :param n_features: no. of features/cells
        :param n_classes: no. of classes/clusters/labels
        :param panel_size: Total no. of features to be taken
        :param weights: List of features we are interested in
        :param pairs: Pairs of genes which should be included or excluded together
        :param alpha: strictness of the panel size
        :param beta: priority coefficient
        :param gamma: likeliness to take pairs together
        :param strict: when True, the model will select exactly the same amount of genes specified in the panel size;
        when False, the model will select less than or equal to the amount of genes specified.
        """
        self.panel_size = panel_size
        self.strict = strict

        inputs = Input(shape=(n_features,), name='inputs')

        feature_regularizer = FeatureRegularizer(l1=0.01, panel_size=panel_size, priority_score=weights, pairs=pairs,
                                                 alpha=alpha, beta=beta, gamma=gamma, strict=strict)
        weighted_layer = OneToOneLayer(kernel_regularizer=feature_regularizer, name='weighted_layer')(inputs)

        hidden1 = Dense(
            units=32, activation='relu', kernel_regularizer=l2(0.01), name='hidden_layer1'
        )(weighted_layer)

        hidden2 = Dense(
            units=16, activation='relu', kernel_regularizer=l2(0.01), name='hidden_layer2'
        )(hidden1)

        outputs = Dense(
            units=n_classes, activation='softmax', kernel_regularizer=l2(0.01), name='outputs'
        )(hidden2)

        self.model = Model(inputs, outputs)

    def compile_model(self, opt=None):
        """
        Compiles the Keras model. It can also be compiled using self.model.compile() function of Keras.
        :param opt: Keras optimizer. If None, Adam optimizer will be used
        """
        if opt is None:
            opt = Adam(learning_rate=0.001)

        self.model.compile(
            optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy', 'Precision', 'Recall']
        )

    def train_model(self, X, y, validation_split=0.2, verbose=2, epochs=200):
        """
        Trains the Keras model. It can also be trained using self.model.fit() function of Keras.
        :param epochs: no. of epochs
        :param X: input data
        :param y: target data/labels. y is assumed not to be one hot encoded
        :param validation_split: fraction of data used as validation data
        :param verbose: set verbosity level
        :return: network train history
        """

        class_labels = np.unique(y)
        class_weight = compute_class_weight(class_weight='balanced', classes=class_labels, y=y)
        class_weight = dict(zip(class_labels, class_weight))

        y_cat = to_categorical(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_cat, test_size=validation_split, random_state=33, stratify=y
        )

        history = self.model.fit(
            X_train, y_train,
            batch_size=64, verbose=verbose,
            validation_data=(X_test, y_test),
            epochs=epochs,
            class_weight=class_weight
        )

        if verbose >= 1:
            print('Loss: %.4f, Val Loss: %.4f' %
                  (history.history['loss'][-1], history.history['val_loss'][-1]))
            print('Accuracy: %.2f, Val Accuracy: %.2f' % (
                history.history['accuracy'][-1] * 100, history.history['val_accuracy'][-1] * 100))

            plot_history(history)

        if verbose == 2:
            y_pred = np.argmax(self.model.predict(X_test), axis=1)
            y_test = np.argmax(y_test, axis=1)

            plot_confusion_matrix(y_test, y_pred, title='Model Confusion Matrix')
        return history

    def get_markers(self, verbose=0):
        # obtain weights of the weighted layer
        weights = abs(self.model.get_layer('weighted_layer').weights[0]).numpy()

        if not self.strict:
            # get weights with significant values
            significant_weights = weights[weights > 0.01]

            total_sig_weights = significant_weights.shape[0]

            if verbose != 0:
                print('Significant Weights Found: ', total_sig_weights)

            if self.panel_size is None:
                self.panel_size = total_sig_weights
            else:
                self.panel_size = min(self.panel_size, total_sig_weights)

        markers = sorted(
            range(len(weights)), key=lambda i: weights[i], reverse=True
        )[: self.panel_size]

        if verbose != 0:
            plot_weights(weights[markers])

        return markers


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
