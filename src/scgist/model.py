from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, overload

import numpy as np
import numpy.typing as npt
from anndata import AnnData
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Model
from tensorflow.keras.callbacks import History
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam, Optimizer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

from scgist.layers import FeatureRegularizer, OneToOneLayer
from scgist.plotting import plot_confusion_matrix, plot_history, plot_marker_weights


class scGIST:
    def __init__(self) -> None:
        self.model: Model | None = None
        self.panel_size: int | None = None
        self.strict: bool = True

    def create_model(
        self,
        n_features: int,
        n_classes: int,
        panel_size: int | None = None,
        priority_scores: Sequence[float] | None = None,
        pairs: Sequence[Sequence[float]] | None = None,
        alpha: float = 0.5,
        beta: float = 0.2,
        gamma: float = 0.5,
        strict: bool = True,
    ) -> None:
        """
        Creates and compiles a DNN model
        :param n_features: no. of features/cells
        :param n_classes: no. of classes/clusters/labels
        :param panel_size: Total no. of features to be taken
        :param priority_scores: List of features we are interested in
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

        feature_regularizer = FeatureRegularizer(
            l1=0.01, panel_size=panel_size, priority_score=priority_scores, pairs=pairs,
            alpha=alpha, beta=beta, gamma=gamma, strict=strict,
        )
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

    def compile_model(self, opt: Optimizer | None = None) -> None:
        """
        Compiles the Keras model. It can also be compiled using self.model.compile() function of Keras.
        :param opt: Keras optimizer. If None, Adam optimizer will be used
        """
        if self.model is None:
            raise ValueError("Call create_model before compile_model")

        if opt is None:
            opt = Adam(learning_rate=0.001)

        self.model.compile(
            optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy', 'Precision', 'Recall']
        )

    def train_model(
        self,
        adata: AnnData | None = None,
        label_column: str | None = None,
        X: npt.NDArray[np.floating[Any]] | None = None,
        y: Sequence[int] | npt.NDArray[np.integer[Any]] | None = None,
        validation_split: float = 0.2,
        verbose: int = 2,
        epochs: int = 200,
    ) -> History:
        """
        Trains the Keras model. It can also be trained using self.model.fit() function of Keras.
        :param label_column: AnnData column name that contains the label names of the cell types
        :param adata: AnnData object
        :param epochs: no. of epochs
        :param X: input data
        :param y: target data/labels. y is assumed not to be one hot encoded
        :param validation_split: fraction of data used as validation data
        :param verbose: set verbosity level
        :return: network train history
        """
        if self.model is None:
            raise ValueError("Call create_model and compile_model before train_model")

        if adata is not None:
            if label_column is None:
                raise ValueError("label_column is required when adata is provided")
            X = np.array(adata.X)
            y_codes, _names = adata.obs[label_column].factorize()
            y = y_codes.tolist()
        elif X is None or y is None:
            raise ValueError("Provide either adata and label_column, or X and y")

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
            print(f"Loss: {history.history['loss'][-1]:.4f}, Val Loss: {history.history['val_loss'][-1]:.4f}")
            print(f"Accuracy: {history.history['accuracy'][-1] * 100:.2f}, "
                  f"Val Accuracy: {history.history['val_accuracy'][-1] * 100:.2f}")

            plot_history(history)

        if verbose == 2:
            y_pred = np.argmax(self.model.predict(X_test), axis=1)
            y_test = np.argmax(y_test, axis=1)

            plot_confusion_matrix(y_test, y_pred, title='Model Confusion Matrix')
        return history

    def get_markers_names(
        self,
        adata: AnnData,
        verbose: int = 0,
        plot_weights: bool = False,
    ) -> list[str]:
        markers_indices, marker_weights = self.get_markers_indices(verbose, return_weights=True)
        markers_names = adata.var_names[markers_indices].tolist()
        if plot_weights:
            plot_marker_weights(markers_names, marker_weights)

        return markers_names

    @overload
    def get_markers_indices(
        self, verbose: int = 0, plot_weights: bool = False, *, return_weights: Literal[False] = False
    ) -> list[int]: ...

    @overload
    def get_markers_indices(
        self, verbose: int = 0, plot_weights: bool = False, *, return_weights: Literal[True]
    ) -> tuple[list[int], npt.NDArray[np.floating[Any]]]: ...

    def get_markers_indices(
        self,
        verbose: int = 0,
        plot_weights: bool = False,
        return_weights: bool = False,
    ) -> list[int] | tuple[list[int], npt.NDArray[np.floating[Any]]]:
        if self.model is None:
            raise ValueError("Call create_model, compile_model and train_model first")

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

        if plot_weights:
            plot_marker_weights(markers, weights[markers])

        if return_weights:
            return markers, weights[markers]
        else:
            return markers
