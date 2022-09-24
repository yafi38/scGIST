from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.initializers import RandomUniform, Constant
from sklearn.utils.class_weight import compute_class_weight
from scGIST.utility import *


class scGIST:
    def __init__(self):
        self.model = None
        self.panel_size = None

    def create_model(self, n_features, n_classes, panel_size=None, weights=None, pairs=None, decision_model=None,
                     alpha=0.5, beta=0.5, gamma=0.5):
        """
        Creates and compiles a DNN model
        :param n_features: no. of features/cells
        :param n_classes: no. of classes/clusters/labels
        :param panel_size: Total no. of features to be taken
        :param weights: List of features we are interested in
        :param pairs: Pairs of genes which should be included or excluded together
        :param decision_model: Keras model used as decision network. If None, neural-marker will
         create one
        """
        self.panel_size = panel_size

        inputs = Input(shape=(n_features,), name='inputs')
        weighted_layer = WeightedLayer(
            kernel_regularizer=FeatureRegularizer(l1=0.01, panel_size=panel_size, priority_score=weights, pairs=pairs, alpha=alpha, beta=beta, gamma=gamma),
            name='weighted_layer', kernel_initializer=Constant(0.5)
        )(inputs)

        if decision_model is None:
            hidden1 = Dense(
                units=32, activation='relu', kernel_regularizer=l2(0.01), name='hidden_layer1'
            )(weighted_layer)

            hidden2 = Dense(
                units=16, activation='relu', kernel_regularizer=l2(0.01), name='hidden_layer2'
            )(hidden1)

            outputs = Dense(
                units=n_classes, activation='softmax', kernel_regularizer=l2(0.01), name='outputs'
            )(hidden2)
        else:
            outputs = decision_model(weighted_layer)

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
        early_stop = EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-6)
        history = self.model.fit(
            X_train, y_train,
            batch_size=64, verbose=verbose,
            validation_data=(X_test, y_test),
            # callbacks=[early_stop],
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
        # only consider significant weights
        significant_weights = weights[weights > 0.01]

        if verbose != 0:
            plot_weights(significant_weights)

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

        return markers

# def main():
#     X, y = load_data()
#
#     n_features = X.shape[1]
#     n_labels = len(np.unique(y))
#
#     model = get_model(n_features, n_labels)
#
#     return model

#
# if __name__ == "__main__":
#     # final_model = main()
#     panel_size = 30
#     X, y_raw = load_pbmc()
#
#     n_features = X.shape[1]
#     n_labels = len(np.unique(y_raw))
#
#     y = to_categorical(y_raw)
#
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=33, stratify=y
#     )
#
#     early_stop = EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-6)
#
#     model = get_model(n_features, n_labels, panel_size=panel_size)
#     history = model.fit(
#         X_train, y_train,
#         batch_size=64, verbose=2,
#         validation_data=(X_test, y_test),
#         callbacks=[early_stop],
#         epochs=500
#     )
#
#     print('Loss: %.4f, Val Loss: %.4f' % (history.history['loss'][-1], history.history['val_loss'][-1]))
#     print('Accuracy: %.2f, Val Accuracy: %.2f' % (
#         history.history['accuracy'][-1] * 100, history.history['val_accuracy'][-1] * 100))
#
#     score = abs(model.get_layer('weighted_layer').weights[0])
#     # plt.bar(np.arange(n_features), weights)
#     # plt.grid()
#     # plt.savefig('3k.svg', format='svg')
#     # plt.show()
#
#     score = score.numpy()
#     significant_weights = score[score > 0.01]
#     plt.bar(np.arange(significant_weights.shape[0]), significant_weights)
#     plt.grid()
#     # plt.savefig('img/head_neck_weighted60.svg', format='svg')
#     plt.show()
#     print(np.sum((score > 0.01).astype('int32')))
#     print(np.sum(significant_weights))
#
#     markers = sorted(range(len(score)), key=lambda i: score[i], reverse=True)[: panel_size]
#     clf = KNeighborsClassifier()
#     accuracy_markers = performance(X[:, markers], y_raw, X[:, markers], y_raw, clf)
#     print('Accuracy:', accuracy_markers)
#
#     # if weights is not None:
#     #     taken = 0
#     #     for i in range(panel_size):
#     #         if weights[markers[i]] != 0:
#     #             # print(markers[i])
#     #             taken += 1
#     #     print("Gene of interest taken:", taken)
