import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


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
    plt.bar(np.arange(weights.shape[0]) + 1, weights)
    plt.xlim([1, weights.shape[0]])
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
