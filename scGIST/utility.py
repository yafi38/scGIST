import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def plot_confusion_matrix(y, y_pred, labels=None, title=None, save_path=None, annot=False):
    if labels is None:
        cm = pd.DataFrame(confusion_matrix(y, y_pred))
    else:
        cm = pd.DataFrame(confusion_matrix(y, y_pred, normalize='true'), index=labels, columns=labels) * 100

    sns.set_theme()
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, vmin=0, vmax=100, annot=annot, annot_kws={'fontsize': 10}, fmt='.0f', cmap="viridis", square=True)

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(size=8, rotation=45, ha='right', rotation_mode='default')
    plt.yticks(size=8)

    if title is not None:
        plt.title(title)
    if save_path is not None:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()


def plot_marker_weights(markers, weights):
    total_markers = len(markers)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(total_markers / 7.5, 4))
    plt.bar(markers, weights)
    plt.xlim([-1, total_markers])
    plt.xticks(size=6, rotation=90, rotation_mode='default')
    plt.grid()
    plt.show()


def plot_history(history):
    sns.set_theme(style="whitegrid")
    f, axs = plt.subplots(1, 2, figsize=(20, 8))

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


def test_classifier(adata=None, label_column=None, X=None, y=None, markers=None, labels=None, clf=None, plot_cm=False, title=None, save_path=None):
    """
    Test performance of the markers using a classifier
    :param label_column: AnnData column name that contains the label names of the cell types
    :param adata: AnnData object
    :param X: data
    :param y: label encoding of cell types
    :param markers: selected markers or None. If None, run the classifier on whole dataset
    :param labels: name of the cell types
    :param clf: a classifier
    :param plot_cm: plots the confusion matrix
    :param title: title of the confusion matrix
    :param save_path: save path of the confusion matrix
    :return: accuracy of the classifier
    """
    if adata is not None:
        if label_column is None:
            print("Please provide the column name in adata.obs to get the cell types")
            return
        X = np.array(adata.X)
        y, names = adata.obs[label_column].factorize()
        y = y.tolist()
    elif X is None or y is None:
        print("Please provide data to train on")
        return

    if clf is None:
        clf = KNeighborsClassifier()

    if markers is not None:
        X = X[:, markers]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=33, stratify=y
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    if plot_cm:
        if title is None:
            title = 'Classifier Confusion Matrix'
        plot_confusion_matrix(y_test, y_pred, labels=labels, title=title, save_path=save_path)

    accuracy = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred, average="macro")

    return accuracy, f1


def get_priority_score_list(adata, gene_priorities):
    if 'gene_name' not in gene_priorities.columns or 'priority' not in gene_priorities.columns:
        print('priority_scores must contain "gene_name" and "priority" column')
        return None

    n_genes = adata.X.shape[1]
    priority_scores = np.zeros(n_genes)

    for _, row in gene_priorities.iterrows():
        if row['gene_name'] in adata.var_names:
            ind = adata.var_names.get_loc(row['gene_name'])
            priority_scores[ind] = row['priority']

    return priority_scores.tolist()
