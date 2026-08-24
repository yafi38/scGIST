import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from scgist.plotting import plot_confusion_matrix


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
