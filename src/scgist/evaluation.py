from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
from anndata import AnnData
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from scgist.plotting import plot_confusion_matrix


def test_classifier(
    adata: AnnData | None = None,
    label_column: str | None = None,
    X: npt.NDArray[np.floating[Any]] | None = None,
    y: Sequence[int] | npt.NDArray[np.integer[Any]] | None = None,
    markers: Sequence[int] | None = None,
    labels: Sequence[str] | None = None,
    clf: ClassifierMixin | None = None,
    plot_cm: bool = False,
    title: str | None = None,
    save_path: str | None = None,
) -> tuple[float, float]:
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
    :return: accuracy and macro F1 score of the classifier
    """
    if adata is not None:
        if label_column is None:
            raise ValueError("label_column is required when adata is provided")
        X = np.array(adata.X)
        y_codes, _names = adata.obs[label_column].factorize()
        y = y_codes.tolist()
    elif X is None or y is None:
        raise ValueError("Provide either adata and label_column, or X and y")

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
