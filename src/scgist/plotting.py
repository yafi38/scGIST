from __future__ import annotations

from typing import Any, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import History


def plot_confusion_matrix(
    y: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[str] | None = None,
    title: str | None = None,
    save_path: str | None = None,
    annot: bool = False,
) -> None:
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


def plot_marker_weights(markers: Sequence[Any], weights: Sequence[float]) -> None:
    total_markers = len(markers)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(total_markers / 7.5, 4))
    plt.bar(markers, weights)
    plt.xlim([-1, total_markers])
    plt.xticks(size=6, rotation=90, rotation_mode='default')
    plt.grid()
    plt.show()


def plot_history(history: History) -> None:
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
