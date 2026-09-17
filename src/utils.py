import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def save_confusion_matrix(y_true, y_pred, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(y_true, y_proba, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_pr_curve(y_true, y_proba, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
