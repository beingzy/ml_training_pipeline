import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
)


def plot_roc_curve(y, proba, curve_type='roc'):
    """Return matplotlib figure object depicting either
       ROC-curve or PR-curve.
    """
    fpr, tpr, _ = roc_curve(y, proba)
    auc_score = auc(fpr, tpr, reorder=True)

    fig = plt.figure()

    plt.subplot(1, 1, 1)
    plt.plot(fpr, tpr, color='darkorange')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('2-Class ROC Curve: AUC={:.3f}'.format(auc_score))
    plt.legend(loc="lower right")

    return fig


def plot_pr_curve(y, proba):
    precision, recall, _ = precision_recall_curve(y, proba)
    avg_precision = average_precision_score(y, proba)

    fig = plt.figure()

    plt.subplot(1, 1, 1)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(
        recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.title('2-Class Precision-Recall curve: AP={:.3f}'
              .format(avg_precision))

    return fig
