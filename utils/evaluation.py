# ROC plots,
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
)

import matplotlib.pyplot as plt


def get_roc_curve_info(pred_label, true_label):
    fpr, tpr, thresholds = roc_curve(true_label, pred_label)
    auc_score = roc_auc_score(true_label, pred_label)
    return fpr, tpr, thresholds, auc_score


def gen_roc_fig(
        x,
        y,
        auc_score=None,
        thresholds=None,
        color="darkorange",
        is_precision_recall=False):
    """ Return a matplotlib figure object by plotting ROC curve.
    """
    if auc_score is not None:
        label = "ROC curve (area = {auc_score:.3f})".format(
            auc_score=auc_score)
    else:
        label = "ROC curve"

    lw = 2

    fig = plt.figure()
    plt.plot(x, y, color=color, lw=lw, label=label)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")

    return fig


def plot_roc_curve(
        pred_label, true_label, **kwargs):
    fpr, tpr, thresholds, auc_score = get_roc_curve_info(
        pred_label, true_label)

    return gen_roc_fig(fpr, tpr, auc_score, **kwargs)
