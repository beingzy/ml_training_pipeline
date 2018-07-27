from collections import namedtuple

from sklearn.metrics import (
    # regression evaluation metrics
    explained_variance_score,
    mean_absolute_error,
    median_absolute_error,
    mean_squared_error,
    # classification evaluation metrics
    roc_auc_score,
    average_precision_score,
    log_loss,
)


def compute_evaluation_metrics(y_true, y_pred, metrics):
    """
    """
    return dict((metric.__name__, metric(y_true, y_pred))
        for metric in metrics)


def calibration(y_true, pred_prob):
    return sum(pred_prob)/sum(y_true)


class BaseEvaluator:
    METRICS = []

    @classmethod
    def compute_evaluation_metrics(clf, y_true, y_pred):
        assert len(clf.METRICS) > 0, "must define at least one" \
            + " function for metric evaluation metrics"
        return compute_evaluation_metrics(
            y_true, y_pred, clf.METRICS)


class BinaryClassificationEvaluator(BaseEvaluator):
    METRICS = [
        roc_auc_score,
        average_precision_score,
        log_loss,
        calibration,
    ]


class RegressionEvaluator(BaseEvaluator):
    METRICS = [
        explained_variance_score,
        mean_absolute_error,
        median_absolute_error,
        mean_squared_error,
    ]
