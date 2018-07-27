from copy import deepcopy
from datetime import datetime
from collections import namedtuple
from abc import (
    ABC,
    abstractclassmethod,
    abstractmethod,
)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
)


def feat_concat(data_dict, namespaces=None):
    """create a wide and flat data set by extracting defined
       namespaces from packaged data
    """
    if namespaces is None:
        namespaces = data_dict.keys()

    subset_list = []
    for ii, name in enumerate(namespaces):
        subset_list.append(data_dict.get(name))

    return np.column_stack(subset_list)


class BaseTrainerAndEvaluatorPipeline(ABC):

    def __init__(self, train_dict, test_dict, train_label, test_label):
        self._train_dict = train_dict
        self._test_dict = test_dict
        self._train_label = train_label
        self._test_label = test_label

    @abstractclassmethod
    def _evaluator(clf):
        raise NotImplementedError

    @abstractclassmethod
    def _trainer(clf):
        raise NotImplementedError

    @abstractmethod
    def sampler_and_labeler(self):
        raise NotImplementedError

    @abstractmethod
    def train_and_evaluate(self):
        raise NotImplementedError


class ClassifierTrainerAndEvaluatorPipeline(
        BaseTrainerAndEvaluatorPipeline):

    def __init__(self, train_dict, test_dict, train_label, test_label):
        self._train_dict = train_dict
        self._test_dict = test_dict
        self._train_label = train_label
        self._test_label = test_label

    @classmethod
    def _evaluator(clf, model, x, y):
        proba = model.predict_proba(x)[:, 1]

        fpr, tpr, _ = roc_curve(y, proba)
        auc_score = auc(fpr, tpr, reorder=True)

        ap = average_precision_score(y, proba)
        calibration = sum(proba) / sum(y)
        obs_positive_rate = sum(y) * 1.0 / len(y)

        EvalResults = namedtuple(
            'EvalResults',
            ['roc_auc_score',
             'average_precision',
             'calibration',
             'observed_positive_rate'])

        return EvalResults(
            auc_score, ap, calibration, obs_positive_rate)

    @classmethod
    def _trainer(clf, model, x, y):
        model.fit(x, y)
        return model

    def sampler_and_labeler(self, namespaces, label):
        train_x = feat_concat(self._train_dict, namespaces)
        test_x = feat_concat(self._test_dict, namespaces)
        train_y = self._train_label[label]
        test_y = self._test_label[label]
        return train_x, test_x, train_y, test_y

    def train_and_evaluate(self, model, namespaces, label):
        """return trained model and performance metrics evaluated
           on both train and test data sets.
        """
        model = deepcopy(model)

        train_x, test_x, train_y, test_y = \
            self.sampler_and_labeler(namespaces, label)

        model = self.__class__._trainer(model, train_x, train_y)

        train_eval = self.__class__._evaluator(model, train_x, train_y)
        test_eval = self.__class__._evaluator(model, test_x, test_y)
        return model, train_eval, test_eval