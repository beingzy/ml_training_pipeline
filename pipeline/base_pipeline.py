# coding=utf-8
"""
"""
import os
import logging

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from pipeline.evaluator import (
    BaseEvaluator,
    BinaryClassificationEvaluator,
    RegressionEvaluator,
)


class BaseModelingPipeline:

    def __init__(
            self,
            train_x,
            train_y,
            test_x=None,
            test_y=None,
            *args,
            **kwargs):
        self.set_train_data(train_x, train_y)
        self.set_test_data(test_x, test_y)
        self._test_size = kwargs.get("test_size", 0.3)
        self._random_seed = kwargs.get("random_seed", None)

    def set_train_data(self, train_x, train_y):
        self._train_x = train_x
        self._train_y = train_y

    def set_test_data(self, test_x, test_y):
        self._test_x = test_x
        self._test_y = test_y

    def set_feature_transformer(self, steps):
        """transformer: transform class or a list of transform classes
        """
        raise NotImplementedError

    def set_estimator(self, estimator):
        if self.steps is not None:
            self.steps = self.steps.append(estimator)
        else:
            self.steps = [estimator]

        return self

    def set_calibrator(self, calibrator):
        # assert isinstance(calibrator)
        assert len(self.steps), "Please, ensure there is at least " + \
            "one estimator which had been inserted into the pipeline."

        self.steps = self.steps.append(calibrator)
        return self

    def set_label_transformer(self):
        raise NotImplementedError

    def set_evaluator(self):
        return NotImplementedError

    def set_hyperparameter_tuner(self):
        return NotImplementedError

    def build_pipeline(self, *args, **kwargs):
        raise NotImplementedError

    def split_train_test(self):
        if self._test_x is not None:
            return self

        train_x, test_x, train_y, test_y = train_test_split(
            self._train_x,
            self._train_y,
            test_size=self._test_size,
            random_state=self._random_seed,
            shuffle=True)

        self.set_train_data(train_x, train_y)
        self.set_test_data(test_x, test_y)

    def fit(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def model_selection(self):
        raise NotImplementedError

    def tune_hyperparameter(self):
        raise NotImplementedError

    def summary_report(self):
        raise NotImplementedError
