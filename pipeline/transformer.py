from pandas import DataFrame

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer, LabelEncoder


class DummyTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class FixedFeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, names):
        self.names = names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.names].values


def BaseCustomTransformer(BaseEstimator, TransformerMixin):
    ENCODER = None

    def fit(self, x, y=0):
        self.ENCODER.fit(x)
        return self

    def transform(self, x, y=0):
        return self.ENCODER.transform(x)

    def fit_transform(self, x, y=0):
        return self.ENCODER.fit_transform(x)


class CustomLabelBinarizer(BaseCustomTransformer):
    ENCODER = LabelBinarizer


class CustomLabelEncoder(BaseCustomTransformer):
    ENCODER = LabelEncoder


# -- problem specific transformer --
class IntegerValuedFeatureSelector(BaseEstimator, TransformerMixin):

    @classmethod
    def _return_feat_index(clf, X, feat_names=None):
        assert isinstance(X, DataFrame) or feat_names is not None, \
            "either X must be DataFrame or feat_names must be provided."

        all_feat = feat_names if feat_names is not None else X.columns.tolist()
        return [idx for idx, feat in enumerate(all_feat)
                if ("_num_" in feat) or feat.startswith("num_")]

    def fit(self, X, y=None):
        return self

    def transform(self, X, feat_names=None):
        num_feat_idx = self.__class__._return_feat_index(X, feat_names)
        return X.iloc[:, num_feat_idx]


class FloatValuedFeatureSelector(IntegerValuedFeatureSelector):

    @classmethod
    def _return_feat_index(clf, X, feat_names=None):
        assert isinstance(X, DataFrame) or feat_names is not None, \
            "either X must be DataFrame or feat_names must be provided."

        all_feat = feat_names if feat_names is not None else X.columns.tolist()
        return [idx for idx, feat in enumerate(all_feat)
                if (("_frac_" in feat) or
                    feat.startswith("frac_") or
                    ('_rate_' in feat) or
                    feat.startswith('rate_'))]
