# -*- coding: utf-8 -*-
import numpy as np

from ..base import BaseEstimator
from ..base import TransformerMixin
from ..base import clone


class VotingImputer(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    estimators : list of (string, estimator) tuples
    """
    def __init__(self, estimators, missing_values="NaN"):
        self.estimators = estimators
        # Нужно для того, чтобы построить маску.
        self.missing_values = missing_values

    def fit(self, X, y=None):
        if self.estimators is None or len(self.estimators) == 0:
            raise AttributeError('Invalid `estimators` attribute, `estimators`'
                                 ' should be a list of (string, estimator)'
                                 ' tuples')
        self.estimators_ = []
        for name, clf in self.estimators:
            fitted_clf = clone(clf).fit(X, y)
            self.estimators_.append(fitted_clf)

        return self

    def transform(self, X):
        if self.missing_values == "NaN" or np.isnan(self.missing_values):
            mask = np.isnan(X)
        else:
            mask = X == self.missing_values

        Xs = [imputer.transform(X) for imputer in self.estimators_]
        X_imputed = reduce(lambda res, X_: res + X_, Xs, np.zeros((X.shape[0], X.shape[1]))) / len(self.estimators_)

        coordinates = np.where(mask.transpose())[::-1]
        X[coordinates] = X_imputed[coordinates]

        return X
