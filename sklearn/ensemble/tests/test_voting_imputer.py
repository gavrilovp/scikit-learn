from sklearn.utils.testing import assert_raise_message
from sklearn.exceptions import NotFittedError
from sklearn import datasets
from sklearn.preprocessing import Imputer
from sklearn.ensemble.voting_imputer import VotingImputer


iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target


def test_estimator_init():
    voting_imputer = VotingImputer([])
    msg = ('Invalid `estimators` attribute, `estimators` should be'
           ' a list of (string, estimator) tuples')
    assert_raise_message(AttributeError, msg, voting_imputer.fit, X, y)

    voting_imputer = VotingImputer([("mean", Imputer(strategy="mean"))])
    voting_imputer.fit(X, y)


def test_notfitted():
    voting_imputer = VotingImputer([("mean", Imputer(strategy="mean"))])
    msg = ("This VotingImputer instance is not fitted yet. Call 'fit' with"
           " appropriate arguments before using this method.")
    assert_raise_message(NotFittedError, msg, voting_imputer.transform, X)
