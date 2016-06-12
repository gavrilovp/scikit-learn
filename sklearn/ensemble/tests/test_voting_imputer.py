from sklearn.utils.testing import assert_raise_message
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
