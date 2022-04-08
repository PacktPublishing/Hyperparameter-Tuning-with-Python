from nose.tools import raises
from hyperband import HyperbandSearchCV

from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint


def setup():
    model = RandomForestClassifier()
    param_dist = {"max_depth": [3, None],
                  "max_features": sp_randint(1, 11),
                  "min_samples_split": sp_randint(2, 11),
                  "min_samples_leaf": sp_randint(1, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    return model, param_dist


@raises(ValueError)
def test_check_min_iter():
    model, param_dist = setup()
    HyperbandSearchCV(model, param_dist, min_iter=-1)._validate_input()


@raises(ValueError)
def test_check_max_iter():
    model, param_dist = setup()
    HyperbandSearchCV(model, param_dist, max_iter=-1)._validate_input()


@raises(ValueError)
def test_check_min_iter_smaller_max_iter():
    model, param_dist = setup()
    HyperbandSearchCV(model, param_dist, min_iter=30, max_iter=15)._validate_input()


@raises(ValueError)
def test_check_skip_last():
    model, param_dist = setup()
    HyperbandSearchCV(model, param_dist, skip_last=-1)._validate_input()


@raises(ValueError)
def test_check_eta():
    model, param_dist = setup()
    HyperbandSearchCV(model, param_dist, eta=0)._validate_input()


@raises(ValueError)
def test_check_resource_param():
    model, param_dist = setup()
    HyperbandSearchCV(model, param_dist, resource_param='wrong_name')._validate_input()
