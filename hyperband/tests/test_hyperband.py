from nose.tools import raises
from hyperband import HyperbandSearchCV

from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint

from sklearn.datasets import load_digits
from sklearn.utils import check_random_state


def setup():
    model = RandomForestClassifier()
    rng = check_random_state(42)
    param_dist = {'max_depth': [3, None],
                  'max_features': sp_randint(1, 11),
                  'min_samples_split': sp_randint(2, 11),
                  'bootstrap': [True, False],
                  'criterion': ['gini', 'entropy']}
    
    digits = load_digits()
    X, y = digits.data, digits.target

    return model, param_dist, X, y, rng


def test_multimetric_hyperband():
    model, param_dist, X, y, rng = setup()

    # multimetric scoring is only supported for 1-D classification
    first_label = (y == 1)
    y[first_label] = 1
    y[~first_label] = 0

    multimetric = [
        'roc_auc',
        'accuracy'
    ]

    search = HyperbandSearchCV(model, param_dist, refit='roc_auc', scoring=multimetric,
                               random_state=rng)
    search.fit(X, y)

    assert('mean_test_roc_auc' in search.cv_results_.keys())
    assert('mean_test_accuracy' in search.cv_results_.keys())
    assert (len(search.cv_results_['hyperband_bracket']) == 187)


def test_min_resource_param():
    model, param_dist, X, y, rng = setup()
    search = HyperbandSearchCV(model, param_dist, min_iter=3, random_state=rng,
                               verbose=1)
    search.fit(X, y)

    assert(search.cv_results_['param_n_estimators'].data.min() == 3)


@raises(ValueError)
def test_skip_last_raise():
    model, param_dist, X, y, rng = setup()
    search = HyperbandSearchCV(model, param_dist, skip_last=10, random_state=rng)
    search.fit(X, y)


def test_skip_last():
    model, param_dist, X, y, rng = setup()
    search = HyperbandSearchCV(model, param_dist, skip_last=1, random_state=rng)
    search.fit(X, y)

    # 177 Because in every round the last search is dropped
    # 187 - (1 + 1 + 1 + 2 + 5)
    assert (len(search.cv_results_['hyperband_bracket']) == 177)
