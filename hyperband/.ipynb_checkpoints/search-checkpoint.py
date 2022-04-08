"""
---------------------------------------------------------------------------
This is a forked version on https://github.com/louisowen6/scikit-hyperband
from the original repo https://github.com/thuijskens/scikit-hyperband.

Several minor changes is made to make this repo compatible with the newer
version of Scikit-learn (version 1.0.1 or above).

Changes made:
- remove `iid` parameter
- remove `self.multimetric_`, set refit_metric = 'score' as default
- remove positional argument for super().fit(X, y=y, groups=groups, **fit_params)
- raise issue if resource is part of the param_distributions
---------------------------------------------------------------------------

=========
Hyperband
=========

This module contains a scikit-learn compatible implementation of the hyperband
algorithm[^1].

Compared to the civismlext implementation, this supports multimetric scoring,
and the option to turn the last round of hyperband (the randomized search
round) off.

References
----------

.. [1] Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A. and Talwalkar, A.,
   2017. Hyperband: A novel bandit-based approach to hyperparameter
   optimization. The Journal of Machine Learning Research, 18(1),
   pp.6765-6816.

"""
import copy

import numpy as np
from scipy.stats import rankdata

from sklearn.utils import check_random_state
from sklearn.model_selection._search import BaseSearchCV, ParameterSampler
from sklearn.base import is_classifier
from sklearn.model_selection._split import check_cv


__all__ = ['HyperbandSearchCV']


class HyperbandSearchCV(BaseSearchCV):
    """Hyperband search on hyper parameters.

    HyperbandSearchCV implements a ``fit`` and a ``score`` method.
    It also implements ``predict``, ``predict_proba``, ``decision_function``,
    ``transform`` and ``inverse_transform`` if they are implemented in the
    estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings using the hyperband
    algorithm [1]_ .

    If all parameters are presented as a list,
    sampling without replacement is performed. If at least one parameter
    is given as a distribution, sampling with replacement is used.
    It is highly recommended to use continuous distributions for continuous
    parameters.

    Read more in the scikit-learn `User Guide
    <http://scikit-learn.org/stable/modules/grid_search.html#randomized-parameter-search>`_.

    Parameters
    ----------
    estimator : estimator object.
        A object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_distributions : dict
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.

    resource_param : str, default='n_estimators'
        The name of the cost parameter for the estimator ``estimator``
        to be fitted. Typically, this is the number of decision trees
        ``n_estimators`` in an ensemble or the number of iterations
        for estimators trained with stochastic gradient descent.

    eta : float, default=3
        The inverse of the proportion of configurations that are discarded
        in each round of hyperband.

    min_iter : int, default=1
        The minimum amount of resource that should be allocated to the cost
        parameter ``resource_param`` for a single configuration of the
        hyperparameters.

    max_iter : int, default=81
        The maximum amount of resource that can be allocated to the cost
        parameter ``resource_param`` for a single configuration of the
        hyperparameters.

    skip_last : int, default=0
        The number of last rounds to skip. For example, this can be used
        to skip the last round of hyperband, which is standard randomized
        search. It can also be used to inspect intermediate results,
        although warm-starting HyperbandSearchCV is not supported.

    scoring : string, callable, list/tuple, dict or None, default: None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.

        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.

        See :ref:`multimetric_grid_search` for an example.

        If None, the estimator's default scorer (if available) is used.

    n_jobs : int, default=1
        Number of jobs to run in parallel.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`sklearn.model_selection.StratifiedKFold`
        is used. In all other cases, :class:`sklearn.model_selection.KFold` is used.

        Refer `User Guide <http://scikit-learn.org/stable/modules/cross_validation.html>`_
        for the various cross-validation strategies that can be used here.

    refit : boolean, or string default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

        For multiple metric evaluation, this needs to be a string denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``HyperbandSearchCV`` instance.

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_parameters_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.

        See ``scoring`` parameter to know more about multiple metric
        evaluation.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    random_state : int, RandomState instance or None, optional, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : boolean, optional, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +------------+-----------+------------+-----------------+---+---------+
        |param_kernel|param_gamma|param_degree|split0_test_score|...|rank_t...|
        +============+===========+============+=================+===+=========+
        |  'poly'    |     --    |      2     |        0.8      |...|    2    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'poly'    |     --    |      3     |        0.7      |...|    4    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.1   |     --     |        0.8      |...|    3    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.2   |     --     |        0.9      |...|    1    |
        +------------+-----------+------------+-----------------+---+---------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel': masked_array(data = ['poly', 'poly', 'rbf', 'rbf'],
                                         mask = [False False False False]...)
            'param_gamma': masked_array(data = [-- -- 0.1 0.2],
                                        mask = [ True  True False False]...),
            'param_degree': masked_array(data = [2.0 3.0 -- --],
                                         mask = [False False  True  True]...),
            'split0_test_score'  : [0.8, 0.7, 0.8, 0.9],
            'split1_test_score'  : [0.82, 0.5, 0.7, 0.78],
            'mean_test_score'    : [0.81, 0.60, 0.75, 0.82],
            'std_test_score'     : [0.02, 0.01, 0.03, 0.03],
            'rank_test_score'    : [2, 4, 3, 1],
            'split0_train_score' : [0.8, 0.9, 0.7],
            'split1_train_score' : [0.82, 0.5, 0.7],
            'mean_train_score'   : [0.81, 0.7, 0.7],
            'std_train_score'    : [0.03, 0.03, 0.04],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
            'mean_score_time'    : [0.007, 0.06, 0.04, 0.04],
            'std_score_time'     : [0.001, 0.002, 0.003, 0.005],
            'params'             : [{'kernel': 'poly', 'degree': 2}, ...],
            }

        NOTE

        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. ('split0_test_precision', 'mean_train_precision' etc.)

    best_estimator_ : estimator or dict
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

        For multi-metric evaluation, this attribute is present only if
        ``refit`` is specified.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_estimator.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    References
    ----------

    .. [1] Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A. and Talwalkar, A.,
           2017. Hyperband: A novel bandit-based approach to hyperparameter
           optimization. The Journal of Machine Learning Research, 18(1),
           pp.6765-6816.

    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    parameter setting (and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    See Also
    --------
    :class:`sklearn.model_selection.GridSearchCV`:
        Does exhaustive search over a grid of parameters.

    :class:`sklearn.model_selection.ParameterSampler`:
        A generator over parameter settings, constructed from
        param_distributions.

    """
    def __init__(self, estimator, param_distributions,
                 resource_param='n_estimators', eta=3, min_iter=1,
                 max_iter=81, skip_last=0, scoring=None, n_jobs=1,
                 refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs', random_state=None,
                 error_score='raise', return_train_score=False):
        self.param_distributions = param_distributions
        self.resource_param = resource_param
        self.eta = eta
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.skip_last = skip_last
        self.random_state = random_state
        self.estimator = estimator
        self.cv = cv

        super(HyperbandSearchCV, self).__init__(
            estimator=estimator, scoring=scoring, n_jobs=n_jobs,
            refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)

    def _run_search(self, evaluate_candidates):
        self._validate_input()

        s_max = int(np.floor(np.log(self.max_iter / self.min_iter) / np.log(self.eta)))
        B = (s_max + 1) * self.max_iter

        refit_metric = 'score'
        random_state = check_random_state(self.random_state)

        if self.skip_last > s_max:
            raise ValueError('skip_last is higher than the total number of rounds')
        
        self.n_resources_ = []
        self.n_candidates_ = []
        self.n_trials_ = []
        for round_index, s in enumerate(reversed(range(s_max + 1))):
            n = int(np.ceil(int(B / self.max_iter / (s + 1)) * np.power(self.eta, s)))

            # initial number of iterations per config
            r = self.max_iter / np.power(self.eta, s)
            configurations = list(ParameterSampler(param_distributions=self.param_distributions,
                                                   n_iter=n,
                                                   random_state=random_state))

            if self.verbose > 0:
                print('Starting bracket {0} (out of {1}) of hyperband'
                      .format(round_index + 1, s_max + 1))
            
            num_SH_trials = (s + 1) - self.skip_last
            self.n_trials_.append(num_SH_trials)

            self.n_resources_SH = []
            self.n_candidates_SH = []
            for i in range(num_SH_trials):

                n_configs = np.floor(n / np.power(self.eta, i))  # n_i
                n_iterations = int(r * np.power(self.eta, i))  # r_i
                n_to_keep = int(np.floor(n_configs / self.eta))

                self.n_resources_SH.append(n_iterations)
                self.n_candidates_SH.append(n_configs)

                if self.verbose > 0:
                    msg = ('Starting successive halving iteration {0} out of'
                           ' {1}. Fitting {2} configurations, with'
                           ' resource_param {3} set to {4}')

                    if n_to_keep > 0:
                        msg += ', and keeping the best {5} configurations.'

                    msg = msg.format(i + 1, s + 1, len(configurations),
                                     self.resource_param, n_iterations,
                                     n_to_keep)
                    print(msg)

                # Set the cost parameter for every configuration
				# Need copy so that the n_iterations of next iteration does
                # not overwrite
                parameters = copy.deepcopy(configurations)
                for configuration in parameters:
                    configuration[self.resource_param] = n_iterations
                cv = self._checked_cv_orig

                more_results = {"SH_iter": [i] * int(n_configs),
                                "SH_n_resources": [n_iterations] * int(n_configs),
                                }

                results = evaluate_candidates(parameters,cv,more_results=more_results)

                if n_to_keep > 0:
                    top_configurations = [x for _, x in sorted(zip(results['rank_test_%s' % refit_metric],
                                                                   results['params']),
                                                               key=lambda x: x[0])]

                    configurations = top_configurations[:n_to_keep]

            if self.skip_last > 0:
                print('Skipping the last {0} successive halving iterations'
                      .format(self.skip_last))

            self.n_resources_.append(self.n_resources_SH)
            self.n_candidates_.append(self.n_candidates_SH)

    def fit(self, X, y=None, groups=None, **fit_params):
        """Run fit with all sets of parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator
        """
        self._checked_cv_orig = check_cv(
            self.cv, y, classifier=is_classifier(self.estimator)
        )

        super().fit(X, y=y, groups=groups, **fit_params)

        
        s_max = int(np.floor(np.log(self.max_iter / self.min_iter) / np.log(self.eta)))
        B = (s_max + 1) * self.max_iter

        brackets = []
        for round_index, s in enumerate(reversed(range(s_max + 1))):
            n = int(np.ceil(int(B / self.max_iter / (s + 1)) * np.power(self.eta, s)))
            n_configs = int(sum([np.floor(n / np.power(self.eta, i))
                                 for i in range((s + 1) - self.skip_last)]))
            bracket = (round_index + 1) * np.ones(n_configs)
            brackets.append(bracket)

        self.cv_results_['hyperband_bracket'] = np.hstack(brackets)

        return self

    def _validate_input(self):
        if not isinstance(self.min_iter, int) or self.min_iter <= 0:
            raise ValueError('min_iter should be a positive integer, got %s' %
                             self.min_iter)

        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError('max_iter should be a positive integer, got %s' %
                             self.max_iter)

        if self.max_iter < self.min_iter:
            raise ValueError('max_iter should be bigger than min_iter, got'
                             'max_iter=%d and min_iter=%d' % (self.max_iter,
                                                              self.min_iter))

        if not isinstance(self.skip_last, int) or self.skip_last < 0:
            raise ValueError('skip_last should be an integer, got %s' %
                             self.skip_last)

        if not isinstance(self.eta, int) or not self.eta > 1:
            raise ValueError('eta should be a positive integer, got %s' %
                             self.eta)

        if self.resource_param not in self.estimator.get_params().keys():
            raise ValueError('resource_param is set to %s, but base_estimator %s '
                             'does not have a parameter with that name' %
                             (self.resource_param,
                              self.estimator.__class__.__name__))

        if any(self.resource_param in candidate for candidate in self.param_distributions):
            # Can only check this now since we need the candidates list
            raise ValueError(
                f"Cannot use parameter {self.resource_param} as the resource since "
                "it is part of the searched parameters."
            )
