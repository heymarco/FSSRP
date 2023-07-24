from __future__ import annotations

import collections
import itertools
import math
import random

import numpy as np
import pandas as pd

from river import base
from river.drift import ADWIN
from river.ensemble.streaming_random_patches import BaseSRPRegressor, BaseSRPClassifier, BaseSRPEstimator
from river.metrics import MAE, Accuracy
from river.metrics.base import ClassificationMetric, Metric, RegressionMetric
from river.tree import HoeffdingTreeClassifier, HoeffdingTreeRegressor
from river.utils.random import poisson

from afs import get_alternative_features_classification, get_alternative_features_regression, get_alternative_features_baseline


class BaseFSSRPEnsemble(base.Wrapper, base.Ensemble):
    """Base class for the sRP ensemble family"""

    _TRAIN_RANDOM_SUBSPACES = "subspaces"
    _TRAIN_RESAMPLING = "resampling"
    _TRAIN_RANDOM_PATCHES = "patches"

    _FEATURES_SQRT = "sqrt"
    _FEATURES_SQRT_INV = "rmsqrt"

    _VALID_TRAINING_METHODS = {
        _TRAIN_RANDOM_PATCHES,
        _TRAIN_RESAMPLING,
        _TRAIN_RESAMPLING,
    }

    @property
    def _min_number_of_models(self):
        return 0

    def __init__(
            self,
            model: base.Estimator | None,
            n_models: int,
            fs_batch_size: int,
            fs_bootstrap_size: float | str,
            fs_function: callable,
            subspace_size: int | float | str,
            training_method: str = "patches",
            lam: float = 6.0,
            drift_detector: base.DriftDetector | None = None,
            warning_detector: base.DriftDetector | None = None,
            disable_detector: str = "off",
            disable_weighted_vote: bool = False,
            seed: int | None = None,
            metric: Metric | None = None,
    ):
        # List of models is properly initialized later
        super().__init__([])  # type: ignore
        self.model = model  # Not restricted to a specific base estimator.
        self.fs_batch_size = fs_batch_size
        self.fs_bootstrap_size = fs_bootstrap_size
        self.n_models = n_models
        self.subspace_size = subspace_size
        self.training_method = training_method
        self.lam = lam
        self.drift_detector = drift_detector
        self.warning_detector = warning_detector
        self.disable_weighted_vote = disable_weighted_vote
        self.disable_detector = disable_detector
        self.metric = metric
        self.seed = seed
        self._rng = random.Random(self.seed)

        self._n_samples_seen = 0
        self._subspaces: list = []

        # defined by extended classes
        self._base_learner_class: BaseSRPClassifier | BaseSRPRegressor | None = None

        # feature selection related
        self._fs_samples_x: pd.DataFrame = None
        self._fs_samples_y: list = []
        self._uses_random_subspaces = True
        self._feature_selection_function = fs_function

    @property
    def _wrapped_model(self):
        return self.model

    @classmethod
    def _unit_test_params(cls):
        yield {"n_models": 3, "seed": 42}

    def _unit_test_skips(self):  # noqa
        return {
            "check_shuffle_features_no_impact",
            "check_emerging_features",
            "check_disappearing_features",
        }

    def learn_one(self, x: dict, y: base.typing.Target, **kwargs):
        self._n_samples_seen += 1

        if not self:
            self._init_ensemble(list(x.keys()))

        self._update_samples_for_fs(x, y)
        if self._uses_random_subspaces and len(self._fs_samples_y) == self.fs_batch_size:
            self._subspaces = self._select_features()
            self._retrain_ensemble()
            self._fs_samples_y = None
            self._fs_samples_y = []
            self._uses_random_subspaces = False

        for model in self:
            # Get prediction for instance
            y_pred = model.predict_one(x)

            # Update performance evaluator
            if y_pred is not None:
                model.metric.update(y_true=y, y_pred=y_pred)

            # Train using random subspaces without resampling,
            # i.e. all instances are used for training.
            if self.training_method == self._TRAIN_RANDOM_SUBSPACES:
                k = 1
            # Train using random patches or resampling,
            # thus we simulate online bagging with Poisson(lambda=...)
            else:
                k = poisson(rate=self.lam, rng=self._rng)
                if k == 0:
                    continue
            model.learn_one(x=x, y=y, sample_weight=k, n_samples_seen=self._n_samples_seen)

        return self

    def _generate_subspaces(self, features: list):
        n_features = len(features)

        self._subspaces = [None] * self.n_models

        if self.training_method != self._TRAIN_RESAMPLING:
            # Set subspaces - This only applies to subspaces and random patches options

            # 1. Calculate the number of features k
            if isinstance(self.subspace_size, float) and 0.0 < self.subspace_size <= 1:
                k = self.subspace_size
                percent = (1.0 + k) / 1.0 if k < 0 else k
                k = round(n_features * percent)
                if k < 2:
                    k = round(n_features * percent) + 1
            elif isinstance(self.subspace_size, int) and self.subspace_size > 2:
                # k is a fixed number of features
                k = self.subspace_size
            elif self.subspace_size == self._FEATURES_SQRT:
                # k is sqrt(M)+1
                k = round(math.sqrt(n_features)) + 1
            elif self.subspace_size == self._FEATURES_SQRT_INV:
                # k is M-(sqrt(M)+1)
                k = n_features - round(math.sqrt(n_features)) + 1
            else:
                raise ValueError(
                    f"Invalid subspace_size: {self.subspace_size}.\n"
                    f"Valid options are: int [2, M], float (0., 1.],"
                    f" {self._FEATURES_SQRT}, {self._FEATURES_SQRT_INV}"
                )
            if k < 0:
                # k is negative, calculate M - k
                k = n_features + k

            # 2. Generate subspaces. The subspaces is a 2D array of shape
            # (n_estimators, k) where each row contains the k-feature indices
            # to be used by each estimator.
            if k != 0 and k < n_features:
                # For low dimensionality it is better to avoid more than
                # 1 classifier with the same subspace, thus we generate all
                # possible combinations of subsets of features and select
                # without replacement.
                # n_features is the total number of features and k is the
                # actual size of a subspace.
                if n_features <= 20 or k < 2:
                    if k == 1 and n_features > 2:
                        k = 2
                    # Generate n_models subspaces from all possible
                    # feature combinations of size k
                    self._subspaces = []
                    for i, combination in enumerate(
                            itertools.cycle(itertools.combinations(features, k))
                    ):
                        if i == self.n_models:
                            break
                        self._subspaces.append(list(combination))
                # For high dimensionality we can't generate all combinations
                # as it is too expensive (memory). On top of that, the chance
                # of repeating a subspace is lower, so we randomly generate
                # subspaces without worrying about repetitions.
                else:
                    self._subspaces = [
                        random_subspace(all_features=features, k=k, rng=self._rng)
                        for _ in range(self.n_models)
                    ]
            else:
                # k == 0 or k > n_features (subspace size is larger than the
                # number of features), then default to re-sampling
                self.training_method = self._TRAIN_RESAMPLING

    def _init_ensemble(self, features: list):
        self._generate_subspaces(features=features)
        subspace_indexes = list(
            range(self.n_models)
        )  # For matching subspaces with ensemble members
        if (
                self.training_method == self._TRAIN_RANDOM_PATCHES
                or self.training_method == self._TRAIN_RANDOM_SUBSPACES
        ):
            # Shuffle indexes
            self._rng.shuffle(subspace_indexes)

        # Initialize the ensemble
        for i in range(self.n_models):
            # If self.training_method == self._TRAIN_RESAMPLING then subspace is None
            subspace = self._subspaces[subspace_indexes[i]]
            self.append(
                self._base_learner_class(  # type: ignore
                    idx_original=i,
                    model=self.model,
                    metric=self.metric,
                    created_on=self._n_samples_seen,
                    drift_detector=self.drift_detector,
                    warning_detector=self.warning_detector,
                    is_background_learner=False,
                    rng=self._rng,
                    features=subspace,
                )
            )

    def _retrain_ensemble(self):
        self.data = []
        subspace_indexes = list(
            range(len(self._subspaces))
        )  # For matching subspaces with ensemble members
        if (
                self.training_method == self._TRAIN_RANDOM_PATCHES
                or self.training_method == self._TRAIN_RANDOM_SUBSPACES
        ):
            # Shuffle indexes
            self._rng.shuffle(subspace_indexes)

        # Initialize the ensemble
        for i in range(len(self._subspaces)):
            # If self.training_method == self._TRAIN_RESAMPLING then subspace is None
            subspace = self._subspaces[subspace_indexes[i]]
            self.append(
                self._base_learner_class(  # type: ignore
                    idx_original=i,
                    model=self.model,
                    metric=self.metric,
                    created_on=self._n_samples_seen,
                    drift_detector=self.drift_detector,
                    warning_detector=self.warning_detector,
                    is_background_learner=False,
                    rng=self._rng,
                    features=subspace,
                )
            )

        for (row_index, row), label in zip(self._fs_samples_x.iterrows(), self._fs_samples_y):
            for model in self:
                # Train using random subspaces without resampling,
                # i.e. all instances are used for training.
                if self.training_method == self._TRAIN_RANDOM_SUBSPACES:
                    k = 1
                # Train using random patches or resampling,
                # thus we simulate online bagging with Poisson(lambda=...)
                else:
                    k = poisson(rate=self.lam, rng=self._rng)
                    if k == 0:
                        continue
                model.learn_one(x=row.to_dict(), y=label,
                                sample_weight=k, n_samples_seen=row_index)

    def reset(self):
        self.data = []
        self._n_samples_seen = 0
        self._rng = random.Random(self.seed)

    def _update_samples_for_fs(self, x: dict, y: base.typing.Target):
        self._fs_samples_y.append(y)
        row = pd.DataFrame(x, index=[0])
        if self._fs_samples_x is None:
            self._fs_samples_x = row
        else:
            self._fs_samples_x = pd.concat([self._fs_samples_x, row], ignore_index=True)

    def _select_features(self) -> list:
        dims = len(self._fs_samples_x.columns)
        relative_size = self.subspace_size
        if self.subspace_size == "sqrt":
            relative_size = np.sqrt(dims) / dims
        elif self.subspace_size > 1:
            relative_size = self.subspace_size / dims
        alt_feature_sets = self._feature_selection_function(data=self._fs_samples_x,
                                                            labels=np.array(self._fs_samples_y),
                                                            n_components=self.n_models,
                                                            relative_size=relative_size,
                                                            bootstrap_size=self.fs_bootstrap_size,
                                                            rng=self._rng)
        alt_feature_sets = unique_featuresets(alt_feature_sets)
        if len(alt_feature_sets) == 0:  # if alternative feature selection did not produce results, fall back  to random
            abs_subspace_size = int(relative_size * dims)
            alt_feature_sets = [
                random_subspace(self._fs_samples_x.columns.tolist(), k=abs_subspace_size, rng=self._rng)
                for _ in range(self.n_models)
            ]
        return alt_feature_sets


def random_subspace(all_features: list, k: int, rng: random.Random):
    """Utility function to generate a random feature subspace of length k

    Parameters
    ----------
    all_features
        List of possible features to select from.
    k
        Subspace length.
    rng
        Random number generator (initialized).
    """
    return rng.sample(all_features, k=k)


def unique_featuresets(all_featuresets: list) -> list:
    """
    Removes empty feature sets and duplicates
    :param all_featuresets: the list of feature sets to filter
    :return: the unique and non-empty feature sets
    """
    if len(all_featuresets) == 0:
        return all_featuresets
    all_featuresets = [np.sort(fs).tolist() for fs in all_featuresets]
    unique_fs = []
    for fs in all_featuresets:
        if fs not in unique_fs and len(fs) > 0:
            unique_fs.append(fs)
    return unique_fs


class FSSRPClassifier(BaseFSSRPEnsemble, base.Classifier):
    """Streaming Random Patches ensemble classifier.

    The Streaming Random Patches (SRP) [^1] is an ensemble method that
    simulates bagging or random subspaces. The default algorithm uses both
    bagging and random subspaces, namely Random Patches. The default base
    estimator is a Hoeffding Tree, but other base estimators can be used
    (differently from random forest variations).

    Parameters
    ----------
    model
        The base estimator.
    n_models
        Number of members in the ensemble.
    subspace_size
        Number of features per subset for each classifier where `M` is the
        total number of features.<br/>
        A negative value means `M - subspace_size`.<br/>
        Only applies when using random subspaces or random patches.<br/>
        * If `int` indicates the number of features to use. Valid range [2, M]. <br/>
        * If `float` indicates the percentage of features to use, Valid range (0., 1.]. <br/>
        * 'sqrt' - `sqrt(M)+1`<br/>
        * 'rmsqrt' - Residual from `M-(sqrt(M)+1)`
    training_method
        The training method to use.<br/>
        * 'subspaces' - Random subspaces.<br/>
        * 'resampling' - Resampling.<br/>
        * 'patches' - Random patches.
    lam
        Lambda value for resampling.
    drift_detector
        Drift detector.
    warning_detector
        Warning detector.
    disable_detector
        Option to disable drift detectors:<br/>
        * If `'off'`, detectors are enabled.<br/>
        * If `'drift'`, disables concept drift detection and the background learner.<br/>
        * If `'warning'`, disables the background learner and ensemble members are
         reset if drift is detected.<br/>
    disable_weighted_vote
        If True, disables weighted voting.
    seed
        Random number generator seed for reproducibility.
    metric
        The metric to track members performance within the ensemble. This
        implementation assumes that larger values are better when using
        weighted votes.

    Examples
    --------

    >>> from river import ensemble
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river.datasets import synth
    >>> from river import tree

    >>> dataset = synth.ConceptDriftStream(
    ...     seed=42,
    ...     position=500,
    ...     width=50
    ... ).take(1000)

    >>> base_model = tree.HoeffdingTreeClassifier(
    ...     grace_period=50, delta=0.01,
    ...     nominal_attributes=['age', 'car', 'zipcode']
    ... )
    >>> model = ensemble.FSSRPClassifier(
    ...     model=base_model, n_models=3, seed=42,
    ... )

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 72.77%

    Notes
    -----
    This implementation uses `n_models=10` as default given the impact on
    processing time. The optimal number of models depends on the data and
    resources available.

    References
    ----------
    [^1]: Heitor Murilo Gomes, Jesse Read, Albert Bifet.
          Streaming Random Patches for Evolving Data Stream Classification.
          IEEE International Conference on Data Mining (ICDM), 2019.

    """

    def __init__(
            self,
            model: base.Estimator | None = None,
            n_models: int = 10,
            fs_batch_size: int = 100,
            fs_bootstrap_size: float | str = 0.5,
            fs_function: callable = get_alternative_features_classification,
            subspace_size: int | float | str = 0.6,
            training_method: str = "patches",
            lam: int = 6,
            drift_detector: base.DriftDetector | None = None,
            warning_detector: base.DriftDetector | None = None,
            disable_detector: str = "off",
            disable_weighted_vote: bool = False,
            seed: int | None = None,
            metric: ClassificationMetric | None = None,
    ):
        if model is None:
            model = HoeffdingTreeClassifier(grace_period=50, delta=0.01)

        if drift_detector is None:
            drift_detector = ADWIN(delta=1e-5)

        if warning_detector is None:
            warning_detector = ADWIN(delta=1e-4)

        if disable_detector == "off":
            pass
        elif disable_detector == "drift":
            drift_detector = None
            warning_detector = None
        elif disable_detector == "warning":
            warning_detector = None
        else:
            raise AttributeError(
                f"{disable_detector} is not a valid value for disable_detector.\n"
                f"Valid options are: 'off', 'drift', 'warning'"
            )

        if metric is None:
            metric = Accuracy()

        super().__init__(
            model=model,
            n_models=n_models,
            fs_bootstrap_size=fs_bootstrap_size,
            fs_batch_size=fs_batch_size,
            fs_function=fs_function,
            subspace_size=subspace_size,
            training_method=training_method,
            lam=lam,
            drift_detector=drift_detector,
            warning_detector=warning_detector,
            disable_detector=disable_detector,
            disable_weighted_vote=disable_weighted_vote,
            seed=seed,
            metric=metric,

        )

        self._base_learner_class = BaseSRPClassifier  # type: ignore
        self._feature_selection_function = get_alternative_features_baseline

    def predict_proba_one(self, x, **kwargs):
        y_pred = collections.Counter()

        if not self.models:
            self._init_ensemble(features=list(x.keys()))
            return y_pred

        for model in self.models:
            y_proba_temp = model.predict_proba_one(x, **kwargs)
            metric_value = model.metric.get()
            if not self.disable_weighted_vote and metric_value > 0.0:
                y_proba_temp = {k: val * metric_value for k, val in y_proba_temp.items()}
            y_pred.update(y_proba_temp)

        total = sum(y_pred.values())
        if total > 0:
            return {label: proba / total for label, proba in y_pred.items()}
        return y_pred


class FSSRPRegressor(BaseFSSRPEnsemble, base.Regressor):
    """Streaming Random Patches ensemble regressor.

    The Streaming Random Patches [^1] ensemble method for regression trains
    each base learner on a subset of features and instances from the
    original data, namely a random patch. This strategy to enforce
    diverse base models is similar to the one in the random forest,
    yet it is not restricted to using decision trees as base learner.

    This method is an adaptation of [^2] for regression.

    Parameters
    ----------
    model
        The base estimator.
    n_models
        Number of members in the ensemble.
    subspace_size
        Number of features per subset for each classifier where `M` is the
        total number of features.<br/>
        A negative value means `M - subspace_size`.<br/>
        Only applies when using random subspaces or random patches.<br/>
        * If `int` indicates the number of features to use. Valid range [2, M]. <br/>
        * If `float` indicates the percentage of features to use, Valid range (0., 1.]. <br/>
        * 'sqrt' - `sqrt(M)+1`<br/>
        * 'rmsqrt' - Residual from `M-(sqrt(M)+1)`
    training_method
        The training method to use.<br/>
        * 'subspaces' - Random subspaces.<br/>
        * 'resampling' - Resampling.<br/>
        * 'patches' - Random patches.
    lam
        Lambda value for bagging.
    drift_detector
        Drift detector.
    warning_detector
        Warning detector.
    disable_detector
        Option to disable drift detectors:<br/>
        * If `'off'`, detectors are enabled.<br/>
        * If `'drift'`, disables concept drift detection and the background learner.<br/>
        * If `'warning'`, disables the background learner and ensemble members are
         reset if drift is detected.<br/>
    disable_weighted_vote
        If True, disables weighted voting.
    drift_detection_criteria
        The criteria used to track drifts.<br/>
        * 'error' - absolute error.<br/>
        * 'prediction' - predicted target values.
    aggregation_method
        The method to use to aggregate predictions in the ensemble.<br/>
        * 'mean'<br/>
        * 'median'
    seed
        Random number generator seed for reproducibility.
    metric
        The metric to track members performance within the ensemble.

    Examples
    --------

    >>> from river import ensemble
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river.datasets import synth
    >>> from river import tree

    >>> dataset = synth.FriedmanDrift(
    ...     drift_type='gsg',
    ...     position=(350, 750),
    ...     transition_window=200,
    ...     seed=42
    ... ).take(1000)

    >>> base_model = tree.HoeffdingTreeRegressor(grace_period=50)
    >>> model = ensemble.SRPRegressor(
    ...     model=base_model,
    ...     training_method="patches",
    ...     n_models=3,
    ...     seed=42
    ... )

    >>> metric = metrics.R2()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    R2: 0.571117

    Notes
    -----
    This implementation uses `n_models=10` as default given the impact on
    processing time. The optimal number of models depends on the data and
    resources available.

    References
    ----------
    [^1]: Heitor Gomes, Jacob Montiel, Saulo Martiello Mastelini,
          Bernhard Pfahringer, and Albert Bifet.
          On Ensemble Techniques for Data Stream Regression.
          IJCNN'20. International Joint Conference on Neural Networks. 2020.

    [^2]: Heitor Murilo Gomes, Jesse Read, Albert Bifet.
          Streaming Random Patches for Evolving Data Stream Classification.
          IEEE International Conference on Data Mining (ICDM), 2019.

    """

    _MEAN = "mean"
    _MEDIAN = "median"
    _ERROR = "error"
    _PREDICTION = "prediction"

    def __init__(
            self,
            model: base.Estimator | None = None,
            n_models: int = 10,
            fs_batch_size: int = 100,
            fs_bootstrap_size: float | str = 0.5,
            fs_function: callable = get_alternative_features_regression,
            subspace_size: int | float | str = 0.6,
            training_method: str = "patches",
            lam: int = 6,
            drift_detector: base.DriftDetector | None = None,
            warning_detector: base.DriftDetector | None = None,
            disable_detector: str = "off",
            disable_weighted_vote: bool = True,
            drift_detection_criteria: str = "error",
            aggregation_method: str = "mean",
            seed=None,
            metric: RegressionMetric | None = None,
    ):
        # Check arguments for parent class
        if model is None:
            model = HoeffdingTreeRegressor(grace_period=50, delta=0.01)

        if drift_detector is None:
            drift_detector = ADWIN(delta=1e-5)

        if warning_detector is None:
            warning_detector = ADWIN(delta=1e-4)

        if disable_detector == "off":
            pass
        elif disable_detector == "drift":
            drift_detector = None
            warning_detector = None
        elif disable_detector == "warning":
            warning_detector = None
        else:
            raise AttributeError(
                f"{disable_detector} is not a valid value for disable_detector.\n"
                f"Valid options are: 'off', 'drift', 'warning'"
            )

        if metric is None:
            metric = MAE()

        super().__init__(
            model=model,
            n_models=n_models,
            fs_bootstrap_size=fs_bootstrap_size,
            fs_batch_size=fs_batch_size,
            fs_function=fs_function,
            subspace_size=subspace_size,
            training_method=training_method,
            lam=lam,
            drift_detector=drift_detector,
            warning_detector=warning_detector,
            disable_detector=disable_detector,
            disable_weighted_vote=disable_weighted_vote,
            seed=seed,
            metric=metric,
        )

        if aggregation_method not in {self._MEAN, self._MEDIAN}:
            raise ValueError(
                f"Invalid aggregation_method: {aggregation_method}.\n"
                f"Valid options are: {[self._MEAN, self._MEDIAN]}"
            )
        self.aggregation_method = aggregation_method

        if drift_detection_criteria not in {self._ERROR, self._PREDICTION}:
            raise ValueError(
                f"Invalid drift_detection_criteria: {drift_detection_criteria}.\n"
                f"Valid options are: {[self._ERROR, self._PREDICTION]}"
            )
        self.drift_detection_criteria = drift_detection_criteria

        self._base_learner_class = BaseSRPRegressor  # type: ignore
        self._feature_selection_function = get_alternative_features_regression

    def predict_one(self, x, **kwargs):
        y_pred = np.zeros(self.n_models)
        weights = np.ones(self.n_models)

        for i, model in enumerate(self.models):
            y_pred[i] = model.predict_one(x, **kwargs)
            if not self.disable_weighted_vote:
                metric_value = model.metric.get()
                weights[i] = metric_value if metric_value >= 0 else 0.0

        if self.aggregation_method == self._MEAN:
            if not self.disable_weighted_vote:
                if not self.metric.bigger_is_better:
                    # Invert weights so smaller values have larger influence
                    weights = -(weights - max(weights))
                if sum(weights) == 0:
                    # Average is undefined
                    return 0.0
            return np.average(y_pred, weights=weights)
        else:  # self.aggregation_method == self._MEDIAN:
            return np.median(y_pred)
