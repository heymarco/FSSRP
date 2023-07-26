import copy
from enum import Enum
from typing import Dict, Union

import numpy as np
import pandas as pd
from mrmr import mrmr_classif, mrmr_regression
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


class AFSType(Enum):
    SAMPLE = "sample"
    WEIGHT = "weight"
    ELIMINATE_IRRELEVANT = "elim. irrelevant"
    SUCCESSIVE_REPLACEMENT = "successive repl."


class TaskType(Enum):
    CLASSIFICATION = 0
    REGRESSION = 1


class FSType(Enum):
    MRMR = 0
    MI = 1


def get_alternative_features(data: pd.DataFrame, labels: np.ndarray,
                             afstype: AFSType, fstype: FSType, tasktype: TaskType,
                             n_components: int,
                             relative_size: Union[str, float],
                             bootstrap_size: Union[str, float],
                             relevance_quantile: float,
                             rng: np.random.Generator
                             ) -> list:
    d = data.shape[1]
    if relative_size == "sqrt":
        relative_size = np.sqrt(d) / d
    fs_size = np.ceil(d * relative_size).astype(int)
    if bootstrap_size == "sqrt":
        bootstrap_size = np.sqrt(data.shape[0]) / data.shape[0]
    afs = None
    if afstype == AFSType.SAMPLE:
        afs = _get_afs_sampling(data, labels, n_components, fs_size=fs_size,
                                bootstrap_size=bootstrap_size, rng=rng, fstype=fstype, tasktype=tasktype)
    elif afstype == AFSType.WEIGHT:
        afs = _get_afs_weighting(data, labels, n_components, fs_size=fs_size,
                                 rng=rng, fstype=fstype, tasktype=tasktype)
    elif afstype == AFSType.ELIMINATE_IRRELEVANT:
        afs = _get_alternative_features_eliminate_irrelevant(data, labels, n_components, fs_size=fs_size,
                                                             tasktype=tasktype, rng=rng,
                                                             relevance_quantile=relevance_quantile)
    elif afstype == AFSType.SUCCESSIVE_REPLACEMENT:
        afs_function = SuccessiveReplacementAFS(relevance_quantile=relevance_quantile, tasktype=tasktype, rng=rng)
        afs = afs_function(data, labels, n_components, relative_size=relative_size,
                           bootstrap_size=bootstrap_size)
    return afs


def _mi_feature_selection_classif(X: pd.DataFrame, y, k):
    mi_values = mutual_info_classif(X.to_numpy(), y)
    quantile = np.quantile(mi_values, 1.0 - k / X.shape[0])
    indices = [i for i in range(len(mi_values)) if mi_values[i] >= quantile]
    return np.array(X.columns)[indices].tolist()


def _mi_feature_selection_regression(X: pd.DataFrame, y, k):
    mi_values = mutual_info_regression(X.to_numpy(), y)
    quantile = np.quantile(mi_values, 1.0 - k / X.shape[0])
    indices = [i for i in range(len(mi_values)) if mi_values[i] >= quantile]
    return np.array(X.columns)[indices].tolist()


def _get_afs_sampling(data: pd.DataFrame,
                      labels: np.ndarray,
                      n_components: int,
                      fs_size: int,
                      bootstrap_size: float,
                      rng: np.random.Generator,
                      fstype: FSType,
                      tasktype: TaskType):
    X_sample, y_sample = _get_sample_df(data, labels, bootstrap_size, rng=rng)
    afs = _get_afs(X_sample, y_sample, n_components, fs_size=fs_size, fstype=fstype, tasktype=tasktype)
    return afs


def _get_afs_weighting(data: pd.DataFrame,
                       labels: np.ndarray,
                       n_components: int,
                       fs_size: int,
                       rng: np.random.Generator,
                       fstype: FSType,
                       tasktype: TaskType):
    X_sample, y_sample = _get_weighted_df(data, labels, rng)
    afs = _get_afs(X_sample, y_sample, n_components, fs_size=fs_size, fstype=fstype, tasktype=tasktype)
    return afs


def _get_afs(X: pd.DataFrame, y: np.ndarray, n_components: int, fs_size: int,
             fstype: FSType, tasktype: TaskType) -> list:
    alternative_feature_sets = []
    for _ in range(n_components):
        if fstype == FSType.MRMR:
            if tasktype == TaskType.CLASSIFICATION:
                this_fs = mrmr_classif(X, y, fs_size, show_progress=False)
            elif tasktype == TaskType.REGRESSION:
                this_fs = mrmr_regression(X, y, fs_size, show_progress=False)
        elif fstype == FSType.MI:
            if tasktype == TaskType.CLASSIFICATION:
                this_fs = _mi_feature_selection_classif(X, y, fs_size)
            elif tasktype == TaskType.REGRESSION:
                this_fs = _mi_feature_selection_regression(X, y, fs_size)
        alternative_feature_sets.append(this_fs)
    return alternative_feature_sets


def _get_alternative_features_eliminate_irrelevant(data: pd.DataFrame,
                                                   labels: np.ndarray,
                                                   n_components: int,
                                                   fs_size: int,
                                                   tasktype: TaskType,
                                                   relevance_quantile: float,
                                                   rng: np.random.Generator):
    random_featuresets = _get_random_subspaces(data.columns, k=fs_size, n_components=n_components, rng=rng)
    if tasktype == TaskType.CLASSIFICATION:
        relevance = mutual_info_classif(data.to_numpy(), labels)
    elif tasktype == TaskType.REGRESSION:
        relevance = mutual_info_regression(data.to_numpy(), labels)
    if np.all(relevance == 0):
        return random_featuresets
    quantile = np.quantile(relevance, relevance_quantile)
    relevant_features = data.columns[relevance > quantile]
    relevant_featuresets = []
    for index, fs in enumerate(random_featuresets):
        fs = np.intersect1d(fs, relevant_features)
        n_removed = fs_size - len(fs)
        if n_removed > 0:
            not_included_relevant_features = np.setdiff1d(relevant_features, fs)
            fs = np.append(fs, rng.choice(not_included_relevant_features, n_removed, replace=False))
            relevant_featuresets.append(fs)
    return relevant_featuresets


def _get_sample_df(X: pd.DataFrame, y: np.ndarray, sample_proba: float, rng: np.random.Generator, replace=True):
    all_indices = np.arange(len(y))
    sample_size = int(len(y) * sample_proba)
    selected_indices = rng.choice(all_indices, sample_size, replace=replace)
    return X.iloc[selected_indices], y[selected_indices]


def _get_weighted_df(X: pd.DataFrame, y: np.ndarray, rng: np.random.Generator):
    weights = rng.poisson(lam=1, size=len(y))
    X_chunks, y_chunks = [], []
    for weight in np.unique(weights):
        if weight == 0:
            mask = weights > weight
            X = X.loc[mask]
            y = y[mask]
            weights = weights[weights > 0]
        elif weight == 1:
            continue
        else:
            mask = weights == weight
            X_repeated = pd.concat([
                X.loc[mask] for _ in range(weight - 1)
            ], ignore_index=True)
            y_repeated = np.concatenate([y[mask] for _ in range(weight - 1)])
            X_chunks.append(X_repeated)
            y_chunks.append(y_repeated)
    X = pd.concat(X_chunks, ignore_index=True)
    y = np.concatenate(y_chunks)
    return X, y


def _get_random_subspaces(columns: pd.Series, k: int, n_components, rng: np.random.Generator) -> list:
    subspaces = [rng.choice(columns, k, replace=False) for _ in range(n_components)]
    return subspaces


class SuccessiveReplacementAFS():
    def __init__(self, relevance_quantile: float, tasktype: TaskType, rng: np.random.Generator):
        self.relevance_quantile = relevance_quantile
        self._feature_correlation_mat: Union[pd.DataFrame, None] = None
        self._feature_relevance: Union[dict, None] = None
        self._tasktype = tasktype
        self._rng = rng

    def __call__(self,
                 data: pd.DataFrame,
                 labels: np.ndarray,
                 n_components: int,
                 relative_size: float,
                 bootstrap_size: float):
        self._compute_relevance(data, labels)
        self._compute_correlation_mat(data)
        relevant_fs = self._get_max_relevant_fs()
        blueprint_quantile = np.quantile(list(self._feature_relevance.values()), 1 - relative_size)
        fs_blueprint = [feature for feature in relevant_fs
                        if self._feature_relevance[feature] > blueprint_quantile]
        unused_relevant_features = np.setdiff1d(relevant_fs, fs_blueprint)
        featuresets = [copy.deepcopy(fs_blueprint) for _ in range(n_components)]
        while len(unused_relevant_features) > 0:
            for this_fs_index in range(n_components):
                this_fs = featuresets[this_fs_index]
                this_fs, unused_relevant_features = self._reduce_redundancy_random(this_fs, unused_relevant_features)
                featuresets[this_fs_index] = this_fs
                if len(unused_relevant_features) == 0:
                    break
        return featuresets

    def _get_max_relevant_fs(self) -> list:
        quantile = np.quantile(list(self._feature_relevance.values()), self.relevance_quantile)
        relevant_features = [
            feature for (feature, relevance) in self._feature_relevance.items()
            if relevance > quantile
        ]
        return relevant_features

    def _reduce_redundancy_random(self, fs: np.ndarray, available_features: np.ndarray) -> (np.ndarray, np.ndarray):
        removed_feature_index = self._rng.choice(range(len(fs)))
        included_feature_index = self._rng.choice(range(len(available_features)))
        fs[removed_feature_index] = available_features[included_feature_index]
        available_features = available_features[available_features != available_features[included_feature_index]]
        return fs, available_features

    def _reduce_redundancy_directed(self, fs: np.ndarray, available_features: np.ndarray) -> (np.ndarray, np.ndarray):
        redundancies_included = self._feature_correlation_mat.loc[fs, fs]
        redundancies_included = [
            redundancies_included.loc[row_index, redundancies_included.columns != row_index]
            for row_index in redundancies_included.index
        ]
        redundancies_included = pd.concat(redundancies_included, axis=1, ignore_index=False)
        redundancy_max_included = np.nanmax(redundancies_included.to_numpy(), axis=1)
        removed_feature_index = np.argmax(redundancy_max_included)

        redundancies_available = self._feature_correlation_mat.loc[fs, available_features]
        redundancies_available = [
            redundancies_available.loc[row_index, redundancies_available.columns != row_index]
            for row_index in redundancies_available.index
        ]
        redundancies_available = pd.concat(redundancies_available, axis=1, ignore_index=False)
        redundancy_max_available = np.nanmax(redundancies_available.to_numpy(), axis=1)
        included_feature_index = np.argmin(redundancy_max_available)

        fs[removed_feature_index] = available_features[included_feature_index]
        available_features = available_features[available_features != available_features[included_feature_index]]
        return fs, available_features

    def _compute_relevance(self, X: pd.DataFrame, y: np.ndarray):
        columns = list(X.columns)
        if self._tasktype == TaskType.CLASSIFICATION:
            mi_values = mutual_info_classif(X.to_numpy(), y)
        elif self._tasktype == TaskType.REGRESSION:
            mi_values = mutual_info_regression(X.to_numpy(), y)
        self._feature_relevance = {
            columns[i]: mi_values[i] for i in range(len(columns))
        }

    def _compute_correlation_mat(self, X: pd.DataFrame):
        mi_values = [mutual_info_regression(X, X[col]) for col in X.columns]
        corr_matrix = pd.DataFrame(mi_values, index=X.columns, columns=X.columns)
        self._feature_correlation_mat = corr_matrix
