from typing import Dict, Union

import numpy as np
import pandas as pd
from mrmr import mrmr_classif, mrmr_regression
from sklearn.feature_selection import mutual_info_classif


def _mi_feature_selection(X: pd.DataFrame, y, k):
    chi2_values = mutual_info_classif(X.to_numpy(), y)
    quantile = np.quantile(chi2_values, 1.0 - k / X.shape[0])
    indices = [i for i in range(len(chi2_values)) if chi2_values[i] >= quantile]
    return np.array(X.columns)[indices].tolist()


def get_alternative_features_classification(data: pd.DataFrame,
                                            labels: np.ndarray,
                                            n_components: int,
                                            relative_size: Union[str, float],
                                            bootstrap_size: Union[str, float],
                                            rng: np.random.Generator):
    d = data.shape[1]
    if relative_size == "sqrt":
        relative_size = np.sqrt(d) / d
    k = np.ceil(d * relative_size).astype(int)
    alternative_feature_sets = []
    if bootstrap_size == "sqrt":
        bootstrap_size = np.sqrt(data.shape[0]) / data.shape[0]
    for _ in range(n_components):
        X_sample, y_sample = get_sample_df(data, labels, sample_proba=bootstrap_size, rng=rng)
        this_fs = _mi_feature_selection(X_sample, y_sample, k)
        # this_fs = mrmr_classif(X_sample, y_sample, k, show_progress=False)
        alternative_feature_sets.append(this_fs)
    print(alternative_feature_sets)
    return alternative_feature_sets


def get_alternative_features_regression(data: pd.DataFrame,
                                        labels: np.ndarray,
                                        n_components: int,
                                        relative_size: float,
                                        rng: np.random.Generator):
    d = data.shape[1]
    if relative_size == "sqrt":
        relative_size = np.sqrt(d) / d
    k = np.ceil(d * relative_size).astype(int)
    alternative_feature_sets = []
    sample_proba = np.sqrt(data.shape[0]) / data.shape[0]
    for _ in range(n_components):
        X_sample, y_sample = get_sample_df(data, labels, sample_proba=sample_proba, rng=rng)
        this_fs = mrmr_regression(X_sample, y_sample, k, show_progress=False)
        alternative_feature_sets.append(this_fs)
    return alternative_feature_sets


def get_alternative_features_baseline(data: pd.DataFrame,
                                      labels: np.ndarray,
                                      n_components: int,
                                      relative_size: Union[str, float],
                                      bootstrap_size: Union[str, float],
                                      rng: np.random.Generator):
    d = data.shape[1]
    if relative_size == "sqrt":
        relative_size = np.sqrt(d) / d
    k = np.ceil(d * relative_size).astype(int)
    random_featuresets = get_random_subspaces(data.columns, k=k, n_components=n_components, rng=rng)
    relevance = mutual_info_classif(data.to_numpy(), labels)
    if np.all(relevance == 0):
        return random_featuresets
    quantile = np.quantile(relevance, .5)
    relevant_features = data.columns[relevance > quantile]
    relevant_featuresets = []
    for index, fs in enumerate(random_featuresets):
        fs = np.intersect1d(fs, relevant_features)
        n_removed = k - len(fs)
        if n_removed > 0:
            not_included_relevant_features = np.setdiff1d(relevant_features, fs)
            fs = np.append(fs, np.random.choice(not_included_relevant_features, n_removed, replace=False))
            relevant_featuresets.append(fs)
    return relevant_featuresets


def get_sample_df(X: pd.DataFrame, y: np.ndarray, sample_proba: float, rng: np.random.Generator, replace=True):
    all_indices = np.arange(len(y))
    sample_size = int(len(y) * sample_proba)
    selected_indices = rng.choice(all_indices, sample_size, replace=replace)
    return X.iloc[selected_indices], y[selected_indices]


def get_random_subspaces(columns: pd.Series, k: int, n_components, rng: np.random.Generator) -> list:
    subspaces = [np.random.choice(columns, k, replace=False) for _ in range(n_components)]
    return subspaces
