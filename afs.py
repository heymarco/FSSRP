from typing import Dict

import numpy as np
import pandas as pd
from mrmr import mrmr_classif, mrmr_regression


def get_alternative_features_classification(data: pd.DataFrame,
                                            labels: np.ndarray,
                                            n_components: int,
                                            relative_size: float):
    d = data.shape[1]
    if relative_size == "sqrt":
        relative_size = np.sqrt(d) / d
    k = np.ceil(d * relative_size).astype(int)
    alternative_feature_sets = []
    sample_proba = np.sqrt(data.shape[0]) / data.shape[0]
    for _ in range(n_components):
        X_sample, y_sample = get_sample_df(data, labels, sample_proba=sample_proba)
        this_fs = mrmr_classif(X_sample, y_sample, k, show_progress=False)
        alternative_feature_sets.append(this_fs)
    return alternative_feature_sets


def get_alternative_features_regression(data: pd.DataFrame,
                                        labels: np.ndarray,
                                        n_components: int,
                                        relative_size: float):
    d = data.shape[1]
    if relative_size == "sqrt":
        relative_size = np.sqrt(d) / d
    k = np.ceil(d * relative_size).astype(int)
    alternative_feature_sets = []
    for _ in range(n_components):
        X_sample, y_sample = get_sample_df(data, labels, sample_proba=0.5)
        this_fs = mrmr_regression(X_sample, y_sample, k, show_progress=False)
        alternative_feature_sets.append(this_fs)
    return alternative_feature_sets


def get_sample_df(X: pd.DataFrame, y: np.ndarray, sample_proba: float, replace=True):
    all_indices = np.arange(len(y))
    sample_size = int(len(y) * sample_proba)
    selected_indices = np.random.choice(all_indices, sample_size, replace=replace)
    return X.iloc[selected_indices], y[selected_indices]
