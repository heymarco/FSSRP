import copy
import multiprocessing
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns
sns.set()

from river import metrics, stats
from river.dummy import StatisticRegressor
from river.ensemble import SRPRegressor, BaggingRegressor, EWARegressor
from river.forest import ARFRegressor
from river.tree import HoeffdingTreeRegressor, HoeffdingAdaptiveTreeRegressor
from river.preprocessing import StandardScaler, TargetStandardScaler, TargetMinMaxScaler

from data import get_rw_classification_datasets, get_rw_binary_classification_datasets, get_rw_regression_datasets
from util import run_async
from evaluate import evaluate
from macros import *
from transformations import drop_dates, drop_categorical

from fssrp import FSSRPRegressor

if __name__ == '__main__':
    n_jobs = multiprocessing.cpu_count()
    reps = 1
    n_estimators = 10
    subspace_size = "sqrt"
    stream_length = 3000

    grace_period = 50
    delta = 0.05
    base_model = HoeffdingTreeRegressor(
        grace_period=grace_period, delta=delta
    )
    approaches = {
        "MeanRegressor": (StatisticRegressor, [
            {"statistic": stats.Mean()}
        ]),
        "HoeffdingTree": (HoeffdingTreeRegressor, [
            {"grace_period": grace_period, "delta": delta}
        ]),
        "HoeffdingAdaptiveTree": (HoeffdingAdaptiveTreeRegressor, [
            {"grace_period": grace_period, "delta": delta, "seed": np.nan}
        ]),
        "AdaptiveRandomForest": (ARFRegressor, [
            {"n_models": n_estimators, "seed": np.nan}
        ]),
        BaggingRegressor.__name__: (BaggingRegressor, [
            {"model": copy.deepcopy(base_model), "n_models": n_estimators, "seed": np.nan}
        ]),
        EWARegressor.__name__: (EWARegressor, [
            {"models": [copy.deepcopy(base_model) for _ in range(n_estimators)]}
        ]),
        "SRP (1)": (SRPRegressor, [
            {"model": copy.deepcopy(base_model), "n_models": 1,
             "subspace_size": subspace_size, "seed": np.nan}
        ]),
        "FSSRP (1)": (FSSRPRegressor, [
            {"model": copy.deepcopy(base_model), "n_models": 1,
             "subspace_size": subspace_size, "seed": np.nan}
        ]),
        "SRP": (SRPRegressor, [
            {"model": copy.deepcopy(base_model), "n_models": n_estimators,
             "subspace_size": subspace_size, "seed": np.nan}
        ]),
        "FSSRP": (FSSRPRegressor, [
            {"model": copy.deepcopy(base_model), "n_models": n_estimators,
             "subspace_size": subspace_size, "seed": np.nan}
        ]),
    }

    metric = metrics.MAE()

    columns = [DATASET, APPROACH, REPETITION, SCORE, METRIC]
    experiments = []
    results = []
    for ds_name, ds in get_rw_regression_datasets().items():
        for approach_name, (approach, args_list) in approaches.items():
            for args in args_list:
                for rep in range(reps):
                    if "seed" in args:
                        args["seed"] = rep
                    dataset = ds()
                    classifier = drop_categorical | drop_dates | StandardScaler()
                    classifier |= TargetMinMaxScaler(approach(**args))
                    # results.append(evaluate(dataset, classifier, rep, metric, stream_length))
                    experiments.append([dataset, classifier, rep, metric, stream_length, approach_name])

    results = run_async(evaluate, experiments, njobs=n_jobs)
    final_df = pd.DataFrame(results, columns=columns)
    final_df.to_parquet(os.path.join(os.getcwd(), "results", "result_regression.parquet"))
