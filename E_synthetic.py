import copy
import multiprocessing
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns

from transformations import drop_dates

sns.set()

from river import metrics
from river.dummy import NoChangeClassifier
from river.ensemble import SRPClassifier, ADWINBaggingClassifier, ADWINBoostingClassifier
from river.forest import ARFClassifier
from river.tree import HoeffdingTreeClassifier, HoeffdingAdaptiveTreeClassifier
from river.preprocessing import StandardScaler

from river.datasets import synth
from util import run_async, separate_name_and_dimensions
from transformations import AddIrrelevantFeaturesTransformer, AddNoisyFeaturesTransformer
from evaluate import evaluate
from macros import *

from fssrp import FSSRPClassifier


if __name__ == '__main__':
    n_jobs = multiprocessing.cpu_count() - 2
    reps = 10
    n_estimators = 10
    subspace_size = "sqrt"
    stream_length = 1000
    n_features = 10

    datasets = {
        synth.RandomRBF.__name__: (synth.RandomRBF, [
            {"seed_model": np.nan, "seed_sample": np.nan, "n_classes": 2, "n_features": n_features}
        ]),
        synth.Hyperplane.__name__: (synth.RandomRBF, [
            {"seed_model": np.nan, "seed_sample": np.nan, "n_classes": 2, "n_features": n_features}
        ]),
        synth.Agrawal.__name__: (synth.Agrawal, [
            {"seed": np.nan}
        ])
    }

    grace_period = 50
    delta = 0.05
    base_model = HoeffdingTreeClassifier(
        grace_period=grace_period, delta=delta
    )
    approaches = {
        NoChangeClassifier.__name__: (NoChangeClassifier, [{}]),
        # ExtremelyFastDecisionTreeClassifier: [
        #     {"grace_period": grace_period, "delta": delta}
        # ],  # DID NOT WORK...
        # "HoeffdingTree": (HoeffdingTreeClassifier, [
        #     {"grace_period": grace_period, "delta": delta}
        # ]),
        # "HoeffdingAdaptiveTree": (HoeffdingAdaptiveTreeClassifier, [
        #     {"grace_period": grace_period, "delta": delta, "seed": np.nan}
        # ]),
        # # "AdaptiveRandomForest": (ARFClassifier, [
        # #     {"n_models": n_estimators, "seed": np.nan}
        # # ]),
        # "ADWINBagging": (ADWINBaggingClassifier, [
        #     {"model": copy.deepcopy(base_model), "n_models": n_estimators, "seed": np.nan}
        # ]),
        # "ADWINBoosting": (ADWINBoostingClassifier, [
        #     {"model": copy.deepcopy(base_model), "n_models": n_estimators, "seed": np.nan}
        # ]),
        # "SRP (1)": (SRPClassifier, [
        #     {"model": copy.deepcopy(base_model), "n_models": 1,
        #      "subspace_size": subspace_size, "seed": np.nan}
        # ]),
        # "FSSRP (1)": (FSSRPClassifier, [
        #     {"model": copy.deepcopy(base_model), "n_models": 1,
        #      "subspace_size": subspace_size, "seed": np.nan,
        #      "fs_batch_size": 500, "fs_bootstrap_size": 1.0}
        # ]),
        "SRP": (SRPClassifier, [
            {"model": copy.deepcopy(base_model), "n_models": n_estimators,
             "subspace_size": subspace_size, "seed": np.nan}
        ]),
        "FSSRP": (FSSRPClassifier, [
            {"model": copy.deepcopy(base_model), "n_models": n_estimators, "fs_bootstrap_size": "sqrt",
             "subspace_size": subspace_size, "seed": np.nan, "fs_batch_size": 100}
        ]),
    }

    metric = metrics.BalancedAccuracy()

    columns = [DATASET, APPROACH, REPETITION, SCORE, METRIC]
    experiments = []
    results = []
    for ds_name, (ds, ds_args_list) in datasets.items():
        for ds_args in ds_args_list:
            for approach_name, (approach, args_list) in approaches.items():
                for args in args_list:
                    for rep in range(reps):
                        if "seed" in args:
                            args["seed"] = rep
                        if "seed" in ds_args:
                            ds_args["seed"] = rep
                        if "seed_model" in ds_args:
                            ds_args["seed_model"] = rep
                            ds_args["seed_sample"] = rep
                        dataset = ds(**ds_args)
                        classifier = StandardScaler()
                        classifier |= AddNoisyFeaturesTransformer(seed=rep)
                        classifier |= approach(**args)
                        # results.append(evaluate(dataset, classifier, rep,
                        #                         metric, stream_length,
                        #                         approach_name, ds_name))
                        experiments.append([dataset, classifier,
                                            rep, metric, stream_length,
                                            approach_name, ds_name])

    results = run_async(evaluate, experiments, njobs=n_jobs)
    final_df = pd.DataFrame(results, columns=columns)
    # app_names, dims = separate_name_and_dimensions(final_df[DATASET])
    # final_df[DATASET] = app_names
    # final_df[DIMENSIONS] = dims
    # final_df[DIMENSIONS] = final_df[DIMENSIONS].astype(int)
    final_df.to_parquet(os.path.join(os.getcwd(), "results", "results_random_rbf.parquet"))

    g = sns.catplot(kind="bar", data=final_df,
                    x=DATASET, y=SCORE, hue=APPROACH)
    sns.move_legend(g, "upper center", ncol=4, title="")
    plt.xlabel("")
    plt.ylabel(final_df[METRIC].iloc[0])
    # plt.xticks(rotation=30, ha="right")
    plt.tight_layout(pad=.5)
    plt.subplots_adjust(top=.85)
    plt.savefig(os.path.join(os.getcwd(), "figures", "results_random_rbf.pdf"))
