import copy
import multiprocessing
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from river.preprocessing import StandardScaler
from tqdm import tqdm
import seaborn as sns

from afs import FSType, AFSType
from transformations import drop_dates, AddIrrelevantFeaturesTransformer, AddNoisyFeaturesTransformer

sns.set()

from river import metrics
from river.dummy import NoChangeClassifier
from river.ensemble import SRPClassifier, ADWINBaggingClassifier, ADWINBoostingClassifier
from river.forest import ARFClassifier
from river.tree import HoeffdingTreeClassifier, HoeffdingAdaptiveTreeClassifier, ExtremelyFastDecisionTreeClassifier

from data import get_rw_classification_datasets, get_rw_binary_classification_datasets
from util import run_async
from evaluate import evaluate
from macros import *

from fssrp import FSSRPClassifier


if __name__ == '__main__':
    n_jobs = multiprocessing.cpu_count() - 2
    reps = 5
    n_estimators = 30
    subspace_size = "sqrt"
    stream_length = 1000

    grace_period = 50
    delta = 0.05
    base_model = HoeffdingTreeClassifier(
        grace_period=grace_period, delta=delta
    )
    approaches = {
        NoChangeClassifier.__name__: (NoChangeClassifier, [{}]),
        "HoeffdingTree": (HoeffdingTreeClassifier, [
            {"grace_period": grace_period, "delta": delta}
        ]),
        "HoeffdingAdaptiveTree": (HoeffdingAdaptiveTreeClassifier, [
            {"grace_period": grace_period, "delta": delta, "seed": np.nan}
        ]),
        "AdaptiveRandomForest": (ARFClassifier, [
            {"n_models": n_estimators, "seed": np.nan}
        ]),
        "ADWINBagging": (ADWINBaggingClassifier, [
            {"model": copy.deepcopy(base_model), "n_models": n_estimators, "seed": np.nan}
        ]),
        "ADWINBoosting": (ADWINBoostingClassifier, [
            {"model": copy.deepcopy(base_model), "n_models": n_estimators, "seed": np.nan}
        ]),
        "SRP (1)": (SRPClassifier, [
            {"model": copy.deepcopy(base_model), "n_models": 1,
             "subspace_size": subspace_size, "seed": np.nan}
        ]),
        f"FSSRP-1 ({AFSType.WEIGHT})": (FSSRPClassifier, [
            {"model": copy.deepcopy(base_model), "n_models": 1,
             "subspace_size": subspace_size, "seed": np.nan,
             "fstype": FSType.MI, "afstype": AFSType.WEIGHT}
        ]),
        "SRP": (SRPClassifier, [
            {"model": copy.deepcopy(base_model), "n_models": n_estimators,
             "subspace_size": subspace_size, "seed": np.nan}
        ]),
        f"FSSRP-{n_estimators} ({AFSType.WEIGHT})": (FSSRPClassifier, [
            {"model": copy.deepcopy(base_model), "n_models": n_estimators,
             "subspace_size": subspace_size, "seed": np.nan,
             "fstype": FSType.MI, "afstype": AFSType.WEIGHT}
        ])
    }

    metric = metrics.BalancedAccuracy()

    columns = [DATASET, APPROACH, REPETITION, SCORE, METRIC]
    experiments = []
    results = []
    for ds_name, ds in get_rw_classification_datasets().items():
        for approach_name, (approach, args_list) in approaches.items():
            for args in args_list:
                for rep in range(reps):
                    if "seed" in args:
                        args["seed"] = rep
                    dataset = ds()
                    classifier = drop_dates | StandardScaler() | AddNoisyFeaturesTransformer(seed=rep)
                    classifier |= copy.deepcopy(approach(**args))
                    # results.append(evaluate(dataset, classifier, rep, metric, stream_length, approach_name))
                    experiments.append([dataset, classifier, rep, metric, stream_length, approach_name])

    results = run_async(evaluate, experiments, njobs=n_jobs)
    final_df = pd.DataFrame(results, columns=columns)
    final_df.to_parquet(os.path.join(os.getcwd(), "results", "results_classification_baseline_selection.parquet"))

    avg_df = final_df.copy()
    avg_df[DATASET] = "Average"
    final_df = pd.concat([avg_df, final_df], ignore_index=True)

    g = sns.catplot(kind="bar", data=final_df, x=DATASET, y=SCORE, hue=APPROACH)
    sns.move_legend(g, "upper center", ncol=4, title="")
    plt.xlabel("")
    plt.ylabel(final_df[METRIC].iloc[0])
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout(pad=.5)
    plt.subplots_adjust(top=.85)
    plt.savefig(os.path.join(os.getcwd(), "figures", "results_classification.pdf"))
