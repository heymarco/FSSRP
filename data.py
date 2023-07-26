from typing import Dict

from river import datasets
from river.datasets import synth

_rw_datasets = [
    # datasets.AirlinePassengers,
    # # datasets.Bananas,  # only 2 features
    # datasets.Bikes,
    # datasets.ChickWeights,
    # # datasets.CreditCard,
    # datasets.Elec2,
    # datasets.HTTP,
    # datasets.Higgs,
    # datasets.ImageSegments,
    # datasets.Insects,
    # datasets.Keystroke,
    datasets.MaliciousURL,  # more than 3 million features
    # datasets.MovieLens100K,
    # datasets.Music,
    # datasets.Phishing,
    # datasets.Restaurants,
    # # datasets.SMSSpam,  # TODO: missing encoding
    # # datasets.SMTP,  # highly imbalanced (only 0.4% positive labels)
    # datasets.SolarFlare,
    # # datasets.TREC07,  # error loading dataset
    # # datasets.Taxis,  # contains outliers, I think.
    # datasets.TrumpApproval,
    # datasets.WaterFlow
]

_syn_datasets = [
    synth.Agrawal,
    synth.AnomalySine,
    synth.ConceptDriftStream,
    synth.Friedman,
    synth.FriedmanDrift,
    synth.Hyperplane,
    synth.LED,
    synth.LEDDrift,
    synth.Logical,
    synth.Mixed,
    synth.Mv,
    synth.Planes2D,
    synth.RandomRBF,
    synth.RandomRBFDrift,
    synth.RandomTree,
    synth.SEA,
    synth.STAGGER,
    synth.Sine,
    synth.Waveform
]


def get_rw_binary_classification_datasets() -> Dict:
    return {
        ds.__name__: ds for ds in _rw_datasets
        if ds().task == datasets.base.BINARY_CLF
    }


def get_rw_classification_datasets() -> Dict:
    return {
        ds.__name__: ds for ds in _rw_datasets
        if ds().task == datasets.base.BINARY_CLF or ds().task == datasets.base.MULTI_CLF
    }


def get_rw_regression_datasets() -> Dict:
    return {
        ds.__name__: ds for ds in _rw_datasets
        if ds().task == datasets.base.REG and ds().n_features > 1
    }


def get_rw_multi_class_classification_datasets() -> Dict:
    return {
        ds.__name__: ds for ds in _rw_datasets
        if ds().task == datasets.base.MULTI_CLF
    }


def get_rw_multi_output_classification_datasets() -> Dict:
    return {
        ds.__name__: ds for ds in _rw_datasets
        if ds().task == datasets.base.MO_BINARY_CLF
    }


def get_rw_multi_output_regression_datasets() -> Dict:
    return {
        ds.__name__: ds for ds in _rw_datasets
        if ds().task == datasets.base.MO_REG
    }


if __name__ == '__main__':
    data_loaders = [
        get_rw_binary_classification_datasets,
        get_rw_classification_datasets,
        get_rw_multi_class_classification_datasets,
        get_rw_multi_output_classification_datasets,
        get_rw_regression_datasets,
        get_rw_multi_output_regression_datasets
    ]
    for loader in data_loaders:
        print(loader())