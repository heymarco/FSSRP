from typing import Dict

from river import datasets

_rw_datasets = [
    datasets.AirlinePassengers,
    datasets.Bananas,
    datasets.Bikes,
    datasets.ChickWeights,
    datasets.CreditCard,
    datasets.Elec2,
    datasets.HTTP,
    datasets.Higgs,
    datasets.ImageSegments,
    datasets.Insects,
    datasets.Keystroke,
    datasets.MaliciousURL,  # more than 3 million features
    # datasets.MovieLens100K,
    datasets.Music,
    datasets.Phishing,
    # datasets.Restaurants,
    # datasets.SMSSpam,  # TODO: missing encoding
    # datasets.SMTP,  # highly imbalanced (only 0.4% positive labels)
    datasets.SolarFlare,
    # datasets.TREC07,  # error loading dataset
    datasets.Taxis,
    datasets.TrumpApproval,
    datasets.WaterFlow
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