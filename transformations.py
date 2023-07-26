from __future__ import annotations

from calendar import calendar

import numpy as np
from river import preprocessing, linear_model
from river.base import Transformer
from river.compose import FuncTransformer
from datetime import datetime
from numbers import Number


def _drop_dates(x: dict) -> dict:
    return {
        key: value for key, value in x.items() if not isinstance(value, datetime)
    }


def _drop_categorical(x: dict) -> dict:
    return {
        key: value for key, value in x.items() if isinstance(value, Number)
    }


class AddIrrelevantFeaturesTransformer(Transformer):
    def __init__(self, seed: int, n_added_features: int):
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.n_added_features = n_added_features
        self._shuffled_keys: list | None = None

    def transform_one(self, x: dict) -> dict:
        x |= {
            f"Irrelevant {i + 1}": self._rng.normal() for i in range(self.n_added_features)
        }
        if self._shuffled_keys is None:
            self._shuffled_keys = list(x.keys())
            self._rng.shuffle(self._shuffled_keys)
        x = {key: x[key] for key in self._shuffled_keys}
        return x


class AddNoisyFeaturesTransformer(Transformer):
    def __init__(self, seed: int, noise_scale: float = 1.0):
        self.seed = seed
        self._noise_scale = noise_scale
        self._rng = np.random.default_rng(seed)
        self._shuffled_keys: list | None = None

    def transform_one(self, x: dict) -> dict:
        x |= {
            f"Noisy {key}": x[key] + self._rng.normal(scale=self._noise_scale)
            for key in list(x.keys())
        }
        if self._shuffled_keys is None:
            self._shuffled_keys = list(x.keys())
            self._rng.shuffle(self._shuffled_keys)
        x = {key: x[key] for key in self._shuffled_keys
             if key in x  #ugly but necessary
             }
        return x



drop_dates = FuncTransformer(_drop_dates)
drop_categorical = FuncTransformer(_drop_categorical)
