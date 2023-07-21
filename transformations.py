from river import preprocessing, linear_model
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


drop_dates = FuncTransformer(_drop_dates)
drop_categorical = FuncTransformer(_drop_categorical)

target_scaler = preprocessing.TargetStandardScaler(regressor=linear_model.LinearRegression(intercept_lr=0.15))

