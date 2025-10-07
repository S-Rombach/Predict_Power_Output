"""Contains custom transformers built for this project."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Union


class MissingFlagTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = list([f"x{i}" for i in range(X.shape[1])])
        return self

    def transform(self, X):
        X_ = X.copy()

        na_mask = (
            pd.isna(X_).to_numpy() if isinstance(X_, pd.DataFrame) else np.isnan(X_)
        )
        return na_mask

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_

        output_features = [f"{c}_isna" for c in input_features]

        return np.array(output_features)


class CategoryFromThresholdTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, thresholds):
        self.thresholds = thresholds

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = list([f"x{i}" for i in range(X.shape[1])])

        if any(isinstance(t, list) for t in self.thresholds):
            if not all(isinstance(t, list) for t in self.thresholds):
                raise ValueError(
                    "'thresholds' must either be a list of skalars or a list of lists."
                )
            self.thresholds = [sorted(t) for t in self.thresholds]
        else:
            self.thresholds = [sorted(self.thresholds)]

        if len(self.feature_names_in_) != len(self.thresholds):
            raise ValueError(
                "number of input features and length of 'thresholds' must be the same."
            )
        return self

    def _get_cat_from_threshold(self, value, thresholds):
        if pd.isna(value):
            return "NA"
        if value >= thresholds[-1]:
            result = f"{thresholds[-1]}+"
        elif value < thresholds[0]:
            result = f"<{thresholds[0]}"
        else:
            idx = next(i for i, val in list(enumerate(thresholds)) if val > value)

            result = f"[{thresholds[idx-1]}-{thresholds[idx]})"
        return result

    def transform(self, X):
        X_ = X.copy()

        if isinstance(X_, pd.DataFrame):
            X_ = X_.values

        res = np.column_stack(
            [
                [self._get_cat_from_threshold(x, th) for x in X_[:, i]]
                for i, th in enumerate(self.thresholds)
            ]
        )

        return res

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_

        return np.array([f"{c}_cat" for c in input_features])


class DayOfYearTransformer(BaseEstimator, TransformerMixin):
    """
    DayOfYearTransformer applies a trigonometric transformation (sin or cos) to the day-of-year
    component of datetime features, enabling cyclical encoding for machine learning models.
    Parameters
    ----------
    trig_function : str
        The trigonometric function to apply. Must be either "sin" or "cos".
    Methods
    -------
    fit(X, y=None)
        Fits the transformer to the input data. Stores feature names for later use.
    transform(X)
        Transforms the input datetime features into their cyclical representation using the
        specified trigonometric function.
    get_feature_names_out(input_features=None)
        Returns the output feature names after transformation.
    Notes
    -----
    - Supports input as pandas Series, pandas DataFrame, or numpy ndarray with datetime64 dtype.
    - Handles leap years correctly when calculating the day-of-year and days-in-year.
    - Raises ValueError for unsupported input types or invalid trig_function values.
    """

    def __init__(self, trig_function):
        self.trig_function = trig_function

    def _extract_dayofyear(
        self, x: Union[pd.Series, pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        if isinstance(x, pd.Series) and np.issubdtype(x.dtype, np.datetime64):
            return x.dt.dayofyear.to_frame()
        elif isinstance(x, pd.DataFrame):
            x = x.copy()
            for col in x.columns:
                x[col] = x[col].dt.dayofyear
            return x
        elif isinstance(x, np.ndarray):
            days = x.astype("datetime64[D]")
            years = days.astype("datetime64[Y]")
            doy = (days - years).astype(int) + 1
            return doy.astype(int)
        else:
            raise ValueError(
                "Input must be a pandas Series with datetime64 dtype"
                " or a DataFrame with datetime64 columns."
            )

    def _extract_days_in_year(
        self, x: Union[pd.Series, pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        if isinstance(x, pd.Series) and np.issubdtype(x.dtype, np.datetime64):
            return x.dt.is_leap_year.map({True: 366, False: 365}).to_frame()
        elif isinstance(x, pd.DataFrame):
            x = x.copy()
            for col in x.columns:
                x[col] = x[col].dt.is_leap_year.map({True: 366, False: 365})
            return x
        elif isinstance(x, np.ndarray):
            years = x.astype("datetime64[Y]").astype(int) + 1970
            leap = (years % 4 == 0) & ((years % 100 != 0) | (years % 400 == 0))
            return np.where(leap, 366, 365)
        else:
            raise ValueError(
                "Input must be a pandas Series with datetime64 dtype"
                " or a DataFrame with datetime64 columns."
            )

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = list([f"x{i}" for i in range(X.shape[1])])
        return self

    def transform(self, X):
        X_ = X.copy()

        if self.trig_function == "sin":
            return np.sin(
                self._extract_dayofyear(X_) / self._extract_days_in_year(X_) * 2 * np.pi
            )
        elif self.trig_function == "cos":
            return np.cos(
                self._extract_dayofyear(X_) / self._extract_days_in_year(X_) * 2 * np.pi
            )
        else:
            raise ValueError("trig_function must be 'sin' or 'cos'")

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([f"doy_{self.trig_function}"])
        return np.array([f"{c}_doy_{self.trig_function}" for c in input_features])


class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that creates lagged features for time series data.
    Parameters
    ----------
    lags : int or list of int
        The lag(s) to apply to the input features. If an integer is provided, a single lag is used.
        If a list of integers is provided, multiple lagged features are created for each lag value.
    Attributes
    ----------
    feature_names_in_ : list of str
        Names of the input features seen during fit.
    Methods
    -------
    fit(X, y=None)
        Fits the transformer to the data, storing feature names.
    transform(X)
        Transforms the input data by adding lagged features.
    get_feature_names_out(input_features=None)
        Returns the names of the output features after transformation.
    Notes
    -----
    - For pandas DataFrame input, lagged columns are created with suffixes indicating the lag.
    - For numpy array input, lagged features are stacked and returned as a new array.
    - Useful for preparing time series data for supervised learning models.
    """

    def __init__(self, lags: Union[int, list[int]]):
        self.lags = lags if isinstance(lags, list) else [lags]

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y=None
    ) -> "LagFeatureTransformer":
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = list([f"x{i}" for i in range(X.shape[1])])
        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        X_ = X.copy()

        if isinstance(X_, pd.DataFrame):
            return X_.add_suffix("_lag").shift(self.lags)
        else:
            return np.vstack([np.roll(X_, lag) for lag in self.lags]).T

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is None:
            input_features = self.feature_names_in_

        return np.array([f"{c}_lag_{lag}" for lag in self.lags for c in input_features])
