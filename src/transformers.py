""" Contains custom transformers built for this project. """

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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
