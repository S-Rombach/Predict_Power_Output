"""Provides utility functions for evaluating regression models."""

import os
import json
import pickle
from datetime import datetime
from typing import Union, Optional, List, Dict, Any, Sequence, Literal

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
    median_absolute_error,
    explained_variance_score,
)


def evaluate_regressor(
    regressor: BaseEstimator,
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    timestamp: datetime,
    model_purpose: str,
    special_features: str,
) -> Dict[str, Any]:
    """
    Evaluate a fitted scikit-learn regressor against true and predicted values,
    compute standard regression metrics, and build a descriptive model identifier.

    Parameters
    ----------
    regressor : BaseEstimator
        The trained regressor instance (any scikit-learn compatible estimator).
    y_true : Union[pandas.Series, numpy.ndarray]
        True target values.
    y_pred : Union[pandas.Series, numpy.ndarray]
        Predicted target values from the regressor.
    timestamp : datetime.datetime
        Timestamp used for naming the model run and result tracking.
    model_purpose : str
        Short description of the model purpose (e.g. "baseline", "forecasting").
    special_features : str
        Description of special features, feature set or preprocessing used.

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - model_name : str
            Unique identifier composed of timestamp, model class,
            R² score, model purpose and special features.
        - timestamp : str
            Timestamp as string in format "%Y%m%d%H%M%S".
        - model_purpose : str
            Purpose description from input.
        - model_class : str
            Name of the regressor class.
        - special_features : str
            Feature description from input.
        - MAE, MSE, RMSE, MAPE, MedAE, R2, ExplainedVar : float
            Regression metrics values.

    Notes
    -----
    - RMSE is computed as sqrt(MSE).
    - Model name encodes performance via scaled R² for easier comparison.
    """

    regression_metrics = {
        "MAE": mean_absolute_error,
        "MSE": mean_squared_error,
        "RMSE": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mean_absolute_percentage_error,
        "MedAE": median_absolute_error,
        "R2": r2_score,
        "ExplainedVar": explained_variance_score,
    }

    scores = {
        metric: func(y_true, y_pred) for metric, func in regression_metrics.items()
    }

    timestamp_str = timestamp.strftime("%Y%m%d%H%M%S")

    model_name = f"{timestamp_str}_{regressor.__class__.__name__}_r2{int(scores.get('R2', 0) * 10000):05d}_{model_purpose}_{special_features}"

    results = {
        "model_name": model_name,
        "timestamp": timestamp_str,
        "model_purpose": model_purpose,
        "model_class": regressor.__class__.__name__,
        "special_features": special_features,
        **scores,
    }

    return results
