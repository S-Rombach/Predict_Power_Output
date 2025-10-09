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
from sklearn.preprocessing import MinMaxScaler


def evaluate_regressor(
    regressor: BaseEstimator,
    y_true: Union[pd.Series, pd.DataFrame, np.ndarray],
    y_pred: Union[pd.Series, pd.DataFrame, np.ndarray],
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

    def __get_scores(
        y_true: Union[pd.Series, pd.DataFrame, np.ndarray],
        y_pred: Union[pd.Series, pd.DataFrame, np.ndarray],
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Compute regression metrics between true and predicted values.

        Parameters
        ----------
        y_true : Union[pandas.Series, numpy.ndarray]
            True target values.
        y_pred : Union[pandas.Series, numpy.ndarray]
            Predicted target values.
        metrics : Optional[Dict[str, Any]], optional
            Dictionary of metric names and corresponding functions to compute them.
            If None, a default set of common regression metrics is used.

        Returns
        -------
        Dict[str, float]
            Dictionary with metric names as keys and computed metric values as values.

        Notes
        -----
        - RMSE is computed as sqrt(MSE).
        """
        if metrics is None:
            metrics = {
                "MAE": mean_absolute_error,
                "MSE": mean_squared_error,
                "RMSE": lambda y_true, y_pred: np.sqrt(
                    mean_squared_error(y_true, y_pred)
                ),
                "MAPE": mean_absolute_percentage_error,
                "MedAE": median_absolute_error,
                "R2": r2_score,
                "ExplainedVar": explained_variance_score,
            }

        scores = {metric: func(y_true, y_pred) for metric, func in metrics.items()}

        return scores

    _y_true = (
        y_true.values
        if isinstance(y_true, pd.DataFrame) or isinstance(y_true, pd.Series)
        else y_true
    )
    _y_pred = (
        y_pred.values
        if isinstance(y_pred, pd.DataFrame) or isinstance(y_pred, pd.Series)
        else y_pred
    )

    all_feature_scores = {}
    if _y_true.ndim > 1 and _y_true.shape[1] > 1:
        columns = (
            y_true.columns
            if isinstance(y_true, pd.DataFrame)
            else (
                y_pred.columns
                if isinstance(y_pred, pd.DataFrame)
                else [f"x_{i}" for i in range(y_true.shape[1])]
            )
        )
        for i, col in enumerate(columns):
            feature_scores = __get_scores(
                y_true=_y_true[:, i],
                y_pred=_y_pred[:, i],
            )
            all_feature_scores[col] = feature_scores

        scaler = MinMaxScaler()
        scores = __get_scores(scaler.fit_transform(_y_true), scaler.transform(_y_pred))
        scores = {f"{k}_scaled": v for k, v in scores.items()}
    else:
        scores = __get_scores(_y_true, _y_pred)

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

    if len(all_feature_scores) > 0:
        results["single_dim_scores"] = all_feature_scores

    return results
