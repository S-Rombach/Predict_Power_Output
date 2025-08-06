"""
Provides utility functions to evaluate classification models, including scoring, 
confusion matrix reconstruction, and formatting of classification reports.
"""

import os
import json
import pickle
from datetime import datetime
from typing import Union, Optional, List, Dict, Any, Sequence, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.ndimage import uniform_filter1d
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import learning_curve


def evaluate_classifier(
    classifier: BaseEstimator,
    labels: List[Union[str, int]],
    target_truth: Union[pd.Series, np.ndarray],
    target_pred: Union[pd.Series, np.ndarray],
    target_pred_proba: Optional[np.ndarray],
    timestamp: datetime,
    model_purpose: str,
    special_features: str,
    avg_mode: str = "weighted",
) -> Dict[str, Any]:
    """
    Evaluates a trained classifier on given predictions and returns a dictionary with performance metrics.

    Parameters
    ----------
    classifier : sklearn.base.BaseEstimator
        Trained classifier object (used for metadata only).
    labels : list
        List of class labels to include in confusion matrix and classification report.
    target_truth : array-like
        True target labels.
    target_pred : array-like
        Predicted target labels.
    target_pred_proba : array-like or None
        Predicted probabilities (used for ROC AUC calculation). Can be None.
    timestamp : datetime
        Timestamp string for unique model identification.
    model_purpose : str
        Description of the model's purpose or context.
    special_features : str
        Indicator string for specific feature configuration used.
    avg_mode : str, default="macro"
        Averaging method for precision, recall, and F1 score.
        Options: {"micro", "macro", "weighted", "samples"}

    Returns
    -------
    dict
        Dictionary containing:
        - model metadata (name, timestamp, class, purpose)
        - evaluation metrics (f1, recall, precision, balanced accuracy, ROC AUC)
        - confusion matrix entries (flattened)
        - classification report (as dict)
    """
    precision_score_val = precision_score(target_truth, target_pred, average=avg_mode)
    recall_score_val = recall_score(target_truth, target_pred, average=avg_mode)
    f1_score_val = f1_score(target_truth, target_pred, average=avg_mode)
    balanced_accuracy_val = balanced_accuracy_score(target_truth, target_pred)

    confusion_matrix_val = confusion_matrix(
        target_truth, target_pred, labels=labels
    )


    roc_auc_score_val = pd.NA
    if target_pred_proba is not None:
        if target_pred_proba.ndim == 2:
            roc_auc_score_val = roc_auc_score(
                target_truth, target_pred_proba, multi_class="ovo", labels=labels
            )
        else:
            roc_auc_score_val = roc_auc_score(
                target_truth==labels[-1], target_pred_proba
            )

    timestamp_str = timestamp.strftime("%Y%m%d%H%M%S")

    model_name = f"{timestamp_str}_{classifier.__class__.__name__}_f1{int(f1_score_val * 10000):05d}_{model_purpose}_{special_features}"

    

    results = {
        "model_name": model_name,
        "timestamp": timestamp_str,
        "model_purpose": model_purpose,
        "model_class": classifier.__class__.__name__,
        "special_features": special_features,
        "predicts": labels,
        "avg_mode": avg_mode,
        "f1": f1_score_val,
        "recall": recall_score_val,
        "precision": precision_score_val,
        "bal_accuracy": balanced_accuracy_val,
        "roc_auc_score": roc_auc_score_val if pd.notna(roc_auc_score_val) else '',
        "conf_matrix": {
            f"C_{true_class}_{pred_class}": int(
                confusion_matrix_val[true_class, pred_class]
            )
            for true_class in range(confusion_matrix_val.shape[0])
            for pred_class in range(confusion_matrix_val.shape[1])
        },
        "classification_report": classification_report(
            target_truth, target_pred, digits=3, output_dict=True, labels=labels
        ),
    }

    return results

def get_confusion_matrix_from_results_as_df(results):
    """
    Reconstructs the confusion matrix from the results dictionary as a labeled pandas DataFrame.

    Parameters
    ----------
    results : dict
        Dictionary returned by `evaluate_classifier()` containing:
        - "predicts": list of class labels
        - "conf_matrix": flattened confusion matrix with keys like "C_0_1"

    Returns
    -------
    pandas.DataFrame
        Confusion matrix as a DataFrame with labeled rows (true classes) and columns (predicted classes).
    """
    labels = results["predicts"]
    conf_matrix = results["conf_matrix"]
    rows = {}
    for k, v in conf_matrix.items():
        sp = k.split("_")
        ri = int(sp[1])
        ci = int(sp[2])

        if ri not in rows:
            rows[ri] = {}

        rows[ri][ci] = v

    df = pd.DataFrame([rows[i] for i in sorted(rows.keys())])
    df = df.rename(index={k: v for k, v in enumerate(labels)})
    df = df.rename(columns={k: v for k, v in enumerate(labels)})

    return df

def get_classification_report_from_results_as_df(results, decimals=3):
    """
    Converts the classification report stored in the results dictionary into a formatted DataFrame.

    Parameters
    ----------
    results : dict
        Dictionary returned by `eval_mevaluate_classifierodel()` containing the key "classification_report"
        with the classification report as a nested dict.
    decimals : int, default=3
        Number of decimal places to round the metric values.

    Returns
    -------
    pandas.DataFrame
        Formatted classification report as DataFrame with string values and support as integers.
        The 'accuracy' row includes only the support value for display purposes.
    """
    df = pd.DataFrame(results["classification_report"]).T.round(decimals)

    df["support"] = df["support"].astype(int)

    df = df.astype('str')

    if "accuracy" in df.index:
        df.loc["accuracy", "support"] = df.loc[ "macro avg", "support"]
        df.loc["accuracy", "precision"] = ''
        df.loc["accuracy", "recall"] = ''

    return df

def save_learning_curve_data(
    model_dir: str,
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    cv: int = 5,
    scoring: str = "f1_macro",
    n_jobs: int = -1,
    train_sizes: Union[np.ndarray, Sequence[float]] = np.linspace(0.1, 1.0, 10),
) -> None:
    """
    Loads model components, computes learning curve data using cross-validation,
    and saves training/validation scores and summary statistics to CSV and JSON files.

    Parameters
    ----------
    model_dir : str
        Path to the directory containing '.model.pkl', '.pipeline.pkl', and
        '.label_encoder.pkl' files.

    X : pd.DataFrame
        Input features (not preprocessed).

    y : Union[pd.Series, np.ndarray]
        Target labels.

    cv : int, optional (default=5)
        Number of cross-validation folds.

    scoring : str, optional (default="f1_macro")
        Scoring metric to evaluate performance.

    n_jobs : int, optional (default=-1)
        Number of parallel jobs to run.

    train_sizes : array-like, optional (default=np.linspace(0.1, 1.0, 10))
        Relative or absolute numbers of training examples to use.

    Returns
    -------
    None
        Writes training/validation learning curve scores and summary JSON to disk.
    """

    model_name = os.path.basename(os.path.normpath(model_dir))
    path_filename_prefix = os.path.join(model_dir, model_name)

    with open(f"{path_filename_prefix}.model.pkl", "rb") as f:
        classifier = pickle.load(f)

    with open(f"{path_filename_prefix}.pipeline.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    with open(f"{path_filename_prefix}.label_encoder.pkl", "rb") as f:
        labelencoder = pickle.load(f)

    features_train_proc = preprocessor.transform(X)
    target_train_enc = labelencoder.transform(y)


    start_timestamp = datetime.now()
    training_sizes, train_scores, val_scores = learning_curve(
        estimator=classifier,
        X=features_train_proc,
        y=target_train_enc,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
    )
    end_timestamp = datetime.now()
    td = end_timestamp - start_timestamp

    df_train = pd.DataFrame(
        {"train_sizes": train_sizes}
        | {
            f"scores_{i}": score
            for rows in train_scores
            for i, score in enumerate(rows)
        }
    )
    df_train.to_csv(
        f"{path_filename_prefix}.learning_curve_train_scores.csv", index=False
    )

    df_train = pd.DataFrame(
        {"train_sizes": train_sizes}
        | {f"scores_{i}": score for rows in val_scores for i, score in enumerate(rows)}
    )
    df_train.to_csv(
        f"{path_filename_prefix}.learning_curve_validation_scores.csv", index=False
    )

    summary = {
        "cv": cv,
        "scoring": scoring,
        "n_jobs": n_jobs,
        "train_sizes": list(train_sizes),
        "training_time_sec": td.seconds,
    }

    with open(f"{path_filename_prefix}.learning_curve_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

def plot_learning_curves_from_file(
    model_dir: str,
    just_save: bool = True,
) -> None:
    """
    Loads learning curve data from CSV and JSON files in the specified model directory,
    generates a learning curve plot with rolling averages and reference lines,
    and saves the plot as a PNG file.

    Parameters
    ----------
    model_dir : str
        Path to the directory containing learning curve CSV and JSON files.
        Required files: '.learning_curve_train_scores.csv',
        '.learning_curve_validation_scores.csv', '.learning_curve_summary.json',
        and '.results.json'.

    just_save : bool, optional (default=True)
        If True, saves the plot to disk without displaying it.
        If False, displays the plot interactively.

    Returns
    -------
    None
        The plot is saved as a PNG file in the given directory.
    """

    model_name = os.path.basename(os.path.normpath(model_dir))
    path_filename_prefix = os.path.join(model_dir, model_name)

    df_train = pd.read_csv(
        f"{path_filename_prefix}.learning_curve_train_scores.csv",
        index_col="train_sizes",
    )
    df_val = pd.read_csv(
        f"{path_filename_prefix}.learning_curve_validation_scores.csv",
        index_col="train_sizes",
    )

    train_sizes = list(df_train.index)
    train_mean = df_train.mean(axis=1)
    val_mean = df_val.mean(axis=1)

    with open(f"{path_filename_prefix}.learning_curve_summary.json", "r") as f:
        summary = json.load(f)

    scoring_mode = summary["scoring"]

    with open(f"{path_filename_prefix}.results.json", "r") as f:
        results = json.load(f)

    fig, ax = plt.subplots(figsize=(16, 5))

    (train_line,) = ax.plot(train_sizes, train_mean, label="Training score")
    (val_line,) = ax.plot(train_sizes, val_mean, label="Validation score")

    train_color = train_line.get_color()
    val_color = val_line.get_color()

    y_lines_min = [train_mean.min(), val_mean.min()]
    y_lines_max = [train_mean.max(), val_mean.max()]

    for y in y_lines_min:
        ax.axhline(y, color="red", linestyle="--", linewidth=0.8, alpha=0.2)
    for y in y_lines_max:
        ax.axhline(y, color="green", linestyle="--", linewidth=0.8, alpha=0.2)

    train_rolling = uniform_filter1d(train_mean, size=3, mode="nearest")
    val_rolling = uniform_filter1d(val_mean, size=3, mode="nearest")

    ax.plot(
        train_sizes,
        train_rolling,
        label="Train rolling avg",
        linestyle=":",
        linewidth=1.5,
        color=train_color,
        alpha=0.5,
    )
    ax.plot(
        train_sizes,
        val_rolling,
        label="Val rolling avg",
        linestyle=":",
        linewidth=1.5,
        color=val_color,
        alpha=0.5,
    )

    yticks = sorted(set(ax.get_yticks().tolist() + y_lines_min + y_lines_max))
    ax.set_yticks(yticks)

    ax.set_xlabel("Training set size")
    ax.set_ylabel(scoring_mode.replace("_", " ").capitalize())

    ax.set_title(f"Learning curve for model '{results["timestamp"]}'")

    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{path_filename_prefix}.learning_curve_img.png", format="png")
    if just_save:
        plt.close(fig)
    else:
        fig.show()

def save_feature_importances_data(
    model_dir: str,
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
) -> None:
    """
    Loads a trained classifier, preprocessor, and label encoder from disk,
    computes permutation feature importances based on weighted F1 score,
    and saves the results as a CSV file.

    Parameters
    ----------
    model_dir : str
        Path to directory containing '.model.pkl', '.pipeline.pkl', and
        '.label_encoder.pkl' files.

    X : pd.DataFrame
        Input features (not preprocessed).

    y : Union[pd.Series, np.ndarray]
        Target labels.

    Returns
    -------
    None
        Writes a CSV file with feature importances to disk.
    """

    model_name = os.path.basename(os.path.normpath(model_dir))
    path_filename_prefix = os.path.join(model_dir, model_name)

    with open(f"{path_filename_prefix}.model.pkl", "rb") as f:
        classifier = pickle.load(f)

    with open(f"{path_filename_prefix}.pipeline.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    with open(f"{path_filename_prefix}.label_encoder.pkl", "rb") as f:
        labelencoder = pickle.load(f)

    features_train_proc = preprocessor.transform(X)
    target_train_enc = labelencoder.transform(y)

    target_train_pred = classifier.predict(features_train_proc)


    f1_base = f1_score(target_train_enc, target_train_pred, average="weighted")


    importances_of_permutations = []

    for feat in X.columns:
        features_permutated = X.copy()
        permutation_run_scores = []
        for i in range(5):
            series_perm = features_permutated[feat].sample(
                frac=1, replace=False, random_state=i
            )

            series_perm.reset_index(drop=True, inplace=True)
            features_permutated[feat] = series_perm
            preprocessed_features_permutated = preprocessor.transform(
                features_permutated
            )

            f1_permutated = f1_score(
                target_train_enc,
                classifier.predict(preprocessed_features_permutated),
                average="weighted",
            )
            permutation_run_scores.append(f1_base - f1_permutated)
        importances_of_permutations.append(permutation_run_scores)


    feature_importances = pd.DataFrame(
        {"feature": X.columns}
        | {
            "importance_mean": np.mean(perm_imp)
            for perm_imp in importances_of_permutations
        }
        | {
            f"imp_{i}": imp
            for perm_imp in importances_of_permutations
            for i, imp in enumerate(perm_imp)
        }
    )


    feature_importances.sort_values(by="importance_mean", ascending=False, inplace=True)
    feature_importances.to_csv(
        f"{path_filename_prefix}.feature_importances.csv", index=False
    )

def plot_feature_importances_from_file(model_dir: Union[str, bytes, os.PathLike], just_save: bool = True) -> None:
    """
    Plots feature importances from a CSV file and saves the plot as PNG.

    Parameters
    ----------
    model_dir : str | PathLike
        Path to the directory containing the model-related `.feature_importances.csv` file.
        The filename must match the directory name (e.g. `model_dir/model_dir.feature_importances.csv`).
    
    just_save : bool, default=True
        If True, the plot is saved and the figure is closed. If False, the plot is shown interactively.

    Returns
    -------
    None
    """

    model_name = os.path.basename(os.path.normpath(model_dir))
    path_filename_prefix = os.path.join(model_dir, model_name)

    feature_importances = pd.read_csv(
        f"{path_filename_prefix}.feature_importances.csv", index_col="feature"
    )

    fig, ax = plt.subplots(figsize=(16, 0.3 * len(feature_importances)))
    sns.barplot(
        x=feature_importances["importance_mean"], y=feature_importances.index, ax=ax
    )
    ax.set_title(f"feature importances for model '{model_name}'")
    for i, (val, label) in enumerate(
        zip(feature_importances["importance_mean"], feature_importances.index)
    ):
        ax.text(val, i, f" {val:.4f}", va="center", ha="left", fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{path_filename_prefix}.feature_importances.png", format="png")
    if just_save:
        plt.close(fig)
    else:
        fig.show()
