"""
This module provides functions for exploratory data analysis (EDA) on pandas DataFrames.
"""

import numpy as np
from IPython.display import display
import pandas as pd
from statsmodels.robust import mad
import matplotlib.pyplot as plt
import seaborn as sns



def overview(df):
    """
    Creates and prints an overview of the DataFrame including data types, counts, missing values,
    unique values, and some basic statistics.
    """
    from pandas.api.types import is_numeric_dtype

    def normalized_entropy_cat(series: pd.Series) -> float:
        """
        Compute the normalized Shannon entropy of a categorical distribution.

        | Entropy in [0,1]| Interpretation         | Example class distribution   |
        | --------------- | ---------------------- | ---------------------------- |
        | 0.00 - 0.20     | Extremely imbalanced   | e.g. 99% / 1%                |
        | 0.20 - 0.40     | Strongly imbalanced    | e.g. 90% / 10% or 80/10/10   |
        | 0.40 - 0.60     | Moderately imbalanced  | e.g. 70/30 or 60/20/20       |
        | 0.60 - 0.80     | Slightly imbalanced    | e.g. 50/25/25                |
        | 0.80 - 1.00     | Balanced               | e.g. 33/33/33 or 25/25/25/25 |


        Returns 0 if only one class is present, 1 for perfectly uniform distribution.
        """
        counts = series.value_counts(normalize=True)
        entropy = -np.sum(counts * np.log2(counts))
        max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1
        return entropy / max_entropy

    display(
        pd.DataFrame(
            {
                "dtype": df.dtypes,
                "total": df.count(),
                "missing": df.isna().sum(),
                "missing%": df.isna().mean() * 100,
                "n_uniques": df.nunique(),
                "uniques%": df.nunique() / df.shape[0] * 100,
                "uniques": [
                    sorted((str(x) for x in df[col].unique())) for col in df.columns
                ],
                "non-numeric": [
                    list(
                        df[col][pd.to_numeric(df[col], errors="coerce").isna()].unique()
                    )
                    for col in df.columns
                ],
                "dev from mean": [
                    (
                        (
                            round(
                                ((df[col].mean() - df[col].min()) / df[col].std()), 1
                            ),
                            round(
                                ((df[col].max() - df[col].mean()) / df[col].std()), 1
                            ),
                        )
                        if is_numeric_dtype(df[col]) and df[col].notna().any()
                        else pd.NA
                    )
                    for col in df.columns
                ],
                "most/least freq": [
                    (
                        (
                            {
                                df[col]
                                .value_counts()
                                .index[i]: list(df[col].value_counts())[i]
                                for i in (0, -1)
                            }
                        )
                        if not is_numeric_dtype(df[col])
                        else pd.NA
                    )
                    for col in df.columns
                ],
                "norm entropy": [
                    (
                        round(normalized_entropy_cat(df[col]), 2)
                        if isinstance(df[col].dtype, pd.CategoricalDtype)
                        else pd.NA
                    )
                    for col in df.columns
                ],
            }
        )
    )


def mark_outliers_mad(
    df,
    std=3,
    show_cum=False,
    show_interesting_rows=False,
    interesting_rows=None,
    return_masks=False,
):
    """
    Detect and visualize outliers in numeric columns using the MAD (Median Absolute Deviation) method.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the data.
    std : float, optional
        Threshold in MAD units to define outliers (default is 3).
    show_cum : bool, optional
        If True, show cumulative distribution plots with outlier thresholds.
    show_interesting_rows : bool, optional
        If True, show histogram restricted to interesting_rows.
    interesting_rows : pd.Series[bool], optional
        Boolean mask indicating rows of interest.
    return_masks : bool, optional
        If True, return dictionary of boolean masks marking outliers per column.

    Returns
    -------
    dict (optional)
        Dictionary of boolean Series with outlier masks for each numeric column.
    """

    def get_mad_outliers_mask(df, column, std=3):
        x = df.loc[df[column].notna(), column]
        outliers = (abs(x - x.median()) / mad(x)) >= std

        mask = pd.Series(False, index=df.index)
        mask.loc[x.index] = outliers
        return mask

    outlier_masks = {}
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    column_config = {col: {} for col in numeric_cols}

    show_interesting_rows = show_interesting_rows and (interesting_rows is not None)

    ncols = 1 + sum([show_cum, show_interesting_rows])
    fig, ax = plt.subplots(
        nrows=len(numeric_cols), ncols=ncols, figsize=(14, len(numeric_cols) * 2 + 2)
    )

    if len(numeric_cols) == 1 and ncols == 1:
        ax = np.array([[ax]])
    elif len(numeric_cols) == 1:
        ax = np.atleast_2d(ax)
    elif ncols == 1:
        ax = np.atleast_2d(ax).T

    for idx, col in enumerate(numeric_cols):
        config = column_config[col]
        axe = ax[idx, 0]
        outlier_mask = get_mad_outliers_mask(df, col, std)

        if return_masks:
            outlier_masks[col] = outlier_mask

        df_temp = pd.concat(
            [df, pd.DataFrame({f"mad_{std}_outlier": outlier_mask})], axis=1
        )

        sns.histplot(data=df_temp, x=col, ax=axe, hue=f"mad_{std}_outlier", bins=100)
        axe.set_yscale("log")
        axe.set_title(f"Histogram: {col}")

        if show_interesting_rows:
            prev_axe = axe
            axe = ax[idx, 1]
            axe.set_xlim(prev_axe.get_xlim())
            axe.set_ylim(prev_axe.get_ylim())

            df_temp = pd.concat(
                [df[[col]].copy(), pd.DataFrame({f"mad_{std}_outlier": outlier_mask})],
                axis=1,
            )
            df_temp.loc[~interesting_rows, col] = np.nan

            sns.histplot(
                data=df_temp, x=col, ax=axe, hue=f"mad_{std}_outlier", bins=100
            )
            axe.set_yscale("log")
            axe.set_title(f"Subset histogram: {col}")

        if show_cum:
            lower_threshold = df[~outlier_mask][col].min()
            upper_threshold = df[~outlier_mask][col].max()

            axe = ax[idx, 1 + show_interesting_rows]
            sns.ecdfplot(df[col].dropna(), ax=axe)

            axe.axvline(
                x=lower_threshold, color="blue", linestyle="--", label="lower threshold"
            )
            axe.axvline(
                x=upper_threshold, color="red", linestyle="--", label="upper threshold"
            )

            col_data = df[col].dropna()
            outlier_low = (col_data < lower_threshold).mean()
            outlier_high = (col_data > upper_threshold).mean()

            axe.set_title(
                f"CDF: {col}\n"
                f"Outliers < {lower_threshold:.2f}: {outlier_low:.3f} | "
                f"> {upper_threshold:.2f}: {outlier_high:.3f}"
            )
            axe.legend()

    plt.suptitle(f"MAD Outlier Detection (±{std} MAD)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    if return_masks:
        return outlier_masks
