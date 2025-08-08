"""
This module provides functions for exploratory data analysis (EDA) on pandas DataFrames.
"""

import numpy as np
from IPython.display import display
import pandas as pd

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
                "uniques": [sorted((str(x) for x in df[col].unique())) for col in df.columns],
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
                                df[col].value_counts().index[i]: list(
                                    df[col].value_counts()
                                )[i]
                                for i in (0, -1)
                            }
                        )
                        if not is_numeric_dtype(df[col])
                        else pd.NA
                    )
                    for col in df.columns
                ],
                "norm entropy": [
                    round(normalized_entropy_cat(df[col]), 2)
                    if isinstance(df[col].dtype, pd.CategoricalDtype)
                    else pd.NA
                    for col in df.columns
                ],
            }
        )
    )