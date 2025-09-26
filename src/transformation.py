"""Custom data transformation functions used in different parts of the project."""

from typing import List, Tuple
import numpy as np
import pandas as pd


def one_hot_encode_weather_descriptions(
    df: pd.DataFrame,
    weather_column: str = "weather_description",
    mandatory_weather_columns: List[str] = [],
) -> pd.DataFrame:
    """
    One-hot encodes the specified weather description column in the given DataFrame.
    Converts the values in the `weather_column` to lowercase, strips whitespace, and replaces spaces with underscores.
    Ensures that the resulting DataFrame contains columns for all specified `mandatory_weather_columns`, adding missing columns with zeros.
    Args:
        df (pd.DataFrame): Input DataFrame containing weather data.
        weather_column (str, optional): Name of the column containing weather descriptions to encode. Defaults to "weather_description".
        mandatory_weather_columns (List[str], optional): List of weather description columns that must be present in the output. Missing columns are added with zeros. Defaults to [].
    Returns:
        pd.DataFrame: DataFrame containing one-hot encoded weather description columns, with all mandatory columns included.
    """
    
    df = df.copy()
    ohe = pd.get_dummies(df[weather_column])
    ohe = ohe.rename(
        columns={col: col.strip().lower().replace(" ", "_") for col in ohe.columns}
    )

    if len(mandatory_weather_columns) > 0:
        ohe = ohe[[col for col in ohe.columns if col in mandatory_weather_columns]]

    for col in mandatory_weather_columns:
        if col not in ohe.columns:
            ohe[col] = 0

    return ohe


def prepare_aggregate_openmeteo_data(

    df: pd.DataFrame,
    weather_column: str = "weather_description",
    mandatory_weather_columns: List[str] = [],
    exclude_columns: List[str] = [],
) -> pd.DataFrame:
    """
    Prepares and aggregates OpenMeteo weather data for further analysis.
    This function processes a DataFrame containing weather data by:
    - Normalizing and converting the 'timestamp' column to a date column.
    - Summing or averaging selected weather-related columns, excluding specified columns.
    - One-hot encoding weather descriptions, ensuring mandatory weather columns are present.
    - Aggregating data by 'installation' and 'date' using appropriate aggregation methods.
    - Sorting the resulting DataFrame by date and removing the 'installation' column.
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing weather data, including a 'timestamp' column.
    weather_column : str, optional
        Name of the column containing weather descriptions to be one-hot encoded (default is "weather_description").
    mandatory_weather_columns : List[str], optional
        List of weather description columns that must be present in the output (default is empty list).
    exclude_columns : List[str], optional
        List of columns to exclude from aggregation (default is empty list).
    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with processed weather features, indexed by date.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize().dt.tz_convert(None)

    sum_cols = [
        c
        for c in ["sol_prod", "sunshine_duration", "direct_radiation"]
        if c in df.columns and c not in exclude_columns
    ]
    mean_cols = [
        c
        for c in ["cloud_cover", "snow_depth", "is_day"]
        if c in df.columns and c not in exclude_columns
    ]

    ohe = one_hot_encode_weather_descriptions(
        df, weather_column, mandatory_weather_columns
    )

    agg_dict = (
        {c: "sum" for c in sum_cols}
        | {c: "mean" for c in mean_cols}
        | {c: "mean" for c in ohe.columns}
    )

    df_doy = (
        pd.concat(
            [
                df.drop(
                    ["weather_description"]
                    + [col for col in exclude_columns if col in df.columns],
                    axis=1,
                ),
                ohe,
            ],
            axis=1,
        )
        .groupby(["installation", "date"])
        .agg(agg_dict)
        .reset_index()
    )

    df_doy = df_doy.sort_values(["date"])

    df_doy = df_doy.drop(["installation"], axis=1)
    return df_doy
