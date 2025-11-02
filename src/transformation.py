"""Custom data transformation functions used in different parts of the project."""

from typing import List, Union, Literal
import datetime
from datetime import datetime as dt
import numpy as np
import pandas as pd
import requests_cache
from retry_requests import retry
import openmeteo_requests


def fetch_openmeteo_weather_data(
    lat: float,
    lon: float,
    start_date: Union[pd.Timestamp, datetime.date, str],
    end_date: Union[pd.Timestamp, datetime.date, str],
    wc_codes: Union[dict[int, str], None] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Fetch weather data from the Open-Meteo API for a given latitude and longitude
    between start_date and end_date.

    The function uses caching and retries to minimize redundant requests
    and handle errors robustly.

    This code is copied and adapted from the Open-Meteo documentation
    https://open-meteo.com/en/docs/historical-weather-api

    Parameters
    ----------
    lat : float
        Latitude of the location.
    lon : float
        Longitude of the location.
    start_date : str or pandas.Timestamp
        Start date for the weather data (inclusive).
    end_date : str or pandas.Timestamp
        End date for the weather data (inclusive).
    wc_codes : dict, optional
        Dictionary mapping weather codes to descriptions.

    Returns
    -------
    hourly_dataframe : pandas.DataFrame
        DataFrame containing hourly weather data for the specified location and period.
    meta_data : dict
        Dictionary with metadata about the location and API response.

    Notes
    -----
    - Uses requests_cache for caching API responses.
    - Retries failed requests up to 5 times.
    - Weather variables fetched: weather_code, cloud_cover, snow_depth,
      sunshine_duration, is_day, direct_radiation.
    - Weather code descriptions are added if wc_codes is provided.
    """

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    start_date = (
        pd.to_datetime(start_date).tz_convert(None)
        if hasattr(start_date, "tzinfo")
        else pd.to_datetime(start_date)
    )
    end_date = (
        pd.to_datetime(end_date).tz_convert(None)
        if hasattr(end_date, "tzinfo")
        else pd.to_datetime(end_date)
    )

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    hist_url = "https://archive-api.open-meteo.com/v1/archive"
    curr_url = "https://api.open-meteo.com/v1/forecast"

    requests = []
    responses = []

    hourly_dataframe = None

    today = pd.to_datetime(dt.now().date())

    if start_date < today:
        if end_date < today:
            requests.append((hist_url, start_date, end_date))
        else:
            requests.append((hist_url, start_date, today - pd.Timedelta(days=1)))
            requests.append((curr_url, today, end_date))
    else:
        requests.append((curr_url, start_date, end_date))

    for url, s_date, e_date in requests:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": s_date.strftime("%Y-%m-%d"),
            "end_date": e_date.strftime("%Y-%m-%d"),
            "hourly": [
                "weather_code",
                "cloud_cover",
                "snow_depth",
                "sunshine_duration",
                "is_day",
                "direct_radiation",
            ],
        }

        resps = openmeteo.weather_api(url, params=params)

        # Process first location. Add a for-loop for multiple locations or weather models
        response = resps[0]
        responses.append(response)

        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()
        hourly_weather_code = hourly.Variables(0).ValuesAsNumpy()
        hourly_cloud_cover = hourly.Variables(1).ValuesAsNumpy()
        hourly_snow_depth = hourly.Variables(2).ValuesAsNumpy()
        hourly_sunshine_duration = hourly.Variables(3).ValuesAsNumpy()
        hourly_is_day = hourly.Variables(4).ValuesAsNumpy()
        hourly_direct_radiation = hourly.Variables(5).ValuesAsNumpy()

        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            )
        }

        hourly_data["weather_code"] = hourly_weather_code
        hourly_data["cloud_cover"] = np.nan_to_num(hourly_cloud_cover, nan=0.0)
        hourly_data["snow_depth"] = np.nan_to_num(hourly_snow_depth, nan=0.0)
        hourly_data["sunshine_duration"] = np.nan_to_num(
            hourly_sunshine_duration, nan=0.0
        )
        hourly_data["is_day"] = hourly_is_day
        hourly_data["direct_radiation"] = np.nan_to_num(
            hourly_direct_radiation, nan=0.0
        )

        if wc_codes is not None:
            hourly_data["weather_description"] = [
                wc_codes.get(code, "Unknown") for code in hourly_weather_code
            ]

        hourly_dataframe = (
            pd.DataFrame(data=hourly_data)
            if hourly_dataframe is None
            else pd.concat(
                [hourly_dataframe, pd.DataFrame(data=hourly_data)], ignore_index=True
            )
        )

    return hourly_dataframe, {
        "actual_lat": responses[0].Latitude(),
        "actual_lon": responses[0].Longitude(),
        "elevation": responses[0].Elevation(),
        "utc_offset_seconds": responses[0].UtcOffsetSeconds(),
    }


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
    # ensure names have the correct format
    mandatory_weather_columns = [
        col.strip().lower().replace(" ", "_") for col in mandatory_weather_columns
    ]

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
    time_horizon: Literal["daily", "hourly"] = "daily",
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
    df["date"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(None).dt.normalize()
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(None).dt.hour

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

    df_doy = pd.concat(
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

    if time_horizon == "daily":
        df_doy = df_doy.drop(["hour"], axis=1)
        df_doy = (
            df_doy.groupby(["installation", "date"])
            .agg(agg_dict)
            .reset_index()
            .sort_values(["date"])
        )
    elif time_horizon == "hourly":
        df_doy = (
            df_doy.groupby(["installation", "date", "hour"])
            .agg(agg_dict)
            .reset_index()
            .sort_values(["date", "hour"])
        )
    else:
        raise ValueError(f"Invalid time_horizon: '{time_horizon}'.")

    df_doy = df_doy.drop(["installation"], axis=1)
    return df_doy
