#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fetches weather data from Open-Meteo for stations in "power_data.csv", merges it with power data.
"""
# -- imports --------------------------------------------------------
import requests_cache
from retry_requests import retry
import os
import pandas as pd
import openmeteo_requests
from src.config import (
    DATA_ORIG_DIR,
    DATA_RAW_DIR,
    DATA_RAW_FILENAME,
    INSTALLATION_DATA_FILENAME,
    POWER_OPENMETEO_WEATHER_FILENAME,
    WC_CODES_FILENAME,
)


def fetch_weather_data(lat, lon, start_date, end_date, wc_codes=None):
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

    start_date = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end_date = pd.to_datetime(end_date).strftime("%Y-%m-%d")

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "weather_code",
            "cloud_cover",
            "snow_depth",
            "sunshine_duration",
            "is_day",
            "direct_radiation",
        ],
    }

    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

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
    hourly_data["cloud_cover"] = hourly_cloud_cover
    hourly_data["snow_depth"] = hourly_snow_depth
    hourly_data["sunshine_duration"] = hourly_sunshine_duration
    hourly_data["is_day"] = hourly_is_day
    hourly_data["direct_radiation"] = hourly_direct_radiation

    if wc_codes is not None:
        hourly_data["weather_description"] = [
            wc_codes.get(code, "Unknown") for code in hourly_weather_code
        ]

    hourly_dataframe = pd.DataFrame(data=hourly_data)
    return hourly_dataframe, {
        "actual_lat": response.Latitude(),
        "actual_lon": response.Longitude(),
        "elevation": response.Elevation(),
        "utc_offset_seconds": response.UtcOffsetSeconds(),
    }


# -- init --------------------------------------------------------
installation_df = pd.read_csv(
    os.path.join(DATA_ORIG_DIR, INSTALLATION_DATA_FILENAME),
    sep=";",
    index_col="installation",
)


power_data_df = pd.read_csv(os.path.join(DATA_RAW_DIR, DATA_RAW_FILENAME), sep=";")
power_data_df["timestamp"] = pd.to_datetime(
    power_data_df["timestamp"], errors="raise", utc=True
)

wc_codes = pd.read_csv(
    os.path.join(DATA_ORIG_DIR, WC_CODES_FILENAME), sep=";", index_col="code_figure"
).to_dict()["code_name"]
""" Weather condition codes and their descriptions. """

for inst in power_data_df["installation"].unique():
    if inst not in installation_df.index:
        raise ValueError(
            f"Installation '{inst}' not found in '{INSTALLATION_DATA_FILENAME}'. Weather data cannot be fetched."
            " Add latitude and longitude of the installation to the file."
        )

    # Get the location for the current installation
    lat = float(installation_df.loc[inst, "latitude"])
    lon = float(installation_df.loc[inst, "longitude"])

    for coord, coord_name in [(lat, "latitude"), (lon, "longitude")]:
        if coord is None or not isinstance(coord, float) or pd.isna(coord):
            raise ValueError(
                f"{coord_name.capitalize()} must be set for installation '{inst}' in '{INSTALLATION_DATA_FILENAME}'"
                f" and be convertible to float."
                f" Is '{installation_df.loc[inst, coord_name]}'"
            )

    power_data_inst_df = power_data_df[power_data_df["installation"] == inst]

    start_date = power_data_inst_df["timestamp"].min()
    end_date = power_data_inst_df["timestamp"].max()

    weather_data, meta_data = fetch_weather_data(
        lat, lon, start_date, end_date, wc_codes=wc_codes
    )
    weather_data = weather_data.reset_index(drop=True)
    weather_data = weather_data.rename(columns={"date": "timestamp"})

    if weather_data is None or weather_data.empty:
        print(
            f"No weather data found for installation '{inst}' with coordinates ({lat}, {lon})."
        )
        continue

    # -- Merge the weather data with the power data --------------------------------------------------------
    weather_data["installation"] = inst
    weather_data = (
        weather_data.set_index("timestamp").resample("15min").ffill().reset_index()
    )
    weather_data["sunshine_duration"] = weather_data["sunshine_duration"].div(4)

    power_data_df = pd.merge(
        power_data_df, weather_data, how="left", on=["timestamp", "installation"]
    )


power_data_df.to_csv(
    os.path.join(DATA_RAW_DIR, POWER_OPENMETEO_WEATHER_FILENAME), sep=";", index=False
)
