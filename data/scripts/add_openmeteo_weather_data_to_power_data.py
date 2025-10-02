#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fetches weather data from Open-Meteo for stations in "power_data.csv", merges it with power data.
"""
# -- imports --------------------------------------------------------
import os
import pandas as pd
from src.config import (
    DATA_ORIG_DIR,
    DATA_RAW_DIR,
    DATA_RAW_FILENAME,
    INSTALLATION_DATA_FILENAME,
    OPENMETEO_WEATHER_FILENAME,
    POWER_OPENMETEO_WEATHER_FILENAME,
    WC_CODES_FILENAME,
)
from src.transformation import fetch_openmeteo_weather_data


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

weather_data_df = pd.read_csv(
    os.path.join(DATA_ORIG_DIR, OPENMETEO_WEATHER_FILENAME), sep=";"
)
weather_data_df["timestamp"] = pd.to_datetime(
    weather_data_df["timestamp"], errors="raise", utc=True
)

for inst in power_data_df["installation"].unique():
    if inst not in weather_data_df["installation"].unique():
        raise ValueError(
            f"Installation '{inst}' not found in '{OPENMETEO_WEATHER_FILENAME}'."
            " Fetch weather data first."
            " Ensure fetched weather data covers the whole time range of the power data."
        )

    weather_data = weather_data_df[weather_data_df["installation"] == inst].copy()

    if weather_data is None or weather_data.empty:
        print(
            f"No weather data found for installation '{inst}' in '{OPENMETEO_WEATHER_FILENAME}'."
        )
        continue

    if (
        weather_data["timestamp"].min()
        > power_data_df[power_data_df["installation"] == inst]["timestamp"].min()
    ):
        raise ValueError(
            f"Weather data for installation '{inst}' in '{OPENMETEO_WEATHER_FILENAME}'"
            " does not cover the entire time range of the power data."
            f" Earliest weather data timestamp is {weather_data['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')},"
            f" but earliest power data timestamp is"
            f" {power_data_df[power_data_df['installation'] == inst]['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')}."
        )
    if (
        weather_data["timestamp"].max()
        < power_data_df[power_data_df["installation"] == inst]["timestamp"].max()
    ):
        raise ValueError(
            f"Weather data for installation '{inst}' in '{OPENMETEO_WEATHER_FILENAME}'"
            " does not cover the entire time range of the power data."
            f" Latest weather data timestamp is {weather_data['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')},"
            f" but latest power data timestamp is"
            f" {power_data_df[power_data_df['installation'] == inst]['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')}."
        )
    
    # -- Merge the weather data with the power data --------------------------------------------------------
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
