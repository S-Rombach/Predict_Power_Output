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
)
from src.data import merge_weather_with_power_data


# -- init --------------------------------------------------------
installation_df = pd.read_csv(
    os.path.join(DATA_ORIG_DIR, INSTALLATION_DATA_FILENAME),
    sep=";",
    index_col="installation",
)


weather_data_df = pd.read_csv(
    os.path.join(DATA_RAW_DIR, OPENMETEO_WEATHER_FILENAME), sep=";"
)
weather_data_df["timestamp"] = pd.to_datetime(
    weather_data_df["timestamp"], errors="raise", utc=True
)


power_data_df = pd.read_csv(os.path.join(DATA_RAW_DIR, DATA_RAW_FILENAME), sep=";")
power_data_df["timestamp"] = pd.to_datetime(
    power_data_df["timestamp"], errors="raise", utc=True
)


power_weather_data_df = merge_weather_with_power_data(power_data_df, weather_data_df)

power_weather_data_df.to_csv(
    os.path.join(DATA_RAW_DIR, POWER_OPENMETEO_WEATHER_FILENAME), sep=";", index=False
)
