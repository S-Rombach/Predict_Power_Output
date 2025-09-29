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

    weather_data, meta_data = fetch_openmeteo_weather_data(
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
