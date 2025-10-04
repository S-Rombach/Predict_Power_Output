#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fetches past weather data from Open-Meteo for stations in 'INSTALLATION_DATA_FILENAME', saves it to 'OPENMETEO_WEATHER_FILENAME'.
"""

import os
import pandas as pd

from datetime import date, timedelta

from src.config import (
    DATA_ORIG_DIR,
    DATA_RAW_DIR,
    WC_CODES_FILENAME,
    INSTALLATION_DATA_FILENAME,
    OPENMETEO_WEATHER_FILENAME,
)
from src.transformation import fetch_openmeteo_weather_data


installation_metadata = pd.read_csv(
    os.path.join(DATA_ORIG_DIR, INSTALLATION_DATA_FILENAME),
    sep=";",
    index_col="installation",
)
all_weather_data = pd.DataFrame()

for installation_name in installation_metadata.index:
    print(f"Processing installation: {installation_name}")

    latitude = float(installation_metadata.loc[installation_name, "latitude"])
    longitude = float(installation_metadata.loc[installation_name, "longitude"])
    start_date = date(2012, 1, 1)
    end_date = date.today() + timedelta(days=-1)

    wc_codes = pd.read_csv(
        os.path.join(DATA_ORIG_DIR, WC_CODES_FILENAME), sep=";", index_col="code_figure"
    ).to_dict()["code_name"]

    weather_data, meta_data = fetch_openmeteo_weather_data(
        latitude, longitude, start_date, end_date, wc_codes=wc_codes
    )
    weather_data = weather_data.reset_index(drop=True)
    weather_data = weather_data.rename(columns={"date": "timestamp"})
    weather_data["installation"] = installation_name
    weather_data = weather_data[
        ["installation"]
        + [col for col in weather_data.columns if col != "installation"]
    ]

    all_weather_data = (
        pd.concat([all_weather_data, weather_data], ignore_index=True)
        if not all_weather_data.empty
        else weather_data
    )

all_weather_data.to_csv(
    os.path.join(DATA_RAW_DIR, OPENMETEO_WEATHER_FILENAME), index=False, sep=";"
)
