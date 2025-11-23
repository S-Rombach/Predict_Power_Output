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
from src.transformation import fetch_weather_data_for_installations


installation_metadata = pd.read_csv(
    os.path.join(DATA_ORIG_DIR, INSTALLATION_DATA_FILENAME),
    sep=";",
    index_col="installation",
).to_dict(orient="index")

wc_codes = pd.read_csv(
    os.path.join(DATA_ORIG_DIR, WC_CODES_FILENAME), sep=";", index_col="code_figure"
).to_dict()["code_name"]

start_date = date(2012, 1, 1)
end_date = date.today() + timedelta(days=-1)


all_weather_data = fetch_weather_data_for_installations(
    installation_metadata, wc_codes, start_date, end_date
)

all_weather_data.to_csv(
    os.path.join(DATA_RAW_DIR, OPENMETEO_WEATHER_FILENAME), index=False, sep=";"
)
