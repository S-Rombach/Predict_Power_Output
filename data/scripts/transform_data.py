#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
import pandas as pd

from src.config import (
    DATA_RAW_DIR,
    DATA_ORIG_DIR,
    DATA_ORIG_FILENAME,
    DATA_RAW_FILENAME,
)

# Ensure the raw data directory exists
os.makedirs(DATA_RAW_DIR, exist_ok=True)

all_data = pd.DataFrame(
    columns=[
        "installation",
        "timestamp",
        "Ladezustand",
        "Batterie (Laden)",
        "Batterie (Entladen)",
        "Netzeinspeisung",
        "Netzbezug",
        "Solarproduktion Tracker 1",
        "Solarproduktion Tracker 2",
        "Solarproduktion",
        "Hausverbrauch",
        "ADDITIONAL Verbrauch",
        "ext. Verbrauch",
        "Σ Verbrauch",
        "Wallbox (ID 0) Gesamtladeleistung",
        "Wallbox (ID 0) Netzbezug",
        "Wallbox (ID 0) Solarladeleistung",
        "Wallbox Gesamtladeleistung",
    ],
    dtype=str,
)

for dir in os.listdir(DATA_ORIG_DIR):
    if not os.path.isdir(os.path.join(DATA_ORIG_DIR, dir)):
        continue

    lst = []
    for file in os.listdir(os.path.join(DATA_ORIG_DIR, dir)):
        if not file.endswith(".csv"):
            continue

        # Construct the full path to the original data file
        orig_file_path = os.path.join(DATA_ORIG_DIR, dir, file)

        # Read the original data file
        df = pd.read_csv(orig_file_path, sep=";", dtype=str)
        lst.extend(df.columns.tolist())

        df["installation"] = dir

        # Append the data to the all_data DataFrame
        all_data = pd.concat([all_data, df], ignore_index=True, axis=0)

column_translations = {
    "installation": "installation",
    "timestamp": "timestamp",
    "Ladezustand": "state of charge",
    "Batterie (Laden)": "battery charging",
    "Batterie (Entladen)": "battery discharging",
    "Netzeinspeisung": "grid feed-in",
    "Netzbezug": "grid consumption",
    "Solarproduktion Tracker 1": "solar production tracker 1",
    "Solarproduktion Tracker 2": "solar production tracker 2",
    "Solarproduktion": "solar production",
    "Hausverbrauch": "house consumption",
    "ADDITIONAL Verbrauch": "additional consumption",
    "ext. Verbrauch": "external consumption",
    "Σ Verbrauch": "total consumption",
    "Wallbox (ID 0) Gesamtladeleistung": "wallbox (id 0) total charging power",
    "Wallbox (ID 0) Netzbezug": "wallbox (id 0) grid consumption",
    "Wallbox (ID 0) Solarladeleistung": "wallbox (id 0) solar charging power",
    "Wallbox Gesamtladeleistung": "wallbox total charging power",
}

all_data = all_data.rename(columns=column_translations)
# After processing all files in the directory, save the combined data
if not all_data.empty:
    all_data.to_csv(os.path.join(DATA_RAW_DIR, DATA_RAW_FILENAME), index=False, sep=";")
