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
    "Ladezustand": "soc",
    "Batterie (Laden)": "bat_charge",
    "Batterie (Entladen)": "bat_discharge",
    "Netzeinspeisung": "grid_feed_in",
    "Netzbezug": "grid_cons",
    "Solarproduktion Tracker 1": "sol_prod_1",
    "Solarproduktion Tracker 2": "sol_prod_2",
    "Solarproduktion": "sol_prod",
    "Hausverbrauch": "house_cons",
    "ADDITIONAL Verbrauch": "add_cons",
    "ext. Verbrauch": "ext_cons",
    "Σ Verbrauch": "tot_cons",
    "Wallbox (ID 0) Gesamtladeleistung": "wb_0_tot_charge",
    "Wallbox (ID 0) Netzbezug": "wb_0_grid_cons",
    "Wallbox (ID 0) Solarladeleistung": "wb_0_sol_charge",
    "Wallbox Gesamtladeleistung": "wb_tot_charge"
}


all_data = all_data.rename(columns=column_translations)
# After processing all files in the directory, save the combined data
if not all_data.empty:
    all_data.to_csv(os.path.join(DATA_RAW_DIR, DATA_RAW_FILENAME), index=False, sep=";")
