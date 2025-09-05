#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import pandas as pd
import datetime

from src.config import (
    DATA_RAW_DIR,
    DATA_ORIG_DIR,
    DATA_RAW_FILENAME,
    INSTALLATION_DATA_FILENAME,
)

# Ensure the raw data directory exists
os.makedirs(DATA_RAW_DIR, exist_ok=True)

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
    "Solarproduktion Tracker 3": "sol_prod_3",
    "Solarproduktion": "sol_prod",
    "Hausverbrauch": "house_cons",
    "ADDITIONAL Verbrauch": "add_cons",
    "ext. Verbrauch": "ext_cons",
    "Σ Verbrauch": "tot_cons",
    "Wallbox (ID 0) Gesamtladeleistung": "wb_0_tot_charge",
    "Wallbox (ID 0) Netzbezug": "wb_0_grid_cons",
    "Wallbox (ID 0) Solarladeleistung": "wb_0_sol_charge",
    "Wallbox Gesamtladeleistung": "wb_tot_charge",
}

power_columns = [
    "bat_charge",
    "bat_discharge",
    "grid_feed_in",
    "grid_cons",
    "sol_prod_1",
    "sol_prod_2",
    "sol_prod_3",
    "sol_prod",
    "house_cons",
    "add_cons",
    "ext_cons",
    "tot_cons",
    "wb_0_tot_charge",
    "wb_0_grid_cons",
    "wb_0_sol_charge",
    "wb_tot_charge",
]
"""The columns that are used for power measurements."""

def ensure_utc_series(s: pd.Series, target_timezone) -> pd.Series:
    """
    Convert a pandas Series of timestamps to UTC timezone-aware datetimes.

    Parameters
    ----------
    s : pd.Series
        Series of datetime strings or naive/aware Timestamps.
    target_timezone : str or tzinfo
        The timezone to localize naive timestamps. Can be a string like 'Europe/Berlin', 'UTC', 'UTC+1', etc.,
        or a tzinfo object. If the Series is already timezone-aware, this is ignored.

    Returns
    -------
    pd.Series
        Series of UTC timezone-aware Timestamps.

    Raises
    ------
    ValueError
        If the Series is naive and no target_timezone is provided.
    """
    s = pd.to_datetime(s, utc=False, errors="raise")
    if s.dt.tz is None:
        if target_timezone is None or target_timezone == "":
            raise ValueError("Series must be timezone-aware or target_timezone must be set."
                             f" Express the timestamps with a timezone or provide a timezone in '{INSTALLATION_DATA_FILENAME}'.")
        elif target_timezone.lower() == "utc":
            s = s.dt.tz_localize("UTC")
        elif target_timezone.lower().startswith("utc+") or target_timezone.lower().startswith("utc-"):
            timeshift_hours = int(target_timezone[3:])
            fixed_tz = datetime.timezone(datetime.timedelta(hours=timeshift_hours))
            s = s.dt.tz_localize(fixed_tz)
        else:
            s = s.dt.tz_localize(target_timezone, ambiguous="infer", nonexistent="shift_forward")
    return s.dt.tz_convert("UTC")


all_power_data = pd.DataFrame(
    columns=column_translations.values(),
    dtype=str,
)

installation_metadata = pd.read_csv(
    os.path.join(DATA_ORIG_DIR, INSTALLATION_DATA_FILENAME),
    sep=";",
    index_col="installation",
)

for dir in os.listdir(DATA_ORIG_DIR):
    if not os.path.isdir(os.path.join(DATA_ORIG_DIR, dir)):
        continue

    peak_power = installation_metadata.loc[dir, "Wp"]
    timezone = installation_metadata.loc[dir, "timezone"]

    print(f"Processing installation: {dir} with peak power {peak_power} W")

    installation_data = pd.DataFrame(
        columns=column_translations.values(),
        dtype=str,
    )

    for file in os.listdir(os.path.join(DATA_ORIG_DIR, dir)):
        if not file.endswith(".csv"):
            continue

        # Construct the full path to the original data file
        orig_file_path = os.path.join(DATA_ORIG_DIR, dir, file)

        # Read the original data file
        df = pd.read_csv(orig_file_path, sep=";", dtype=str)

        df["installation"] = dir
        df = df.rename(columns=column_translations)

        df["soc"] = df["soc"].apply(pd.to_numeric, errors="coerce").div(100)

        this_power_columns = [col for col in power_columns if col in df.columns]

        # normalize all power values as part of the peak power
        # provides comparability across installations
        # anonymize the installation owner further by not giving details about the installation
        df[this_power_columns] = (
            df[this_power_columns].apply(pd.to_numeric, errors="coerce").div(peak_power)
        )

        df["timestamp"] = ensure_utc_series(df["timestamp"], timezone)

        # Append the data to the installation_data DataFrame
        installation_data = pd.concat(
            [installation_data, df], ignore_index=True, axis=0
        )

    # Append the data to the all_data DataFrame
    all_power_data = pd.concat([all_power_data, installation_data], ignore_index=True, axis=0)

# After processing all files in the directory, save the combined data
if not all_power_data.empty:
    all_power_data.to_csv(os.path.join(DATA_RAW_DIR, DATA_RAW_FILENAME), index=False, sep=";")
