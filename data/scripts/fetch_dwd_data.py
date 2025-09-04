#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fetches weather data from the DWD. Fills missing values and converts it into 15 minute intervals.
"""
# -- imports --------------------------------------------------------
import pandas as pd
from wetterdienst.provider.dwd.observation import DwdObservationRequest

import datetime as dt

import os
from src.config import (
    DATA_ORIG_DIR,
)

# -- init --------------------------------------------------------
installation_df = pd.read_csv(
    os.path.join(DATA_ORIG_DIR, "installation_data.csv"),
    sep=";",
    index_col="installation",
)

station_ids = str(
    installation_df.loc["elegant_eagle", "closest_weather_station_ids"]
).split("|")

start_date = dt.datetime(2020, 4, 1)
end_date = dt.datetime(2025, 9, 1)


params = [
    ("10_minutes", "solar", "radiation_global"),
    ("10_minutes", "solar", "sunshine_duration"),
]

all_period_data = {}

# -- load data --------------------------------------------------------
for period in ("historical", "recent"):
    req = DwdObservationRequest(
        parameters=params,
        start_date=start_date,
        end_date=end_date,
        periods=period,
    )

    stations = req.filter_by_station_id(station_id=station_ids)

    stations_df = stations.df.to_pandas()
    value_df = stations.values.all().df.to_pandas()

    all_period_data[period] = stations_df[
        ["resolution", "dataset", "station_id", "name"]
    ].merge(value_df, on=["station_id", "resolution", "dataset"], how="left")

historical_date_range = (
    all_period_data["historical"]
    .groupby(["resolution", "dataset", "station_id", "parameter"])
    .agg({"date": ["min", "max"]})
)

# -- merge time periods --------------------------------------------------------

# add all data from recent thats not in historical
# prefer historical, because it has higher quality assurance
key_cols = ["resolution", "dataset", "station_id", "parameter", "date"]
"""Columns to identify unique records."""

recent_data_not_in_hist = all_period_data["recent"][
    ~all_period_data["recent"]
    .set_index(key_cols)
    .index.isin(all_period_data["historical"].set_index(key_cols).index)
]

data_combined = pd.concat([all_period_data["historical"], recent_data_not_in_hist])
"""Historical data and recent data combined, to provide a comprehensive timeline for each station and parameter."""

# -- merge station data --------------------------------------------------------

relevant_data = data_combined[data_combined["station_id"] == station_ids[0]]
"""Data is reduced to data from the nearest station. Missing data is filled with data from other stations."""

key_cols_wo_station = ["resolution", "dataset", "parameter", "date"]
"""Columns to keep when merging data across stations."""

for stid in station_ids[1:]:
    rel_station_data = data_combined[data_combined["station_id"] == stid]

    station_data_not_in_rel = rel_station_data[
        ~rel_station_data.set_index(key_cols_wo_station).index.isin(
            relevant_data.set_index(key_cols_wo_station).index
        )
    ]

    relevant_data = pd.concat([relevant_data, station_data_not_in_rel])

# -- parameters to columns --------------------------------------------------------

# All information except date and values is removed.
rad_sun_data = relevant_data.drop(
    columns=["station_id", "name", "resolution", "dataset", "quality"]
)
"""A full timeline for each parameter across all stations."""

rad_sun_data = rad_sun_data.pivot(
    index="date", columns="parameter", values="value"
).reset_index()
rad_sun_data.columns.name = None

# -- create missing time steps and fill missing values --------------------------------------------------------
rad_sun_data = rad_sun_data.sort_values("date").set_index("date")

# create full 10 min index
full_index = pd.date_range(
    start=rad_sun_data.index.min(), end=rad_sun_data.index.max(), freq="10min"
)

# Reindex → missing data is NaN
rad_sun_data = rad_sun_data.reindex(full_index).reset_index()
rad_sun_data.rename(columns={"index": "date"}, inplace=True)

rad_sun_data["data_temp"] = rad_sun_data["date"].dt.strftime("%m-%d-%H-%M")

rad_sun_data["mean_rad"] = (
    rad_sun_data["radiation_global"]
    .groupby(rad_sun_data["data_temp"])
    .transform("mean")
)
rad_sun_data["mean_sun"] = (
    rad_sun_data["sunshine_duration"]
    .groupby(rad_sun_data["data_temp"])
    .transform("mean")
)

nans = rad_sun_data.isna().any(axis=1)
notnans = rad_sun_data.notna().any(axis=1)

# filling the nans with the mean over all years may be way off
# but for the sake of having a complete dataset, ...
rad_sun_data.fillna({"radiation_global": rad_sun_data["mean_rad"]}, inplace=True)
rad_sun_data.fillna({"sunshine_duration": rad_sun_data["mean_sun"]}, inplace=True)

rad_sun_data.drop(columns=["data_temp", "mean_rad", "mean_sun"], inplace=True)

# -- restructuring to 15 min steps --------------------------------------------------------

rad_sun_data = rad_sun_data.set_index("date")

# split the 10 min intervals into 5 min intervals
# ATTENTION: The 00:55 is missing
rs_5min = rad_sun_data.resample("5min").ffill() * (5 / 10)

step = pd.Timedelta("5min")

# create all 5 min intervals, including the missing 00:55
idx5 = pd.date_range(
    start=rad_sun_data.index.min(),
    end=rad_sun_data.index.max() + step,
    freq=step,
)

# fill the 00:55 interval with the value from 00:50
# wich was already halved, so both have the correct value
rs_5min = rs_5min.reindex(idx5).ffill()

# sum the 5 min intervals to 15 min intervals
rad_sun_data_15min = rs_5min.resample("15min").sum()
display(rad_sun_data_15min)
