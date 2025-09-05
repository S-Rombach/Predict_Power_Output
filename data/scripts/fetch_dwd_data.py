#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fetches weather data from the DWD. Fills missing values and converts it into 15 minute intervals.
"""
# -- imports --------------------------------------------------------
import pandas as pd
from wetterdienst.provider.dwd.observation import DwdObservationRequest

import os
from src.config import (
    DATA_ORIG_DIR,
    DATA_RAW_DIR,
    DATA_RAW_FILENAME,
    INSTALLATION_DATA_FILENAME,
    POWER_WEATHER_FILENAME
)


def fetch_sunshine_radiation_data(station_ids, start_date, end_date):
    """
    Fetches and processes weather data from the German Weather Service (DWD) for specified stations and date range.
    This function retrieves 10-minute interval solar radiation and sunshine duration data for the given station IDs
    and time period, merges historical and recent datasets (preferring historical data for quality), fills missing
    values using data from additional stations and mean values over all years, and resamples the data into 15-minute
    intervals.
    Parameters
    ----------
    station_ids : list of int or str
        List of DWD station IDs, ordered by proximity (first is nearest).
    start_date : str or pandas.Timestamp
        Start date for data retrieval (inclusive).
    end_date : str or pandas.Timestamp
        End date for data retrieval (inclusive).
    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by 15-minute intervals with columns:
        - 'radiation_global': Global solar radiation (filled and resampled).
        - 'sunshine_duration': Sunshine duration (filled and resampled).
    Notes
    -----
    - Missing values are filled using data from additional stations and, if still missing, by the mean value for the
      corresponding time across all years.
    - The function combines historical and recent DWD datasets, preferring historical data where available.
    - The returned DataFrame covers the full date range at 15-minute intervals.
    """
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
    rad_sun_data = rad_sun_data.rename(columns={"index": "date"})

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

    # filling the nans with the mean over all years may be way off
    # but for the sake of having a complete dataset, ...
    rad_sun_data = rad_sun_data.fillna({"radiation_global": rad_sun_data["mean_rad"]})
    rad_sun_data = rad_sun_data.fillna({"sunshine_duration": rad_sun_data["mean_sun"]})

    rad_sun_data = rad_sun_data.drop(columns=["data_temp", "mean_rad", "mean_sun"])

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
    rad_sun_data_15min.index.name = "timestamp"

    return rad_sun_data_15min


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

for inst in power_data_df["installation"].unique():
    if inst not in installation_df.index:
        raise ValueError(
            f"Installation '{inst}' not found in '{INSTALLATION_DATA_FILENAME}'. Weather data cannot be fetched."
            " Add the nearest weather station IDs for the installation to the file."
        )

    # Get the weather station IDs for the current installation
    station_ids = str(installation_df.loc[inst, "closest_weather_station_ids"]).split(
        "|"
    )

    if not station_ids or station_ids == [""]:
        raise ValueError(
            f"No weather station IDs found for installation '{inst}' in '{INSTALLATION_DATA_FILENAME}'."
        )

    power_data_inst_df = power_data_df[power_data_df["installation"] == inst]

    start_date = power_data_inst_df["timestamp"].min()
    end_date = power_data_inst_df["timestamp"].max()

    weather_data = fetch_sunshine_radiation_data(station_ids, start_date, end_date)
    weather_data = weather_data.reset_index()

    if weather_data is None or weather_data.empty:
        print(
            f"No weather data found for installation '{inst}' with stations {station_ids}."
        )
        continue

    # -- Merge the weather data with the power data --------------------------------------------------------

    weather_data["installation"] = inst

    power_data_df = pd.merge(
        power_data_df, weather_data, how="left", on=["timestamp", "installation"]
    )

power_data_df.to_csv(
    os.path.join(DATA_RAW_DIR, POWER_WEATHER_FILENAME), sep=";", index=False
)
