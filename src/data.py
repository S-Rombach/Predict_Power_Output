import os
import pandas as pd
import datetime


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
            raise ValueError(
                "Series must be timezone-aware or target_timezone must be set."
                " Express the timestamps with a timezone or provide a timezone."
            )
        elif target_timezone.lower() == "utc":
            s = s.dt.tz_localize("UTC")
        elif target_timezone.lower().startswith(
            "utc+"
        ) or target_timezone.lower().startswith("utc-"):
            timeshift_hours = int(target_timezone[3:])
            fixed_tz = datetime.timezone(datetime.timedelta(hours=timeshift_hours))
            s = s.dt.tz_localize(fixed_tz)
        else:
            s = s.dt.tz_localize(
                target_timezone, ambiguous="infer", nonexistent="shift_forward"
            )
    return s.dt.tz_convert("UTC")


def gather_and_transform_data(
    installation_metadata: pd.DataFrame,
    orig_data_dir_name: str,
) -> pd.DataFrame:
    """
    Gather and transform raw installation CSV files into a single, normalized dataset and save it.

    This function walks through subdirectories of `orig_data_dir_name`.
    All per-file rows are concatenated into a single DataFrame and written (semicolon-separated)
    to `raw_data_pathfilename`. The raw data directory is created if it does not exist.

    Parameters
    ----
        orig_data_dir_name : str
            Directory containing one subdirectory per installation with raw CSVs.
        installation_metadata : pd.DataFrame
            DataFrame containing installation metadata indexed by "installation" and
            expected to contain at least the columns "Wp" (peak power) and "timezone".

    Returns
    -------
    pd.DataFrame
        The combined and transformed power data from all installations.
    """

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
    """Mapping of original column names to standardized column names."""

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

    all_power_data = pd.DataFrame(
        columns=column_translations.values(),
        dtype=str,
    )

    for dir in os.listdir(orig_data_dir_name):
        if not os.path.isdir(os.path.join(orig_data_dir_name, dir)):
            continue

        peak_power = float(installation_metadata.loc[dir, "Wp"])
        timezone = installation_metadata.loc[dir, "timezone"]

        print(f"Processing installation: {dir} with peak power {peak_power} W")

        installation_data = pd.DataFrame(
            columns=column_translations.values(),
            dtype=str,
        )

        for file in os.listdir(os.path.join(orig_data_dir_name, dir)):
            if not file.endswith(".csv"):
                continue

            # Construct the full path to the original data file
            orig_file_path = os.path.join(orig_data_dir_name, dir, file)

            # Read the original data file
            df = pd.read_csv(orig_file_path, sep=";", dtype=str)

            df["installation"] = dir
            df = df.rename(columns=column_translations)

            df["soc"] = df["soc"].apply(pd.to_numeric, errors="coerce").div(100)

            # normalize all power values as part of the peak power
            # provides comparability across installations
            # anonymize the installation owner further by not giving details about the installation
            this_power_columns = [col for col in power_columns if col in df.columns]

            df[this_power_columns] = (
                df[this_power_columns]
                .apply(pd.to_numeric, errors="coerce")
                .div(peak_power)
            )

            df["timestamp"] = ensure_utc_series(df["timestamp"], timezone)

            # Append the data to the installation_data DataFrame
            installation_data = pd.concat(
                [installation_data, df], ignore_index=True, axis=0
            )

        # Append the data to the all_data DataFrame
        all_power_data = pd.concat(
            [all_power_data, installation_data], ignore_index=True, axis=0
        )

    return all_power_data


def merge_weather_with_power_data(
    power_data_df: pd.DataFrame, weather_data_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge power and weather DataFrames by installation and timestamp.
    Parameters
    ----------
    power_data_df : pd.DataFrame
        Power measurements containing at least 'timestamp' and 'installation' columns.
    weather_data_df : pd.DataFrame
        Weather observations containing 'timestamp', 'installation' and 'sunshine_duration'.
    Returns
    -------
    pd.DataFrame
        The power_data_df enriched with matching weather columns. Weather is resampled
        to 15-minute intervals with forward-fill and sunshine_duration is scaled accordingly.
    Raises
    ------
    ValueError
        If an installation in power_data_df is missing in weather_data_df or if weather
        data does not cover the full time range of the power data for any installation.
    """
    for inst in power_data_df["installation"].unique():
        if inst not in weather_data_df["installation"].unique():
            raise ValueError(
                f"Installation '{inst}' not found in weather data."
                " Fetch weather data first."
                " Ensure fetched weather data covers the whole time range of the power data."
            )

        weather_data = weather_data_df[weather_data_df["installation"] == inst].copy()

        if weather_data is None or weather_data.empty:
            print(f"No weather data found for installation '{inst}'.")
            continue

        if (
            weather_data["timestamp"].min()
            > power_data_df[power_data_df["installation"] == inst]["timestamp"].min()
        ):
            raise ValueError(
                f"Weather data for installation '{inst}' does not cover the entire time range of the power data."
                f" Earliest weather data timestamp is {weather_data['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')},"
                f" but earliest power data timestamp is"
                f" {power_data_df[power_data_df['installation'] == inst]['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')}."
            )
        if (
            weather_data["timestamp"].max()
            < power_data_df[power_data_df["installation"] == inst]["timestamp"].max()
        ):
            raise ValueError(
                f"Weather data for installation '{inst}' does not cover the entire time range of the power data."
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

    return power_data_df
