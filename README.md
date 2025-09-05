# Predict_Power_Output

## Usage

To gather all power data of all installations:

* in the `data\orig` directory:
  * for each installation
    * create a directory, the name of the directory must be a unique identifier. It will be used throughout the whole project.
    * copy its power data into it.
    * add an entry to the `installation_data.csv`, using the directories name as key in the column `installation`.
    * with `find_nearest_dwd_station.ipynb` you can find the mandatory `closest_weather_station_ids`. At least one is necessary.
* run `transform_data.py`, creating `data\raw\power_data.csv`
* run `add_weather_data_to_power_data.py`, creating `data\raw\power_weather_data.csv`

## Conventions

* All serialized timestamps are expressed in utc with timezone info. This is to ensure clarity about the time across multiple data inputs. For example, the DWD uses utc for all its reports, but installation timestamps are local times without timezone information.

## Installation metadata file

This file contains metadata about all solar stations covered by this project. To keep the actual station information private the metadata file is checked in another (private) repository and only referenced.

The file contains the following data:

| information                   | type           | semantic |
| ----------------------------- | -------------- | -------- |
| installation                  | string         | the name of the station |
| Wp                            | int            | the peak power output in W |
| closest_weather_station_names | string         | \| -separated, the names of the closest weather stations, to be human readable |
| closest_weather_station_ids   | string         | \| -separated, the ids of the closest weather stations |
| timezone                      | string rep. tz | The timezone of the timestamps of the installation are in. If the timestamps are not localized, this timezone is used, otherwise an error is raised. The timezone to localize naive timestamps. Can be a string like 'Europe/Berlin', 'UTC', 'UTC+1', etc., or a tzinfo object. If the Series is already timezone-aware, this is ignored. |
