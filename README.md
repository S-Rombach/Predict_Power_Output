# Predict Power Output

## Goal

The goal of this project is to predict the power output of solar panels of specific installations.

The benefits over existing solutions are:

* _accessible_ predictions: The manufacturer of the installation does provide predictions, but does not allow to export them. This project provides data ready to feed in other applications.
* _customized_ per installation: The manufacturer provides predictions based on a model trained on data of multiple sites. This one is tailored to a specific location.

## Setup

To gather all power data of all installations:

* in the [data/orig](data/orig/) directory:  
  for each installation
  * create a directory, the name of the directory must be a unique identifier. It will be used throughout the whole project.
  * copy its power data into it.
  * add an entry to the [installation_data.csv](data/orig/installation_data.csv),
    see [Installation metadata file](#installation-metadata-file) for details
    * use the directories name as key in the column `installation`.
    * peak power production in W.
    * at least one of the following
      * if you want to use _DWD_ reports: Use [find_nearest_dwd_station.ipynb](data/scripts/find_nearest_dwd_station.ipynb) to find the  `closest_weather_station_ids`. At least one is necessary. `closest_weather_station_names` is optional.
      * if you want to use _OpenMeteo_ reforecasts: Enter longitude and latitude of the installation.
    * The timezone of the installation (e.g. utc+1).
* run [data/scripts/transform_data.py](data/scripts/transform_data.py), creating [data/raw/power_data.csv](data/raw/power_data.csv)
* run [add_[WDP]_weather_data_to_power_data.py](data/scripts/add_openmeteo_weather_data_to_power_data.py), creating [data\raw\power_[WDP]_weather_data.csv](data/raw/power_openmeteo_weather_data.csv). `[WDP]` is a weather data provider (either `dwd` or `openmeteo`).

## Usage

Notebooks to train models are located in [model_training](model_training). Run all cells of the notebook to train a model. The configuration and results will be saved in a directory in [models](models). The binaries are not saved, because the source code of custom models is not packaged.

## Conventions

* All serialized timestamps are expressed in utc with timezone info. This is to ensure clarity about the time across multiple data inputs. For example, the DWD uses utc for all its reports, but installation timestamps are local times without timezone information.
* According to the [dwd documentation](https://wetterdienst.readthedocs.io/en/latest/data/parameters.html#list-of-parameters) the unit of sunshine duration in the dwd dataset `sd_10` (data point every 10 minutes) is hours, yet the maximum value is 601.2. The value is considered to be in seconds.
* Not all custom models are pickled, because the source code is not packaged. Therefore, in some cases, only the config and results are serialized.

## Installation metadata file

This file contains metadata about all solar stations covered by this project. To keep the actual station information private the metadata file is checked in another (private) repository and only referenced.

The file contains the following data:

| information                   | type           | semantic |
| ----------------------------- | -------------- | -------- |
| installation                  | string         | the name of the station |
| Wp                            | int            | the peak power output in W |
| closest_weather_station_names | string         | \|-separated, the names of the closest weather stations, to be human readable |
| closest_weather_station_ids   | string         | \|-separated, the ids of the closest weather stations |
| timezone                      | string rep. tz | The timezone of the timestamps of the installation are in. If the timestamps are not localized, this timezone is used, otherwise an error is raised. The timezone to localize naive timestamps. Can be a string like 'Europe/Berlin', 'UTC', 'UTC+1', etc., or a tzinfo object. If the Series is already timezone-aware, this is ignored. |
| latitude                      | float          | The latitude of the installation. |
| longitude                     | float          | The longitude of the installation. |

## Data

### Data Dictionary

The data dictionary is located in [data](data/data_dictionary.ipynb).

### Source DWD

Weather data by [Dwd](https://www.dwd.de) using the package [wetterdienst](https://github.com/earthobservations/wetterdienst/tree/v0.112.0) [https://doi.org/10.5281/zenodo.3960624](https://doi.org/10.5281/zenodo.3960624)

### Source OpenMeteo

Weather data by [Open-Meteo.com](https://open-meteo.com/).

Zippenfenig, P. (2023). Open-Meteo.com Weather API [Computer software]. Zenodo. [https://doi.org/10.5281/ZENODO.7970649](https://doi.org/10.5281/ZENODO.7970649)

### Weather codes

The file [wc_4677_codes_simplified.csv](data/orig/wc_4677_codes_simplified.csv) is a shrunken version of the list of all [weather codes](https://www.nodc.noaa.gov/archive/arc0021/0002199/1.1/data/0-data/HTML/WMO-CODE/WMO4677.HTM) to reduce complexity. The shrinking was done with common sense, but without particular meteorological domain knowledge. If in doubt, use the weather codes instead.
