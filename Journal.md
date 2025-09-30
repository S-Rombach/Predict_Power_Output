# Journal

## Compromises

* While fetching DWD data,
  * data completeness is prioritized over data accuracy
    * Data from different stations is mixed.
    * Missing values are filled with means from the last years, even when
      * there is not much data to build the mean and
      * the mean may differ much from the true value.

## Doings  

### 29.09.25

* abstract some code to make data preprocessing available everywhere
* read docs and create basic streamlit app  

### 26.09.25

* abstract some code
  * to make loading of serialized models possible
  * to make data preprocessing available everywhere
* add link required by license
* extend readme

### 22.09.25

* create open meteo model

### 19.09.25

* eda of open meteo data
* readme clean up
* data dictionary clean up

### 18.09.25

* use open meteo to add weather data to power data. This data is reforecast data and not measurements, so the inherent imprecision has to be learned by the model. This should help to prevent overfitting.

### 17.09.25

* Explore open meteo documentation

### 13.09.25

* add nb to compare all trained models

### 12.09.25

* save regression models to be able to compare metrics
* convention about serializing models

### 10.09.25

* fixed the training of the baseline model

### 09.09.25

* Created model to predict power production from sunshine duration.
* Autocorrelation of sunshine duration.

### 08.09.25

* Added convention to Readme.
* Extended data dict.

### 05.09.25

* Adding timezone convention
* Implementing timezone convention
* Learned about pd.tz_localize, pd.tz_convert, pd.melt
* Eda of weather data
  * The high correlation of sunshine duration, global radiation and power production suggests using a formula instead of a ml-model.

### 04.09.25

* Converting the `fetch_dwd_data`-skript into a transforming skript.
