import os
import json
import pickle
import pandas as pd

from sklearn.pipeline import Pipeline
import streamlit as st

from src.config import (
    DATA_ORIG_DIR,
    SOLAR_PROD_DAILY_MODELS_DIR,
    SOLAR_PROD_HOURLY_MODELS_DIR,
    WC_CODES_FILENAME,
)
from src.data import gather_and_transform_data, merge_weather_with_power_data
from src.transformation import fetch_openmeteo_weather_data

from datetime import date, timedelta

TIME_RESOLUTION_DAILY = "Daily"
TIME_RESOLUTION_HOURLY = "Hourly"
time_resolution_options = {
    TIME_RESOLUTION_HOURLY: SOLAR_PROD_HOURLY_MODELS_DIR,
    TIME_RESOLUTION_DAILY: SOLAR_PROD_DAILY_MODELS_DIR,
}


def retrain_model(data_path: str, selected_models_dir: str, model_name: str) -> None:

    installation_name = "Unknown"
    meta_data_dict = {
        "timezone": st.secrets["timezone"],
        "Wp": st.secrets["Wp"],
        "installation": installation_name,
        "latitude": st.secrets["latitude"],
        "longitude": st.secrets["longitude"],
    }
    installation_metadata = (
        pd.Series(
            meta_data_dict,
            name=meta_data_dict["installation"],
        )
        .to_frame()
        .T
    )

    # transform solar data
    all_power_data = gather_and_transform_data(
        installation_metadata=installation_metadata, orig_data_dir_name=data_path
    )

    # fetch weather data
    latitude = float(meta_data_dict["latitude"])
    longitude = float(meta_data_dict["longitude"])
    start_date = date(2012, 1, 1)
    end_date = date.today() + timedelta(days=-1)

    wc_codes = pd.read_csv(
        os.path.join(DATA_ORIG_DIR, WC_CODES_FILENAME), sep=";", index_col="code_figure"
    ).to_dict()["code_name"]

    weather_data, meta_data = fetch_openmeteo_weather_data(
        latitude, longitude, start_date, end_date, wc_codes=wc_codes
    )
    weather_data = weather_data.reset_index(drop=True)
    weather_data = weather_data.rename(columns={"date": "timestamp"})
    weather_data["installation"] = installation_name
    weather_data = weather_data[
        ["installation"]
        + [col for col in weather_data.columns if col != "installation"]
    ]

    # merge weather data with power data
    merged_data = merge_weather_with_power_data(
        power_data_df=all_power_data, weather_data_df=weather_data
    )

    from sklearn.model_selection import train_test_split

    Xy_train, Xy_test = train_test_split(
        merged_data, test_size=0.2, random_state=387
    )
    X_train = Xy_train.drop(columns=["power_output"])
    y_train = Xy_train["power_output"]
    X_test = Xy_test.drop(columns=["power_output"])
    y_test = Xy_test["power_output"]

    # load model
    model_name = os.path.basename(os.path.normpath(selected_models_dir))
    path_filename_prefix = os.path.join(selected_models_dir, model_name)

    with open(f"{path_filename_prefix}.model.pkl", "rb") as f:
        estimator = pickle.load(f)

    with open(f"{path_filename_prefix}.pipeline.pkl", "rb") as f:
        preprocessor = pickle.load(f)


    
    # retrain model
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])
    pipe.fit(X_train, y_train)

    # save updated model


def render():
    st.set_page_config(page_title="Model Retraining", layout="centered")
    st.title("Model Retraining")
    st.write(
        "These are the models in the `models/` directory."
        " Below you can select a model to retrain it."
    )

    # define which metrics to show ###############################################
    error_metrics = ["MAE", "MSE", "RMSE", "MedAE"]
    high_exp_error_metrics = ["MAPE"]
    score_metrics = ["R2", "ExplainedVar"]

    target_keys = (
        [
            "timestamp",
            "model_purpose",
            "special_features",
            "model_class",
        ]
        + error_metrics
        + high_exp_error_metrics
        + score_metrics
    )

    selected_time_resolution = st.selectbox(
        "Select time resolution",
        list(time_resolution_options.keys()),
        key="tabretrain_eval_time_res",
    )
    selected_models_dir = time_resolution_options[selected_time_resolution]

    # find model subdirs with results ###########################################
    subdirs = [
        d
        for d in os.listdir(selected_models_dir)
        if os.path.isdir(os.path.join(selected_models_dir, d))
        and not d.startswith(".")
        and os.path.exists(os.path.join(selected_models_dir, d, f"{d}.results.json"))
    ]

    rows = []

    for d in subdirs:
        result_path = os.path.join(selected_models_dir, d, f"{d}.results.json")
        row = {}
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                data = json.load(f)

            for key in target_keys:
                val = data
                for part in key.split("__"):
                    if isinstance(val, dict) and part in val:
                        val = val[part]
                    else:
                        val = None
                        break
                row[key] = val
        else:
            continue
        row["name"] = d
        rows.append(row)

    df = pd.DataFrame(rows)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    df = df.sort_values(by="R2", ascending=False)

    # show list of models with metrics ###########################################
    st.dataframe(df.set_index("name"))

    # prepare comparison of models ###############################################
    df_show = df.copy()
    df_baseline = df_show[df_show["model_purpose"].str.startswith("baseline")]

    # aggregate metrics for min/max band and median line
    df_agg = df[error_metrics + high_exp_error_metrics + score_metrics].agg(
        ["min", "max", "median"]
    )

    df_show[error_metrics] = df_show[error_metrics]
    # df_show = df_show[~df_show["model_purpose"].str.startswith("baseline")]

    # Example selectors
    target_model_timestamp = st.selectbox(
        "Select model to retrain",
        options=list(map(str, pd.Index(df_show.index).unique())),
        format_func=lambda x: df_show.loc[df_show.index == x, "name"].values[0],
        key="tabretrain_eval_model_timestamp",
    )

    uploaded_files = st.file_uploader(
        "Upload new training data (CSV files)",
        accept_multiple_files=True,
        type=["csv"],
        key="tabretrain_upload_new_training_data",
    )

    if len(uploaded_files) == 0:
        st.info("Please upload at least one CSV file with new training data.")
        return

    # make the retrain data directory session specific, so multiple users don't conflict
    # this is obviously not suitable for production use
    retrain_data_path = os.path.join(
        DATA_ORIG_DIR, f"retrain_data_{date.today().isoformat()}"
    )

    os.makedirs(retrain_data_path, exist_ok=True)

    for uploaded_file in uploaded_files:
        with open(os.path.join(retrain_data_path, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

    if st.button("Retrain Model"):
        retrain_model(retrain_data_path, selected_models_dir, target_model_timestamp)
