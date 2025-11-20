import os
import json
import pandas as pd

import streamlit as st

from src.config import SOLAR_PROD_DAILY_MODELS_DIR, SOLAR_PROD_HOURLY_MODELS_DIR

TIME_RESOLUTION_DAILY = "Daily"
TIME_RESOLUTION_HOURLY = "Hourly"
time_resolution_options = {
    TIME_RESOLUTION_HOURLY: SOLAR_PROD_HOURLY_MODELS_DIR,
    TIME_RESOLUTION_DAILY: SOLAR_PROD_DAILY_MODELS_DIR,
}


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
    
    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            None
