import os
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import seaborn as sns
import plotly.express as px
from datetime import date, timedelta


from sklearn.pipeline import Pipeline
from src.config import (
    DATA_ORIG_DIR,
    SOLAR_PROD_DAILY_MODELS_DIR,
    SOLAR_PROD_HOURLY_MODELS_DIR,
    WC_CODES_FILENAME,
)
from src.transformation import (
    fetch_openmeteo_weather_data,
    prepare_aggregate_openmeteo_data,
)

TIME_RESOLUTION_DAILY = "Daily"
TIME_RESOLUTION_HOURLY = "Hourly"


def render():
    st.set_page_config(page_title="Predict Power Output", layout="centered")

    st.title("Predict Power Output")

    time_resolution_options = {
        TIME_RESOLUTION_HOURLY: SOLAR_PROD_HOURLY_MODELS_DIR,
        TIME_RESOLUTION_DAILY: SOLAR_PROD_DAILY_MODELS_DIR,
    }

    selected_time_resolution = st.selectbox(
        "Select time resolution", list(time_resolution_options.keys())
    )

    earliest_date = date.today() - timedelta(
        days=(365 if selected_time_resolution == TIME_RESOLUTION_DAILY else 3)
    )
    latest_date = date.today() + timedelta(
        days=(10 if selected_time_resolution == TIME_RESOLUTION_DAILY else 3)
    )

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Select start date",
            value=date.today(),
            format="YYYY-MM-DD",
            min_value=earliest_date,
            max_value=latest_date,
        )

    with col2:
        if "end_date" not in st.session_state:
            st.session_state["end_date"] = date.today() + timedelta(days=7)

        st.session_state["end_date"] = max(
            min(st.session_state["end_date"], latest_date), start_date
        )

        end_date = st.date_input(
            "Select end date",
            format="YYYY-MM-DD",
            min_value=start_date,
            max_value=latest_date,
            key="end_date",
        )

    selected_models_dir = time_resolution_options[selected_time_resolution]

    subdirs = [
        d
        for d in os.listdir(selected_models_dir)
        if os.path.isdir(os.path.join(selected_models_dir, d))
        and not d.startswith(".")
        and os.path.exists(os.path.join(selected_models_dir, d, f"{d}.model.pkl"))
        and os.path.exists(os.path.join(selected_models_dir, d, f"{d}.pipeline.pkl"))
    ]

    selected_model = st.selectbox("Select model", subdirs)

    start = st.button("Predict Power Output")

    installation_name = "elegant_eagle"

    if start:
        lat = float(st.secrets[installation_name]["latitude"])
        lon = float(st.secrets[installation_name]["longitude"])

        model_name = selected_model
        filename = os.path.join(selected_models_dir, model_name, model_name)

        with open(f"{filename}.model.pkl", "rb") as f:
            lreg = pickle.load(f)

        with open(f"{filename}.pipeline.pkl", "rb") as f:
            lpreprocessor = pickle.load(f)

        lp = Pipeline(steps=[("prep", lpreprocessor), ("reg", lreg)])

        wc_codes = pd.read_csv(
            os.path.join(DATA_ORIG_DIR, WC_CODES_FILENAME),
            sep=";",
            index_col="code_figure",
        ).to_dict()["code_name"]

        weather_data, meta_data = fetch_openmeteo_weather_data(
            lat, lon, start_date, end_date, wc_codes=wc_codes
        )
        weather_data = weather_data.reset_index(drop=True)
        weather_data = weather_data.rename(columns={"date": "timestamp"})
        weather_data["installation"] = installation_name

        df_doy = prepare_aggregate_openmeteo_data(
            weather_data,
            time_horizon={
                TIME_RESOLUTION_HOURLY: "hourly",
                TIME_RESOLUTION_DAILY: "daily",
            }[selected_time_resolution],
            weather_column="weather_description",
            mandatory_weather_columns=[
                "clear_sky",
                "cloudy",
                "drizzle",
                "rain",
                "solid_precipitation",
            ],
        )

        df_doy["predicted"] = lp.predict(df_doy)
        if selected_time_resolution == TIME_RESOLUTION_DAILY:
            df_doy["time"] = pd.to_datetime(df_doy["date"])
        else:
            df_doy["time"] = pd.to_datetime(df_doy["date"]) + pd.to_timedelta(
                df_doy["hour"], unit="h"
            )

        fig = px.line(
            data_frame=df_doy,
            x="time",
            y="predicted",
            title="Predicted Power Output",
            labels={
                "predicted": "peak power hours [W<sub>p</sub>h]",
                "time": (
                    "Date"
                    if selected_time_resolution == TIME_RESOLUTION_DAILY
                    else "Hour"
                ),
            },
        )

        fig.update_xaxes(
            tickvals=df_doy["time"].unique(),
            ticktext=[
                (
                    d.strftime("%Y-%m-%d")
                    if selected_time_resolution == TIME_RESOLUTION_DAILY
                    else (
                        d.strftime("%H" + "<br>" + "%Y-%m-%d")
                        if d.hour == 0
                        else (d.strftime("%H") if d.hour % 3 == 0 else "")
                    )
                )
                for d in df_doy["time"]
            ],
            tickangle=45 if selected_time_resolution == TIME_RESOLUTION_DAILY else 0,
        )

        fig.update_yaxes(range=[0, None], rangemode="tozero")

        fig.update_layout(margin={"l": 40, "r": 60, "t": 40, "b": 40})

        # fig.add_annotation(
        #     x=1,
        #     y=0,
        #     xref="paper",
        #     yref="paper",
        #     xanchor="right",
        #     yanchor="bottom",
        #     text="Contains weather data by https://open-meteo.com/",
        #     showarrow=False,
        #     font=dict(size=8, color="gray"),
        # )

        st.plotly_chart(fig)
        if selected_time_resolution == TIME_RESOLUTION_DAILY:
            col1, col2 = st.columns(2)

            with col1:
                df_doy["datef"] = pd.to_datetime(df_doy["date"]).dt.date
                st.dataframe(
                    df_doy[["datef", "predicted"]]
                    .rename(
                        columns={
                            "datef": "Date",
                            "predicted": "Predicted Power Output [Wph]",
                        }
                    )
                    .set_index("Date")
                    .style.format("{:.3f}")
                )
