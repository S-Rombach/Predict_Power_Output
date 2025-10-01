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
from src.config import DATA_ORIG_DIR, MODELS_DIR, WC_CODES_FILENAME
from src.transformation import (
    fetch_openmeteo_weather_data,
    prepare_aggregate_openmeteo_data,
)

st.set_page_config(page_title="Predict Power Output", layout="centered")

st.title("Predict Power Output")
earliest_date = date.today() - timedelta(days=365)
latest_date = date.today() + timedelta(days=10)
start_date = st.date_input(
    "Select start date",
    value=date.today(),
    format="YYYY-MM-DD",
    min_value=earliest_date,
    max_value=latest_date,
)
end_date = st.date_input(
    "Select end date",
    value=date.today() + timedelta(days=1),
    format="YYYY-MM-DD",
    min_value=start_date,
    max_value=latest_date,
)
start = st.button("Start")

installation_name = "elegant_eagle"

if start:
    lat = float(st.secrets[installation_name]["latitude"])
    lon = float(st.secrets[installation_name]["longitude"])

    model_name = "20250929162510_Ridge_r208799_predict_mult-weather,feateng-doytrig"
    filename = os.path.join(MODELS_DIR, model_name, model_name)

    with open(f"{filename}.model.pkl", "rb") as f:
        lreg = pickle.load(f)

    with open(f"{filename}.pipeline.pkl", "rb") as f:
        lpreprocessor = pickle.load(f)

    lp = Pipeline(steps=[("prep", lpreprocessor), ("reg", lreg)])

    wc_codes = pd.read_csv(
        os.path.join(DATA_ORIG_DIR, WC_CODES_FILENAME), sep=";", index_col="code_figure"
    ).to_dict()["code_name"]

    weather_data, meta_data = fetch_openmeteo_weather_data(
        lat, lon, start_date, end_date, wc_codes=wc_codes
    )
    weather_data = weather_data.reset_index(drop=True)
    weather_data = weather_data.rename(columns={"date": "timestamp"})
    weather_data["installation"] = installation_name

    df_doy = prepare_aggregate_openmeteo_data(
        weather_data,
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

    fig = px.line(
        data_frame=df_doy,
        x="date",
        y="predicted",
        title="Predicted Power Output",
        labels={"predicted": "peak power hours [W<sub>p</sub>h]", "date": "Date"},
    )

    fig.update_xaxes(
        tickvals=df_doy["date"].unique(),
        ticktext=[d.strftime("%Y-%m-%d") for d in df_doy["date"].unique()],
        tickangle=45,
    )

    fig.update_yaxes(range=[0, None])

    fig.add_annotation(
        x=1,
        y=0,
        xref="paper",
        yref="paper",
        xanchor="right",
        yanchor="bottom",
        text="Contains weather data by https://open-meteo.com/",
        showarrow=False,
        font=dict(size=8, color="gray"),
    )

    st.plotly_chart(fig)
