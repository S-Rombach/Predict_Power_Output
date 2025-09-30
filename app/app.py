import os
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta


from sklearn.pipeline import Pipeline
from src.config import DATA_ORIG_DIR, MODELS_DIR, WC_CODES_FILENAME
from src.transformation import (
    fetch_openmeteo_weather_data,
    prepare_aggregate_openmeteo_data,
)

st.set_page_config(page_title="Date Plot Demo", layout="centered")

st.title("Date Plot Demo")
start_date = st.date_input("Select start date", value=date.today(), format="YYYY-MM-DD")
end_date = st.date_input(
    "Select end date", value=date.today() + timedelta(days=1), format="YYYY-MM-DD"
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

    # df_doy["date"] = pd.to_datetime(df_doy["date"])
    df_doy["predicted"] = lp.predict(df_doy)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=df_doy, x="date", y="predicted", label="Predicted", ax=ax)
    ax.set_xticks(df_doy["date"].unique())
    ax.set_xticklabels(
        [d.strftime("%Y-%m-%d") for d in df_doy["date"].unique()],
        rotation=45,
        ha="right",
    )
    ax.set_ylim(0, None)
    ax.text(
        0.98,
        0.02,
        "Contains weather data by https://open-meteo.com/",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        horizontalalignment="right",
        color="gray",
    )

    st.pyplot(fig)
