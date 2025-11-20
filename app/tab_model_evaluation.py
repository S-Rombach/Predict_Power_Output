import os
import json
import pandas as pd
import numpy as np

import streamlit as st

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from src.config import SOLAR_PROD_DAILY_MODELS_DIR, SOLAR_PROD_HOURLY_MODELS_DIR

TIME_RESOLUTION_DAILY = "Daily"
TIME_RESOLUTION_HOURLY = "Hourly"
time_resolution_options = {
    TIME_RESOLUTION_HOURLY: SOLAR_PROD_HOURLY_MODELS_DIR,
    TIME_RESOLUTION_DAILY: SOLAR_PROD_DAILY_MODELS_DIR,
}


def render():
    st.set_page_config(page_title="Model Evaluation", layout="centered")
    st.title("Model Evaluation")
    st.write(
        "These are the models in the `models/` directory and their evaluation metrics."
        " Below you can select a model to highlight it in the comparison plot."
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
        key="eval_time_res",
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
        "Select model to show",
        options=list(map(str, pd.Index(df_show.index).unique())),
        format_func=lambda x: df_show.loc[df_show.index == x, "name"].values[0],
    )

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Error Metrics", "Error Metrics", "Score Metrics"),
        column_widths=[
            len(error_metrics),
            len(high_exp_error_metrics),
            len(score_metrics),
        ],
        horizontal_spacing=0.11,
    )

    # aggregate metrics: min/max band and median line ##############################
    for col, metrics in (
        (1, error_metrics),
        (2, high_exp_error_metrics),
        (3, score_metrics),
    ):
        # min/max band
        mn = df_agg[metrics].loc["min"].values.astype(float)
        mx = df_agg[metrics].loc["max"].values.astype(float)
        fig.add_trace(
            go.Scatter(
                x=metrics,
                y=mn,
                mode="lines" if len(metrics) > 1 else "markers",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
                marker=dict(size=10, color="gray", symbol="triangle-up"),
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=metrics,
                y=mx,
                mode="lines" if len(metrics) > 1 else "lines+markers",
                fill="tonexty",
                fillcolor="rgba(128,128,128,0.25)",
                line={"color": "rgba(128,128,128,0.6)", "width": 1},
                name="range min to max",
                showlegend=(col == 1),
                hovertemplate="%{x}: %{y:.4g}<extra></extra>",
                marker=dict(size=10, color="gray", symbol="triangle-down"),
            ),
            row=1,
            col=col,
        )
        # median
        med = df_agg[metrics].loc["median"].values.astype(float)
        fig.add_trace(
            go.Scatter(
                x=metrics,
                y=med,
                mode="lines" if len(metrics) > 1 else "markers",
                line={"color": "gray", "dash": "dash"},
                name="median",
                showlegend=(col == 1),
                hovertemplate="%{x}: %{y:.4g}<extra></extra>",
                marker=dict(size=10, color="gray", symbol="line-ew-open"),
            ),
            row=1,
            col=col,
        )

    # baselines (optional) ########################################################
    if df_baseline is not None and not df_baseline.empty:
        for i, (_, row) in enumerate(df_baseline.iterrows()):
            label = f"baseline: {row.get('name', str(i))}"
            showleg = True
            for col, metrics in (
                (1, error_metrics),
                (2, high_exp_error_metrics),
                (3, score_metrics),
            ):
                fig.add_trace(
                    go.Scatter(
                        x=metrics,
                        y=row[metrics].values.astype(float),
                        mode="lines+markers",
                        line={"width": 1.5, "color": "black"},
                        marker={"size": 5},
                        name=label if showleg and col == 1 else None,
                        showlegend=(showleg and col == 1),
                        hovertemplate="%{x}: %{y:.4g}<extra></extra>",
                    ),
                    row=1,
                    col=col,
                )
            showleg = False  # legend only once

    # target model (highlight) ##################################################
    if target_model_timestamp is None or target_model_timestamp == "":
        target_model_timestamp = df_show.index.max()
    if target_model_timestamp not in df_show.index:
        raise ValueError(f"timestamp '{target_model_timestamp}' not found in df.index")
    tgt = df_show.loc[target_model_timestamp]

    for col, metrics in (
        (1, error_metrics),
        (2, high_exp_error_metrics),
        (3, score_metrics),
    ):
        fig.add_trace(
            go.Scatter(
                x=metrics,
                y=tgt[metrics].values.astype(float),
                mode="lines+markers",
                line={"width": 1, "color": "red"},
                marker={"size": 7},
                name=f"model: {tgt['name']}" if col == 1 else None,
                showlegend=(col == 1),
                hovertemplate="%{x}: %{y:.4g}<extra></extra>",
            ),
            row=1,
            col=col,
        )

    # axes, layout ##############################################################
    fig.update_xaxes(tickangle=0, row=1, col=1)
    fig.update_yaxes(
        row=1,
        col=1,
        title_text="MSE: (W<sub>p</sub>h)<sup>2</sup>, MAPE: None, else W<sub>p</sub>h",
    )

    fig.update_xaxes(tickangle=0, row=1, col=2)
    fig.update_yaxes(
        row=1,
        col=2,
        type="log",
        title_text="MSE: (W<sub>p</sub>h)<sup>2</sup>, MAPE: None, else W<sub>p</sub>h",
    )

    fig.update_xaxes(tickangle=0, row=1, col=3)
    fig.update_yaxes(
        range=[0, 1],
        row=1,
        col=3,
        title_text="Score",
        automargin=True,
    )

    fig.for_each_trace(
        lambda t: t.update(line={"dash": "dash"} if t.name == "median" else {})
    )

    fig.update_layout(
        title="Comparison of models",
        legend={
            "orientation": "h",
            "y": -0.2,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        autosize=True,
        margin={"l": 40, "r": 60, "t": 50, "b": 80},
        hovermode="x unified",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)
