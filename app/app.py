import streamlit as st
import tab_predict_power_output
import tab_model_evaluation

tab1, tab2 = st.tabs(["Predict Power Output", "Model Evaluation"])

with tab1:
    tab_predict_power_output.render()

with tab2:
    tab_model_evaluation.render()
