import streamlit as st
import tab_predict_power_output
import tab_model_evaluation
import tab_model_retraining

tab1, tab2, tab3 = st.tabs(["Predict Power Output", "Model Evaluation", "Model Retraining"])

with tab1:
    tab_predict_power_output.render()

with tab2:
    tab_model_evaluation.render()

with tab3:
    tab_model_retraining.render()