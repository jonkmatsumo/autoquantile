import streamlit as st
import pandas as pd
import json
import traceback
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, Dict, Any, List


current_dir = os.path.dirname(os.path.abspath(__file__))

from src.utils.compatibility import apply_backward_compatibility
apply_backward_compatibility()

from src.xgboost.model import SalaryForecaster
from src.app.config_ui import render_config_ui
from src.app.data_analysis import render_data_analysis_ui
from src.app.model_analysis import render_model_analysis_ui
from src.app.train_ui import render_training_ui
from src.utils.config_loader import get_config
from src.app.caching import load_data_cached as load_data
from src.utils.logger import setup_logging


from src.services.model_registry import ModelRegistry



def render_inference_ui() -> None:
    """Renders the inference interface."""
    st.header("Salary Inference")
    
    registry = ModelRegistry()
    
    # Display format: "RunID (Date) - Metric"
    runs = registry.list_models()

    
    if not runs:
        st.warning("No trained models found in MLflow. Please train a new model.")
        return


    def fmt_score(x):
        try:
            return f"{float(x):.4f}"
        except (ValueError, TypeError):
            return str(x)

    run_options = {f"{r['start_time'].strftime('%Y-%m-%d %H:%M')} | CV:{fmt_score(r.get('metrics.cv_mean_score', 'N/A'))} | ID:{r['run_id'][:8]}": r['run_id'] for r in runs}
    
    selected_label = st.selectbox("Select Model Version", options=list(run_options.keys()))
    
    if not selected_label:
        return
        
    run_id = run_options[selected_label]
    

    if "forecaster" not in st.session_state or st.session_state.get("current_run_id") != run_id:
        with st.spinner(f"Loading model from MLflow run {run_id}..."):
            try:
                st.session_state["forecaster"] = registry.load_model(run_id)
                st.session_state["current_run_id"] = run_id
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                return
    
    forecaster: SalaryForecaster = st.session_state["forecaster"]
    
    with st.form("inference_form"):
        st.subheader("Candidate Details")
        c1, c2 = st.columns(2)
        
        with c1:
            level_map = forecaster.level_encoder.mapping
            levels = list(level_map.keys()) if level_map else ["E3", "E4", "E5"]
            level = st.selectbox("Level", levels)
            
            location = st.text_input("Location", "New York")
            
        with c2:
            yoe = st.number_input("Years of Experience", 0, 30, 5)
            yac = st.number_input("Years at Company", 0, 30, 0)
            
        if st.form_submit_button("Predict Compensation"):
            input_df = pd.DataFrame([{
                "Level": level,
                "Location": location,
                "YearsOfExperience": yoe,
                "YearsAtCompany": yac
            }])
            
            with st.spinner("Predicting..."):
                results = forecaster.predict(input_df)
                
            st.subheader("Prediction Results")
            st.markdown(f"**Target Location Zone:** {forecaster.loc_encoder.mapper.get_zone(location)}")
            
            # Prepare data for display
            res_data = []
            for target, preds in results.items():
                row = {"Component": target}
                for q_key, val in preds.items():
                    row[q_key] = val[0]
                res_data.append(row)
                
            res_df = pd.DataFrame(res_data)
            
            # Visualization (Interactive Line Chart)
            # X-axis = Percentiles (p10, p25...), Lines = Components
            chart_df = res_df.set_index("Component").T
            
            # Sort index (percentiles) numerically

            try:
                # Extract integer part for sorting
                sorted_index = sorted(chart_df.index, key=lambda x: int(x.replace("p", "")))
                chart_df = chart_df.reindex(sorted_index)
            except ValueError:
                # Fallback if index format is unexpected
                pass
                
            st.line_chart(chart_df)
            

            st.dataframe(res_df.style.format({c: "${:,.0f}" for c in res_df.columns if c != "Component"}))



def main() -> None:
    """Main entry point for the Streamlit application."""
    st.set_page_config(page_title="Salary Forecaster", layout="wide")
    
    config = get_config()
    st.session_state["config_override"] = config

    
    st.sidebar.title("Navigation")
    

    if "nav" not in st.session_state:
        st.session_state["nav"] = "Inference"
        
    options = ["Inference", "Training", "Data Analysis", "Model Analysis", "Configuration"]
    default_index = 0
    if st.session_state.get("nav") in options:
        default_index = options.index(st.session_state["nav"])
        
    nav = st.sidebar.radio("Go to", options, index=default_index, key="nav_radio")
    

    st.session_state["nav"] = nav
    
    if nav == "Inference":
        render_inference_ui()
    elif nav == "Training":
        render_training_ui()
    elif nav == "Data Analysis":
        render_data_analysis_ui()
    elif nav == "Model Analysis":
        render_model_analysis_ui()
    elif nav == "Configuration":
        new_config = render_config_ui(config)
        st.session_state["config_override"] = new_config

if __name__ == "__main__":
    setup_logging()
    main()
