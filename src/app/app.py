import streamlit as st
import pandas as pd
import json
import traceback
import os
import pickle
import glob
import sys
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, Dict, Any, List

# Ensure src can be imported if running from inside src/app or root
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../.."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.model.model import SalaryForecaster
from src.app.config_ui import render_config_ui
from src.app.data_analysis import render_data_analysis_ui
from src.app.model_analysis import render_model_analysis_ui
from src.utils.config_loader import get_config
from src.app.caching import load_data_cached as load_data

def render_inference_ui() -> None:
    """Renders the inference interface."""
    st.header("Salary Inference")
    
    # Check if model is loaded
    if "forecaster" not in st.session_state:
        # Try to find models
        model_files = glob.glob("*.pkl")
        if not model_files:
            st.warning("No model files found. Please train a model first.")
            if st.button("Go to Training"):
                st.session_state["nav"] = "Training"
                st.rerun()
            return
            
        selected_model = st.selectbox("Select Model", model_files)
        if st.button("Load Model"):
            try:
                with open(selected_model, "rb") as f:
                    st.session_state["forecaster"] = pickle.load(f)
                st.success(f"Loaded {selected_model}")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading model: {e}")
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
            
            # Display as a table
            res_data = []
            for target, preds in results.items():
                row = {"Component": target}
                for q_key, val in preds.items():
                    row[q_key] = val[0]
                res_data.append(row)
                
            res_df = pd.DataFrame(res_data)
            st.dataframe(res_df.style.format({c: "${:,.0f}" for c in res_df.columns if c != "Component"}))

def render_training_ui() -> None:
    """Renders the model training interface."""
    st.header("Model Training")
    
    st.info("Configure settings in 'Configuration' page before training.")
    
    csv_file = st.text_input("Training Data CSV Path", "data/salary_data.csv")
    
    do_tune = st.checkbox("Run Hyperparameter Tuning", value=False)
    num_trials = 20
    if do_tune:
        num_trials = st.number_input("Number of Trials", 5, 100, 20)
        
    remove_outliers = st.checkbox("Remove Outliers (IQR)", value=True)
    
    if st.button("Start Training"):
        if not os.path.exists(csv_file):
            st.error(f"File not found: {csv_file}")
            return
            
        status_container = st.empty()
        log_container = st.empty()
        logs = []
        
        def streamlit_callback(msg: str, data: Optional[Dict[str, Any]] = None) -> None:
            status_container.markdown(f"**Status:** {msg}")
            logs.append(msg)
            if len(logs) > 10:
                logs.pop(0)
            log_container.code("\n".join(logs))
            
        try:
            df = load_data(csv_file)
            
            # Use current config
            forecaster = SalaryForecaster()
            
            if do_tune:
                streamlit_callback("Starting tuning...")
                best_params = forecaster.tune(df, n_trials=num_trials)
                st.json(best_params)
            
            streamlit_callback("Starting training...")
            forecaster.train(df, callback=streamlit_callback, remove_outliers=remove_outliers)
            
            # Save
            output_path = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            with open(output_path, "wb") as f:
                pickle.dump(forecaster, f)
                
            st.success(f"Training Complete! Model saved to {output_path}")
            st.session_state["forecaster"] = forecaster
            
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.code(traceback.format_exc())

def main() -> None:
    """Main entry point for the Streamlit application."""
    st.set_page_config(page_title="Salary Forecaster", layout="wide")
    
    # Initialize
    config = get_config()
    st.session_state["config_override"] = config
    
    st.sidebar.title("Navigation")
    
    # Use session state for nav persistence if needed, but sidebar widget handles it
    if "nav" not in st.session_state:
        st.session_state["nav"] = "Inference"
        
    # We can't easily sync st.sidebar.radio with session_state unless we use index/on_change
    # For now, let's just let the user pick
    # But wait, if we redirect from Inference to Training, we might want to update the sidebar.
    # Streamlit sidebar navigation is tricky to programmatically change without rerun.
    # We will use the simple approach: defaults to Inference, if session_state["nav"] is set, try to use it.
    
    options = ["Inference", "Training", "Data Analysis", "Model Analysis", "Configuration"]
    default_index = 0
    if st.session_state.get("nav") in options:
        default_index = options.index(st.session_state["nav"])
        
    nav = st.sidebar.radio("Go to", options, index=default_index, key="nav_radio")
    
    # Update session state to match (to sync back if user clicked radio)
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
    main()
