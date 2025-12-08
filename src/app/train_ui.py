import streamlit as st
import pandas as pd
import time
from src.app.caching import load_data_cached as load_data
from src.services.training_service import TrainingService
from src.services.model_registry import ModelRegistry


@st.cache_resource
def get_training_service() -> TrainingService:
    return TrainingService()

def render_training_ui() -> None:
    """Renders the model training interface."""
    st.header("Model Training")
    
    st.info("Configure settings in 'Configuration' page before training.")
    

    df = None
    if "training_data" in st.session_state:
        df = st.session_state["training_data"]
        st.success(f"Using loaded data from Data Analysis ({len(df)} rows).")
        if st.button("Use Different File"):
            del st.session_state["training_data"]
            st.rerun()
    else:
        uploaded_file = st.file_uploader("Upload Training CSV", type=["csv"])
        if uploaded_file:
            try:
                df = load_data(uploaded_file)
                st.session_state["training_data"] = df

                st.success(f"Loaded {len(df)} rows.")
                st.info("💡 **Tip**: If this is a new dataset, go to the **Configuration** page to generate an optimal config.")
            except Exception as e:
                st.error(f"Error loading file: {e}")
                
    do_tune = st.checkbox("Run Hyperparameter Tuning", value=False)
    num_trials = 20
    if do_tune:
        num_trials = st.number_input("Number of Trials", 5, 100, 20)
        
    remove_outliers = st.checkbox("Remove Outliers (IQR)", value=True)
    

    display_charts = st.checkbox("Show Training Performance Chart", value=True)
    
    custom_name = st.text_input("Model Output Filename (Optional)", placeholder="e.g. my_custom_model.pkl")
    

    training_service = get_training_service()
    registry = ModelRegistry()


    if "training_job_id" not in st.session_state:
        st.session_state["training_job_id"] = None
        
    job_id = st.session_state["training_job_id"]


    if job_id is None:
        if st.button("Start Training (Async)"):
            if df is None:
                st.error("No data loaded.")
                return
                
            job_id = training_service.start_training_async(
                df, 
                remove_outliers=remove_outliers,
                do_tune=do_tune,
                n_trials=num_trials
            )
            st.session_state["training_job_id"] = job_id
            st.rerun()


    else:
        status = training_service.get_job_status(job_id)
        
        if status is None:
            st.error("Job not found. Clearing state.")
            st.session_state["training_job_id"] = None
            st.rerun()
            return

        state = status["status"]
        st.info(f"Training Status: **{state}**")
        

        if state in ["QUEUED", "RUNNING"]:
            with st.spinner("Training in progress... (You can switch tabs, but stay in app to see completion)"):
                # Poll every 2 seconds
                time.sleep(2) 
                st.rerun()
                

        with st.expander("Training Logs", expanded=(state != "COMPLETED")):
            st.code("\n".join(status["logs"]))


        if state == "COMPLETED":
            st.success("Training Finished Successfully!")
            
            # Result Visualization

            history = status.get("history", [])
            results_data = []
            
            for event in history:
                if event.get("stage") == "cv_end":
                    results_data.append({
                        "Model": event.get("model_name"),
                        "Best Round": event.get("best_round"),
                        "Score": event.get("best_score")
                    })
            
            if results_data:
                res_df = pd.DataFrame(results_data)
                if display_charts:
                    st.line_chart(res_df.set_index("Model")["Score"])
                st.dataframe(res_df.style.format({"Score": "{:.4f}"}))

            
            forecaster = status["result"]
            run_id = status.get("run_id", "N/A")
            
            st.session_state["forecaster"] = forecaster
            st.session_state["current_run_id"] = run_id
            
            st.info(f"Model logged to MLflow with Run ID: **{run_id}**")
            st.markdown("[Open MLflow UI](http://localhost:5000) to view details.")
                
            if st.button("Start New Training"):
                st.session_state["training_job_id"] = None
                st.rerun()
                
        elif state == "FAILED":
            st.error(f"Training Failed: {status.get('error')}")
            if st.button("Retry"):
                st.session_state["training_job_id"] = None
                st.rerun()
