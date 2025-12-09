import streamlit as st
import pandas as pd
import time
from src.app.caching import load_data_cached as load_data
from src.services.training_service import TrainingService
from src.services.model_registry import ModelRegistry
from src.app.config_ui import render_workflow_wizard, _reset_workflow_state
from src.services.workflow_service import get_workflow_providers


@st.cache_resource
def get_training_service() -> TrainingService:
    return TrainingService()

def render_training_ui() -> None:
    """Renders the model training interface."""
    st.header("Model Training")
    

    df = None
    if "training_data" in st.session_state:
        df = st.session_state["training_data"]
        dataset_name = st.session_state.get("training_dataset_name", "Unknown")
        st.success(f"Using loaded data: **{dataset_name}** ({len(df)} rows).")
        if st.button("Use Different File"):
            del st.session_state["training_data"]
            if "training_dataset_name" in st.session_state:
                del st.session_state["training_dataset_name"]
            st.rerun()
    else:
        uploaded_file = st.file_uploader("Upload Training CSV", type=["csv"])
        if uploaded_file:
            try:
                df = load_data(uploaded_file)
                st.session_state["training_data"] = df
                st.session_state["training_dataset_name"] = uploaded_file.name

                st.success(f"Loaded {len(df)} rows.")
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    if df is None:
        st.info("Please upload a CSV file to begin.")
        return
    
    # AI-Powered Configuration Wizard (Required before training)
    st.markdown("---")
    wizard_completed = st.session_state.get("workflow_phase") == "complete"
    
    if not wizard_completed:
        with st.expander("AI-Powered Configuration Wizard", expanded=True):
            st.write("**Required:** Complete the configuration wizard before you can start training.")
            st.info("Generate optimal configuration using an intelligent multi-step workflow.")
            
            # Provider selection
            available_providers = get_workflow_providers()
            if not available_providers:
                available_providers = ["openai", "gemini"]
            
            provider = st.selectbox(
                "LLM Provider",
                available_providers,
                index=0,
                key="wizard_provider_training"
            )
            
            result = render_workflow_wizard(df, provider)
            
            if result:
                st.session_state["config_override"] = result
                st.success("✅ Configuration generated and applied! You can now proceed with training.")
                st.rerun()
    else:
        st.success("✅ Configuration wizard completed. You can now configure training options below.")
        if st.button("Re-run Configuration Wizard"):
            # Reset workflow state to allow re-running
            _reset_workflow_state()
            st.rerun()
    
    # Training controls - only show if wizard is completed
    if not wizard_completed:
        st.info("⏳ Please complete the AI-Powered Configuration Wizard above to enable training options.")
        return
                
    st.markdown("---")
    st.subheader("Training Configuration")
    
    do_tune = st.checkbox("Run Hyperparameter Tuning", value=False)
    num_trials = 20
    if do_tune:
        num_trials = st.number_input("Number of Trials", 5, 100, 20)
        
    remove_outliers = st.checkbox("Remove Outliers (IQR)", value=True)
    

    display_charts = st.checkbox("Show Training Performance Chart", value=True)
    
    additional_tag = st.text_input("Additional Tag (Optional)", placeholder="e.g. experimental-v1")
    

    training_service = get_training_service()
    registry = ModelRegistry()


    if "training_job_id" not in st.session_state:
        st.session_state["training_job_id"] = None
        
    job_id = st.session_state["training_job_id"]


    if job_id is None:
        if st.button("Start Training (Async)", type="primary"):
            if df is None:
                st.error("No data loaded.")
                return
                
            dataset_name = st.session_state.get("training_dataset_name", "Unknown Data")
            
            job_id = training_service.start_training_async(
                df, 
                remove_outliers=remove_outliers,
                do_tune=do_tune,
                n_trials=num_trials,
                additional_tag=additional_tag if additional_tag.strip() else None,
                dataset_name=dataset_name
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
