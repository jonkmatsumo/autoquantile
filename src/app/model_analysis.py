import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
from src.services.model_registry import ModelRegistry
from src.services.analytics_service import AnalyticsService

def render_model_analysis_ui() -> None:
    """Renders the model analysis dashboard."""
    st.header("Model Analysis")
    
    registry = ModelRegistry()
    
    model_files = registry.list_models()
    if not model_files:
        st.warning("No model files (*.pkl) found in the root directory. Please train a model first.")
        return

    selected_model_file = st.selectbox("Select Model to Analyze", model_files)
    
    if selected_model_file:
        try:
            forecaster = registry.load_model(selected_model_file)
            st.success(f"Loaded `{selected_model_file}`")
            
            st.subheader("Feature Importance")
            st.info("Visualize which features drive the predictions (Gain metric).")
            
            analytics_service = AnalyticsService()
            targets = analytics_service.get_available_targets(forecaster)
            
            if not targets:
                 st.error("This model file does not appear to contain trained models.")
                 return

            selected_target = st.selectbox("Select Target Component", targets)
            
            if selected_target:
                quantiles = analytics_service.get_available_quantiles(forecaster, selected_target)
                selected_q_val = st.selectbox("Select Quantile", quantiles, format_func=lambda x: f"P{int(x*100)}")
                
                if selected_q_val is not None:
                    df_imp = analytics_service.get_feature_importance(forecaster, selected_target, selected_q_val)
                    
                    if df_imp is None or df_imp.empty:
                        st.warning("No feature importance scores found.")
                    else:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(data=df_imp, x="Gain", y="Feature", hue="Feature", ax=ax, palette="viridis", legend=False)
                        ax.set_title(f"Feature Importance (Gain) - {selected_target} P{int(selected_q_val*100)}")
                        st.pyplot(fig)
                        
                        with st.expander("View Raw Scores"):
                            st.dataframe(df_imp)
                        
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.code(traceback.format_exc())
