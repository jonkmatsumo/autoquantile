import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import streamlit as st
from src.app.model_analysis import render_model_analysis_ui

@pytest.fixture
def mock_streamlit():
    with patch("src.app.model_analysis.st") as mock_st:
        # Defaults
        mock_st.selectbox.return_value = None
        yield mock_st

@pytest.fixture
def mock_registry():
    with patch("src.app.model_analysis.ModelRegistry") as MockReg:
        yield MockReg.return_value

@pytest.fixture
def mock_analytics():
    with patch("src.app.model_analysis.AnalyticsService") as MockAn:
        yield MockAn.return_value

def test_no_models_shows_warning(mock_streamlit, mock_registry):
    mock_registry.list_models.return_value = []
    render_model_analysis_ui()
    mock_streamlit.warning.assert_called_with("No model files (*.pkl) found in the root directory. Please train a model first.")

def test_load_valid_model(mock_streamlit, mock_registry, mock_analytics):
    mock_registry.list_models.return_value = ["model.pkl"]
    mock_streamlit.selectbox.side_effect = ["model.pkl", "BaseSalary", 0.5] # Model, Target, Quantile
    
    # Mock Forecaster object
    mock_forecaster = MagicMock()
    mock_registry.load_model.return_value = mock_forecaster
    
    # Mock Analytics Responses
    mock_analytics.get_available_targets.return_value = ["BaseSalary"]
    mock_analytics.get_available_quantiles.return_value = [0.5]
    
    # Mock Feature Importance
    df_imp = pd.DataFrame({"Feature": ["A"], "Gain": [10.0]})
    mock_analytics.get_feature_importance.return_value = df_imp
    
    render_model_analysis_ui()
    
    # Verify success loading
    mock_streamlit.success.assert_called()
    
    # Verify plotting
    # Assuming get_feature_importance returns valid DF, it plots
    mock_streamlit.pyplot.assert_called()

def test_empty_importance(mock_streamlit, mock_registry, mock_analytics):
    mock_registry.list_models.return_value = ["model.pkl"]
    mock_streamlit.selectbox.side_effect = ["model.pkl", "BaseSalary", 0.5]
    
    mock_forecaster = MagicMock()
    mock_registry.load_model.return_value = mock_forecaster
    
    mock_analytics.get_available_targets.return_value = ["BaseSalary"]
    mock_analytics.get_available_quantiles.return_value = [0.5]
    
    # Return empty importance
    mock_analytics.get_feature_importance.return_value = pd.DataFrame()
    
    render_model_analysis_ui()
        
    mock_streamlit.warning.assert_called()
    assert "No feature importance scores found" in mock_streamlit.warning.call_args_list[-1][0][0]
