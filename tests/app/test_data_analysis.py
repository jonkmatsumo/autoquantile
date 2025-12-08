import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import streamlit as st
from src.app.data_analysis import render_data_analysis_ui

@pytest.fixture
def mock_streamlit():
    with patch("src.app.data_analysis.st") as mock_st:
        # Setup session state mock
        mock_st.session_state = {}
        # Setup columns mock to return 3 items
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        # Setup columns(2) as well just in case (e.g. for breakdown) - side_effect can handle this
        def columns_side_effect(spec):
             if isinstance(spec, list):
                 count = len(spec)
             else:
                 count = spec
             return [MagicMock() for _ in range(count)]
        mock_st.columns.side_effect = columns_side_effect
        
        # Setup selectbox to return a valid column
        mock_st.selectbox.return_value = "BaseSalary"
        
        yield mock_st

@pytest.fixture
def mock_load_data():
    with patch("src.app.data_analysis.load_data") as mock_ld:
        yield mock_ld

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Level": ["E3", "E4", "E3"],
        "Location": ["NY", "SF", "NY"],
        "BaseSalary": [100, 150, 110],
        "TotalComp": [120, 180, 130],
        "Stock": [10, 20, 10],
        "Bonus": [10, 10, 10],
        "YearsOfExperience": [1, 5, 2],
        "YearsAtCompany": [1, 2, 1]
    })

def test_render_no_data_shows_uploader(mock_streamlit):
    # Setup: No training_data in session state
    mock_streamlit.session_state = {}
    
    render_data_analysis_ui()
    
    # Assert Uploader is shown
    mock_streamlit.file_uploader.assert_called_with("Upload CSV", type=["csv"], key="analysis_uploader")
    # Assert Error/Success not shown (as no upload happened yet)
    mock_streamlit.success.assert_not_called()

def test_render_upload_loads_data(mock_streamlit, mock_load_data, sample_df):
    mock_streamlit.session_state = {}
    mock_streamlit.file_uploader.return_value = "dummy.csv"
    mock_load_data.return_value = sample_df
    
    render_data_analysis_ui()
    
    # Assert data loaded into session state
    assert "training_data" in mock_streamlit.session_state
    assert mock_streamlit.session_state["training_data"].equals(sample_df)
    
    # Assert rerun called
    mock_streamlit.rerun.assert_called()

def test_render_existing_data_shows_analysis(mock_streamlit, sample_df):
    mock_streamlit.session_state = {"training_data": sample_df}
    
    render_data_analysis_ui()
    
    # Assert Metrics shown
    assert mock_streamlit.columns.call_count >= 1
    # Verify we tried to show metrics
    # Note: st.columns returns list of columns, which we then call .metric on.
    # It's a bit complex to verify deep calls on return values without intricate mocking,
    # but we can verify general flow.
    
    # Assert Charts called
    mock_streamlit.bar_chart.assert_called() # Categorical breakdown
    
    # Assert Pyplot called (for Histograms - sns)
    mock_streamlit.pyplot.assert_called()

def test_clear_data_button(mock_streamlit, sample_df):
    mock_streamlit.session_state = {"training_data": sample_df}
    
    # Mock button to return True for "Clear Data"
    # Note: st.button might be called multiple times. We need to distinguish them.
    # "Clear Data" is the first button in the flow if data exists.
    mock_streamlit.button.side_effect = lambda label: label == "Clear Data"
    
    render_data_analysis_ui()
    
    # Assert data removed
    assert "training_data" not in mock_streamlit.session_state
    mock_streamlit.rerun.assert_called()

@patch("src.app.data_analysis.ConfigGenerator")
def test_config_generation_ui_flow(MockGenerator, mock_streamlit, sample_df):
    mock_streamlit.session_state = {"training_data": sample_df}
    
    # Mock ConfigGenerator
    mock_gen_instance = MockGenerator.return_value
    mock_gen_instance.generate_config_template.return_value = {"mapping": "heuristic"}
    mock_gen_instance.generate_config_with_llm.return_value = {"mapping": "llm"}
    mock_gen_instance.infer_levels.return_value = {"L": 1}
    mock_gen_instance.infer_locations.return_value = {"Loc": 2}
    
    # Simulate "Generate Config" Expander being open? Streamlit runs everything anyway.
    
    # 1. Test "Use AI" checkbox and "Generate Proposal"
    mock_streamlit.checkbox.return_value = True # Use AI
    
    # Selectbox side effect: Return "openai" for Provider, "BaseSalary" for component
    def selectbox_side_effect(label, options, **kwargs):
        if label == "Provider": return "openai"
        if label == "Select Component": return "BaseSalary"
        return options[0]
    mock_streamlit.selectbox.side_effect = selectbox_side_effect
    
    # Mock "Generate Proposal" button click
    # Mock button side effects: "Clear Data" -> False, "Generate Proposal" -> True, "Apply" -> False
    def button_side_effect(label, **kwargs):
        if label == "Generate Proposal": return True
        return False
    mock_streamlit.button.side_effect = button_side_effect
    
    # Mock data_editor to return empty DF (simplification)
    mock_streamlit.data_editor.return_value = pd.DataFrame()
    
    with patch("src.app.data_analysis.get_env_var", return_value="fake_key"):
        render_data_analysis_ui()
    
    # Verify LLM generation called
    mock_gen_instance.generate_config_with_llm.assert_called()
    assert mock_streamlit.session_state["gen_config_proposal"] == {"mapping": "llm"}

@patch("src.app.data_analysis.ConfigGenerator")
def test_apply_configuration(MockGenerator, mock_streamlit, sample_df):
    mock_streamlit.session_state = {
        "training_data": sample_df, 
        "gen_config_proposal": {"mappings": {"levels": {}}, "model": {}}
    }
    
    mock_gen_instance = MockGenerator.return_value
    mock_gen_instance.generate_config_template.return_value = {"mappings": {}, "model": {}}
    mock_gen_instance.infer_locations.return_value = {"Loc": 2}

    # Mock edits
    mapping_df = pd.DataFrame([
        {"Column": "BaseSalary", "Role": "Target", "Monotone": 0},
        {"Column": "Years", "Role": "Feature", "Monotone": 1}
    ])
    levels_df = pd.DataFrame([{"Level": "E3", "Rank": 0}])
    
    # mock data_editor returns: 1. mapping_df, 2. levels_df
    # AND mock selectbox for the plot that happens after re-run logic (or before)
    mock_streamlit.data_editor.side_effect = [mapping_df, levels_df]
    mock_streamlit.selectbox.return_value = "BaseSalary" # For plots
    
    # Mock "Apply Configuration" button
    mock_streamlit.button.side_effect = lambda label: label == "Apply Configuration"
    
    render_data_analysis_ui()
    
    # Verify config override set
    assert "config_override" in mock_streamlit.session_state
    override = mock_streamlit.session_state["config_override"]
    assert override["model"]["targets"] == ["BaseSalary"]
    
    # Check features contains Years (ignoring order/extras)
    feature_names = [f["name"] for f in override["model"]["features"]]
    assert "Years" in feature_names
    
    # Also Check implicit encoders
    assert "Level_Enc" in feature_names
    assert "Location_Enc" in feature_names
