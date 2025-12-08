import pytest
from unittest.mock import patch, MagicMock
from src.app.train_ui import render_training_ui

@pytest.fixture
def mock_streamlit():
    with patch("src.app.train_ui.st") as mock_st:
        mock_st.session_state = {}
        yield mock_st

@pytest.fixture
def mock_load_data():
    with patch("src.app.train_ui.load_data") as mock_ld:
        yield mock_ld

@pytest.fixture
def mock_training_service():
    with patch("src.app.train_ui.get_training_service") as mock_get_svc:
        yield mock_get_svc

@pytest.fixture
def mock_registry():
    with patch("src.app.train_ui.ModelRegistry") as mock_reg:
        yield mock_reg

def test_render_training_ui_upload_redirect(mock_streamlit, mock_load_data, mock_training_service, mock_registry):
    # Setup: No training_data, uploader active
    mock_streamlit.session_state = {}
    mock_upload_file = MagicMock()
    mock_streamlit.file_uploader.return_value = mock_upload_file
    
    # Validation succeeds
    df = MagicMock()
    df.__len__.return_value = 10
    mock_load_data.return_value = df
    
    render_training_ui()
    
    # Assert data loaded
    assert "training_data" in mock_streamlit.session_state
    
    # Assert Success Message
    mock_streamlit.success.assert_called()
    
    # Assert Redirect Info (Tip)
    found_redirect = False
    for call in mock_streamlit.info.call_args_list:
        if "Tip" in call[0][0] and "Configuration" in call[0][0]:
            found_redirect = True
            break
    assert found_redirect, "Redirect tip message not found in Training UI"
            
def test_render_training_ui_starts_job(mock_streamlit, mock_load_data, mock_training_service):
    # Setup state
    df = MagicMock()
    mock_streamlit.session_state = {
        "training_data": df,
        "training_dataset_name": "dataset.csv",
        "training_job_id": None
    }
    
    # Mock inputs
    mock_streamlit.checkbox.return_value = False # no tune
    mock_streamlit.number_input.return_value = 20
    # text_input for additional tag
    mock_streamlit.text_input.return_value = "tag-v1"
    
    # Mock Start Button -> True
    mock_streamlit.button.side_effect = [False, True] # First (load different) False, Second (Start) True
    
    # Mock Service
    service_instance = mock_training_service.return_value
    service_instance.start_training_async.return_value = "new_job_id"
    
    render_training_ui()
    
    service_instance.start_training_async.assert_called_with(
        df,
        remove_outliers=False,
        do_tune=False,
        n_trials=20,
        additional_tag="tag-v1",
        dataset_name="dataset.csv"
    )
    
    assert mock_streamlit.session_state["training_job_id"] == "new_job_id"
    mock_streamlit.rerun.assert_called()
