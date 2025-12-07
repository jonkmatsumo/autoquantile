import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from src.app.config_ui import render_levels_editor, render_location_targets_editor, render_location_settings_editor, render_config_ui

@pytest.fixture
def sample_config():
    return {
        "mappings": {
            "levels": {"E3": 0, "E4": 1},
            "location_targets": {"New York": 1, "Austin": 2}
        },
        "location_settings": {"max_distance_km": 50}
    }

def test_render_levels_editor(sample_config):
    with patch("src.app.config_ui.st") as mock_st:
        # Mock data_editor return value (dataframe)
        mock_df = pd.DataFrame([
            {"Level": "E3", "Rank": 0},
            {"Level": "E4", "Rank": 1},
            {"Level": "E5", "Rank": 2} # Added one
        ])
        mock_st.data_editor.return_value = mock_df
        
        updated_levels = render_levels_editor(sample_config)
        
        mock_st.subheader.assert_called_with("Levels Configuration")
        mock_st.data_editor.assert_called_once()
        
        assert updated_levels == {"E3": 0, "E4": 1, "E5": 2}

def test_render_location_targets_editor(sample_config):
    with patch("src.app.config_ui.st") as mock_st:
        mock_df = pd.DataFrame([
            {"City": "New York", "Tier/Rank": 1},
            {"City": "Austin", "Tier/Rank": 3} # Changed rank
        ])
        mock_st.data_editor.return_value = mock_df
        
        updated_loc = render_location_targets_editor(sample_config)
        
        mock_st.subheader.assert_called_with("Location Targets")
        assert updated_loc == {"New York": 1, "Austin": 3}

def test_render_location_settings_editor(sample_config):
    with patch("src.app.config_ui.st") as mock_st:
        mock_st.slider.return_value = 100
        
        updated_settings = render_location_settings_editor(sample_config)
        
        mock_st.subheader.assert_called_with("Location Settings")
        mock_st.slider.assert_called_with(
            "Max Distance (km) for Proximity Matching",
            min_value=0, max_value=200, value=50, step=5
        )
        assert updated_settings == {"max_distance_km": 100}

def test_render_config_ui(sample_config):
    # Integration test of the wrapper
    with patch("src.app.config_ui.st") as mock_st, \
         patch("src.app.config_ui.render_levels_editor") as mock_levels, \
         patch("src.app.config_ui.render_location_targets_editor") as mock_loc, \
         patch("src.app.config_ui.render_location_settings_editor") as mock_settings, \
         patch("src.app.config_ui.render_model_config_editor") as mock_model, \
         patch("src.app.config_ui.render_save_load_controls"): # Also patch save/load to avoid recursion/calls? 
         # Actually render_save_load_controls is harmless if mocked, but patching it avoids the JSON error too.
         # But the JSON error happens INSIDE render_save_load_controls.
         # If I patch it, the code inside won't run. That's good for THIS test (integration of wrappers).
        
        mock_levels.return_value = {"L": 1}
        mock_loc.return_value = {"C": 2}
        mock_settings.return_value = {"dist": 99}
        mock_model.return_value = {"targets": []}
        
        # Mock st.columns inside render_model_config_editor which is called by render_config_ui
        # Actually since we patched render_model_config_editor, we don't need to mock st.columns anymore!
        # mock_st.columns.return_value = [MagicMock(), MagicMock()] # Removed
        
        new_config = render_config_ui(sample_config)
        
        assert new_config["mappings"]["levels"] == {"L": 1}
        assert new_config["mappings"]["location_targets"] == {"C": 2}
        assert new_config["location_settings"] == {"dist": 99}
        assert new_config["model"] == {"targets": []}
        # Verify deep copy didn't affect original if we care (not strictly required by implementation but good practice)
        assert sample_config["location_settings"]["max_distance_km"] == 50

def test_render_model_config_editor(sample_config):
    from src.app.config_ui import render_model_config_editor
    
    with patch("src.app.config_ui.st") as mock_st:
        # Mock st.columns to return 2 objects
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        
        # Mock returns for data editors
        # 1. Targets
        mock_st.data_editor.side_effect = [
            pd.DataFrame([{"Target": "T1"}, {"Target": "T2"}]), # Targets
            pd.DataFrame([{"Quantile": 0.1}, {"Quantile": 0.9}]), # Quantiles
            pd.DataFrame([{"name": "F1", "monotone_constraint": 1}]) # Features
        ]
        
        # Mock return for inputs
        # 1. Sample Weight (number_input) -> 1.5
        # 2. Verbosity (number_input) -> 1
        # 3. Num Boost Rounds (number_input) -> 200
        # 4. N Folds (number_input) -> 10
        # 5. Early Stopping (number_input) -> 5
        mock_st.number_input.side_effect = [1.5, 1, 200, 10, 5]
        
        # Mock text_input for Objective -> "reg:squaredlogerror"
        mock_st.text_input.return_value = "reg:squaredlogerror"
        
        # Mock selectbox for Tree Method -> "approx"
        mock_st.selectbox.return_value = "approx"
        
        updated_model = render_model_config_editor(sample_config)
        
        # Verify structure
        assert updated_model["targets"] == ["T1", "T2"]
        assert updated_model["quantiles"] == [0.1, 0.9]
        assert updated_model["sample_weight_k"] == 1.5
        
        hp = updated_model["hyperparameters"]
        assert hp["training"]["objective"] == "reg:squaredlogerror"
        assert hp["training"]["tree_method"] == "approx"
        assert hp["training"]["verbosity"] == 1
        
        assert hp["cv"]["num_boost_round"] == 200
        assert hp["cv"]["nfold"] == 10
        
        feat = updated_model["features"]
        assert len(feat) == 1
        assert feat[0]["name"] == "F1"
        feat = updated_model["features"]
        assert len(feat) == 1
        assert feat[0]["name"] == "F1"
        assert feat[0]["monotone_constraint"] == 1

def test_render_save_load_controls_save():
    from src.app.config_ui import render_save_load_controls
    config = {"a": 1}
    
    with patch("src.app.config_ui.st") as mock_st, \
         patch("json.dumps", return_value='{"a": 1}') as mock_json_dumps:
        
        mock_st.file_uploader.return_value = None
        
        render_save_load_controls(config)
        
        mock_st.download_button.assert_called_once()
        args, kwargs = mock_st.download_button.call_args
        assert kwargs["data"] == '{"a": 1}'
        assert kwargs["file_name"] == "config.json"

def test_render_save_load_controls_load_success():
    from src.app.config_ui import render_save_load_controls
    import json
    
    config = {"a": 1}
    loaded_config = {"a": 2}
    
    with patch("src.app.config_ui.st") as mock_st:
        # Mock file uploader returning a file
        mock_file = MagicMock()
        mock_st.file_uploader.return_value = mock_file
        
        # Mock json load
        with patch("json.load", return_value=loaded_config):
            # Mock session state
            mock_st.session_state = {}
            
            render_save_load_controls(config)
            
            assert mock_st.session_state['config_override'] == loaded_config
            mock_st.rerun.assert_called_once()

def test_render_config_ui_uses_override(sample_config):
    from src.app.config_ui import render_config_ui
    
    override_config = {
        "mappings": {"levels": {"O": 99}, "location_targets": {}},
        "location_settings": {"max_distance_km": 10},
        "model": {} 
    }
    
    with patch("src.app.config_ui.st") as mock_st, \
         patch("src.app.config_ui.render_levels_editor") as mock_levels, \
         patch("src.app.config_ui.render_location_targets_editor") as mock_loc, \
         patch("src.app.config_ui.render_location_settings_editor") as mock_settings, \
         patch("src.app.config_ui.render_model_config_editor") as mock_model, \
         patch("src.app.config_ui.render_save_load_controls") as mock_save_load:

        # Use side_effects to return the existing values so config isn't mutated to empty
        mock_levels.side_effect = lambda c: c["mappings"].get("levels", {})
        mock_loc.side_effect = lambda c: c["mappings"].get("location_targets", {})
        mock_settings.side_effect = lambda c: c.get("location_settings", {})
        mock_model.side_effect = lambda c: c.get("model", {})

        # Set overrides in session state
        mock_st.session_state = {"config_override": override_config}
        
        # Mock st.columns needed by render_model_config_editor if called?
        # We mocked render_model_config_editor so we are good.
        
        render_config_ui(sample_config) # pass sample, but expect override used
        
        # Check that render_levels_editor was called with OVERRIDE config, not sample
        # We need to verify the arguments passed to the sub-helpers
        args, _ = mock_levels.call_args
        assert args[0] == override_config

