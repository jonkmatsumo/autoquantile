from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.cli.train_cli import generate_config_from_data, load_config_from_file, main, train_workflow

# Import conftest function directly (pytest will handle the path)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from conftest import create_test_config


@patch("src.cli.train_cli.Console")
@patch("src.cli.train_cli.load_config_from_file")
@patch("src.cli.train_cli.train_workflow")
def test_train_cli_main(mock_train_workflow, mock_load_config, MockConsole):
    # Mock config
    from conftest import create_test_config
    config = create_test_config()
    mock_load_config.return_value = config
    
    # Run
    with patch("sys.argv", ["script", "--csv", "data.csv", "--config", "config.json", "--tune"]):
        main()

    # Verify train_workflow was called with correct arguments
    mock_train_workflow.assert_called_once()
    call_args = mock_train_workflow.call_args
    assert call_args[0][0] == "data.csv"  # CSV path
    assert call_args[0][1] == config  # Config
    assert call_args[1]["do_tune"] is True  # Tune flag


@patch("src.cli.train_cli.Console")
@patch("src.cli.train_cli.load_config_from_file")
def test_train_cli_defaults(mock_load_config, mock_console):
    # Verify defaults - now requires --config or --generate-config
    from conftest import create_test_config
    
    config = create_test_config()
    mock_load_config.return_value = config
    
    with patch("src.cli.train_cli.train_workflow") as mock_workflow:
        with patch("sys.argv", ["script", "--config", "config.json"]):
            main()
            mock_workflow.assert_called_once()
            call_args = mock_workflow.call_args[0]
            assert call_args[0] == "salaries-list.csv"  # CSV
            assert call_args[2] is None  # Output default changed to None


@patch("sys.argv", ["prog", "--config", "config.json", "--tune", "--num-trials", "50"])
@patch("src.cli.train_cli.load_config_from_file")
def test_train_cli_tune(mock_load_config):
    from conftest import create_test_config
    
    config = create_test_config()
    mock_load_config.return_value = config
    
    with (
        patch("src.cli.train_cli.Console") as MockConsole,
        patch("src.cli.train_cli.train_workflow") as mock_train_workflow,
    ):
        main()

        mock_train_workflow.assert_called_once()
        args, kwargs = mock_train_workflow.call_args
        assert kwargs["do_tune"] is True
        assert kwargs["num_trials"] == 50


@patch("src.cli.train_cli.mlflow")
def test_train_workflow(mock_mlflow):
    # Mock dependencies
    with (
        patch("src.cli.train_cli.load_data") as mock_load_data,
        patch("src.cli.train_cli.SalaryForecaster") as MockForecaster,
        patch("os.path.exists", return_value=True),
        patch("src.cli.train_cli.Live"),
        patch("src.cli.train_cli.Group"),
    ):

        # Setup mocks
        mock_df = MagicMock()
        mock_df.__len__.return_value = 100  # Simulate loaded data
        mock_load_data.return_value = mock_df

        mock_model = MockForecaster.return_value
        # Mock predict to return dictionary logic for output printing
        mock_model.quantiles = [0.10, 0.50, 0.90]
        mock_model.predict.return_value = {
            "BaseSalary": {"p10": [150000], "p50": [200000], "p90": [250000]}
        }

        mock_console = MagicMock()
        config = create_test_config()

        # Run train_workflow
        train_workflow(
            csv_path="test.csv",
            config=config,
            output_path="test_model.pkl",
            console=mock_console,
        )

        # Verify interactions
        mock_load_data.assert_called_once_with("test.csv")
        MockForecaster.assert_called_once_with(config=config)
        # Verify train was called (we can't easily assert the local callback function equality)
        assert mock_model.train.call_count == 1
        args, kwargs = mock_model.train.call_args
        assert args[0] == mock_df
        assert "callback" in kwargs
        assert callable(kwargs["callback"])

        # Verify MLflow calls
        mock_mlflow.start_run.assert_called()
        mock_mlflow.log_params.assert_called()
        mock_mlflow.pyfunc.log_model.assert_called()


def test_load_config_from_file():
    """Test loading config from file."""
    import json
    import tempfile

    config = create_test_config()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        temp_path = f.name

    try:
        loaded_config = load_config_from_file(temp_path)
        # Validation adds hyperparameters field with default {}, so account for that
        expected_config = config.copy()
        if "hyperparameters" not in expected_config.get("model", {}):
            expected_config.setdefault("model", {})["hyperparameters"] = {}
        assert loaded_config == expected_config
    finally:
        import os

        os.unlink(temp_path)


def test_load_config_from_file_not_found():
    """Test loading config from non-existent file raises error."""
    with pytest.raises(FileNotFoundError):
        load_config_from_file("nonexistent.json")


def test_load_config_from_file_invalid():
    """Test loading invalid config raises error."""
    import json
    import tempfile

    invalid_config = {"invalid": "config"}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(invalid_config, f)
        temp_path = f.name

    try:
        with pytest.raises(ValueError) as exc_info:
            load_config_from_file(temp_path)
        assert "validation failed" in str(exc_info.value).lower()
    finally:
        import os

        os.unlink(temp_path)


@patch("src.cli.train_cli.WorkflowService")
def test_generate_config_from_data(mock_workflow_service_class):
    """Test generating config from data using workflow."""
    mock_service = MagicMock()
    mock_workflow_service_class.return_value = mock_service

    # Mock workflow steps
    mock_service.start_workflow.return_value = {"status": "success", "phase": "classification"}
    mock_service.confirm_classification.return_value = {"status": "success", "phase": "encoding"}
    mock_service.confirm_encoding.return_value = {"status": "success", "phase": "configuration"}
    mock_service.get_final_config.return_value = create_test_config()

    with patch("src.cli.train_cli.load_data") as mock_load_data:
        mock_df = pd.DataFrame({"A": [1, 2, 3]})
        mock_load_data.return_value = mock_df

        with patch("os.path.exists", return_value=True):
            config = generate_config_from_data("test.csv")

            assert config == create_test_config()
            mock_service.start_workflow.assert_called_once()
            mock_service.confirm_classification.assert_called_once()
            mock_service.confirm_encoding.assert_called_once()
            mock_service.get_final_config.assert_called_once()


@patch("src.cli.train_cli.WorkflowService")
def test_generate_config_from_data_workflow_error(mock_workflow_service_class):
    """Test generating config when workflow fails."""
    mock_service = MagicMock()
    mock_workflow_service_class.return_value = mock_service

    mock_service.start_workflow.return_value = {"status": "error", "error": "Workflow failed"}

    with patch("src.cli.train_cli.load_data") as mock_load_data:
        mock_df = pd.DataFrame({"A": [1, 2, 3]})
        mock_load_data.return_value = mock_df

        with patch("os.path.exists", return_value=True):
            with pytest.raises(RuntimeError) as exc_info:
                generate_config_from_data("test.csv")
            assert "Workflow failed" in str(exc_info.value)


def test_train_workflow_with_missing_config():
    """Test train_workflow with missing config."""
    mock_console = MagicMock()
    # train_workflow now requires config as positional arg
    # The function should handle None/empty config gracefully
    with patch("os.path.exists", return_value=True):
        train_workflow("test.csv", None, None, mock_console)
        # Should print error message about config being required
        calls = [str(call) for call in mock_console.print.call_args_list]
        assert any("Config is required" in call or "Error" in call for call in calls)


@patch("src.cli.train_cli.mlflow")
def test_train_workflow_with_empty_config(mock_mlflow):
    """Test train_workflow with empty config."""
    mock_console = MagicMock()
    with patch("os.path.exists", return_value=True):
        train_workflow("test.csv", {}, None, mock_console)
        # Should return early without error (graceful handling)
        mock_console.print.assert_called()


@patch("src.cli.train_cli.Console")
def test_train_cli_missing_config_argument(mock_console):
    """Test that CLI requires --config or --generate-config."""
    with patch("sys.argv", ["script", "--csv", "data.csv"]):
        # Should fail because neither --config nor --generate-config provided
        with pytest.raises(SystemExit):
            main()


@patch("src.cli.train_cli.Console")
@patch("src.cli.train_cli.generate_config_from_data")
def test_train_cli_generate_config_flag(mock_generate_config, mock_console):
    """Test --generate-config flag."""
    config = create_test_config()
    mock_generate_config.return_value = config

    with (
        patch("sys.argv", ["script", "--csv", "data.csv", "--generate-config"]),
        patch("os.path.exists", return_value=True),
        patch("src.cli.train_cli.train_workflow") as mock_train_workflow,
    ):
        main()

        mock_generate_config.assert_called_once()
        mock_train_workflow.assert_called_once()
        # Verify config was passed to train_workflow
        call_args = mock_train_workflow.call_args
        assert call_args[0][1] == config  # config is second positional arg
