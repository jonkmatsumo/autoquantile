import sys
import unittest
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd

from src.cli.generate_config_cli import generate_config_workflow, main

# Import conftest function directly (pytest will handle the path)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from conftest import create_test_config


class TestGenerateConfigCLI(unittest.TestCase):
    @patch("src.cli.generate_config_cli.argparse.ArgumentParser.parse_args")
    @patch("src.cli.generate_config_cli.pd.read_csv")
    @patch("src.cli.generate_config_cli.ConfigGenerator")
    @patch("src.cli.generate_config_cli.os.path.exists")
    @patch("src.cli.generate_config_cli.validate_config_dict")
    @patch("builtins.print")
    def test_heuristic_generation(
        self, mock_print, mock_validate, mock_exists, MockGenerator, mock_read_csv, mock_ears
    ):
        # Setup
        mock_exists.return_value = True
        config = create_test_config()
        mock_validate.return_value.model_dump.return_value = config
        mock_ears.return_value = MagicMock(
            input_file="data.csv",
            output=None,
            heuristic=True,
            verbose=False,
            preset=None,
            provider="openai",
        )
        mock_read_csv.return_value = pd.DataFrame({"A": [1]})

        mock_gen_instance = MockGenerator.return_value
        mock_gen_instance.generate_config_template.return_value = config

        # Run
        main()

        # Verify
        mock_gen_instance.generate_config_template.assert_called_with(mock_read_csv.return_value)
        mock_validate.assert_called_once()
        mock_print.assert_called()
        args = mock_print.call_args[0][0]
        self.assertIn("mappings", args)

    @patch("src.cli.generate_config_cli.argparse.ArgumentParser")
    @patch("src.cli.generate_config_cli.pd.read_csv")
    @patch("src.cli.generate_config_cli.generate_config_workflow")
    @patch("src.cli.generate_config_cli.os.path.exists")
    @patch("src.cli.generate_config_cli.validate_config_dict")
    @patch("builtins.open", new_callable=mock_open)
    def test_workflow_generation_file_output(
        self, mock_file, mock_validate, mock_exists, mock_workflow, mock_read_csv, MockParser
    ):
        # Setup
        mock_exists.return_value = True
        config = create_test_config()
        mock_validate.return_value.model_dump.return_value = config
        mock_workflow.return_value = config

        # Setup Parser Mock
        mock_parser_instance = MockParser.return_value
        mock_parser_instance.parse_args.return_value = MagicMock(
            input_file="data.csv",
            output="out.json",
            heuristic=False,
            provider="openai",
            verbose=True,
            preset="salary",
        )

        mock_read_csv.return_value = pd.DataFrame({"A": [1]})

        # Run
        main()

        # Verify
        mock_workflow.assert_called_once_with(
            mock_read_csv.return_value, provider="openai", preset="salary"
        )
        mock_validate.assert_called_once()
        mock_file.assert_called_with("out.json", "w")
        self.assertTrue(mock_file.return_value.write.called)

    @patch("src.cli.generate_config_cli.argparse.ArgumentParser.parse_args")
    @patch("src.cli.generate_config_cli.os.path.exists")
    def test_file_not_found(self, mock_exists, mock_ears):
        mock_exists.return_value = False
        mock_ears.return_value = MagicMock(
            input_file="missing.csv", verbose=False, heuristic=False, provider="openai", preset=None
        )

        with self.assertRaises(SystemExit) as cm:
            main()
        self.assertEqual(cm.exception.code, 1)


class TestGenerateConfigWorkflow(unittest.TestCase):
    """Tests for workflow-based config generation."""

    @patch("src.cli.generate_config_cli.WorkflowService")
    def test_generate_config_workflow_success(self, MockWorkflowService):
        """Test successful config generation using workflow."""
        mock_service = MagicMock()
        MockWorkflowService.return_value = mock_service

        config = create_test_config()

        # Mock workflow steps
        mock_service.start_workflow.return_value = {"status": "success", "phase": "classification"}
        mock_service.confirm_classification.return_value = {"status": "success", "phase": "encoding"}
        mock_service.confirm_encoding.return_value = {"status": "success", "phase": "configuration"}
        mock_service.get_final_config.return_value = config

        df = pd.DataFrame({"A": [1, 2, 3]})

        result = generate_config_workflow(df, provider="openai", preset="salary")

        self.assertEqual(result, config)
        mock_service.start_workflow.assert_called_once()
        mock_service.confirm_classification.assert_called_once()
        mock_service.confirm_encoding.assert_called_once()
        mock_service.get_final_config.assert_called_once()

    @patch("src.cli.generate_config_cli.WorkflowService")
    def test_generate_config_workflow_start_error(self, MockWorkflowService):
        """Test workflow error during start."""
        mock_service = MagicMock()
        MockWorkflowService.return_value = mock_service

        mock_service.start_workflow.return_value = {"status": "error", "error": "Start failed"}

        df = pd.DataFrame({"A": [1, 2, 3]})

        with self.assertRaises(RuntimeError) as context:
            generate_config_workflow(df)

        self.assertIn("Workflow failed", str(context.exception))
        self.assertIn("Start failed", str(context.exception))

    @patch("src.cli.generate_config_cli.WorkflowService")
    def test_generate_config_workflow_classification_error(self, MockWorkflowService):
        """Test workflow error during classification confirmation."""
        mock_service = MagicMock()
        MockWorkflowService.return_value = mock_service

        mock_service.start_workflow.return_value = {"status": "success"}
        mock_service.confirm_classification.return_value = {
            "status": "error",
            "error": "Classification failed",
        }

        df = pd.DataFrame({"A": [1, 2, 3]})

        with self.assertRaises(RuntimeError) as context:
            generate_config_workflow(df)

        self.assertIn("Classification confirmation failed", str(context.exception))

    @patch("src.cli.generate_config_cli.WorkflowService")
    def test_generate_config_workflow_encoding_error(self, MockWorkflowService):
        """Test workflow error during encoding confirmation."""
        mock_service = MagicMock()
        MockWorkflowService.return_value = mock_service

        mock_service.start_workflow.return_value = {"status": "success"}
        mock_service.confirm_classification.return_value = {"status": "success"}
        mock_service.confirm_encoding.return_value = {"status": "error", "error": "Encoding failed"}

        df = pd.DataFrame({"A": [1, 2, 3]})

        with self.assertRaises(RuntimeError) as context:
            generate_config_workflow(df)

        self.assertIn("Encoding confirmation failed", str(context.exception))

    @patch("src.cli.generate_config_cli.WorkflowService")
    def test_generate_config_workflow_no_final_config(self, MockWorkflowService):
        """Test workflow when final config is None."""
        mock_service = MagicMock()
        MockWorkflowService.return_value = mock_service

        mock_service.start_workflow.return_value = {"status": "success"}
        mock_service.confirm_classification.return_value = {"status": "success"}
        mock_service.confirm_encoding.return_value = {"status": "success"}
        mock_service.get_final_config.return_value = None

        df = pd.DataFrame({"A": [1, 2, 3]})

        with self.assertRaises(RuntimeError) as context:
            generate_config_workflow(df)

        self.assertIn("Failed to generate configuration", str(context.exception))


class TestConfigValidation(unittest.TestCase):
    """Tests for config validation in generate_config_cli."""

    @patch("src.cli.generate_config_cli.argparse.ArgumentParser.parse_args")
    @patch("src.cli.generate_config_cli.pd.read_csv")
    @patch("src.cli.generate_config_cli.ConfigGenerator")
    @patch("src.cli.generate_config_cli.os.path.exists")
    @patch("src.cli.generate_config_cli.validate_config_dict")
    def test_config_validation_failure(self, mock_validate, mock_exists, MockGenerator, mock_read_csv, mock_ears):
        """Test that invalid config causes exit."""
        mock_exists.return_value = True
        mock_ears.return_value = MagicMock(
            input_file="data.csv",
            output=None,
            heuristic=True,
            verbose=False,
            preset=None,
            provider="openai",
        )
        mock_read_csv.return_value = pd.DataFrame({"A": [1]})

        mock_gen_instance = MockGenerator.return_value
        invalid_config = {"invalid": "config"}
        mock_gen_instance.generate_config_template.return_value = invalid_config

        # Mock validation to raise error
        from pydantic import ValidationError

        mock_validate.side_effect = ValidationError.from_exception_data(
            "Config",
            [{"type": "missing", "loc": ("model",), "msg": "Field required", "input": {}}],
        )

        with self.assertRaises(SystemExit) as cm:
            main()
        self.assertEqual(cm.exception.code, 1)

    @patch("src.cli.generate_config_cli.argparse.ArgumentParser.parse_args")
    @patch("src.cli.generate_config_cli.pd.read_csv")
    @patch("src.cli.generate_config_cli.generate_config_workflow")
    @patch("src.cli.generate_config_cli.os.path.exists")
    @patch("src.cli.generate_config_cli.validate_config_dict")
    @patch("builtins.print")
    def test_workflow_output_validation(
        self, mock_print, mock_validate, mock_exists, mock_workflow, mock_read_csv, mock_ears
    ):
        """Test that workflow output is validated."""
        mock_exists.return_value = True
        config = create_test_config()
        mock_validate.return_value.model_dump.return_value = config
        mock_workflow.return_value = config

        mock_ears.return_value = MagicMock(
            input_file="data.csv",
            output=None,
            heuristic=False,
            verbose=False,
            preset=None,
            provider="openai",
        )
        mock_read_csv.return_value = pd.DataFrame({"A": [1]})

        main()

        # Verify validation was called
        mock_validate.assert_called_once_with(config)
        # Verify output was printed
        mock_print.assert_called()
