import unittest
from unittest.mock import patch, MagicMock
from src.services.model_registry import ModelRegistry
import pandas as pd
from datetime import datetime

class TestModelRegistry(unittest.TestCase):
    @patch("src.services.model_registry.mlflow")
    @patch("src.services.model_registry.MlflowClient")
    def setUp(self, MockClient, mock_mlflow):
        self.registry = ModelRegistry()

    @patch("src.services.model_registry.mlflow.search_runs")
    def test_list_models(self, mock_search):
        # Mock dataframe return from search_runs
        mock_df = pd.DataFrame({
            "run_id": ["run1", "run2"],
            "start_time": [datetime(2023,1,1), datetime(2023,1,2)],
            "status": ["FINISHED", "FINISHED"],
            "metrics.cv_mean_score": [0.95, 0.96]
        })
        mock_search.return_value = mock_df
        
        models = self.registry.list_models()
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0]["run_id"], "run1")

    @patch("src.services.model_registry.mlflow.search_runs")
    def test_list_models_missing_col(self, mock_search):
        # Mock dataframe WITHOUT metric column
        mock_df = pd.DataFrame({
            "run_id": ["run1"],
            "start_time": [datetime(2023,1,1)],
            "status": ["FINISHED"]
        })
        mock_search.return_value = mock_df
        
        models = self.registry.list_models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["run_id"], "run1")
        # Ensure it didn't crash and metric key is absent
        self.assertNotIn("metrics.cv_mean_score", models[0])

    @patch("src.services.model_registry.mlflow.pyfunc.load_model")
    def test_load_model(self, mock_load):
        mock_model_wrapper = MagicMock()
        # First unwrap returns the SalaryForecasterWrapper instance
        # Second unwrap (called on Wrapper) returns the Inner Model
        
        # We simulate this chain
        mock_pyfunc_model = MagicMock()
        mock_wrapper = MagicMock()
        mock_wrapper.unwrap_python_model.return_value = "RealModel"
        
        mock_pyfunc_model.unwrap_python_model.return_value = mock_wrapper
        mock_load.return_value = mock_pyfunc_model
        
        model = self.registry.load_model("run123")
        self.assertEqual(model, "RealModel")
        mock_load.assert_called_with("runs:/run123/model")

    def test_save_model_deprecated(self):
        # Just ensure it doesn't crash on dummy call
        self.registry.save_model(None, "test")
