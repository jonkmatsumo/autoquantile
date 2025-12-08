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

    @patch("src.services.model_registry.mlflow.pyfunc.load_model")
    def test_load_model(self, mock_load):
        mock_model_wrapper = MagicMock()
        mock_model_wrapper.unwrap_python_model.return_value = "RealModel"
        mock_load.return_value = mock_model_wrapper
        
        model = self.registry.load_model("run123")
        self.assertEqual(model, "RealModel")
        mock_load.assert_called_with("runs:/run123/model")

    def test_save_model_deprecated(self):
        # Just ensure it doesn't crash on dummy call
        self.registry.save_model(None, "test")
