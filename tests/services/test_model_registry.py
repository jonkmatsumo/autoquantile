import os
import unittest
import pickle
from unittest.mock import patch, MagicMock
from src.services.model_registry import ModelRegistry
from src.model.model import SalaryForecaster

class TestModelRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = ModelRegistry(".")

    @patch("glob.glob")
    def test_list_models(self, mock_glob):
        mock_glob.return_value = ["model1.pkl", "model2.pkl"]
        models = self.registry.list_models()
        self.assertEqual(models, ["model1.pkl", "model2.pkl"])
        mock_glob.assert_called_with("./*.pkl")

    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    @patch("pickle.load")
    @patch("os.path.exists")
    def test_load_model(self, mock_exists, mock_pickle_load, mock_open):
        mock_exists.return_value = True
        mock_model = MagicMock(spec=SalaryForecaster)
        mock_pickle_load.return_value = mock_model
        
        loaded = self.registry.load_model("test.pkl")
        self.assertEqual(loaded, mock_model)
        mock_open.assert_called_with("test.pkl", "rb")

    @patch("os.path.exists")
    def test_load_model_not_found(self, mock_exists):
        mock_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            self.registry.load_model("missing.pkl")

    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    @patch("pickle.dump")
    def test_save_model(self, mock_pickle_dump, mock_open):
        mock_model = MagicMock(spec=SalaryForecaster)
        path = self.registry.save_model(mock_model, "output")
        
        # Check it appended .pkl
        self.assertTrue(path.endswith(".pkl"))
        mock_open.assert_called_with("./output.pkl", "wb")
        mock_pickle_dump.assert_called_with(mock_model, mock_open())
