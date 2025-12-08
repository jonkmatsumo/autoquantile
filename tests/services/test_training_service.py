import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from src.services.training_service import TrainingService
from src.model.model import SalaryForecaster

class TestTrainingService(unittest.TestCase):
    def setUp(self):
        self.service = TrainingService()
        self.df = pd.DataFrame({"col": [1, 2, 3]})

    @patch("src.services.training_service.SalaryForecaster")
    def test_train_model(self, MockForecaster):
        # Setup mock instance
        mock_instance = MockForecaster.return_value
        
        callback = MagicMock()
        
        model = self.service.train_model(self.df, remove_outliers=True, callback=callback)
        
        # Verify Forecaster was instantiated
        MockForecaster.assert_called_once()
        
        # Verify train was called
        mock_instance.train.assert_called_with(self.df, callback=callback, remove_outliers=True)
        
        # Verify callback initial call
        callback.assert_any_call("Starting training...", None)
        
        self.assertEqual(model, mock_instance)

    @patch("src.services.training_service.SalaryForecaster")
    def test_start_training_async(self, MockForecaster):
        job_id = self.service.start_training_async(self.df)
        self.assertIsNotNone(job_id)
        
        status = self.service.get_job_status(job_id)
        self.assertIsNotNone(status)
        self.assertIn(status["status"], ["QUEUED", "RUNNING", "COMPLETED"])
        self.assertIsInstance(status["history"], list)

    def test_get_job_status_invalid(self):
        status = self.service.get_job_status("invalid_id")
        self.assertNone(status)
