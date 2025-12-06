import pytest
from unittest.mock import patch, MagicMock
from src.model.train import train_model

def test_train_model():
    # Mock dependencies
    with patch('src.model.train.load_data') as mock_load_data, \
         patch('src.model.train.SalaryForecaster') as MockForecaster, \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', new_callable=MagicMock), \
         patch('pickle.dump') as mock_pickle_dump: # Mock pickle dump
        
        # Setup mocks
        mock_df = MagicMock()
        mock_df.__len__.return_value = 100 # Simulate loaded data
        mock_load_data.return_value = mock_df
        
        mock_model = MockForecaster.return_value
        
        # Run train_model
        train_model(csv_path="test.csv", config_path="test_config.json", output_path="test_model.pkl")
        
        # Verify interactions
        mock_load_data.assert_called_once_with("test.csv")
        MockForecaster.assert_called_once()
        mock_model.train.assert_called_once_with(mock_df)
        
        # Verify pickle.dump was called
        mock_pickle_dump.assert_called_once()
        
        # Verify inference was attempted (predict called)
        assert mock_model.predict.call_count >= 1

def test_train_model_no_data():
    # Test case where CSV doesn't exist
    with patch('os.path.exists', return_value=False), \
         patch('builtins.print') as mock_print:
        
        train_model(csv_path="missing.csv")
        
        # Verify error message
        assert any("not found" in str(call) for call in mock_print.call_args_list)
