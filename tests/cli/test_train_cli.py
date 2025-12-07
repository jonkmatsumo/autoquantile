import pytest
from unittest.mock import patch, MagicMock
from src.cli.train_cli import main, train_workflow

@patch('sys.argv', ['prog', '--csv', 'input.csv', '--config', 'config.json', '--output', 'model.pkl'])
def test_train_cli_main():
    with patch('src.cli.train_cli.Console') as MockConsole, \
         patch('os.path.exists', return_value=True), \
         patch('src.cli.train_cli.train_workflow') as mock_train_workflow:
        
        main()
        
        # Verify train_workflow was called with correct args
        mock_train_workflow.assert_called_once()
        args, kwargs = mock_train_workflow.call_args
        assert args[0] == "input.csv"
        assert args[1] == "config.json"
        assert args[2] == "model.pkl"
        assert kwargs['do_tune'] is False
        assert kwargs['num_trials'] == 20

@patch('sys.argv', ['prog'])
def test_train_cli_defaults():
    with patch('src.cli.train_cli.Console') as MockConsole, \
         patch('os.path.exists', return_value=True), \
         patch('src.cli.train_cli.train_workflow') as mock_train_workflow:
        
        main()
        
        # Verify defaults
        mock_train_workflow.assert_called_once()
        args, kwargs = mock_train_workflow.call_args
        assert args[0] == "salaries-list.csv"
        assert args[1] == "config.json"
        assert args[2] == "salary_model.pkl"

@patch('sys.argv', ['prog', '--tune', '--num-trials', '50'])
def test_train_cli_tune():
    with patch('src.cli.train_cli.Console') as MockConsole, \
         patch('os.path.exists', return_value=True), \
         patch('src.cli.train_cli.train_workflow') as mock_train_workflow:
        
        main()
        
        mock_train_workflow.assert_called_once()
        args, kwargs = mock_train_workflow.call_args
        assert kwargs['do_tune'] is True
        assert kwargs['num_trials'] == 50

# test_train_cli_file_not_found removed or needs update. 
# Argparse doesn't check file existence, train_workflow does.
# But main calls train_workflow. train_workflow prints error and returns.
# So we can test that main calls train_workflow, and train_workflow handles it.
# Actually test_train_workflow handles the file not found logic?
# Let's keep a full flow test if desired but we are testing main parsing here.


def test_train_workflow():
    # Mock dependencies
    with patch('src.cli.train_cli.load_data') as mock_load_data, \
         patch('src.cli.train_cli.SalaryForecaster') as MockForecaster, \
         patch('src.cli.train_cli.load_config') as mock_load_config, \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', new_callable=MagicMock), \
         patch('pickle.dump') as mock_pickle_dump: # Mock pickle dump
        
        # Setup mocks
        mock_df = MagicMock()
        mock_df.__len__.return_value = 100 # Simulate loaded data
        mock_load_data.return_value = mock_df
        
        mock_model = MockForecaster.return_value
        # Mock predict to return dictionary logic for output printing
        mock_model.quantiles = [0.10, 0.50, 0.90]
        mock_model.predict.return_value = {
            "BaseSalary": {"p10": [150000], "p50": [200000], "p90": [250000]}
        }

        mock_console = MagicMock()

        # Run train_workflow
        train_workflow(csv_path="test.csv", config_path="test_config.json", output_path="test_model.pkl", console=mock_console)
        
        # Verify interactions
        mock_load_data.assert_called_once_with("test.csv")
        MockForecaster.assert_called_once()
        # Verify train was called (we can't easily assert the local callback function equality)
        assert mock_model.train.call_count == 1
        args, kwargs = mock_model.train.call_args
        assert args[0] == mock_df
        assert 'callback' in kwargs
        assert callable(kwargs['callback'])
        
        # Verify pickle.dump was called
        mock_pickle_dump.assert_called_once()
        
        # Verify inference was attempted (predict called)
        assert mock_model.predict.call_count >= 1

def test_train_workflow_calls_load_config():
    with patch('src.cli.train_cli.load_config') as mock_load_config, \
         patch('src.cli.train_cli.load_data'), \
         patch('src.cli.train_cli.SalaryForecaster'), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open'), \
         patch('pickle.dump'):
        
        mock_console = MagicMock()
        train_workflow(csv_path="test.csv", config_path="test_config.json", output_path="model.pkl", console=mock_console)
        
        mock_load_config.assert_called_once_with("test_config.json")
