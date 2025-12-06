import pytest
from unittest.mock import patch, MagicMock
from src.cli.train_cli import main

def test_train_cli_main():
    # Mock inputs: 
    # 1. input.csv (CSV)
    # 2. config.json (Config)
    # 3. model.pkl (Output)
    inputs = ["input.csv", "config.json", "model.pkl"]
    
    with patch('src.cli.train_cli.Console') as MockConsole, \
         patch('os.path.exists', return_value=True), \
         patch('src.cli.train_cli.train_model') as mock_train_model:
        
        mock_console = MockConsole.return_value
        mock_console.input.side_effect = inputs
        
        main()
        
        # Verify train_model was called with correct args
        mock_train_model.assert_called_once_with("input.csv", "config.json", "model.pkl")

def test_train_cli_defaults():
    # Mock inputs: Enter (default), Enter (default), Enter (default)
    inputs = ["", "", ""]
    
    with patch('src.cli.train_cli.Console') as MockConsole:
        mock_console = MockConsole.return_value
        # Mock input method of console
        mock_console.input.side_effect = inputs
        
        with patch('os.path.exists', return_value=True), \
             patch('src.cli.train_cli.train_model') as mock_train_model:
            
            main()
            
            # Verify defaults
            mock_train_model.assert_called_once_with("salaries-list.csv", "config.json", "salary_model.pkl")

def test_train_cli_file_not_found():
    # Mock inputs: missing.csv, ...
    inputs = ["missing.csv", "config.json", "model.pkl"]
    
    with patch('src.cli.train_cli.Console') as MockConsole:
        mock_console = MockConsole.return_value
        mock_console.input.side_effect = inputs
        
        # Mock os.path.exists to return False for the first file
        with patch('os.path.exists', side_effect=[False, True]): 
            
            main()
            
            # Verify error message printed
            # Check if any print call contains "Error"
            error_printed = any("Error" in str(call) for call in mock_console.print.call_args_list)
            assert error_printed
