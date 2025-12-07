import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.cli.inference_cli import collect_user_data, main

def test_collect_user_data():
    # Mock input to return: E5, New York, 5, 2
    inputs = ["E5", "New York", "5", "2"]
    with patch('builtins.input', side_effect=inputs):
        df = collect_user_data()
        
        assert len(df) == 1
        assert df.iloc[0]["Level"] == "E5"
        assert df.iloc[0]["Location"] == "New York"
        assert df.iloc[0]["YearsOfExperience"] == 5
        assert df.iloc[0]["YearsAtCompany"] == 2

def test_collect_user_data_invalid_level():
    # Mock input: Invalid, then E5, New York, 5, 2
    with patch('builtins.input', side_effect=["Invalid", "E5", "New York", "5", "2"]):
        df = collect_user_data()
        assert df.iloc[0]["Level"] == "E5"

def test_main_flow():
    # Mock load_model to return a mock model
    mock_model = MagicMock()
    # Mock quantiles: 10th and 90th percentile
    mock_model.quantiles = [0.1, 0.9]
    mock_model.predict.return_value = {
        "BaseSalary": {"p10": [100000], "p90": [140000]}
    }
    
    # Mock inputs: 
    # 1. E5 (Level)
    # 2. New York (Location)
    # 3. 5 (YOE)
    # 4. 2 (YAC)
    # 5. n (Stop)
    inputs = ["E5", "New York", "5", "2", "n"]
    
    with patch('src.cli.inference_cli.load_model', return_value=mock_model), \
         patch('builtins.input', side_effect=inputs), \
         patch('src.cli.inference_cli.Console') as MockConsole, \
         patch('src.cli.inference_cli.select_model', return_value="mock_model.pkl"), \
         patch('src.cli.inference_cli.plt') as mock_plt, \
         patch('sys.argv', ['src/cli/inference_cli.py']), \
         patch('src.cli.inference_cli.logger') as mock_logger:
        
        mock_console_instance = MockConsole.return_value
        
        main()
        
        # Verify predict was called
        mock_model.predict.assert_called_once()
        
        # Verify logger was used for status
        assert mock_logger.info.called
        assert mock_logger.info.call_count >= 3 # Loading, Calculating, Visualizing
        
        # Verify console was used for Welcome and Table
        assert mock_console_instance.print.call_count >= 2
        
        # Verify that a Table object was printed
        # One of the calls to print should have a Table argument
        # Since we mock Console, we can't easily check isinstance(arg, Table) unless we import Table here too.
        # However, checking if "Table" is in the type name of the argument is robust enough for a mock.
        
        table_printed = False
        for call in mock_console_instance.print.call_args_list:
            args, _ = call
            if len(args) > 0:
                arg = args[0]
                if "Table" in str(type(arg)) or "Table" in str(arg):
                     table_printed = True
                     break
                     
        assert table_printed, "A rich Table should have been printed"
