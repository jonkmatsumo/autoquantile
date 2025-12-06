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
         patch('src.cli.inference_cli.plt') as mock_plt: # Mock plotext
        
        mock_console_instance = MockConsole.return_value
        
        main()
        
        # Verify predict was called
        mock_model.predict.assert_called_once()
        
        # Verify output contains expected strings (checking args passed to print)
        # We can just check if "BaseSalary" was printed to the console
        # Rich console.print is called with a Table object, so checking string content is harder directly.
        # But we can check if table was added.
        assert mock_console_instance.print.call_count >= 5
        
        # Verify that a Table object was printed
        # One of the calls to print should have a Table argument
        table_printed = False
        for call in mock_console_instance.print.call_args_list:
            args, _ = call
            if len(args) > 0 and "Table" in str(type(args[0])):
                table_printed = True
                # Check if columns were added dynamically?
                # It's hard to inspect the Table object deeply without importing rich, 
                # but we can assume if it didn't crash and printed a table, logic is likely ok.
                # Ideally we'd inspect table.columns but that requires the real object.
                break
        
        # Since we mock Console, we can't easily check isinstance(arg, Table) unless we import Table here too.
        from rich.table import Table
        table_printed = any(isinstance(call.args[0], Table) for call in mock_console_instance.print.call_args_list)
        assert table_printed, "A rich Table should have been printed"
