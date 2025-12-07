import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import json
import io
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

def test_main_interactive():
    mock_model = MagicMock()
    mock_model.quantiles = [0.5]
    mock_model.predict.return_value = {"BaseSalary": {"p50": [120000]}}
    
    # Simulate: No args (default interactive), input E5, NY, 5, 2, Stop
    with patch('src.cli.inference_cli.load_model', return_value=mock_model), \
         patch('builtins.input', side_effect=["E5", "New York", "5", "2", "n"]), \
         patch('src.cli.inference_cli.Console') as MockConsole, \
         patch('src.cli.inference_cli.select_model', return_value="model.pkl"), \
         patch('sys.argv', ['cli']), \
         patch('src.cli.inference_cli.plt'): # Mock plotting
         
         main()
         
         mock_model.predict.assert_called()

def test_main_non_interactive_json():
    mock_model = MagicMock()
    mock_model.quantiles = [0.5]
    mock_model.predict.return_value = {"BaseSalary": {"p50": [120000]}}
    
    # Simulate: Flags provided, --json
    args = [
        'cli', 
        '--model', 'model.pkl',
        '--level', 'E5', 
        '--location', 'New York',
        '--yoe', '5', 
        '--yac', '2',
        '--json'
    ]
    
    with patch('src.cli.inference_cli.load_model', return_value=mock_model), \
         patch('sys.argv', args), \
         patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
         
         main()
         
         output = mock_stdout.getvalue()
         data = json.loads(output)
         
         assert data["BaseSalary"]["p50"] == 120000
         mock_model.predict.assert_called_once()
         # Verify correct input dataframe was passed
         call_arg = mock_model.predict.call_args[0][0]
         assert call_arg.iloc[0]["Level"] == "E5"

def test_main_partial_args_error():
    # Missing --yac
    args = ['cli', '--level', 'E5', '--location', 'NY', '--yoe', '5']
    
    with patch('sys.argv', args), \
         patch('src.cli.inference_cli.Console') as MockConsole, \
         patch('sys.exit', side_effect=SystemExit) as mock_exit:
         
         with pytest.raises(SystemExit):
             main()
         
         mock_exit.assert_called_with(1)
         # Should print error message
         MockConsole.return_value.print.assert_called()
