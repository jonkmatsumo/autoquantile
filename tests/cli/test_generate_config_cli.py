import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import pandas as pd
from src.cli.generate_config_cli import main

class TestGenerateConfigCLI(unittest.TestCase):
    @patch('src.cli.generate_config_cli.argparse.ArgumentParser.parse_args')
    @patch('src.cli.generate_config_cli.pd.read_csv')
    @patch('src.cli.generate_config_cli.ConfigGenerator')
    @patch('src.cli.generate_config_cli.os.path.exists')
    @patch('builtins.print')
    def test_heuristic_generation(self, mock_print, mock_exists, MockGenerator, mock_read_csv, mock_ears):
        # Setup
        mock_exists.return_value = True
        mock_ears.return_value = MagicMock(input_file="data.csv", output=None, llm=False, verbose=False)
        mock_read_csv.return_value = pd.DataFrame({"A": [1]})
        
        mock_gen_instance = MockGenerator.return_value
        mock_gen_instance.generate_config_template.return_value = {"mapping": "heuristic"}
        
        # Run
        main()
        
        # Verify
        mock_gen_instance.generate_config_template.assert_called_once()
        mock_print.assert_called() # Should print JSON to stdout
        args = mock_print.call_args[0][0]
        self.assertIn("heuristic", args)

    @patch('src.cli.generate_config_cli.argparse.ArgumentParser')
    @patch('src.cli.generate_config_cli.pd.read_csv')
    @patch('src.cli.generate_config_cli.ConfigGenerator')
    @patch('src.cli.generate_config_cli.os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_llm_generation_file_output(self, mock_file, mock_exists, MockGenerator, mock_read_csv, MockParser):
        # Setup
        mock_exists.return_value = True
        
        # Setup Parser Mock
        mock_parser_instance = MockParser.return_value
        mock_parser_instance.parse_args.return_value = MagicMock(input_file="data.csv", output="out.json", llm=True, provider="openai", verbose=True)
        
        mock_gen_instance = MockGenerator.return_value
        mock_gen_instance.generate_config_with_llm.return_value = {"mapping": "llm"}
        
        # Run
        main()
        
        # Verify
        mock_gen_instance.generate_config_with_llm.assert_called_once()
        mock_file.assert_called_with("out.json", "w")
        # json.dump calls write multiple times, just check called
        self.assertTrue(mock_file.return_value.write.called)

    @patch('src.cli.generate_config_cli.argparse.ArgumentParser.parse_args')
    @patch('src.cli.generate_config_cli.os.path.exists')
    def test_file_not_found(self, mock_exists, mock_ears):
        mock_exists.return_value = False
        mock_ears.return_value = MagicMock(input_file="missing.csv", verbose=False)
        
        with self.assertRaises(SystemExit) as cm:
            main()
        self.assertEqual(cm.exception.code, 1)
