import unittest
from unittest.mock import patch, mock_open
import os
from src.utils.prompt_loader import load_prompt

class TestPromptLoader(unittest.TestCase):
    @patch("src.utils.prompt_loader.os.path.exists")
    @patch("src.utils.prompt_loader.open", new_callable=mock_open, read_data="Mock Prompt Content")
    def test_load_prompt_success(self, mock_file, mock_exists):
        mock_exists.return_value = True
        
        content = load_prompt("test_prompt")
        
        self.assertEqual(content, "Mock Prompt Content")
        
        # Verify path logic implies calling exists on a path ending in test_prompt.md
        args, _ = mock_file.call_args
        self.assertTrue(args[0].endswith("test_prompt.md"))

    @patch("src.utils.prompt_loader.os.path.exists")
    def test_load_prompt_not_found(self, mock_exists):
        mock_exists.return_value = False
        
        with self.assertRaises(FileNotFoundError):
            load_prompt("missing_prompt")
