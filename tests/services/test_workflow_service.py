"""Tests for WorkflowService."""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from src.services.workflow_service import WorkflowService, get_workflow_providers


class TestWorkflowService(unittest.TestCase):
    """Test cases for WorkflowService."""
    
    @patch("src.services.workflow_service.get_langchain_llm")
    def test_init_success(self, mock_get_llm):
        """Test successful initialization."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        service = WorkflowService(provider="openai")
        
        self.assertIsNotNone(service.llm)
        mock_get_llm.assert_called_once_with(provider="openai", model=None)
    
    @patch("src.services.workflow_service.get_langchain_llm")
    def test_init_failure(self, mock_get_llm):
        """Test initialization failure."""
        mock_get_llm.side_effect = ValueError("Invalid API key")
        
        with self.assertRaises(ValueError):
            WorkflowService(provider="openai")
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.workflow_service.ConfigWorkflow")
    def test_start_workflow(self, mock_workflow_class, mock_get_llm):
        """Test starting a workflow."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        # Mock workflow
        mock_workflow = MagicMock()
        mock_workflow.start.return_value = {
            "column_classification": {
                "targets": ["Salary"],
                "features": ["Level", "Location"],
                "ignore": ["ID"],
                "reasoning": "Test reasoning"
            },
            "classification_confirmed": False,
            "current_phase": "classification"
        }
        mock_workflow.get_current_phase.return_value = "classification"
        mock_workflow_class.return_value = mock_workflow
        
        service = WorkflowService(provider="openai")
        
        df = pd.DataFrame({
            "ID": [1, 2],
            "Level": ["L3", "L4"],
            "Location": ["NY", "SF"],
            "Salary": [100000, 150000]
        })
        
        result = service.start_workflow(df)
        
        self.assertEqual(result["phase"], "classification")
        self.assertEqual(result["status"], "success")
        self.assertIn("data", result)
        self.assertEqual(result["data"]["targets"], ["Salary"])
    
    @patch("src.services.workflow_service.get_langchain_llm")
    def test_get_current_state_not_started(self, mock_get_llm):
        """Test getting state before workflow starts."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        service = WorkflowService(provider="openai")
        state = service.get_current_state()
        
        self.assertEqual(state["phase"], "not_started")
        self.assertEqual(state["status"], "pending")
    
    @patch("src.services.workflow_service.get_langchain_llm")
    def test_is_complete_false(self, mock_get_llm):
        """Test is_complete when workflow not started."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        service = WorkflowService(provider="openai")
        self.assertFalse(service.is_complete())
    
    @patch("src.services.workflow_service.get_langchain_llm")
    def test_get_final_config_none(self, mock_get_llm):
        """Test get_final_config when workflow not complete."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        service = WorkflowService(provider="openai")
        self.assertIsNone(service.get_final_config())


class TestGetWorkflowProviders(unittest.TestCase):
    """Test cases for get_workflow_providers function."""
    
    @patch("src.services.workflow_service.get_available_providers")
    def test_get_providers(self, mock_get_available):
        """Test getting available providers."""
        mock_get_available.return_value = ["openai", "gemini"]
        
        providers = get_workflow_providers()
        
        self.assertEqual(providers, ["openai", "gemini"])


if __name__ == "__main__":
    unittest.main()

