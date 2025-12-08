import pytest
from unittest.mock import MagicMock, patch
from src.llm.client import OpenAIClient, GeminiClient, DebugClient, get_llm_client

@patch("src.llm.client.OpenAI")
@patch("src.llm.client.get_env_var")
def test_openai_client(mock_get_env, mock_openai):
    mock_get_env.return_value = "fake-key"
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = "Response"
    mock_openai.return_value.chat.completions.create.return_value = mock_completion
    
    client = OpenAIClient()
    response = client.generate("Hello")
    assert response == "Response"
    mock_openai.return_value.chat.completions.create.assert_called_once()

@patch("src.llm.client.genai")
@patch("src.llm.client.get_env_var")
def test_gemini_client(mock_get_env, mock_genai):
    mock_get_env.return_value = "fake-key"
    mock_model = MagicMock()
    mock_model.generate_content.return_value.text = "Response"
    mock_genai.GenerativeModel.return_value = mock_model
    
    client = GeminiClient()
    response = client.generate("Hello")
    assert response == "Response"
    mock_model.generate_content.assert_called_once()

def test_debug_client():
    client = DebugClient()
    assert client.generate("test") == "MOCK_RESPONSE"

def test_get_llm_client():
    assert isinstance(get_llm_client("debug"), DebugClient)
    
    with patch("src.llm.client.get_env_var", return_value="key"):
        with patch("src.llm.client.OpenAI"):
             assert isinstance(get_llm_client("openai"), OpenAIClient)
