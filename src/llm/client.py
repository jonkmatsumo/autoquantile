"""
LLM Client module.

Provides both legacy LLM clients and LangChain-compatible wrappers
for use with the agentic workflow.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import os
import google.generativeai as genai
from openai import OpenAI
from src.utils.env_loader import get_env_var
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Legacy LLM Clients (for backward compatibility)
# =============================================================================

class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generates text from the LLM."""
        pass


class OpenAIClient(LLMClient):
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo-preview"):
        self.api_key = api_key or get_env_var("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found.")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise


class GeminiClient(LLMClient):
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-pro"):
        self.api_key = api_key or get_env_var("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        # Gemini handling of system prompts varies, effectively prepending is often easiest for basic usage
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\nUser: {prompt}"
            
        try:
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise


class DebugClient(LLMClient):
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        logger.info("DebugClient: Generating mock response.")
        return "MOCK_RESPONSE"


def get_llm_client(provider: str = "openai") -> LLMClient:
    """
    Get a legacy LLM client instance.
    
    Args:
        provider: Provider name ("openai", "gemini", or "debug").
        
    Returns:
        LLMClient instance.
    """
    if provider.lower() == "openai":
        return OpenAIClient()
    elif provider.lower() == "gemini":
        return GeminiClient()
    elif provider.lower() == "debug":
        return DebugClient()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


# =============================================================================
# LangChain-Compatible LLM Wrappers (for agentic workflow)
# =============================================================================

def get_langchain_llm(
    provider: str = "openai",
    model: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs
):
    """
    Get a LangChain-compatible LLM instance for use with agents.
    
    Args:
        provider: Provider name ("openai" or "gemini").
        model: Optional model name override.
        temperature: Temperature for generation (default 0.0 for determinism).
        **kwargs: Additional arguments passed to the LLM constructor.
        
    Returns:
        LangChain BaseChatModel instance.
        
    Raises:
        ValueError: If provider is not supported.
        ImportError: If required LangChain package is not installed.
    """
    provider_lower = provider.lower()
    
    if provider_lower == "openai":
        return _get_langchain_openai(model, temperature, **kwargs)
    elif provider_lower == "gemini":
        return _get_langchain_gemini(model, temperature, **kwargs)
    else:
        raise ValueError(f"Unknown LangChain provider: {provider}. Supported: openai, gemini")


def _get_langchain_openai(
    model: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs
):
    """
    Get a LangChain ChatOpenAI instance.
    
    Args:
        model: Model name (default: gpt-4-turbo-preview).
        temperature: Temperature for generation.
        **kwargs: Additional arguments.
        
    Returns:
        ChatOpenAI instance.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai is required for OpenAI LangChain support. "
            "Install with: pip install langchain-openai"
        )
    
    api_key = get_env_var("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment.")
    
    model_name = model or "gpt-4-turbo-preview"
    
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        **kwargs
    )


def _get_langchain_gemini(
    model: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs
):
    """
    Get a LangChain ChatGoogleGenerativeAI instance.
    
    Args:
        model: Model name (default: gemini-1.5-pro).
        temperature: Temperature for generation.
        **kwargs: Additional arguments.
        
    Returns:
        ChatGoogleGenerativeAI instance.
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ImportError(
            "langchain-google-genai is required for Gemini LangChain support. "
            "Install with: pip install langchain-google-genai"
        )
    
    api_key = get_env_var("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment.")
    
    model_name = model or "gemini-1.5-pro"
    
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key,
        **kwargs
    )


def get_available_providers() -> list:
    """
    Get list of available LLM providers based on installed packages and API keys.
    
    Returns:
        List of available provider names.
    """
    available = []
    
    # Check OpenAI
    try:
        from langchain_openai import ChatOpenAI
        if get_env_var("OPENAI_API_KEY"):
            available.append("openai")
    except ImportError:
        pass
    
    # Check Gemini
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        if get_env_var("GEMINI_API_KEY"):
            available.append("gemini")
    except ImportError:
        pass
    
    return available


def validate_provider(provider: str) -> bool:
    """
    Check if a provider is available and properly configured.
    
    Args:
        provider: Provider name to check.
        
    Returns:
        True if provider is available, False otherwise.
    """
    try:
        llm = get_langchain_llm(provider)
        return llm is not None
    except (ValueError, ImportError):
        return False
