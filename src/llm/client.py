from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import os
import google.generativeai as genai
from openai import OpenAI
from src.utils.env_loader import get_env_var
from src.utils.logger import get_logger

logger = get_logger(__name__)

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
    if provider.lower() == "openai":
        return OpenAIClient()
    elif provider.lower() == "gemini":
        return GeminiClient()
    elif provider.lower() == "debug":
        return DebugClient()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
