import json
import pandas as pd
from typing import Dict, Any, Optional
from src.llm.client import get_llm_client, LLMClient
from src.utils.prompt_loader import load_prompt
from src.utils.logger import get_logger

class LLMService:
    def __init__(self, provider: str = "openai"):
        self.logger = get_logger(__name__)
        try:
            self.client = get_llm_client(provider)
        except Exception as e:
            self.logger.warning(f"Failed to initialize LLM client for provider '{provider}': {e}")
            self.client = None

    def generate_config(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generates configuration from dataframe using LLM."""
        if not self.client:
            raise RuntimeError("LLM Client not initialized. Check API keys.")

        # Prepare sample
        sample = df.head(50).to_csv(index=False)
        dtypes = df.dtypes.to_string()
        
        user_prompt_template = load_prompt("config_generation_user")
        system_prompt = load_prompt("config_generation_system")
        
        prompt = user_prompt_template.format(data_sample=sample, dtypes=dtypes)
        
        self.logger.info("Sending request to LLM...")
        response_text = self.client.generate(prompt, system_prompt=system_prompt)
        
        try:
            # Basic parsing, assumed markdown code block stripping
            cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
            config = json.loads(cleaned_text)
            self.logger.info("Successfully generated config from LLM.")
            return config
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response: {response_text}")
            raise ValueError("LLM did not return valid JSON.") from e
