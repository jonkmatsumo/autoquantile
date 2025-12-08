import json
import pandas as pd
from typing import Dict, Any, Optional
from src.llm.client import get_llm_client, LLMClient
from src.utils.prompt_loader import load_prompt
from src.utils.logger import get_logger
from src.model.config_schema_model import Config

class LLMService:
    def __init__(self, provider: str = "openai"):
        self.logger = get_logger(__name__)
        try:
            self.client = get_llm_client(provider)
        except Exception as e:
            self.logger.warning(f"Failed to initialize LLM client for provider '{provider}': {e}")
            self.client = None

    def generate_config(self, df: pd.DataFrame, preset: str = "none") -> Dict[str, Any]:
        """
        Generates configuration from dataframe using LLM, optionally applying a preset.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            preset (str): Preset name (e.g. 'salary', 'none'). Defaults to 'none'.
            
        Returns:
            Dict[str, Any]: Validated configuration dictionary.
        """
        if not self.client:
            raise RuntimeError("LLM Client not initialized. Check API keys.")


        sample = df.head(50).to_csv(index=False)
        dtypes = df.dtypes.to_string()
        
        user_prompt_template = load_prompt("config_generation_user")
        system_prompt = load_prompt("config_generation_system")
        

        preset_content = ""
        if preset and preset.lower() != "none":
            try:
                preset_content = load_prompt(f"presets/{preset}")
            except Exception as e:
                self.logger.warning(f"Failed to load preset '{preset}': {e}")
        

        if preset_content:
            system_prompt += f"\n\n{preset_content}"
        
        prompt = user_prompt_template.format(data_sample=sample, dtypes=dtypes)
        
        self.logger.info(f"Sending request to LLM (Preset: {preset})...")
        response_text = self.client.generate(prompt, system_prompt=system_prompt)
        
        try:

            cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
            config_dict = json.loads(cleaned_text)
            

            validated_config = Config.model_validate(config_dict)
            self.logger.info("Successfully generated and validated config from LLM.")
            return validated_config.model_dump()
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response JSON: {response_text}")
            raise ValueError("LLM did not return valid JSON.") from e
        except Exception as e:
            self.logger.error(f"Config validation failed: {e}")
            raise ValueError(f"Generated config failed validation: {e}")
