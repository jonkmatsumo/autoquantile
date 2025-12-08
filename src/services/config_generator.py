import pandas as pd
import re
from typing import Dict, Any, List, Optional
from src.utils.logger import get_logger
from src.services.llm_service import LLMService

class ConfigGenerator:
    """Service to generate configuration from data."""
    
    def __init__(self):
        self.logger = get_logger(__name__)

    def infer_levels(self, df: pd.DataFrame, level_col: str = "Level") -> Dict[str, int]:
        """Infers level ranking based on heuristics.
        
        Logic:
        1. Extract first integer found in string (e.g. "E5" -> 5, "L3" -> 3).
        2. Sort by integer, then by string.
        3. If no integer, sort alphabetically.
        """
        if level_col not in df.columns:
            return {}
            
        unique_levels = df[level_col].dropna().unique().tolist()
        
        def extract_rank(val: str):
            # Find first integer
            match = re.search(r'\d+', str(val))
            if match:
                return int(match.group())
            return -1 # Default low rank for non-numeric levels
            
        # Sort tuple: (extracted_rank, string_val)
        # We assume higher number = higher rank usually
        sorted_levels = sorted(unique_levels, key=lambda x: (extract_rank(x), x))
        
        # Map to 0-indexed rank
        return {lvl: i for i, lvl in enumerate(sorted_levels)}

    def infer_locations(self, df: pd.DataFrame, loc_col: str = "Location") -> Dict[str, int]:
        """Infers locations from column. Default tier is 2."""
        if loc_col not in df.columns:
            return {}
            
        unique_locs = sorted(df[loc_col].dropna().unique().tolist())
        return {loc: 2 for loc in unique_locs} # Default tier 2

    CONFIG_TEMPLATE = {
        "mappings": {
            "levels": {},
            "location_targets": {}
        },
        "location_settings": {
            "max_distance_km": 50
        },
        "model": {
            "targets": [], 
            "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
            "sample_weight_k": 1.0,
            "features": [],
            "hyperparameters": {
                "training": {"objective": "reg:quantileerror", "tree_method": "hist", "verbosity": 0},
                "cv": {"num_boost_round": 100, "nfold": 5, "early_stopping_rounds": 10}
            }
        }
    }

    def generate_config_template(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generates a template config using heuristics."""
        levels = self.infer_levels(df)
        locations = self.infer_locations(df)
        
        config = self.CONFIG_TEMPLATE.copy()
        config["mappings"] = {
            "levels": levels,
            "location_targets": locations
        }
        return config

    def generate_config(self, df: pd.DataFrame, use_llm: bool = True, provider: str = "openai", preset: str = "none") -> Dict[str, Any]:
        """
        Generates configuration from dataframe.
        
        Args:
            df (pd.DataFrame): Input data.
            use_llm (bool): Whether to use LLM.
            provider (str): LLM provider.
            preset (str): LLM preset.
            
        Returns:
            Dict[str, Any]: Configuration dictionary.
        """
        if use_llm:
            try:
                llm_service = LLMService(provider=provider)
                config = llm_service.generate_config(df, preset=preset)
                return config
            except Exception as e:
                self.logger.error(f"LLM Config Generation failed: {e}")
                raise e # Propagate error to let UI handle it
        else:
            return self.generate_config_template(df)
