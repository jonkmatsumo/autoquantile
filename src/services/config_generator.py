import pandas as pd
import re
from typing import Dict, Any, List, Optional
from src.utils.logger import get_logger

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

    def generate_config_with_llm(self, df: pd.DataFrame, provider: str = "openai") -> Dict[str, Any]:
        """Generates config using LLM, merging with heuristics for robustness."""
        from src.services.llm_service import LLMService
        
        # 1. Heuristic Baseline (ensures we have all levels/locs)
        base_config = self.generate_config_template(df)
        
        try:
            llm_service = LLMService(provider=provider)
            llm_config = llm_service.generate_config(df)
            
            # 2. Merge Strategies
            self.logger.info("Merging LLM config with baseline...")
            
            # A. Levels: Use LLM ranks for known levels, keep heuristic for others
            if "mappings" in llm_config and "levels" in llm_config["mappings"]:
                llm_levels = llm_config["mappings"]["levels"]
                for lvl, rank in llm_levels.items():
                    # Only update if level exists in data (fuzzy match could be added later)
                    if lvl in base_config["mappings"]["levels"]:
                        base_config["mappings"]["levels"][lvl] = rank
                        
            # B. Locations: Use LLM tiers
            if "mappings" in llm_config and "location_targets" in llm_config["mappings"]:
                llm_locs = llm_config["mappings"]["location_targets"]
                for loc, tier in llm_locs.items():
                    if loc in base_config["mappings"]["location_targets"]:
                        base_config["mappings"]["location_targets"][loc] = tier
            
            # C. Model: Targets and Features
            # We trust LLM identification of targets/features over nothing
            if "model" in llm_config:
                if "targets" in llm_config["model"]:
                    # Validate columns exist
                    valid_targets = [t for t in llm_config["model"]["targets"] if t in df.columns]
                    if valid_targets:
                        base_config["model"]["targets"] = valid_targets
                        
                if "features" in llm_config["model"]:
                    valid_features = []
                    for feat in llm_config["model"]["features"]:
                        if feat["name"] in df.columns or feat["name"] in ["Level_Enc", "Location_Enc"]:
                            valid_features.append(feat)
                    if valid_features:
                        base_config["model"]["features"] = valid_features

            return base_config
            
        except Exception as e:
            self.logger.error(f"LLM Config Generation failed: {e}")
            self.logger.info("Falling back to heuristic config.")
            return base_config
