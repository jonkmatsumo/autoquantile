import pandas as pd
import numpy as np
from datetime import datetime
from .config_loader import get_config

class LevelEncoder:
    """
    Maps company levels to ordinal integers based on config.
    """
    def __init__(self):
        config = get_config()
        self.mapping = config["mappings"]["levels"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X is expected to be a Series or list of level strings
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        return X.map(self.mapping).fillna(-1).astype(int)

class LocationEncoder:
    """
    Maps locations to Cost Zones based on config.
    """
    def __init__(self):
        config = get_config()
        self.zone_mapping = config["mappings"]["locations"]
        # Zone 4 is default

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        
        # Helper to map a single value
        def map_loc(loc):
            if not isinstance(loc, str):
                return 4
            for key, zone in self.zone_mapping.items():
                if key.lower() in loc.lower():
                    return zone
            return 4 # Default
            
        return X.apply(map_loc)

class SampleWeighter:
    """
    Calculates sample weights based on recency.
    Weight = 1 / (1 + Age_in_Years)^k
    """
    def __init__(self, k=None, ref_date=None):
        if k is None:
            config = get_config()
            self.k = config["model"].get("sample_weight_k", 1.0)
        else:
            self.k = k
            
        self.ref_date = pd.to_datetime(ref_date) if ref_date else datetime.now()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X is expected to be a Series of dates
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        
        X = pd.to_datetime(X)
        
        # Calculate age in years
        age_days = (self.ref_date - X).dt.days
        age_years = age_days / 365.25
        
        # Clip negative age (future dates) to 0
        age_years = age_years.clip(lower=0)
        
        weights = 1 / (1 + age_years) ** self.k
        return weights
