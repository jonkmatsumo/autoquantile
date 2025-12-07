import pandas as pd
import numpy as np
from datetime import datetime
from src.utils.config_loader import get_config

class LevelEncoder:
    """
    Maps company levels to ordinal integers based on config.

    Attributes:
        mapping (dict): Dictionary mapping level names (e.g., 'E3') to integer ranks (e.g., 0).
    """
    def __init__(self):
        """Initializes the encoder by loading level mappings from configuration."""
        config = get_config()
        self.mapping = config["mappings"]["levels"]

    def fit(self, X, y=None):
        """
        Fits the encoder (no-op as mapping is static from config).

        Args:
            X: Input data.
            y: Target data (optional).

        Returns:
            self
        """
        return self

    def transform(self, X):
        """
        Transforms levels to their integer representation.

        Args:
            X (pd.DataFrame or pd.Series): The input data containing 'Level' column or series of levels.

        Returns:
            pd.Series: Integer encoded levels. Unknown levels are mapped to -1.
        """
        # X is expected to be a Series or list of level strings
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        return X.map(self.mapping).fillna(-1).astype(int)

from src.utils.geo_utils import GeoMapper

class LocationEncoder:
    """
    Maps locations to Cost Zones based on proximity to target cities.

    Attributes:
        mapper (GeoMapper): Utility to calculate proximity zones.
    """
    def __init__(self):
        """Initializes the encoder with a GeoMapper instance."""
        self.mapper = GeoMapper()

    def fit(self, X, y=None):
        """
        Fits the encoder (no-op).

        Args:
            X: Input data.
            y: Target data (optional).

        Returns:
            self
        """
        return self

    def transform(self, X):
        """
        Transforms location names to their cost zone integers.

        Args:
            X (pd.DataFrame or pd.Series): Input containing location names.

        Returns:
            pd.Series: Cost zones (1, 2, 3, or 4 for unknown).
        """
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        
        # Helper to map a single value
        def map_loc(loc):
            if not isinstance(loc, str):
                return 4
            return self.mapper.get_zone(loc)
            
        return X.apply(map_loc)

class SampleWeighter:
    """
    Calculates sample weights based on recency.
    Weight = 1 / (1 + Age_in_Years)^k

    Attributes:
        k (float): Decay rate parameter.
        ref_date (datetime): Reference date to calculate age from.
    """
    def __init__(self, k=None, ref_date=None):
        """
        Initializes the weighter.

        Args:
            k (float, optional): Decay parameter. If None, loads from config.
            ref_date (str or datetime, optional): Reference date. Defaults to current time.
        """
        if k is None:
            config = get_config()
            self.k = config["model"].get("sample_weight_k", 1.0)
        else:
            self.k = k
            
        self.ref_date = pd.to_datetime(ref_date) if ref_date else datetime.now()

    def fit(self, X, y=None):
        """
        Fits the weighter (no-op).

        Args:
            X: Input data.
            y: Target data (optional).

        Returns:
            self
        """
        return self

    def transform(self, X):
        """
        Calculates weights for the input dates.

        Args:
            X (pd.DataFrame or pd.Series): Input dates.

        Returns:
            pd.Series or np.ndarray: Calculated weights.
        """
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
