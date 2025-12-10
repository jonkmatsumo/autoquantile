import pandas as pd
from datetime import datetime
from typing import Union, Optional, Any, Dict
from src.utils.config_loader import get_config

class RankedCategoryEncoder:
    """Maps ordinal categorical values to integers based on a provided mapping.

    Attributes:
        mapping (dict): Dictionary mapping category names to integer ranks.
    """
    def __init__(self, mapping: Optional[Dict[str, int]] = None, config_key: Optional[str] = None) -> None:
        """Initialize encoder.
        
        Args:
            mapping: Direct mapping dictionary.
            config_key: Key in config['mappings'] to load mapping from.
        """
        if mapping is not None:
            self.mapping = mapping
        elif config_key is not None:
            config = get_config()
            self.mapping = config.get("mappings", {}).get(config_key, {})
        else:
            self.mapping = {}


    def fit(self, X: Any, y: Optional[Any] = None) -> "RankedCategoryEncoder":
        """Fits the encoder (no-op as mapping is static).
        
        Args:
            X: Input data.
            y: Target data (optional).
            
        Returns:
            self
        """
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series, list]) -> pd.Series:
        """Transforms categories to their integer representation.
        
        Args:
            X: Input categories.
            
        Returns:
            Integer encoded categories. Unknown categories mapped to -1.
        """
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        return pd.Series(X).map(self.mapping).fillna(-1).astype(int)

from src.utils.geo_utils import GeoMapper

class ProximityEncoder:
    """Maps locations to Cost Zones based on proximity to target cities.

    Attributes:
        mapper (GeoMapper): Utility to calculate proximity zones.
    """
    def __init__(self) -> None:
        self.mapper = GeoMapper()


    def fit(self, X: Any, y: Optional[Any] = None) -> "ProximityEncoder":
        """Fits the encoder (no-op).
        
        Args:
            X: Input data.
            y: Target data (optional).
            
        Returns:
            self
        """
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Transforms location names to their cost zone integers.
        
        Args:
            X: Input locations.
            
        Returns:
            Cost zones (1-4).
        """
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        
        def map_loc(loc: Any) -> int:
            if not isinstance(loc, str):
                return 4
            return self.mapper.get_zone(loc)
            
        return X.apply(map_loc)

class SampleWeighter:
    """Calculates sample weights based on recency.
    
    Formula: Weight = 1 / (1 + Age_in_Years)^k

    Attributes:
        k (float): Decay rate parameter.
        ref_date (datetime): Reference date to calculate age from.
        date_col (str): Name of the date column to use if dataframe passed.
    """
    def __init__(self, k: Optional[float] = None, ref_date: Optional[Union[str, datetime]] = None, date_col: str = "Date") -> None:
        """Initialize the weighter.
        
        Args:
            k: Decay parameter. If None, loads from config.
            ref_date: Reference date. Defaults to now.
            date_col: Column name for date if dataframe is passed. Defaults to "Date".
        """
        if k is None:
            config = get_config()
            self.k = config["model"].get("sample_weight_k", 1.0)
        else:
            self.k = k
            
        self.ref_date = pd.to_datetime(ref_date) if ref_date else datetime.now()
        self.date_col = date_col

    def fit(self, X: Any, y: Optional[Any] = None) -> "SampleWeighter":
        """Fits the weighter (no-op).
        
        Args:
            X: Input data.
            y: Target data (optional).
            
        Returns:
            self
        """
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Calculates weights for the input dates.
        
        Args:
            X: Input dates or dataframe containing date_col.
            
        Returns:
            Calculated weights.
        """
        if isinstance(X, pd.DataFrame):
            if self.date_col in X.columns:
                X = X[self.date_col]
            else:
                # Date column doesn't exist - try first column as fallback
                # but only if it can be parsed as dates
                X = X.iloc[:, 0]
        
        # Try to parse as datetime, but handle errors gracefully
        try:
            X_parsed = pd.to_datetime(X, errors='coerce')
            # If all values are NaT (couldn't parse), return uniform weights
            if X_parsed.isna().all():
                return pd.Series(1.0, index=X.index if isinstance(X, pd.Series) else range(len(X)))
            X = X_parsed
        except Exception:
            # If parsing fails completely, return uniform weights
            return pd.Series(1.0, index=X.index if isinstance(X, pd.Series) else range(len(X)))
        
        age_days = (self.ref_date - X).dt.days
        age_years = age_days / 365.25
        
        age_years = age_years.clip(lower=0)
        
        weights = 1 / (1 + age_years) ** self.k
        return weights

class CostOfLivingEncoder:
    """Maps locations to cost of living tiers.
    
    Uses the same proximity-based approach as ProximityEncoder but
    specifically for cost of living classification.
    
    Attributes:
        mapper (GeoMapper): Utility to calculate proximity zones.
    """
    def __init__(self) -> None:
        self.mapper = GeoMapper()

    def fit(self, X: Any, y: Optional[Any] = None) -> "CostOfLivingEncoder":
        """Fits the encoder (no-op).
        
        Args:
            X: Input data.
            y: Target data (optional).
            
        Returns:
            self
        """
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Transforms location names to their cost of living tier integers.
        
        Args:
            X: Input locations.
            
        Returns:
            Cost of living tiers (1-4, where 1 is highest cost).
        """
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        
        def map_loc(loc: Any) -> int:
            if not isinstance(loc, str):
                return 4
            return self.mapper.get_zone(loc)
            
        return X.apply(map_loc)


class MetroPopulationEncoder:
    """Maps locations to metro area population values.
    
    This is a placeholder implementation. In a production system,
    this would use an external API or database to get actual population data.
    For now, it uses a simple heuristic based on proximity zones.
    
    Attributes:
        mapper (GeoMapper): Utility to calculate proximity zones.
        population_map (Dict[int, int]): Mapping from zone to approximate population.
    """
    def __init__(self) -> None:
        self.mapper = GeoMapper()
        # Approximate population mapping based on cost zones
        # Zone 1 (highest cost) typically has larger populations
        self.population_map = {
            1: 5000000,  # Major metro areas
            2: 2000000,  # Large cities
            3: 500000,   # Medium cities
            4: 100000    # Small cities/rural
        }

    def fit(self, X: Any, y: Optional[Any] = None) -> "MetroPopulationEncoder":
        """Fits the encoder (no-op).
        
        Args:
            X: Input data.
            y: Target data (optional).
            
        Returns:
            self
        """
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Transforms location names to approximate population values.
        
        Args:
            X: Input locations.
            
        Returns:
            Approximate population values.
        """
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        
        def map_loc(loc: Any) -> int:
            if not isinstance(loc, str):
                return self.population_map[4]
            zone = self.mapper.get_zone(loc)
            return self.population_map.get(zone, self.population_map[4])
            
        return X.apply(map_loc)


class DateNormalizer:
    """Normalizes dates to 0-1 range based on min/max dates.
    
    Supports two modes:
    - normalize_recent: Most recent date = 1.0, least recent = 0.0
    - least_recent: Least recent date = 0.0, most recent = 1.0
    
    Attributes:
        mode (str): Normalization mode ('normalize_recent' or 'least_recent').
        min_date (Optional[pd.Timestamp]): Minimum date (set during fit).
        max_date (Optional[pd.Timestamp]): Maximum date (set during fit).
    """
    def __init__(self, mode: str = "normalize_recent") -> None:
        """Initialize the normalizer.
        
        Args:
            mode: Normalization mode ('normalize_recent' or 'least_recent').
        """
        self.mode = mode
        self.min_date: Optional[pd.Timestamp] = None
        self.max_date: Optional[pd.Timestamp] = None

    def fit(self, X: Any, y: Optional[Any] = None) -> "DateNormalizer":
        """Fits the normalizer by computing min/max dates.
        
        Args:
            X: Input dates.
            y: Target data (optional, ignored).
            
        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        
        X = pd.to_datetime(X, errors='coerce')
        valid_dates = X.dropna()
        
        if len(valid_dates) > 0:
            self.min_date = valid_dates.min()
            self.max_date = valid_dates.max()
        else:
            self.min_date = pd.Timestamp.now()
            self.max_date = pd.Timestamp.now()
        
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Normalizes dates to 0-1 range.
        
        Args:
            X: Input dates.
            
        Returns:
            Normalized dates (0.0 to 1.0).
        """
        if self.min_date is None or self.max_date is None:
            raise ValueError("DateNormalizer must be fitted before transform")
        
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        
        X = pd.to_datetime(X, errors='coerce')
        
        date_range = (self.max_date - self.min_date).total_seconds()
        
        if date_range == 0:
            return pd.Series([1.0] * len(X), index=X.index)
        
        if self.mode == "normalize_recent":
            # Most recent = 1.0, least recent = 0.0
            normalized = (X - self.min_date).dt.total_seconds() / date_range
        else:  # least_recent
            # Least recent = 0.0, most recent = 1.0 (same calculation)
            normalized = (X - self.min_date).dt.total_seconds() / date_range
        
        # Handle NaT values
        normalized = normalized.fillna(0.0)
        
        # Clip to [0, 1] range
        normalized = normalized.clip(0.0, 1.0)
        
        return normalized


# Backward Compatibility Aliases
LevelEncoder = RankedCategoryEncoder
LocationEncoder = ProximityEncoder
