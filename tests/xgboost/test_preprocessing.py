import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.xgboost.preprocessing import LevelEncoder, LocationEncoder, SampleWeighter

# --- LevelEncoder Tests ---

def test_level_encoder_transform():
    mock_config = {
        "mappings": {
            "levels": {"E3": 0, "E4": 1, "E5": 2}
        }
    }
    with patch('src.xgboost.preprocessing.get_config', return_value=mock_config):
        encoder = LevelEncoder()
        
        # Test Series input
        X = pd.Series(["E3", "E5", "E4", "Unknown"])
        result = encoder.transform(X)
        
        expected = np.array([0, 2, 1, -1])
        np.testing.assert_array_equal(result, expected)

def test_level_encoder_dataframe_input():
    mock_config = {
        "mappings": {
            "levels": {"E3": 0}
        }
    }
    with patch('src.xgboost.preprocessing.get_config', return_value=mock_config):
        encoder = LevelEncoder()
        df = pd.DataFrame({"Level": ["E3"]})
        result = encoder.transform(df)
        assert result[0] == 0

# --- LocationEncoder Tests ---

def test_location_encoder_transform():
    # Mock GeoMapper
    with patch('src.xgboost.preprocessing.GeoMapper') as MockGeoMapper:
        mock_mapper = MockGeoMapper.return_value
        # Setup mock behavior: NY -> 1, SF -> 2, Unknown -> 4
        mock_mapper.get_zone.side_effect = lambda x: 1 if x == "NY" else (2 if x == "SF" else 4)
        
        encoder = LocationEncoder()
        
        X = pd.Series(["NY", "SF", "Other", 123]) # 123 to test non-string
        result = encoder.transform(X)
        
        expected = np.array([1, 2, 4, 4])
        np.testing.assert_array_equal(result, expected)

# --- SampleWeighter Tests ---

def test_sample_weighter_transform():
    # Test with explicit k
    weighter = SampleWeighter(k=1.0, ref_date="2023-01-01")
    
    # Dates: 0 years old, 1 year old, 2 years old
    dates = pd.Series(["2023-01-01", "2022-01-01", "2021-01-01"])
    weights = weighter.transform(dates)
    
    # Expected: 1/(1+0)^1 = 1, 1/(1+1)^1 = 0.5, 1/(1+2)^1 = 0.333
    np.testing.assert_almost_equal(weights[0], 1.0)
    np.testing.assert_almost_equal(weights[1], 0.5, decimal=3)
    np.testing.assert_almost_equal(weights[2], 1/3, decimal=3)

def test_sample_weighter_future_dates():
    # Future dates should be treated as age 0
    weighter = SampleWeighter(k=1.0, ref_date="2023-01-01")
    dates = pd.Series(["2024-01-01"])
    weights = weighter.transform(dates)
    assert weights[0] == 1.0

def test_level_encoder_edge_cases():
    mock_config = {
        "mappings": {
            "levels": {"E3": 0}
        }
    }
    with patch('src.xgboost.preprocessing.get_config', return_value=mock_config):
        encoder = LevelEncoder()
        
        # Test with None, NaN, Empty string - should map to -1 (unknown)
        X = pd.Series([None, np.nan, "", "Unknown"])
        result = encoder.transform(X)
        
        expected = np.array([-1, -1, -1, -1])
        np.testing.assert_array_equal(result, expected)

def test_location_encoder_edge_cases():
    with patch('src.xgboost.preprocessing.GeoMapper') as MockGeoMapper:
        mock_mapper = MockGeoMapper.return_value
        # If input is not a string (e.g. NaN), get_zone might not even be called if we handle it in transform
        # or we rely on get_zone to handle it. 
        # Looking at implementation: 
        # def map_loc(loc):
        #     if not isinstance(loc, str): return 4
        #     return self.mapper.get_zone(loc)
        
        encoder = LocationEncoder()
        
        # None, NaN -> Should return 4 (Unknown) without calling mapper.get_zone
        X = pd.Series([None, np.nan, 123])
        result = encoder.transform(X)
        
        expected = np.array([4, 4, 4])
        np.testing.assert_array_equal(result, expected)
        
        # Empty string -> Should call mapper.get_zone("") -> let's say mapper returns 4 for empty
        mock_mapper.get_zone.return_value = 4
        
        X_str = pd.Series(["", "   "])
        result_str = encoder.transform(X_str)
        
        expected_str = np.array([4, 4])
        np.testing.assert_array_equal(result_str, expected_str)

def test_sample_weighter_edge_cases():
    weighter = SampleWeighter(k=1.0)
    
    # NaT handling
    dates = pd.Series([pd.NaT, "2023-01-01"])
    # Age of NaT will be NaT/NaN. 1/(1+NaN)^k = NaN
    weights = weighter.transform(dates)
    
    assert np.isnan(weights[0])
    assert not np.isnan(weights[1])

def test_sample_weighter_k_zero():
    # If k=0, weights should always be 1.0 regardless of age
    weighter = SampleWeighter(k=0.0)
    dates = pd.Series(["2020-01-01", "2023-01-01"])
    weights = weighter.transform(dates)
    
    np.testing.assert_array_equal(weights, np.array([1.0, 1.0]))
