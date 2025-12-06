import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.model.preprocessing import LevelEncoder, LocationEncoder, SampleWeighter

# --- LevelEncoder Tests ---

def test_level_encoder_transform():
    mock_config = {
        "mappings": {
            "levels": {"E3": 0, "E4": 1, "E5": 2}
        }
    }
    with patch('src.model.preprocessing.get_config', return_value=mock_config):
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
    with patch('src.model.preprocessing.get_config', return_value=mock_config):
        encoder = LevelEncoder()
        df = pd.DataFrame({"Level": ["E3"]})
        result = encoder.transform(df)
        assert result[0] == 0

# --- LocationEncoder Tests ---

def test_location_encoder_transform():
    # Mock GeoMapper
    with patch('src.model.preprocessing.GeoMapper') as MockGeoMapper:
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
