import pytest
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from src.model.model import SalaryForecaster

@pytest.fixture(scope="module")
def trained_model():
    # Create a small manual dataset for testing constraints
    data = []
    levels = ["E3", "E4", "E5", "E6", "E7"]
    locations = ["New York", "San Francisco", "Seattle", "Austin"]
    
    for _ in range(200): # Smaller dataset for unit tests
        for level_idx, level in enumerate(levels):
            for loc in locations:
                base = 100000 + level_idx * 50000
                if loc == "New York": base *= 1.2
                
                data.append({
                    "Level": level,
                    "Location": loc,
                    "Date": pd.Timestamp.now(),
                    "YearsOfExperience": level_idx * 3 + 2,
                    "YearsAtCompany": 2,
                    "BaseSalary": base,
                    "Stock": base * 0.5,
                    "Bonus": base * 0.1,
                    "TotalComp": base * 1.6
                })
    
    df = pd.DataFrame(data)
    model = SalaryForecaster()
    model.train(df)
    return model

def test_monotonicity_level(trained_model):
    """
    Verify that higher levels result in higher (or equal) salary, holding other factors constant.
    """
    # Create base input
    base_input = {
        "Location": "New York",
        "YearsOfExperience": 5,
        "YearsAtCompany": 2
    }
    
    levels = ["E3", "E4", "E5", "E6", "E7"]
    preds = []
    
    for level in levels:
        inp = pd.DataFrame([{**base_input, "Level": level}])
        res = trained_model.predict(inp)
        preds.append(res["BaseSalary"]["p50"][0])
        
    # Check if sorted
    print(f"\nLevel Predictions (E3-E7): {preds}")
    assert np.all(np.diff(preds) >= -1e-9), "Base Salary P50 should be monotonic with Level"

def test_monotonicity_yoe(trained_model):
    """
    Verify that more experience results in higher (or equal) salary.
    """
    # Create base input
    base_input = {
        "Level": "E5",
        "Location": "New York",
        "YearsAtCompany": 2
    }
    
    yoes = [2, 5, 8, 12, 15]
    preds = []
    
    for yoe in yoes:
        inp = pd.DataFrame([{**base_input, "YearsOfExperience": yoe}])
        res = trained_model.predict(inp)
        preds.append(res["BaseSalary"]["p50"][0])
        
    # Check if sorted
    print(f"\nYOE Predictions (2, 5, 8, 12, 15): {preds}")
    assert np.all(np.diff(preds) >= -1e-9), "Base Salary P50 should be monotonic with YOE"

def test_quantiles_ordering(trained_model):
    """
    Verify P25 <= P50 <= P75
    """
    inp = pd.DataFrame([{
        "Level": "E5",
        "Location": "New York",
        "YearsOfExperience": 8,
        "YearsAtCompany": 2
    }])
    
    res = trained_model.predict(inp)
    
    for target in res:
        p25 = res[target]["p25"][0]
        p50 = res[target]["p50"][0]
        p75 = res[target]["p75"][0]
        
        assert p25 <= p50, f"{target}: P25 ({p25}) > P50 ({p50})"
        assert p50 <= p75, f"{target}: P50 ({p50}) > P75 ({p75})"

def test_location_impact(trained_model):
    """
    Verify NY > Austin (Zone 1 > Zone 3)
    """
    inp_ny = pd.DataFrame([{
        "Level": "E5", "Location": "New York", "YearsOfExperience": 8, "YearsAtCompany": 2
    }])
    inp_austin = pd.DataFrame([{
        "Level": "E5", "Location": "Austin", "YearsOfExperience": 8, "YearsAtCompany": 2
    }])
    
    pred_ny = trained_model.predict(inp_ny)["BaseSalary"]["p50"][0]
    pred_austin = trained_model.predict(inp_austin)["BaseSalary"]["p50"][0]
    
    print(f"\nLocation Check: NY={pred_ny}, Austin={pred_austin}")
    assert pred_ny > pred_austin, "NY salary should be higher than Austin"
class TestConfigHyperparams(unittest.TestCase):
    def setUp(self):
        # Sample data
        self.df = pd.DataFrame({
            "Level": ["E3", "E4"],
            "Location": ["New York", "San Francisco"],
            "YearsOfExperience": [1, 2],
            "YearsAtCompany": [0, 1],
            "Date": ["2023-01-01", "2023-01-02"],
            "BaseSalary": [100000, 120000]
        })
        
        # Minimal config with custom hyperparams
        self.custom_config = {
            "mappings": {
                "levels": {"E3": 0, "E4": 1},
                "location_targets": {"New York": 1, "San Francisco": 1}
            },
            "location_settings": {"max_distance_km": 50},
            "model": {
                "targets": ["BaseSalary"],
                "quantiles": [0.5],
                "features": [
                     {"name": "Level_Enc", "monotone_constraint": 1}
                ],
                "hyperparameters": {
                    "training": {
                        "objective": "reg:quantileerror",
                        "max_depth": 7,  # Custom value to check
                        "eta": 0.01      # Custom value to check
                    },
                    "cv": {
                        "num_boost_round": 10,
                        "nfold": 2
                    }
                }
            }
        }

    @patch("src.model.model.xgb.train")
    @patch("src.model.model.xgb.cv")
    @patch("src.model.model.xgb.DMatrix")
    def test_custom_hyperparams_passed_to_xgb(self, mock_dmatrix, mock_cv, mock_train):
        # Mock cv results
        mock_cv.return_value = pd.DataFrame({'test-quantile-mean': [0.5, 0.4, 0.3]})
        
        forecaster = SalaryForecaster(config=self.custom_config)
        forecaster.train(self.df)
        
        # Check if train was called with custom params
        # We need to inspect the call to xgb.train
        # Call args: (params, dtrain, num_boost_round)
        
        self.assertTrue(mock_train.called)
        
        # Get arguments of the first call
        call_args = mock_train.call_args
        params_arg = call_args[0][0]
        
        # Verify custom params exist
        self.assertEqual(params_arg.get("max_depth"), 7)
        self.assertEqual(params_arg.get("eta"), 0.01)
        
        # Verify merged params exist
        self.assertEqual(params_arg.get("quantile_alpha"), 0.5)

    @patch("src.model.model.xgb.train")
    @patch("src.model.model.xgb.cv")
    @patch("src.model.model.xgb.DMatrix")
    def test_custom_cv_params_passed(self, mock_dmatrix, mock_cv, mock_train):
        mock_cv.return_value = pd.DataFrame({'test-quantile-mean': [0.5]})
        
        forecaster = SalaryForecaster(config=self.custom_config)
        forecaster.train(self.df)
        
        self.assertTrue(mock_cv.called)
        
        # Check kwargs
        call_kwargs = mock_cv.call_args[1]
        
        # Or checking positional/keyword args depending on implementation
        # The implementation passes arguments as kwargs or positional?
        # cv_results = xgb.cv(params, dtrain, num_boost_round=..., nfold=...)
        
        # Let's inspect call_kwargs
        self.assertEqual(call_kwargs.get("num_boost_round"), 10)
        self.assertEqual(call_kwargs.get("nfold"), 2)

    def test_analyze_cv_results_static(self):
        """Test the static helper method _analyze_cv_results directly."""
        cv_df = pd.DataFrame({
            'test-quantile-mean': [0.5, 0.4, 0.45, 0.6],
            'other-metric': [1, 2, 3, 4]
        })
        
        # Expected: Best score is min (0.4) at index 1 (round 2)
        best_round, best_score = SalaryForecaster._analyze_cv_results(cv_df, 'test-quantile-mean')
        
        self.assertEqual(best_round, 2)
        self.assertEqual(best_score, 0.4)
        
        # Test error on missing column
        with self.assertRaises(ValueError):
            SalaryForecaster._analyze_cv_results(cv_df, 'missing-metric')
