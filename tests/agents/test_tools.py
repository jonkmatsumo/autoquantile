"""Tests for agent analysis tools."""

import unittest
import json
import pandas as pd
import numpy as np
from src.agents.tools import (
    compute_correlation_matrix,
    get_column_statistics,
    get_unique_value_counts,
    detect_ordinal_patterns,
    detect_column_dtype,
)


class TestComputeCorrelationMatrix(unittest.TestCase):
    """Tests for compute_correlation_matrix tool."""
    
    def test_basic_correlation(self):
        """Test basic correlation computation."""
        df = pd.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "B": [2, 4, 6, 8, 10],  # Perfect correlation with A
            "C": [5, 4, 3, 2, 1]   # Perfect negative correlation with A
        })
        
        result = compute_correlation_matrix.invoke({"df_json": df.to_json()})
        result_dict = json.loads(result)
        
        self.assertIn("correlations", result_dict)
        self.assertIn("columns_analyzed", result_dict)
        
        # Find A-B correlation
        ab_corr = next(
            c for c in result_dict["correlations"] 
            if (c["column_1"] == "A" and c["column_2"] == "B") or
               (c["column_1"] == "B" and c["column_2"] == "A")
        )
        self.assertAlmostEqual(ab_corr["correlation"], 1.0, places=4)
    
    def test_specific_columns(self):
        """Test correlation with specific columns."""
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [2, 4, 6],
            "C": [1, 1, 1],
            "D": "text"  # Non-numeric
        })
        
        result = compute_correlation_matrix.invoke({
            "df_json": df.to_json(),
            "columns": "A, B"
        })
        result_dict = json.loads(result)
        
        self.assertEqual(set(result_dict["columns_analyzed"]), {"A", "B"})
    
    def test_insufficient_columns(self):
        """Test with less than 2 numeric columns."""
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": ["x", "y", "z"]
        })
        
        result = compute_correlation_matrix.invoke({"df_json": df.to_json()})
        result_dict = json.loads(result)
        
        self.assertIn("error", result_dict)


class TestGetColumnStatistics(unittest.TestCase):
    """Tests for get_column_statistics tool."""
    
    def test_numeric_column(self):
        """Test statistics for numeric column."""
        df = pd.DataFrame({
            "values": [1, 2, 3, 4, 5, None]
        })
        
        result = get_column_statistics.invoke({
            "df_json": df.to_json(),
            "column": "values"
        })
        result_dict = json.loads(result)
        
        self.assertEqual(result_dict["column"], "values")
        self.assertEqual(result_dict["total_count"], 6)
        self.assertEqual(result_dict["null_count"], 1)
        self.assertIn("numeric_stats", result_dict)
        self.assertEqual(result_dict["numeric_stats"]["mean"], 3.0)
    
    def test_categorical_column(self):
        """Test statistics for categorical column."""
        df = pd.DataFrame({
            "category": ["A", "B", "A", "C", "A"]
        })
        
        result = get_column_statistics.invoke({
            "df_json": df.to_json(),
            "column": "category"
        })
        result_dict = json.loads(result)
        
        self.assertEqual(result_dict["unique_count"], 3)
        self.assertNotIn("numeric_stats", result_dict)
    
    def test_missing_column(self):
        """Test with non-existent column."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        
        result = get_column_statistics.invoke({
            "df_json": df.to_json(),
            "column": "NonExistent"
        })
        result_dict = json.loads(result)
        
        self.assertIn("error", result_dict)


class TestGetUniqueValueCounts(unittest.TestCase):
    """Tests for get_unique_value_counts tool."""
    
    def test_basic_counts(self):
        """Test basic value counting."""
        df = pd.DataFrame({
            "status": ["Active", "Active", "Inactive", "Active", "Pending"]
        })
        
        result = get_unique_value_counts.invoke({
            "df_json": df.to_json(),
            "column": "status"
        })
        result_dict = json.loads(result)
        
        self.assertEqual(result_dict["total_unique_values"], 3)
        
        # Check Active count
        active_count = next(
            v for v in result_dict["value_counts"] 
            if v["value"] == "Active"
        )
        self.assertEqual(active_count["count"], 3)
    
    def test_limit(self):
        """Test limiting results."""
        df = pd.DataFrame({
            "id": list(range(100))
        })
        
        result = get_unique_value_counts.invoke({
            "df_json": df.to_json(),
            "column": "id",
            "limit": 5
        })
        result_dict = json.loads(result)
        
        self.assertEqual(result_dict["showing_top"], 5)
        self.assertEqual(len(result_dict["value_counts"]), 5)


class TestDetectOrdinalPatterns(unittest.TestCase):
    """Tests for detect_ordinal_patterns tool."""
    
    def test_numeric_pattern(self):
        """Test detection of numeric patterns like L1, L2, L3."""
        df = pd.DataFrame({
            "level": ["L1", "L2", "L3", "L4", "L5"]
        })
        
        result = detect_ordinal_patterns.invoke({
            "df_json": df.to_json(),
            "column": "level"
        })
        result_dict = json.loads(result)
        
        self.assertTrue(result_dict["is_ordinal"])
        self.assertIn("numeric_in_string", result_dict["patterns_detected"])
        self.assertIn("suggested_mapping", result_dict)
        
        # Verify ordering: L1 < L2 < L3 < L4 < L5
        mapping = result_dict["suggested_mapping"]
        self.assertTrue(mapping["L1"] < mapping["L5"])
    
    def test_keyword_pattern(self):
        """Test detection of ordinal keywords."""
        df = pd.DataFrame({
            "seniority": ["Junior Developer", "Senior Developer", "Lead Developer"]
        })
        
        result = detect_ordinal_patterns.invoke({
            "df_json": df.to_json(),
            "column": "seniority"
        })
        result_dict = json.loads(result)
        
        self.assertTrue(result_dict["is_ordinal"])
        self.assertIn("ordinal_keywords", result_dict["patterns_detected"])
    
    def test_non_ordinal(self):
        """Test non-ordinal column."""
        df = pd.DataFrame({
            "color": ["Red", "Blue", "Green", "Yellow"]
        })
        
        result = detect_ordinal_patterns.invoke({
            "df_json": df.to_json(),
            "column": "color"
        })
        result_dict = json.loads(result)
        
        self.assertFalse(result_dict["is_ordinal"])


class TestDetectColumnDtype(unittest.TestCase):
    """Tests for detect_column_dtype tool."""
    
    def test_numeric_continuous(self):
        """Test detection of continuous numeric column."""
        df = pd.DataFrame({
            "price": [10.5, 20.3, 15.7, 30.2]
        })
        
        result = detect_column_dtype.invoke({
            "df_json": df.to_json(),
            "column": "price"
        })
        result_dict = json.loads(result)
        
        self.assertEqual(result_dict["semantic_type"], "numeric_continuous")
    
    def test_categorical(self):
        """Test detection of categorical column."""
        df = pd.DataFrame({
            "category": ["A", "B", "A", "C", "B"] * 10
        })
        
        result = detect_column_dtype.invoke({
            "df_json": df.to_json(),
            "column": "category"
        })
        result_dict = json.loads(result)
        
        self.assertEqual(result_dict["semantic_type"], "categorical")
    
    def test_identifier(self):
        """Test detection of ID column."""
        df = pd.DataFrame({
            "user_id": list(range(100))
        })
        
        result = detect_column_dtype.invoke({
            "df_json": df.to_json(),
            "column": "user_id"
        })
        result_dict = json.loads(result)
        
        self.assertEqual(result_dict["semantic_type"], "identifier")
    
    def test_boolean(self):
        """Test detection of boolean column."""
        df = pd.DataFrame({
            "is_active": [True, False, True, False]
        })
        
        result = detect_column_dtype.invoke({
            "df_json": df.to_json(),
            "column": "is_active"
        })
        result_dict = json.loads(result)
        
        self.assertEqual(result_dict["semantic_type"], "boolean")


if __name__ == "__main__":
    unittest.main()

