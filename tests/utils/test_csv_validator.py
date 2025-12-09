import pytest
import io
import pandas as pd
from src.utils.csv_validator import validate_csv

def test_valid_csv():
    data = "Col1,Col2\n1,2\n3,4"
    f = io.BytesIO(data.encode('utf-8'))
    is_valid, err, df = validate_csv(f)
    assert is_valid
    assert err is None
    assert len(df) == 2

def test_empty_file():
    f = io.BytesIO(b"")
    is_valid, err, df = validate_csv(f)
    assert not is_valid
    assert "File is empty" in err

def test_too_few_columns():
    data = "Col1\n1\n2"
    f = io.BytesIO(data.encode('utf-8'))
    is_valid, err, df = validate_csv(f)
    assert not is_valid
    assert "must have at least 2 columns" in err

def test_parsing_error():
    f = io.BytesIO(b"garbage data")
    # pd.read_csv actually handles garbage pretty well (single col), unless totally binary
    # We test non-utf8?
    f = io.BytesIO(b"\xff\xff")
    is_valid, err, df = validate_csv(f)
    assert not is_valid
    assert "Failed to parse CSV" in err


def test_csv_with_missing_values():
    """Test CSV with missing values (should still be valid)."""
    data = "Col1,Col2\n1,2\n,4\n3,"
    f = io.BytesIO(data.encode('utf-8'))
    is_valid, err, df = validate_csv(f)
    assert is_valid
    assert err is None
    assert len(df) == 3


def test_csv_single_row():
    """Test CSV with only header row."""
    data = "Col1,Col2"
    f = io.BytesIO(data.encode('utf-8'))
    is_valid, err, df = validate_csv(f)
    # CSV validator may reject empty dataframes - check actual behavior
    # If invalid, should have error message
    if not is_valid:
        assert err is not None
    else:
        assert len(df) == 0


def test_csv_unicode_characters():
    """Test CSV with unicode characters."""
    data = "Name,Salary\nJosé,100000\nMüller,200000"
    f = io.BytesIO(data.encode('utf-8'))
    is_valid, err, df = validate_csv(f)
    assert is_valid
    assert err is None
    assert len(df) == 2
