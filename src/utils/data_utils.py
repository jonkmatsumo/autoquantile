import pandas as pd
import numpy as np
import re

from typing import Union

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads and cleans the salary data from CSV.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned dataframe with parsed dates and numeric columns.
    """
    df = pd.read_csv(filepath)
    
    # Columns are already renamed in the CSV to match model expectations
    # Level,TotalComp,BaseSalary,Stock,Bonus,YearsOfExperience,YearsAtCompany,Date,Location
    
    # Clean numeric columns that might have strings like "11+" or "5-10"
    def clean_years(val: Union[int, float, str]) -> float:
        """Helper to parse year strings like '11+' or '5-10'."""
        if isinstance(val, (int, float)):
            return float(val)
        val = str(val).strip()
        if "+" in val:
            return float(val.replace("+", ""))
        if "-" in val:
            # Take average of range
            parts = val.split("-")
            return (float(parts[0]) + float(parts[1])) / 2
        return float(val)

    df["YearsOfExperience"] = df["YearsOfExperience"].apply(clean_years)
    df["YearsAtCompany"] = df["YearsAtCompany"].apply(clean_years)
    
    # Parse dates
    # format='mixed' allows parsing different formats in the same column (e.g. '2023-01-01', 'Jan 1, 2023')
    # dayfirst=False is generally safer for US-centric tech salary data unless known otherwise
    try:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce', format='mixed')
    except ValueError:
        # Fallback for older pandas versions or extremely weird data
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    
    # Ensure numeric targets
    targets = ["BaseSalary", "Stock", "Bonus", "TotalComp"]
    for col in targets:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    return df
