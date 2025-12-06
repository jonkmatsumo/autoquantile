import pytest
import pandas as pd
import os
from src.utils.data_utils import load_data

def test_load_data(tmp_path):
    # Create a dummy CSV
    csv_content = """Level,Location,YearsOfExperience,YearsAtCompany,BaseSalary,Stock,Bonus,TotalComp,Date
E3,NY,2,1,100000,50000,10000,160000,2023-01-01
E4,SF,5-10,3+,150000,80000,20000,250000,2023-02-01
"""
    csv_file = tmp_path / "test_salaries.csv"
    csv_file.write_text(csv_content)
    
    df = load_data(str(csv_file))
    
    assert len(df) == 2
    
    # Check numeric cleaning
    # "5-10" -> 7.5
    assert df.iloc[1]["YearsOfExperience"] == 7.5
    # "3+" -> 3.0
    assert df.iloc[1]["YearsAtCompany"] == 3.0
    
    # Check date parsing
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])
    
    # Check numeric targets
    assert df.iloc[0]["BaseSalary"] == 100000
    assert df.iloc[0]["TotalComp"] == 160000
