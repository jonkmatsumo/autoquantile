import pandas as pd
from typing import Tuple, Optional

def validate_csv(file_buffer) -> Tuple[bool, Optional[str], Optional[pd.DataFrame]]:
    """
    Validates a CSV file buffer.
    
    Args:
        file_buffer: File-like object (e.g. UploadedFile).
        
    Returns:
        (is_valid, error_message, dataframe)
    """
    try:
        # Try reading a few bytes check emptiness
        file_buffer.seek(0)
        first_byte = file_buffer.read(1)
        if not first_byte:
            return False, "File is empty", None
        
        file_buffer.seek(0)
        df = pd.read_csv(file_buffer)
        
        if df.empty:
            return False, "CSV contains no data rows", None
            
        if len(df.columns) < 2:
            return False, "CSV must have at least 2 columns", None
            
        return True, None, df
        
    except Exception as e:
        return False, f"Failed to parse CSV: {str(e)}", None
    finally:
        # Reset pointer for downstream usage if needed, though streamlit usually handles this or we pass DF
        file_buffer.seek(0)
