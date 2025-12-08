import sys
from src.utils.logger import get_logger

def apply_backward_compatibility():
    """
    Maps legacy 'src.model' modules to 'src.xgboost' to allow unpickling
    old MLflow models that reference the old package structure.
    """
    logger = get_logger(__name__)
    # Backward compatibility for 'src.model' import path has been removed 
    # as strict migration to 'src.xgboost' is enforced and 'src.model' is now used for Pydantic models.
    pass
