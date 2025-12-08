import sys
from src.utils.logger import get_logger

def apply_backward_compatibility():
    """
    Maps legacy 'src.model' modules to 'src.xgboost' to allow unpickling
    old MLflow models that reference the old package structure.
    """
    logger = get_logger(__name__)
    
    if 'src.model' not in sys.modules:
        try:
            import src.xgboost
            import src.xgboost.model
            import src.xgboost.preprocessing
            
            # Map the old module names to the new ones
            sys.modules['src.model'] = src.xgboost
            sys.modules['src.model.model'] = src.xgboost.model
            sys.modules['src.model.preprocessing'] = src.xgboost.preprocessing
            
            logger.info("Applied backward compatibility mapping: src.model -> src.xgboost")
        except ImportError as e:
            logger.warning(f"Failed to apply backward compatibility: {e}")
