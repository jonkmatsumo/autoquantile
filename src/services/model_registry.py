import glob
import os
import pickle
from typing import List, Optional
from src.model.model import SalaryForecaster

class ModelRegistry:
    """Service for managing model persistence and retrieval."""
    
    def __init__(self, model_dir: str = "."):
        self.model_dir = model_dir

    def list_models(self) -> List[str]:
        """Lists all available model files (.pkl) in the model directory."""
        return glob.glob(os.path.join(self.model_dir, "*.pkl"))

    def load_model(self, path: str) -> SalaryForecaster:
        """Loads a model from the specified path."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        with open(path, "rb") as f:
            return pickle.load(f)

    def save_model(self, model: SalaryForecaster, path: str) -> str:
        """Saves a model to the specified path."""
        # Ensure path ends with .pkl
        if not path.endswith(".pkl"):
            path += ".pkl"
            
        full_path = os.path.join(self.model_dir, path)
        with open(full_path, "wb") as f:
            pickle.dump(model, f)
            
        return full_path
