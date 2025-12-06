import pandas as pd
import pickle
import os
from src.model.model import SalaryForecaster
from src.utils.data_utils import load_data
from src.utils.config_loader import load_config

def train_model(csv_path="salaries-list.csv", config_path="config.json", output_path="salary_model.pkl"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # Load config if provided (though config_loader handles it globally, explicit load is good for testing)
    if config_path and os.path.exists(config_path):
        load_config(config_path)

    print(f"Loading data from {csv_path}...")
    df = load_data(csv_path)
    print(f"Loaded {len(df)} samples.")
    
    print("Initializing model...")
    forecaster = SalaryForecaster()
    
    print("Training model...")
    forecaster.train(df)
    
    print(f"Saving model to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(forecaster, f)
    print("Model saved.")
    
    # Simple inference check
    print("\nRunning sample inference...")
    sample_input = pd.DataFrame([{
        "Level": "E4",
        "Location": "New York",
        "YearsOfExperience": 3,
        "YearsAtCompany": 0
    }])
    
    prediction = forecaster.predict(sample_input)
    print("Prediction for E4 New Hire in NY (3 YOE):")
    for target, preds in prediction.items():
        print(f"  {target}: P25={preds['p25'][0]:.0f}, P50={preds['p50'][0]:.0f}, P75={preds['p75'][0]:.0f}")

if __name__ == "__main__":
    train_model()
