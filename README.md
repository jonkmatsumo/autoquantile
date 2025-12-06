# Salary Forecasting Engine

A machine learning system to predict compensation distributions (Base Salary, Stock, Bonus, Total Comp) based on candidate attributes. It uses **XGBoost** with Quantile Regression to forecast the 25th, 50th, and 75th percentiles, enforcing monotonic constraints on Level and Years of Experience.

## Installation

1.  Create a virtual environment (optional but recommended):
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training
To train the model interactively:

```bash
python3 -m src.cli.train_cli
```

You can specify the input CSV, config file, and output model path.

### Inference (CLI)
To run the interactive CLI for making predictions:

```bash
python3 -m src.cli.inference_cli
```

Follow the prompts to select a model and enter candidate details.

## Testing

To run the unit tests:

```bash
python3 -m pytest tests/
```

## Configuration

The model and data processing are highly configurable via `config.json`. You can create your own configuration file to adapt the model to different datasets or requirements.

### Structure

#### 1. Mappings (`mappings`)
Defines how categorical data is mapped to numerical values.

- **`levels`**: Maps job levels (e.g., "E3", "E4") to ordinal integers (0, 1, 2...).
  ```json
  "levels": {
      "E3": 0,
      "E4": 1,
      ...
  }
  ```
- **`location_targets`**: Maps major cities to "Cost Zones" (integers). Lower numbers typically represent higher cost of living.
  ```json
  "location_targets": {
      "New York, NY": 1,
      "San Francisco, CA": 1,
      "Austin, TX": 3,
      ...
  }
  ```

#### 2. Location Settings (`location_settings`)
Controls the proximity matching logic.

- **`max_distance_km`**: The maximum distance (in km) for a city to be considered part of a target city's zone. If a city is further than this from any target, it falls into a default "Unknown" zone.

#### 3. Model Settings (`model`)
Configures the XGBoost model and feature engineering.

- **`targets`**: List of salary components to predict (e.g., "BaseSalary", "Stock").
- **`quantiles`**: List of quantiles to predict (e.g., 0.25, 0.50, 0.75).
- **`sample_weight_k`**: Decay factor for sample weighting based on recency. Higher `k` gives more weight to recent data.
- **`features`**: List of features to use in the model.
  - **`name`**: Feature name (must match column in processed data).
  - **`monotone_constraint`**: Enforces monotonic relationships.
    - `1`: Increasing constraint (higher feature value -> higher prediction).
    - `0`: No constraint.
    - `-1`: Decreasing constraint.

### Example `config.json`

```json
{
    "mappings": {
        "levels": {"E3": 0, "E4": 1},
        "location_targets": {"New York, NY": 1}
    },
    "location_settings": {"max_distance_km": 50},
    "model": {
        "targets": ["BaseSalary"],
        "quantiles": [0.5],
        "sample_weight_k": 1.0,
        "features": [
            {"name": "Level_Enc", "monotone_constraint": 1}
        ]
    }
}
```

## Proximity Matching

The system uses `geopy` to automatically map input locations to the nearest target city defined in `config.json`.
- **Dynamic Matching**: "Newark" maps to "New York" (Zone 1) because it is within the configured `max_distance_km` (default 50km).
- **Caching**: Geocoding results are cached in `city_cache.json` to speed up subsequent runs and reduce API usage.
- **O(1) Lookup**: Once a city is processed, its zone is cached in memory for instant lookup.
