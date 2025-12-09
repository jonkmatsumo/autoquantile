# AutoQuantile

A comprehensive framework for **Multi-Target Quantile Regression** using **XGBoost**. It automates the complex lifecycle of probabilistic modeling—from feature engineering and monotonic constraint enforcement to hyperparameter tuning and model versioning.

Key features include:
- **Automated Versioning**: Automatically tracks and versions trained models using **MLFlow**.
- **Auto-Tuning**: Integrated Hyperparameter Optimization using **Optuna** to automatically find the best model parameters.
- **LLM-Assisted Feature Engineering**: Uses Generative AI to intelligently infer feature and target variables, along with encodings and monotonic constraints.
- **Outlier Detection** (IQR) to filter extreme data points and improve model generalization.
- **Proximity Matching**: Geo-spatial grouping of cities into cost zones.

## Installation

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    
2.  Upgrade pip (Required for pyproject.toml based installs):
    ```bash
    pip install --upgrade pip
    ```

3.  Install the package:
    ```bash
    pip install -e .
    # Or for development (includes test dependencies):
    pip install -e ".[dev]"
    ```

## Usage

### Web Application (Streamlit)
The easiest way to use the system is via the web interface:

```bash
streamlit run src/app/app.py
```

This launches a dashboard where you can:
- **Train Models**: Upload a CSV, adjust configurations (like quantiles), and train new models interactively.
- **Run Inference**: Select a trained model, enter candidate details, and visualize the predicted salary distribution.

### Training (CLI)
To train the model via terminal:

```bash
salary-forecast-train
# Or: python3 -m src.cli.train_cli
```

You can specify the input CSV, config file, and output model path.

### Inference (CLI)
You can run the CLI in two modes:

**1. Interactive Mode**:
```bash
salary-forecast-infer
```
Follow the prompts to select a model and enter candidate details.

**2. Non-Interactive (Automation) Mode**:
Pass all required arguments via flags to skip prompts. Useful for scripts.

```bash
salary-forecast-infer --model salary_model.pkl --level E5 --location "New York" --yoe 5 --yac 2
```

**JSON Output**:
Add the `--json` flag to output results as a machine-readable JSON object (suppresses charts and tables).

```bash
salary-forecast-infer ... --json
```

## Testing

To run the unit tests:

```bash
python3 -m pytest tests/
```

## AI-Powered Configuration Workflow

AutoQuantile features an advanced **multi-step agentic workflow** powered by **LangGraph** that intelligently generates model configurations through a collaborative process between specialized AI agents and human oversight.

### Workflow Overview

The configuration generation process follows a structured 3-phase workflow, with each phase handled by a specialized AI agent:

#### Phase 1: Column Classification
The **Column Classification Agent** analyzes your dataset to identify:
- **Targets**: Columns to predict (e.g., salary components)
- **Features**: Columns to use as input features
- **Ignored**: Columns to exclude (e.g., IDs, metadata)

The agent uses data analysis tools to:
- Compute correlation matrices between columns
- Analyze column statistics (dtypes, null counts, unique values)
- Detect semantic types (numeric, categorical, datetime, boolean)
- Provide reasoning for each classification decision

**Human Review**: You can review and modify the agent's classifications before proceeding.

#### Phase 2: Feature Encoding
The **Feature Encoding Agent** determines optimal encoding strategies for categorical features:
- **Ordinal Encoding**: For features with inherent ordering (e.g., job levels: E3 < E4 < E5)
- **One-Hot Encoding**: For nominal categories with few unique values
- **Proximity Encoding**: For geographic features (cities grouped by distance)
- **Label Encoding**: For high-cardinality categorical features

The agent uses tools to:
- Detect ordinal patterns in categorical data
- Analyze unique value counts and distributions
- Examine correlation with target variables
- Generate encoding mappings where applicable

**Human Review**: You can adjust encoding types and mappings before finalizing.

#### Phase 3: Model Configuration
The **Model Configuration Agent** proposes:
- **Monotonic Constraints**: Enforces relationships between features and predictions
  - `1`: Increasing (higher feature → higher prediction)
  - `0`: No constraint
  - `-1`: Decreasing (higher feature → lower prediction)
- **Quantiles**: Optimal quantile levels for prediction (e.g., [0.1, 0.25, 0.5, 0.75, 0.9])
- **Hyperparameters**: XGBoost training parameters (max_depth, learning rate, etc.)

The agent considers:
- Feature correlations with targets
- Data characteristics (sample size, feature distributions)
- Best practices for quantile regression

**Human Review**: You can fine-tune hyperparameters, quantiles, and constraints.

### Using the Workflow

#### Via Web Interface
1. Launch the Streamlit app: `streamlit run src/app/app.py`
2. Upload your CSV dataset
3. Click **"Start AI-Powered Configuration Wizard"**
4. Review and confirm each phase:
   - Modify classifications if needed
   - Adjust encoding strategies
   - Refine model parameters
5. The workflow generates a complete configuration ready for training

#### Supported LLM Providers
- **OpenAI** (GPT-4, GPT-3.5): Requires `OPENAI_API_KEY` environment variable
- **Google Gemini**: Requires `GEMINI_API_KEY` environment variable

The system automatically detects available providers based on installed packages and API keys.

### Manual Configuration

You can also provide a configuration file directly (`config.json`) when using the CLI, or edit configurations manually in the web interface after generation.
