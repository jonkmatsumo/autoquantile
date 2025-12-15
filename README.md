# AutoQuantile

A framework for **Multi-Target Quantile Regression** using **XGBoost**. Automates the lifecycle of probabilistic modeling—from feature engineering to hyperparameter tuning and model versioning.

## Requirements

- Python 3.9 or higher (tested with Python 3.12)

Key features include:
- **Automated Versioning**: Automatically tracks and versions trained models using **MLflow**.
- **Auto-Tuning**: Integrated Hyperparameter Optimization using **Optuna** to automatically find the best model parameters.
- **LLM-Assisted Feature Engineering**: Uses Generative AI (OpenAI GPT-4 or Google Gemini) to intelligently infer feature and target variables, along with encodings and monotonic constraints through a multi-step agentic workflow.
- **Outlier Detection**: IQR-based outlier filtering to improve model generalization.
- **Proximity Matching**: Geo-spatial grouping of cities into cost zones using distance calculations.
- **Prompt Injection Detection**: Security validation to prevent malicious inputs in the AI workflow.
- **Type Safety**: Comprehensive type annotations with mypy static type checking.

## Installation

1. Create a virtual environment (recommended):
    ```bash
    python3.12 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

2. Upgrade pip (required for pyproject.toml based installs):
    ```bash
    pip install --upgrade pip
    ```

3. Install the package:
    ```bash
    # Development (recommended)
    pip install -e ".[dev]"
    
    # Or production only
    pip install -r requirements.txt
    ```

4. Set up environment variables (for LLM features):
    ```bash
    export OPENAI_API_KEY=your_openai_key_here
    export GEMINI_API_KEY=your_gemini_key_here  # Optional
    ```

## Usage

### Web Application (Streamlit)
The easiest way to use the system is via the web interface:

```bash
streamlit run src/app/app.py
```

This launches a dashboard with training and inference pages:
- **Training**: Upload CSV files, use AI-powered configuration wizard, train models with hyperparameter tuning
- **Inference**: Select models, make predictions, view visualizations and feature importance

## Testing & Type Checking

Run tests:
```bash
python3 -m pytest tests/
```

Run type checking:
```bash
mypy src/
```

## AI-Powered Configuration Workflow

Multi-step agentic workflow powered by **LangGraph** that generates model configurations through specialized AI agents:

1. **Column Classification**: Identifies targets, features, and columns to ignore
2. **Feature Encoding**: Determines optimal encodings (ordinal, one-hot, proximity, label)
3. **Model Configuration**: Proposes monotonic constraints, quantiles, and hyperparameters

**Usage**: Launch Streamlit app, upload CSV, click "Start AI-Powered Configuration Wizard", and review/confirm each phase. Supports OpenAI (GPT-4/3.5) and Google Gemini.

## REST API

Comprehensive REST API built with **FastAPI** for programmatic access. Features: model management, inference (single/batch), training jobs, configuration workflow, and analytics.

**Start server:**
```bash
uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

**Documentation**: `http://localhost:8000/docs` (Swagger UI) or `/redoc` (ReDoc)

**Authentication** (optional):
```bash
export API_KEY=your_api_key_here
```

**Example:**
```python
import requests
response = requests.get(
    "http://localhost:8000/api/v1/models",
    headers={"X-API-Key": "your_api_key"}
)
```

All endpoints prefixed with `/api/v1`.

## MCP Server (Model Context Protocol)

Native **MCP** server implementation for agent-native interactions via JSON-RPC 2.0. Includes 11 tools for model operations, inference, training, configuration workflow, and analytics.

**Endpoint**: `POST /mcp/rpc` (mounted as FastAPI sub-application)

**Protocol**: JSON-RPC 2.0

**Example request:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "predict_salary",
    "arguments": {"run_id": "abc123", "features": {"Level": "L5"}}
  },
  "id": 1
}
```

**Tool discovery**: Use `{"method": "tools/list", "id": 1}` to list all available tools. Each tool includes semantic descriptions, JSON Schema definitions, and examples for LLM integration.

For details, see `API_DESIGN.md`.
