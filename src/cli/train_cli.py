import argparse
import contextlib
import io
import json
import os
import traceback
from typing import Any, Dict, Optional

import mlflow
import pandas as pd
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from src.model.config_schema_model import validate_config_dict
from src.services.model_registry import SalaryForecasterWrapper, get_experiment_name
from src.services.workflow_service import WorkflowService
from src.utils.data_utils import load_data
from src.utils.logger import setup_logging
from src.xgboost.model import SalaryForecaster


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load and validate config from JSON file. Args: config_path (str): Config file path. Returns: Dict[str, Any]: Validated configuration. Raises: FileNotFoundError: If config file not found. ValueError: If config is invalid."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    try:
        validated_config = validate_config_dict(config_dict)
        return validated_config.model_dump()
    except Exception as e:
        raise ValueError(f"Config validation failed: {e}. Please ensure your config matches the required schema.") from e


def generate_config_from_data(csv_path: str, provider: str = "openai", preset: Optional[str] = None) -> Dict[str, Any]:
    """Generate config using LLM workflow from data. Args: csv_path (str): Training data CSV path. provider (str): LLM provider. preset (Optional[str]): Optional preset name. Returns: Dict[str, Any]: Generated configuration. Raises: FileNotFoundError: If CSV file not found. RuntimeError: If workflow fails."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = load_data(csv_path)

    try:
        workflow_service = WorkflowService(provider=provider)
        result = workflow_service.start_workflow(df, sample_size=50, preset=preset)

        if result.get("status") == "error":
            raise RuntimeError(f"Workflow failed: {result.get('error', 'Unknown error')}")

        # Confirm classification
        result = workflow_service.confirm_classification()
        if result.get("status") == "error":
            raise RuntimeError(f"Classification confirmation failed: {result.get('error', 'Unknown error')}")

        # Confirm encoding
        result = workflow_service.confirm_encoding()
        if result.get("status") == "error":
            raise RuntimeError(f"Encoding confirmation failed: {result.get('error', 'Unknown error')}")

        # Get final config
        final_config = workflow_service.get_final_config()
        if not final_config:
            raise RuntimeError("Failed to generate configuration from workflow")

        return final_config
    except Exception as e:
        raise RuntimeError(f"Failed to generate config using workflow: {e}") from e


def train_workflow(
    csv_path: str,
    config: Dict[str, Any],
    output_path: str,
    console: Any,
    do_tune: bool = False,
    num_trials: int = 20,
    remove_outliers: bool = False,
) -> None:
    """Execute the model training workflow. Args: csv_path (str): Training data CSV path. config (Dict[str, Any]): Required configuration dictionary. output_path (str): Model output path (deprecated). console (Any): Rich console. do_tune (bool): Run hyperparameter tuning. num_trials (int): Tuning trials. remove_outliers (bool): Remove outliers. Returns: None. Raises: FileNotFoundError: If CSV file not found. ValueError: If config is invalid."""
    if not os.path.exists(csv_path):
        console.print(f"[bold red]Error: {csv_path} not found.[/bold red]")
        return

    if not config:
        console.print("[bold red]Error: Config is required.[/bold red]")
        return

    status_text = Text("Status: Preparing...", style="bold blue")

    results_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    results_table.add_column("Component", style="cyan")
    results_table.add_column("Percentile", justify="right")
    results_table.add_column("Best Round", justify="right")
    results_table.add_column("Metric")
    results_table.add_column("Score", justify="right")

    output_group = Group(status_text, Text(""), results_table)

    with Live(output_group, console=console, refresh_per_second=4, transient=False):
        status_text.plain = f"Status: Loading data from {csv_path}..."
        df = load_data(csv_path)

        status_text.plain = "Status: Starting training workflow..."

        with contextlib.redirect_stdout(io.StringIO()):
            forecaster = SalaryForecaster(config=config)

        if do_tune:
            status_text.plain = f"Status: Tuning hyperparameters (Trials={num_trials})..."
            best_params = forecaster.tune(df, n_trials=num_trials)
            console.print(f"[dim]Best Params: {best_params}[/dim]")

        status_text.plain = "Status: Starting training..."

        def console_callback(msg: str, data: Optional[dict] = None) -> None:
            if data and data.get("stage") == "start":
                model_name = data["model_name"]
                status_text.plain = f"Status: Training {model_name}..."

            elif data and data.get("stage") == "cv_end":
                metric = data.get("metric_name", "metric")
                best_round = str(data.get("best_round"))
                best_score = f"{data.get('best_score'):.4f}"
                model_name = data.get("model_name", "Unknown")

                if "_p" in model_name:
                    parts = model_name.rsplit("_", 1)
                    component = parts[0]
                    percentile = parts[1]
                else:
                    component = model_name
                    percentile = "-"

                results_table.add_row(component, percentile, best_round, metric, best_score)

            elif data and data.get("stage") == "cv_start":
                pass

        forecaster.train(df, callback=console_callback, remove_outliers=remove_outliers)

        experiment_name = get_experiment_name()
        status_text.plain = f"Status: Logging model to MLflow (Experiment: {experiment_name})..."

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run() as run:
            mlflow.log_params(
                {
                    "remove_outliers": remove_outliers,
                    "do_tune": do_tune,
                    "n_trials": num_trials if do_tune else 0,
                    "data_rows": len(df),
                }
            )

            wrapper = SalaryForecasterWrapper(forecaster)
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=wrapper,
                pip_requirements=["xgboost", "pandas", "scikit-learn"],
            )

            console.print(f"[dim]Run ID: {run.info.run_id}[/dim]")

        status_text.plain = "Status: Completed"


def main() -> None:
    """Main entry point for training CLI. Returns: None."""
    setup_logging()
    console = Console()
    console.print("[bold green]Salary Forecasting Training CLI[/bold green]")

    parser = argparse.ArgumentParser(
        description="Train Salary Forecast Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with existing config file
  python -m src.cli.train_cli --csv data.csv --config config.json

  # Generate config from data and train
  python -m src.cli.train_cli --csv data.csv --generate-config

  # Generate config with specific LLM provider
  python -m src.cli.train_cli --csv data.csv --generate-config --provider gemini
        """,
    )
    parser.add_argument("--csv", default="salaries-list.csv", help="Path to training CSV")
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "--config",
        help="Path to config JSON file (required if not using --generate-config)",
    )
    config_group.add_argument(
        "--generate-config",
        action="store_true",
        help="Generate configuration using LLM workflow from data",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "gemini"],
        help="LLM provider for config generation (default: openai, only used with --generate-config)",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="Preset name for config generation (only used with --generate-config)",
    )
    parser.add_argument(
        "--output", default=None, help="Deprecated: Output path (now logs to MLflow)"
    )
    parser.add_argument("--tune", action="store_true", help="Enable Optuna hyperparameter tuning")
    parser.add_argument("--num-trials", type=int, default=20, help="Number of tuning trials")
    parser.add_argument(
        "--remove-outliers", action="store_true", help="Remove outliers using IQR before training"
    )

    args = parser.parse_args()

    try:
        # Load or generate config
        if args.generate_config:
            console.print("[bold blue]Generating configuration from data using LLM workflow...[/bold blue]")
            try:
                config = generate_config_from_data(args.csv, provider=args.provider, preset=args.preset)
                console.print("[bold green]✓ Configuration generated successfully![/bold green]")
            except Exception as e:
                console.print(f"[bold red]Failed to generate config: {e}[/bold red]")
                traceback.print_exc()
                return
        else:
            if not args.config:
                console.print("[bold red]Error: --config is required when not using --generate-config[/bold red]")
                return
            try:
                config = load_config_from_file(args.config)
                console.print(f"[bold green]✓ Configuration loaded from {args.config}[/bold green]")
            except FileNotFoundError as e:
                console.print(f"[bold red]Error: {e}[/bold red]")
                console.print("[yellow]Hint: Use --generate-config to create a config from your data[/yellow]")
                return
            except ValueError as e:
                console.print(f"[bold red]Config validation error: {e}[/bold red]")
                return

        train_workflow(
            args.csv,
            config,
            args.output,
            console,
            do_tune=args.tune,
            num_trials=args.num_trials,
            remove_outliers=args.remove_outliers,
        )
        console.print(f"\n[bold green]Training workflow completed![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Training failed: {e}[/bold red]")
        traceback.print_exc()


if __name__ == "__main__":
    main()
