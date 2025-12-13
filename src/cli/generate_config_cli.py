import argparse
import json
import os
import sys
from typing import Dict, Any, Optional

import pandas as pd

from src.model.config_schema_model import validate_config_dict
from src.services.config_generator import ConfigGenerator
from src.services.workflow_service import WorkflowService
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def generate_config_workflow(
    df: pd.DataFrame, provider: str = "openai", preset: Optional[str] = None
) -> Dict[str, Any]:
    """Generate config using LLM workflow. Args: df (pd.DataFrame): Input data. provider (str): LLM provider. preset (Optional[str]): Optional preset name. Returns: Dict[str, Any]: Generated configuration. Raises: RuntimeError: If workflow fails."""
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


def main() -> None:
    """Main entry point for config generation CLI. Returns: None."""
    parser = argparse.ArgumentParser(
        description="Generate salary forecast configuration from data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate config using LLM workflow (default)
  python -m src.cli.generate_config_cli data.csv

  # Generate config using heuristic method
  python -m src.cli.generate_config_cli data.csv --heuristic

  # Generate config with specific LLM provider
  python -m src.cli.generate_config_cli data.csv --provider gemini

  # Save config to file
  python -m src.cli.generate_config_cli data.csv -o config.json
        """,
    )
    parser.add_argument("input_file", help="Path to input CSV file.")
    parser.add_argument("-o", "--output", help="Output JSON file path (default: stdout).")
    parser.add_argument(
        "--heuristic",
        action="store_true",
        help="Use heuristic-based configuration generation (default: LLM workflow).",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "gemini"],
        help="LLM provider for workflow generation (default: openai, only used with --workflow).",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="Preset name for workflow generation (only used with --workflow).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")

    args = parser.parse_args()

    log_level = "INFO" if args.verbose else "WARNING"
    setup_logging(level=log_level)

    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)

    try:
        df = pd.read_csv(args.input_file)
        logger.info(f"Loaded data with shape {df.shape}")

        # Determine generation method (workflow is default unless --heuristic is specified)
        use_workflow = not args.heuristic

        if use_workflow:
            logger.info(f"Generating config using LLM workflow (provider={args.provider})...")
            config = generate_config_workflow(df, provider=args.provider, preset=args.preset)
        else:
            logger.info("Generating config using heuristic method...")
            generator = ConfigGenerator()
            config = generator.generate_config_template(df)

        # Validate config before output
        try:
            validated_config = validate_config_dict(config)
            config = validated_config.model_dump()
            logger.info("Configuration validated successfully")
        except Exception as e:
            logger.error(f"Config validation failed: {e}")
            sys.exit(1)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(config, f, indent=4)
            logger.info(f"Configuration written to {args.output}")
        else:
            print(json.dumps(config, indent=4))

    except Exception as e:
        logger.error(f"Failed to generate config: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
