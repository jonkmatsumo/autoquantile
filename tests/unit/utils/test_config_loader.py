import pytest
from pydantic import ValidationError

from src.model.config_schema_model import validate_config_dict

# Import conftest function directly (pytest will handle the path)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from conftest import create_test_config


def test_validate_config_dict_success():
    """Verify validation accepts valid configuration."""
    config = create_test_config()
    validated = validate_config_dict(config)
    validated_dict = validated.model_dump()
    # Validation may add default fields like hyperparameters
    if "hyperparameters" not in config.get("model", {}):
        config.setdefault("model", {})["hyperparameters"] = {}
    assert validated_dict == config


def test_validate_config_dict_missing_required_keys():
    """Verify validation catches incomplete configuration."""
    incomplete_config = {"mappings": {}}

    with pytest.raises(ValidationError) as exc_info:
        validate_config_dict(incomplete_config)

    assert "location_settings" in str(exc_info.value) or "model" in str(exc_info.value)


def test_validate_config_dict_missing_model_targets():
    """Verify validation catches missing model targets."""
    config = create_test_config()
    del config["model"]["targets"]

    with pytest.raises(ValidationError) as exc_info:
        validate_config_dict(config)

    assert "targets" in str(exc_info.value)


def test_validate_config_dict_missing_model_quantiles():
    """Verify validation uses default quantiles when missing."""
    config = create_test_config()
    del config["model"]["quantiles"]

    validated = validate_config_dict(config)
    assert validated.model.quantiles == [0.1, 0.25, 0.5, 0.75, 0.9]


def test_validate_config_dict_invalid_quantiles():
    """Verify validation catches invalid quantile values."""
    config = create_test_config()
    config["model"]["quantiles"] = [1.5, 2.0]

    with pytest.raises(ValidationError) as exc_info:
        validate_config_dict(config)

    assert (
        "quantile" in str(exc_info.value).lower() or "greater than 1" in str(exc_info.value).lower()
    )


def test_validate_config_dict_invalid_monotone_constraint():
    """Verify validation catches invalid monotone constraint values."""
    config = create_test_config()
    config["model"]["features"][0]["monotone_constraint"] = 5

    with pytest.raises(ValidationError) as exc_info:
        validate_config_dict(config)

    assert "monotone_constraint" in str(exc_info.value).lower()


def test_validate_config_dict_duplicate_feature_names():
    """Verify validation catches duplicate feature names."""
    config = create_test_config()
    config["model"]["features"].append({"name": "Level_Enc", "monotone_constraint": 0})

    with pytest.raises(ValidationError) as exc_info:
        validate_config_dict(config)

    assert "unique" in str(exc_info.value).lower() or "duplicate" in str(exc_info.value).lower()


def test_validate_config_dict_empty_config():
    """Verify validation rejects empty configuration."""
    with pytest.raises(ValidationError):
        validate_config_dict({})


def test_validate_config_dict_returns_validated_object():
    """Verify validation returns a Config Pydantic model."""
    config = create_test_config()
    validated = validate_config_dict(config)
    from src.model.config_schema_model import Config

    assert isinstance(validated, Config)
