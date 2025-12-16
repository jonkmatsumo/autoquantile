"""Unit tests for health check router."""

import asyncio
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, status

from src.api.dto.health import DependencyStatus, HealthStatusResponse, ReadyStatusResponse
from src.api.routers.health import (
    _check_mlflow_sync,
    check_mlflow_connectivity,
    get_model_registry,
    health_check,
    readiness_check,
)
from src.services.model_registry import ModelRegistry


class TestCheckMlflowSync:
    """Tests for _check_mlflow_sync function."""

    def test_check_mlflow_sync_success_via_search_experiments(self):
        """Test successful MLflow check via search_experiments."""
        registry = MagicMock()
        registry.client = MagicMock()
        registry.client.search_experiments.return_value = []

        _check_mlflow_sync(registry)

        registry.client.search_experiments.assert_called_once()

    def test_check_mlflow_sync_success_via_list_models(self):
        """Test successful MLflow check via list_models fallback."""
        registry = MagicMock()
        registry.client = MagicMock()
        registry.client.search_experiments.side_effect = Exception("Search failed")
        registry.list_models.return_value = []

        _check_mlflow_sync(registry)

        registry.client.search_experiments.assert_called_once()
        registry.list_models.assert_called_once()

    def test_check_mlflow_sync_failure(self):
        """Test MLflow check failure."""
        registry = MagicMock()
        registry.client = MagicMock()
        registry.client.search_experiments.side_effect = Exception("Connection failed")
        registry.list_models.side_effect = Exception("Connection failed")

        with pytest.raises(Exception, match="Connection failed"):
            _check_mlflow_sync(registry)


class TestCheckMlflowConnectivity:
    """Tests for check_mlflow_connectivity async function."""

    def test_check_mlflow_connectivity_success(self):
        """Test successful MLflow connectivity check."""
        registry = MagicMock()
        registry.client = MagicMock()
        registry.client.search_experiments.return_value = []

        result = asyncio.run(check_mlflow_connectivity(registry, timeout_seconds=1.0))

        assert isinstance(result, DependencyStatus)
        assert result.name == "mlflow"
        assert result.status == "healthy"
        assert result.message == "MLflow connection successful"
        assert result.error is None

    def test_check_mlflow_connectivity_timeout(self):
        """Test MLflow connectivity check timeout."""
        registry = MagicMock()

        with patch("src.api.routers.health.asyncio.wait_for") as mock_wait:
            mock_wait.side_effect = asyncio.TimeoutError()

            result = asyncio.run(check_mlflow_connectivity(registry, timeout_seconds=0.1))

        assert isinstance(result, DependencyStatus)
        assert result.name == "mlflow"
        assert result.status == "unhealthy"
        assert "timeout" in result.message.lower()
        assert result.error is not None

    def test_check_mlflow_connectivity_exception(self):
        """Test MLflow connectivity check with exception."""
        registry = MagicMock()
        registry.client = MagicMock()
        registry.client.search_experiments.side_effect = Exception("Connection error")
        registry.list_models = MagicMock()
        registry.list_models.side_effect = Exception("Connection error")

        result = asyncio.run(check_mlflow_connectivity(registry, timeout_seconds=1.0))

        assert isinstance(result, DependencyStatus)
        assert result.name == "mlflow"
        assert result.status == "unhealthy"
        assert result.message == "MLflow connection failed"
        assert "Exception" in result.error or "Connection error" in result.error


class TestHealthCheck:
    """Tests for health_check endpoint."""

    def test_health_check_healthy(self):
        """Test health check when all dependencies are healthy."""
        registry = MagicMock()

        with patch("src.api.routers.health.check_mlflow_connectivity") as mock_check:
            mock_check.return_value = DependencyStatus(
                name="mlflow",
                status="healthy",
                message="MLflow connection successful",
            )

            result = asyncio.run(health_check(registry=registry))

        assert isinstance(result, HealthStatusResponse)
        assert result.status == "healthy"
        assert result.service == "AutoQuantile API"
        assert result.version == "1.0.0"
        assert isinstance(result.timestamp, datetime)
        assert len(result.dependencies) == 1
        assert result.dependencies[0].status == "healthy"

    def test_health_check_unhealthy(self):
        """Test health check when dependencies are unhealthy."""
        registry = MagicMock()

        with patch("src.api.routers.health.check_mlflow_connectivity") as mock_check:
            mock_check.return_value = DependencyStatus(
                name="mlflow",
                status="unhealthy",
                message="MLflow connection failed",
                error="Connection timeout",
            )

            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(health_check(registry=registry))

        assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        detail = exc_info.value.detail
        assert detail["status"] == "unhealthy"
        assert detail["dependencies"][0]["status"] == "unhealthy"


class TestReadinessCheck:
    """Tests for readiness_check endpoint."""

    def test_readiness_check(self):
        """Test readiness check endpoint."""
        result = asyncio.run(readiness_check())

        assert isinstance(result, ReadyStatusResponse)
        assert result.status == "ready"
        assert result.service == "AutoQuantile API"
        assert isinstance(result.timestamp, datetime)


class TestGetModelRegistry:
    """Tests for get_model_registry dependency."""

    def test_get_model_registry(self):
        """Test model registry dependency function."""
        registry = get_model_registry()

        assert isinstance(registry, ModelRegistry)
