"""Integration tests for health check endpoints."""

from unittest.mock import MagicMock, patch

from fastapi import status


def test_health_check_healthy(client_no_auth):
    """Test health check endpoint when all dependencies are healthy. Args: client_no_auth: Test client without auth."""
    with patch("src.api.routers.health.ModelRegistry") as mock_registry_class:
        mock_registry = MagicMock()
        mock_client = MagicMock()
        mock_client.search_experiments.return_value = []
        mock_registry.client = mock_client
        mock_registry_class.return_value = mock_registry

        response = client_no_auth.get("/health")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "AutoQuantile API"
        assert data["version"] == "1.0.0"
        assert "timestamp" in data
        assert len(data["dependencies"]) == 1
        assert data["dependencies"][0]["name"] == "mlflow"
        assert data["dependencies"][0]["status"] == "healthy"


def test_health_check_unhealthy_mlflow(client_no_auth):
    """Test health check endpoint when MLflow is unavailable. Args: client_no_auth: Test client without auth."""
    with patch("src.api.routers.health.ModelRegistry") as mock_registry_class:
        mock_registry = MagicMock()
        mock_client = MagicMock()
        mock_client.search_experiments.side_effect = Exception("MLflow connection failed")
        mock_registry.client = mock_client
        mock_registry.list_models.side_effect = Exception("MLflow connection failed")
        mock_registry_class.return_value = mock_registry

        response = client_no_auth.get("/health")
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()["detail"]
        assert data["status"] == "unhealthy"
        assert len(data["dependencies"]) == 1
        assert data["dependencies"][0]["name"] == "mlflow"
        assert data["dependencies"][0]["status"] == "unhealthy"
        assert "error" in data["dependencies"][0]


def test_health_check_no_auth(client_no_auth):
    """Test health check works without authentication. Args: client_no_auth: Test client without auth."""
    with patch("src.api.routers.health.ModelRegistry") as mock_registry_class:
        mock_registry = MagicMock()
        mock_client = MagicMock()
        mock_client.search_experiments.return_value = []
        mock_registry.client = mock_client
        mock_registry_class.return_value = mock_registry

        response = client_no_auth.get("/health")
        assert response.status_code == status.HTTP_200_OK


def test_health_check_with_auth(client):
    """Test health check works with authentication. Args: client: Test client."""
    with patch("src.api.routers.health.ModelRegistry") as mock_registry_class:
        mock_registry = MagicMock()
        mock_client = MagicMock()
        mock_client.search_experiments.return_value = []
        mock_registry.client = mock_client
        mock_registry_class.return_value = mock_registry

        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK


def test_ready_check(client_no_auth):
    """Test readiness check endpoint. Args: client_no_auth: Test client without auth."""
    response = client_no_auth.get("/ready")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "ready"
    assert data["service"] == "AutoQuantile API"
    assert "timestamp" in data


def test_ready_check_no_auth(client_no_auth):
    """Test readiness check works without authentication. Args: client_no_auth: Test client without auth."""
    response = client_no_auth.get("/ready")
    assert response.status_code == status.HTTP_200_OK


def test_health_check_timeout(client_no_auth):
    """Test health check handles MLflow timeout gracefully. Args: client_no_auth: Test client without auth."""
    from src.api.dto.health import DependencyStatus

    with patch("src.api.routers.health.check_mlflow_connectivity") as mock_check:
        mock_check.return_value = DependencyStatus(
            name="mlflow",
            status="unhealthy",
            message="MLflow connection timeout",
            error="Connection check exceeded timeout",
        )

        response = client_no_auth.get("/health")
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()["detail"]
        assert data["status"] == "unhealthy"
        assert data["dependencies"][0]["status"] == "unhealthy"
