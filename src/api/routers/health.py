"""Health check API endpoints."""

import asyncio
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dto.health import DependencyStatus, HealthStatusResponse, ReadyStatusResponse
from src.services.model_registry import ModelRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


def get_model_registry() -> ModelRegistry:
    """Get model registry instance.

    Returns:
        ModelRegistry: Model registry.
    """
    return ModelRegistry()


async def check_mlflow_connectivity(
    registry: ModelRegistry, timeout_seconds: float = 2.0
) -> DependencyStatus:
    """Check MLflow connectivity with timeout.

    Args:
        registry (ModelRegistry): Model registry instance.
        timeout_seconds (float): Timeout in seconds.

    Returns:
        DependencyStatus: MLflow dependency status.
    """
    try:
        loop = asyncio.get_event_loop()
        await asyncio.wait_for(
            loop.run_in_executor(None, _check_mlflow_sync, registry),
            timeout=timeout_seconds,
        )
        return DependencyStatus(
            name="mlflow",
            status="healthy",
            message="MLflow connection successful",
        )
    except asyncio.TimeoutError:
        logger.warning("MLflow health check timed out")
        return DependencyStatus(
            name="mlflow",
            status="unhealthy",
            message="MLflow connection timeout",
            error="Connection check exceeded timeout",
        )
    except Exception as e:
        logger.error(f"MLflow health check failed: {type(e).__name__}: {e}")
        return DependencyStatus(
            name="mlflow",
            status="unhealthy",
            message="MLflow connection failed",
            error=f"{type(e).__name__}: {str(e)}",
        )


def _check_mlflow_sync(registry: ModelRegistry) -> None:
    """Synchronous MLflow connectivity check.

    Args:
        registry (ModelRegistry): Model registry instance.

    Raises:
        Exception: If MLflow is not accessible.
    """
    try:
        registry.client.search_experiments()
    except Exception:
        registry.list_models()


@router.get("/health", response_model=HealthStatusResponse, status_code=status.HTTP_200_OK)
async def health_check(
    registry: ModelRegistry = Depends(get_model_registry),
) -> HealthStatusResponse:
    """Health check endpoint with dependency verification.

    Args:
        registry (ModelRegistry): Model registry.

    Returns:
        HealthStatusResponse: Health status with dependencies.

    Raises:
        HTTPException: If service is unhealthy (503).
    """
    dependencies = []
    mlflow_status = await check_mlflow_connectivity(registry)
    dependencies.append(mlflow_status)

    all_healthy = all(dep.status == "healthy" for dep in dependencies)
    overall_status = "healthy" if all_healthy else "unhealthy"

    response = HealthStatusResponse(
        status=overall_status,
        timestamp=datetime.now(),
        dependencies=dependencies,
    )

    if not all_healthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=response.model_dump(mode="json"),
        )

    return response


@router.get("/ready", response_model=ReadyStatusResponse, status_code=status.HTTP_200_OK)
async def readiness_check() -> ReadyStatusResponse:
    """Readiness check endpoint for load balancer integration.

    Returns:
        ReadyStatusResponse: Readiness status.

    Raises:
        HTTPException: If service is not ready (503).
    """
    response = ReadyStatusResponse(
        status="ready",
        timestamp=datetime.now(),
    )

    return response
