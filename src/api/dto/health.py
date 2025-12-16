"""DTOs for health check endpoints."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class DependencyStatus(BaseModel):
    """Status of a single dependency."""

    name: str = Field(..., description="Dependency name")
    status: str = Field(..., description="Status: 'healthy' or 'unhealthy'")
    message: Optional[str] = Field(default=None, description="Status message")
    error: Optional[str] = Field(default=None, description="Error details if unhealthy")


class HealthStatusResponse(BaseModel):
    """Health check response with service and dependency status."""

    status: str = Field(..., description="Overall status: 'healthy' or 'unhealthy'")
    timestamp: datetime = Field(..., description="Check timestamp")
    service: str = Field(default="AutoQuantile API", description="Service name")
    version: str = Field(default="1.0.0", description="Service version")
    dependencies: List[DependencyStatus] = Field(
        default_factory=list, description="Dependency statuses"
    )


class ReadyStatusResponse(BaseModel):
    """Readiness check response."""

    status: str = Field(..., description="Readiness status: 'ready' or 'not_ready'")
    timestamp: datetime = Field(..., description="Check timestamp")
    service: str = Field(default="AutoQuantile API", description="Service name")
