"""Common DTOs for API responses and pagination."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class BaseResponse(BaseModel):
    """Base response model with status and optional message."""

    status: str = Field(default="success", description="Response status")
    message: Optional[str] = Field(default=None, description="Optional message")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Response data")


class ErrorDetail(BaseModel):
    """Error detail model."""

    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")


class ErrorResponse(BaseModel):
    """Error response model."""

    status: str = Field(default="error", description="Response status")
    error: ErrorDetail = Field(..., description="Error details")


class PaginationParams(BaseModel):
    """Pagination parameters for list endpoints."""

    limit: int = Field(default=50, ge=1, le=100, description="Maximum number of items to return")
    offset: int = Field(default=0, ge=0, description="Number of items to skip")


class PaginationResponse(BaseModel):
    """Pagination information in list responses."""

    total: int = Field(..., ge=0, description="Total number of items")
    limit: int = Field(..., ge=1, description="Limit used")
    offset: int = Field(..., ge=0, description="Offset used")
    has_more: bool = Field(..., description="Whether there are more items")
