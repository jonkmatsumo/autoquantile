"""DTOs for analytics endpoints."""

from typing import Dict, List, Tuple

from pydantic import BaseModel, Field


class DataSummary(BaseModel):
    """Summary statistics for a dataset."""

    total_samples: int = Field(..., ge=0, description="Total number of samples")
    shape: Tuple[int, int] = Field(..., description="Dataset shape (rows, columns)")
    unique_counts: Dict[str, int] = Field(
        default_factory=dict, description="Unique value counts by column"
    )


class DataSummaryRequest(BaseModel):
    """Request for data summary."""

    data: str = Field(..., description="JSON string of DataFrame (records orient)", min_length=1)


class DataSummaryResponse(BaseModel):
    """Response containing data summary."""

    total_samples: int = Field(..., ge=0, description="Total number of samples")
    shape: Tuple[int, int] = Field(..., description="Dataset shape (rows, columns)")
    unique_counts: Dict[str, int] = Field(
        default_factory=dict, description="Unique value counts by column"
    )


class FeatureImportance(BaseModel):
    """Feature importance information."""

    name: str = Field(..., description="Feature name")
    gain: float = Field(..., ge=0.0, description="Feature importance gain score")


class FeatureImportanceResponse(BaseModel):
    """Response containing feature importance."""

    features: List[FeatureImportance] = Field(
        ..., description="List of features with importance scores"
    )
