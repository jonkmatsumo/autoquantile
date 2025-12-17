"""DTOs for inference/prediction endpoints."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request for a single prediction."""

    features: Dict[str, Any] = Field(..., description="Feature name to value mapping", min_length=1)


class PredictionMetadata(BaseModel):
    """Metadata about a prediction."""

    model_run_id: str = Field(..., description="MLflow run ID of the model used")
    prediction_timestamp: datetime = Field(
        default_factory=datetime.now, description="When the prediction was made"
    )
    location_zone: Optional[str] = Field(default=None, description="Location zone if applicable")


class PredictionResponse(BaseModel):
    """Response containing prediction results."""

    predictions: Dict[str, Dict[str, float]] = Field(
        ...,
        description="Target name -> {quantile_key: value} mapping (e.g., {'BaseSalary': {'p10': 120000.0, 'p50': 150000.0}})",
    )
    metadata: PredictionMetadata = Field(..., description="Prediction metadata")


class BatchPredictionItem(BaseModel):
    """Individual item in batch prediction response."""

    prediction: Optional[PredictionResponse] = Field(
        default=None, description="Prediction result if successful"
    )
    status_code: int = Field(..., description="HTTP status code for this item")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    index: int = Field(..., description="Original request index", ge=0)


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""

    features: List[Dict[str, Any]] = Field(
        ..., description="List of feature dictionaries", min_length=1, max_length=1000
    )
    concurrency: Optional[int] = Field(
        default=None, description="Number of concurrent predictions", ge=1, le=100
    )


class BatchPredictionResponse(BaseModel):
    """Response containing batch prediction results."""

    predictions: List[PredictionResponse] = Field(
        ..., description="List of prediction responses (backward compatibility)"
    )
    items: List[BatchPredictionItem] = Field(
        ..., description="List of batch prediction items with individual status codes"
    )
    total: int = Field(..., description="Total number of predictions", ge=0)
    success_count: int = Field(..., description="Number of successful predictions", ge=0)
    failure_count: int = Field(..., description="Number of failed predictions", ge=0)
    progress: Optional[float] = Field(
        default=None, description="Progress indicator (0.0-1.0)", ge=0.0, le=1.0
    )
