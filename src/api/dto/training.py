"""DTOs for training endpoints."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from src.api.dto.analytics import DataSummary


class DataUploadResponse(BaseModel):
    """Response after uploading training data."""

    dataset_id: str = Field(..., description="Unique dataset identifier")
    row_count: int = Field(..., ge=1, description="Number of rows in the dataset")
    column_count: int = Field(..., ge=1, description="Number of columns in the dataset")
    summary: DataSummary = Field(..., description="Data summary statistics")


class TrainingJobRequest(BaseModel):
    """Request to start a training job."""

    dataset_id: str = Field(..., description="Dataset identifier from upload")
    config: Dict[str, Any] = Field(..., description="Model configuration dictionary")
    remove_outliers: bool = Field(default=True, description="Whether to remove outliers using IQR")
    do_tune: bool = Field(default=False, description="Whether to run hyperparameter tuning")
    n_trials: Optional[int] = Field(
        default=None, ge=1, le=500, description="Number of tuning trials (required if do_tune=True)"
    )
    additional_tag: Optional[str] = Field(
        default=None, max_length=200, description="Optional tag/label for the model"
    )
    dataset_name: Optional[str] = Field(
        default=None, max_length=200, description="Optional dataset name"
    )

    @field_validator("n_trials")
    @classmethod
    def validate_n_trials(cls, v: Optional[int], info) -> Optional[int]:
        """Validate that n_trials is provided when do_tune is True. Returns: Optional[int]: Validated n_trials."""
        if info.data.get("do_tune") and v is None:
            raise ValueError("n_trials is required when do_tune=True")
        return v


class TrainingJobResponse(BaseModel):
    """Response after starting a training job."""

    job_id: str = Field(..., description="Training job identifier")
    status: Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED"] = Field(
        default="QUEUED", description="Job status"
    )
    created_at: datetime = Field(default_factory=datetime.now, description="Job creation time")


class TrainingResult(BaseModel):
    """Result of a completed training job."""

    run_id: str = Field(..., description="MLflow run ID")
    model_type: str = Field(default="XGBoost", description="Model type")
    cv_mean_score: Optional[float] = Field(
        default=None, description="Cross-validation mean score"
    )


class TrainingJobStatusResponse(BaseModel):
    """Status response for a training job."""

    job_id: str = Field(..., description="Training job identifier")
    status: Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED"] = Field(
        ..., description="Current job status"
    )
    progress: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Training progress (0.0-1.0)"
    )
    logs: List[str] = Field(default_factory=list, description="Training logs")
    submitted_at: Optional[datetime] = Field(default=None, description="Job submission time")
    completed_at: Optional[datetime] = Field(default=None, description="Job completion time")
    result: Optional[TrainingResult] = Field(
        default=None, description="Training result (if completed)"
    )
    error: Optional[str] = Field(default=None, description="Error message (if failed)")
    run_id: Optional[str] = Field(
        default=None, description="MLflow run ID (if completed)"
    )


class TrainingJobSummary(BaseModel):
    """Summary of a training job for list endpoints."""

    job_id: str = Field(..., description="Training job identifier")
    status: Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED"] = Field(
        ..., description="Job status"
    )
    submitted_at: datetime = Field(..., description="Job submission time")
    completed_at: Optional[datetime] = Field(default=None, description="Job completion time")
    run_id: Optional[str] = Field(default=None, description="MLflow run ID if completed")

