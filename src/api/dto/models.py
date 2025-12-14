"""DTOs for model management endpoints."""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ModelMetadata(BaseModel):
    """Metadata about a trained model."""

    run_id: str = Field(..., description="MLflow run ID")
    start_time: datetime = Field(..., description="Training start time")
    model_type: str = Field(default="XGBoost", description="Model type")
    cv_mean_score: Optional[float] = Field(default=None, description="Cross-validation mean score")
    dataset_name: str = Field(..., description="Dataset name used for training")
    additional_tag: Optional[str] = Field(default=None, description="Additional tag/label")


class RankedFeatureSchema(BaseModel):
    """Schema for a ranked/categorical feature."""

    name: str = Field(..., description="Feature column name")
    levels: List[str] = Field(..., description="Valid categorical levels")
    encoding_type: str = Field(default="ranked", description="Encoding type")


class ProximityFeatureSchema(BaseModel):
    """Schema for a proximity-based feature (e.g., location)."""

    name: str = Field(..., description="Feature column name")
    encoding_type: str = Field(default="proximity", description="Encoding type")


class ModelSchema(BaseModel):
    """Complete model schema including all feature types."""

    ranked_features: List[RankedFeatureSchema] = Field(
        default_factory=list, description="Ranked/categorical features"
    )
    proximity_features: List[ProximityFeatureSchema] = Field(
        default_factory=list, description="Proximity-based features"
    )
    numerical_features: List[str] = Field(
        default_factory=list, description="Numerical feature names"
    )


class ModelSchemaResponse(BaseModel):
    """Response containing model schema."""

    run_id: str = Field(..., description="MLflow run ID")
    schema: ModelSchema = Field(..., description="Model schema")


class ModelDetailsResponse(BaseModel):
    """Complete model details including metadata and schema."""

    run_id: str = Field(..., description="MLflow run ID")
    metadata: ModelMetadata = Field(..., description="Model metadata")
    schema: ModelSchema = Field(..., description="Model schema")
    feature_names: List[str] = Field(..., description="All feature names")
    targets: List[str] = Field(..., description="Target column names")
    quantiles: List[float] = Field(
        ..., description="Prediction quantiles", min_length=1, max_length=20
    )

