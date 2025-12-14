"""DTOs for configuration workflow endpoints."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class WorkflowStartRequest(BaseModel):
    """Request to start a configuration workflow."""

    data: str = Field(..., description="JSON string of DataFrame (records orient)", min_length=1)
    columns: List[str] = Field(..., description="List of column names", min_length=1)
    dtypes: Dict[str, str] = Field(
        ..., description="Dictionary mapping column names to data types", min_length=1
    )
    dataset_size: int = Field(..., ge=1, description="Total number of rows in the dataset")
    provider: str = Field(default="openai", description="LLM provider name")
    preset: Optional[str] = Field(
        default=None, max_length=100, description="Optional preset prompt name"
    )


class WorkflowState(BaseModel):
    """Workflow state information."""

    phase: str = Field(..., description="Current workflow phase")
    status: Literal["success", "error", "pending"] = Field(
        ..., description="Workflow status"
    )
    current_result: Optional[Dict[str, Any]] = Field(
        default=None, description="Current phase result data"
    )


class WorkflowStartResponse(BaseModel):
    """Response after starting a workflow."""

    workflow_id: str = Field(..., description="Unique workflow identifier")
    phase: Literal["classification", "encoding", "configuration", "complete"] = Field(
        ..., description="Current workflow phase"
    )
    state: WorkflowState = Field(..., description="Current workflow state")


class WorkflowStateResponse(BaseModel):
    """Response containing current workflow state."""

    workflow_id: str = Field(..., description="Unique workflow identifier")
    phase: str = Field(..., description="Current workflow phase")
    state: Dict[str, Any] = Field(..., description="Complete workflow state dictionary")
    current_result: Dict[str, Any] = Field(..., description="Current phase result")


class ClassificationModifications(BaseModel):
    """Modifications to column classification."""

    targets: List[str] = Field(default_factory=list, description="List of target column names")
    features: List[str] = Field(default_factory=list, description="List of feature column names")
    ignore: List[str] = Field(default_factory=list, description="List of columns to ignore")


class ClassificationConfirmationRequest(BaseModel):
    """Request to confirm classification phase."""

    modifications: ClassificationModifications = Field(
        ..., description="Modified classification"
    )


class EncodingConfig(BaseModel):
    """Configuration for a single feature encoding."""

    type: Literal["numeric", "ordinal", "onehot", "proximity", "label"] = Field(
        ..., description="Encoding type"
    )
    mapping: Optional[Dict[str, Any]] = Field(
        default=None, description="Mapping for ordinal encoding (value -> rank)"
    )
    reasoning: Optional[str] = Field(default=None, description="Reasoning for encoding choice")


class OptionalEncodingConfig(BaseModel):
    """Configuration for optional encoding (e.g., cost_of_living, normalize_recent)."""

    type: Literal[
        "cost_of_living",
        "metro_population",
        "normalize_recent",
        "weight_recent",
        "least_recent",
    ] = Field(..., description="Optional encoding type")
    params: Dict[str, Any] = Field(default_factory=dict, description="Optional parameters")


class EncodingModifications(BaseModel):
    """Modifications to feature encodings."""

    encodings: Dict[str, EncodingConfig] = Field(
        ..., description="Dictionary of column name to encoding configuration"
    )
    optional_encodings: Dict[str, OptionalEncodingConfig] = Field(
        default_factory=dict,
        description="Dictionary of optional encodings (e.g., cost_of_living for locations)",
    )


class EncodingConfirmationRequest(BaseModel):
    """Request to confirm encoding phase."""

    modifications: EncodingModifications = Field(..., description="Modified encodings")


class FeatureConfig(BaseModel):
    """Configuration for a model feature."""

    name: str = Field(..., description="Feature name", min_length=1)
    monotone_constraint: Literal[-1, 0, 1] = Field(
        ..., description="Monotonic constraint: 1 (increasing), 0 (none), -1 (decreasing)"
    )


class Hyperparameters(BaseModel):
    """Model hyperparameters."""

    training: Dict[str, Any] = Field(..., description="Training hyperparameters")
    cv: Dict[str, Any] = Field(..., description="Cross-validation hyperparameters")


class ConfigurationFinalizationRequest(BaseModel):
    """Request to finalize configuration."""

    features: List[FeatureConfig] = Field(
        ..., description="Feature configurations", min_length=1
    )
    quantiles: List[float] = Field(
        ...,
        description="Prediction quantiles",
        min_length=1,
        max_length=20,
    )
    hyperparameters: Hyperparameters = Field(..., description="Model hyperparameters")
    location_settings: Optional[Dict[str, Any]] = Field(
        default=None, description="Location proximity settings"
    )

    @field_validator("quantiles")
    @classmethod
    def validate_quantiles_range(cls, v: List[float]) -> List[float]:
        """Validate that quantiles are between 0 and 1. Returns: List[float]: Validated quantiles."""
        for q in v:
            if not 0 <= q <= 1:
                raise ValueError(f"Quantiles must be between 0 and 1, got {q}")
        return v


class WorkflowProgressResponse(BaseModel):
    """Response after progressing workflow to next phase."""

    workflow_id: str = Field(..., description="Unique workflow identifier")
    phase: str = Field(..., description="New workflow phase")
    result: Dict[str, Any] = Field(..., description="Phase result data")


class WorkflowCompleteResponse(BaseModel):
    """Response when workflow is complete."""

    workflow_id: str = Field(..., description="Unique workflow identifier")
    phase: Literal["complete"] = Field(default="complete", description="Workflow phase")
    final_config: Dict[str, Any] = Field(..., description="Final configuration dictionary")

