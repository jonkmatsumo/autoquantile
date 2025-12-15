"""Data Transfer Objects (DTOs) for API requests and responses."""

from src.api.dto.common import (
    BaseResponse,
    ErrorDetail,
    ErrorResponse,
    PaginationParams,
    PaginationResponse,
)
from src.api.dto.models import (
    ModelDetailsResponse,
    ModelMetadata,
    ModelSchema,
    ModelSchemaResponse,
    ProximityFeatureSchema,
    RankedFeatureSchema,
)
from src.api.dto.inference import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionMetadata,
    PredictionRequest,
    PredictionResponse,
)
from src.api.dto.training import (
    DataUploadResponse,
    TrainingJobRequest,
    TrainingJobResponse,
    TrainingJobStatusResponse,
    TrainingJobSummary,
)
from src.api.dto.workflow import (
    ClassificationConfirmationRequest,
    ClassificationModifications,
    ConfigurationFinalizationRequest,
    EncodingConfirmationRequest,
    EncodingModifications,
    WorkflowCompleteResponse,
    WorkflowProgressResponse,
    WorkflowStartRequest,
    WorkflowStartResponse,
    WorkflowStateResponse,
)
from src.api.dto.analytics import (
    DataSummaryRequest,
    DataSummaryResponse,
    FeatureImportance,
    FeatureImportanceResponse,
)

__all__ = [
    # Common
    "BaseResponse",
    "ErrorDetail",
    "ErrorResponse",
    "PaginationParams",
    "PaginationResponse",
    # Models
    "ModelMetadata",
    "RankedFeatureSchema",
    "ProximityFeatureSchema",
    "ModelSchema",
    "ModelSchemaResponse",
    "ModelDetailsResponse",
    # Inference
    "PredictionRequest",
    "PredictionResponse",
    "PredictionMetadata",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    # Training
    "DataUploadResponse",
    "TrainingJobRequest",
    "TrainingJobResponse",
    "TrainingJobStatusResponse",
    "TrainingJobSummary",
    # Workflow
    "WorkflowStartRequest",
    "WorkflowStartResponse",
    "WorkflowStateResponse",
    "ClassificationModifications",
    "ClassificationConfirmationRequest",
    "WorkflowProgressResponse",
    "EncodingModifications",
    "EncodingConfirmationRequest",
    "ConfigurationFinalizationRequest",
    "WorkflowCompleteResponse",
    # Analytics
    "DataSummaryRequest",
    "DataSummaryResponse",
    "FeatureImportance",
    "FeatureImportanceResponse",
]
