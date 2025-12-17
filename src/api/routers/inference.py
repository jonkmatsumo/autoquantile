"""Inference/prediction API endpoints."""

from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, Request

from src.api.dependencies import get_current_user
from src.api.dto.inference import (
    BatchPredictionItem,
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.api.exceptions import InvalidInputError
from src.api.exceptions import ModelNotFoundError as APIModelNotFoundError
from src.api.rate_limiting import (
    BATCH_INFERENCE_CONCURRENCY,
    BATCH_INFERENCE_MAX_SIZE,
    BATCH_INFERENCE_TIMEOUT,
    INFERENCE_LIMIT,
    limiter,
)
from src.services.inference_service import InferenceService
from src.services.inference_service import InvalidInputError as ServiceInvalidInputError
from src.services.inference_service import ModelNotFoundError
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/models", tags=["inference"])


def get_inference_service() -> InferenceService:
    """Get inference service instance.

    Returns:
        InferenceService: Inference service.
    """
    return InferenceService()


@router.post("/{run_id}/predict", response_model=PredictionResponse)
@limiter.limit(INFERENCE_LIMIT)
async def predict(
    request: Request,
    run_id: str,
    prediction_request: PredictionRequest,
    user: str = Depends(get_current_user),
    inference_service: InferenceService = Depends(get_inference_service),
):
    """Predict salary quantiles for given features.

    Args:
        request (Request): FastAPI request object.
        run_id (str): MLflow run ID.
        prediction_request (PredictionRequest): Prediction request.
        user (str): Current user.
        inference_service (InferenceService): Inference service.

    Returns:
        PredictionResponse: Prediction results.

    Raises:
        APIModelNotFoundError: If model not found.
        InvalidInputError: If input validation fails.
    """
    try:
        model = inference_service.load_model(run_id)
        result = inference_service.predict(model, prediction_request.features)

        from src.api.dto.inference import PredictionMetadata

        return PredictionResponse(
            predictions=result.predictions,
            metadata=PredictionMetadata(
                model_run_id=run_id,
                prediction_timestamp=datetime.now(),
                location_zone=result.metadata.get("location_zone"),
            ),
        )
    except ModelNotFoundError as e:
        raise APIModelNotFoundError(run_id) from e
    except ServiceInvalidInputError as e:
        raise InvalidInputError(str(e)) from e


@router.post("/{run_id}/predict/batch", response_model=BatchPredictionResponse)
@limiter.limit(INFERENCE_LIMIT)
async def predict_batch(
    request: Request,
    run_id: str,
    batch_request: BatchPredictionRequest,
    user: str = Depends(get_current_user),
    inference_service: InferenceService = Depends(get_inference_service),
):
    """Batch predict salary quantiles for multiple feature sets.

    Args:
        request (Request): FastAPI request object.
        run_id (str): MLflow run ID.
        batch_request (BatchPredictionRequest): Batch prediction request.
        user (str): Current user.
        inference_service (InferenceService): Inference service.

    Returns:
        BatchPredictionResponse: Batch prediction results.

    Raises:
        APIModelNotFoundError: If model not found.
        InvalidInputError: If input validation fails.
    """
    try:
        if len(batch_request.features) > BATCH_INFERENCE_MAX_SIZE:
            raise InvalidInputError(
                f"Batch size {len(batch_request.features)} exceeds maximum of {BATCH_INFERENCE_MAX_SIZE}"
            )

        model = inference_service.load_model(run_id)

        concurrency = batch_request.concurrency or BATCH_INFERENCE_CONCURRENCY
        concurrency = min(concurrency, BATCH_INFERENCE_CONCURRENCY)

        batch_results = inference_service.predict_batch_parallel(
            model, batch_request.features, concurrency=concurrency, timeout=BATCH_INFERENCE_TIMEOUT
        )

        from src.api.dto.inference import PredictionMetadata

        predictions: List[PredictionResponse] = []
        items: List[BatchPredictionItem] = []
        success_count = 0
        failure_count = 0

        for index, result in batch_results:
            if isinstance(result, Exception):
                if isinstance(result, ServiceInvalidInputError):
                    status_code = 400
                else:
                    status_code = 500
                error_msg = str(result)
                items.append(
                    BatchPredictionItem(
                        prediction=None,
                        status_code=status_code,
                        error=error_msg,
                        index=index,
                    )
                )
                failure_count += 1
                logger.warning(f"Batch prediction failed for item {index}: {error_msg}")
            else:
                prediction_response = PredictionResponse(
                    predictions=result.predictions,
                    metadata=PredictionMetadata(
                        model_run_id=run_id,
                        prediction_timestamp=datetime.now(),
                        location_zone=result.metadata.get("location_zone"),
                    ),
                )
                predictions.append(prediction_response)
                items.append(
                    BatchPredictionItem(
                        prediction=prediction_response,
                        status_code=200,
                        error=None,
                        index=index,
                    )
                )
                success_count += 1

        total = len(batch_request.features)
        progress = 1.0

        return BatchPredictionResponse(
            predictions=predictions,
            items=items,
            total=total,
            success_count=success_count,
            failure_count=failure_count,
            progress=progress,
        )
    except ModelNotFoundError as e:
        raise APIModelNotFoundError(run_id) from e
    except ServiceInvalidInputError as e:
        raise InvalidInputError(str(e)) from e
