"""Rate limiting configuration and utilities for API endpoints."""

from typing import Optional

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.responses import Response

from src.utils.env_loader import get_env_var
from src.utils.logger import get_logger

logger = get_logger(__name__)

_rate_limit_enabled_str = get_env_var("RATE_LIMIT_ENABLED", "true")
RATE_LIMIT_ENABLED = _rate_limit_enabled_str.lower() == "true" if _rate_limit_enabled_str else False
RATE_LIMIT_STORAGE_URL = get_env_var("RATE_LIMIT_STORAGE_URL")

INFERENCE_LIMIT = get_env_var("RATE_LIMIT_INFERENCE", "100/minute") or "100/minute"
TRAINING_LIMIT = get_env_var("RATE_LIMIT_TRAINING", "10/minute") or "10/minute"
ANALYTICS_LIMIT = get_env_var("RATE_LIMIT_ANALYTICS", "50/minute") or "50/minute"
MODELS_LIMIT = get_env_var("RATE_LIMIT_MODELS", "50/minute") or "50/minute"
WORKFLOW_LIMIT = get_env_var("RATE_LIMIT_WORKFLOW", "20/minute") or "20/minute"

BATCH_INFERENCE_CONCURRENCY = int(get_env_var("BATCH_INFERENCE_CONCURRENCY", "10") or "10")
BATCH_INFERENCE_MAX_SIZE = int(get_env_var("BATCH_INFERENCE_MAX_SIZE", "1000") or "1000")
BATCH_INFERENCE_TIMEOUT = int(get_env_var("BATCH_INFERENCE_TIMEOUT", "300") or "300")


def get_rate_limit_key(request: Request) -> str:
    """Get rate limit key from request (API key or IP address).

    Args:
        request (Request): FastAPI request object.

    Returns:
        str: Rate limit key identifier.
    """
    api_key: Optional[str] = None

    authorization = request.headers.get("Authorization")
    if authorization and authorization.startswith("Bearer "):
        api_key = authorization.replace("Bearer ", "")

    if not api_key:
        api_key = request.headers.get("X-API-Key")

    if api_key:
        return f"api_key:{api_key}"

    return f"ip:{get_remote_address(request)}"


if RATE_LIMIT_STORAGE_URL:
    limiter = Limiter(
        key_func=get_rate_limit_key,
        storage_uri=RATE_LIMIT_STORAGE_URL,
        default_limits=["1000/hour"],
        headers_enabled=True,
        enabled=RATE_LIMIT_ENABLED,
    )
else:
    limiter = Limiter(
        key_func=get_rate_limit_key,
        default_limits=["1000/hour"],
        headers_enabled=True,
        enabled=RATE_LIMIT_ENABLED,
    )


_original_inject_headers = limiter._inject_headers


def _patched_inject_headers(
    self, response: Optional[Response], current_limit
) -> Optional[Response]:
    """Patched version of _inject_headers that handles None responses.

    Args:
        response (Optional[Response]): Response object or None.
        current_limit: Current rate limit information.

    Returns:
        Optional[Response]: Response with headers injected, or None if response was None.
    """
    if response is None or current_limit is None:
        return response
    return _original_inject_headers(response, current_limit)


limiter._inject_headers = _patched_inject_headers.__get__(limiter, Limiter)
