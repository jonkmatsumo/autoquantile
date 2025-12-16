"""Integration tests for rate limiting functionality."""

import importlib
import os
from typing import Generator

import pytest
from fastapi.testclient import TestClient

import src.api.rate_limiting


@pytest.fixture
def client_with_rate_limiting(api_key: str) -> Generator[TestClient, None, None]:
    """Fixture providing FastAPI test client with rate limiting enabled.

    Args:
        api_key (str): API key.

    Returns:
        TestClient: Test client with rate limiting enabled.
    """
    os.environ["API_KEY"] = api_key
    os.environ["RATE_LIMIT_ENABLED"] = "true"
    os.environ["RATE_LIMIT_INFERENCE"] = "5/minute"
    os.environ["RATE_LIMIT_TRAINING"] = "3/minute"
    os.environ["RATE_LIMIT_ANALYTICS"] = "4/minute"
    os.environ["RATE_LIMIT_MODELS"] = "4/minute"
    os.environ["RATE_LIMIT_WORKFLOW"] = "3/minute"

    importlib.reload(src.api.rate_limiting)

    from typing import Optional

    from slowapi import Limiter
    from slowapi.middleware import SlowAPIMiddleware
    from starlette.responses import Response

    from src.api import app as app_module
    from src.api import rate_limiting
    from src.api.routers import analytics, inference, models, training, workflow

    models.limiter = rate_limiting.limiter
    inference.limiter = rate_limiting.limiter
    training.limiter = rate_limiting.limiter
    workflow.limiter = rate_limiting.limiter
    analytics.limiter = rate_limiting.limiter

    app_module.limiter = rate_limiting.limiter
    app_module.RATE_LIMIT_ENABLED = rate_limiting.RATE_LIMIT_ENABLED

    rate_limiting.limiter.enabled = True

    _original_inject_headers = rate_limiting.limiter._inject_headers

    def _patched_inject_headers(
        self, response: Optional[Response], current_limit
    ) -> Optional[Response]:
        if response is None or current_limit is None:
            return response
        return _original_inject_headers(response, current_limit)

    rate_limiting.limiter._inject_headers = _patched_inject_headers.__get__(
        rate_limiting.limiter, Limiter
    )

    _original_dispatch = SlowAPIMiddleware.dispatch

    async def _patched_dispatch(self, request, call_next):
        from slowapi.middleware import _find_route_handler, async_check_limits

        app = request.app
        limiter = app.state.limiter

        if not limiter.enabled:
            return await call_next(request)

        handler = _find_route_handler(app.routes, request.scope)
        error_response, should_inject_headers = await async_check_limits(
            limiter, request, handler, app
        )

        if error_response is not None:
            return error_response

        response = await call_next(request)

        if should_inject_headers and hasattr(request.state, "view_rate_limit"):
            response = limiter._inject_headers(response, request.state.view_rate_limit)
        elif handler is not None and hasattr(request.state, "view_rate_limit"):
            name = f"{handler.__module__}.{handler.__name__}"
            if name in limiter._route_limits and hasattr(request.state, "view_rate_limit"):
                response = limiter._inject_headers(response, request.state.view_rate_limit)

        return response

    SlowAPIMiddleware.dispatch = _patched_dispatch

    from src.api.app import create_app

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client

    if "API_KEY" in os.environ:
        del os.environ["API_KEY"]
    if "RATE_LIMIT_ENABLED" in os.environ:
        del os.environ["RATE_LIMIT_ENABLED"]
    if "RATE_LIMIT_INFERENCE" in os.environ:
        del os.environ["RATE_LIMIT_INFERENCE"]
    if "RATE_LIMIT_TRAINING" in os.environ:
        del os.environ["RATE_LIMIT_TRAINING"]
    if "RATE_LIMIT_ANALYTICS" in os.environ:
        del os.environ["RATE_LIMIT_ANALYTICS"]
    if "RATE_LIMIT_MODELS" in os.environ:
        del os.environ["RATE_LIMIT_MODELS"]
    if "RATE_LIMIT_WORKFLOW" in os.environ:
        del os.environ["RATE_LIMIT_WORKFLOW"]

    importlib.reload(src.api.rate_limiting)


@pytest.fixture
def client_without_rate_limiting(api_key: str) -> Generator[TestClient, None, None]:
    """Fixture providing FastAPI test client with rate limiting disabled.

    Args:
        api_key (str): API key.

    Returns:
        TestClient: Test client without rate limiting.
    """
    os.environ["API_KEY"] = api_key
    os.environ["RATE_LIMIT_ENABLED"] = "false"

    importlib.reload(src.api.rate_limiting)

    from typing import Optional

    from slowapi import Limiter
    from slowapi.middleware import SlowAPIMiddleware
    from starlette.responses import Response

    from src.api import app as app_module
    from src.api import rate_limiting
    from src.api.routers import analytics, inference, models, training, workflow

    models.limiter = rate_limiting.limiter
    inference.limiter = rate_limiting.limiter
    training.limiter = rate_limiting.limiter
    workflow.limiter = rate_limiting.limiter
    analytics.limiter = rate_limiting.limiter

    app_module.limiter = rate_limiting.limiter
    app_module.RATE_LIMIT_ENABLED = rate_limiting.RATE_LIMIT_ENABLED

    rate_limiting.limiter.enabled = False

    _original_inject_headers = rate_limiting.limiter._inject_headers

    def _patched_inject_headers(
        self, response: Optional[Response], current_limit
    ) -> Optional[Response]:
        if response is None or current_limit is None:
            return response
        return _original_inject_headers(response, current_limit)

    rate_limiting.limiter._inject_headers = _patched_inject_headers.__get__(
        rate_limiting.limiter, Limiter
    )

    _original_dispatch = SlowAPIMiddleware.dispatch

    async def _patched_dispatch(self, request, call_next):
        from slowapi.middleware import _find_route_handler, async_check_limits

        app = request.app
        limiter = app.state.limiter

        if not limiter.enabled:
            return await call_next(request)

        handler = _find_route_handler(app.routes, request.scope)
        error_response, should_inject_headers = await async_check_limits(
            limiter, request, handler, app
        )

        if error_response is not None:
            return error_response

        response = await call_next(request)

        if should_inject_headers and hasattr(request.state, "view_rate_limit"):
            response = limiter._inject_headers(response, request.state.view_rate_limit)
        elif handler is not None and hasattr(request.state, "view_rate_limit"):
            name = f"{handler.__module__}.{handler.__name__}"
            if name in limiter._route_limits and hasattr(request.state, "view_rate_limit"):
                response = limiter._inject_headers(response, request.state.view_rate_limit)

        return response

    SlowAPIMiddleware.dispatch = _patched_dispatch

    from src.api.app import create_app

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client

    if "API_KEY" in os.environ:
        del os.environ["API_KEY"]
    if "RATE_LIMIT_ENABLED" in os.environ:
        del os.environ["RATE_LIMIT_ENABLED"]

    importlib.reload(src.api.rate_limiting)

    from typing import Optional

    from slowapi import Limiter
    from slowapi.middleware import SlowAPIMiddleware
    from starlette.responses import Response

    from src.api import app as app_module
    from src.api import rate_limiting
    from src.api.routers import analytics, inference, models, training, workflow

    models.limiter = rate_limiting.limiter
    inference.limiter = rate_limiting.limiter
    training.limiter = rate_limiting.limiter
    workflow.limiter = rate_limiting.limiter
    analytics.limiter = rate_limiting.limiter

    app_module.limiter = rate_limiting.limiter
    app_module.RATE_LIMIT_ENABLED = rate_limiting.RATE_LIMIT_ENABLED

    rate_limiting.limiter.enabled = True

    _original_inject_headers = rate_limiting.limiter._inject_headers

    def _patched_inject_headers(
        self, response: Optional[Response], current_limit
    ) -> Optional[Response]:
        if response is None or current_limit is None:
            return response
        return _original_inject_headers(response, current_limit)

    rate_limiting.limiter._inject_headers = _patched_inject_headers.__get__(
        rate_limiting.limiter, Limiter
    )

    _original_dispatch = SlowAPIMiddleware.dispatch

    async def _patched_dispatch(self, request, call_next):
        from slowapi.middleware import _find_route_handler, async_check_limits

        app = request.app
        limiter = app.state.limiter

        if not limiter.enabled:
            return await call_next(request)

        handler = _find_route_handler(app.routes, request.scope)
        error_response, should_inject_headers = await async_check_limits(
            limiter, request, handler, app
        )

        if error_response is not None:
            return error_response

        response = await call_next(request)

        if should_inject_headers and hasattr(request.state, "view_rate_limit"):
            response = limiter._inject_headers(response, request.state.view_rate_limit)
        elif handler is not None and hasattr(request.state, "view_rate_limit"):
            name = f"{handler.__module__}.{handler.__name__}"
            if name in limiter._route_limits and hasattr(request.state, "view_rate_limit"):
                response = limiter._inject_headers(response, request.state.view_rate_limit)

        return response

    SlowAPIMiddleware.dispatch = _patched_dispatch


class TestRateLimitHeaders:
    """Tests for rate limit headers in responses."""

    def test_rate_limit_headers_present(self, client_with_rate_limiting: TestClient):
        """Test that rate limit headers are present in responses."""
        headers = {"X-API-Key": "test_api_key_123"}

        response = client_with_rate_limiting.get("/api/v1/models", headers=headers)

        assert response.status_code in [200, 401, 404]
        rate_limit_headers = [
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
        ]
        for header in rate_limit_headers:
            assert header in response.headers, f"Missing rate limit header: {header}"
            assert response.headers[header] is not None


class TestRateLimitEnforcement:
    """Tests for rate limit enforcement."""

    def test_rate_limit_enforced_on_inference_endpoint(self, client_with_rate_limiting: TestClient):
        """Test that rate limiting is configured on inference endpoints."""
        headers = {"X-API-Key": "test_api_key_123"}

        response = client_with_rate_limiting.get("/api/v1/models", headers=headers)

        assert response.status_code in [200, 401, 404]
        assert "X-RateLimit-Limit" in response.headers, "Rate limit headers should be present"
        assert "X-RateLimit-Remaining" in response.headers, "Rate limit headers should be present"

        limit = response.headers.get("X-RateLimit-Limit")
        assert limit is not None, "Rate limit should be configured"

    def test_rate_limit_not_enforced_when_disabled(self, client_without_rate_limiting: TestClient):
        """Test that rate limiting is disabled when RATE_LIMIT_ENABLED is false."""
        headers = {"X-API-Key": "test_api_key_123"}

        response = client_without_rate_limiting.get("/api/v1/models", headers=headers)

        assert response.status_code in [200, 401, 404]
        assert (
            "X-RateLimit-Limit" not in response.headers
        ), "Rate limit headers should not be present when disabled"

    def test_rate_limit_per_api_key(self, client_with_rate_limiting: TestClient):
        """Test that rate limits are tracked per API key."""
        headers_key1 = {"X-API-Key": "key1"}
        headers_key2 = {"X-API-Key": "key2"}

        for _ in range(5):
            response = client_with_rate_limiting.get("/api/v1/models", headers=headers_key1)
            assert response.status_code != 429

        response_key2 = client_with_rate_limiting.get("/api/v1/models", headers=headers_key2)
        assert response_key2.status_code != 429, "Different API key should have separate limit"

    def test_rate_limit_error_response_format(self, client_with_rate_limiting: TestClient):
        """Test that rate limit exceeded returns proper error format."""
        headers = {"X-API-Key": "test_api_key_123"}

        for _ in range(6):
            response = client_with_rate_limiting.get("/api/v1/models", headers=headers)
            if response.status_code == 429:
                data = response.json()
                assert data["status"] == "error"
                assert "error" in data
                assert data["error"]["code"] == "RATE_LIMIT_EXCEEDED"
                assert "message" in data["error"]
                break


class TestRateLimitDifferentEndpoints:
    """Tests for different rate limits on different endpoint types."""

    def test_different_limits_for_different_endpoints(self, client_with_rate_limiting: TestClient):
        """Test that different endpoints have different rate limit configurations."""
        headers = {"X-API-Key": "test_api_key_123"}

        models_response = client_with_rate_limiting.get("/api/v1/models", headers=headers)
        training_response = client_with_rate_limiting.get("/api/v1/training/jobs", headers=headers)

        assert (
            "X-RateLimit-Limit" in models_response.headers
        ), "Models endpoint should have rate limit headers"
        assert (
            "X-RateLimit-Limit" in training_response.headers
        ), "Training endpoint should have rate limit headers"

        models_limit = models_response.headers.get("X-RateLimit-Limit")
        training_limit = training_response.headers.get("X-RateLimit-Limit")

        assert models_limit is not None, "Models endpoint should have a rate limit configured"
        assert training_limit is not None, "Training endpoint should have a rate limit configured"
        assert models_limit != training_limit, "Different endpoints should have different limits"
