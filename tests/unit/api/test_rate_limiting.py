"""Unit tests for rate limiting functionality."""

from unittest.mock import MagicMock, patch

from fastapi import Request
from slowapi.errors import RateLimitExceeded

from src.api.rate_limiting import get_rate_limit_key, limiter


class TestGetRateLimitKey:
    """Tests for get_rate_limit_key function."""

    def test_get_rate_limit_key_with_bearer_token(self):
        """Test rate limit key extraction with Bearer token."""
        request = MagicMock(spec=Request)
        request.headers = {"Authorization": "Bearer test_api_key_123"}

        key = get_rate_limit_key(request)

        assert key == "api_key:test_api_key_123"

    def test_get_rate_limit_key_with_x_api_key_header(self):
        """Test rate limit key extraction with X-API-Key header."""
        request = MagicMock(spec=Request)
        request.headers = {"X-API-Key": "test_api_key_456"}

        key = get_rate_limit_key(request)

        assert key == "api_key:test_api_key_456"

    def test_get_rate_limit_key_prefers_bearer_over_x_api_key(self):
        """Test that Bearer token takes precedence over X-API-Key."""
        request = MagicMock(spec=Request)
        request.headers = {
            "Authorization": "Bearer bearer_key",
            "X-API-Key": "x_api_key",
        }

        key = get_rate_limit_key(request)

        assert key == "api_key:bearer_key"

    def test_get_rate_limit_key_with_ip_fallback(self):
        """Test rate limit key falls back to IP address when no API key."""
        request = MagicMock(spec=Request)
        request.headers = {}
        request.client = MagicMock()
        request.client.host = "192.168.1.1"

        with patch("src.api.rate_limiting.get_remote_address", return_value="192.168.1.1"):
            key = get_rate_limit_key(request)

        assert key == "ip:192.168.1.1"


class TestRateLimiterConfiguration:
    """Tests for rate limiter configuration."""

    def test_limiter_initialized(self):
        """Test that limiter is properly initialized."""
        assert limiter is not None
        assert hasattr(limiter, "limit")

    def test_rate_limit_constants_exist(self):
        """Test that rate limit constants are defined."""
        from src.api.rate_limiting import (
            ANALYTICS_LIMIT,
            INFERENCE_LIMIT,
            MODELS_LIMIT,
            TRAINING_LIMIT,
            WORKFLOW_LIMIT,
        )

        assert INFERENCE_LIMIT is not None
        assert TRAINING_LIMIT is not None
        assert ANALYTICS_LIMIT is not None
        assert MODELS_LIMIT is not None
        assert WORKFLOW_LIMIT is not None

    def test_rate_limit_constants_format(self):
        """Test that rate limit constants follow expected format."""
        from src.api.rate_limiting import (
            ANALYTICS_LIMIT,
            INFERENCE_LIMIT,
            MODELS_LIMIT,
            TRAINING_LIMIT,
            WORKFLOW_LIMIT,
        )

        limits = [INFERENCE_LIMIT, TRAINING_LIMIT, ANALYTICS_LIMIT, MODELS_LIMIT, WORKFLOW_LIMIT]

        for limit in limits:
            assert limit is not None, "Limit should not be None"
            assert "/" in limit, f"Limit {limit} should contain '/' separator"
            parts = limit.split("/")
            assert len(parts) == 2, f"Limit {limit} should have format 'number/period'"
            assert parts[0].isdigit(), f"Limit {limit} should start with a number"
            assert parts[1] in ["minute", "hour", "day"], f"Limit {limit} should have valid period"


class TestRateLimitExceededHandler:
    """Tests for rate limit exceeded exception handler."""

    def test_rate_limit_exceeded_handler_in_app(self):
        """Test that rate limit exception handler is registered in app."""
        from src.api.app import create_app

        app = create_app()

        assert RateLimitExceeded in app.exception_handlers
        assert callable(app.exception_handlers[RateLimitExceeded])

    def test_rate_limit_exceeded_response_format(self):
        """Test that rate limit exceeded handler is properly configured."""
        from src.api.app import create_app

        app = create_app()

        handler = app.exception_handlers.get(RateLimitExceeded)
        assert handler is not None, "Rate limit exception handler should be registered"
        assert callable(handler), "Rate limit exception handler should be callable"
