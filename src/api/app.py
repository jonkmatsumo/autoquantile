"""FastAPI application setup."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.api.dto.common import ErrorDetail, ErrorResponse
from src.api.exceptions import APIException
from src.api.rate_limiting import RATE_LIMIT_ENABLED, limiter
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        FastAPI: Configured application.
    """
    app = FastAPI(
        title="AutoQuantile API",
        description="REST API for salary forecasting model inference and training",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if RATE_LIMIT_ENABLED:
        app.state.limiter = limiter
        app.add_middleware(SlowAPIMiddleware)

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
        """Handle rate limit exceeded exceptions.

        Args:
            request (Request): Request object.
            exc (RateLimitExceeded): Rate limit exception.

        Returns:
            JSONResponse: Error response.
        """
        logger.warning(
            f"Rate limit exceeded: {exc.detail}",
            extra={"detail": exc.detail, "retry_after": getattr(exc, "retry_after", None)},
        )
        return JSONResponse(
            status_code=429,
            content=ErrorResponse(
                status="error",
                error=ErrorDetail(
                    code="RATE_LIMIT_EXCEEDED",
                    message=f"Rate limit exceeded: {exc.detail}",
                    details={"retry_after": exc.retry_after} if hasattr(exc, "retry_after") else {},
                ),
            ).model_dump(),
        )

    @app.exception_handler(APIException)
    async def api_exception_handler(request: Request, exc: APIException):
        """Handle API exceptions.

        Args:
            request (Request): Request object.
            exc (APIException): Exception instance.

        Returns:
            JSONResponse: Error response.
        """
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                status="error",
                error=ErrorDetail(
                    code=exc.code,
                    message=exc.message,
                    details=exc.details,
                ),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions.

        Args:
            request (Request): Request object.
            exc (Exception): Exception instance.

        Returns:
            JSONResponse: Error response.
        """
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                status="error",
                error=ErrorDetail(
                    code="INTERNAL_ERROR",
                    message="An internal error occurred",
                    details={"type": type(exc).__name__},
                ),
            ).model_dump(),
        )

    from src.api.mcp.server import register_mcp_tools
    from src.api.routers import analytics, health, inference, models, training, workflow

    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(inference.router)
    app.include_router(training.router)
    app.include_router(workflow.router)
    app.include_router(analytics.router)

    register_mcp_tools(app)

    return app
