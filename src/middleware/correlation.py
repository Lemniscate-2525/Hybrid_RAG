"""
Correlation ID & Structured Request Observability Middleware (Phase 8).
Injects a unique correlation ID into every request, traces it across logs,
and records request latency.
"""
import uuid
import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("api.access")

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        # Extract existing correlation ID or generate a new UUID
        correlation_id = request.headers.get("X-Correlation-ID") or request.headers.get("X-Request-ID")
        if not correlation_id:
            correlation_id = str(uuid.uuid4())

        request.state.correlation_id = correlation_id
        start_time = time.time()

        response = await call_next(request)

        duration_ms = int((time.time() - start_time) * 1000)
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Response-Time-Ms"] = str(duration_ms)

        # Log structured request summary
        logger.info(
            f"{request.method} {request.url.path} - status={response.status_code} "
            f"duration={duration_ms}ms correlation_id={correlation_id}"
        )

        return response

