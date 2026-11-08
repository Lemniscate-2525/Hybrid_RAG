"""
Health Check and Operational Readiness Endpoints (Phase 8 & 10).
Provides liveness and readiness probes for orchestrators (Docker, Kubernetes).
"""
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, status, Response
from sqlalchemy.orm import Session
from sqlalchemy import text
from src.config import settings
from src.db.database import get_db
from src.api.schemas import HealthLiveResponse, HealthReadyResponse, ComponentStatus
from src.middleware.rate_limiter import redis_manager
from src.services.qdrant_service import qdrant_service

router = APIRouter(prefix="/health", tags=["Health & Observability"])

@router.get("/live", response_model=HealthLiveResponse)
def liveness_probe():
    """Liveness probe: verifies application process is running."""
    return HealthLiveResponse(
        status="UP",
        version=settings.APP_VERSION,
        timestamp=datetime.now(timezone.utc)
    )

@router.get("/ready", response_model=HealthReadyResponse)
def readiness_probe(response: Response, db: Session = Depends(get_db)):
    """Readiness probe: validates database, Qdrant, and Redis connections."""
    components = {}
    is_ready = True

    # Check Database
    try:
        db.execute(text("SELECT 1"))
        components["database"] = ComponentStatus(status="UP", details="SQLAlchemy session responsive")
    except Exception as e:
        is_ready = False
        components["database"] = ComponentStatus(status="DOWN", details=str(e))

    # Check Redis
    redis_client = redis_manager.get_client()
    if redis_client:
        try:
            redis_client.ping()
            components["redis"] = ComponentStatus(status="UP", details="Redis responsive")
        except Exception as e:
            components["redis"] = ComponentStatus(status="DOWN", details=str(e))
    else:
        components["redis"] = ComponentStatus(status="DOWN", details="Redis connection unavailable")

    # Check Qdrant
    try:
        exists = qdrant_service.client.collection_exists(collection_name=qdrant_service.collection_name)
        components["qdrant"] = ComponentStatus(
            status="UP",
            details=f"Collection '{qdrant_service.collection_name}' ready (exists={exists})"
        )
    except Exception as e:
        is_ready = False
        components["qdrant"] = ComponentStatus(status="DOWN", details=str(e))

    overall_status = "UP" if is_ready else "DOWN"
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthReadyResponse(
        status=overall_status,
        timestamp=datetime.now(timezone.utc),
        components=components
    )

