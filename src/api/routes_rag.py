"""
Hybrid RAG Query Endpoints.
Exposes secure query interface with rate limiting, multi-tenant isolation,
prompt injection guardrails, and optional protected evaluation telemetry.
"""
import secrets
from typing import Optional
from fastapi import APIRouter, Depends, Request, Header, HTTPException, status
from src.db.models import User
from src.api.schemas import QueryRequest, QueryResponse
from src.security.auth import get_current_user
from src.services.rag_service import rag_service
from src.middleware.rate_limiter import RateLimiter
from src.config import settings

router = APIRouter(prefix="/rag", tags=["RAG Query Engine"])
query_rate_limiter = RateLimiter(
    limit=settings.RATE_LIMIT_QUERY_PER_MINUTE,
    window_seconds=60,
    prefix="query",
    fail_closed=True
)

@router.post("/query", response_model=QueryResponse, dependencies=[Depends(query_rate_limiter)])
async def execute_rag_query(
    request: Request,
    query_in: QueryRequest,
    current_user: User = Depends(get_current_user),
    x_evaluation_token: Optional[str] = Header(default=None, alias="X-Evaluation-Token")
):
    """Execute a tenant-isolated RAG query; raw chunk telemetry requires a separate secret."""
    include_evaluation_trace = False
    if x_evaluation_token is not None:
        configured = settings.EVALUATION_DEBUG_TOKEN
        if not configured or not secrets.compare_digest(x_evaluation_token, configured):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid evaluation token")
        include_evaluation_trace = True

    correlation_id = getattr(request.state, "correlation_id", None)
    return await rag_service.process_query(
        query_in=query_in,
        user=current_user,
        correlation_id=correlation_id,
        include_evaluation_trace=include_evaluation_trace
    )
