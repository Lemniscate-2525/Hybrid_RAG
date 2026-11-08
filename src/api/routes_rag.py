"""
Hybrid RAG Query Endpoints.
Exposes secure query interface with rate limiting, multi-tenant isolation,
prompt injection guardrails, and citation generation.
"""
from fastapi import APIRouter, Depends, Request
from src.db.models import User
from src.api.schemas import QueryRequest, QueryResponse
from src.security.auth import get_current_user
from src.services.rag_service import rag_service
from src.middleware.rate_limiter import RateLimiter
from src.config import settings

router = APIRouter(prefix="/rag", tags=["RAG Query Engine"])

# Rate limiter dependency for query endpoints: 30 requests/minute per user
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
    current_user: User = Depends(get_current_user)
):
    """
    Execute a secure hybrid search query against the user's indexed documents.
    Enforces BM42 sparse + dense retrieval, Reciprocal Rank Fusion, Cross-Encoder reranking,
    prompt injection sanitization, and structured citations.
    """
    correlation_id = getattr(request.state, "correlation_id", None)
    response = await rag_service.process_query(
        query_in=query_in,
        user=current_user,
        correlation_id=correlation_id
    )
    return response

