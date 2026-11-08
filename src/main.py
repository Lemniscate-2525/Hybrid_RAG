"""
Main FastAPI Application Entrypoint.
Configures CORS, Correlation ID, Rate Limiting, Routers, and Observability.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.config import settings
from src.db.database import init_db
from src.middleware.correlation import CorrelationIdMiddleware
from src.api.routes_auth import router as auth_router
from src.api.routes_documents import router as documents_router
from src.api.routes_rag import router as rag_router
from src.api.routes_health import router as health_router

# Configure structured logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s (%(threadName)s): %(message)s"
)
logger = logging.getLogger("api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Application Lifecycle...")
    # Initialize database tables
    init_db()
    logger.info("Application startup complete.")
    yield
    logger.info("Application shutting down...")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-grade Multi-Tenant Hybrid RAG API with BM42, Qdrant, and MistralAI",
    lifespan=lifespan
)

# Correlation ID Middleware (must be first)
app.add_middleware(CorrelationIdMiddleware)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Safe Exception Handler (prevents leaking internal traces to end users)
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    correlation_id = getattr(request.state, "correlation_id", "unknown")
    logger.error(f"Unhandled Exception for correlation_id={correlation_id}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred. Please contact support with the correlation ID.",
            "correlation_id": correlation_id
        }
    )

# Include Routers
app.include_router(auth_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")
app.include_router(rag_router, prefix="/api/v1")
app.include_router(health_router, prefix="/api/v1")

@app.get("/")
def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "healthy",
        "docs_url": "/docs"
    }
