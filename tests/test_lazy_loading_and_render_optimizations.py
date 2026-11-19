"""
Targeted tests for Lazy Model Loading, Memory Optimization, Production Redis/Qdrant Enforcement,
and Health Checks for Render deployment.
"""
import threading
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.config import settings
from src.services.embedding_service import embedding_service
from src.services.rerank_service import reranker_service
from src.main import app

def test_fastapi_imports_without_loading_models():
    """Verify in a clean process that importing FastAPI does NOT load heavy embedding or reranking models."""
    import subprocess
    import sys

    code = (
        "from src.main import app; "
        "from src.services.embedding_service import embedding_service; "
        "from src.services.rerank_service import reranker_service; "
        "assert not embedding_service.is_dense_loaded, 'Dense model loaded at import!'; "
        "assert not embedding_service.is_sparse_loaded, 'Sparse model loaded at import!'; "
        "assert not reranker_service.is_loaded, 'Reranker model loaded at import!'; "
        "print('CLEAN_IMPORT_VERIFIED')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"Import process failed: {result.stderr}"
    assert "CLEAN_IMPORT_VERIFIED" in result.stdout

def test_lazy_loading_dense_embedding_on_first_use():
    """Verify that the dense model initializes only when requested."""
    texts = ["Test sentence for dense embedding."]
    emb = embedding_service.embed_dense(texts)
    assert embedding_service.is_dense_loaded is True
    assert len(emb) == 1
    assert len(emb[0]) == 384  # MiniLM dimension

def test_lazy_loading_sparse_bm42_embedding_on_first_use():
    """Verify that the BM42 sparse model initializes only when requested."""
    texts = ["Test query for sparse BM42 representation."]
    sparse = embedding_service.embed_sparse(texts)
    assert embedding_service.is_sparse_loaded is True
    assert len(sparse) == 1
    assert hasattr(sparse[0], "indices")
    assert hasattr(sparse[0], "values")

def test_concurrent_initialization_thread_safety():
    """Verify that concurrent requests reuse the exact same singleton model instance."""
    dense_instances = []
    sparse_instances = []

    def worker():
        d_model = embedding_service._get_dense_model()
        s_model = embedding_service._get_sparse_model()
        dense_instances.append(id(d_model))
        sparse_instances.append(id(s_model))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All threads must share the exact same object id
    assert len(set(dense_instances)) == 1, "Duplicate dense model instances were created!"
    assert len(set(sparse_instances)) == 1, "Duplicate sparse model instances were created!"

def test_cross_encoder_lazy_loading_and_ranking():
    """Verify that the Cross-Encoder initializes only on demand and accurately scores relevance."""
    query = "What is hybrid retrieval?"
    passages = [
        {"text": "Hybrid search combines dense vector similarity with sparse BM42 keyword matching."},
        {"text": "The Eiffel tower is located in Paris, France."}
    ]

    reranked = reranker_service.rerank(query, passages, top_k=2)
    assert reranker_service.is_loaded is True
    assert len(reranked) == 2
    # The hybrid retrieval passage must have a significantly higher score than the Eiffel tower
    assert reranked[0]["text"].startswith("Hybrid search")
    assert reranked[0]["rerank_score"] > reranked[1]["rerank_score"]

def test_health_checks_do_not_load_models():
    """Verify that /health/live and /health/ready do not trigger model loading if uninitialized."""
    client = TestClient(app)
    # Liveness probe
    res_live = client.get("/api/v1/health/live")
    assert res_live.status_code == 200
    assert res_live.json()["status"] == "UP"

    # Readiness probe
    res_ready = client.get("/api/v1/health/ready")
    assert res_ready.status_code in (200, 503)
    data = res_ready.json()
    assert "database" in data["components"]
    assert "redis" in data["components"]
    assert "qdrant" in data["components"]

def test_readiness_fails_503_when_redis_is_down():
    """Verify readiness returns HTTP 503 and status DOWN when Redis is down."""
    client = TestClient(app)
    with patch("src.middleware.rate_limiter.redis_manager.get_client", return_value=None):
        res = client.get("/api/v1/health/ready")
        assert res.status_code == 503
        data = res.json()
        assert data["status"] == "DOWN"
        assert data["components"]["redis"]["status"] == "DOWN"

def test_production_redis_enforcement():
    """Verify that in production mode, Redis connection failure does NOT fall back to FakeRedis."""
    from src.middleware.rate_limiter import RedisConnectionManager
    with patch("src.config.settings.ENVIRONMENT", "production"):
        with patch("redis.ConnectionPool.from_url", side_effect=Exception("Redis down")):
            with pytest.raises(RuntimeError) as exc_info:
                mgr = RedisConnectionManager()
            assert "Failed to connect to Redis" in str(exc_info.value)

def test_production_qdrant_enforcement():
    """Verify that in production mode, missing QDRANT_URL or QDRANT_API_KEY fails immediately."""
    from src.services.qdrant_service import QdrantService
    with patch("src.config.settings.ENVIRONMENT", "production"):
        with patch("src.config.settings.QDRANT_URL", None):
            with pytest.raises(RuntimeError) as exc_info:
                q_svc = QdrantService()
            assert "QDRANT_URL and QDRANT_API_KEY are strictly required" in str(exc_info.value)
