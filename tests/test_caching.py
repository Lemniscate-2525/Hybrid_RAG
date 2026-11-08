"""
Unit & Integration Tests for Multi-Tenant Redis Caching & Invalidation (Phase 5).
"""
import pytest
from src.services.cache_service import cache_service
from tests.test_multitenancy import get_auth_token

def test_cache_set_get_and_tenant_isolation():
    user_a = "user-uuid-111"
    user_b = "user-uuid-222"
    query = "What is Reciprocal Rank Fusion?"

    payload_a = {"query": query, "answer": "Answer for User A", "citations": []}
    cache_service.set_cached_response(user_id=user_a, query=query, response_data=payload_a)

    # User A gets cache hit
    hit_a = cache_service.get_cached_response(user_id=user_a, query=query)
    assert hit_a is not None
    assert hit_a["answer"] == "Answer for User A"

    # User B MUST get cache miss (strict tenant boundary)
    miss_b = cache_service.get_cached_response(user_id=user_b, query=query)
    assert miss_b is None

def test_cache_invalidation():
    user_id = "user-uuid-invalidate-test"
    query = "Explain CrossEncoder reranking."
    payload = {"query": query, "answer": "Reranking answer", "citations": []}

    cache_service.set_cached_response(user_id=user_id, query=query, response_data=payload)
    assert cache_service.get_cached_response(user_id=user_id, query=query) is not None

    # Invalidate user cache
    invalidated_count = cache_service.invalidate_user_cache(user_id)
    assert invalidated_count >= 1

    # Now cache must be a miss
    assert cache_service.get_cached_response(user_id=user_id, query=query) is None

