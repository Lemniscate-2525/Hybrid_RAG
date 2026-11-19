"""
Unit & Integration Tests for Redis-Backed Rate Limiting (Phase 4).
"""
import pytest
from tests.test_multitenancy import get_auth_token

def test_rate_limiter_exceeded(client):
    from src.api.routes_rag import query_rate_limiter
    token = get_auth_token(client, "rate_limiter_user@example.com", "PasswordRate123!")

    # Temporarily set limit to 5 to test rate limiter response and headers without delay
    original_limit = query_rate_limiter.limit
    query_rate_limiter.limit = 5
    try:
        responses = []
        for _ in range(7):
            res = client.post(
                "/api/v1/rag/query",
                json={"query": "Explain what BM42 sparse representations are."},
                headers={"Authorization": f"Bearer {token}"}
            )
            responses.append(res)
            if res.status_code == 429:
                break

        # At least one request should be rate-limited
        last_res = responses[-1]
        assert last_res.status_code == 429
        assert "Rate limit exceeded" in last_res.json()["detail"]
        assert "Retry-After" in last_res.headers
        assert last_res.headers["X-RateLimit-Remaining"] == "0"
    finally:
        query_rate_limiter.limit = original_limit

