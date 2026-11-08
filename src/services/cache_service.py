"""
Redis-Backed Multi-Tenant Caching Service (Phase 5).
Safely caches RAG responses with strict tenant isolation, TTL expiration,
and automatic cache invalidation on document deletion.
"""
import json
import hashlib
import logging
from typing import Optional, Dict, Any
from src.config import settings
from src.middleware.rate_limiter import redis_manager

logger = logging.getLogger(__name__)

class CacheService:
    def __init__(self):
        self.ttl = settings.REDIS_CACHE_TTL_SECONDS
        self.enabled = settings.REDIS_CACHE_ENABLED

    def _generate_cache_key(self, user_id: str, query: str, filename: Optional[str] = None) -> str:
        """
        Construct a collision-proof tenant-isolated cache key.
        Key strictly binds to user_id, optional filename scope, and normalized query hash.
        """
        normalized_query = query.strip().lower()
        scope = filename.strip().lower() if filename else "all"
        query_hash = hashlib.sha256(f"{scope}:{normalized_query}".encode("utf-8")).hexdigest()
        return f"cache:v1:rag:{user_id}:{query_hash}"

    def get_cached_response(self, user_id: str, query: str, filename: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve cached response if available."""
        if not self.enabled:
            return None

        redis_client = redis_manager.get_client()
        if not redis_client:
            return None

        key = self._generate_cache_key(user_id, query, filename)
        try:
            val = redis_client.get(key)
            if val:
                logger.info(f"Cache hit for user {user_id} on query.")
                return json.loads(val)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")

        return None

    def set_cached_response(
        self,
        user_id: str,
        query: str,
        response_data: Dict[str, Any],
        filename: Optional[str] = None
    ) -> None:
        """Store response in cache with tenant-isolated key and TTL."""
        if not self.enabled:
            return

        redis_client = redis_manager.get_client()
        if not redis_client:
            return

        key = self._generate_cache_key(user_id, query, filename)
        try:
            payload = json.dumps(response_data)
            redis_client.set(key, payload, ex=self.ttl)
            logger.info(f"Cached response for user {user_id} with TTL {self.ttl}s")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def invalidate_user_cache(self, user_id: str) -> int:
        """
        Invalidate all cached responses for a specific user.
        Invoked whenever user uploads, modifies, or deletes documents.
        """
        redis_client = redis_manager.get_client()
        if not redis_client:
            return 0

        pattern = f"cache:v1:rag:{user_id}:*"
        try:
            keys = list(redis_client.scan_iter(match=pattern, count=100))
            if keys:
                deleted = redis_client.delete(*keys)
                logger.info(f"Invalidated {deleted} cache keys for user {user_id}")
                return deleted
        except Exception as e:
            logger.warning(f"Cache invalidation error for user {user_id}: {e}")

        return 0

cache_service = CacheService()

