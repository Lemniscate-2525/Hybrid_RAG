"""
Redis-Backed Distributed Rate Limiting Middleware (Phase 4).
Uses an atomic Redis Lua sliding-window script to prevent race conditions.
Supports per-user (authenticated) and per-IP (unauthenticated with proxy verification) limiting.
"""
import time
import logging
from typing import Optional, Callable
import redis
from fastapi import Request, HTTPException, status
from src.config import settings

logger = logging.getLogger(__name__)

# Atomic sliding window script in Lua
SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local clearBefore = now - window

redis.call('ZREMRANGEBYSCORE', key, 0, clearBefore)
local currentRequests = redis.call('ZCARD', key)

if currentRequests < limit then
    redis.call('ZADD', key, now, now)
    redis.call('EXPIRE', key, math.ceil(window))
    return {1, limit - currentRequests - 1, 0}
else
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local retryAfter = 1
    if oldest and #oldest >= 2 then
        retryAfter = math.ceil(tonumber(oldest[2]) + window - now)
    end
    if retryAfter < 1 then retryAfter = 1 end
    return {0, 0, retryAfter}
end
"""

class RedisConnectionManager:
    _instance = None

    def __init__(self):
        self.client: Optional[redis.Redis] = None
        self.is_connected = False
        self._init_connection()

    def _init_connection(self):
        try:
            r = redis.Redis.from_url(
                settings.REDIS_URL,
                socket_timeout=settings.REDIS_TIMEOUT_MS / 1000.0,
                socket_connect_timeout=settings.REDIS_TIMEOUT_MS / 1000.0,
                decode_responses=True
            )
            r.ping()
            self.client = r
            self.is_connected = True
            logger.info("Connected to distributed Redis server.")
        except Exception as e:
            logger.warning(f"Could not connect to Redis at {settings.REDIS_URL}: {e}. Falling back to FakeRedis for development/testing.")
            try:
                import fakeredis
                self.client = fakeredis.FakeRedis(decode_responses=True)
                self.is_connected = True
                logger.info("Initialized in-memory FakeRedis fallback.")
            except Exception as fe:
                logger.error(f"Failed to initialize FakeRedis: {fe}")
                self.client = None
                self.is_connected = False

    def get_client(self) -> Optional[redis.Redis]:
        return self.client

redis_manager = RedisConnectionManager()

def get_client_ip(request: Request) -> str:
    """
    Safely extract client IP address.
    Only inspects X-Forwarded-For if request comes from a configured trusted proxy.
    """
    client_host = request.client.host if request.client else "127.0.0.1"

    if client_host in settings.TRUSTED_PROXIES:
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # First IP in list is original client
            client_ip = forwarded_for.split(",")[0].strip()
            return client_ip

    return client_host

class RateLimiter:
    """
    Callable rate limiter dependency for FastAPI endpoints.
    """
    def __init__(self, limit: int, window_seconds: int, prefix: str, fail_closed: bool = True):
        self.limit = limit
        self.window_seconds = window_seconds
        self.prefix = prefix
        self.fail_closed = fail_closed

    async def __call__(self, request: Request):
        if not settings.REDIS_RATE_LIMIT_ENABLED:
            return

        redis_client = redis_manager.get_client()
        if not redis_client:
            if self.fail_closed:
                logger.error(f"Rate limiting unavailable (Redis down) - failing closed for {self.prefix}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Rate limiting service temporarily unavailable. Please retry later."
                )
            else:
                logger.warning(f"Rate limiting unavailable - failing open for {self.prefix}")
                return

        # Determine identifier: verified user ID from Bearer token or client IP
        identifier = None
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            try:
                from src.security.auth import decode_access_token
                payload = decode_access_token(token)
                user_id = payload.get("sub")
                if user_id:
                    identifier = f"user:{user_id}"
            except Exception:
                pass

        if not identifier:
            user = getattr(request.state, "user", None)
            if user and hasattr(user, "id"):
                identifier = f"user:{user.id}"
            else:
                client_ip = get_client_ip(request)
                identifier = f"ip:{client_ip}"

        key = f"ratelimit:{self.prefix}:{identifier}"
        now = time.time()

        try:
            res = redis_client.eval(SLIDING_WINDOW_LUA, 1, key, now, self.window_seconds, self.limit)
            allowed, remaining, retry_after = int(res[0]), int(res[1]), int(res[2])

            if allowed == 0:
                logger.warning(f"Rate limit exceeded on {self.prefix} for {identifier}. Retry after {retry_after}s.")
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Maximum {self.limit} requests per {self.window_seconds}s.",
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(self.limit),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(retry_after)
                    }
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Rate limiter Redis execution error: {e}")
            if self.fail_closed:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Rate limiting validation error."
                )

