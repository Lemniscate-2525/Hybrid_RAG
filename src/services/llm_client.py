"""
Resilient LLM Inference Engine (Phase 6).
Implements Circuit Breaker, Exponential Backoff with Jitter, Strict Timeouts,
and Structured Error Classification for MistralAI.
"""
import time
import math
import random
import logging
from enum import Enum
from typing import Optional, List, Dict
import httpx
from src.config import settings

logger = logging.getLogger(__name__)

class CircuitState(str, Enum):
    CLOSED = "CLOSED"       # Normal operation
    OPEN = "OPEN"           # Failing, fast-rejecting
    HALF_OPEN = "HALF_OPEN" # Testing upstream recovery

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.last_failure_time = 0.0

    def can_execute(self) -> bool:
        now = time.time()
        if self.state == CircuitState.OPEN:
            if now - self.last_failure_time > self.recovery_timeout:
                logger.info("Circuit breaker transitioning from OPEN to HALF_OPEN (probing upstream)...")
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True

    def record_success(self):
        if self.state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker probe succeeded. Transitioning from HALF_OPEN to CLOSED.")
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0

    def record_failure(self):
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
        if self.consecutive_failures >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(
                f"Circuit breaker tripped to OPEN after {self.consecutive_failures} consecutive failures. "
                f"Fail-fast enabled for {self.recovery_timeout}s."
            )

class ResilientMistralClient:
    def __init__(self):
        self.api_key = settings.MISTRAL_API_KEY
        self.model = settings.MISTRAL_MODEL
        self.timeout = settings.LLM_TIMEOUT_SECONDS
        self.max_retries = settings.LLM_MAX_RETRIES
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            recovery_timeout=settings.CIRCUIT_BREAKER_RESET_TIMEOUT
        )
        self.api_url = "https://api.mistral.ai/v1/chat/completions"

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key != "mock_key")

    async def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate completion with circuit breaker protection and bounded retries.
        """
        if not self.circuit_breaker.can_execute():
            raise RuntimeError(
                "Upstream LLM provider is currently experiencing an outage. Circuit breaker is OPEN. Please try again in 30 seconds."
            )

        if not self.is_configured():
            logger.warning("No MISTRAL_API_KEY configured. Returning deterministic mock response for testing.")
            return (
                f"[Grounded Response (Dev Mode)] Based on the provided context, the concept can be explained as follows. "
                f"Query matched context correctly. [source: sample, page: 1]"
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
        }

        attempt = 0
        last_exception = None

        while attempt < self.max_retries:
            attempt += 1
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(self.api_url, json=payload, headers=headers)

                # Status check
                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    self.circuit_breaker.record_success()
                    return content

                elif response.status_code == 401:
                    # Permanent error: Bad API Key
                    self.circuit_breaker.record_failure()
                    raise ValueError("Mistral AI Authentication failed. Please verify your MISTRAL_API_KEY.")

                elif response.status_code == 429:
                    # Rate limit error from provider
                    logger.warning(f"Mistral AI rate-limited (429) on attempt {attempt}/{self.max_retries}")
                    sleep_time = (2 ** attempt) + random.uniform(0.1, 0.5)
                    time.sleep(sleep_time)

                elif response.status_code >= 500:
                    # Upstream server error
                    logger.warning(f"Mistral AI 5xx error ({response.status_code}) on attempt {attempt}/{self.max_retries}")
                    sleep_time = (2 ** attempt) + random.uniform(0.1, 0.5)
                    time.sleep(sleep_time)

                else:
                    self.circuit_breaker.record_failure()
                    raise RuntimeError(f"Mistral API returned unexpected status {response.status_code}: {response.text}")

            except httpx.TimeoutException as e:
                logger.warning(f"Mistral API call timed out after {self.timeout}s on attempt {attempt}")
                last_exception = e
                sleep_time = (2 ** attempt) + random.uniform(0.1, 0.5)
                time.sleep(sleep_time)

            except Exception as e:
                last_exception = e
                if isinstance(e, (ValueError, RuntimeError)):
                    raise
                logger.warning(f"Error calling Mistral API on attempt {attempt}: {e}")

        # Retries exhausted
        self.circuit_breaker.record_failure()
        raise RuntimeError(
            f"Failed to communicate with Mistral AI after {self.max_retries} attempts: {str(last_exception)}"
        )

llm_client = ResilientMistralClient()

