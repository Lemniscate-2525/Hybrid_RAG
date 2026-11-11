"""
Resilient LLM Inference Engine.
Supports OpenRouter (free and paid models), Mistral AI, and standard OpenAI-compatible endpoints.
Implements Circuit Breaker, Exponential Backoff with Jitter, and Structured Error Handling.
"""
import os
import re
import time
import random
import logging
from enum import Enum
from typing import Optional, Dict, Any
import httpx
from src.config import settings

logger = logging.getLogger(__name__)

class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

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
                logger.info("Circuit breaker transitioning from OPEN to HALF_OPEN...")
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True

    def record_success(self):
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0

    def record_failure(self):
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
        if self.consecutive_failures >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(f"Circuit breaker tripped to OPEN after {self.consecutive_failures} failures.")

class ResilientLLMClient:
    def __init__(self):
        self.timeout = settings.LLM_TIMEOUT_SECONDS
        self.max_retries = settings.LLM_MAX_RETRIES
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            recovery_timeout=settings.CIRCUIT_BREAKER_RESET_TIMEOUT
        )

    def get_credentials(self) -> Dict[str, str]:
        """Dynamically resolve API key, endpoint, and model from environment."""
        # Check OpenRouter, generic LLM key, or Mistral
        api_key = (
            os.getenv("OPENROUTER_API_KEY") or
            settings.OPENROUTER_API_KEY or
            os.getenv("LLM_API_KEY") or
            settings.LLM_API_KEY or
            os.getenv("MistralAI") or
            settings.MISTRAL_API_KEY
        )

        if not api_key or "your-" in api_key.lower() or api_key in ["mock_key", "change-me"]:
            return {"configured": False}

        api_key = api_key.strip()

        # If OpenRouter key format (sk-or-...) or specified via OPENROUTER_API_KEY
        if api_key.startswith("sk-or-") or os.getenv("OPENROUTER_API_KEY") or settings.OPENROUTER_API_KEY:
            api_url = "https://openrouter.ai/api/v1/chat/completions"
            model = os.getenv("OPENROUTER_MODEL") or os.getenv("LLM_MODEL") or settings.LLM_MODEL or "meta-llama/llama-3.2-3b-instruct:free"
            provider = "openrouter"
        elif "mistral" in settings.LLM_BASE_URL.lower() and not api_key.startswith("sk-or-"):
            api_url = "https://api.mistral.ai/v1/chat/completions"
            model = settings.MISTRAL_MODEL
            provider = "mistral"
        else:
            api_url = os.getenv("LLM_BASE_URL") or settings.LLM_BASE_URL
            model = os.getenv("LLM_MODEL") or settings.LLM_MODEL
            provider = "generic"

        return {
            "configured": True,
            "api_key": api_key,
            "api_url": api_url,
            "model": model,
            "provider": provider
        }

    def _fallback_local_synthesis(self, user_prompt: str) -> str:
        """
        Synthesize an extracted summary directly from retrieved evidence
        if no external LLM API key has been configured yet.
        """
        match = re.search(r'<retrieved_evidence[^>]*>(.*?)</retrieved_evidence>', user_prompt, re.DOTALL)
        if match:
            passage = match.group(1).strip()
            # Clean sentences
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', passage) if len(s.strip()) > 15]
            summary = " ".join(sentences[:3]) if sentences else passage[:300]
            return summary
        return "I could not find relevant passages in your uploaded documents to answer this question."

    async def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate completion with circuit breaker protection and bounded retries.
        """
        creds = self.get_credentials()

        if not creds.get("configured"):
            logger.info("No LLM API key configured. Generating local passage extraction.")
            return self._fallback_local_synthesis(user_prompt)

        if not self.circuit_breaker.can_execute():
            raise RuntimeError(
                "Upstream LLM provider is currently experiencing issues. Circuit breaker is OPEN. Please retry in 30 seconds."
            )

        api_url = creds["api_url"]
        api_key = creds["api_key"]
        model = creds["model"]

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # OpenRouter-specific metadata headers
        if creds.get("provider") == "openrouter":
            headers["HTTP-Referer"] = "http://localhost:8501"
            headers["X-Title"] = "Hybrid RAG"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1024,
        }

        attempt = 0
        last_exception = None

        while attempt < self.max_retries:
            attempt += 1
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(api_url, json=payload, headers=headers)

                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices", [])
                    if choices and isinstance(choices[0].get("message"), dict):
                        choice = choices[0]
                        message = choice["message"]
                        content = message.get("content")

                        # OpenRouter can return HTTP 200 with no user-facing content
                        # (for example, a reasoning-only or tool-call completion).
                        # Never silently pass None into downstream JSON parsing.
                        if isinstance(content, list):
                            text_parts = [
                                part.get("text", "")
                                for part in content
                                if isinstance(part, dict) and part.get("type") == "text"
                            ]
                            content = "\\n".join(part for part in text_parts if part).strip()

                        if isinstance(content, str) and content.strip():
                            self.circuit_breaker.record_success()
                            return content

                        finish_reason = choice.get("finish_reason", "unknown")
                        message_keys = sorted(message.keys())
                        has_reasoning = bool(message.get("reasoning") or message.get("reasoning_content"))
                        has_tool_calls = bool(message.get("tool_calls"))
                        self.circuit_breaker.record_failure()
                        raise RuntimeError(
                            "LLM provider returned HTTP 200 but no text content. "
                            f"provider={creds.get('provider')}, model={model}, "
                            f"finish_reason={finish_reason}, message_keys={message_keys}, "
                            f"reasoning_present={has_reasoning}, tool_calls_present={has_tool_calls}. "
                            "Check the selected OpenRouter model/provider; it must support "
                            "ordinary chat completions and return message.content."
                        )

                    raise RuntimeError(
                        "Unexpected LLM response structure: "
                        f"top_level_keys={sorted(data.keys())}, choices_type={type(choices).__name__}, "
                        f"choices_count={len(choices) if isinstance(choices, list) else 'n/a'}"
                    )

                elif response.status_code == 401:
                    self.circuit_breaker.record_failure()
                    err_msg = f"⚠️ The configured API key was rejected by {creds.get('provider')}. Please check your OPENROUTER_API_KEY in `.env`.\n\n"
                    logger.error(err_msg)
                    return err_msg + self._fallback_local_synthesis(user_prompt)

                elif response.status_code == 429:
                    logger.warning(f"LLM API rate-limited (429) on attempt {attempt}/{self.max_retries}")
                    time.sleep((2 ** attempt) + random.uniform(0.1, 0.5))

                elif response.status_code >= 500:
                    logger.warning(f"LLM API 5xx ({response.status_code}) on attempt {attempt}/{self.max_retries}")
                    time.sleep((2 ** attempt) + random.uniform(0.1, 0.5))

                else:
                    self.circuit_breaker.record_failure()
                    raise RuntimeError(f"LLM API returned status {response.status_code}: {response.text}")

            except httpx.TimeoutException as e:
                logger.warning(f"LLM API call timed out after {self.timeout}s on attempt {attempt}")
                last_exception = e
                time.sleep((2 ** attempt) + random.uniform(0.1, 0.5))

            except Exception as e:
                last_exception = e
                if isinstance(e, (ValueError, RuntimeError)):
                    raise
                logger.warning(f"Error calling LLM API on attempt {attempt}: {e}")

        # Retries exhausted
        self.circuit_breaker.record_failure()
        raise RuntimeError(f"Failed to communicate with LLM provider after {self.max_retries} attempts: {str(last_exception)}")

llm_client = ResilientLLMClient()
