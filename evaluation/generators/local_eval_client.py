"""Dedicated local FreeLLMAPI client for evaluation.

Never falls back to the application's OpenRouter credentials.
"""
import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import dotenv_values


class LocalEvalLLMClient:
    def __init__(self) -> None:
        env_file = Path(__file__).resolve().parents[2] / ".env"
        file_values = dotenv_values(env_file) if env_file.exists() else {}

        def setting(name: str) -> str | None:
            value = os.getenv(name) or file_values.get(name)
            return str(value).strip() if value and str(value).strip() else None

        self.api_key = setting("LOCAL_EVAL_API")
        self.base_url = (
            setting("LOCAL_EVAL_BASE_URL")
            or "http://127.0.0.1:31415/v1"
        ).rstrip("/")
        self.model = setting("LOCAL_EVAL_MODEL")

        if not self.api_key:
            raise RuntimeError(
                "LOCAL_EVAL_API is missing from .env. "
                "Evaluation will not use OPENROUTER_API_KEY."
            )

        self.endpoint = self.base_url
        if not self.endpoint.endswith("/chat/completions"):
            self.endpoint += "/chat/completions"

    async def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 2048,
        }

        # Let FreeLLMAPI choose its configured default unless explicitly overridden.
        if self.model:
            payload["model"] = self.model

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    self.endpoint, json=payload, headers=headers
                )
        except httpx.RequestError as exc:
            raise RuntimeError(
                "Could not reach local FreeLLMAPI at "
                f"{self.endpoint}. Confirm the FreeLLMAPI app is running."
            ) from exc

        if response.status_code != 200:
            raise RuntimeError(
                f"Local FreeLLMAPI request failed: HTTP {response.status_code}: "
                f"{response.text[:1000]}"
            )

        data = response.json()
        choices = data.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        content = message.get("content") if isinstance(message, dict) else None

        if isinstance(content, list):
            content = "\\n".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ).strip()

        if not isinstance(content, str) or not content.strip():
            finish_reason = choices[0].get("finish_reason", "unknown") if choices else "no choices"
            raise RuntimeError(
                "Local FreeLLMAPI returned no text in message.content. "
                f"finish_reason={finish_reason}; response_keys={sorted(data.keys())}"
            )
        return content.strip()
