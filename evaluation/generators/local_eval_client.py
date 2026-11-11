"""Dedicated FreeLLMAPI client for offline/local evaluation generation.

This client intentionally never falls back to OPENROUTER_API_KEY or the
application's production LLM credentials.
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

        def setting(name: str, *aliases: str) -> str | None:
            for key in (name, *aliases):
                value = os.getenv(key) or file_values.get(key)
                if value and str(value).strip():
                    return str(value).strip()
            return None

        self.api_key = setting("LOCAL_EVAL_API")
        self.base_url = setting(
            "LOCAL_EVAL_BASE_URL", "LOCAL_EVAL_API_URL", "FREE_LLM_API_BASE_URL"
        )
        self.model = setting("LOCAL_EVAL_MODEL", "FREE_LLM_API_MODEL")

        missing = [
            name for name, value in (
                ("LOCAL_EVAL_API", self.api_key),
                ("LOCAL_EVAL_BASE_URL", self.base_url),
                ("LOCAL_EVAL_MODEL", self.model),
            ) if not value
        ]
        if missing:
            raise RuntimeError(
                "Evaluation LLM configuration missing: "
                + ", ".join(missing)
                + ". Set these in .env. Evaluation is deliberately isolated "
                  "from OPENROUTER_API_KEY and the application's LLM settings."
            )

        if not self.base_url.endswith("/chat/completions"):
            self.base_url = self.base_url.rstrip("/") + "/chat/completions"

    async def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 2048,
        }

        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(self.base_url, json=payload, headers=headers)

        if response.status_code != 200:
            raise RuntimeError(
                f"FreeLLMAPI evaluation request failed: HTTP {response.status_code}: "
                f"{response.text[:1000]}"
            )

        data = response.json()
        choices = data.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        content = message.get("content") if isinstance(message, dict) else None

        if isinstance(content, list):
            content = "\\n".join(
                part.get("text", "") for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ).strip()

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(
                "FreeLLMAPI returned HTTP 200 without message.content. "
                f"model={self.model}, finish_reason="
                f"{choices[0].get('finish_reason', 'unknown') if choices else 'no choices'}"
            )
        return content
