"""OpenAI-compatible local Ollama client dedicated to benchmark evaluation.

Evaluation configuration is isolated from the application's LLM client.
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

        self.base_url = (
            setting("LOCAL_EVAL_BASE_URL") or "http://127.0.0.1:11434/v1"
        ).rstrip("/")
        self.model = setting("LOCAL_EVAL_MODEL") or "qwen3:1.7b"
        self.api_key = setting("LOCAL_EVAL_API_KEY") or "ollama"

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
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": 4096,
            "stream": False,
            "think": False,
            "response_format": {"type": "json_object"},
        }

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0)
            ) as client:
                response = await client.post(
                    self.endpoint, json=payload, headers=headers
                )
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.endpoint}. "
                "Start Ollama and confirm the model is available."
            ) from exc

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama evaluation request failed: HTTP {response.status_code}: "
                f"{response.text[:1200]}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("Ollama returned a non-JSON HTTP response.") from exc

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
            finish_reason = (
                choices[0].get("finish_reason", "unknown")
                if choices else "no choices"
            )
            reasoning = message.get("reasoning") if isinstance(message, dict) else None
            raise RuntimeError(
                "Ollama returned no message.content. "
                f"finish_reason={finish_reason}; reasoning_present={bool(reasoning)}; "
                f"response_keys={sorted(data.keys())}"
            )
        return content.strip()
