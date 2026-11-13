"""Dedicated local Ollama client for evaluation.

Evaluation credentials and inference stay isolated from application providers.
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
        ).replace("localhost", "127.0.0.1").rstrip("/")
        self.model = setting("LOCAL_EVAL_MODEL") or "qwen3:1.7b"
        self.api_key = setting("LOCAL_EVAL_API_KEY") or "ollama"
        self.endpoint = (
            self.base_url
            if self.base_url.endswith("/chat/completions")
            else f"{self.base_url}/chat/completions"
        )

    async def _generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt + "\n/no_think",
                },
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0 if json_mode else 0.3,
            "max_tokens": 512 if json_mode else 4096,
            "stream": False,
            "frequency_penalty": 0.2 if json_mode else 0.0,
        }
        if json_mode:
            # Ollama's OpenAI-compatible endpoint supports JSON response mode.
            payload["response_format"] = {"type": "json_object"}

        timeout = httpx.Timeout(
            connect=10.0,
            read=900.0,
            write=60.0,
            pool=30.0,
        )
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"Ollama timed out while generating with model {self.model} "
                f"at {self.endpoint}: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:1200]
            raise RuntimeError(
                f"Ollama returned HTTP {exc.response.status_code}: {body}"
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.endpoint}: "
                f"{type(exc).__name__}: {exc}. Check that Ollama is running."
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Ollama returned non-JSON HTTP content: {response.text[:800]!r}"
            ) from exc

        choices = data.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        content = message.get("content") if isinstance(message, dict) else None

        if isinstance(content, list):
            content = "\n".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ).strip()

        if not isinstance(content, str) or not content.strip():
            finish_reason = (
                choices[0].get("finish_reason", "unknown")
                if choices else "no choices"
            )
            raise RuntimeError(
                "Ollama returned no text in message.content. "
                f"finish_reason={finish_reason}; response_keys={sorted(data.keys())}"
            )
        return content.strip()

    async def generate_response(
        self, system_prompt: str, user_prompt: str
    ) -> str:
        return await self._generate(
            system_prompt, user_prompt, json_mode=False
        )

    async def generate_json_response(
        self, system_prompt: str, user_prompt: str
    ) -> str:
        return await self._generate(
            system_prompt, user_prompt, json_mode=True
        )
