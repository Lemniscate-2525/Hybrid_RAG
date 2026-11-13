
"""Dedicated native Ollama client for local benchmark evaluation."""

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

        base_url = (
            setting("LOCAL_EVAL_BASE_URL")
            or "http://127.0.0.1:11434"
        ).replace("localhost", "127.0.0.1").rstrip("/")

        # Accept either the native Ollama URL or the previous OpenAI URL.
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]

        self.endpoint = f"{base_url}/api/chat"
        self.model = setting("LOCAL_EVAL_MODEL") or "qwen3:1.7b"

    async def _generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool,
    ) -> str:

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,

            # Explicitly disable Qwen3 reasoning.
            "think": False,

            "options": {
                "temperature": 0,
                "num_predict": 512,
                "num_ctx": 16384,
            },
        }

        if json_mode:
            payload["format"] = "json"

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
                )
                response.raise_for_status()

        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"Ollama timed out at {self.endpoint}: {exc}"
            ) from exc

        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Ollama HTTP {exc.response.status_code}: "
                f"{exc.response.text[:1000]}"
            ) from exc

        except httpx.RequestError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.endpoint}: {exc}. "
                "Check that Ollama is running."
            ) from exc

        data = response.json()
        message = data.get("message") or {}
        content = message.get("content")

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(
                "Ollama returned empty content. "
                f"done_reason={data.get('done_reason')}; "
                f"message_keys={list(message.keys())}"
            )

        return content.strip()

    async def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        return await self._generate(
            system_prompt,
            user_prompt,
            json_mode=False,
        )

    async def generate_json_response(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        return await self._generate(
            system_prompt,
            user_prompt,
            json_mode=True,
        )
