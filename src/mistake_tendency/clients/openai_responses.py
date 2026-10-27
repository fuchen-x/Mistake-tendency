from __future__ import annotations

import os
from typing import Any

from .base import GenerationResult, LLMClient


class OpenAIResponsesClient(LLMClient):
    """GPT adapter using the OpenAI Responses API."""

    def __init__(self, provider_name: str, settings: dict[str, Any]) -> None:
        super().__init__(provider_name, settings["model"])
        api_key = os.getenv(settings["api_key_env"], "")
        if not api_key:
            raise RuntimeError(f"Missing environment variable {settings['api_key_env']}")
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key,
            timeout=float(settings.get("timeout_seconds", 180)),
            max_retries=int(settings.get("max_retries", 3)),
        )
        self.settings = settings

    def generate_json(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        response = self.client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=user_prompt,
            max_output_tokens=int(self.settings.get("max_tokens", 4096)),
        )
        usage = getattr(response, "usage", None)
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        elif usage is None:
            usage = {}
        return GenerationResult(
            text=response.output_text or "",
            request_id=getattr(response, "_request_id", None),
            usage=usage,
        )

