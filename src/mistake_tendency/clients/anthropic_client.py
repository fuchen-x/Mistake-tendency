from __future__ import annotations

import os
from typing import Any

from .base import GenerationResult, LLMClient


class AnthropicClient(LLMClient):
    def __init__(self, provider_name: str, settings: dict[str, Any]) -> None:
        super().__init__(provider_name, settings["model"])
        api_key = os.getenv(settings["api_key_env"], "")
        if not api_key:
            raise RuntimeError(f"Missing environment variable {settings['api_key_env']}")
        import anthropic

        self.client = anthropic.Anthropic(
            api_key=api_key,
            timeout=float(settings.get("timeout_seconds", 180)),
            max_retries=int(settings.get("max_retries", 3)),
        )
        self.settings = settings

    def generate_json(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=int(self.settings.get("max_tokens", 4096)),
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )
        usage = getattr(message, "usage", None)
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        elif usage is None:
            usage = {}
        return GenerationResult(
            text=text,
            request_id=getattr(message, "_request_id", None),
            usage=usage,
        )

