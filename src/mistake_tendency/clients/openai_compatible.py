from __future__ import annotations

import os
from typing import Any

from .base import GenerationResult, LLMClient


def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    result = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, field, None)
        if value is not None:
            result[field] = value
    return result


class OpenAICompatibleClient(LLMClient):
    """Chat Completions adapter used by DeepSeek and Qwen-compatible endpoints."""

    def __init__(self, provider_name: str, settings: dict[str, Any]) -> None:
        super().__init__(provider_name, settings["model"])
        api_key = os.getenv(settings["api_key_env"], "")
        if not api_key:
            raise RuntimeError(f"Missing environment variable {settings['api_key_env']}")
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key,
            base_url=settings.get("base_url"),
            timeout=float(settings.get("timeout_seconds", 180)),
            max_retries=int(settings.get("max_retries", 3)),
        )
        self.settings = settings

    def generate_json(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        base_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": int(self.settings.get("max_tokens", 4096)),
            "stream": False,
        }
        temperature = self.settings.get("temperature")
        if temperature is not None:
            base_kwargs["temperature"] = float(temperature)

        # DeepSeek enables thinking mode by default.  Only send this vendor-specific
        # field when it is explicitly configured, so Qwen-compatible endpoints are
        # unaffected.
        thinking = self.settings.get("thinking")
        if thinking is not None:
            base_kwargs["extra_body"] = {
                "thinking": {"type": str(thinking).strip().lower()}
            }

        json_mode = bool(self.settings.get("json_mode", True))
        fallback = bool(self.settings.get("json_mode_fallback", False))
        modes = [json_mode]
        if json_mode and fallback:
            modes.append(False)

        empty_diagnostics: list[str] = []
        for use_json_mode in modes:
            kwargs = dict(base_kwargs)
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(**kwargs)
            if not response.choices:
                empty_diagnostics.append(
                    f"json_mode={use_json_mode}, choices=0, "
                    f"usage={_usage_dict(getattr(response, 'usage', None))}"
                )
                continue

            choice = response.choices[0]
            message = choice.message
            content = message.content or ""
            if content.strip():
                return GenerationResult(
                    text=content,
                    request_id=getattr(response, "_request_id", None),
                    usage=_usage_dict(getattr(response, "usage", None)),
                )

            reasoning = getattr(message, "reasoning_content", None) or ""
            empty_diagnostics.append(
                f"json_mode={use_json_mode}, "
                f"finish_reason={getattr(choice, 'finish_reason', None)!r}, "
                f"reasoning_chars={len(reasoning)}, "
                f"usage={_usage_dict(getattr(response, 'usage', None))}"
            )

        details = "; ".join(empty_diagnostics)
        raise RuntimeError(
            f"{self.provider_name} API returned empty content: {details}"
        )

