from __future__ import annotations

from typing import Any

from .anthropic_client import AnthropicClient
from .base import LLMClient
from .mock import MockClient
from .openai_compatible import OpenAICompatibleClient
from .openai_responses import OpenAIResponsesClient


def create_client(settings: dict[str, Any]) -> LLMClient:
    kind = settings["kind"]
    provider_name = settings["name"]
    if kind == "openai_compatible":
        return OpenAICompatibleClient(provider_name, settings)
    if kind == "openai_responses":
        return OpenAIResponsesClient(provider_name, settings)
    if kind == "anthropic":
        return AnthropicClient(provider_name, settings)
    if kind == "mock":
        return MockClient(provider_name, settings)
    raise ValueError(f"Unsupported provider kind: {kind!r}")
