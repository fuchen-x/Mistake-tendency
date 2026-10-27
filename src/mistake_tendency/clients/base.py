from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationResult:
    text: str
    request_id: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


class LLMClient(ABC):
    def __init__(self, provider_name: str, model: str) -> None:
        self.provider_name = provider_name
        self.model = model

    @abstractmethod
    def generate_json(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        raise NotImplementedError

