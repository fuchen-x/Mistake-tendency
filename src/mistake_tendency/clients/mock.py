from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .base import GenerationResult, LLMClient


class MockClient(LLMClient):
    """Deterministic offline client for pipeline tests; it is not an experimental model."""

    def __init__(self, provider_name: str, settings: dict[str, Any]) -> None:
        super().__init__(provider_name, settings.get("model", "deterministic-mock"))

    def generate_json(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        payload = json.loads(user_prompt)
        task = payload.get("task")
        if task == "infer_profile":
            history = payload.get("error_history", [])
            counts = Counter(
                event.get("student_option")
                for event in history
                if event.get("student_option")
            )
            total = sum(counts.values()) or 1
            bias = {key: value / total for key, value in sorted(counts.items())}
            output = {
                "profile_version": "stage1_v1",
                "tendencies": [
                    {
                        "tendency_id": "MOCK_POSITION_BIAS",
                        "description": "Historical wrong-option position preference.",
                        "scope": {"skill_codes": [], "answer_types": ["Multiple Choice"]},
                        "expected_error_features": ["option_position_bias"],
                        "strength": max(bias.values(), default=0.0),
                        "confidence": min(1.0, total / 20),
                        "evidence_event_ids": [event["event_id"] for event in history[:5]],
                        "prototype_wrong_answers": [],
                        "counterevidence_event_ids": [],
                    }
                ],
                "global_patterns": {"option_position_bias": bias},
            }
        elif task == "predict_wrong_options":
            profile = payload.get("profile") or {}
            bias = profile.get("global_patterns", {}).get("option_position_bias", {})
            output_predictions = []
            for item in payload.get("items", []):
                labels = item["wrong_option_labels"]
                raw = {label: float(bias.get(label, 1.0)) for label in labels}
                total = sum(raw.values()) or len(labels) or 1
                probs = {label: value / total for label, value in raw.items()}
                predicted = max(probs, key=probs.get) if probs else None
                output_predictions.append(
                    {
                        "task_id": item["task_id"],
                        "probabilities": probs,
                        "predicted_option": predicted,
                        "activated_tendencies": ["MOCK_POSITION_BIAS"] if bias else [],
                    }
                )
            output = {"predictions": output_predictions}
        else:
            raise ValueError(f"Unsupported mock task: {task!r}")
        return GenerationResult(text=json.dumps(output, ensure_ascii=False), request_id="mock")

