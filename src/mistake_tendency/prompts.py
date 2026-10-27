from __future__ import annotations

import json
from typing import Any


PROFILE_SYSTEM_PROMPT = """You infer recurring student mistake tendencies from prior incorrect math responses.
Return one valid JSON object only. Do not diagnose psychology, ability, disability, or demographic traits.
Use only repeated evidence. A tendency must describe how the student tends to answer incorrectly, its scope,
and supporting event ids. If evidence is weak, return fewer tendencies and lower confidence.
Do not mention a test item because test responses are never provided.

Required JSON shape:
{
  "profile_version": "stage1_v1",
  "tendencies": [
    {
      "tendency_id": "T01",
      "description": "short behavioral error pattern",
      "scope": {"skill_codes": ["..."], "answer_types": ["..."]},
      "expected_error_features": ["machine-friendly tags"],
      "strength": 0.0,
      "confidence": 0.0,
      "evidence_event_ids": ["..."],
      "prototype_wrong_answers": ["..."],
      "counterevidence_event_ids": ["..."]
    }
  ],
  "global_patterns": {
    "numeric_overestimation": 0.0,
    "numeric_underestimation": 0.0,
    "option_position_bias": {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
  }
}

All strength and confidence values must be in [0,1]. Tendencies may use a latent identifier and prototypes
when a human-readable mechanism cannot be justified. Output JSON, not markdown."""


PREDICTION_SYSTEM_PROMPT = """Predict which incorrect option a student would choose, conditional on the
student answering the item incorrectly. Return one valid JSON object only. For each item, assign probabilities
only to wrong_option_labels. Probabilities must be nonnegative and sum to 1. Do not include the correct option.
Use the supplied profile or history only when relevant; otherwise return a near-uniform distribution.

Required JSON shape:
{
  "predictions": [
    {
      "task_id": "...",
      "probabilities": {"A": 0.5, "B": 0.5},
      "predicted_option": "A",
      "activated_tendencies": ["T01"]
    }
  ]
}
Output JSON, not markdown."""


def profile_user_prompt(profile_input: dict[str, Any]) -> str:
    payload = {
        "task": "infer_profile",
        "instruction": "Infer a compact error-tendency profile from these training-period errors.",
        "error_history": profile_input["error_history"],
    }
    return json.dumps(payload, ensure_ascii=False)


def prediction_user_prompt(
    items: list[dict[str, Any]],
    condition: str,
    profile: dict[str, Any] | None = None,
    raw_history: list[dict[str, Any]] | None = None,
) -> str:
    safe_items = []
    for item in items:
        safe_items.append(
            {
                "task_id": item["task_id"],
                "problem_id": item["problem_id"],
                "question": item["question"],
                "options": item["options"],
                "correct_option": item["correct_option"],
                "correct_answer": item["correct_answer"],
                "wrong_option_labels": item["wrong_option_labels"],
                "skill_codes": item["skill_codes"],
                "skill_names": item["skill_names"],
            }
        )
    payload: dict[str, Any] = {
        "task": "predict_wrong_options",
        "condition": condition,
        "items": safe_items,
    }
    if profile is not None:
        payload["profile"] = profile
    if raw_history is not None:
        payload["raw_error_history"] = raw_history
    return json.dumps(payload, ensure_ascii=False)

