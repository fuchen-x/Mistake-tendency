from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .clients import create_client
from .config import provider_config, slug
from .io_utils import append_jsonl, extract_json_object, read_jsonl
from .prompts import PROFILE_SYSTEM_PROMPT, profile_user_prompt


def profile_output_path(config: dict[str, Any], provider: str | None = None) -> Path:
    selected = provider_config(config, provider)
    filename = f"profiles__{slug(selected['name'])}__{slug(selected['model'])}.jsonl"
    return config["paths"]["result_dir"] / filename


def _validate_profile(profile: dict[str, Any]) -> dict[str, Any]:
    tendencies = profile.get("tendencies", [])
    if not isinstance(tendencies, list):
        raise ValueError("profile.tendencies must be a list")
    cleaned = []
    for index, tendency in enumerate(tendencies, start=1):
        if not isinstance(tendency, dict):
            continue
        tendency.setdefault("tendency_id", f"T{index:02d}")
        for field in ("strength", "confidence"):
            try:
                tendency[field] = min(1.0, max(0.0, float(tendency.get(field, 0.0))))
            except (TypeError, ValueError):
                tendency[field] = 0.0
        tendency.setdefault("scope", {"skill_codes": [], "answer_types": []})
        tendency.setdefault("expected_error_features", [])
        tendency.setdefault("evidence_event_ids", [])
        tendency.setdefault("prototype_wrong_answers", [])
        tendency.setdefault("counterevidence_event_ids", [])
        cleaned.append(tendency)
    profile["tendencies"] = cleaned
    profile.setdefault("profile_version", "stage1_v1")
    profile.setdefault("global_patterns", {})
    return profile


def generate_profiles(
    config: dict[str, Any],
    provider: str | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> Path:
    selected = provider_config(config, provider)
    output_path = profile_output_path(config, provider)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and output_path.exists():
        output_path.unlink()

    completed = set()
    if output_path.exists():
        completed = {record["student_id"] for record in read_jsonl(output_path) if record.get("status") == "ok"}

    inputs = list(read_jsonl(config["paths"]["processed_data_dir"] / "profile_inputs.jsonl"))
    inputs = [record for record in inputs if record["student_id"] not in completed]
    if limit is not None:
        inputs = inputs[:limit]
    retries = int(config["experiment"].get("json_retries", 3))
    concurrency = int(config["experiment"].get("concurrency", 1))

    def process(record: dict[str, Any]) -> dict[str, Any]:
        client = create_client(selected)
        started = time.perf_counter()
        last_error: Exception | None = None
        raw_response = ""
        generation = None
        for _ in range(retries):
            try:
                generation = client.generate_json(
                    PROFILE_SYSTEM_PROMPT, profile_user_prompt(record)
                )
                raw_response = generation.text
                profile = _validate_profile(extract_json_object(raw_response))
                return {
                    "student_id": record["student_id"],
                    "status": "ok",
                    "provider": selected["name"],
                    "model": selected["model"],
                    "profile": profile,
                    "cutoff_time": record.get("cutoff_time"),
                    "history_event_count": len(record["error_history"]),
                    "latency_seconds": round(time.perf_counter() - started, 4),
                    "request_id": generation.request_id,
                    "usage": generation.usage,
                }
            except Exception as exc:  # API and schema failures are checkpointed.
                last_error = exc
        return {
            "student_id": record["student_id"],
            "status": "error",
            "provider": selected["name"],
            "model": selected["model"],
            "cutoff_time": record.get("cutoff_time"),
            "history_event_count": len(record["error_history"]),
            "latency_seconds": round(time.perf_counter() - started, 4),
            "error": repr(last_error),
            "raw_response": raw_response[:2000],
        }

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = [executor.submit(process, record) for record in inputs]
        for future in as_completed(futures):
            result = future.result()
            append_jsonl(output_path, result)
            print(f"profile {result['student_id']}: {result['status']}")
    return output_path

