from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .clients import create_client
from .config import provider_config, slug
from .io_utils import append_jsonl, extract_json_object, read_jsonl
from .prompts import PREDICTION_SYSTEM_PROMPT, prediction_user_prompt


VALID_CONDITIONS = {"no_profile", "profile", "raw_history", "shuffled_profile"}


def prediction_output_path(
    config: dict[str, Any], provider: str | None, condition: str, split: str
) -> Path:
    selected = provider_config(config, provider)
    filename = (
        f"predictions__{slug(selected['name'])}__{slug(selected['model'])}__"
        f"{slug(condition)}__{slug(split)}.jsonl"
    )
    return config["paths"]["result_dir"] / filename


def _normalize_predictions(
    parsed: dict[str, Any], items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    expected = {str(item["task_id"]): item for item in items}
    received = {
        str(item.get("task_id")): item
        for item in parsed.get("predictions", [])
        if isinstance(item, dict)
    }
    results = []
    for task_id, task in expected.items():
        prediction = received.get(task_id)
        if prediction is None:
            raise ValueError(f"Missing prediction for task {task_id}")
        labels = task["wrong_option_labels"]
        raw = prediction.get("probabilities", {})
        probabilities = {}
        for label in labels:
            try:
                probabilities[label] = max(0.0, float(raw.get(label, 0.0)))
            except (TypeError, ValueError):
                probabilities[label] = 0.0
        total = sum(probabilities.values())
        if total <= 0:
            raise ValueError(f"Nonpositive probability total for task {task_id}")
        probabilities = {label: value / total for label, value in probabilities.items()}
        predicted = max(probabilities, key=probabilities.get)
        results.append(
            {
                "task_id": task_id,
                "probabilities": probabilities,
                "predicted_option": predicted,
                "activated_tendencies": prediction.get("activated_tendencies", []),
            }
        )
    return results


def predict_errors(
    config: dict[str, Any],
    condition: str,
    split: str = "test",
    provider: str | None = None,
    profiles_path: str | Path | None = None,
    limit_students: int | None = None,
    overwrite: bool = False,
) -> Path:
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; choose from {sorted(VALID_CONDITIONS)}")
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")

    selected = provider_config(config, provider)
    output_path = prediction_output_path(config, provider, condition, split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and output_path.exists():
        output_path.unlink()
    completed = set()
    if output_path.exists():
        completed = {
            str(record["task_id"])
            for record in read_jsonl(output_path)
            if record.get("status") == "ok"
        }

    processed_dir = config["paths"]["processed_data_dir"]
    tasks = list(read_jsonl(processed_dir / f"{split}_tasks.jsonl"))
    histories = {
        record["student_id"]: record["error_history"]
        for record in read_jsonl(processed_dir / "profile_inputs.jsonl")
    }

    profiles: dict[str, dict[str, Any]] = {}
    if condition in {"profile", "shuffled_profile"}:
        if profiles_path is None:
            from .generate_profiles import profile_output_path

            profiles_path = profile_output_path(config, provider)
        profiles = {
            record["student_id"]: record["profile"]
            for record in read_jsonl(profiles_path)
            if record.get("status") == "ok"
        }

    by_student: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        if str(task["task_id"]) in completed:
            continue
        by_student.setdefault(task["student_id"], []).append(task)
    student_ids = sorted(by_student)
    if condition in {"profile", "shuffled_profile"}:
        student_ids = [student_id for student_id in student_ids if student_id in profiles]
    if limit_students is not None:
        student_ids = student_ids[:limit_students]

    shuffled_map: dict[str, str] = {}
    if condition == "shuffled_profile":
        source_ids = student_ids.copy()
        if len(source_ids) > 1:
            rng = random.Random(int(config["preprocess"].get("seed", 20261020)))
            offset = rng.randrange(1, len(source_ids))
            source_ids = source_ids[offset:] + source_ids[:offset]
        shuffled_map = dict(zip(student_ids, source_ids))

    batch_size = int(config["experiment"].get("batch_size", 10))
    jobs: list[tuple[str, list[dict[str, Any]]]] = []
    for student_id in student_ids:
        student_tasks = by_student[student_id]
        for start in range(0, len(student_tasks), batch_size):
            jobs.append((student_id, student_tasks[start : start + batch_size]))

    retries = int(config["experiment"].get("json_retries", 3))
    concurrency = int(config["experiment"].get("concurrency", 1))

    def process(student_id: str, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        client = create_client(selected)
        profile = None
        history = None
        profile_source = None
        if condition == "profile":
            profile = profiles[student_id]
            profile_source = student_id
        elif condition == "shuffled_profile":
            profile_source = shuffled_map[student_id]
            profile = profiles[profile_source]
        elif condition == "raw_history":
            history = histories[student_id]

        prompt = prediction_user_prompt(batch, condition, profile, history)
        started = time.perf_counter()
        last_error: Exception | None = None
        raw_response = ""
        for _ in range(retries):
            try:
                generation = client.generate_json(PREDICTION_SYSTEM_PROMPT, prompt)
                raw_response = generation.text
                parsed = extract_json_object(raw_response)
                predictions = _normalize_predictions(parsed, batch)
                elapsed = round(time.perf_counter() - started, 4)
                return [
                    {
                        "student_id": student_id,
                        "task_id": prediction["task_id"],
                        "status": "ok",
                        "split": split,
                        "condition": condition,
                        "provider": selected["name"],
                        "model": selected["model"],
                        "profile_source_student_id": profile_source,
                        "probabilities": prediction["probabilities"],
                        "predicted_option": prediction["predicted_option"],
                        "activated_tendencies": prediction["activated_tendencies"],
                        "latency_seconds_batch": elapsed,
                        "request_id": generation.request_id,
                        "usage_batch": generation.usage,
                    }
                    for prediction in predictions
                ]
            except Exception as exc:
                last_error = exc
        return [
            {
                "student_id": student_id,
                "task_id": str(task["task_id"]),
                "status": "error",
                "split": split,
                "condition": condition,
                "provider": selected["name"],
                "model": selected["model"],
                "error": repr(last_error),
                "raw_response": raw_response[:2000],
            }
            for task in batch
        ]

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        future_map = {
            executor.submit(process, student_id, batch): (student_id, len(batch))
            for student_id, batch in jobs
        }
        for future in as_completed(future_map):
            student_id, count = future_map[future]
            records = future.result()
            for record in records:
                append_jsonl(output_path, record)
            status = "ok" if all(record["status"] == "ok" for record in records) else "error"
            print(f"predict {student_id} ({count} tasks): {status}")
    return output_path

