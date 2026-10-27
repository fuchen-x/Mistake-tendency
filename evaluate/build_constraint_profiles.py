from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def probability_for_actual(record: dict[str, Any], actual: str, floor: float) -> float:
    probabilities = record.get("probabilities", {})
    try:
        return max(floor, float(probabilities.get(actual, 0.0)))
    except (TypeError, ValueError):
        return floor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach automatic validation evidence and export constraint-ready profiles"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--profile-predictions", required=True, type=Path)
    parser.add_argument("--no-profile-predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    project_root = args.config.resolve().parent.parent
    sys.path.insert(0, str(project_root / "src"))
    from mistake_tendency.config import load_config

    config = load_config(args.config)

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else project_root / path

    profiles = {
        row["student_id"]: row
        for row in read_jsonl(resolve(args.profiles))
        if row.get("status") == "ok"
    }
    tasks = {
        str(row["task_id"]): row
        for row in read_jsonl(config["paths"]["processed_data_dir"] / "validation_tasks.jsonl")
    }
    profile_predictions = {
        str(row["task_id"]): row
        for row in read_jsonl(resolve(args.profile_predictions))
        if row.get("status") == "ok"
    }
    no_profile_predictions = {
        str(row["task_id"]): row
        for row in read_jsonl(resolve(args.no_profile_predictions))
        if row.get("status") == "ok"
    }

    floor = float(config["experiment"].get("probability_floor", 1e-12))
    minimum_tasks = int(config["experiment"].get("min_profile_validation_tasks", 3))
    paired: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for task_id in set(profile_predictions).intersection(no_profile_predictions, tasks):
        task = tasks[task_id]
        actual = task["actual_wrong_option"]
        profile_record = profile_predictions[task_id]
        no_profile_record = no_profile_predictions[task_id]
        p_profile = probability_for_actual(profile_record, actual, floor)
        p_no_profile = probability_for_actual(no_profile_record, actual, floor)
        top_profile = float(profile_record.get("predicted_option") == actual)
        top_no_profile = float(no_profile_record.get("predicted_option") == actual)
        paired[task["student_id"]].append(
            (-math.log(p_profile), -math.log(p_no_profile), top_profile, top_no_profile)
        )

    exported = []
    for student_id, source in sorted(profiles.items()):
        record = copy.deepcopy(source)
        values = paired.get(student_id, [])
        task_count = len(values)
        if values:
            profile_nll = sum(value[0] for value in values) / task_count
            no_profile_nll = sum(value[1] for value in values) / task_count
            profile_top1 = sum(value[2] for value in values) / task_count
            no_profile_top1 = sum(value[3] for value in values) / task_count
            nll_gain = no_profile_nll - profile_nll
            top1_gain = profile_top1 - no_profile_top1
        else:
            profile_nll = no_profile_nll = nll_gain = 0.0
            profile_top1 = no_profile_top1 = top1_gain = 0.0
        usable = task_count >= minimum_tasks and nll_gain > 0
        validation = {
            "validation_task_count": task_count,
            "profile_nll": profile_nll,
            "no_profile_nll": no_profile_nll,
            "nll_gain": nll_gain,
            "profile_top1": profile_top1,
            "no_profile_top1": no_profile_top1,
            "top1_gain": top1_gain,
            "usable_for_constraint": usable,
        }
        record["validation"] = validation
        for tendency in record.get("profile", {}).get("tendencies", []):
            strength = float(tendency.get("strength", 0.0))
            confidence = float(tendency.get("confidence", 0.0))
            tendency["usable_for_constraint"] = usable
            tendency["effective_weight"] = strength * confidence if usable else 0.0
        exported.append(record)

    output_path = resolve(args.output)
    write_jsonl(output_path, exported)
    usable_count = sum(bool(row["validation"]["usable_for_constraint"]) for row in exported)
    print(f"Exported {len(exported)} profiles; {usable_count} passed automatic validation")
    print(output_path)


if __name__ == "__main__":
    main()
