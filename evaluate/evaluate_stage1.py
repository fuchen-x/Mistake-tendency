from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _bootstrap_ci(values: np.ndarray, iterations: int, seed: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _normalize_probabilities(raw: dict[str, Any], labels: list[str]) -> dict[str, float]:
    values = {}
    for label in labels:
        try:
            values[label] = max(0.0, float(raw.get(label, 0.0)))
        except (TypeError, ValueError):
            values[label] = 0.0
    total = sum(values.values())
    if total <= 0:
        return {label: 1.0 / len(labels) for label in labels}
    return {label: value / total for label, value in values.items()}


def _score_record(probabilities: dict[str, float], actual: str, floor: float) -> dict[str, float]:
    ordered = sorted(probabilities, key=probabilities.get, reverse=True)
    rank = ordered.index(actual) + 1 if actual in ordered else len(ordered) + 1
    actual_probability = max(floor, probabilities.get(actual, 0.0))
    brier = sum(
        (probability - (1.0 if label == actual else 0.0)) ** 2
        for label, probability in probabilities.items()
    )
    return {
        "nll": -math.log(actual_probability),
        "top1": float(bool(ordered) and ordered[0] == actual),
        "top2": float(actual in ordered[:2]),
        "mrr": 1.0 / rank,
        "brier": brier,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _make_baselines(
    events_path: Path, tasks: list[dict[str, Any]], smoothing: float = 1.0
) -> dict[str, dict[str, dict[str, float]]]:
    events = pd.read_parquet(events_path)
    train = events[
        events["split"].eq("train")
        & events["is_mapped_single_wrong"]
        & events["selected_option"].notna()
    ].copy()
    global_counts = Counter(train["selected_option"].astype(str))
    item_counts: dict[str, Counter] = defaultdict(Counter)
    skill_counts: dict[str, Counter] = defaultdict(Counter)
    for row in train.itertuples(index=False):
        label = str(row.selected_option)
        item_counts[str(row.problem_id)][label] += 1
        try:
            codes = json.loads(row.skill_codes_json)
        except (TypeError, json.JSONDecodeError):
            codes = []
        for code in codes:
            skill_counts[str(code)][label] += 1

    outputs = {"item_baseline": {}, "skill_baseline": {}, "global_baseline": {}}
    for task in tasks:
        labels = task["wrong_option_labels"]
        task_id = str(task["task_id"])

        def distribution(counts: Counter) -> dict[str, float]:
            raw = {label: counts.get(label, 0) + smoothing for label in labels}
            total = sum(raw.values())
            return {label: value / total for label, value in raw.items()}

        outputs["global_baseline"][task_id] = distribution(global_counts)
        item_counter = item_counts.get(str(task["problem_id"]), global_counts)
        outputs["item_baseline"][task_id] = distribution(item_counter)
        combined_skill = Counter()
        for code in task.get("skill_codes", []):
            combined_skill.update(skill_counts.get(str(code), Counter()))
        outputs["skill_baseline"][task_id] = distribution(combined_skill or global_counts)
    return outputs


def evaluate(
    config_path: Path,
    split: str,
    prediction_specs: list[str],
    output_dir: Path | None,
) -> None:
    project_root = config_path.resolve().parent.parent
    sys.path.insert(0, str(project_root / "src"))
    from mistake_tendency.config import load_config

    config = load_config(config_path)
    processed = config["paths"]["processed_data_dir"]
    result_dir = output_dir or config["paths"]["result_dir"] / "evaluation"
    result_dir.mkdir(parents=True, exist_ok=True)
    tasks = _read_jsonl(processed / f"{split}_tasks.jsonl")
    task_by_id = {str(task["task_id"]): task for task in tasks}

    condition_predictions: dict[str, dict[str, dict[str, float]]] = _make_baselines(
        processed / "events.parquet", tasks
    )
    for spec in prediction_specs:
        if "=" not in spec:
            raise ValueError("--prediction must be condition=path")
        condition, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        if not path.is_absolute():
            path = project_root / path
        predictions = {}
        for record in _read_jsonl(path):
            if record.get("status") == "ok":
                predictions[str(record["task_id"])] = record["probabilities"]
        condition_predictions[condition] = predictions

    floor = float(config["experiment"].get("probability_floor", 1e-12))
    scored_rows = []
    for condition, predictions in condition_predictions.items():
        for task_id, raw_probabilities in predictions.items():
            task = task_by_id.get(task_id)
            if task is None:
                continue
            probabilities = _normalize_probabilities(raw_probabilities, task["wrong_option_labels"])
            scores = _score_record(probabilities, task["actual_wrong_option"], floor)
            scored_rows.append(
                {
                    "condition": condition,
                    "student_id": task["student_id"],
                    "task_id": task_id,
                    **scores,
                }
            )
    scored = pd.DataFrame(scored_rows)
    if scored.empty:
        raise RuntimeError("No predictions matched the selected test tasks")

    student_metrics = (
        scored.groupby(["condition", "student_id"], as_index=False)[
            ["nll", "top1", "top2", "mrr", "brier"]
        ]
        .mean()
    )
    coverage = scored.groupby("condition")["task_id"].nunique() / len(tasks)
    iterations = int(config["experiment"].get("bootstrap_iterations", 1000))
    seed = int(config["preprocess"].get("seed", 20261020))
    summary_rows = []
    for condition, group in student_metrics.groupby("condition"):
        row: dict[str, Any] = {
            "condition": condition,
            "students": int(group["student_id"].nunique()),
            "tasks": int(scored[scored["condition"].eq(condition)]["task_id"].nunique()),
            "task_coverage": float(coverage.get(condition, 0.0)),
        }
        for metric in ("nll", "top1", "top2", "mrr", "brier"):
            values = group[metric].to_numpy(dtype=float)
            low, high = _bootstrap_ci(values, iterations, seed)
            row[metric] = float(np.mean(values))
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("nll")

    comparison_rows = []
    conditions = sorted(student_metrics["condition"].unique())
    references = [name for name in ("no_profile", "item_baseline") if name in conditions]
    for reference in references:
        reference_frame = student_metrics[student_metrics["condition"].eq(reference)].set_index(
            "student_id"
        )
        for condition in conditions:
            if condition == reference:
                continue
            candidate = student_metrics[student_metrics["condition"].eq(condition)].set_index(
                "student_id"
            )
            common = reference_frame.index.intersection(candidate.index)
            if len(common) == 0:
                continue
            for metric in ("nll", "top1"):
                difference = candidate.loc[common, metric].to_numpy() - reference_frame.loc[
                    common, metric
                ].to_numpy()
                low, high = _bootstrap_ci(difference, iterations, seed)
                comparison_rows.append(
                    {
                        "condition": condition,
                        "reference": reference,
                        "metric": metric,
                        "students": len(common),
                        "mean_difference": float(np.mean(difference)),
                        "ci_low": low,
                        "ci_high": high,
                        "better_than_reference": bool(
                            high < 0 if metric == "nll" else low > 0
                        ),
                    }
                )

    summary.to_csv(result_dir / f"metrics_summary_{split}.csv", index=False, encoding="utf-8-sig")
    student_metrics.to_csv(
        result_dir / f"student_metrics_{split}.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(comparison_rows).to_csv(
        result_dir / f"pairwise_comparisons_{split}.csv", index=False, encoding="utf-8-sig"
    )
    print(summary.to_string(index=False))
    print(f"Saved evaluation to {result_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate stage-1 wrong-option predictions")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        help="condition=path; repeat for multiple model conditions",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    evaluate(args.config, args.split, args.prediction, args.output_dir)


if __name__ == "__main__":
    main()
