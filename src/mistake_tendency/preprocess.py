from __future__ import annotations

import html
import json
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ensure_output_dirs
from .io_utils import write_jsonl


TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
CHOICE_PREFIX_RE = re.compile(r"^\s*[A-Ha-h]\s*[\.\)\]:：、-]\s*")
PUNCT_RE = re.compile(r"[\s\u00a0\.,;:!?，。；：！？'\"`]+")
FULL_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\d*\.\d+)(?:%|/\d+)?$")


def html_to_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value)
    text = re.sub(r"<\s*br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"</\s*p\s*>", " ", text, flags=re.I)
    text = TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = text.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u00d7", "*").replace("\u00f7", "/")
    return SPACE_RE.sub(" ", text).strip()


def normalize_answer(value: Any, remove_choice_prefix: bool = True) -> str:
    text = html_to_text(value).casefold()
    if remove_choice_prefix:
        text = CHOICE_PREFIX_RE.sub("", text)
    text = text.replace("&nbsp;", " ")
    text = PUNCT_RE.sub("", text)
    return text.strip()


def split_multi_value(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [part.strip() for part in str(value).split("||") if part.strip()]


def option_label(index: int) -> str:
    return chr(ord("A") + index)


def parse_numeric(value: Any) -> Fraction | None:
    text = html_to_text(value).replace(",", "").replace(" ", "")
    if not FULL_NUMBER_RE.fullmatch(text):
        return None
    try:
        if text.endswith("%"):
            return Fraction(text[:-1]) / 100
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def map_answer_to_option(answer: Any, options: list[str]) -> tuple[str | None, str, float]:
    """Map raw answer text to one option without using the correctness label."""
    if not options or answer is None or not str(answer).strip():
        return None, "empty", 0.0

    answer_norm = normalize_answer(answer)
    option_norms = [normalize_answer(option) for option in options]
    exact = [idx for idx, value in enumerate(option_norms) if value == answer_norm]
    if len(exact) == 1:
        return option_label(exact[0]), "normalized_exact", 1.0

    answer_number = parse_numeric(answer)
    if answer_number is not None:
        numeric_matches = [
            idx for idx, option in enumerate(options) if parse_numeric(option) == answer_number
        ]
        if len(numeric_matches) == 1:
            return option_label(numeric_matches[0]), "numeric_equivalent", 0.99

    scores = [SequenceMatcher(None, answer_norm, value).ratio() for value in option_norms]
    if scores:
        order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        best = order[0]
        runner_up = scores[order[1]] if len(order) > 1 else 0.0
        if scores[best] >= 0.93 and scores[best] - runner_up >= 0.05:
            return option_label(best), "fuzzy_unique", float(scores[best])
    return None, "unresolved", 0.0


def _safe_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def _event_payload(row: pd.Series, max_problem_chars: int) -> dict[str, Any]:
    options = json.loads(row["options_json"]) if row["options_json"] else {}
    return {
        "event_id": str(row["id"]),
        "problem_id": str(row["problem_id"]),
        "timestamp": row["end_time"],
        "problem_type": _safe_text(row["problem_type"]),
        "answer_type": _safe_text(row["answer_type"]),
        "question": _safe_text(row["stem_text"])[:max_problem_chars],
        "options": options,
        "correct_answer": _safe_text(row["correct_answer_text"]),
        "correct_option": row["correct_option"] if pd.notna(row["correct_option"]) else None,
        "student_answer": _safe_text(row["answer_text"]),
        "student_option": row["selected_option"] if pd.notna(row["selected_option"]) else None,
        "skill_codes": json.loads(row["skill_codes_json"]),
        "skill_names": json.loads(row["skill_names_json"]),
    }


def preprocess_dataset(config: dict[str, Any]) -> dict[str, Any]:
    ensure_output_dirs(config)
    raw_dir: Path = config["paths"]["raw_data_dir"]
    output_dir: Path = config["paths"]["processed_data_dir"]
    settings = config["preprocess"]

    interactions_path = raw_dir / "Interactions.csv"
    problems_path = raw_dir / "Problems.csv"
    skills_path = raw_dir / "Skills.csv"
    skill_set_path = raw_dir / "Skill_Set.csv"
    for path in (interactions_path, problems_path, skills_path, skill_set_path):
        if not path.exists():
            raise FileNotFoundError(path)

    problems_raw = pd.read_csv(problems_path, dtype=str)
    excluded = set(str(value) for value in settings.get("exclude_problem_ids", []))
    problems = problems_raw[~problems_raw["problem_id"].isin(excluded)].drop_duplicates(
        "problem_id", keep="first"
    ).copy()
    problems = problems.rename(
        columns={
            "Problem Type": "problem_type",
            "Answer Types": "answer_type",
            "Problem Body": "problem_body_html",
            "Multiple Choice Options": "mc_options_raw",
            "Multiple Choice Answers": "mc_answers_raw",
            "Fill-in Answers": "fill_answers_raw",
        }
    )
    problems["stem_text"] = problems["problem_body_html"].map(html_to_text)

    def build_options(value: Any) -> str:
        parts = split_multi_value(value)
        return json.dumps(
            {option_label(index): html_to_text(text) for index, text in enumerate(parts)},
            ensure_ascii=False,
        )

    problems["options_json"] = problems["mc_options_raw"].map(build_options)
    problems["correct_answer_text"] = problems["mc_answers_raw"].fillna(
        problems["fill_answers_raw"]
    )

    correct_labels: list[str | None] = []
    correct_methods: list[str] = []
    for row in problems.itertuples(index=False):
        options = split_multi_value(row.mc_options_raw)
        label, method, _ = map_answer_to_option(row.mc_answers_raw, options)
        correct_labels.append(label)
        correct_methods.append(method)
    problems["correct_option"] = correct_labels
    problems["correct_option_map_method"] = correct_methods

    skill_set = pd.read_csv(skill_set_path, dtype=str)
    description_by_code = dict(zip(skill_set["skill_code"], skill_set["full_description"]))
    skills = pd.read_csv(skills_path, dtype=str)
    skills["description"] = skills["node_code"].map(description_by_code).fillna("")

    def unique_list(values: pd.Series) -> list[str]:
        return list(dict.fromkeys(str(value) for value in values if pd.notna(value) and str(value)))

    skill_groups = skills.groupby("problem_id", sort=False).agg(
        skill_codes=("node_code", unique_list),
        skill_names=("node_name", unique_list),
        skill_descriptions=("description", unique_list),
    )
    for column in ("skill_codes", "skill_names", "skill_descriptions"):
        skill_groups[f"{column}_json"] = skill_groups[column].map(
            lambda values: json.dumps(values, ensure_ascii=False)
        )

    use_columns = [
        "id",
        "problem_id",
        "hint_count",
        "answer_text",
        "saw_answer",
        "discrete_score",
        "end_time",
        "user_id",
    ]
    interactions = pd.read_csv(interactions_path, usecols=use_columns, dtype=str)
    raw_interaction_rows = len(interactions)
    interactions = interactions.drop_duplicates("id", keep="first")
    deduplicated_rows = len(interactions)
    interactions = interactions[interactions["discrete_score"].isin(["0.0", "1.0"])]
    if settings.get("require_saw_answer_false", True):
        interactions = interactions[interactions["saw_answer"].eq("False")]
    if settings.get("require_hint_count_zero", True):
        interactions = interactions[interactions["hint_count"].eq("0")]
    interactions = interactions[
        interactions["answer_text"].fillna("").str.strip().ne("")
        & ~interactions["problem_id"].isin(excluded)
    ].copy()
    interactions["timestamp"] = pd.to_datetime(
        interactions["end_time"], errors="coerce", utc=True
    )
    interactions = interactions[interactions["timestamp"].notna()]
    interactions = interactions.sort_values(["user_id", "timestamp", "id"])
    if settings.get("keep_first_student_problem", True):
        interactions = interactions.drop_duplicates(["user_id", "problem_id"], keep="first")

    problem_columns = [
        "problem_id",
        "problem_type",
        "answer_type",
        "problem_body_html",
        "stem_text",
        "options_json",
        "correct_answer_text",
        "correct_option",
        "correct_option_map_method",
        "mc_options_raw",
    ]
    events = interactions.merge(problems[problem_columns], on="problem_id", how="inner")
    events = events.merge(
        skill_groups[["skill_codes_json", "skill_names_json", "skill_descriptions_json"]],
        left_on="problem_id",
        right_index=True,
        how="left",
    )
    for column in ("skill_codes_json", "skill_names_json", "skill_descriptions_json"):
        events[column] = events[column].fillna("[]")

    selected_options: list[str | None] = []
    selected_methods: list[str] = []
    selected_confidence: list[float] = []
    for row in events.itertuples(index=False):
        raw_options = split_multi_value(row.mc_options_raw)
        label, method, confidence = map_answer_to_option(row.answer_text, raw_options)
        selected_options.append(label)
        selected_methods.append(method)
        selected_confidence.append(confidence)
    events["selected_option"] = selected_options
    events["selected_option_map_method"] = selected_methods
    events["selected_option_map_confidence"] = selected_confidence
    events["is_wrong"] = events["discrete_score"].eq("0.0")

    events = events.sort_values(["user_id", "timestamp", "id"]).reset_index(drop=True)
    events["student_position"] = events.groupby("user_id").cumcount()
    events["student_event_count"] = events.groupby("user_id")["id"].transform("size")
    relative_position = events["student_position"] / events["student_event_count"]
    train_ratio = float(settings["train_ratio"])
    validation_end = train_ratio + float(settings["validation_ratio"])
    events["split"] = np.select(
        [relative_position < train_ratio, relative_position < validation_end],
        ["train", "validation"],
        default="test",
    )

    is_single = events["problem_type"].eq("Multiple Choice (select 1)")
    is_mapped_wrong = (
        events["is_wrong"]
        & is_single
        & events["selected_option"].notna()
        & events["correct_option"].notna()
        & events["selected_option"].ne(events["correct_option"])
    )
    events["is_mapped_single_wrong"] = is_mapped_wrong

    train_wrong = (
        events[events["split"].eq("train") & events["is_wrong"]]
        .groupby("user_id")
        .size()
    )
    test_single_wrong = (
        events[events["split"].eq("test") & events["is_mapped_single_wrong"]]
        .groupby("user_id")
        .size()
    )
    eligible = sorted(
        set(train_wrong[train_wrong >= int(settings["min_train_wrong"])].index)
        & set(test_single_wrong[test_single_wrong >= int(settings["min_test_single_wrong"])].index)
    )
    events["eligible_student"] = events["user_id"].isin(eligible)

    output_columns = [
        "id",
        "user_id",
        "problem_id",
        "end_time",
        "discrete_score",
        "is_wrong",
        "hint_count",
        "saw_answer",
        "answer_text",
        "selected_option",
        "selected_option_map_method",
        "selected_option_map_confidence",
        "problem_type",
        "answer_type",
        "stem_text",
        "options_json",
        "correct_answer_text",
        "correct_option",
        "correct_option_map_method",
        "skill_codes_json",
        "skill_names_json",
        "skill_descriptions_json",
        "split",
        "is_mapped_single_wrong",
        "eligible_student",
    ]
    events[output_columns].to_parquet(output_dir / "events.parquet", index=False)

    eligible_frame = pd.DataFrame({"student_id": eligible})
    eligible_frame["train_wrong_count"] = eligible_frame["student_id"].map(train_wrong).fillna(0).astype(int)
    eligible_frame["test_single_wrong_count"] = eligible_frame["student_id"].map(test_single_wrong).fillna(0).astype(int)
    eligible_frame.to_csv(output_dir / "eligible_students.csv", index=False, encoding="utf-8-sig")

    max_events = int(settings.get("max_profile_events", 50))
    max_problem_chars = int(settings.get("max_problem_chars", 3500))
    profile_records: list[dict[str, Any]] = []
    eligible_events = events[events["eligible_student"]]
    for student_id, group in eligible_events.groupby("user_id", sort=True):
        history = group[group["split"].eq("train") & group["is_wrong"]].tail(max_events)
        profile_records.append(
            {
                "student_id": student_id,
                "cutoff_time": group[group["split"].eq("train")]["end_time"].iloc[-1],
                "error_history": [
                    _event_payload(row, max_problem_chars) for _, row in history.iterrows()
                ],
            }
        )
    write_jsonl(output_dir / "profile_inputs.jsonl", profile_records)

    task_counts: dict[str, int] = {}
    for split in ("validation", "test"):
        subset = eligible_events[
            eligible_events["split"].eq(split) & eligible_events["is_mapped_single_wrong"]
        ]
        tasks: list[dict[str, Any]] = []
        for _, row in subset.iterrows():
            payload = _event_payload(row, max_problem_chars)
            correct = payload["correct_option"]
            payload["task_id"] = payload.pop("event_id")
            payload["wrong_option_labels"] = [
                label for label in payload["options"] if label != correct
            ]
            tasks.append(
                {
                    "student_id": row["user_id"],
                    "actual_wrong_option": row["selected_option"],
                    **payload,
                }
            )
        write_jsonl(output_dir / f"{split}_tasks.jsonl", tasks)
        task_counts[split] = len(tasks)

    mapping_status = Counter(
        events.loc[events["problem_type"].eq("Multiple Choice (select 1)"), "selected_option_map_method"]
    )
    manifest = {
        "raw_interaction_rows": raw_interaction_rows,
        "deduplicated_interaction_rows": deduplicated_rows,
        "clean_first_attempt_rows": int(len(events)),
        "students": int(events["user_id"].nunique()),
        "eligible_students": len(eligible),
        "profile_records": len(profile_records),
        "validation_tasks": task_counts.get("validation", 0),
        "test_tasks": task_counts.get("test", 0),
        "single_choice_mapping_status": dict(mapping_status),
        "settings": settings,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest
