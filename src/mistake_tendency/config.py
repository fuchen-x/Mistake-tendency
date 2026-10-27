from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and resolve project-relative paths."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    project_root = (config_path.parent / config.get("project_root", "..")).resolve()
    config["project_root"] = project_root
    config["config_path"] = config_path

    paths = config.setdefault("paths", {})
    for key in ("raw_data_dir", "processed_data_dir", "result_dir"):
        value = Path(paths[key]).expanduser()
        if not value.is_absolute():
            value = project_root / value
        paths[key] = value.resolve()

    load_dotenv(project_root / ".env", override=False)
    return config


def provider_config(config: dict[str, Any], provider: str | None = None) -> dict[str, Any]:
    name = provider or config.get("provider", "deepseek")
    providers = config.get("providers", {})
    if name not in providers:
        raise KeyError(f"Unknown provider {name!r}. Available: {sorted(providers)}")
    selected = copy.deepcopy(providers[name])
    selected["name"] = name

    base_url_env = selected.get("base_url_env")
    if base_url_env and os.getenv(base_url_env):
        selected["base_url"] = os.environ[base_url_env]
    return selected


def ensure_output_dirs(config: dict[str, Any]) -> None:
    config["paths"]["processed_data_dir"].mkdir(parents=True, exist_ok=True)
    config["paths"]["result_dir"].mkdir(parents=True, exist_ok=True)


def slug(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    return "".join(ch if ch in allowed else "-" for ch in value).strip("-")
