"""Configuration management for Prism."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

PRISM_DIR = Path.home() / ".prism"
CONFIG_FILE = PRISM_DIR / "config.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "openai_api_key": "",
    "semantic_scholar_api_key": "",  # optional; get one at semanticscholar.org/product/api
    "model": "gpt-4o",
    "user_profile": {
        "name": "",
        "research_areas": [],
    },
    "observation": {
        "paper_limit": 10,
        "year_from": None,          # None = no lower bound; int = earliest year
        "fields_filter": [],        # e.g. ["Computer Science"] to restrict
    },
    "thinker": {
        "num_ideas_per_paper": 3,
        "use_future_explorations": True,
    },
    "judge": {
        "comparisons_per_idea": 5,  # how many reference papers to compare against
        "novelty_threshold": 0.6,   # min win-rate to call an idea novel
    },
    "backend": {
        "type": "mock",             # "mock" | "local" | "remote"
        "base_url": "http://localhost:8000/v1",
        "api_key": "EMPTY",
        "thinker_model": "OpenMOSS-Team/SciThinker-30B",
        "judge_model": "OpenMOSS-Team/SciJudge-30B",
    },
    "active_investigation_id": None,
}


def _ensure_prism_dir() -> None:
    PRISM_DIR.mkdir(parents=True, exist_ok=True)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config() -> dict[str, Any]:
    """Load configuration from ~/.prism/config.json."""
    _ensure_prism_dir()
    if not CONFIG_FILE.exists():
        return _deep_merge({}, DEFAULT_CONFIG)
    try:
        with open(CONFIG_FILE) as f:
            data = json.load(f)
        return _deep_merge(DEFAULT_CONFIG, data)
    except (json.JSONDecodeError, OSError):
        return _deep_merge({}, DEFAULT_CONFIG)


def save_config(data: dict[str, Any]) -> None:
    """Save configuration to ~/.prism/config.json."""
    _ensure_prism_dir()
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_api_key() -> str:
    env_key = os.environ.get("OPENAI_API_KEY", "")
    if env_key:
        return env_key
    config = load_config()
    return config.get("openai_api_key", "")


def get_ss_api_key() -> str:
    """Return the Semantic Scholar API key, or empty string if not configured."""
    env_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    if env_key:
        return env_key
    config = load_config()
    return config.get("semantic_scholar_api_key", "")


def get_model() -> str:
    config = load_config()
    return config.get("model", "gpt-4o")


def get_active_investigation() -> str | None:
    config = load_config()
    return config.get("active_investigation_id")


def set_active_investigation(investigation_id: str | None) -> None:
    config = load_config()
    config["active_investigation_id"] = investigation_id
    save_config(config)


def get_user_profile() -> dict[str, Any]:
    config = load_config()
    return config.get("user_profile", DEFAULT_CONFIG["user_profile"])


def get_observation_config() -> dict[str, Any]:
    config = load_config()
    return config.get("observation", DEFAULT_CONFIG["observation"])


def get_backend_config() -> dict[str, Any]:
    config = load_config()
    return config.get("backend", DEFAULT_CONFIG["backend"])


def get_thinker_config() -> dict[str, Any]:
    config = load_config()
    return config.get("thinker", DEFAULT_CONFIG["thinker"])


def get_judge_config() -> dict[str, Any]:
    config = load_config()
    return config.get("judge", DEFAULT_CONFIG["judge"])


def get_backend() -> "ModelBackend":  # type: ignore[name-defined]  # noqa: F821
    """Instantiate and return the configured model backend."""
    from prism.taste.mock import MockBackend

    cfg = get_backend_config()
    backend_type = cfg.get("type", "mock")

    if backend_type == "mock":
        return MockBackend()

    if backend_type in ("local", "remote"):
        from prism.taste.api import APIBackend
        backend = APIBackend(
            base_url=cfg.get("base_url", "http://localhost:8000/v1"),
            thinker_model=cfg.get("thinker_model", "OpenMOSS-Team/SciThinker-30B"),
            judge_model=cfg.get("judge_model", "OpenMOSS-Team/SciJudge-30B"),
            api_key=cfg.get("api_key", "EMPTY"),
        )
        if not backend.is_available():
            from prism.display import print_warning
            print_warning(
                f"Backend at {cfg.get('base_url')} is not reachable. "
                "Falling back to mock backend.\n"
                "Run 'prism serve start' to start the local model server."
            )
            return MockBackend()
        return backend

    return MockBackend()
