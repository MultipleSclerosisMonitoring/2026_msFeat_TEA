"""
Helpers for loading project configuration and environment variables.
"""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def load_project_dotenv(start_path: Path | None = None) -> Path | None:
    """
    Load a project-local `.env` file if present.

    Search order:
    1. Current working directory and its parents
    2. The repository root inferred from this package location
    """
    search_roots: list[Path] = []
    if start_path is not None:
        start = start_path if start_path.is_dir() else start_path.parent
        search_roots.append(start.resolve())

    repo_root = Path(__file__).resolve().parents[3]
    if repo_root not in search_roots:
        search_roots.append(repo_root)

    for root in search_roots:
        for candidate in [root, *root.parents]:
            dotenv_path = candidate / ".env"
            if dotenv_path.exists():
                load_dotenv(dotenv_path, override=False)
                return dotenv_path

    return None


def _expand_env_vars(value: Any) -> Any:
    """Expand environment variables of the form `${VAR_NAME}` inside config values."""
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    if isinstance(value, str):
        pattern = re.compile(r"\$\{([^}]+)\}")

        def replacer(match: re.Match[str]) -> str:
            var_name = match.group(1)
            env_value = os.getenv(var_name)
            if env_value is None:
                raise ValueError(f"Environment variable '{var_name}' is not defined.")
            return env_value

        return pattern.sub(replacer, value)
    return value


def load_project_config(config_path: str | Path) -> dict:
    """Load YAML config after populating environment variables from `.env`."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    load_project_dotenv(start_path=path)

    with path.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}

    return _expand_env_vars(raw_config)
