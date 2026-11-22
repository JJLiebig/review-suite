from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def rounds_dir(state_dir: Path) -> Path:
    return state_dir / "rounds"


def round_path(state_dir: Path, round_id: str) -> Path:
    return rounds_dir(state_dir) / f"{round_id}.json"


def _round_state_dir_key(path: Path) -> str:
    resolved = path.resolve(strict=False)
    return str(resolved).lower() if sys.platform == "win32" else str(resolved)


def unique_round_state_dirs(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = _round_state_dir_key(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def load_round(state_dir: Path, round_id: str) -> dict[str, Any]:
    path = round_path(state_dir, round_id)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"unknown round_id: {round_id}") from None
