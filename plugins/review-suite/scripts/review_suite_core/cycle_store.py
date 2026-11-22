from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


CALLER_PUBLIC_ID_RE = re.compile(r"rvw_[a-z0-9]{8,32}")
CYCLE_KEY_RE = re.compile(r"orc-[0-9a-f]{20}")


class UnknownReviewIdError(ValueError):
    pass


def orchestrator_dir(state_dir: Path) -> Path:
    return state_dir / "orchestrator"


def cycles_dir(state_dir: Path) -> Path:
    return orchestrator_dir(state_dir) / "cycles"


def index_path(state_dir: Path) -> Path:
    return orchestrator_dir(state_dir) / "index.json"


def cycle_path(state_dir: Path, cycle_key: str) -> Path:
    return cycles_dir(state_dir) / f"{cycle_key}.json"


def public_id_path(state_dir: Path, public_id: str) -> Path | None:
    if CALLER_PUBLIC_ID_RE.fullmatch(public_id):
        return orchestrator_dir(state_dir) / "ids" / f"{public_id}.json"
    return None


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid orchestrator JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"orchestrator JSON must be an object: {path}")
    return payload


def load_cycle_by_key(state_dir: Path, cycle_key: str) -> dict[str, Any] | None:
    return read_json(cycle_path(state_dir, cycle_key)) or None


def read_public_id_lookup(path: Path) -> str | None:
    try:
        key = read_json(path).get("cycle_key")
    except (OSError, ValueError):
        return None  # An unavailable lookup leaves the aggregate index authoritative.
    return key if isinstance(key, str) and CYCLE_KEY_RE.fullmatch(key) else None


def load_cycle_by_public_id(state_dir: Path, public_id: str) -> dict[str, Any]:
    review_id = str(public_id or "").strip()
    if not review_id:
        raise ValueError("--id is required")
    path = public_id_path(state_dir, review_id)
    if path and (key := read_public_id_lookup(path)):
        state = load_cycle_by_key(state_dir, key)
        if state and state.get("public_id") == review_id and state.get("cycle_key") == key:
            return state
    # Cycles created by older runtimes and writes interrupted before publishing
    # their lookup are still discoverable through the original index.
    raw_ids = read_json(index_path(state_dir)).get("ids")
    cycle_key = str(raw_ids.get(review_id) or "") if isinstance(raw_ids, dict) else ""
    if not cycle_key:
        raise UnknownReviewIdError(f"unknown review cycle id: {review_id}")
    state = load_cycle_by_key(state_dir, cycle_key)
    if state is None:
        raise UnknownReviewIdError(f"unknown review cycle id: {review_id}")
    state.setdefault("public_id", review_id)
    return state
