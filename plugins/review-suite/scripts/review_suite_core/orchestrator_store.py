from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from .orchestrator_state import record_contract_replan

if os.name == "nt":
    import msvcrt
else:
    import fcntl


ORCHESTRATOR_INDEX_SCHEMA_VERSION = 1
ORCHESTRATOR_CYCLES_LOCK = "orchestrator-cycles"


def orchestrator_dir(state_dir: Path) -> Path:
    return state_dir / "orchestrator"


def cycles_dir(state_dir: Path) -> Path:
    return orchestrator_dir(state_dir) / "cycles"


def index_path(state_dir: Path) -> Path:
    return orchestrator_dir(state_dir) / "index.json"


def cycle_path(state_dir: Path, cycle_key: str) -> Path:
    return cycles_dir(state_dir) / f"{cycle_key}.json"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid orchestrator JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"orchestrator JSON must be an object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(tmp_path, path)


def _empty_index() -> dict[str, Any]:
    return {
        "schema_version": ORCHESTRATOR_INDEX_SCHEMA_VERSION,
        "ids": {},
        "cycle_keys": {},
    }


def load_index(state_dir: Path) -> dict[str, Any]:
    raw = _read_json(index_path(state_dir))
    if not raw:
        return _empty_index()
    ids = raw.get("ids") if isinstance(raw.get("ids"), dict) else {}
    cycle_keys = (
        raw.get("cycle_keys") if isinstance(raw.get("cycle_keys"), dict) else {}
    )
    return {
        "schema_version": ORCHESTRATOR_INDEX_SCHEMA_VERSION,
        "ids": {str(key): str(value) for key, value in ids.items()},
        "cycle_keys": {str(key): str(value) for key, value in cycle_keys.items()},
    }


def _write_index(state_dir: Path, index: dict[str, Any]) -> None:
    _atomic_write_json(index_path(state_dir), index)


def _try_lock_file(handle) -> bool:
    try:
        if os.name == "nt":
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock_file(handle) -> None:
    if os.name == "nt":
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def orchestrator_store_lock(
    *,
    state_dir: Path,
    name: str,
    timeout_seconds: int = 30,
    poll_seconds: float = 0.1,
):
    locks_dir = state_dir / ".locks"
    locks_dir.mkdir(parents=True, exist_ok=True)
    lock_path = locks_dir / f"{name}.lock"
    with lock_path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        deadline = time.monotonic() + timeout_seconds
        while True:
            if _try_lock_file(handle):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for lock: {lock_path}")
            time.sleep(poll_seconds)
        try:
            yield
        finally:
            _unlock_file(handle)


def _public_id_candidates(cycle_key: str) -> list[str]:
    digest = cycle_key.removeprefix("orc-")
    lengths = (8, 10, 12, 16, len(digest))
    return [f"rvw_{digest[:length]}" for length in lengths if length <= len(digest)]


def public_id_for_cycle_key(state_dir: Path, cycle_key: str) -> str:
    with orchestrator_store_lock(state_dir=state_dir, name="orchestrator-index"):
        index = load_index(state_dir)
        existing = str(index["cycle_keys"].get(cycle_key) or "")
        if existing:
            if index["ids"].get(existing) != cycle_key:
                index["ids"][existing] = cycle_key
                _write_index(state_dir, index)
            return existing
        for candidate in _public_id_candidates(cycle_key):
            if candidate not in index["ids"]:
                index["ids"][candidate] = cycle_key
                index["cycle_keys"][cycle_key] = candidate
                _write_index(state_dir, index)
                return candidate
    raise ValueError(f"could not allocate public id for cycle key: {cycle_key}")


def load_cycle_by_key(state_dir: Path, cycle_key: str) -> dict[str, Any] | None:
    path = cycle_path(state_dir, cycle_key)
    if not path.exists():
        return None
    return _read_json(path)


def load_cycle_by_public_id(state_dir: Path, public_id: str) -> dict[str, Any]:
    review_id = str(public_id or "").strip()
    if not review_id:
        raise ValueError("--id is required")
    index = load_index(state_dir)
    cycle_key = str(index["ids"].get(review_id) or "")
    if not cycle_key:
        raise ValueError(f"unknown review cycle id: {review_id}")
    state = load_cycle_by_key(state_dir, cycle_key)
    if state is None:
        raise ValueError(f"review cycle state is missing for id: {review_id}")
    state.setdefault("public_id", review_id)
    return state


def save_cycle(state_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    with orchestrator_store_lock(state_dir=state_dir, name=ORCHESTRATOR_CYCLES_LOCK):
        cycle_key = str(state.get("cycle_key") or "").strip()
        current = load_cycle_by_key(state_dir, cycle_key) if cycle_key else None
        payload = deepcopy(state)
        if current and current.get("stage") == "followup-pending" and payload.get("stage") in {"fix-pending", "followup-pending"}:
            from review_suite_local import _process_is_running

            if _process_is_running(dict(current.get("pending_action") or {}).get("wrapper_pid")):
                for key in ("stage", "pending_action"):
                    payload[key] = deepcopy(current[key])
        if current and str(
            dict(current.get("superseded_by") or {}).get("review") or ""
        ):
            for key in (
                "stage",
                "pending_action",
                "recovery",
                "superseded_by",
                "convergence",
            ):
                payload[key] = deepcopy(current.get(key))
        return _save_cycle_unlocked(state_dir, payload)


def _save_cycle_unlocked(state_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    cycle_key = str(state.get("cycle_key") or "").strip()
    if not cycle_key:
        raise ValueError("cycle_key is required")
    payload = deepcopy(state)
    payload["public_id"] = public_id_for_cycle_key(state_dir, cycle_key)
    _atomic_write_json(cycle_path(state_dir, cycle_key), payload)
    return payload


def reserve_cycle_successor(
    state_dir: Path,
    *,
    source: dict[str, Any],
    successor: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    source_key = str(source.get("cycle_key") or "").strip()
    if not source_key:
        raise ValueError("source cycle_key is required")
    with orchestrator_store_lock(state_dir=state_dir, name=ORCHESTRATOR_CYCLES_LOCK):
        current = load_cycle_by_key(state_dir, source_key)
        if current is None:
            raise ValueError(f"source review cycle state is missing: {source_key}")
        superseded = dict(current.get("superseded_by") or {})
        existing_id = str(superseded.get("review") or "").strip()
        if existing_id:
            existing_key = str(superseded.get("cycle_key") or "").strip()
            requested_key = str(successor.get("cycle_key") or "").strip()
            if existing_key != requested_key:
                raise ValueError(
                    f"source review cycle already has a different successor: {existing_id}"
                )
            return load_cycle_by_public_id(state_dir, existing_id), False
        replanned = None
        if dict(source.get("superseded_by") or {}).get("kind") == "contract-replan":
            if any(
                source.get(key) != current.get(key)
                for key in ("identity", "active_findings")
            ):
                raise ValueError(
                    "source review changed during contract replan; inspect its status and retry"
                )
            replanned = record_contract_replan(
                current, conflict=source["convergence"]["conflict"]
            )
        saved_successor = _save_cycle_unlocked(state_dir, successor)
        saved_source = deepcopy(current)
        for key in ("stage", "pending_action", "recovery", "superseded_by"):
            saved_source[key] = deepcopy(source.get(key))
        if replanned is not None:
            saved_source["convergence"] = deepcopy(replanned["convergence"])
        redirect = dict(saved_source.get("superseded_by") or {})
        redirect["review"] = str(saved_successor["public_id"])
        redirect["cycle_key"] = str(saved_successor["cycle_key"])
        saved_source["superseded_by"] = redirect
        _save_cycle_unlocked(state_dir, saved_source)
        return saved_successor, True
