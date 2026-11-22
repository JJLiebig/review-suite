from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager, suppress
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from .cycle_status import convergence_summary
from .cycle_store import (
    CALLER_PUBLIC_ID_RE as CALLER_PUBLIC_ID_RE,
    UnknownReviewIdError as UnknownReviewIdError,
    cycle_path as cycle_path,
    cycles_dir as cycles_dir,
    index_path as index_path,
    load_cycle_by_key as load_cycle_by_key,
    load_cycle_by_public_id as load_cycle_by_public_id,
    orchestrator_dir as orchestrator_dir,
    public_id_path as public_id_path,
    read_json,
    read_public_id_lookup,
)

if os.name == "nt":
    import msvcrt
else:
    import fcntl


ORCHESTRATOR_INDEX_SCHEMA_VERSION = 1
ORCHESTRATOR_CYCLES_LOCK = "orchestrator-cycles"


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
    raw = read_json(index_path(state_dir))
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
    timeout_seconds: float = 30,
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


def create_cycle_with_public_id(
    state_dir: Path, state: dict[str, Any], public_id: str
) -> dict[str, Any]:
    """Save a new cycle under a caller-chosen review id."""
    cycle_key = str(state["cycle_key"])
    with orchestrator_store_lock(state_dir=state_dir, name=ORCHESTRATOR_CYCLES_LOCK):
        with orchestrator_store_lock(state_dir=state_dir, name="orchestrator-index"):
            index = load_index(state_dir)
            bound = index["ids"].get(public_id)
            if bound and bound != cycle_key:
                raise ValueError(
                    f"review id {public_id} already belongs to another review; "
                    "use a new id"
                )
            index["ids"][public_id] = cycle_key
            index["cycle_keys"][cycle_key] = public_id
            _write_index(state_dir, index)
        current = load_cycle_by_key(state_dir, cycle_key)
        if current is not None:
            if current.get("public_id") == public_id:
                _publish_public_id_lookup(state_dir, public_id, cycle_key)
            current["public_id"] = public_id
            return current
        return _save_cycle_unlocked(state_dir, state)


def save_cycle(
    state_dir: Path, state: dict[str, Any], *, claim_followup: bool = False, preserve_validation: bool = False, validation_only: bool = False
) -> dict[str, Any]:
    with orchestrator_store_lock(state_dir=state_dir, name=ORCHESTRATOR_CYCLES_LOCK):
        cycle_key = str(state.get("cycle_key") or "").strip()
        current = load_cycle_by_key(state_dir, cycle_key) if cycle_key else None
        payload = deepcopy(state)
        if validation_only and current is not None:
            validation = payload.get("validation")
            payload = deepcopy(current)
            payload["validation"] = deepcopy(validation)
        if preserve_validation and current is not None:
            payload["validation"] = deepcopy(current.get("validation") or {})
        if claim_followup and current is not None:
            from review_suite_local import _process_is_running

            if (
                current.get("stage") not in {"fix-pending", "followup-pending"}
                or convergence_summary(current)["status"] != "ACTIVE"
                or dict(current.get("active_findings") or {}).get("round_id")
                != dict(payload.get("active_findings") or {}).get("round_id")
            ):
                raise ValueError(
                    "source review changed before follow-up launch; inspect its status and follow the current action"
                )
            if _process_is_running(
                dict(current.get("pending_action") or {}).get("wrapper_pid")
            ):
                raise ValueError(
                    "follow-up wrapper is already running; wait for its owning review command"
                )
            payload["convergence"] = deepcopy(current["convergence"])
        if current and dict(current.get("superseded_by") or {}).get("review"):
            return current
        if current and current.get("stage") == "followup-pending":
            from review_suite_local import _process_is_running

            if _process_is_running(
                dict(current.get("pending_action") or {}).get("wrapper_pid")
            ):
                if payload.get("stage") in {"fix-pending", "followup-pending"}:
                    for key in ("stage", "pending_action"):
                        payload[key] = deepcopy(current[key])
                elif not (
                    payload.get("stage") == "decision-pending"
                    and dict(payload.get("pending_action") or {}).get("lane")
                    == "review-followup"
                    and dict(payload.get("pending_action") or {}).get("round_id")
                    == dict(current.get("pending_action") or {}).get("round_id")
                ):
                    raise ValueError(
                        "follow-up wrapper is running; finish or stop its owning review command before changing convergence"
                    )
        incoming_convergence = dict(payload.get("convergence") or {})
        if (
            current
            and incoming_convergence.get("status") == "DECISION_REQUIRED"
            and incoming_convergence.get("reason") == "contract_conflict"
            and incoming_convergence != current.get("convergence")
            and (
                current.get("stage") != "fix-pending"
                or current.get("active_findings") != payload.get("active_findings")
            )
        ):
            raise ValueError(
                "source findings changed before contract conflict was recorded; inspect its status and follow the current action"
            )
        return _save_cycle_unlocked(state_dir, payload)


def _save_cycle_unlocked(state_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    cycle_key = str(state.get("cycle_key") or "").strip()
    if not cycle_key:
        raise ValueError("cycle_key is required")
    payload = deepcopy(state)
    payload["public_id"] = public_id_for_cycle_key(state_dir, cycle_key)
    if payload.get("stage") in {"review-green", "local-green-handoff"}:
        from .patch_identity import reviewed_receipt
        try:
            receipt = reviewed_receipt(payload)
            if receipt:
                payload["review_patch"] = receipt
        except (OSError, ValueError):
            pass
    _atomic_write_json(cycle_path(state_dir, cycle_key), payload)
    _publish_public_id_lookup(state_dir, payload["public_id"], cycle_key)
    return payload


def _publish_public_id_lookup(state_dir: Path, public_id: str, cycle_key: str) -> None:
    path = public_id_path(state_dir, public_id)
    if path is not None and read_public_id_lookup(path) != cycle_key:
        # The cycle and aggregate index are already committed at this point.
        with suppress(OSError):
            _atomic_write_json(path, {"cycle_key": cycle_key})


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
            from .orchestrator_state import record_contract_replan

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
