from __future__ import annotations

import json
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from review_suite_core.orchestrator_state import (
    STAGE_ABORTED,
    STAGE_DISMISSED,
    STAGE_LOCAL_GREEN_HANDOFF,
    STAGE_REVIEW_GREEN,
)
from review_suite_core.orchestrator_store import (
    ORCHESTRATOR_CYCLES_LOCK,
    cycles_dir,
    index_path,
    load_index,
    orchestrator_store_lock,
    public_id_path,
)
from review_suite_local import (
    ORCHESTRATOR_ROUND_STATE_DIR,
    write_json,
)


TERMINAL_CYCLE_STAGES = {
    STAGE_ABORTED,
    STAGE_DISMISSED,
    STAGE_LOCAL_GREEN_HANDOFF,
    STAGE_REVIEW_GREEN,
}
ROUND_STATE_DIRS = (Path(), ORCHESTRATOR_ROUND_STATE_DIR)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _older_than(path: Path, cutoff: datetime) -> bool:
    try:
        return _mtime(path) < cutoff
    except OSError:
        return False


def _path_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _timestamp(payload: dict[str, Any], *keys: str) -> datetime | None:
    for key in keys:
        value = str(payload.get(key) or "").strip()
        if not value:
            continue
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
                timezone.utc
            )
        except ValueError:
            continue
    return None


def _round_ids(payload: dict[str, Any]) -> set[str]:
    return {
        str(item.get("round_id") or "").strip()
        for item in list(payload.get("rounds") or [])
        if isinstance(item, dict) and str(item.get("round_id") or "").strip()
    }


def _pruned_index(
    state_dir: Path, deleted_keys: set[str]
) -> tuple[dict[str, Any], int, list[str]]:
    path = index_path(state_dir)
    index = load_index(state_dir)
    removed_ids = [
        public_id for public_id, cycle_key in index["ids"].items()
        if cycle_key in deleted_keys
    ]
    index["ids"] = {
        public_id: cycle_key
        for public_id, cycle_key in index["ids"].items()
        if cycle_key not in deleted_keys
    }
    index["cycle_keys"] = {
        cycle_key: public_id
        for cycle_key, public_id in index["cycle_keys"].items()
        if cycle_key not in deleted_keys
    }
    after = len((json.dumps(index, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return index, max(0, _path_size(path) - after), removed_ids


def _cycle_is_prunable(path: Path, payload: dict[str, Any], cutoff: datetime) -> bool:
    return (
        _older_than(path, cutoff)
        and not dict(payload.get("pending_action") or {})
        and str(payload.get("stage") or "").strip() in TERMINAL_CYCLE_STAGES
    )


def _prune_orchestrator_cycles(
    state_dir: Path, *, cutoff: datetime, apply: bool
) -> tuple[dict[str, int], set[str]]:
    result = {"checked": 0, "deleted": 0, "saved_b": 0}
    protected_round_ids: set[str] = set()
    deleted_keys: set[str] = set()
    lock = (
        orchestrator_store_lock(state_dir=state_dir, name=ORCHESTRATOR_CYCLES_LOCK)
        if apply
        else nullcontext()
    )
    with lock:
        for path in sorted(cycles_dir(state_dir).glob("*.json")):
            payload = _load_json(path)
            if payload is None:
                continue
            result["checked"] += 1
            if not _cycle_is_prunable(path, payload, cutoff):
                protected_round_ids.update(_round_ids(payload))
                continue
            size = _path_size(path)
            if apply:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    protected_round_ids.update(_round_ids(payload))
                    continue
            deleted_keys.add(path.stem)
            result["deleted"] += 1
            result["saved_b"] += size
        if deleted_keys:
            if apply:
                with orchestrator_store_lock(
                    state_dir=state_dir, name="orchestrator-index"
                ):
                    index, saved, removed_ids = _pruned_index(state_dir, deleted_keys)
                    write_json(index_path(state_dir), index)
            else:
                _, saved, removed_ids = _pruned_index(state_dir, deleted_keys)
            for public_id in removed_ids:
                if lookup := public_id_path(state_dir, public_id):
                    saved += _path_size(lookup)
                    if apply:
                        lookup.unlink(missing_ok=True)
            result["saved_b"] += saved
    return result, protected_round_ids


def _round_is_prunable(
    path: Path,
    payload: dict[str, Any],
    *,
    cutoff: datetime,
    protected_round_ids: set[str],
) -> bool:
    round_id = str(payload.get("round_id") or path.stem).strip()
    if round_id in protected_round_ids:
        return False
    status = str(payload.get("status") or "").strip()
    if status == "dismissed":
        completed_at = _timestamp(
            payload, "dismissed_at", "review_completed_at", "sampled_at"
        )
    elif status == "completed" and str(payload.get("graded_at") or "").strip():
        completed_at = _timestamp(payload, "graded_at", "review_completed_at")
    else:
        return False
    return completed_at < cutoff if completed_at else _older_than(path, cutoff)


def _prune_rounds(
    state_dir: Path,
    *,
    cutoff: datetime,
    protected_round_ids: set[str],
    apply: bool,
) -> dict[str, int]:
    result = {"checked": 0, "deleted": 0, "saved_b": 0}
    for relative_state_dir in ROUND_STATE_DIRS:
        round_state_dir = state_dir / relative_state_dir
        for path in sorted((round_state_dir / "rounds").glob("*.json")):
            payload = _load_json(path)
            if payload is None:
                continue
            result["checked"] += 1
            if not _round_is_prunable(
                path,
                payload,
                cutoff=cutoff,
                protected_round_ids=protected_round_ids,
            ):
                continue
            size = _path_size(path)
            if apply:
                latest = _load_json(path)
                if latest is None or not _round_is_prunable(
                    path,
                    latest,
                    cutoff=cutoff,
                    protected_round_ids=protected_round_ids,
                ):
                    continue
                size = _path_size(path)
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    continue
            result["deleted"] += 1
            result["saved_b"] += size
    return result


def prune_review_state(
    state_dir: Path,
    *,
    apply: bool = False,
    older_than_days: int = 14,
    now: datetime | None = None,
) -> dict[str, Any]:
    state_dir = Path(state_dir)
    cutoff = (now or _utc_now()) - timedelta(days=max(1, older_than_days))
    cycles, protected_round_ids = _prune_orchestrator_cycles(
        state_dir, cutoff=cutoff, apply=apply
    )
    rounds = _prune_rounds(
        state_dir,
        cutoff=cutoff,
        protected_round_ids=protected_round_ids,
        apply=apply,
    )
    saved = cycles["saved_b"] + rounds["saved_b"]
    return {
        "applied": bool(apply),
        "cutoff": cutoff.isoformat().replace("+00:00", "Z"),
        "saved_b": saved,
        "orchestrator_cycles": cycles,
        "rounds": rounds,
    }
