"""Permanent Arena retirement, independent of ratings and ordinary signoff jobs."""

from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .orchestrator_store import orchestrator_store_lock


RETIREMENTS_FILENAME = "arena_retirements.json"
RETIREMENTS_LOCK = "arena-retirements"


class ArenaRetiredError(ValueError):
    """A queued Arena invocation must stop without starting a reviewer."""


def retirement_state_dir(state_dir: Path) -> Path:
    """Arena and nested orchestrator rounds share one retirement ledger."""
    path = Path(state_dir)
    if path.parts[-2:] == ("orchestrator", "review-rounds"):
        return path.parent.parent
    return path


def model_id(variant: dict[str, Any]) -> str:
    # Old/custom rosters remain usable; release revisions sharing an API route
    # must supply model_id explicitly, as the shipped roster does.
    return str(variant.get("model_id") or variant.get("model") or variant["id"])


def _lineage_positions(roster: dict[str, Any]) -> dict[str, tuple[str, int]]:
    lineages = roster.get("lineages", {})
    if not isinstance(lineages, dict):
        raise ValueError("roster lineages must be an object of ordered model ids")
    seen: set[str] = set()
    positions: dict[str, tuple[str, int]] = {}
    for lineage, generations in lineages.items():
        if not isinstance(generations, list) or not generations:
            raise ValueError(f"lineage {lineage} must list released model ids in order")
        for generation in generations:
            if not isinstance(generation, str) or not generation.strip():
                raise ValueError(f"lineage {lineage} contains an empty model id")
            if generation in seen:
                raise ValueError(
                    f"model id occurs more than once in lineages: {generation}"
                )
            seen.add(generation)
        for position, generation in enumerate(reversed(generations)):
            positions[generation] = (lineage, position)
    return positions


def _empty_ledger() -> dict[str, Any]:
    return {"version": 1, "models": {}, "variants": {}}


def _read_ledger(state_dir: Path) -> dict[str, Any]:
    path = retirement_state_dir(state_dir) / RETIREMENTS_FILENAME
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _empty_ledger()
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid Arena retirement ledger: {path}") from exc
    if not isinstance(ledger, dict) or ledger.get("version") != 1:
        raise ValueError(f"unsupported Arena retirement ledger: {path}")
    for collection in ("models", "variants"):
        entries = ledger.get(collection)
        if not isinstance(entries, dict) or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, dict)
            or not isinstance(value.get("reason"), str)
            or not isinstance(value.get("retired_at"), str)
            for key, value in entries.items()
        ):
            raise ValueError(f"invalid {collection} in Arena retirement ledger: {path}")
    return ledger


def _write_ledger(state_dir: Path, ledger: dict[str, Any]) -> None:
    path = retirement_state_dir(state_dir) / RETIREMENTS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve(
    roster: dict[str, Any],
    ledger: dict[str, Any],
    *,
    persist_date: bool,
    retire_model_id: str | None = None,
    reason: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = deepcopy(roster)
    ledger = deepcopy(ledger)
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    entry_date = {"retired_at": now} if persist_date else {}
    positions = _lineage_positions(roster)
    retired_models = {
        generation: f"Superseded: outside the latest two generations of {lineage}"
        for generation, (lineage, position) in positions.items() if position >= 2
    }
    variants_by_model: dict[str, list[dict[str, Any]]] = {}
    for variant in resolved.get("variants", []):
        variants_by_model.setdefault(model_id(variant), []).append(variant)
    for generation, variants in variants_by_model.items():
        if all(variant.get("state") == "retired" for variant in variants):
            retired_models.setdefault(generation, "Retired in the roster")
    if retire_model_id is not None:
        if (
            retire_model_id not in variants_by_model
            and retire_model_id not in ledger["models"]
        ):
            raise ValueError(f"unknown Arena model id: {retire_model_id}")
        retired_models[retire_model_id] = str(
            reason or "Manually retired from Arena"
        ).strip()
        if not retired_models[retire_model_id]:
            raise ValueError("retirement reason must not be empty")
    for generation, retirement_reason in retired_models.items():
        ledger["models"].setdefault(
            generation, {**entry_date, "reason": retirement_reason}
        )
    for variant in resolved.get("variants", []):
        variant_id = str(variant["id"])
        generation = model_id(variant)
        retirement = ledger["variants"].get(variant_id) or ledger["models"].get(
            generation
        )
        if retirement is None and variant.get("state") == "retired":
            retirement = {**entry_date, "reason": "Retired in the roster"}
        if retirement is not None:
            ledger["variants"].setdefault(
                variant_id, {**retirement, "model_id": generation}
            )
            variant["state"] = "retired"
            variant["lifecycle"] = {
                "status": "retired",
                **{
                    key: retirement[key]
                    for key in ("retired_at", "reason")
                    if key in retirement
                },
            }
        else:
            variant["lifecycle"] = {
                "status": "active" if variant.get("state", "active") == "active"
                and variant.get("arena_eligible", True) is True else "disabled"
            }
        if generation in positions:
            lineage, position = positions[generation]
            variant["lifecycle"].update(
                lineage=lineage,
                generation="current" if position == 0 else "previous" if position == 1 else "older",
            )
    resolved["_arena_retirements"] = ledger
    return resolved, ledger


def resolve_roster_lifecycle(
    roster: dict[str, Any],
    *,
    state_dir: Path | None = None,
    persist: bool = False,
    retire_model_id: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Apply the policy automatically; persisted tombstones always win later edits.

    Pure previews do not create files or claim a retirement observation date.
    """
    if persist and state_dir is None:
        raise ValueError("persisting Arena retirement requires a state directory")
    if persist:
        root = retirement_state_dir(state_dir)
        with orchestrator_store_lock(state_dir=root, name=RETIREMENTS_LOCK):
            previous = _read_ledger(root)
            resolved, ledger = _resolve(
                roster,
                previous,
                persist_date=True,
                retire_model_id=retire_model_id,
                reason=reason,
            )
            if ledger != previous:
                _write_ledger(root, ledger)
            return resolved
    previous = (
        _read_ledger(state_dir)
        if state_dir is not None
        else roster.get("_arena_retirements", _empty_ledger())
    )
    return _resolve(
        roster,
        previous,
        persist_date=False,
        retire_model_id=retire_model_id,
        reason=reason,
    )[0]


@contextmanager
def arena_launch_guard(roster: dict[str, Any], *, state_dir: Path, variant_id: str):
    """Serialize retirement with a child start, including stale plans and retries."""
    root = retirement_state_dir(state_dir)
    with orchestrator_store_lock(state_dir=root, name=RETIREMENTS_LOCK):
        source = roster.get("_roster_path")
        current = (
            json.loads(Path(source).read_text(encoding="utf-8")) if source else roster
        )
        previous = _read_ledger(root)
        resolved, ledger = _resolve(current, previous, persist_date=True)
        if ledger != previous:
            _write_ledger(root, ledger)
        variant = next(
            (v for v in resolved.get("variants", []) if v["id"] == variant_id), None
        )
        if variant is None:
            raise ArenaRetiredError(
                f"Arena reviewer {variant_id} is no longer in the roster; reroll or dismiss this round"
            )
        if variant.get("state") == "retired":
            raise ArenaRetiredError(
                f"Arena reviewer {variant_id} is permanently retired; reroll or dismiss this round"
            )
        yield variant
