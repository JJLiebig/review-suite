"""Resolve Review Suite state paths without initializing or migrating them."""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from .cycle_store import read_json


_selected_state: ContextVar[Path | None] = ContextVar(
    "review_suite_state", default=None
)


@contextmanager
def state_directory(path: str | None):
    token = _selected_state.set(Path(path).expanduser().absolute() if path else None)
    try:
        yield
    finally:
        _selected_state.reset(token)


def _same_state(left: Path, right: Path) -> bool:
    return left.exists() and right.exists() and left.samefile(right)


def state_override() -> Path | None:
    if selected := _selected_state.get():
        return selected
    override = os.environ.get("REVIEW_SUITE_STATE_DIR", "").strip()
    return Path(override).expanduser().absolute() if override else None


def state_locations() -> tuple[Path, list[Path], Path]:
    home = Path.home()
    target = home / ".agents" / "state" / "review-suite"
    codex = Path(os.environ.get("CODEX_HOME") or home / ".codex").expanduser().absolute()
    sources = list(dict.fromkeys([
        codex / "state" / "review-suite",
        home / ".codex" / "state" / "review-suite",
    ]))
    return target, sources, target.parent / ".review-suite-migration"


def read_state_dir() -> Path:
    if override := state_override():
        return override
    target, sources, control = state_locations()
    # During a cross-device migration the source remains authoritative until
    # it is retired. Readers use it without taking the writer lock.
    pending = read_json(control / "pending.json")
    if legacy_text := pending.get("legacy"):
        legacy = Path(legacy_text)
        if legacy.is_dir() and not _same_state(legacy, target):
            return legacy
    if target.exists():
        return target
    return next((path for path in sources if path.exists()), target)
