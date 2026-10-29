"""One-time, offline migration of saved review names to base/deep."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from review_suite_core.axi_output import AxiArgumentParser, emit_error, emit_toon
from review_suite_core.config import _toml_settings


KEY_NAMES = {
    "phase_review": "base_review",
    "pr_review": "deep_review",
    "review_t1": "review_base",
    "review_t3": "review_deep",
    "arena_phase": "arena_base",
    "discovery_phase": "discovery_base",
    "normal_signoff": "base_signoff",
    "signoff_normal_model": "signoff_base_model",
    "normal_arena_loops": "base_arena_loops",
    "t1_sessions": "base_sessions",
    "t3_sessions": "deep_sessions",
}
NAME_FIELDS = {
    "task_class",
    "task_classes",
    "taskName",
    "task",
    "public_task",
    "lane",
    "source_lane",
    "last_reviewed_lane",
    "tool_name",
    "pool",
    "model_ref",
    "loop_ref",
    "subtract_loop_ref",
}
STEP_FIELDS = {
    "name",
    "step",
    "step_name",
    "orchestrator_step",
    "next_step_name",
    "profile_step",
}
STEP_NAMES = {
    "precision-signoff": "base-signoff",
    "fast-signoff": "base-signoff",
    "arena-phase-review": "arena-base-review",
    "arena-pr-review": "arena-deep-review",
}
STATE_DIRECTORIES = ("rounds", "workflow", "review_cost_rows", "orchestrator", "tools")
STATE_FILES = (
    "settings.toml",
    "config.json",
    "runs.jsonl",
    "wrapper_sessions.jsonl",
    "summary.json",
    "operational_state.json",
)


def _step_name(value: str) -> str:
    for old, new in STEP_NAMES.items():
        if value == old or re.fullmatch(re.escape(old) + r"-\d+", value):
            return new + value[len(old) :]
    return value


def rename_payload(
    value: Any,
    *,
    deep_rounds: set[str],
    field: str = "",
    path: tuple[str, ...] = (),
    settings: bool = False,
) -> Any:
    if isinstance(value, list):
        return [
            rename_payload(
                item, deep_rounds=deep_rounds, field=field, path=path, settings=settings
            )
            for item in value
        ]
    if isinstance(value, str):
        if field in NAME_FIELDS:
            return KEY_NAMES.get(value, value)
        if field in STEP_FIELDS:
            return _step_name(value)
        if value == "normal" and (
            field in {"mode", "requested_mode", "effective_mode"}
            or path[-2:-1] == ("mode",)
            and field in {"requested", "effective"}
        ):
            return "base"
        return value
    if not isinstance(value, dict):
        return value

    result: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"review_brief", "reviewer_output", "requested_prompt"}:
            result[key] = item
            continue
        new_key = KEY_NAMES.get(key, key)
        if (
            key == "normal"
            and settings
            and (not path or len(path) >= 2 and path[-2] == "profiles")
        ):
            new_key = "base"
        renamed = rename_payload(
            item,
            deep_rounds=deep_rounds,
            field=key,
            path=(*path, key),
            settings=settings,
        )
        if new_key in result and result[new_key] != renamed:
            raise ValueError(
                f"conflicting saved values for {'.'.join((*path, new_key))}"
            )
        result[new_key] = renamed

    # Old fixed-model deep signoffs were mislabeled as phase reviews.
    is_deep = any(value.get(key) == "deep-signoff" for key in STEP_FIELDS)
    profile_step = value.get("profile_step")
    is_deep = (
        is_deep
        or isinstance(profile_step, dict)
        and profile_step.get("name") == "deep-signoff"
    )
    is_deep = is_deep or any(
        value.get(key) in deep_rounds
        for key in ("round_id", "source_round_id", "origin_round_id")
    )
    if is_deep:
        if "task_class" in result:
            result["task_class"] = "deep_review"
        for key in ("lane", "public_task", "taskName", "source_lane"):
            if result.get(key) == "review_base":
                result[key] = "review_deep"
    if (
        value.get("name") in {"precision-signoff", "fast-signoff", "deep-signoff"}
        or value.get("kind") == "review"
        and "name" in value
        and "count" in value
    ):
        result["lane"] = "review_deep" if is_deep else "review_base"
        result["task_class"] = "deep_review" if is_deep else "base_review"
    return result


def _files(state_dir: Path, snapshots: list[Path]) -> list[Path]:
    paths = [state_dir / name for name in STATE_FILES if (state_dir / name).is_file()]
    for name in STATE_DIRECTORIES:
        directory = state_dir / name
        paths.extend(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix in {".json", ".jsonl"}
        )
    return sorted(set(paths + snapshots))


def _read(path: Path) -> tuple[bytes, Any]:
    original = path.read_bytes()
    try:
        text = original.decode("utf-8-sig")
        if path.suffix == ".toml":
            payload = tomllib.loads(text)
        elif path.suffix == ".jsonl":
            payload = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            payload = json.loads(text)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"cannot read saved review {path}: {exc}") from exc
    return original, payload


def migrate_names(
    state_dir: Path, snapshots: list[Path], *, apply: bool = False
) -> dict[str, Any]:
    state_dir = state_dir.resolve(strict=True)
    if not state_dir.is_dir():
        raise ValueError(
            "--state-dir must name an existing Review Suite state directory"
        )
    snapshots = [path.resolve(strict=True) for path in snapshots]
    paths = _files(state_dir, snapshots)
    deep_rounds: set[str] = set()
    for directory in (
        state_dir / "rounds",
        state_dir / "orchestrator/review-rounds/rounds",
    ):
        for path in directory.glob("*.json"):
            _, payload = _read(path)
            if (
                isinstance(payload, dict)
                and payload.get("orchestrator_step") == "deep-signoff"
            ):
                deep_rounds.add(payload["round_id"])

    changes: list[tuple[Path, bytes, bytes]] = []
    for path in paths:
        original, payload = _read(path)
        renamed = rename_payload(
            payload,
            deep_rounds=deep_rounds,
            settings=path.name in {"settings.toml", "config.json"},
        )
        if renamed == payload:
            continue
        if path.suffix == ".toml":
            content = _toml_settings(renamed)
        elif path.suffix == ".jsonl":
            content = "".join(
                json.dumps(row, ensure_ascii=False) + "\n" for row in renamed
            )
        else:
            content = json.dumps(renamed, indent=2, ensure_ascii=False) + "\n"
        changes.append((path, original, content.encode("utf-8")))

    result: dict[str, Any] = {"status": "dry_run", "files": len(changes)}
    if not apply or not changes:
        if apply:
            result["status"] = "unchanged"
        return result

    # Preflight every input and back up all originals before replacing any file.
    for path, original, _ in changes:
        if path.read_bytes() != original:
            raise ValueError(
                f"saved reviews changed during migration; stop writers and retry: {path}"
            )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = state_dir / f"base-deep-backup-{stamp}"
    manifest: dict[str, str] = {}
    for path, original, _ in changes:
        relative = (
            path.relative_to(state_dir)
            if path.is_relative_to(state_dir)
            else Path("snapshots")
            / hashlib.sha256(str(path).encode()).hexdigest()[:12]
            / path.name
        )
        destination = backup / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(original)
        manifest[str(path)] = relative.as_posix()
    (backup / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    try:
        for path, original, content in changes:
            if path.read_bytes() != original:
                raise ValueError(f"saved reviews changed during migration: {path}")
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(content)
            try:
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"migration interrupted; originals are saved at {backup}: {exc}"
        ) from exc
    return {"status": "migrated", "files": len(changes), "backup": str(backup)}


def main() -> int:
    parser = AxiArgumentParser(
        description="Migrate saved Review Suite names to base/deep while all writers are stopped.",
        epilog="Dry run: migrate_review_names.py --state-dir <path>. "
        "Apply: add --apply. Include an API snapshot or tray last-round file with --snapshot <path>.",
    )
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--snapshot", action="append", default=[], type=Path)
    parser.add_argument("--apply", action="store_true")
    try:
        args = parser.parse_args()
    except ValueError as exc:
        return emit_error(
            str(exc),
            status="usage_error",
            help_items=[
                "Run migrate_review_names.py --help for arguments and examples."
            ],
        )
    try:
        emit_toon(migrate_names(args.state_dir, args.snapshot, apply=args.apply))
    except (OSError, ValueError) as exc:
        return emit_error(
            str(exc),
            help_items=[
                "Stop review, daemon, tray, and API writers before applying the migration."
            ],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
