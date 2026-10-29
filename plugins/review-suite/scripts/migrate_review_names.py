"""One-time, offline migration of saved review names to base/deep."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from review_suite_core.axi_output import AxiArgumentParser, emit_error, emit_toon
from review_suite_core.config import _toml_settings
from review_suite_core.name_migration import rename_payload


STATE_DIRECTORIES = ("rounds", "workflow", "review_cost_rows", "orchestrator", "tools")
STATE_FILES = (
    "settings.toml",
    "config.json",
    "runs.jsonl",
    "wrapper_sessions.jsonl",
    "summary.json",
    "operational_state.json",
)


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
    followups: dict[str, list[str]] = {}
    for directory in (
        state_dir / "rounds",
        state_dir / "orchestrator/review-rounds/rounds",
    ):
        for path in directory.glob("*.json"):
            _, payload = _read(path)
            if not isinstance(payload, dict) or not isinstance(
                payload.get("round_id"), str
            ):
                continue
            round_id = payload["round_id"]
            if (
                payload.get("orchestrator_step") == "deep-signoff"
                or payload.get("task_class") in {"pr_review", "deep_review"}
                or payload.get("public_task") in {"review_t3", "review_deep"}
            ):
                deep_rounds.add(round_id)
            source = payload.get("review_scope", {}).get(
                "source_round_id"
            ) or payload.get("source_round_id")
            if isinstance(source, str):
                followups.setdefault(source, []).append(round_id)
    pending = list(deep_rounds)
    while pending:
        for round_id in followups.get(pending.pop(), []):
            if round_id not in deep_rounds:
                deep_rounds.add(round_id)
                pending.append(round_id)

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
