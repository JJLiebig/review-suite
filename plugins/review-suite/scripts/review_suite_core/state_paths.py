"""Shared state location and one-time, serialized migration from Codex homes."""

from __future__ import annotations

import atexit
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from .orchestrator_store import _try_lock_file, _unlock_file, orchestrator_store_lock

_leases: dict[str, object] = {}


def _lease(control: Path, legacy: Path) -> None:
    key = str(legacy)
    if key in _leases:
        return
    path = (
        control / f"use-{os.getpid()}-{hashlib.sha256(key.encode()).hexdigest()[:16]}"
    )
    handle = path.open("a+b")
    handle.write(b"0")
    handle.flush()
    if not _try_lock_file(handle):
        handle.close()
        raise OSError("cannot reserve Review Suite state")
    _leases[key] = handle

    def close() -> None:
        handle.close()
        path.unlink(missing_ok=True)

    atexit.register(close)


def _held(path: Path) -> bool:
    if path.is_dir():  # Older Arena writers use mkdir locks.
        return True
    with path.open("a+b") as handle:
        if not _try_lock_file(handle):
            return True
        _unlock_file(handle)
    return False


def _processes() -> list[tuple[int, int, str]]:
    if os.name == "nt":
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        rows = json.loads(result.stdout)
        return [
            (row["ProcessId"], row["ParentProcessId"], row.get("CommandLine") or "")
            for row in (rows if isinstance(rows, list) else [rows])
        ]
    result = subprocess.run(
        ["ps", "-ax", "-o", "pid=,ppid=,args="],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    return [
        (int(parts[0]), int(parts[1]), parts[2])
        for line in result.stdout.splitlines()
        if len(parts := line.strip().split(None, 2)) == 3
    ]


def _in_use(legacy: Path, control: Path) -> bool:
    # A process that deferred migration must keep that choice until it exits.
    for path in control.glob("use-*"):
        if _held(path):
            return True
        path.unlink(missing_ok=True)
    for path in legacy.rglob(".locks/*"):
        if _held(path):
            return True
    rows = _processes()
    ancestors = {os.getpid()}
    parents = {pid: parent for pid, parent, _ in rows}
    current = os.getpid()
    while (current := parents.get(current, 0)) and current not in ancestors:
        ancestors.add(current)
    # Old releases have no lifetime state lease. Their Python entry points
    # (including the tray's settings helper) must finish before moving state.
    for pid, _, command in rows:
        tokens = [
            quoted or plain for quoted, plain in re.findall(r'"([^"]*)"|(\S+)', command)
        ]
        # cmd /c and shell wrappers may quote the whole child command.
        tokens = [
            part
            for token in tokens
            for part in (token.split() if ".py " in token else [token])
        ]
        if "--help" in tokens or "-h" in tokens:
            continue
        if "--state-dir" in tokens:
            index = tokens.index("--state-dir") + 1
            if (
                index < len(tokens)
                and Path(tokens[index]).is_absolute()
                and Path(tokens[index]) != legacy
            ):
                continue
        if "review-suite-settings.py" in command:
            paths = [Path(token) for token in tokens if token.endswith("settings.toml")]
            if paths and paths[0].is_absolute() and paths[0].parent != legacy:
                continue
        # A separately installed copy under an isolated home cannot write this
        # home's defaults. Keep repository launches conservative (cwd unknown).
        script = next((token for token in tokens if token.endswith(".py")), "")
        normalized = script.replace("\\", "/").casefold()
        install_home = re.split(r"/\.(?:codex|agents)/", normalized)[0]
        if install_home != normalized and install_home not in {
            str(Path.home()).replace("\\", "/").casefold(),
            str(legacy.parents[1].parent).replace("\\", "/").casefold(),
        }:
            continue
        if pid not in ancestors and re.search(
            r"(?:[/\\\s\"])(?:review(?:_[\w]+)?|review-suite-settings|migrate_review_names)\.py(?:[\s\"]|$)",
            command,
        ):
            return True
    # Detached reviewers can outlive the wrapper that recorded their PID.
    live = {pid for pid, _, _ in rows}
    for path in legacy.rglob("rounds/*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if any(
            run.get("pid") in live
            for run in payload.get("runs", [])
            if run.get("status") not in {"completed", "failed", "cancelled"}
        ):
            return True
    return False


def _link(legacy: Path, target: Path) -> None:
    if os.name == "nt":
        # Native junction creation; no shell interpolation or symlink privilege.
        import _winapi

        _winapi.CreateJunction(str(target), str(legacy))
    else:
        legacy.symlink_to(target, target_is_directory=True)


def _tree(root: Path) -> dict[str, tuple]:
    result = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            key = path.relative_to(root).as_posix()
            if path.is_symlink() or path.is_junction():
                result[key] = ("link", os.readlink(path))
                if name in dirs:
                    dirs.remove(name)
            elif path.is_dir():
                result[key] = ("dir",)
            else:
                with path.open("rb") as handle:
                    result[key] = (
                        "file",
                        hashlib.file_digest(handle, "sha256").hexdigest(),
                    )
    return result


def _move(legacy: Path, target: Path, control: Path) -> None:
    journal = control / "pending.json"
    staging = target.with_name(".review-suite-copy")
    retired = legacy.with_name(".review-suite-moved")
    journal.write_text(json.dumps({"legacy": str(legacy)}), encoding="utf-8")
    try:
        legacy.rename(target)
    except OSError as error:
        if error.errno != errno.EXDEV:
            journal.unlink()
            raise
        shutil.rmtree(staging, ignore_errors=True)
        shutil.copytree(legacy, staging, symlinks=True)
        if _tree(legacy) != _tree(staging):
            raise OSError("Review Suite state copy verification failed")
        staging.rename(target)
        legacy.rename(retired)
    try:
        _link(legacy, target)
    except OSError:
        # Do not leave older installs able to create a separate empty state.
        if retired.exists():
            retired.rename(legacy)
            shutil.rmtree(target)
        else:
            target.rename(legacy)
        journal.unlink()
        raise
    if retired.exists():
        shutil.rmtree(retired)
    journal.unlink()


def _recover(target: Path, control: Path) -> None:
    journal = control / "pending.json"
    if not journal.exists():
        return
    legacy = Path(json.loads(journal.read_text(encoding="utf-8"))["legacy"])
    retired = legacy.with_name(".review-suite-moved")
    if target.exists():
        if legacy.exists() and not (legacy.is_symlink() or legacy.is_junction()):
            # Interrupted copy promotion: leave the original authoritative and
            # retry the verified copy, rather than guessing which copy changed.
            if _in_use(legacy, control):
                raise OSError("Review Suite state is still in use")
            shutil.rmtree(target)
        elif not legacy.exists():
            _link(legacy, target)
    if retired.exists():
        shutil.rmtree(retired)
    shutil.rmtree(target.with_name(".review-suite-copy"), ignore_errors=True)
    journal.unlink()


def default_state_dir() -> Path:
    override = os.environ.get("REVIEW_SUITE_STATE_DIR", "").strip()
    if override:
        # An explicit independent state must not import the user's history.
        return Path(override).expanduser().absolute()
    target = Path.home() / ".agents" / "state" / "review-suite"
    codex = (
        Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
        .expanduser()
        .absolute()
    )
    sources = list(
        dict.fromkeys(
            [
                codex / "state" / "review-suite",
                Path.home() / ".codex" / "state" / "review-suite",
            ]
        )
    )
    control = target.parent / ".review-suite-migration"
    # No directory creation, process scan or lock on the normal migrated path.
    if not (control / "pending.json").exists() and not any(
        path.exists() and not (path.is_symlink() or path.is_junction())
        for path in sources
    ):
        return target
    with orchestrator_store_lock(state_dir=control, name="migration"):
        try:
            _recover(target, control)
        except OSError, ValueError, subprocess.SubprocessError:
            # Never select a partially promoted destination after interruption.
            for legacy in sources:
                if legacy.exists() and not (
                    legacy.is_symlink() or legacy.is_junction()
                ):
                    _lease(control, legacy)
                    return legacy
            raise
        for legacy in sources:
            if not legacy.exists() or legacy.is_symlink() or legacy.is_junction():
                continue
            if target.exists():
                marker = control / (
                    "reported-" + hashlib.sha256(str(legacy).encode()).hexdigest()[:16]
                )
                if not marker.exists():
                    print(
                        f"Review Suite uses {target}; existing state at {legacy} was left unchanged.",
                        file=sys.stderr,
                    )
                    marker.touch()
                continue
            try:
                if _in_use(legacy, control):
                    _lease(control, legacy)
                    return legacy
                target.parent.mkdir(parents=True, exist_ok=True)
                _move(legacy, target, control)
            except OSError, ValueError, subprocess.SubprocessError:
                if legacy.exists() and not (
                    legacy.is_symlink() or legacy.is_junction()
                ):
                    _lease(control, legacy)
                    return legacy
                raise
        return target
