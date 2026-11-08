"""Runs Review Suite from a build in its state folder instead of from the install.

Updating an install replaces its folder, which is not atomic. So each launch copies the
install's runtime files to `<state>/runtimes/<content hash>` (or reuses the ready build
with that content), leases that build for its process and runs the entry script from it.
A build is removed as soon as no live process leases it and a launch finds the install
contents changed (two installs with different contents re-copy each other's idle builds).
Source checkouts run in place.
"""

from __future__ import annotations

import atexit
import ctypes
import functools
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

PLUGIN_NAME = "review-suite"
DISABLE_ENV = "REVIEW_SUITE_DISABLE_RUNTIME_BOOTSTRAP"
LAUNCHER_SCRIPT_ENV = "REVIEW_SUITE_LAUNCHER_SCRIPT"
METADATA_FILENAME = "runtime_metadata.json"
READY_MARKER = ".review-suite-runtime-ready"
RETIRING_MARKER = ".review-suite-runtime-retiring"
LEASE_PREFIX = ".active"
RUNTIME_ITEMS = (
    "scripts",
    "references",
    "assets",
    "config",
    ".codex-plugin",
    "default_settings.toml",
)
EXCLUDED_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
# An install update replaces its folder in well under a second.
INSTALL_SETTLE_SECONDS = 10.0
INSTALL_RETRY_INTERVAL_SECONDS = 0.2
RETIRING_STALE_SECONDS = 30.0


class InstallChanged(Exception):
    """The install changed or was incomplete while it was being read."""


def bootstrap_runtime(
    script_file: str | os.PathLike[str],
    *,
    argv: Sequence[str] | None = None,
    executable: str | None = None,
    execv: Callable[[str, Sequence[str]], object] = os.execv,
    run: Callable[[str, Sequence[str]], int] | None = None,
    platform_name: str | None = None,
) -> bool:
    """Re-runs an entry script from its install's build; returns False to run in place."""
    script_path = Path(script_file).resolve(strict=False)
    plugin_root = find_plugin_root(script_path)
    if plugin_root is None or os.environ.get(DISABLE_ENV):
        return False
    if is_runtime_build(plugin_root):
        _lease_for_life(plugin_root)
        return False
    if _is_source_checkout(plugin_root):
        return False
    runtimes_root = _runtimes_root()
    build = ensure_runtime_build(plugin_root, runtimes_root)
    prune_runtimes(runtimes_root, build=build)
    relative_script = script_path.relative_to(plugin_root)
    exe = str(executable or sys.executable)
    current_argv = tuple(sys.argv if argv is None else argv)
    next_argv = (exe, str(build / relative_script), *current_argv[1:])
    previous_launcher = os.environ.get(LAUNCHER_SCRIPT_ENV)
    # Commands printed for later keep using the install, so they run its current build.
    os.environ[LAUNCHER_SCRIPT_ENV] = str(plugin_root / relative_script)
    try:
        if (platform_name or os.name) == "nt":
            exit_code = (
                run(exe, next_argv) if run else _run_runtime_process(exe, next_argv)
            )
            raise SystemExit(exit_code)
        # The process keeps its PID, so the build's lease stays with it.
        execv(exe, next_argv)
        raise RuntimeError("runtime bootstrap exec returned unexpectedly")
    finally:
        _lease_path(build, os.getpid()).unlink(missing_ok=True)
        if previous_launcher is None:
            os.environ.pop(LAUNCHER_SCRIPT_ENV, None)
        else:
            os.environ[LAUNCHER_SCRIPT_ENV] = previous_launcher


def launcher_script_path(
    current_file: str | os.PathLike[str], name: str | None = None
) -> Path:
    launcher = os.environ.get(LAUNCHER_SCRIPT_ENV)
    current = Path(current_file).resolve(strict=False)
    base = Path(launcher).resolve(strict=False) if launcher else current
    return base.with_name(name or current.name)


def lease_child_process(pid: int) -> None:
    """Leases the build this code runs from for a child process, until it exits."""
    build = current_runtime_build()
    if build is None:
        return
    try:
        _write_lease(_lease_path(build, pid), pid)
    except OSError:
        pass  # this process's own lease still covers the child while it runs


def release_child_process(pid: int) -> None:
    build = current_runtime_build()
    if build is not None:
        _lease_path(build, pid).unlink(missing_ok=True)


@functools.cache
def current_runtime_build() -> Path | None:
    plugin_root = find_plugin_root(Path(__file__).resolve(strict=False))
    return plugin_root if plugin_root and is_runtime_build(plugin_root) else None


def is_runtime_build(plugin_root: Path) -> bool:
    return (plugin_root / METADATA_FILENAME).is_file()


def find_plugin_root(path: Path) -> Path | None:
    for candidate in (path.parent, *path.parents):
        manifest_path = candidate / ".codex-plugin" / "plugin.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except OSError, json.JSONDecodeError:
            continue
        if str(manifest.get("name") or "") == PLUGIN_NAME:
            return candidate.resolve(strict=False)
    return None


def _is_source_checkout(plugin_root: Path) -> bool:
    return (plugin_root.parent.parent / ".git").exists()


def _runtimes_root() -> Path:
    # The state folder resolver lives in the install's engine, which may be mid-update.
    deadline = time.monotonic() + INSTALL_SETTLE_SECONDS
    while True:
        try:
            from review_suite_core.state_paths import default_state_dir

            return default_state_dir() / "runtimes"
        except ImportError, SyntaxError:
            if time.monotonic() >= deadline:
                raise
            for name in [
                name
                for name in sys.modules
                if name == "review_suite_core" or name.startswith("review_suite_core.")
            ]:
                del sys.modules[name]
            importlib.invalidate_caches()
            time.sleep(INSTALL_RETRY_INTERVAL_SECONDS)


def _run_runtime_process(executable: str, argv: Sequence[str]) -> int:
    return subprocess.run(list(argv), executable=executable, check=False).returncode


def ensure_runtime_build(plugin_root: Path, runtimes_root: Path) -> Path:
    """Returns the ready build of the install's current contents, leased for this process."""
    deadline = time.monotonic() + INSTALL_SETTLE_SECONDS
    while True:
        try:
            return _claim_or_build(plugin_root, runtimes_root)
        except (OSError, InstallChanged) as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"cannot prepare a {PLUGIN_NAME} runtime build from {plugin_root}: {exc}"
                ) from exc
            time.sleep(INSTALL_RETRY_INTERVAL_SECONDS)


def _claim_or_build(plugin_root: Path, parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    prune_builds(parent)
    _version(plugin_root)  # an install mid-replacement may lack its manifest
    target = parent / _build_name(content_hash(plugin_root))
    if (target / READY_MARKER).exists() and _claim(target):
        return target
    staging = parent / f".staging-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        _copy_runtime_items(plugin_root, staging)
        version = _version(staging)
        digest = content_hash(staging)
        # A copy taken while the install was being replaced mixes or misses files.
        if content_hash(plugin_root) != digest:
            raise InstallChanged(f"{plugin_root} changed while it was copied")
        target = parent / _build_name(digest)
        _write_metadata(
            staging, plugin_root=plugin_root, version=version, digest=digest
        )
        # Leased and ready from the moment it appears, so cleanup never takes it.
        _write_lease(_lease_path(staging, os.getpid()), os.getpid())
        (staging / READY_MARKER).touch()
        while True:
            try:
                staging.rename(target)
                return target
            except OSError:
                existed = target.exists()
                if existed and (target / READY_MARKER).exists() and _claim(target):
                    return target
                # Retry only while cleanup is deciding on that build, or once it removed it.
                if existed and (not target.exists() or _is_retiring(target)):
                    time.sleep(0.05)
                    continue
                raise
    finally:
        _remove(staging)


def _claim(build: Path) -> bool:
    """Leases a build, then confirms cleanup did not retire it first."""
    lease = _lease_path(build, os.getpid())
    try:
        _write_lease(lease, os.getpid())
    except OSError:
        return False
    if (build / READY_MARKER).exists():
        return True
    lease.unlink(missing_ok=True)
    return False


def _lease_for_life(build: Path) -> None:
    pid = os.getpid()
    lease = _lease_path(build, pid)
    try:
        _write_lease(lease, pid)
    except OSError:
        return
    atexit.register(lease.unlink, missing_ok=True)


def _lease_path(build: Path, pid: int) -> Path:
    return build / f"{LEASE_PREFIX}-{pid}.json"


def _write_lease(path: Path, pid: int) -> None:
    path.write_text(json.dumps({"pid": pid}) + "\n", encoding="utf-8")


def prune_runtimes(runtimes_root: Path, *, build: Path) -> None:
    """Removes every build no live process leases, except the one this launch uses.

    Cleanup never fails a launch: a folder that cannot be moved (on Windows, a process still
    has it open) is skipped and retried by a later launch.
    """
    prune_builds(runtimes_root, keep=build, retire_finished=True)
    _prune_legacy_builds()


def prune_builds(
    parent: Path, *, keep: Path | None = None, retire_finished: bool = False
) -> None:
    """Removes leftovers; with `retire_finished`, also every other unleased build."""
    for root in _directories(parent):
        try:
            if root == keep:
                continue
            if root.name.startswith(".trash-"):
                _remove(root)
                continue
            if root.name.startswith(".staging-"):
                if not _process_alive(_staging_pid(root.name)):
                    _retire(parent, root)
                continue
            if root.name.startswith("."):
                continue
            if not (root / READY_MARKER).exists():
                if _is_retiring(root):
                    continue
                # An interrupted retirement: keep the build so its leases are checked again.
                if (root / RETIRING_MARKER).exists():
                    _restore_ready(root)
                    continue
                # Recheck: another cleanup may have just restored its marker.
                if not (root / READY_MARKER).exists():
                    _retire(parent, root)
                continue
            if retire_finished:
                _retire_unleased(parent, root)
        except OSError:
            pass


def _prune_legacy_builds() -> None:
    """Removes the builds earlier versions kept in the Codex home."""
    codex_home = Path(
        os.environ.get("CODEX_HOME", "").strip() or Path.home() / ".codex"
    ).expanduser()
    legacy_root = codex_home / "plugin-runtimes" / PLUGIN_NAME
    for root in _directories(legacy_root):
        if root.name.startswith(".trash-") or not _has_live_lease(root):
            _retire(legacy_root, root)
    try:
        legacy_root.rmdir()
    except OSError:
        pass


# Cleanup sets a build's ready marker aside before checking its leases, while a launcher
# leases a build before checking that marker. So either cleanup sees the lease and keeps the
# build, or the launcher sees the build retiring and does not use it.
def _retire_unleased(parent: Path, root: Path) -> None:
    ready = root / READY_MARKER
    try:
        # Freshen the marker first, so the retirement never looks interrupted to others.
        os.utime(ready)
        os.replace(ready, root / RETIRING_MARKER)
    except OSError:
        return
    if not _has_live_lease(root) and _retire(parent, root):
        return
    _restore_ready(root)


def _restore_ready(root: Path) -> None:
    try:
        os.replace(root / RETIRING_MARKER, root / READY_MARKER)
    except OSError:
        pass


def _is_retiring(root: Path) -> bool:
    try:
        age = time.time() - (root / RETIRING_MARKER).stat().st_mtime
    except OSError:
        return False
    return age < RETIRING_STALE_SECONDS


def _retire(parent: Path, root: Path) -> bool:
    """Moves the folder aside in one step so a partly deleted build is never left behind."""
    trash = parent / f".trash-{uuid.uuid4().hex}"
    try:
        root.rename(trash)
    except OSError:
        return False
    _remove(trash)
    return True


def _remove(root: Path) -> None:
    shutil.rmtree(root, ignore_errors=True)


def _has_live_lease(root: Path) -> bool:
    try:
        leases = [path for path in root.iterdir() if path.name.startswith(LEASE_PREFIX)]
    except OSError:
        return True
    for lease in leases:
        try:
            if _process_alive(
                int(json.loads(lease.read_text(encoding="utf-8"))["pid"])
            ):
                return True
        except OSError, ValueError, KeyError, TypeError:
            pass
        lease.unlink(missing_ok=True)
    return False


def _staging_pid(name: str) -> int:
    try:
        return int(name.split("-")[1])
    except IndexError, ValueError:
        return 0


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except PermissionError:
            return True
        except OSError:
            return False
        return True
    # os.kill(pid, 0) on Windows sends a console control event instead of probing.
    synchronize, wait_timeout, access_denied = 0x00100000, 0x102, 5
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(synchronize, False, pid)
    if not handle:
        return ctypes.get_last_error() == access_denied
    try:
        return kernel32.WaitForSingleObject(handle, 0) == wait_timeout
    finally:
        kernel32.CloseHandle(handle)


def _directories(parent: Path) -> list[Path]:
    try:
        return [path for path in parent.iterdir() if path.is_dir()]
    except OSError:
        return []


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def content_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in _iter_runtime_files(root):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _version(root: Path) -> str:
    manifest_path = root / ".codex-plugin" / "plugin.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InstallChanged(f"cannot read {manifest_path}") from exc
    if str(manifest.get("name") or "") != PLUGIN_NAME:
        raise InstallChanged(f"plugin manifest is not {PLUGIN_NAME}: {manifest_path}")
    return str(manifest.get("version") or "0.0.0").strip() or "0.0.0"


def _build_name(digest: str) -> str:
    return digest[:8]


def _iter_runtime_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for item_name in RUNTIME_ITEMS:
        item = root / item_name
        if item.is_file() and not _excluded(item):
            files.append(item)
            continue
        if not item.is_dir():
            continue
        for path in item.rglob("*"):
            if path.is_file() and not _excluded(path):
                files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def _excluded(path: Path) -> bool:
    if path.suffix in EXCLUDED_SUFFIXES:
        return True
    return any(part in EXCLUDED_NAMES for part in path.parts)


def _ignore_runtime_items(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in EXCLUDED_NAMES or Path(name).suffix in EXCLUDED_SUFFIXES
    }


def _copy_runtime_items(plugin_root: Path, staging: Path) -> None:
    staging.mkdir(parents=True)
    for item_name in RUNTIME_ITEMS:
        source = plugin_root / item_name
        destination = staging / item_name
        if source.is_dir():
            shutil.copytree(source, destination, ignore=_ignore_runtime_items)
        elif source.is_file():
            shutil.copy2(source, destination)


def _write_metadata(
    root: Path, *, plugin_root: Path, version: str, digest: str
) -> None:
    payload = {
        "plugin": PLUGIN_NAME,
        "source_path": str(plugin_root),
        "version": version,
        "content_hash": digest,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    (root / METADATA_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
