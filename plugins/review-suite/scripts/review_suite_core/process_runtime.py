from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from review_suite_runtime_bootstrap import lease_child_process, release_child_process

IS_WINDOWS = os.name == "nt"
POSIX_SIGTERM = 15
POSIX_SIGKILL = 9


@dataclass(frozen=True)
class CapturedChildProcess:
    process: subprocess.Popen[str]
    stdout_path: Path
    stderr_path: Path
    started_monotonic: float


@dataclass(frozen=True)
class CapturedChildWaitResult:
    returncode: int
    timed_out: bool
    elapsed_seconds: float


def launch_captured_child_process(
    *,
    command: list[str],
    cwd: Path,
    env: dict[str, str] | None = None,
    stdin_text: str | None = None,
    stdout_prefix: str,
    stderr_prefix: str | None = None,
    stdout_suffix: str = ".stdout.txt",
    stderr_suffix: str = ".stderr.txt",
) -> CapturedChildProcess:
    stdout_tmp = tempfile.NamedTemporaryFile(
        prefix=stdout_prefix, suffix=stdout_suffix, delete=False
    )
    stderr_tmp = tempfile.NamedTemporaryFile(
        prefix=stderr_prefix or stdout_prefix, suffix=stderr_suffix, delete=False
    )
    stdout_path = Path(stdout_tmp.name)
    stderr_path = Path(stderr_tmp.name)
    started_monotonic = time.monotonic()
    try:
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdin=subprocess.PIPE if stdin_text else None,
            stdout=stdout_tmp,
            stderr=stderr_tmp,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        # Detached reviewer runs can outlive this process.
        lease_child_process(proc.pid)
        if stdin_text and proc.stdin is not None:
            proc.stdin.write(stdin_text)
            proc.stdin.close()
        return CapturedChildProcess(
            process=proc,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            started_monotonic=started_monotonic,
        )
    except Exception:
        stdout_tmp.close()
        stderr_tmp.close()
        for path in (stdout_path, stderr_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    finally:
        stdout_tmp.close()
        stderr_tmp.close()


def _posix_process_tree(root_pid: int) -> list[int]:
    result = subprocess.run(
        ["ps", "-e", "-o", "pid=,ppid="],
        capture_output=True,
        text=True,
        check=False,
    )
    children: dict[int, list[int]] = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        child_pid, parent_pid = map(int, fields)
        children.setdefault(parent_pid, []).append(child_pid)
    descendants: list[int] = []
    pending = list(children.get(root_pid, []))
    while pending:
        child_pid = pending.pop()
        descendants.append(child_pid)
        pending.extend(children.get(child_pid, []))
    return [root_pid, *descendants]


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def terminate_process_tree(pid: int | None, *, grace_seconds: float = 0.5) -> None:
    if not pid:
        return
    if IS_WINDOWS:
        subprocess.run(
            ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    targets = _posix_process_tree(int(pid))
    for target in targets:
        try:
            os.kill(target, POSIX_SIGTERM)
        except OSError:
            pass
    deadline = time.monotonic() + max(0.0, grace_seconds)
    while time.monotonic() < deadline and any(
        _pid_exists(target) for target in targets
    ):
        time.sleep(0.05)
    for target in reversed(targets):
        if not _pid_exists(target):
            continue
        try:
            os.kill(target, POSIX_SIGKILL)
        except OSError:
            pass


def wait_for_captured_child_process(
    *,
    process: subprocess.Popen[str],
    started_monotonic: float,
    start_line: str | None,
    heartbeat_line: Callable[[int], str],
    timeout_line: Callable[[int], str],
    progress_interval_seconds: int,
    timeout_seconds: int,
    poll_interval_seconds: float = 1.0,
) -> CapturedChildWaitResult:
    last_progress = started_monotonic
    timed_out = False
    if start_line:
        print(start_line, file=sys.stderr, flush=True)
    while process.poll() is None:
        now = time.monotonic()
        elapsed = int(now - started_monotonic)
        if timeout_seconds > 0 and elapsed >= timeout_seconds:
            terminate_process_tree(process.pid)
            timed_out = True
            print(timeout_line(elapsed), file=sys.stderr, flush=True)
            break
        if (
            progress_interval_seconds >= 0
            and now - last_progress >= progress_interval_seconds
        ):
            print(heartbeat_line(elapsed), file=sys.stderr, flush=True)
            last_progress = now
        time.sleep(max(0.0, poll_interval_seconds))
    returncode = process.wait()
    release_child_process(process.pid)
    return CapturedChildWaitResult(
        returncode=returncode,
        timed_out=timed_out,
        elapsed_seconds=round(time.monotonic() - started_monotonic, 3),
    )
