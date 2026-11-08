from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from review_suite_core.process_runtime import (
    launch_captured_child_process,
    wait_for_captured_child_process,
)


def test_launch_captured_child_process_starts_clock_before_stdin_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    captured: dict[str, object] = {}

    class FakeStdin:
        def write(self, value: str) -> None:
            events.append(f"write:{value}")

        def close(self) -> None:
            events.append("close")

    class FakeProcess:
        pid = 42
        stdin = FakeStdin()

    def fake_monotonic() -> float:
        events.append("clock")
        return 123.0

    def fake_popen(*args, **kwargs) -> FakeProcess:
        events.append("popen")
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(
        "review_suite_core.process_runtime.time.monotonic", fake_monotonic
    )
    monkeypatch.setattr(
        "review_suite_core.process_runtime.subprocess.Popen", fake_popen
    )

    child = launch_captured_child_process(
        command=[sys.executable, "-c", "pass"],
        cwd=tmp_path,
        env={"SCOPED": "1"},
        stdin_text="prompt",
        stdout_prefix="child-stdout-",
        stderr_prefix="child-stderr-",
    )
    try:
        assert child.started_monotonic == 123.0
        assert events[:4] == ["clock", "popen", "write:prompt", "close"]
        assert captured["env"] == {"SCOPED": "1"}
    finally:
        child.stdout_path.unlink(missing_ok=True)
        child.stderr_path.unlink(missing_ok=True)


def test_wait_for_captured_child_process_keeps_child_stderr_captured(
    capsys,
    tmp_path: Path,
) -> None:
    child = launch_captured_child_process(
        command=[
            sys.executable,
            "-c",
            "import sys, time; print('child stdout'); print('child stderr', file=sys.stderr); time.sleep(0.05)",
        ],
        cwd=tmp_path,
        stdout_prefix="child-stdout-",
        stderr_prefix="child-stderr-",
        stdout_suffix=".txt",
        stderr_suffix=".txt",
    )
    try:
        result = wait_for_captured_child_process(
            process=child.process,
            started_monotonic=child.started_monotonic,
            start_line="[parent] waiting for child",
            heartbeat_line=lambda elapsed: f"OK {max(1, elapsed // 60)}m: child",
            timeout_line=lambda elapsed: f"child timed out after {elapsed}s",
            progress_interval_seconds=0,
            timeout_seconds=0,
            poll_interval_seconds=0.01,
        )
        captured = capsys.readouterr()

        assert result.returncode == 0
        assert "child stdout" in child.stdout_path.read_text(encoding="utf-8")
        assert "child stderr" in child.stderr_path.read_text(encoding="utf-8")
        assert "[parent] waiting for child" in captured.err
        assert "OK 1m: child" in captured.err
        assert "child stderr" not in captured.err
    finally:
        child.stdout_path.unlink(missing_ok=True)
        child.stderr_path.unlink(missing_ok=True)


def test_wait_timeout_terminates_process_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    terminated: list[int] = []

    class FakeProcess:
        pid = 42
        polls = 0

        def poll(self) -> int | None:
            self.polls += 1
            return None if self.polls == 1 else -1

        def wait(self) -> int:
            return -1

    clock = iter([1.0, 1.0])
    monkeypatch.setattr(
        "review_suite_core.process_runtime.time.monotonic", lambda: next(clock)
    )
    monkeypatch.setattr(
        "review_suite_core.process_runtime.terminate_process_tree",
        lambda pid: terminated.append(int(pid)),
    )

    result = wait_for_captured_child_process(
        process=FakeProcess(),
        started_monotonic=0.0,
        start_line=None,
        heartbeat_line=lambda _: "heartbeat",
        timeout_line=lambda _: "timeout",
        progress_interval_seconds=60,
        timeout_seconds=1,
        poll_interval_seconds=0,
    )

    assert result.timed_out is True
    assert terminated == [42]


def test_posix_process_tree_escalates_survivors_to_sigkill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr("review_suite_core.process_runtime.IS_WINDOWS", False)
    monkeypatch.setattr(
        "review_suite_core.process_runtime._posix_process_tree", lambda _: [10, 11]
    )
    monkeypatch.setattr("review_suite_core.process_runtime._pid_exists", lambda _: True)
    monkeypatch.setattr("review_suite_core.process_runtime.time.sleep", lambda _: None)
    monkeypatch.setattr(
        "review_suite_core.process_runtime.os.kill",
        lambda pid, sig: signals.append((pid, sig)),
    )

    from review_suite_core.process_runtime import terminate_process_tree

    terminate_process_tree(10, grace_seconds=0)

    assert signals == [
        (10, 15),
        (11, 15),
        (11, 9),
        (10, 9),
    ]
