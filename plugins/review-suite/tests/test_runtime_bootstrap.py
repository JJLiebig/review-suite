from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator, Sequence

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import review
import review_suite_arena
import review_suite_runtime_bootstrap as runtime_bootstrap
from review_suite_local import _review_status_command
from review_suite_runtime_bootstrap import (
    READY_MARKER,
    bootstrap_runtime,
    content_hash,
    ensure_runtime_build,
    prune_runtimes,
)


class ExecCalled(Exception):
    pass


LAUNCHER_ENTRYPOINTS = (
    "review.py",
    "review_deslop.py",
    "review_followup.py",
    "review_github.py",
    "review_plan.py",
    "review_suite_arena.py",
)


@pytest.fixture
def runtimes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("REVIEW_SUITE_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.delenv(runtime_bootstrap.DISABLE_ENV, raising=False)
    monkeypatch.delenv(runtime_bootstrap.LAUNCHER_SCRIPT_ENV, raising=False)
    return tmp_path / "state" / "runtimes"


@pytest.fixture
def live_process() -> Iterator[subprocess.Popen[bytes]]:
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        yield process
    finally:
        process.kill()
        process.wait()


def _dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_plugin(root: Path, *, version: str = "1.2.3") -> Path:
    plugin_root = root / "plugins" / "review-suite"
    _write(
        plugin_root / ".codex-plugin" / "plugin.json",
        json.dumps({"name": "review-suite", "version": version}),
    )
    _write(plugin_root / "scripts" / "review.py", "print('review')\n")
    _write(plugin_root / "default_settings.toml", '[base]\nmodel = "first"\n')
    return plugin_root.resolve()


def _copy_real_plugin(root: Path) -> Path:
    plugin_root = root / "plugins" / "review-suite"
    shutil.copytree(
        SCRIPT_DIR.parent,
        plugin_root,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests"),
    )
    return plugin_root.resolve()


def _lease(build: Path, pid: int) -> None:
    _write(build / f".active-{pid}.json", json.dumps({"pid": pid}))


def _release_own_lease(build: Path) -> None:
    (build / f".active-{os.getpid()}.json").unlink()


@pytest.mark.parametrize("entrypoint", LAUNCHER_ENTRYPOINTS)
@pytest.mark.usefixtures("review_process_lock")
def test_entrypoint_runs_from_state_build_and_releases_it(
    tmp_path: Path, runtimes: Path, entrypoint: str
) -> None:
    plugin_root = _copy_real_plugin(tmp_path / "install")
    env = os.environ.copy()
    env.pop("PYTHONDONTWRITEBYTECODE", None)

    result = subprocess.run(
        [sys.executable, str(plugin_root / "scripts" / entrypoint), "--help"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode in {0, 2}, result.stderr
    assert not list(plugin_root.rglob("__pycache__"))
    (build,) = [path for path in runtimes.iterdir() if path.is_dir()]
    assert list(build.rglob("*.pyc"))
    assert (build / READY_MARKER).exists()
    assert not list(build.glob(".active*"))


def test_launcher_execs_build_under_its_lease(tmp_path: Path, runtimes: Path) -> None:
    plugin_root = _make_plugin(tmp_path / "install")
    seen: list[tuple[Sequence[str], str | None, bool]] = []

    def fake_execv(_executable: str, argv: Sequence[str]) -> object:
        build = Path(argv[1]).parents[1]
        lease = build / f".active-{os.getpid()}.json"
        seen.append(
            (
                argv,
                os.environ.get(runtime_bootstrap.LAUNCHER_SCRIPT_ENV),
                lease.exists(),
            )
        )
        raise ExecCalled

    with pytest.raises(ExecCalled):
        bootstrap_runtime(
            plugin_root / "scripts" / "review.py",
            argv=["review.py", "--mode", "base"],
            executable="python",
            execv=fake_execv,
            platform_name="posix",
        )

    ((argv, launcher, leased),) = seen
    assert argv[0] == "python"
    assert Path(argv[1]).parents[1] == runtimes / content_hash(plugin_root)[:8]
    assert Path(argv[1]).read_text(encoding="utf-8") == "print('review')\n"
    assert tuple(argv[2:]) == ("--mode", "base")
    assert launcher == str(plugin_root / "scripts" / "review.py")
    assert leased
    assert not list(Path(argv[1]).parents[1].glob(".active*"))
    assert os.environ.get(runtime_bootstrap.LAUNCHER_SCRIPT_ENV) is None


def test_windows_launcher_holds_lease_until_child_exits(
    tmp_path: Path, runtimes: Path
) -> None:
    plugin_root = _make_plugin(tmp_path / "install")
    seen: list[bool] = []

    def fake_run(_executable: str, argv: Sequence[str]) -> int:
        seen.append((Path(argv[1]).parents[1] / f".active-{os.getpid()}.json").exists())
        return 42

    with pytest.raises(SystemExit) as exc:
        bootstrap_runtime(
            plugin_root / "scripts" / "review.py",
            argv=["review.py"],
            executable="python",
            run=fake_run,
            platform_name="nt",
        )

    assert exc.value.code == 42
    assert seen == [True]
    assert not list(runtimes.rglob(".active*"))


def test_source_checkout_runs_in_place(tmp_path: Path, runtimes: Path) -> None:
    plugin_root = _make_plugin(tmp_path / "repo")
    (tmp_path / "repo" / ".git").mkdir()

    assert bootstrap_runtime(plugin_root / "scripts" / "review.py") is False
    assert not runtimes.exists()


def test_launch_removes_every_unleased_build_but_the_current_one(
    tmp_path: Path, runtimes: Path, live_process: subprocess.Popen[bytes]
) -> None:
    plugin_root = _make_plugin(tmp_path / "install")
    script = plugin_root / "scripts" / "review.py"
    leased = ensure_runtime_build(plugin_root, runtimes)
    _release_own_lease(leased)
    _lease(leased, live_process.pid)
    script.write_text("print('second')\n", encoding="utf-8")
    unleased = ensure_runtime_build(plugin_root, runtimes)
    _release_own_lease(unleased)
    _lease(unleased, _dead_pid())
    parent = unleased.parent
    (parent / f".staging-{_dead_pid()}-abandoned").mkdir()
    (parent / ".trash-abandoned").mkdir()
    script.write_text("print('third')\n", encoding="utf-8")

    current = ensure_runtime_build(plugin_root, runtimes)
    prune_runtimes(runtimes, build=current)

    assert sorted(path.name for path in parent.iterdir() if path.is_dir()) == sorted(
        [leased.name, current.name]
    )
    assert (leased / READY_MARKER).exists()

    _release_own_lease(current)
    # Reaped, so it no longer counts as running.
    live_process.kill()
    live_process.wait()
    again = ensure_runtime_build(plugin_root, runtimes)
    prune_runtimes(runtimes, build=again)

    assert again == current
    assert [path.name for path in parent.iterdir() if path.is_dir()] == [current.name]


def test_legacy_codex_builds_follow_the_earlier_retention_rule(
    tmp_path: Path, runtimes: Path
) -> None:
    legacy = tmp_path / "codex" / "plugin-runtimes" / "review-suite"
    _write(legacy / "0.1.0-idle" / "runtime_metadata.json", "{}")
    _write(legacy / "0.1.0-stale-lease" / ".active.1.json", "{}")
    os.utime(legacy / "0.1.0-stale-lease" / ".active.1.json", (0, 0))
    # Earlier versions refresh a running build's lease every minute.
    _write(legacy / "0.1.0-running" / ".active.2.json", "")
    for folder in legacy.iterdir():
        os.utime(folder, (0, 0))
    # An earlier launcher may still be copying or starting this one.
    (legacy / ".staging.tmp.3.recent").mkdir()
    plugin_root = _make_plugin(tmp_path / "install")

    build = ensure_runtime_build(plugin_root, runtimes)
    prune_runtimes(runtimes, build=build)

    assert sorted(path.name for path in legacy.iterdir()) == [
        ".staging.tmp.3.recent",
        "0.1.0-running",
    ]


def test_copy_taken_during_install_replacement_is_retried(
    tmp_path: Path, runtimes: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin_root = _make_plugin(tmp_path / "install")
    script = plugin_root / "scripts" / "review.py"
    original_copy = runtime_bootstrap._copy_runtime_items
    calls: list[int] = []

    def copy_during_replacement(source: Path, staging: Path) -> None:
        calls.append(len(calls))
        if len(calls) == 1:
            original_copy(source, staging)
            script.write_text("print('new version')\n", encoding="utf-8")
        elif len(calls) == 2:
            staging.mkdir()
            raise FileNotFoundError(source / "scripts" / "review.py")
        else:
            original_copy(source, staging)

    monkeypatch.setattr(
        runtime_bootstrap, "_copy_runtime_items", copy_during_replacement
    )
    monkeypatch.setattr(runtime_bootstrap, "INSTALL_RETRY_INTERVAL_SECONDS", 0)

    build = ensure_runtime_build(plugin_root, runtimes)

    assert len(calls) == 3
    assert (build / "scripts" / "review.py").read_text(encoding="utf-8") == (
        "print('new version')\n"
    )
    assert content_hash(build) == content_hash(plugin_root)
    assert [path.name for path in build.parent.iterdir() if path.is_dir()] == [
        build.name
    ]


def test_detached_child_keeps_build_leased_after_launch(
    tmp_path: Path, runtimes: Path
) -> None:
    build = ensure_runtime_build(_copy_real_plugin(tmp_path / "install"), runtimes)
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "from pathlib import Path;"
        "from review_suite_core.process_runtime import launch_captured_child_process;"
        "child = launch_captured_child_process(command=[sys.executable, '-c', "
        "'import time; time.sleep(60)'], cwd=Path.cwd(), stdout_prefix='lease-test-');"
        "print(child.process.pid)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(build / "scripts")],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    child_pid = int(result.stdout.strip())
    try:
        assert runtime_bootstrap._has_live_lease(build)
        assert (build / f".active-{child_pid}.json").exists()
    finally:
        _kill(child_pid)
    _release_own_lease(build)
    assert not runtime_bootstrap._has_live_lease(build)


def _kill(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
    else:
        os.kill(pid, 9)
    deadline = time.monotonic() + 10
    while runtime_bootstrap._process_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)


def test_runtime_hash_ignores_volatile_state(tmp_path: Path) -> None:
    plugin_root = _make_plugin(tmp_path / "install")
    initial_hash = content_hash(plugin_root)

    _write(
        plugin_root / "scripts" / "__pycache__" / "review.cpython-314.pyc", "bytecode"
    )
    _write(plugin_root / ".pytest_cache" / "README.md", "cache")

    assert content_hash(plugin_root) == initial_hash


def test_followup_command_uses_launcher_script_path_after_reexec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime_script = (
        tmp_path / "state" / "runtimes" / "key" / "build" / "scripts" / "review.py"
    )
    launcher_script = tmp_path / "install" / "scripts" / "review.py"

    monkeypatch.setattr(review, "__file__", str(runtime_script))
    monkeypatch.setenv(runtime_bootstrap.LAUNCHER_SCRIPT_ENV, str(launcher_script))

    command = review._review_command("rvw_123", "--decision", "clean")

    assert str(launcher_script) in command
    assert str(runtime_script) not in command


def test_action_command_helpers_use_launcher_paths_after_reexec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launcher_script = tmp_path / "install" / "scripts" / "review.py"
    monkeypatch.setenv(runtime_bootstrap.LAUNCHER_SCRIPT_ENV, str(launcher_script))

    commands = [
        review_suite_arena._grade_command(
            round_id="round", state_dir=tmp_path / "state"
        ),
        _review_status_command(review_cwd=tmp_path, base="main"),
    ]
    command_text = "\n".join(commands).replace("\\", "/")

    assert str(launcher_script.parent).replace("\\", "/") in command_text
    assert str(SCRIPT_DIR).replace("\\", "/") not in command_text
    assert "review_suite_arena.py grade" in command_text


def test_runtime_loads_split_settings_with_existing_user_overrides(
    tmp_path: Path, runtimes: Path
) -> None:
    runtime = ensure_runtime_build(_copy_real_plugin(tmp_path / "install"), runtimes)
    state = tmp_path / "settings-state"
    state.mkdir()
    (state / "settings.toml").write_text(
        '[base]\nmodel = "gpt-6-sol"\n[arena]\nenabled = true\n[orchestrator.stable_defaults]\nbase_arena_loops = 2\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
            "from review_suite_core.config import load_config; "
            "state = Path(sys.argv[2]); config = load_config(state); "
            "assert config['arena']['enabled'] is True; "
            "assert config['arena']['pools']['arena_base']['variant_groups']; "
            "assert config['orchestrator']['stable_defaults']['base_arena_loops'] == 2; "
            "assert config['base']['model'] == 'gpt-6-sol'",
            str(runtime / "scripts"),
            str(state),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
