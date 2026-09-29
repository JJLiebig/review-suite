from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import review
import review_suite_arena
import review_suite_runtime_bootstrap as runtime_bootstrap
from review_suite_local import _review_status_command
from review_suite_runtime_bootstrap import (
    METADATA_FILENAME,
    bootstrap_from_installed_cache,
    content_hash_for_runtime,
    ensure_runtime_copy,
    prepare_runtime_bootstrap,
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
    "review_t1.py",
    "review_t3.py",
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_cache_plugin(codex_home: Path, *, version: str = "1.2.3") -> Path:
    plugin_root = codex_home / "plugins" / "cache" / "market" / "review-suite" / "local"
    _write(
        plugin_root / ".codex-plugin" / "plugin.json",
        json.dumps({"name": "review-suite", "version": version}),
    )
    _write(plugin_root / "scripts" / "review.py", "print('review')\n")
    _write(
        plugin_root / "scripts" / "review_suite_runtime_bootstrap.py",
        "print('bootstrap')\n",
    )
    _write(plugin_root / "default_settings.toml", '[normal]\nmodel = "first"\n')
    _write(plugin_root / "assets" / "logo.txt", "asset\n")
    return plugin_root


@pytest.mark.parametrize("entrypoint", LAUNCHER_ENTRYPOINTS)
def test_installed_cache_launcher_does_not_write_source_bytecode(
    tmp_path: Path, entrypoint: str
) -> None:
    codex_home = tmp_path / "codex"
    source_root = SCRIPT_DIR.parent
    plugin_root = (
        codex_home / "plugins" / "cache" / "review-suite" / "review-suite" / "local"
    )
    shutil.copytree(
        source_root,
        plugin_root,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    env = os.environ.copy()
    env["CODEX_HOME"] = str(codex_home)
    env.pop(runtime_bootstrap.DISABLE_ENV, None)
    env.pop(runtime_bootstrap.LAUNCHER_SCRIPT_ENV, None)
    env.pop("PYTHONDONTWRITEBYTECODE", None)

    result = subprocess.run(
        [sys.executable, str(plugin_root / "scripts" / entrypoint), "--help"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode in {0, 2}, result.stderr
    assert not list(plugin_root.rglob("__pycache__"))
    assert not list(plugin_root.rglob("*.pyc"))
    assert list((codex_home / "plugin-runtimes" / "review-suite").rglob("*.pyc"))


def test_cache_launcher_reexecs_to_runtime_and_preserves_argv(
    tmp_path: Path,
) -> None:
    codex_home = tmp_path / "codex"
    plugin_root = _make_cache_plugin(codex_home)
    calls: list[tuple[str, Sequence[str]]] = []

    def fake_execv(executable: str, argv: Sequence[str]) -> object:
        calls.append((executable, argv))
        raise ExecCalled

    with pytest.raises(ExecCalled):
        bootstrap_from_installed_cache(
            plugin_root / "scripts" / "review.py",
            argv=["review.py", "--mode", "normal", "--cd", "repo"],
            environ={"CODEX_HOME": str(codex_home)},
            executable="python",
            execv=fake_execv,
            platform_name="posix",
        )

    assert len(calls) == 1
    executable, argv = calls[0]
    assert executable == "python"
    assert argv[0] == "python"
    assert Path(argv[1]).is_relative_to(codex_home / "plugin-runtimes" / "review-suite")
    assert Path(argv[1]).name == "review.py"
    assert tuple(argv[2:]) == ("--mode", "normal", "--cd", "repo")
    assert not Path(argv[1]).is_relative_to(plugin_root)


def test_windows_cache_launcher_waits_for_runtime_and_returns_child_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    codex_home = tmp_path / "codex"
    plugin_root = _make_cache_plugin(codex_home)
    execv_calls: list[tuple[str, Sequence[str]]] = []
    run_calls: list[tuple[str, Sequence[str]]] = []
    launcher_values: list[str | None] = []

    def fake_run(executable: str, argv: Sequence[str]) -> int:
        launcher_values.append(os.environ.get(runtime_bootstrap.LAUNCHER_SCRIPT_ENV))
        run_calls.append((executable, argv))
        return 42

    with pytest.raises(SystemExit) as exc:
        bootstrap_from_installed_cache(
            plugin_root / "scripts" / "review.py",
            argv=["review.py", "--mode", "fast", "--base", "main"],
            environ={"CODEX_HOME": str(codex_home)},
            executable="python",
            execv=lambda executable, argv: execv_calls.append((executable, argv)),
            run=fake_run,
            platform_name="nt",
        )

    assert exc.value.code == 42
    assert execv_calls == []
    assert len(run_calls) == 1
    executable, argv = run_calls[0]
    assert executable == "python"
    assert Path(argv[1]).is_relative_to(codex_home / "plugin-runtimes" / "review-suite")
    assert tuple(argv[2:]) == ("--mode", "fast", "--base", "main")
    assert launcher_values == [str(plugin_root / "scripts" / "review.py")]
    assert os.environ.get(runtime_bootstrap.LAUNCHER_SCRIPT_ENV) is None


def test_already_running_from_runtime_does_not_reexec(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex"
    plugin_root = _make_cache_plugin(codex_home)
    runtime_root = ensure_runtime_copy(plugin_root, codex_home=codex_home)
    calls: list[tuple[str, Sequence[str]]] = []

    result = bootstrap_from_installed_cache(
        runtime_root / "scripts" / "review.py",
        argv=["review.py", "--id", "rvw_123"],
        environ={"CODEX_HOME": str(codex_home)},
        executable="python",
        execv=lambda executable, argv: calls.append((executable, argv)),
    )

    assert result is False
    assert calls == []


def test_runtime_directory_is_created_and_reused(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex"
    plugin_root = _make_cache_plugin(codex_home)

    first_root = ensure_runtime_copy(plugin_root, codex_home=codex_home)
    second_root = ensure_runtime_copy(plugin_root, codex_home=codex_home)
    expected_hash = content_hash_for_runtime(plugin_root)

    assert first_root == second_root
    assert first_root.name.startswith("1.2.3-")
    assert (first_root / "scripts" / "review.py").read_text(
        encoding="utf-8"
    ) == "print('review')\n"
    assert (
        first_root / "default_settings.toml"
    ).read_text() == '[normal]\nmodel = "first"\n'
    (plugin_root / "default_settings.toml").write_text(
        '[normal]\nmodel = "updated"\n', encoding="utf-8"
    )
    updated_root = ensure_runtime_copy(plugin_root, codex_home=codex_home)
    assert updated_root != first_root
    assert (
        updated_root / "default_settings.toml"
    ).read_text() == '[normal]\nmodel = "updated"\n'
    metadata = json.loads((first_root / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert metadata["source_path"] == str(plugin_root.resolve(strict=False))
    assert metadata["version"] == "1.2.3"
    assert metadata["content_hash"] == expected_hash
    assert metadata["created_at"]
    assert list(first_root.parent.glob("*.tmp.*")) == []


def test_runtime_reuse_does_not_stage_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    codex_home = tmp_path / "codex"
    plugin_root = _make_cache_plugin(codex_home)
    runtime_root = ensure_runtime_copy(plugin_root, codex_home=codex_home)

    def fail_copy(_source_root: Path, _temp_root: Path) -> None:
        raise AssertionError("reuse should not create a staging copy")

    monkeypatch.setattr(runtime_bootstrap, "_copy_runtime_items", fail_copy)

    assert ensure_runtime_copy(plugin_root, codex_home=codex_home) == runtime_root


def test_runtime_cleanup_keeps_current_and_unowned_roots(tmp_path: Path) -> None:
    parent = tmp_path / "codex" / "plugin-runtimes" / "review-suite"
    parent.mkdir(parents=True)

    def make_runtime(name: str, created_at: str) -> Path:
        root = parent / name
        root.mkdir()
        _write(
            root / METADATA_FILENAME,
            json.dumps(
                {
                    "plugin": "review-suite",
                    "runtime_key": name,
                    "created_at": created_at,
                }
            ),
        )
        return root

    current = make_runtime("0.1.0-current", "2026-01-05T00:00:00Z")
    active = make_runtime("0.1.0-active", "2026-01-01T00:00:00Z")
    _write(active / ".active.123.json", "{}\n")
    fresh = make_runtime("0.1.0-fresh", "2999-01-01T00:00:00Z")
    old = make_runtime("0.1.0-old", "2026-01-02T00:00:00Z")
    no_metadata = parent / "0.1.0-no-metadata"
    no_metadata.mkdir()

    runtime_bootstrap._cleanup_stale_runtime_roots(parent, keep_root=current)

    assert current.exists()
    assert active.exists()
    assert fresh.exists()
    assert no_metadata.exists()
    assert not old.exists()


def test_runtime_hash_ignores_volatile_state(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex"
    plugin_root = _make_cache_plugin(codex_home)
    initial_hash = content_hash_for_runtime(plugin_root)

    _write(
        plugin_root / "scripts" / "__pycache__" / "review.cpython-314.pyc", "bytecode"
    )
    _write(plugin_root / ".pytest_cache" / "README.md", "cache")

    assert content_hash_for_runtime(plugin_root) == initial_hash


def test_runtime_key_matches_staged_copy_when_source_changes_during_install(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    codex_home = tmp_path / "codex"
    plugin_root = _make_cache_plugin(codex_home)
    original_copy = runtime_bootstrap._copy_runtime_items

    def copy_then_mutate_source(source_root: Path, temp_root: Path) -> None:
        _write(
            source_root / ".codex-plugin" / "plugin.json",
            json.dumps({"name": "review-suite", "version": "9.9.9"}),
        )
        original_copy(source_root, temp_root)
        _write(source_root / "scripts" / "review.py", "print('changed after copy')\n")

    monkeypatch.setattr(
        runtime_bootstrap, "_copy_runtime_items", copy_then_mutate_source
    )

    runtime_root = ensure_runtime_copy(plugin_root, codex_home=codex_home)
    metadata = json.loads(
        (runtime_root / METADATA_FILENAME).read_text(encoding="utf-8")
    )

    assert metadata["content_hash"] == content_hash_for_runtime(runtime_root)
    assert metadata["content_hash"] != content_hash_for_runtime(plugin_root)
    assert metadata["version"] == "9.9.9"
    assert runtime_root.name.startswith("9.9.9-")
    assert (runtime_root / "scripts" / "review.py").read_text(
        encoding="utf-8"
    ) == "print('review')\n"


def test_prepare_runtime_bootstrap_returns_none_outside_installed_cache(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "repo" / "plugins" / "review-suite"
    _write(
        source_root / ".codex-plugin" / "plugin.json",
        json.dumps({"name": "review-suite", "version": "1.2.3"}),
    )
    _write(source_root / "scripts" / "review.py", "print('review')\n")

    plan = prepare_runtime_bootstrap(
        source_root / "scripts" / "review.py",
        argv=["review.py"],
        environ={"CODEX_HOME": str(tmp_path / "codex")},
        executable="python",
    )

    assert plan is None


def test_followup_command_uses_launcher_script_path_after_reexec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    codex_home = tmp_path / "codex"
    plugin_root = _make_cache_plugin(codex_home)
    runtime_root = ensure_runtime_copy(plugin_root, codex_home=codex_home)
    runtime_script = runtime_root / "scripts" / "review.py"
    launcher_script = plugin_root / "scripts" / "review.py"

    monkeypatch.setattr(review, "__file__", str(runtime_script))
    monkeypatch.setenv(runtime_bootstrap.LAUNCHER_SCRIPT_ENV, str(launcher_script))

    command = review._review_command("rvw_123", "--decision", "clean")

    assert str(launcher_script) in command
    assert str(runtime_script) not in command


def test_action_command_helpers_use_launcher_paths_after_reexec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    codex_home = tmp_path / "codex"
    plugin_root = _make_cache_plugin(codex_home)
    runtime_root = ensure_runtime_copy(plugin_root, codex_home=codex_home)
    launcher_script = plugin_root / "scripts" / "review.py"

    monkeypatch.setenv(runtime_bootstrap.LAUNCHER_SCRIPT_ENV, str(launcher_script))

    commands = [
        review_suite_arena._grade_command(
            round_id="round", state_dir=tmp_path / "state"
        ),
        _review_status_command(review_cwd=tmp_path, base="main"),
    ]
    command_text = "\n".join(commands).replace("\\", "/")

    assert str(plugin_root / "scripts").replace("\\", "/") in command_text
    assert str(runtime_root).replace("\\", "/") not in command_text
    assert "review_suite_arena.py grade" in command_text


def test_runtime_loads_split_settings_with_existing_user_overrides(
    tmp_path: Path,
) -> None:
    runtime = ensure_runtime_copy(SCRIPT_DIR.parent, codex_home=tmp_path / "codex")
    state = tmp_path / "state"
    state.mkdir()
    (state / "settings.toml").write_text(
        '[normal]\nmodel = "gpt-6-sol"\n[arena]\nenabled = true\n[orchestrator.stable_defaults]\nnormal_arena_loops = 2\n',
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
            "assert config['arena']['pools']['arena_phase']['variant_groups']; "
            "assert config['orchestrator']['stable_defaults']['normal_arena_loops'] == 2; "
            "assert config['normal']['model'] == 'gpt-6-sol'",
            str(runtime / "scripts"),
            str(state),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
