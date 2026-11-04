from __future__ import annotations

import errno
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from review_suite_core import state_paths
from review_suite_core.config import load_config
from review_suite_core.orchestrator_store import orchestrator_store_lock


@pytest.fixture
def home(tmp_path, monkeypatch):
    for key in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(key, str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex"))
    monkeypatch.delenv("REVIEW_SUITE_STATE_DIR", raising=False)
    monkeypatch.delenv("REVIEW_SUITE_DISABLE_RUNTIME_BOOTSTRAP", raising=False)
    yield tmp_path
    for handle in state_paths._leases.values():
        handle.close()
    state_paths._leases.clear()


def seed(home):
    legacy = home / ".codex/state/review-suite"
    legacy.mkdir(parents=True)
    (legacy / "runs.jsonl").write_bytes(b'{"history":"preserve"}\n')
    (legacy / "names-backup-123").mkdir()
    (legacy / "names-backup-123/original.json").write_bytes(b"backup")
    return legacy, home / ".agents/state/review-suite"


def resolve_process():
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"import sys; sys.path.insert(0, {str(SCRIPTS)!r}); "
            "from review_suite_core.state_paths import default_state_dir; print(default_state_dir())",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


@pytest.mark.parametrize("route", ["skills", "plugin"])
@pytest.mark.parametrize("settings", ["config.json", "settings.toml"])
def test_install_first_run_moves_history_links_old_path_and_migrates_settings(
    home, route, settings
):
    legacy, target = seed(home)
    if settings == "config.json":
        (legacy / settings).write_text(json.dumps({"arena": {"enabled": False}}))
    else:
        (legacy / settings).write_text(
            '[arena]\nenabled = false\n[normal]\nreasoning = "low"\n'
        )
    destination = (
        home / ".agents/skills/review-suite/plugins/review-suite"
        if route == "skills"
        else home / ".codex/plugins/cache/review-suite/review-suite/local"
    )
    shutil.copytree(
        SCRIPTS.parent,
        destination,
        ignore=shutil.ignore_patterns("tests", "__pycache__"),
    )
    command = [
        sys.executable,
        str(destination / "scripts/review.py"),
        "--status",
        "--base",
        "HEAD",
        "--cd",
        str(SCRIPTS.parents[2]),
    ]
    first = subprocess.run(command, capture_output=True, text=True, timeout=40)
    assert first.returncode == 0, first.stdout + first.stderr
    assert target.is_dir()
    assert legacy.is_junction() or legacy.is_symlink()
    assert (target / "runs.jsonl").read_bytes() == b'{"history":"preserve"}\n'
    assert (target / "names-backup-123/original.json").read_bytes() == b"backup"
    assert load_config()["arena"]["enabled"] is False
    if settings == "settings.toml":
        assert load_config()["base"]["reasoning"] == "low"
        assert len(list(target.glob("settings-before-base-deep-*.toml"))) == 1
    (legacy / "older-install.txt").write_text("shared")
    before = (target / "runs.jsonl").stat().st_mtime_ns
    second = subprocess.run(command, capture_output=True, text=True, timeout=40)
    assert second.returncode == 0, second.stderr
    assert (target / "older-install.txt").read_text() == "shared"
    assert (target / "runs.jsonl").stat().st_mtime_ns == before


def test_both_exist_never_merge_and_report_once(home, capsys):
    legacy, target = seed(home)
    target.mkdir(parents=True)
    (target / "runs.jsonl").write_text("new history")
    assert state_paths.default_state_dir() == target
    assert str(legacy) in capsys.readouterr().err
    assert state_paths.default_state_dir() == target
    assert capsys.readouterr().err == ""
    assert (target / "runs.jsonl").read_text() == "new history"
    assert (legacy / "runs.jsonl").read_bytes() == b'{"history":"preserve"}\n'


def test_held_state_defers_until_writer_and_deferred_reader_exit(home):
    legacy, target = seed(home)
    with orchestrator_store_lock(state_dir=legacy, name="settings"):
        child = resolve_process()
        stdout, stderr = child.communicate(timeout=30)
        assert child.returncode == 0, stderr
        assert Path(stdout.strip()) == legacy
        assert not target.exists()
    child = resolve_process()
    stdout, stderr = child.communicate(timeout=30)
    assert child.returncode == 0, stderr
    assert Path(stdout.strip()) == target
    assert legacy.samefile(target)


def test_active_old_settings_writer_defers_migration(home):
    legacy, target = seed(home)
    script = home / "review-suite-settings.py"
    script.write_text("import sys; print('ready', flush=True); sys.stdin.read()")
    with subprocess.Popen(
        [sys.executable, str(script), str(legacy / "settings.toml")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as writer:
        assert writer.stdout.readline().strip() == "ready"
        child = resolve_process()
        stdout, stderr = child.communicate(timeout=30)
        assert child.returncode == 0, stderr
        assert Path(stdout.strip()) == legacy
        assert not target.exists()
        writer.communicate("", timeout=10)
    child = resolve_process()
    stdout, stderr = child.communicate(timeout=30)
    assert child.returncode == 0, stderr
    assert Path(stdout.strip()) == target


def test_concurrent_first_runs_share_one_complete_state(home):
    legacy, target = seed(home)
    children = [resolve_process() for _ in range(4)]
    for child in children:
        stdout, stderr = child.communicate(timeout=40)
        assert child.returncode == 0, stderr
        assert Path(stdout.strip()) == target
    assert legacy.samefile(target)
    assert (target / "runs.jsonl").read_bytes() == b'{"history":"preserve"}\n'


@pytest.mark.parametrize("failure", [None, "corrupt", "link"])
def test_cross_volume_copy_verifies_before_removing_source(home, monkeypatch, failure):
    legacy, target = seed(home)
    rename = Path.rename
    copytree = shutil.copytree

    def cross_volume(path, destination):
        if path == legacy and destination == target:
            raise OSError(errno.EXDEV, "different volumes")
        return rename(path, destination)

    def copy(source, destination, *args, **kwargs):
        result = copytree(source, destination, *args, **kwargs)
        if source == legacy and failure == "corrupt":
            (destination / "runs.jsonl").write_bytes(b"corrupted")
        return result

    monkeypatch.setattr(Path, "rename", cross_volume)
    monkeypatch.setattr(shutil, "copytree", copy)
    if failure == "link":
        monkeypatch.setattr(
            state_paths,
            "_link",
            lambda *_: (_ for _ in ()).throw(OSError("link denied")),
        )
    resolved = state_paths.default_state_dir()
    assert resolved == (legacy if failure else target)
    assert (resolved / "runs.jsonl").read_bytes() == b'{"history":"preserve"}\n'
    if failure:
        assert not target.exists()
    else:
        assert legacy.samefile(target)
        assert not legacy.with_name(".review-suite-moved").exists()


def test_explicit_override_keeps_independent_state(home, monkeypatch):
    legacy, target = seed(home)
    custom = home / "custom"
    monkeypatch.setenv("REVIEW_SUITE_STATE_DIR", str(custom))
    assert state_paths.default_state_dir() == custom
    assert legacy.is_dir() and not target.exists()


def test_interrupted_rename_recovers_old_link(home):
    legacy, target = seed(home)
    control = target.parent / ".review-suite-migration"
    control.mkdir(parents=True)
    (control / "pending.json").write_text(json.dumps({"legacy": str(legacy)}))
    legacy.rename(target)
    assert state_paths.default_state_dir() == target
    assert legacy.samefile(target)
    assert not (control / "pending.json").exists()


@pytest.mark.skipif(
    sys.platform != "win32", reason="Windows denies renaming open files"
)
def test_windows_open_file_leaves_state_until_handle_closes(home):
    legacy, target = seed(home)
    with (legacy / "runs.jsonl").open("rb"):
        child = resolve_process()
        stdout, stderr = child.communicate(timeout=30)
        assert child.returncode == 0, stderr
        assert Path(stdout.strip()) == legacy
        assert not target.exists()
    child = resolve_process()
    stdout, stderr = child.communicate(timeout=30)
    assert child.returncode == 0, stderr
    assert Path(stdout.strip()) == target


def test_deferred_process_keeps_using_old_state_for_its_lifetime(home):
    legacy, target = seed(home)
    code = (
        f"import sys; sys.path.insert(0, {str(SCRIPTS)!r}); "
        "from review_suite_core.state_paths import default_state_dir; "
        "print(default_state_dir(), flush=True); sys.stdin.read()"
    )
    with orchestrator_store_lock(state_dir=legacy, name="settings"):
        reader = subprocess.Popen(
            [sys.executable, "-c", code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert Path(reader.stdout.readline().strip()) == legacy
    try:
        child = resolve_process()
        stdout, stderr = child.communicate(timeout=30)
        assert child.returncode == 0, stderr
        assert Path(stdout.strip()) == legacy
        assert not target.exists()
    finally:
        reader.communicate("", timeout=10)
    child = resolve_process()
    stdout, stderr = child.communicate(timeout=30)
    assert child.returncode == 0, stderr
    assert Path(stdout.strip()) == target


def test_old_install_recreating_source_never_discards_moved_history(home, monkeypatch):
    legacy, target = seed(home)
    original_link = state_paths._link

    def competing_writer(old, new):
        old.mkdir()
        (old / "new-run.json").write_text("new review")
        raise FileExistsError("old install recreated its state")

    monkeypatch.setattr(state_paths, "_link", competing_writer)
    assert state_paths.default_state_dir() == target
    monkeypatch.setattr(state_paths, "_link", original_link)
    assert state_paths.default_state_dir() == target
    assert (target / "runs.jsonl").read_bytes() == b'{"history":"preserve"}\n'
    assert (legacy / "new-run.json").read_text() == "new review"


def test_existing_legacy_link_keeps_same_history(home):
    legacy, target = seed(home)
    external = home / "external-state"
    legacy.rename(external)
    state_paths._link(legacy, external)
    assert state_paths.default_state_dir() == target
    assert target.samefile(external)
    assert legacy.samefile(target)


def test_explicit_cli_state_does_not_migrate_default(home):
    legacy, target = seed(home)
    custom = home / "independent"
    custom.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "review_suite_arena.py"),
            "compact-rounds",
            "--state-dir",
            str(custom),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert legacy.is_dir() and not target.exists()


@pytest.mark.parametrize("edited_copy", [False, True])
def test_interrupted_cross_volume_copy_retries_original_or_preserves_edited_copy(
    home, monkeypatch, edited_copy
):
    legacy, target = seed(home)
    rename = Path.rename

    def interrupted(path, destination):
        if path == legacy and destination == target:
            raise OSError(errno.EXDEV, "different volumes")
        if path == legacy and destination == legacy.with_name(".review-suite-moved"):
            raise SystemExit("power loss after copy promotion")
        return rename(path, destination)

    monkeypatch.setattr(Path, "rename", interrupted)
    with pytest.raises(SystemExit):
        state_paths.default_state_dir()
    (legacy / "later.json").write_text("old install wrote after interruption")
    if edited_copy:
        (target / "new.json").write_text("new copy edited independently")
    monkeypatch.setattr(Path, "rename", rename)
    assert state_paths.default_state_dir() == target
    assert (legacy / "later.json").read_text() == "old install wrote after interruption"
    if edited_copy:
        assert (target / "new.json").read_text() == "new copy edited independently"
        assert not legacy.samefile(target)
    else:
        assert legacy.samefile(target)
        assert (
            target / "later.json"
        ).read_text() == "old install wrote after interruption"
    assert (target / "runs.jsonl").read_bytes() == b'{"history":"preserve"}\n'
