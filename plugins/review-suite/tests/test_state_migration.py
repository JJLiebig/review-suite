from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

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


def resolve_process(writer_pid=None):
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"import sys; sys.path.insert(0, {str(SCRIPTS)!r}); "
            "from review_suite_core import state_paths as state; "
            "snapshot = state._processes; "
            f"state._processes = lambda: [row for row in snapshot() if row[0] == {writer_pid!r}] if {writer_pid!r} is not None else []; "
            "print(state.default_state_dir())",
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


@pytest.mark.parametrize("review_process", [False, True])
@pytest.mark.parametrize("folder", ["", "names-backup-123"])
def test_recorded_run_only_defers_for_live_review_suite_pid(
    home, monkeypatch, review_process, folder
):
    legacy, target = seed(home)
    rounds = legacy / folder / "rounds"
    rounds.mkdir()
    (rounds / "old.json").write_text(json.dumps({"runs": [{"pid": os.getpid()}]}))
    script = "review.py" if review_process else "unrelated.py"
    # The current process is excluded from direct writer detection, so the
    # recorded run must decide whether migration is safe.
    monkeypatch.setattr(
        state_paths, "_processes", lambda: [(os.getpid(), 0, f"python {script}")]
    )

    assert state_paths.default_state_dir() == (legacy if review_process else target)
    assert target.exists() is not review_process
    if not review_process:
        assert legacy.samefile(target)
        assert (target / folder / "rounds/old.json").is_file()


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


def test_stale_mkdir_lock_does_not_block_migration(home):
    legacy, target = seed(home)
    lock = legacy / ".locks/cost-reports.lock"
    lock.mkdir(parents=True)
    child = resolve_process()
    stdout, stderr = child.communicate(timeout=30)
    assert child.returncode == 0, stderr
    assert Path(stdout.strip()) == legacy
    assert not target.exists()

    stale = time.time() - 7 * 24 * 60 * 60
    os.utime(lock, (stale, stale))
    child = resolve_process()
    stdout, stderr = child.communicate(timeout=30)
    assert child.returncode == 0, stderr
    assert Path(stdout.strip()) == target
    assert legacy.samefile(target)
    assert (target / "runs.jsonl").read_bytes() == b'{"history":"preserve"}\n'
    assert (target / ".locks/cost-reports.lock").is_dir()


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
        child = resolve_process(writer.pid)
        stdout, stderr = child.communicate(timeout=30)
        assert child.returncode == 0, stderr
        assert Path(stdout.strip()) == legacy
        assert not target.exists()
        writer.communicate("", timeout=10)
    child = resolve_process(writer.pid)
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
    monkeypatch.setattr(state_paths, "_processes", lambda: [])
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


@pytest.mark.parametrize("deny_link", [False, True])
def test_interrupted_rename_recovers_old_link(home, monkeypatch, deny_link):
    legacy, target = seed(home)
    control = target.parent / ".review-suite-migration"
    control.mkdir(parents=True)
    (control / "pending.json").write_text(json.dumps({
        "legacy": str(legacy), "original_id": state_paths._identity(legacy),
    }))
    legacy.rename(target)
    if deny_link:
        monkeypatch.setattr(state_paths, "_link", lambda *_: (_ for _ in ()).throw(PermissionError("junction denied")))
        assert state_paths.default_state_dir() == legacy
        assert not target.exists()
        assert (legacy / "runs.jsonl").read_bytes() == b'{"history":"preserve"}\n'
        assert not (control / "pending.json").exists()
        return
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
    monkeypatch.setattr(state_paths, "_processes", lambda: [])
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
    assert state_paths.default_state_dir() == (legacy if edited_copy else target)
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


@pytest.mark.parametrize("failure", ["late_write", "late_writer", "retirement_locked"])
def test_cross_volume_cutover_failure_keeps_original_authoritative(
    home, monkeypatch, failure
):
    writers = []
    monkeypatch.setattr(state_paths, "_processes", lambda: writers)
    legacy, target = seed(home)
    rename = Path.rename

    def race(path, destination):
        if path == legacy and destination == target:
            raise OSError(errno.EXDEV, "different volumes")
        if path == legacy and destination == legacy.with_name(".review-suite-moved"):
            if failure == "retirement_locked":
                raise PermissionError("open Windows handle prevents retirement")
            (legacy / "late.json").write_text("writer raced copy verification")
            if failure == "late_writer":
                writers.append((999999999, 0, "python C:/repo/scripts/review.py --mode fast"))
        return rename(path, destination)

    monkeypatch.setattr(Path, "rename", race)
    assert state_paths.default_state_dir() == legacy
    assert (legacy / "runs.jsonl").read_bytes() == b'{"history":"preserve"}\n'
    if failure != "retirement_locked":
        assert (legacy / "late.json").read_text() == "writer raced copy verification"
    if failure == "late_writer":
        assert state_paths.default_state_dir() == legacy
        writers.clear()
        for handle in state_paths._leases.values():
            handle.close()
        state_paths._leases.clear()
        monkeypatch.setattr(Path, "rename", rename)
        assert state_paths.default_state_dir() == target
        assert legacy.samefile(target)
        assert (target / "late.json").read_text() == "writer raced copy verification"
    else:
        assert not target.exists()


@pytest.mark.parametrize("interruption", [SystemExit, OSError])
def test_interrupted_retirement_restores_original_before_recovery(
    home, monkeypatch, interruption
):
    legacy, target = seed(home)
    rename = Path.rename
    monkeypatch.setattr(state_paths, "_processes", lambda: [])
    original_in_use = state_paths._in_use

    def cross_volume(path, destination):
        if path == legacy and destination == target:
            raise OSError(errno.EXDEV, "different volumes")
        return rename(path, destination)

    def interrupted_check(path, control, **kwargs):
        retired = legacy.with_name(".review-suite-moved")
        if retired.exists():
            (retired / "late.json").write_text("write not in promoted copy")
            raise interruption("interrupted before verification")
        return original_in_use(path, control, **kwargs)

    monkeypatch.setattr(Path, "rename", cross_volume)
    monkeypatch.setattr(state_paths, "_in_use", interrupted_check)
    if interruption is SystemExit:
        with pytest.raises(SystemExit):
            state_paths.default_state_dir()
    else:
        assert state_paths.default_state_dir() == legacy
        assert (legacy / "late.json").read_text() == "write not in promoted copy"
        return
    monkeypatch.setattr(state_paths, "_in_use", original_in_use)
    assert state_paths.default_state_dir() == target
    assert legacy.samefile(target)
    assert (target / "late.json").read_text() == "write not in promoted copy"


def test_independent_linked_legacy_state_is_reported_once(home, capsys):
    legacy, target = seed(home)
    external = home / "external-state"
    legacy.rename(external)
    state_paths._link(legacy, external)
    target.mkdir(parents=True)
    assert state_paths.default_state_dir() == target
    assert str(legacy) in capsys.readouterr().err
    assert state_paths.default_state_dir() == target
    assert capsys.readouterr().err == ""
    assert (external / "runs.jsonl").read_bytes() == b'{"history":"preserve"}\n'


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process environment inspection")
@pytest.mark.parametrize("override", ["absolute", "relative", "tilde", "cli_equals", "cli_spaced", "aliased"])
def test_environment_selected_writer_keeps_promoted_copy_alive(home, monkeypatch, override):
    if override == "cli_spaced":
        home = home / "home with spaces"
        for key in ("HOME", "USERPROFILE"):
            monkeypatch.setenv(key, str(home))
        monkeypatch.setenv("CODEX_HOME", str(home / ".codex"))
    legacy, target = seed(home)
    rename = Path.rename
    monkeypatch.setattr(state_paths, "_processes", lambda: [])

    def interrupted(path, destination):
        if path == legacy and destination == target:
            raise OSError(errno.EXDEV, "different volumes")
        if path == legacy and destination == legacy.with_name(".review-suite-moved"):
            raise SystemExit("power loss after copy promotion")
        return rename(path, destination)

    monkeypatch.setattr(Path, "rename", interrupted)
    with pytest.raises(SystemExit):
        state_paths.default_state_dir()
    monkeypatch.setattr(Path, "rename", rename)
    script = home / "review.py"
    script.write_text("import time; print('ready', flush=True); time.sleep(30)")
    alias = home / "alias"
    alias.symlink_to(target, target_is_directory=True)
    selected = {"absolute": str(target), "relative": ".agents/state/../state/review-suite", "tilde": "~/.agents/state/review-suite", "cli_equals": "", "cli_spaced": "", "aliased": str(alias)}[override]
    arguments = [f"--state-dir={alias}"] if override == "cli_equals" else ["--state-dir", str(target)] if override == "cli_spaced" else []
    writer = subprocess.Popen(
        [sys.executable, str(script), *arguments], cwd=home,
        env={**os.environ, "REVIEW_SUITE_STATE_DIR": selected},
        stdout=subprocess.PIPE, text=True,
    )
    monkeypatch.setattr(state_paths, "_processes", lambda: [(writer.pid, 0, " ".join([sys.executable, str(script), *arguments]))])
    try:
        assert writer.stdout.readline().strip() == "ready"
        assert state_paths.default_state_dir() == legacy
        assert not legacy.samefile(target)
        assert (target / "runs.jsonl").read_bytes() == b'{"history":"preserve"}\n'
    finally:
        writer.terminate()
        writer.wait(timeout=10)


@pytest.mark.parametrize("failure", [OSError, SystemExit])
def test_partial_retired_cleanup_does_not_block_next_run(home, monkeypatch, failure):
    legacy, target = seed(home)
    rename = Path.rename
    remove = shutil.rmtree
    retired = legacy.with_name(".review-suite-moved")
    monkeypatch.setattr(state_paths, "_processes", lambda: [])

    def cross_volume(path, destination):
        if path == legacy and destination == target:
            raise OSError(errno.EXDEV, "different volumes")
        return rename(path, destination)

    def partial_cleanup(path, *args, **kwargs):
        if path == retired:
            (path / "runs.jsonl").unlink()
            raise failure("cleanup interrupted")
        return remove(path, *args, **kwargs)

    monkeypatch.setattr(Path, "rename", cross_volume)
    monkeypatch.setattr(shutil, "rmtree", partial_cleanup)
    if failure is SystemExit:
        with pytest.raises(SystemExit):
            state_paths.default_state_dir()
    else:
        assert state_paths.default_state_dir() == target
    monkeypatch.setattr(shutil, "rmtree", remove)
    assert state_paths.default_state_dir() == target
    assert legacy.samefile(target)
    assert not retired.exists()
    assert (target / "runs.jsonl").read_bytes() == b'{"history":"preserve"}\n'


def test_cross_volume_source_recreation_reports_preserved_original(
    home, monkeypatch, capsys
):
    legacy, target = seed(home)
    rename = Path.rename
    link = state_paths._link
    retired = legacy.with_name(".review-suite-moved")
    monkeypatch.setattr(state_paths, "_processes", lambda: [])

    def cross_volume(path, destination):
        if path == legacy and destination == target:
            raise OSError(errno.EXDEV, "different volumes")
        return rename(path, destination)

    def recreation(old, new):
        old.mkdir()
        (old / "new.json").write_text("independent new state")
        (retired / "late.json").write_text("original writer finished late")
        return link(old, new)

    monkeypatch.setattr(Path, "rename", cross_volume)
    monkeypatch.setattr(state_paths, "_link", recreation)
    assert state_paths.default_state_dir() == target
    assert (retired / "late.json").read_text() == "original writer finished late"
    assert str(retired) in capsys.readouterr().err
    assert (legacy / "new.json").read_text() == "independent new state"
    assert (target / "runs.jsonl").read_bytes() == b'{"history":"preserve"}\n'


def test_recovery_preserves_retired_files_changed_after_partial_cleanup(
    home, monkeypatch, capsys
):
    legacy, target = seed(home)
    rename = Path.rename
    remove = shutil.rmtree
    retired = legacy.with_name(".review-suite-moved")
    monkeypatch.setattr(state_paths, "_processes", lambda: [])

    def cross_volume(path, destination):
        if path == legacy and destination == target:
            raise OSError(errno.EXDEV, "different volumes")
        return rename(path, destination)

    def partial_cleanup(path, *args, **kwargs):
        if path == retired:
            (path / "names-backup-123/original.json").unlink()
            raise OSError("locked remainder")
        return remove(path, *args, **kwargs)

    monkeypatch.setattr(Path, "rename", cross_volume)
    monkeypatch.setattr(shutil, "rmtree", partial_cleanup)
    assert state_paths.default_state_dir() == target
    (retired / "runs.jsonl").write_text("changed remainder")
    monkeypatch.setattr(shutil, "rmtree", remove)
    assert state_paths.default_state_dir() == target
    assert (retired / "runs.jsonl").read_text() == "changed remainder"
    assert str(retired) in capsys.readouterr().err
    assert (target / "runs.jsonl").read_bytes() == b'{"history":"preserve"}\n'


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows held files prevent the directory rename itself",
)
def test_cutover_checks_locks_in_the_retired_tree(home, monkeypatch):
    from contextlib import ExitStack

    legacy, target = seed(home)
    rename = Path.rename
    monkeypatch.setattr(state_paths, "_processes", lambda: [])
    # Existing lock file means acquiring the lock changes no tree contents.
    with orchestrator_store_lock(state_dir=legacy, name="settings"):
        pass
    with ExitStack() as scope:

        def cross_volume(path, destination):
            if path == legacy and destination == target:
                raise OSError(errno.EXDEV, "different volumes")
            if path == legacy and destination == legacy.with_name(
                ".review-suite-moved"
            ):
                scope.enter_context(
                    orchestrator_store_lock(state_dir=legacy, name="settings")
                )
            return rename(path, destination)

        monkeypatch.setattr(Path, "rename", cross_volume)
        assert state_paths.default_state_dir() == legacy
        assert not legacy.samefile(target)
    assert (legacy / "runs.jsonl").read_bytes() == b'{"history":"preserve"}\n'
    for handle in state_paths._leases.values():
        handle.close()
    state_paths._leases.clear()
    monkeypatch.setattr(Path, "rename", rename)
    assert state_paths.default_state_dir() == target
    assert legacy.samefile(target)


def test_rollback_preserves_independently_changed_promoted_copy(
    home, monkeypatch, capsys
):
    legacy, target = seed(home)
    rename = Path.rename
    monkeypatch.setattr(state_paths, "_processes", lambda: [])

    def cross_volume(path, destination):
        if path == legacy and destination == target:
            raise OSError(errno.EXDEV, "different volumes")
        if path == legacy and destination == legacy.with_name(".review-suite-moved"):
            (target / "new-run.json").write_text("explicit writer used new state")
            (legacy / "late.json").write_text("old writer changed original")
        return rename(path, destination)

    monkeypatch.setattr(Path, "rename", cross_volume)
    assert state_paths.default_state_dir() == legacy
    assert (target / "new-run.json").read_text() == "explicit writer used new state"
    assert (legacy / "late.json").read_text() == "old writer changed original"
    assert str(target) in capsys.readouterr().err
    # A subsequent launch must still use the authoritative original even after
    # the writers and the first deferred caller have exited.
    for handle in state_paths._leases.values():
        handle.close()
    state_paths._leases.clear()
    assert state_paths.default_state_dir() == legacy
    assert not legacy.samefile(target)
    assert capsys.readouterr().err == ""


def test_linked_codex_home_reports_independent_home_default_immediately(
    home, monkeypatch, capsys
):
    legacy, target = seed(home)
    codex = home / "custom-codex"
    linked = codex / "state/review-suite"
    linked.parent.mkdir(parents=True)
    store = home / "external"
    store.mkdir()
    state_paths._link(linked, store)
    monkeypatch.setenv("CODEX_HOME", str(codex))
    assert state_paths.default_state_dir() == target
    assert target.samefile(store)
    assert str(legacy) in capsys.readouterr().err
    assert (legacy / "runs.jsonl").read_bytes() == b'{"history":"preserve"}\n'
