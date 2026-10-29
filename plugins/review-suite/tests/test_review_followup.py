from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import review_followup


@pytest.fixture(autouse=True)
def _isolate_default_state_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(review_followup, "default_state_dir", lambda: tmp_path / "state")


def test_load_followup_note_rejects_conflicting_sources() -> None:
    with pytest.raises(ValueError, match="use either --note or --note-file"):
        review_followup.load_followup_note(note="x", note_file="note.txt")


def test_load_followup_note_requires_non_empty_content(tmp_path: Path) -> None:
    note_path = tmp_path / "note.txt"
    note_path.write_text(" \n\t ", encoding="utf-8")

    with pytest.raises(ValueError, match="follow-up note must not be empty"):
        review_followup.load_followup_note(note=None, note_file=str(note_path))


def test_load_followup_note_resolves_relative_file_against_repo_root(
    tmp_path: Path,
) -> None:
    review_root = tmp_path / "repo"
    note_path = review_root / ".review-suite" / "fix-note.md"
    note_path.parent.mkdir(parents=True)
    note_path.write_text("invariant: keep owner state coherent", encoding="utf-8")

    text = review_followup.load_followup_note(
        note=None,
        note_file=".review-suite/fix-note.md",
        review_root=review_root,
    )

    assert text == "invariant: keep owner state coherent"


def test_main_uses_recorded_anchor_and_records_new_followup_anchor(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(review_followup, "resolve_repo_root", lambda cd: tmp_path)
    monkeypatch.setattr(
        review_followup, "ensure_clean_git_worktree", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        review_followup,
        "effective_base_ref",
        lambda review_root, base: {"base": base, "requested_base": base},
    )
    monkeypatch.setattr(
        review_followup, "has_committed_diff", lambda review_root, start, end: True
    )
    monkeypatch.setattr(
        review_followup,
        "inspect_workflow_status",
        lambda **kwargs: {
            "status": "ok",
            "recommendation": "review-followup",
            "last_reviewed_head": "abc123",
        },
    )
    monkeypatch.setattr(review_followup, "current_head", lambda review_root: "def456")
    monkeypatch.setattr(
        review_followup, "merge_base", lambda review_root, base: "base123"
    )
    monkeypatch.setattr(
        review_followup,
        "validated_linear_review_range",
        lambda review_root, start, end, label: {
            "start": start,
            "end": end,
            "resolved_start": start,
            "resolved_end": end,
            "head": end,
        },
    )

    def fake_run_codex_review(**kwargs):
        captured["prompt"] = kwargs["prompt"]
        captured["review_base"] = kwargs["base"]
        return {
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "final_message": "No findings.",
            "session_id": "sess-1",
            "elapsed_seconds": 1.2,
            "timed_out": False,
        }

    monkeypatch.setattr(review_followup, "run_codex_review", fake_run_codex_review)
    monkeypatch.setattr(
        review_followup,
        "record_review_anchor",
        lambda **kwargs: captured.setdefault("anchor", kwargs) or {},
    )

    def fake_emit_result(**kwargs):
        captured["result"] = kwargs
        return 0

    monkeypatch.setattr(review_followup, "emit_result", fake_emit_result)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review_followup.py",
            "--base",
            "main",
            "--note",
            "invariant: branch fix must update the owning state machine",
        ],
    )

    exit_code = review_followup.main()

    assert exit_code == 0
    assert "Fixer root-cause note:" in str(captured["prompt"])
    assert "abc123" in str(captured["prompt"])
    assert "do not stop after the first issue" in str(captured["prompt"])
    assert "unbounded agent-context injection" in str(captured["prompt"])
    assert "=== BEGIN DIFF ===" not in str(captured["prompt"])
    assert captured["review_base"] == "abc123"
    assert captured["anchor"]["state_dir"] == tmp_path / "state"
    assert (tmp_path / "state" / "settings.toml").is_file()
    assert captured["anchor"]["lane"] == "review-followup"
    assert captured["anchor"]["reviewed_head"] == "def456"
    assert captured["anchor"]["review_scope"]["commit"] == "abc123"
    assert captured["anchor"]["review_scope"]["base"] == "abc123"
    assert captured["anchor"]["review_scope"]["branch_base"] == "main"
    assert captured["anchor"]["review_scope"]["merge_base"] == "base123"


def test_main_records_effective_branch_base_for_followup_anchor(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}
    status_bases: list[str] = []

    monkeypatch.setattr(review_followup, "resolve_repo_root", lambda cd: tmp_path)
    monkeypatch.setattr(
        review_followup, "ensure_clean_git_worktree", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        review_followup,
        "effective_base_ref",
        lambda review_root, base: {
            "base": "origin/main",
            "requested_base": "main",
            "base_upstream": "origin/main",
            "requested_base_head": "old-main",
            "effective_base_head": "new-main",
            "base_ref_stale": True,
        },
    )

    def fake_inspect_workflow_status(**kwargs):
        status_bases.append(str(kwargs["base"]))
        return {
            "status": "ok",
            "recommendation": "review-followup",
            "last_reviewed_head": "abc123",
        }

    monkeypatch.setattr(
        review_followup, "inspect_workflow_status", fake_inspect_workflow_status
    )
    monkeypatch.setattr(review_followup, "current_head", lambda review_root: "def456")
    monkeypatch.setattr(
        review_followup,
        "merge_base",
        lambda review_root, base: f"merge-base-for-{base}",
    )
    monkeypatch.setattr(
        review_followup, "has_committed_diff", lambda review_root, start, end: True
    )
    monkeypatch.setattr(
        review_followup,
        "validated_linear_review_range",
        lambda review_root, start, end, label: {
            "start": start,
            "end": end,
            "resolved_start": start,
            "resolved_end": end,
            "head": end,
        },
    )
    monkeypatch.setattr(
        review_followup,
        "run_codex_review",
        lambda **kwargs: {
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "final_message": "No findings.",
            "session_id": "sess-1",
            "elapsed_seconds": 1.2,
            "timed_out": False,
        },
    )
    monkeypatch.setattr(
        review_followup,
        "record_review_anchor",
        lambda **kwargs: captured.setdefault("anchor", kwargs) or {},
    )
    monkeypatch.setattr(review_followup, "emit_result", lambda **kwargs: 0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review_followup.py",
            "--base",
            "main",
            "--note",
            "invariant: branch fix must update the owning state machine",
        ],
    )

    assert review_followup.main() == 0
    assert status_bases == ["origin/main"]
    assert captured["anchor"]["base"] == "origin/main"
    assert captured["anchor"]["review_scope"]["base"] == "abc123"
    assert captured["anchor"]["review_scope"]["branch_base"] == "origin/main"
    assert captured["anchor"]["review_scope"]["requested_base"] == "main"
    assert (
        captured["anchor"]["review_scope"]["merge_base"] == "merge-base-for-origin/main"
    )
    assert captured["anchor"]["review_scope"]["base_upstream"] == "origin/main"
    assert captured["anchor"]["review_scope"]["base_ref_stale"] is True


def test_main_rejects_empty_followup_interdiff(monkeypatch, tmp_path: Path) -> None:
    errors: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(review_followup, "resolve_repo_root", lambda cd: tmp_path)
    monkeypatch.setattr(
        review_followup,
        "effective_base_ref",
        lambda review_root, base: {"base": base, "requested_base": base},
    )
    monkeypatch.setattr(
        review_followup,
        "inspect_workflow_status",
        lambda **kwargs: {
            "status": "ok",
            "recommendation": "review-followup",
            "last_reviewed_head": "abc123",
        },
    )
    monkeypatch.setattr(review_followup, "current_head", lambda review_root: "def456")
    monkeypatch.setattr(
        review_followup,
        "validated_linear_review_range",
        lambda review_root, start, end, label: {
            "start": start,
            "end": end,
            "resolved_start": start,
            "resolved_end": end,
            "head": end,
        },
    )
    monkeypatch.setattr(
        review_followup, "has_committed_diff", lambda review_root, start, end: False
    )
    monkeypatch.setattr(
        review_followup,
        "run_codex_review",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("review should not run")),
    )
    monkeypatch.setattr(
        review_followup,
        "emit_error",
        lambda message, **kwargs: errors.append((message, dict(kwargs))) or 2,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review_followup.py",
            "--base",
            "main",
            "--note",
            "invariant: branch fix must update the owning state machine",
        ],
    )

    assert review_followup.main() == 2
    assert "no committed diff" in errors[0][0]
    assert errors[0][1]["status"] == "usage_error"


def test_main_rejects_allow_dirty_flag(monkeypatch) -> None:
    errors: list[tuple[str, dict[str, object]]] = []

    def fake_emit_error(message: str, **kwargs: object) -> int:
        errors.append((message, dict(kwargs)))
        return 2

    monkeypatch.setattr(review_followup, "emit_error", fake_emit_error)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review_followup.py",
            "--base",
            "main",
            "--allow-dirty",
            "--note",
            "invariant: only committed follow-up diffs are reviewable",
        ],
    )

    exit_code = review_followup.main()

    assert exit_code == 2
    assert errors
    assert "unrecognized arguments: --allow-dirty" in errors[0][0]


def test_resolve_since_head_requires_anchor_when_not_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        review_followup, "inspect_workflow_status", lambda **kwargs: {"status": "ok"}
    )
    with pytest.raises(
        ValueError, match="requires --since or an existing recorded review anchor"
    ):
        review_followup.resolve_since_head(
            explicit_since=None,
            state_dir=tmp_path / "state",
            review_cwd=tmp_path,
            base="main",
            force=False,
        )


def test_resolve_since_head_rejects_large_recorded_delta_without_force(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        review_followup,
        "inspect_workflow_status",
        lambda **kwargs: {
            "status": "ok",
            "recommendation": "coherence-review",
            "last_reviewed_head": "abc123",
            "note": "The post-review delta is large enough that coherence/reset review is safer than another narrow follow-up.",
        },
    )

    with pytest.raises(ValueError, match="coherence/reset"):
        review_followup.resolve_since_head(
            explicit_since=None,
            state_dir=tmp_path / "state",
            review_cwd=tmp_path,
            base="main",
            force=False,
        )


def test_resolve_since_head_allows_large_explicit_delta_with_force(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        review_followup,
        "diff_stats",
        lambda review_cwd, start_ref, end_ref: {
            "commits_since_anchor": 8,
            "files_changed": 9,
            "lines_changed": 900,
        },
    )

    resolved = review_followup.resolve_since_head(
        explicit_since="abc123",
        state_dir=tmp_path / "state",
        review_cwd=tmp_path,
        base="main",
        force=True,
    )

    assert resolved == "abc123"


def test_resolve_since_head_allows_valid_explicit_anchor_even_when_stored_state_is_stale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        review_followup,
        "inspect_workflow_status",
        lambda **kwargs: {
            "status": "ok",
            "recommendation": "full-review",
            "reason": "review_anchor_not_ancestor",
            "note": "The latest recorded review anchor is stale.",
        },
    )
    monkeypatch.setattr(
        review_followup,
        "resolve_ref",
        lambda review_cwd, ref: "abc123-resolved",
    )
    monkeypatch.setattr(review_followup, "is_ancestor", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        review_followup,
        "diff_stats",
        lambda review_cwd, start_ref, end_ref: {
            "commits_since_anchor": 1,
            "files_changed": 1,
            "lines_changed": 20,
        },
    )

    resolved = review_followup.resolve_since_head(
        explicit_since="abc123",
        state_dir=tmp_path / "state",
        review_cwd=tmp_path,
        base="main",
        force=False,
    )

    assert resolved == "abc123-resolved"


def test_resolve_since_head_rejects_explicit_anchor_when_branch_review_pressure_is_high(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        review_followup,
        "inspect_workflow_status",
        lambda **kwargs: {
            "status": "ok",
            "recommendation": "coherence-review",
            "reason": "branch_review_pressure_exceeded",
            "note": "The branch already carries too much review churn for another narrow follow-up.",
        },
    )

    with pytest.raises(ValueError, match="too much review churn"):
        review_followup.resolve_since_head(
            explicit_since="abc123",
            state_dir=tmp_path / "state",
            review_cwd=tmp_path,
            base="main",
            force=False,
        )


def test_resolve_since_head_rejects_explicit_anchor_that_is_not_ancestor_without_force(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        review_followup, "inspect_workflow_status", lambda **kwargs: {"status": "ok"}
    )
    monkeypatch.setattr(
        review_followup, "resolve_ref", lambda review_cwd, ref: "abc123-resolved"
    )
    monkeypatch.setattr(review_followup, "is_ancestor", lambda *args, **kwargs: False)

    with pytest.raises(ValueError, match="ancestor of HEAD"):
        review_followup.resolve_since_head(
            explicit_since="abc123",
            state_dir=tmp_path / "state",
            review_cwd=tmp_path,
            base="main",
            force=False,
        )


def test_resolve_since_head_rejects_large_explicit_delta_without_force(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        review_followup, "inspect_workflow_status", lambda **kwargs: {"status": "ok"}
    )
    monkeypatch.setattr(
        review_followup, "resolve_ref", lambda review_cwd, ref: "abc123-resolved"
    )
    monkeypatch.setattr(review_followup, "is_ancestor", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        review_followup,
        "diff_stats",
        lambda review_cwd, start_ref, end_ref: {
            "commits_since_anchor": 8,
            "files_changed": 9,
            "lines_changed": 900,
        },
    )

    with pytest.raises(ValueError, match="coherence/reset"):
        review_followup.resolve_since_head(
            explicit_since="abc123",
            state_dir=tmp_path / "state",
            review_cwd=tmp_path,
            base="main",
            force=False,
        )
