from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import review
import review_deslop
import review_followup
import review_github
import review_plan
import review_suite_arena
import review_suite_local

from review_suite_local import (
    build_local_review_request,
    build_base_instructions,
    build_deep_instructions,
    load_custom_instructions,
)


@pytest.fixture(autouse=True)
def default_effective_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        review_suite_local,
        "effective_base_ref",
        lambda review_cwd, base: {"base": base, "requested_base": base},
    )
    monkeypatch.setattr(
        review_suite_local,
        "merge_base",
        lambda review_cwd, base, right_ref="HEAD": "merge-base-sha",
    )
    monkeypatch.setattr(
        review_suite_local, "current_head", lambda review_cwd: "head-sha"
    )
    monkeypatch.setattr(
        review_suite_local, "ensure_clean_git_worktree", lambda *args, **kwargs: None
    )


def _subparser_help(parser: argparse.ArgumentParser, name: str) -> str:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices[name].format_help()
    raise AssertionError(f"subparser not found: {name}")


def test_load_custom_instructions_rejects_conflicting_sources() -> None:
    with pytest.raises(
        ValueError, match="use either --instructions or --instructions-file"
    ):
        load_custom_instructions(
            instructions="focus on correctness", instructions_file="instructions.txt"
        )


@pytest.mark.parametrize(
    ("instructions", "file_body"),
    [
        ("   ", None),
        (None, " \n\t "),
    ],
)
def test_load_custom_instructions_rejects_empty_content(
    tmp_path: Path,
    instructions: str | None,
    file_body: str | None,
) -> None:
    instructions_file = None
    if file_body is not None:
        path = tmp_path / "instructions.txt"
        path.write_text(file_body, encoding="utf-8")
        instructions_file = str(path)

    with pytest.raises(ValueError, match="custom instructions must not be empty"):
        load_custom_instructions(
            instructions=instructions, instructions_file=instructions_file
        )


def test_standard_review_contract_includes_current_codex_review_dimensions() -> None:
    prompt = build_base_instructions("commit `abc123`")

    assert (
        "Reviewer output is advisory risk input, not authoritative product direction"
        in prompt
    )
    assert "do not stop after the first issue" in prompt
    assert "line number when available" in prompt
    assert "A finding is only valid" in prompt
    assert "UX preference" in prompt
    assert "backwards-compat speculation" in prompt
    assert "oversized or hard-to-stage diffs" in prompt
    assert "external integration surface breaks" in prompt
    assert "missing regression or integration coverage" in prompt
    assert "unbounded agent-context injection" in prompt
    assert "Do not assume backwards compatibility" in prompt
    assert "Scope questions / suggestions (non-findings)" in prompt
    assert (
        "Do not recommend code changes that reverse explicit product intent" in prompt
    )
    assert (
        "focused review-relevant checks can be enough to launch the next review round"
        in prompt
    )
    assert "full-suite/CI remains a merge-readiness requirement" in prompt
    assert "pending, passed, failed, or explicitly waived with a reason" in prompt
    assert (
        "do not call a PR final or merge-ready while that status is unknown" in prompt
    )
    assert "No findings." in prompt
    assert "Review result: clean" in prompt
    assert "Review result: findings" in prompt


def test_build_local_review_request_keeps_terminal_contract_after_custom_block(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        raise AssertionError(args)

    monkeypatch.setattr(review_suite_local.subprocess, "run", fake_run)

    request = build_local_review_request(
        review_cwd=tmp_path,
        base="main",
        commit_values=["abc123"],
        instruction_builder=build_base_instructions,
        custom_instructions="Do not spend time on backwards compatibility concerns.",
    )

    assert calls == []
    assert request.review_scope["commit"] == "abc123"
    assert "base" not in request.review_scope
    assert request.prompt.index(
        "The review target is commit `abc123`."
    ) < request.prompt.index("Additional review instructions:")
    assert "Do not spend time on backwards compatibility concerns." in request.prompt
    assert request.prompt.index(
        "Additional review instructions:"
    ) < request.prompt.rindex("Review result:")
    assert request.prompt.rstrip().endswith(
        "`Review result: findings` if you reported one or more valid findings."
    )
    assert "=== BEGIN DIFF ===" not in request.prompt


def test_build_local_review_request_linear_commit_range_uses_native_base(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        review_suite_local,
        "validated_linear_review_range",
        lambda review_cwd, start, end, label: {
            "start": start,
            "end": end,
            "resolved_start": "abc123-resolved",
            "resolved_end": "def456-resolved",
            "head": "def456-resolved",
        },
    )

    request = build_local_review_request(
        review_cwd=tmp_path,
        base="main",
        commit_values=["abc123", "def456"],
        instruction_builder=build_base_instructions,
        custom_instructions=None,
    )

    assert request.review_scope["base"] == "abc123"
    assert request.review_scope["commit"] == "abc123"
    assert request.review_scope["commit_end"] == "def456"
    assert request.review_scope["reviewed_head"] == "def456-resolved"
    assert "commit range `abc123..def456`" in request.prompt
    assert "=== BEGIN DIFF ===" not in request.prompt


def test_build_local_review_request_rejects_non_linear_commit_range(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def reject_range(*args, **kwargs):
        raise ValueError(
            "native commit-range review requires the range start to be an ancestor of the range end"
        )

    monkeypatch.setattr(
        review_suite_local, "validated_linear_review_range", reject_range
    )

    with pytest.raises(ValueError, match="range start to be an ancestor"):
        build_local_review_request(
            review_cwd=tmp_path,
            base="main",
            commit_values=["abc123", "def456"],
            instruction_builder=build_base_instructions,
            custom_instructions=None,
        )


def test_build_local_review_request_pr_base_without_custom_instructions_includes_standard_prompt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if args[:2] == ["git", "diff"]:
            if "--quiet" in args:
                return subprocess.CompletedProcess(args, 1, stdout="", stderr="")
            return subprocess.CompletedProcess(
                args, 0, stdout="diff --git a/app.py b/app.py\n", stderr=""
            )
        raise AssertionError(args)

    monkeypatch.setattr(review_suite_local.subprocess, "run", fake_run)

    request = build_local_review_request(
        review_cwd=tmp_path,
        base="main",
        commit_values=None,
        instruction_builder=build_deep_instructions,
        custom_instructions=None,
    )

    assert calls == [["git", "diff", "--quiet", "merge-base-sha..HEAD"]]
    assert request.review_scope["base"] == "main"
    assert request.review_scope["merge_base"] == "merge-base-sha"
    assert request.review_scope["reviewed_head"] == "head-sha"
    assert (
        "Perform a deep review of this branch diff for correctness and regression risk."
        in request.prompt
    )
    assert "Review result: clean" in request.prompt
    assert "Additional review instructions:" not in request.prompt


def test_build_local_review_request_uses_effective_upstream_base(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(
        review_suite_local,
        "effective_base_ref",
        lambda review_cwd, base: {
            "base": "origin/main",
            "requested_base": "main",
            "base_upstream": "origin/main",
            "requested_base_head": "old-main",
            "effective_base_head": "new-main",
            "base_ref_stale": True,
        },
    )

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if args[:2] == ["git", "diff"]:
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="")
        raise AssertionError(args)

    monkeypatch.setattr(review_suite_local.subprocess, "run", fake_run)
    monkeypatch.setattr(
        review_suite_local,
        "merge_base",
        lambda review_cwd, base, right_ref="HEAD": "upstream-merge-base",
    )

    request = build_local_review_request(
        review_cwd=tmp_path,
        base="main",
        commit_values=None,
        instruction_builder=build_deep_instructions,
        custom_instructions=None,
    )

    assert request.review_scope["base"] == "origin/main"
    assert request.review_scope["requested_base"] == "main"
    assert request.review_scope["base_upstream"] == "origin/main"
    assert request.review_scope["base_ref_stale"] is True
    assert "requested `main`" in request.target_label


def test_build_local_review_request_pr_base_with_custom_instructions_stays_native(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if args[:2] == ["git", "diff"]:
            if "--quiet" in args:
                return subprocess.CompletedProcess(args, 1, stdout="", stderr="")
            return subprocess.CompletedProcess(
                args, 0, stdout="diff --git a/x b/x\n", stderr=""
            )
        raise AssertionError(args)

    monkeypatch.setattr(review_suite_local.subprocess, "run", fake_run)

    request = build_local_review_request(
        review_cwd=tmp_path,
        base="main",
        commit_values=None,
        instruction_builder=build_deep_instructions,
        custom_instructions="Stay focused on correctness only.",
    )

    assert calls[0] == ["git", "diff", "--quiet", "merge-base-sha..HEAD"]
    assert not any("--patch" in call for call in calls)
    assert request.review_scope["base"] == "main"
    assert request.review_scope["merge_base"] == "merge-base-sha"
    assert request.review_scope["reviewed_head"] == "head-sha"
    assert (
        "Perform a deep review of this branch diff for correctness and regression risk."
        in request.prompt
    )
    assert "Additional review instructions:" in request.prompt
    assert "base `main`" in request.prompt
    assert request.prompt.index(
        "Additional review instructions:"
    ) < request.prompt.rindex("Review result:")
    assert request.prompt.rstrip().endswith(
        "`Review result: findings` if you reported one or more valid findings."
    )
    assert "=== BEGIN DIFF ===" not in request.prompt


def test_build_local_review_request_blank_custom_instructions_still_requires_clean_worktree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clean_checks: list[Path] = []

    def fake_run(args, **kwargs):
        if args[:2] == ["git", "diff"]:
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="")
        raise AssertionError(args)

    monkeypatch.setattr(review_suite_local.subprocess, "run", fake_run)
    monkeypatch.setattr(
        review_suite_local,
        "ensure_clean_git_worktree",
        lambda review_cwd: clean_checks.append(review_cwd),
    )

    request = build_local_review_request(
        review_cwd=tmp_path,
        base="main",
        commit_values=None,
        instruction_builder=build_deep_instructions,
        custom_instructions="",
    )

    assert clean_checks == [tmp_path]
    assert (
        "Perform a deep review of this branch diff for correctness and regression risk."
        in request.prompt
    )
    assert "Additional review instructions:" not in request.prompt
    assert "=== BEGIN DIFF ===" not in request.prompt


def test_build_local_review_request_pr_base_with_empty_committed_diff_reports_dirty_recovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if args[:2] == ["git", "diff"]:
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        raise AssertionError(args)

    monkeypatch.setattr(review_suite_local.subprocess, "run", fake_run)
    monkeypatch.setattr(
        review_suite_local,
        "merge_base",
        lambda review_cwd, base, right_ref="HEAD": "head-sha",
    )
    monkeypatch.setattr(
        review_suite_local, "has_worktree_changes", lambda review_cwd: True
    )

    with pytest.raises(ValueError, match="no committed diff"):
        build_local_review_request(
            review_cwd=tmp_path,
            base="main",
            commit_values=None,
            instruction_builder=build_deep_instructions,
            custom_instructions="Stay focused on correctness only.",
        )

    assert calls[0] == ["git", "diff", "--quiet", "head-sha..HEAD"]


def test_build_local_review_request_native_pr_base_with_empty_committed_diff_reports_dirty_recovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run(args, **kwargs):
        if args[:2] == ["git", "merge-base"]:
            return subprocess.CompletedProcess(args, 0, stdout="head-sha\n", stderr="")
        if args[:3] == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(args, 0, stdout="head-sha\n", stderr="")
        if args[:2] == ["git", "diff"]:
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        raise AssertionError(args)

    monkeypatch.setattr(review_suite_local.subprocess, "run", fake_run)
    monkeypatch.setattr(
        review_suite_local, "has_worktree_changes", lambda review_cwd: True
    )

    with pytest.raises(ValueError, match="no committed diff"):
        build_local_review_request(
            review_cwd=tmp_path,
            base="main",
            commit_values=None,
            instruction_builder=build_deep_instructions,
            custom_instructions=None,
        )


def test_build_local_review_request_pr_base_without_custom_instructions_requires_clean_worktree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if args[:2] == ["git", "merge-base"]:
            return subprocess.CompletedProcess(
                args, 0, stdout="merge-base-sha\n", stderr=""
            )
        if args[:3] == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(args, 0, stdout="head-sha\n", stderr="")
        if args[:2] == ["git", "diff"]:
            if "--quiet" in args:
                return subprocess.CompletedProcess(args, 1, stdout="", stderr="")
            return subprocess.CompletedProcess(
                args, 0, stdout="diff --git a/app.py b/app.py\n", stderr=""
            )
        raise AssertionError(args)

    monkeypatch.setattr(review_suite_local.subprocess, "run", fake_run)
    monkeypatch.setattr(
        review_suite_local,
        "ensure_clean_git_worktree",
        lambda review_cwd: (_ for _ in ()).throw(
            ValueError("review-suite requires a clean worktree.")
        ),
    )

    with pytest.raises(ValueError, match="clean worktree"):
        build_local_review_request(
            review_cwd=tmp_path,
            base="main",
            commit_values=None,
            instruction_builder=build_deep_instructions,
            custom_instructions=None,
        )


def test_build_local_review_request_with_custom_instructions_does_not_build_patch_diff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if args[:2] == ["git", "merge-base"]:
            return subprocess.CompletedProcess(
                args, 0, stdout="merge-base-sha\n", stderr=""
            )
        if args[:3] == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(args, 0, stdout="head-sha\n", stderr="")
        if args[:2] == ["git", "diff"]:
            assert "--quiet" in args
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="")
        raise AssertionError(args)

    monkeypatch.setattr(review_suite_local.subprocess, "run", fake_run)
    request = build_local_review_request(
        review_cwd=tmp_path,
        base="main",
        commit_values=None,
        instruction_builder=build_deep_instructions,
        custom_instructions="Stay focused on correctness.",
    )

    assert request.review_scope["base"] == "main"
    assert not any("--patch" in call for call in calls)
    assert "=== BEGIN DIFF ===" not in request.prompt


def test_review_plan_deslop_and_github_do_not_expose_custom_instruction_flags() -> None:
    assert "--instructions" not in review_plan.build_parser().format_help()
    assert "--instructions-file" not in review_plan.build_parser().format_help()
    assert "--instructions" not in review_deslop.build_parser().format_help()
    assert "--instructions-file" not in review_deslop.build_parser().format_help()
    assert "--instructions" not in review_github.build_parser().format_help()
    assert "--instructions-file" not in review_github.build_parser().format_help()


def test_agent_wrapper_help_omits_low_roi_runtime_knobs() -> None:
    removed_flags = (
        "--model",
        "--reasoning-effort",
        "--progress-interval-seconds",
        "--timeout-seconds",
        "--poll-seconds",
        "--timeout-minutes",
        "--status-interval-seconds",
        "--re-request-after-seconds",
        "--max-request-attempts",
        "--settle-seconds",
    )
    help_texts = [
        review_plan.build_parser().format_help(),
        review_deslop.build_parser().format_help(),
        review_followup.build_parser().format_help(),
        _subparser_help(review_github.build_parser(), "run"),
        _subparser_help(review_suite_arena.build_parser(), "run"),
        _subparser_help(review_suite_arena.build_parser(), "run-round"),
        _subparser_help(review_suite_arena.build_parser(), "resume-round"),
        _subparser_help(review_suite_arena.build_parser(), "reroll-slot"),
    ]

    for help_text in help_texts:
        for flag in removed_flags:
            assert flag not in help_text


def test_primary_wrappers_hide_operator_state_knobs_from_help() -> None:
    hidden_flags = (
        "--task-id",
        "--seed",
        "--roster",
        "--rubric",
        "--state-dir",
        "--sqlite-path",
        "--caller-id",
        "--ignore-pending-grades",
        "--rating-pool-id",
        "--rank",
        "--basis",
        "--bot-login",
    )
    help_texts = [
        review.build_parser().format_help(),
        review_followup.build_parser().format_help(),
        _subparser_help(review_github.build_parser(), "run"),
    ]

    for help_text in help_texts:
        for flag in hidden_flags:
            assert flag not in help_text


def test_agent_wrapper_help_keeps_useful_targeting_controls_visible() -> None:
    assert "--input-file" in review_plan.build_parser().format_help()
    assert "--skip-git-repo-check" in review_plan.build_parser().format_help()
    assert "--focus" in review_deslop.build_parser().format_help()
    assert "--note-file" in review_followup.build_parser().format_help()
    assert "--pr-number" in _subparser_help(review_github.build_parser(), "run")
    assert "--base" in review.build_parser().format_help()
    assert "--status" in review.build_parser().format_help()


def test_review_plan_prompt_chooses_the_best_credible_solution_shape() -> None:
    prompt = review_plan.build_prompt("Keep the existing wrapper.")

    for requirement in (
        "root cause, canonical owner, and invariants",
        "materially distinct credible solution shapes",
        "only credible one",
        "contract correctness",
        "diff minimality only as a tie-breaker",
        "exactly one verdict: PROCEED, REVISE, or RETHINK",
        "RETHINK as scope authority",
    ):
        assert requirement in prompt
    assert prompt.endswith(
        "=== BEGIN PLAN ===\nKeep the existing wrapper.\n=== END PLAN ==="
    )


def test_local_review_wrappers_expose_short_wsl_flag() -> None:
    for help_text in (
        review_followup.build_parser().format_help(),
        review_plan.build_parser().format_help(),
        review_deslop.build_parser().format_help(),
    ):
        assert "--wsl" in help_text
        assert "--allow-unsafe-windows-codex-wsl-fallback" not in help_text


def test_review_plan_input_file_prefers_current_repo_when_file_is_not_in_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = tmp_path / "plan.md"
    repo = tmp_path / "repo"
    plan.write_text("Plan", encoding="utf-8")

    def fake_resolve_repo_root(value: object) -> Path:
        if value == plan.parent:
            raise ValueError("not a repo")
        if value is None:
            return repo
        raise AssertionError(value)

    monkeypatch.setattr(review_plan, "resolve_repo_root", fake_resolve_repo_root)
    args = review_plan.build_parser().parse_args(["--input-file", str(plan)])

    assert review_plan.resolve_review_root(args) == repo


def test_review_plan_cd_keeps_raw_path_for_repo_resolver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    captured: dict[str, object] = {}

    def fake_resolve_repo_root(value: object) -> Path:
        captured["value"] = value
        return repo

    monkeypatch.setattr(review_plan, "resolve_repo_root", fake_resolve_repo_root)
    args = review_plan.build_parser().parse_args(
        ["--cd", "/mnt/c/Code/sample-repo", "--input-text", "Plan"]
    )

    assert review_plan.resolve_review_root(args) == repo
    assert captured["value"] == "/mnt/c/Code/sample-repo"


def test_review_plan_forwards_skip_git_repo_check(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan = tmp_path / "plan.md"
    plan.write_text("Plan", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(review_plan, "resolve_repo_root", lambda value: tmp_path)
    monkeypatch.setattr(
        review_plan,
        "lens_model_config",
        lambda name: SimpleNamespace(
            model="gpt-5.5", reasoning_effort="medium", service_tier=None
        ),
    )

    def fake_run_codex(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "returncode": 0,
            "session_id": None,
            "elapsed_seconds": 0.0,
            "final_message": "No findings.",
        }

    monkeypatch.setattr(review_plan, "run_codex", fake_run_codex)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review_plan.py",
            "--skip-git-repo-check",
            "--input-file",
            str(plan),
        ],
    )

    assert review_plan.main() == 0
    assert captured["skip_git_repo_check"] is True
    assert "status: ok" in capsys.readouterr().out


def test_review_plan_skip_git_repo_check_uses_requested_directory_directly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    requested = tmp_path / "not-a-repo"
    args = review_plan.build_parser().parse_args(
        ["--skip-git-repo-check", "--cd", str(requested), "--input-text", "Plan"]
    )

    def fail_resolve_repo_root(value: object) -> Path:
        raise AssertionError(value)

    monkeypatch.setattr(review_plan, "resolve_repo_root", fail_resolve_repo_root)

    assert review_plan.resolve_review_root(args) == requested.resolve(strict=False)


def test_review_plan_skip_git_repo_check_uses_cd_path_normalization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    captured: dict[str, object] = {}
    args = review_plan.build_parser().parse_args(
        [
            "--skip-git-repo-check",
            "--cd",
            "/mnt/c/Code/sample-repo",
            "--input-text",
            "Plan",
        ]
    )

    def fake_resolve_cd_path(value: object) -> Path:
        captured["value"] = value
        return repo

    def fail_resolve_repo_root(value: object) -> Path:
        raise AssertionError(value)

    monkeypatch.setattr(review_plan, "resolve_cd_path", fake_resolve_cd_path)
    monkeypatch.setattr(review_plan, "resolve_repo_root", fail_resolve_repo_root)

    assert review_plan.resolve_review_root(args) == repo
    assert captured["value"] == "/mnt/c/Code/sample-repo"


def test_review_plan_skip_git_repo_check_uses_current_directory_without_cd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    args = review_plan.build_parser().parse_args(
        ["--skip-git-repo-check", "--input-text", "Plan"]
    )

    def fail_resolve_repo_root(value: object) -> Path:
        raise AssertionError(value)

    monkeypatch.setattr(review_plan, "resolve_repo_root", fail_resolve_repo_root)

    assert review_plan.resolve_review_root(args) == tmp_path.resolve(strict=False)
