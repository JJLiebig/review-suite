from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import review_suite_core.workflow_state as workflow_state_module
import review
import review_suite_core.review_branch_status as review_state

from review_suite_core.workflow_state import (
    branch_token,
    effective_base_ref,
    inspect_workflow_status,
    latest_base_review_context_anchor,
    load_workflow_state,
    record_review_anchor,
    workflow_state_path,
)


_GIT_ENV = os.environ | {
    "GIT_AUTHOR_EMAIL": "codex@example.invalid",
    "GIT_AUTHOR_NAME": "Codex",
    "GIT_COMMITTER_EMAIL": "codex@example.invalid",
    "GIT_COMMITTER_NAME": "Codex",
    "GIT_TERMINAL_PROMPT": "0",
}


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=_GIT_ENV,
    )
    if proc.returncode != 0:
        raise AssertionError(
            proc.stderr or proc.stdout or f"git {' '.join(args)} failed"
        )
    return proc.stdout.strip()


def _init_repo(repo: Path) -> None:
    repo.mkdir()
    _git(repo, "init", "-b", "main")


def _commit_file(repo: Path, relative_path: str, content: str, message: str) -> str:
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _git(repo, "add", relative_path)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _write_orchestrator_cycle(
    state_dir: Path, name: str, payload: dict[str, object]
) -> None:
    path = state_dir / "orchestrator" / "cycles" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _use_default_state_dir(monkeypatch, state_dir: Path) -> None:
    monkeypatch.setattr(review_state, "default_state_dir", lambda: state_dir)


def test_review_state_status_rejects_state_dir(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["review.py", "--status", "--state-dir", str(tmp_path / "state")]
    )

    assert review.main() == 2
    rendered = capsys.readouterr().out
    assert "status: usage_error" in rendered
    assert "unrecognized arguments: --state-dir" in rendered


@pytest.mark.parametrize("default_branch", ["main", "master"])
def test_effective_base_ref_detects_remote_default_branch(
    tmp_path: Path, default_branch: str
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    head = _commit_file(repo, "app.txt", "content\n", "initial")
    _git(repo, "remote", "add", "origin", str(tmp_path / "origin.git"))
    _git(repo, "update-ref", f"refs/remotes/origin/{default_branch}", head)
    _git(
        repo,
        "symbolic-ref",
        "refs/remotes/origin/HEAD",
        f"refs/remotes/origin/{default_branch}",
    )

    payload = effective_base_ref(repo, None)

    assert payload == {
        "base": f"origin/{default_branch}",
        "requested_base": f"origin/{default_branch}",
    }


def test_effective_base_ref_keeps_explicit_local_base(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    head = _commit_file(repo, "app.txt", "content\n", "initial")
    _git(repo, "remote", "add", "origin", str(tmp_path / "origin.git"))
    _git(repo, "update-ref", "refs/remotes/origin/main", head)
    _git(repo, "branch", "origin/renamed", head)
    _git(
        repo,
        "symbolic-ref",
        "refs/remotes/origin/HEAD",
        "refs/remotes/origin/main",
    )

    payload = effective_base_ref(repo, "main")

    assert payload == {"base": "main", "requested_base": "main"}


def test_effective_base_ref_ignores_dangling_remote_head(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    head = _commit_file(repo, "app.txt", "content\n", "initial")
    _git(repo, "remote", "add", "origin", str(tmp_path / "origin.git"))
    _git(repo, "update-ref", "refs/remotes/origin/main", head)
    _git(
        repo,
        "symbolic-ref",
        "refs/remotes/origin/HEAD",
        "refs/remotes/origin/renamed",
    )

    payload = effective_base_ref(repo, None)

    assert payload == {"base": "origin/main", "requested_base": "origin/main"}


def test_effective_base_ref_falls_back_to_local_main(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "content\n", "initial")

    payload = effective_base_ref(repo, None)

    assert payload == {"base": "main", "requested_base": "main"}


def test_effective_base_ref_does_not_treat_tag_as_local_branch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "content\n", "initial")
    _git(repo, "branch", "-m", "trunk")
    _git(repo, "tag", "main")

    with pytest.raises(ValueError, match="pass --base <ref>"):
        effective_base_ref(repo, None)


def test_inspect_workflow_status_without_anchor_recommends_full_review(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "one\n", "initial")

    payload = inspect_workflow_status(
        state_dir=tmp_path / "state",
        review_cwd=repo,
        base="main",
    )

    assert payload["recommendation"] == "full-review"
    assert payload["reason"] == "no_review_anchor"
    assert "state_file" not in payload


def test_review_state_status_json_emits_sanitized_running_review(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    repo.mkdir()
    _write_orchestrator_cycle(
        state_dir,
        "running.json",
        {
            "public_id": "rvw_running",
            "stage": "decision-pending",
            "mode": {"requested": "normal", "effective": "normal"},
            "identity": {
                "cwd": str(review_state.normalize_review_cwd_value(repo)),
                "base": "main",
                "branch": "feature/json",
            },
            "pending_action": {
                "kind": "decision",
                "lane": "review_t1",
                "round_id": "round-private",
                "step": "broad-discovery",
                "step_index": 0,
            },
            "review_plan": {"steps": [{"name": "broad-discovery", "kind": "review"}]},
        },
    )
    monkeypatch.setattr(review_state, "resolve_repo_root", lambda cd: repo)
    monkeypatch.setattr(
        review_state,
        "inspect_workflow_status",
        lambda **kwargs: {
            "status": "ok",
            "base": "main",
            "branch": "feature/json",
            "head": "head-1",
        },
    )
    _use_default_state_dir(monkeypatch, state_dir)
    monkeypatch.setattr(
        sys, "argv", ["review.py", "--status", "--json", "--base", "main"]
    )

    assert review.main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "base": "main",
        "branch": "feature/json",
        "done": False,
        "head": "head-1",
        "mode": "normal",
        "next_action": "continue",
        "progress": "review 1/1 broad-discovery",
        "review": "rvw_running",
        "review_ladder": "pending",
    }


def test_review_state_status_json_emits_caller_convergence_decision(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    repo.mkdir()
    _write_orchestrator_cycle(
        state_dir,
        "decision.json",
        {
            "public_id": "rvw_decision",
            "stage": "decision-required",
            "mode": {"requested": "fast", "effective": "fast"},
            "identity": {
                "cwd": str(review_state.normalize_review_cwd_value(repo)),
                "base": "main",
                "branch": "feature/json",
            },
            "convergence": {
                "status": "DECISION_REQUIRED",
                "reason": "budget_exhausted",
                "accepted_findings_limit": 3,
                "accepted_findings_heads": [
                    {"head": f"head-{index}", "material_id": f"tree-{index}"}
                    for index in range(3)
                ],
                "continue_used": False,
            },
        },
    )
    monkeypatch.setattr(review_state, "resolve_repo_root", lambda cd: repo)
    monkeypatch.setattr(
        review_state,
        "inspect_workflow_status",
        lambda **kwargs: {
            "status": "ok",
            "base": "main",
            "branch": "feature/json",
            "head": "head-3",
        },
    )
    _use_default_state_dir(monkeypatch, state_dir)
    monkeypatch.setattr(
        sys, "argv", ["review.py", "--status", "--json", "--base", "main"]
    )

    assert review.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "decision_required"
    assert payload["next_action"] == "caller_decision"
    assert payload["convergence"]["reason"] == "budget_exhausted"
    assert payload["convergence"]["accepted_findings_heads"] == 3


def test_review_state_status_json_without_current_review(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setattr(review_state, "resolve_repo_root", lambda cd: tmp_path)
    monkeypatch.setattr(
        review_state,
        "inspect_workflow_status",
        lambda **kwargs: {
            "status": "ok",
            "base": "main",
            "branch": "feature/json",
            "head": "head-1",
            "recommendation": "full-review",
            "reason": "no_review_anchor",
        },
    )
    _use_default_state_dir(monkeypatch, tmp_path / "state")
    monkeypatch.setattr(
        sys, "argv", ["review.py", "--status", "--json", "--base", "main"]
    )

    assert review.main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["head"] == "head-1"
    assert payload["reason"] == "no_review_anchor"
    assert "review" not in payload
    assert "Action" not in payload


def test_review_state_status_json_preserves_ambiguous_match_failure(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        review_state,
        "resolve_repo_root",
        lambda cd: (_ for _ in ()).throw(
            ValueError("multiple active review cycles match this repo")
        ),
    )
    monkeypatch.setattr(sys, "argv", ["review.py", "--status", "--json"])

    assert review.main() == 2
    assert "multiple active review cycles match this repo" in capsys.readouterr().out


def test_review_state_status_default_output_is_unchanged(
    monkeypatch, tmp_path: Path
) -> None:
    emitted: list[dict[str, object]] = []
    monkeypatch.setattr(review_state, "resolve_repo_root", lambda cd: tmp_path)
    monkeypatch.setattr(
        review_state,
        "inspect_workflow_status",
        lambda **kwargs: {
            "status": "ok",
            "base": "main",
            "branch": "feature/json",
            "head": "head-1",
            "recommendation": "full-review",
            "reason": "no_review_anchor",
        },
    )
    _use_default_state_dir(monkeypatch, tmp_path / "state")
    monkeypatch.setattr(review_state, "emit_toon", emitted.append)
    monkeypatch.setattr(sys, "argv", ["review.py", "--status", "--base", "main"])

    assert review.main() == 0
    assert list(emitted[0]) == ["status", "recommendation", "reason", "Action"]
    assert "base" not in emitted[0]
    assert "branch" not in emitted[0]
    assert "head" not in emitted[0]


def test_review_state_status_surfaces_orchestrator_progress(
    monkeypatch, tmp_path: Path
) -> None:
    emitted: list[dict[str, object]] = []
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    repo.mkdir()
    _write_orchestrator_cycle(
        state_dir,
        "orc-progress.json",
        {
            "public_id": "rvw_progress",
            "stage": "decision-pending",
            "mode": {"requested": "normal", "effective": "normal"},
            "identity": {
                "cwd": str(review_state.normalize_review_cwd_value(repo)),
                "base": "main",
                "branch": "feature/progress",
            },
            "pending_action": {
                "kind": "decision",
                "lane": "review_t1",
                "round_id": "round-progress",
                "step": "broad-discovery",
                "step_index": 1,
            },
            "review_plan": {
                "steps": [
                    {"name": "deslop", "kind": "deslop"},
                    {"name": "broad-discovery", "kind": "review"},
                    {"name": "precision-signoff", "kind": "review"},
                ],
            },
        },
    )
    monkeypatch.setattr(review_state, "resolve_repo_root", lambda cd: repo)
    monkeypatch.setattr(
        review_state,
        "inspect_workflow_status",
        lambda **kwargs: {
            "recommendation": "coherence-review",
            "reason": "diff_churn_exceeded",
            "branch": "feature/progress",
        },
    )
    _use_default_state_dir(monkeypatch, state_dir)
    monkeypatch.setattr(
        review_state, "emit_toon", lambda payload: emitted.append(payload)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["review.py", "--status", "--base", "main"],
    )

    exit_code = review.main()

    assert exit_code == 0
    assert emitted[0]["review"] == "rvw_progress"
    assert emitted[0]["progress"] == "review 1/2 broad-discovery"
    assert "recommendation" not in emitted[0]
    assert "reason" not in emitted[0]
    action = emitted[0]["Action"]
    assert set(action) == {"choices", "note", "restart"}
    assert action["note"] == (
        "Classify the reviewer output, then record clean or findings."
    )
    assert "--id rvw_progress --decision clean" in str(action["choices"]["clean"])
    assert "--id rvw_progress --decision findings" in str(action["choices"]["findings"])
    assert "--id rvw_progress --restart-mode deep --reason REASON" in str(
        emitted[0]["Action"]["restart"]["cmd"]
    )
    assert emitted[0]["Action"]["restart"]["mode"] == "deep"
    assert "--state-dir" not in str(action["choices"]["clean"])
    assert "--state-dir" not in str(action["choices"]["findings"])
    assert "--state-dir" not in str(emitted[0]["Action"]["restart"]["cmd"])
    assert str(state_dir.resolve(strict=False)) not in str(action["choices"]["clean"])
    assert "review_t1.py" not in str(emitted[0]["Action"])


def test_review_state_status_keeps_persisted_classified_blocked(
    monkeypatch, tmp_path: Path
) -> None:
    emitted: list[dict[str, object]] = []
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    head = "head-1"
    repo.mkdir()
    _write_orchestrator_cycle(
        state_dir,
        "orc-done.json",
        {
            "public_id": "rvw_done",
            "stage": "review-green",
            "mode": {"requested": "normal", "effective": "normal"},
            "identity": {
                "cwd": str(review_state.normalize_review_cwd_value(repo)),
                "base": "main",
                "branch": "feature/progress",
                "head": head,
            },
            "review_heads": {"last_reviewed_head": head},
            "github_review": {"status": "clean", "reviewed_head": head},
            "validation": {"full_suite": "passed", "ci": "classified"},
            "deslop": {"tracked": False},
        },
    )
    monkeypatch.setattr(review_state, "resolve_repo_root", lambda cd: repo)
    monkeypatch.setattr(
        review_state,
        "inspect_workflow_status",
        lambda **kwargs: {
            "recommendation": "coherence-review",
            "reason": "diff_churn_exceeded",
            "branch": "feature/progress",
            "head": head,
        },
    )
    _use_default_state_dir(monkeypatch, state_dir)
    monkeypatch.setattr(
        review_state, "emit_toon", lambda payload: emitted.append(payload)
    )
    monkeypatch.setattr(sys, "argv", ["review.py", "--status", "--base", "main"])

    assert review.main() == 0

    assert emitted[0]["review"] == "rvw_done"
    assert "status" not in emitted[0]
    assert emitted[0]["done"] is False
    assert emitted[0]["review_ladder"] == "pending"
    assert emitted[0]["next_action"] == "validation"
    assert emitted[0]["Action"]["blocked_by"] == ["ci:classified"]


def test_review_state_status_keeps_patch_equivalent_base_drift_done(
    monkeypatch, tmp_path: Path
) -> None:
    emitted: list[dict[str, object]] = []
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    old_head = "old-head"
    new_head = "new-head"
    repo.mkdir()
    _write_orchestrator_cycle(
        state_dir,
        "orc-base-drift-done.json",
        {
            "public_id": "rvw_done",
            "stage": "review-green",
            "mode": {"requested": "normal", "effective": "normal"},
            "identity": {
                "cwd": str(review_state.normalize_review_cwd_value(repo)),
                "base": "main",
                "branch": "feature/progress",
                "head": old_head,
            },
            "review_heads": {"last_reviewed_head": old_head},
            "github_review": {"status": "clean", "reviewed_head": old_head},
            "validation": {
                "full_suite": "passed",
                "ci": "waived",
                "note": "CI unavailable for this docs-only change",
            },
            "deslop": {"tracked": False},
            "base_drift": {
                "patch_equivalent": True,
                "reviewed_head": old_head,
                "equivalent_reviewed_head": new_head,
            },
        },
    )
    monkeypatch.setattr(review_state, "resolve_repo_root", lambda cd: repo)
    monkeypatch.setattr(
        review_state,
        "inspect_workflow_status",
        lambda **kwargs: {
            "recommendation": "coherence-review",
            "reason": "diff_churn_exceeded",
            "branch": "feature/progress",
            "head": new_head,
        },
    )
    _use_default_state_dir(monkeypatch, state_dir)
    monkeypatch.setattr(
        review_state, "emit_toon", lambda payload: emitted.append(payload)
    )
    monkeypatch.setattr(sys, "argv", ["review.py", "--status", "--base", "main"])

    assert review.main() == 0

    assert emitted[0]["status"] == "done"
    assert emitted[0]["done"] is True
    assert emitted[0]["review_ladder"] == "complete"
    assert "Action" not in emitted[0]


def test_review_state_status_uses_bare_id_for_structured_verdict(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    round_state_dir = state_dir / "orchestrator" / "review-rounds"
    rounds_dir = round_state_dir / "rounds"
    rounds_dir.mkdir(parents=True)
    (rounds_dir / "round-1.json").write_text(
        json.dumps(
            {
                "round_id": "round-1",
                "status": "completed",
                "runs": [
                    {
                        "slot": "alpha",
                        "review_status": "completed",
                        "reviewer_output": "Review result: findings",
                        "terminal_command": "clean",
                        "grade_blocked": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    action = review_state._orchestrator_action(
        {
            "stage": "decision-pending",
            "pending_action": {
                "kind": "decision",
                "round_id": "round-1",
                "lane": "review_t1",
            },
            "rounds": [
                {
                    "round_id": "round-1",
                    "lane": "review_t1",
                    "review_status": "completed",
                    "round_state_dir": str(round_state_dir),
                    "runs": [
                        {
                            "slot": "alpha",
                            "review_status": "completed",
                            "summary": "No findings.",
                            "ref": "round-1/alpha.txt",
                            "grade_blocked": False,
                        }
                    ],
                }
            ],
        },
        "rvw_progress",
        state_dir=state_dir,
    )

    assert "--id rvw_progress" in str(action["cmd"])
    assert "--decision" not in str(action["cmd"])
    assert "--id rvw_progress --decision clean" in str(action["override"]["clean"])
    assert "--id rvw_progress --decision findings" in str(
        action["override"]["findings"]
    )
    briefed_fix = review_state._orchestrator_action(
        {"stage": "fix-pending", "review_brief": "Frozen contract"},
        "rvw_progress",
        state_dir=state_dir,
    )
    briefless_fix = review_state._orchestrator_action(
        {"stage": "fix-pending"}, "rvw_progress", state_dir=state_dir
    )
    assert "--contract-conflict <dimension>" in str(briefed_fix["note"])
    assert "--fixes-validated" in str(briefless_fix["choices"])


def test_review_state_status_surfaces_grade_before_structured_verdict(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    round_state_dir = state_dir / "orchestrator" / "review-rounds"
    rounds_dir = round_state_dir / "rounds"
    rounds_dir.mkdir(parents=True)
    (rounds_dir / "arena-round-1.json").write_text(
        json.dumps(
            {
                "round_id": "arena-round-1",
                "arena_round": True,
                "status": "completed",
                "task_id_hint": "feature/arena",
                "rating_pool_id": "discovery-phase-gpt-5.6-v1",
                "runs": [
                    {
                        "slot": slot,
                        "review_status": "completed",
                        "reviewer_output": "No findings.\n\nReview result: clean",
                        "grade_blocked": False,
                    }
                    for slot in ("alpha", "bravo", "charlie", "delta")
                ],
            }
        ),
        encoding="utf-8",
    )

    action = review_state._orchestrator_action(
        {
            "public_id": "rvw_progress",
            "stage": "decision-pending",
            "pending_action": {
                "kind": "decision",
                "round_id": "arena-round-1",
                "lane": "review_t1",
            },
            "identity": {"branch": "feature/arena"},
            "rounds": [
                {
                    "round_id": "arena-round-1",
                    "lane": "review_t1",
                    "review_status": "completed",
                    "status": "completed",
                    "grading_required": True,
                    "arena_round": True,
                    "round_state_dir": str(round_state_dir),
                }
            ],
        },
        "rvw_progress",
        state_dir=state_dir,
    )

    assert "review_suite_arena.py grade" in str(action["cmd"])
    assert "--round-id arena-round-1" in str(action["cmd"])
    assert "--task-id feature/arena" in str(action["cmd"])
    assert "--rating-pool-id discovery-phase-gpt-5.6-v1" in str(action["cmd"])
    assert str(action["cmd"]).count("--rank") == 4
    assert str(round_state_dir) in str(action["cmd"])
    assert "--id rvw_progress" in str(action["next"])
    assert "--decision" not in str(action["cmd"])
    assert "override" not in action


def test_orchestrator_action_routes_superseded_reviews(tmp_path: Path) -> None:
    action = review_state._orchestrator_action(
        {"stage": "aborted", "superseded_by": {"review": "rvw_new"}},
        "rvw_old",
        state_dir=tmp_path / "state",
    )

    assert "--id rvw_new" in str(action["cmd"])
    assert "superseded" in str(action["note"])
    assert "restart" not in action


def test_orchestrator_action_omits_initial_fast_github_handoff(
    tmp_path: Path,
) -> None:
    action = review_state._orchestrator_action(
        {
            "stage": "review-green",
            "mode": {"effective": "fast"},
            "github_review": {"status": "unknown"},
        },
        "rvw_fast",
        state_dir=tmp_path / "state",
    )

    assert action is None


def test_orchestrator_action_routes_terminal_github_result_to_validation(
    tmp_path: Path,
) -> None:
    action = review_state._orchestrator_action(
        {
            "stage": "review-green",
            "mode": {"effective": "fast"},
            "identity": {"head": "old-head"},
            "review_heads": {
                "last_reviewed_head": "head-1",
            },
            "github_review": {"status": "clean", "reviewed_head": "head-1"},
            "validation": {"full_suite": "unknown", "ci": "unknown"},
        },
        "rvw_fast",
        state_dir=tmp_path / "state",
        current_head="head-1",
    )

    assert action is not None
    assert action["blocked_by"] == ["full_suite:unknown", "ci:unknown"]
    assert "--full-suite FULL_SUITE_STATUS --ci CI_STATUS" in str(action["cmd"])
    assert '--validation-note "reason"' in str(action["note"])
    assert "alt" not in action
    assert "--github-review" not in str(action["cmd"])
    repair = review_state._orchestrator_validation_blocker_action(
        "rvw_fast", ["ci:waived_without_note"]
    )
    assert "--ci waived --validation-note WAIVER_REASON" in str(repair["cmd"])


def test_orchestrator_action_suppresses_stale_terminal_github_result(
    tmp_path: Path,
) -> None:
    action = review_state._orchestrator_action(
        {
            "stage": "review-green",
            "mode": {"effective": "normal"},
            "identity": {"head": "old-head"},
            "review_heads": {
                "last_reviewed_head": "old-head",
                "last_followup_head": "fixed-head",
            },
            "github_review": {"status": "clean", "reviewed_head": "old-head"},
            "validation": {"full_suite": "passed", "ci": "passed"},
        },
        "rvw_progress",
        state_dir=tmp_path / "state",
        current_head="fixed-head",
    )

    assert action is None


def test_orchestrator_action_uses_clean_followup_head_for_freshness(
    tmp_path: Path,
) -> None:
    action = review_state._orchestrator_action(
        {
            "stage": "review-green",
            "mode": {"effective": "normal"},
            "review_heads": {
                "last_reviewed_head": "old-head",
                "last_followup_head": "fixed-head",
            },
            "github_review": {"status": "unknown"},
            "validation": {"full_suite": "passed", "ci": "passed"},
        },
        "rvw_progress",
        state_dir=tmp_path / "state",
        current_head="fixed-head",
    )

    assert action is not None
    assert "--github-review" in str(action["cmd"])


def test_review_state_status_ignores_stale_green_orchestrator_cycle(
    monkeypatch, tmp_path: Path
) -> None:
    emitted: list[dict[str, object]] = []
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    repo.mkdir()
    _write_orchestrator_cycle(
        state_dir,
        "orc-stale-green.json",
        {
            "public_id": "rvw_stale",
            "stage": "review-green",
            "identity": {
                "cwd": str(review_state.normalize_review_cwd_value(repo)),
                "base": "main",
                "branch": "feature/progress",
                "head": "old-head",
            },
            "review_heads": {"last_reviewed_head": "old-head"},
        },
    )
    monkeypatch.setattr(review_state, "resolve_repo_root", lambda cd: repo)
    monkeypatch.setattr(
        review_state,
        "inspect_workflow_status",
        lambda **kwargs: {
            "recommendation": "coherence-review",
            "reason": "diff_churn_exceeded",
            "recommended_lane": "review_t1",
            "branch": "feature/progress",
            "head": "new-head",
        },
    )
    _use_default_state_dir(monkeypatch, state_dir)
    monkeypatch.setattr(
        review_state, "emit_toon", lambda payload: emitted.append(payload)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["review.py", "--status", "--base", "main"],
    )

    exit_code = review.main()

    assert exit_code == 0
    assert "review" not in emitted[0]
    assert emitted[0]["recommendation"] == "coherence-review"
    assert emitted[0]["reason"] == "diff_churn_exceeded"
    assert "review.py" in str(emitted[0]["Action"]["cmd"])
    assert "--mode normal" in str(emitted[0]["Action"]["cmd"])
    assert "review_t1.py" not in str(emitted[0]["Action"])


def test_review_state_status_rejects_windows_wsl_unc_before_git(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured_errors: list[str] = []

    monkeypatch.setattr(review_state.sys, "platform", "win32")
    monkeypatch.setattr(
        review_state,
        "resolve_repo_root",
        lambda cd: (_ for _ in ()).throw(
            AssertionError("UNC WSL path should fail before git")
        ),
    )
    monkeypatch.setattr(
        review,
        "emit_error",
        lambda message, **kwargs: captured_errors.append(message) or 2,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review.py",
            "--status",
            "--cd",
            "//wsl.localhost/Ubuntu/home/alice/code/repo",
            "--base",
            "main",
            "--wsl",
        ],
    )

    exit_code = review.main()

    assert exit_code == 2
    assert "Run review.py --status from native WSL" in captured_errors[0]
    assert "--wsl is not useful" in captured_errors[0]


def test_inspect_workflow_status_warns_after_six_same_tier_runs_without_counting_followups(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/convergence")
    reviewed_head = _commit_file(repo, "app.txt", "base\nfeature\n", "feature")
    merge_base = _git(repo, "merge-base", "main", "HEAD")

    for index in range(6):
        record_review_anchor(
            state_dir=state_dir,
            review_cwd=repo,
            lane="review_t3",
            base="main",
            reviewed_head=reviewed_head,
            review_scope={
                "base": "main",
                "reviewed_head": reviewed_head,
                "merge_base": merge_base,
            },
            round_id=f"t3-{index}",
        )
    for index in range(4):
        record_review_anchor(
            state_dir=state_dir,
            review_cwd=repo,
            lane="review-followup",
            base="main",
            reviewed_head=reviewed_head,
            review_scope={"commit": reviewed_head, "commit_end": reviewed_head},
            round_id=f"followup-{index}",
        )

    payload = inspect_workflow_status(state_dir=state_dir, review_cwd=repo, base="main")

    assert payload["recommendation"] == "none"
    assert payload["convergence"]["status"] == "caution"
    assert payload["convergence"]["tier"] == "review_t3"
    assert payload["convergence"]["same_tier_true_run_count"] == 6
    assert "not review-followup" in payload["convergence"]["note"]
    assert "converging and patch-sized" in payload["convergence"]["instruction"]


def test_inspect_workflow_status_high_pressure_after_ten_same_tier_runs(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/high-pressure")
    reviewed_head = _commit_file(repo, "app.txt", "base\nfeature\n", "feature")
    merge_base = _git(repo, "merge-base", "main", "HEAD")

    for index in range(10):
        record_review_anchor(
            state_dir=state_dir,
            review_cwd=repo,
            lane="review_t3",
            base="main",
            reviewed_head=reviewed_head,
            review_scope={
                "base": "main",
                "reviewed_head": reviewed_head,
                "merge_base": merge_base,
            },
            round_id=f"t3-{index}",
        )

    payload = inspect_workflow_status(state_dir=state_dir, review_cwd=repo, base="main")

    assert payload["recommendation"] == "none"
    assert payload["convergence"]["status"] == "high_pressure"
    assert payload["convergence"]["same_tier_true_run_count"] == 10
    assert "full diff" in payload["convergence"]["instruction"]
    assert "pause and discuss" in payload["convergence"]["instruction"]


def test_record_review_anchor_compacts_tool_only_state(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    reviewed_head = _commit_file(repo, "app.txt", "one\n", "initial")

    record_review_anchor(
        state_dir=state_dir,
        review_cwd=repo,
        lane="review-followup",
        base="main",
        review_scope={
            "commit": reviewed_head,
            "commit_end": reviewed_head,
            "branch_base": "main",
            "requested_base": "main",
            "base_upstream": "origin/main",
            "base_upstream_head": reviewed_head,
            "effective_base_head": reviewed_head,
            "merge_base": reviewed_head,
            "target_label": f"interdiff `{reviewed_head}..{reviewed_head}`",
            "manual_prompt_reason": "custom instructions",
        },
        reviewed_head=reviewed_head,
        output_refs=["rollout://should-not-be-routing-state"],
        session_id="session-should-not-be-routing-state",
        note="invariant=" + ("very verbose root-cause note " * 20),
    )

    state = load_workflow_state(state_dir=state_dir, review_cwd=repo)
    assert state is not None
    anchor = state["anchors"][-1]
    assert "note" not in anchor
    assert "output_refs" not in anchor
    assert "session_id" not in anchor
    assert anchor["review_scope"] == {
        "commit": reviewed_head,
        "commit_end": reviewed_head,
        "branch_base": "main",
        "requested_base": "main",
        "base_upstream": "origin/main",
        "base_upstream_head": reviewed_head,
        "effective_base_head": reviewed_head,
        "merge_base": reviewed_head,
    }


def test_latest_base_review_context_anchor_matches_branch_base_metadata() -> None:
    anchor = {
        "lane": "review_t3",
        "review_scope": {
            "base": "origin/main",
            "branch_base": "main",
            "merge_base": "merge-base-sha",
        },
    }

    assert (
        latest_base_review_context_anchor({"anchors": [anchor]}, requested_base="main")
        is anchor
    )


def test_inspect_workflow_status_recommends_followup_for_small_delta(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    reviewed_head = _commit_file(repo, "app.txt", "one\n", "initial")
    record_review_anchor(
        state_dir=state_dir,
        review_cwd=repo,
        lane="review_t1",
        base="main",
        reviewed_head=reviewed_head,
    )
    _commit_file(repo, "app.txt", "one\ntwo\n", "small fix")

    payload = inspect_workflow_status(
        state_dir=state_dir,
        review_cwd=repo,
        base="main",
    )

    assert payload["recommendation"] == "review-followup"
    assert payload["reason"] == "small_delta_after_review"
    assert payload["last_reviewed_head"] == reviewed_head
    assert payload["commits_since_anchor"] == 1


def test_inspect_workflow_status_recommends_coherence_for_large_delta(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    reviewed_head = _commit_file(repo, "app.txt", "one\n", "initial")
    record_review_anchor(
        state_dir=state_dir,
        review_cwd=repo,
        lane="review_t3",
        base="main",
        reviewed_head=reviewed_head,
    )
    for index in range(7):
        _commit_file(repo, f"src/file_{index}.txt", f"{index}\n", f"touch file {index}")

    payload = inspect_workflow_status(
        state_dir=state_dir,
        review_cwd=repo,
        base="main",
    )

    assert payload["recommendation"] == "coherence-review"
    assert payload["reason"] == "diff_churn_exceeded"
    assert payload["files_changed"] == 7


def test_inspect_workflow_status_escalates_small_delta_when_branch_review_pressure_is_high(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "base.txt", "base\n", "initial")
    _git(repo, "checkout", "-b", "feature/pressure")

    latest_reviewed_head = _commit_file(
        repo, "app.txt", "reviewed\n", "latest reviewed"
    )
    merge_base = _git(repo, "merge-base", "main", "HEAD")
    real_commit_distance = workflow_state_module.commit_distance

    def fake_commit_distance(
        review_cwd: Path, start_ref: str, end_ref: str = "HEAD"
    ) -> int:
        if start_ref == merge_base and end_ref == "HEAD":
            return 25
        return real_commit_distance(review_cwd, start_ref, end_ref)

    monkeypatch.setattr(workflow_state_module, "commit_distance", fake_commit_distance)

    for index in range(12):
        record_review_anchor(
            state_dir=state_dir,
            review_cwd=repo,
            lane="review-followup",
            base="main",
            reviewed_head=latest_reviewed_head,
            review_scope={
                "commit": latest_reviewed_head,
                "commit_end": latest_reviewed_head,
                "merge_base": merge_base,
            },
        )
    _commit_file(repo, "app.txt", "tip\n", "small fix")

    payload = inspect_workflow_status(
        state_dir=state_dir,
        review_cwd=repo,
        base="main",
    )

    assert payload["recommendation"] == "coherence-review"
    assert payload["reason"] == "branch_review_pressure_exceeded"
    assert payload["commits_since_anchor"] == 1
    assert payload["branch_commits_since_base"] >= 25
    assert payload["recorded_review_anchor_count"] >= 12
    assert payload["followup_anchor_count"] >= 5


def test_inspect_workflow_status_escalates_after_too_many_followups_since_full_checkpoint(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "base.txt", "base\n", "initial")
    _git(repo, "checkout", "-b", "feature/followup-cap")
    checkpoint_head = _commit_file(repo, "app.txt", "checkpoint\n", "checkpoint")
    merge_base_at_checkpoint = _git(repo, "merge-base", "main", "HEAD")
    record_review_anchor(
        state_dir=state_dir,
        review_cwd=repo,
        lane="review_t3",
        base="main",
        reviewed_head=checkpoint_head,
        review_scope={
            "base": "main",
            "reviewed_head": checkpoint_head,
            "merge_base": merge_base_at_checkpoint,
        },
    )

    previous_head = checkpoint_head
    for index in range(1, 4):
        next_head = _commit_file(
            repo, "app.txt", f"followup {index}\n", f"followup {index}"
        )
        record_review_anchor(
            state_dir=state_dir,
            review_cwd=repo,
            lane="review-followup",
            base="main",
            reviewed_head=next_head,
            review_scope={
                "commit": previous_head,
                "commit_end": next_head,
                "merge_base": merge_base_at_checkpoint,
            },
        )
        previous_head = next_head
    _commit_file(repo, "app.txt", "small fix\n", "small fix")

    payload = inspect_workflow_status(
        state_dir=state_dir,
        review_cwd=repo,
        base="main",
    )

    assert payload["recommendation"] == "coherence-review"
    assert payload["reason"] == "followup_cycle_limit_exceeded"
    assert payload["commits_since_anchor"] == 1
    assert payload["followup_anchor_count_since_full_review"] == 3
    assert payload["last_full_review_lane"] == "review_t3"


def test_inspect_workflow_status_github_review_resets_followup_cycle_pressure(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "base.txt", "base\n", "initial")
    _git(repo, "checkout", "-b", "feature/post-github")
    checkpoint_head = _commit_file(repo, "app.txt", "checkpoint\n", "checkpoint")
    merge_base_at_checkpoint = _git(repo, "merge-base", "main", "HEAD")
    record_review_anchor(
        state_dir=state_dir,
        review_cwd=repo,
        lane="review_t3",
        base="main",
        reviewed_head=checkpoint_head,
        review_scope={
            "base": "main",
            "reviewed_head": checkpoint_head,
            "merge_base": merge_base_at_checkpoint,
        },
    )

    previous_head = checkpoint_head
    for index in range(1, 4):
        next_head = _commit_file(
            repo, "app.txt", f"followup {index}\n", f"followup {index}"
        )
        record_review_anchor(
            state_dir=state_dir,
            review_cwd=repo,
            lane="review-followup",
            base="main",
            reviewed_head=next_head,
            review_scope={
                "commit": previous_head,
                "commit_end": next_head,
                "merge_base": merge_base_at_checkpoint,
            },
        )
        previous_head = next_head
    record_review_anchor(
        state_dir=state_dir,
        review_cwd=repo,
        lane="review-github",
        base="main",
        reviewed_head=previous_head,
        review_scope={
            "base": "main",
            "reviewed_head": previous_head,
            "merge_base": merge_base_at_checkpoint,
        },
    )
    _commit_file(repo, "app.txt", "github fix\n", "fix github finding")

    payload = inspect_workflow_status(
        state_dir=state_dir,
        review_cwd=repo,
        base="main",
    )

    assert payload["recommendation"] == "review-followup"
    assert payload["reason"] == "small_delta_after_review"
    assert "followup_anchor_count_since_full_review" not in payload


def test_inspect_workflow_status_recommends_full_review_when_anchor_is_not_ancestor(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    first = _commit_file(repo, "app.txt", "one\n", "initial")
    reviewed_head = _commit_file(repo, "app.txt", "one\ntwo\n", "reviewed")
    record_review_anchor(
        state_dir=state_dir,
        review_cwd=repo,
        lane="review_t3",
        base="main",
        reviewed_head=reviewed_head,
        review_scope={"base": "main", "reviewed_head": reviewed_head},
    )
    _git(repo, "reset", "--hard", first)
    _commit_file(repo, "app.txt", "one\nthree\n", "rebased replacement")

    payload = inspect_workflow_status(
        state_dir=state_dir,
        review_cwd=repo,
        base="main",
    )

    assert payload["recommendation"] == "full-review"
    assert payload["reason"] == "review_anchor_not_ancestor"


def test_inspect_workflow_status_accepts_non_overlapping_equivalent_merge_base_drift(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    base_at_review = _commit_file(repo, "base.txt", "one\n", "initial")
    _git(repo, "checkout", "-b", "feature/test")
    reviewed_head = _commit_file(repo, "feature.txt", "feat\n", "feature work")
    record_review_anchor(
        state_dir=state_dir,
        review_cwd=repo,
        lane="review_t3",
        base="main",
        reviewed_head=reviewed_head,
        review_scope={
            "base": "main",
            "reviewed_head": reviewed_head,
            "merge_base": base_at_review,
        },
    )
    _git(repo, "checkout", "main")
    _commit_file(repo, "base.txt", "one\ntwo\n", "main moves")
    _git(repo, "checkout", "feature/test")
    _git(repo, "merge", "--no-ff", "main", "-m", "merge main")

    payload = inspect_workflow_status(
        state_dir=state_dir,
        review_cwd=repo,
        base="main",
    )

    assert payload["recommendation"] == "none"
    assert payload["reason"] == "current_head_already_reviewed"
    assert payload["base_drift_review_equivalent"] is True
    assert payload["base_drift_overlap_paths"] == []


def test_inspect_workflow_status_recommends_full_review_when_merge_base_drift_overlaps_branch_paths(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    base_at_review = _commit_file(repo, "app.txt", "one\n", "initial")
    _git(repo, "checkout", "-b", "feature/test")
    reviewed_head = _commit_file(repo, "app.txt", "one\nfeature\n", "feature work")
    record_review_anchor(
        state_dir=state_dir,
        review_cwd=repo,
        lane="review_t3",
        base="main",
        reviewed_head=reviewed_head,
        review_scope={
            "base": "main",
            "reviewed_head": reviewed_head,
            "merge_base": base_at_review,
        },
    )
    _git(repo, "checkout", "main")
    _commit_file(repo, "app.txt", "one\nmain\n", "main moves")
    _git(repo, "checkout", "feature/test")
    _git(repo, "merge", "-s", "ours", "--no-ff", "main", "-m", "merge main")

    payload = inspect_workflow_status(
        state_dir=state_dir,
        review_cwd=repo,
        base="main",
    )

    assert payload["recommendation"] == "full-review"
    assert payload["reason"] == "base_merge_base_changed"
    assert payload["base_drift_overlap_paths"] == ["app.txt"]


def test_inspect_workflow_status_uses_latest_base_review_context_after_followup(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    stale_base = _commit_file(repo, "app.txt", "one\n", "initial")
    _commit_file(repo, "app.txt", "one\nmain\n", "main moves")
    current_base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-b", "feature/test")
    reviewed_head = _commit_file(repo, "feature.txt", "feat\n", "feature work")
    record_review_anchor(
        state_dir=state_dir,
        review_cwd=repo,
        lane="review_t1",
        base="main",
        reviewed_head=reviewed_head,
        review_scope={
            "base": "main",
            "reviewed_head": reviewed_head,
            "merge_base": current_base,
        },
    )
    final_head = _commit_file(repo, "feature.txt", "feat\nfix\n", "followup fix")
    record_review_anchor(
        state_dir=state_dir,
        review_cwd=repo,
        lane="review-followup",
        base="main",
        reviewed_head=final_head,
        review_scope={
            "commit": reviewed_head,
            "commit_end": final_head,
            "merge_base": stale_base,
        },
    )

    payload = inspect_workflow_status(
        state_dir=state_dir,
        review_cwd=repo,
        base="main",
    )

    assert payload["recommendation"] == "none"
    assert payload["reason"] == "current_head_already_reviewed"
    assert payload["recorded_merge_base"] == current_base


def test_commit_only_review_anchor_does_not_advance_branch_review_state(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    initial_head = _commit_file(repo, "app.txt", "one\n", "initial")
    reviewed_head = _commit_file(repo, "app.txt", "one\ntwo\n", "reviewed")
    record_review_anchor(
        state_dir=state_dir,
        review_cwd=repo,
        lane="review_t3",
        base="main",
        reviewed_head=reviewed_head,
        review_scope={"base": "main", "reviewed_head": reviewed_head},
    )
    _commit_file(repo, "app.txt", "one\ntwo\nthree\n", "middle commit")
    _commit_file(repo, "app.txt", "one\ntwo\nthree\nfour\n", "tip commit")
    record_review_anchor(
        state_dir=state_dir,
        review_cwd=repo,
        lane="review_t1",
        base="main",
        review_scope={"commit": initial_head},
    )

    payload = inspect_workflow_status(
        state_dir=state_dir,
        review_cwd=repo,
        base="main",
    )

    assert payload["recommendation"] == "review-followup"
    assert payload["last_reviewed_head"] == reviewed_head


def test_inspect_workflow_status_falls_back_to_older_valid_branch_anchor(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    first = _commit_file(repo, "app.txt", "one\n", "initial")
    reviewed_head = _commit_file(repo, "app.txt", "one\ntwo\n", "reviewed")
    record_review_anchor(
        state_dir=state_dir,
        review_cwd=repo,
        lane="review_t3",
        base="main",
        reviewed_head=reviewed_head,
        review_scope={"base": "main", "reviewed_head": reviewed_head},
    )
    _commit_file(repo, "app.txt", "one\ntwo\nthree\n", "head")
    _git(repo, "checkout", "-b", "side-review", first)
    unrelated_reviewed_head = _commit_file(repo, "side.txt", "other\n", "side review")
    _git(repo, "checkout", "main")
    record_review_anchor(
        state_dir=state_dir,
        review_cwd=repo,
        lane="review_t1",
        base="main",
        reviewed_head=unrelated_reviewed_head,
        review_scope={"commit": unrelated_reviewed_head},
    )

    payload = inspect_workflow_status(
        state_dir=state_dir,
        review_cwd=repo,
        base="main",
    )

    assert payload["recommendation"] == "review-followup"
    assert payload["last_reviewed_head"] == reviewed_head


def test_commit_only_review_anchor_overrides_older_branch_anchor_when_it_is_closer_to_head(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    reviewed_head = _commit_file(repo, "app.txt", "one\n", "initial")
    record_review_anchor(
        state_dir=state_dir,
        review_cwd=repo,
        lane="review_t3",
        base="main",
        reviewed_head=reviewed_head,
        review_scope={"base": "main", "reviewed_head": reviewed_head},
    )
    focused_head = _commit_file(repo, "app.txt", "one\ntwo\n", "focused commit")
    record_review_anchor(
        state_dir=state_dir,
        review_cwd=repo,
        lane="review_t1",
        base="main",
        review_scope={"commit": focused_head},
    )
    _commit_file(repo, "app.txt", "one\ntwo\nthree\n", "tip commit")

    payload = inspect_workflow_status(
        state_dir=state_dir,
        review_cwd=repo,
        base="main",
    )

    assert payload["recommendation"] == "review-followup"
    assert payload["last_reviewed_head"] == focused_head
    assert payload["last_reviewed_lane"] == "review_t1"


def test_commit_only_review_anchor_bootstraps_when_no_branch_anchor_exists(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    reviewed_head = _commit_file(repo, "app.txt", "one\n", "initial")
    _commit_file(repo, "app.txt", "one\ntwo\n", "small fix")
    record_review_anchor(
        state_dir=state_dir,
        review_cwd=repo,
        lane="review_t1",
        base="main",
        review_scope={"commit": reviewed_head},
    )

    payload = inspect_workflow_status(
        state_dir=state_dir,
        review_cwd=repo,
        base="main",
    )

    assert payload["recommendation"] == "review-followup"
    assert payload["last_reviewed_head"] == reviewed_head


def test_inspect_workflow_status_recommends_followup_for_dirty_same_head(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "base.txt", "base\n", "initial")
    _git(repo, "checkout", "-b", "feature/dirty-followup")
    reviewed_head = _commit_file(repo, "app.txt", "one\n", "feature work")
    record_review_anchor(
        state_dir=state_dir,
        review_cwd=repo,
        lane="review_t3",
        base="main",
        reviewed_head=reviewed_head,
        review_scope={"base": "main", "reviewed_head": reviewed_head},
    )
    (repo / "app.txt").write_text("one\ntwo\n", encoding="utf-8")

    payload = inspect_workflow_status(
        state_dir=state_dir,
        review_cwd=repo,
        base="main",
    )

    assert payload["recommendation"] == "review-followup"
    assert payload["reason"] == "dirty_worktree_after_review"
    assert payload["worktree_dirty"] is True
    assert payload["files_changed"] == 1
    assert payload["last_reviewed_head"] == reviewed_head


def test_inspect_workflow_status_ignores_dirty_paths_outside_branch_diff_on_same_head(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "base.txt", "base\n", "initial")
    _git(repo, "checkout", "-b", "feature/clean-signoff")
    reviewed_head = _commit_file(repo, "src/app.txt", "feature\n", "feature work")
    record_review_anchor(
        state_dir=state_dir,
        review_cwd=repo,
        lane="review_t3",
        base="main",
        reviewed_head=reviewed_head,
        review_scope={
            "base": "main",
            "reviewed_head": reviewed_head,
            "merge_base": _git(repo, "merge-base", "main", "HEAD"),
        },
    )
    (repo / "docs.md").write_text("notes\n", encoding="utf-8")

    payload = inspect_workflow_status(
        state_dir=state_dir,
        review_cwd=repo,
        base="main",
    )

    assert payload["recommendation"] == "none"
    assert payload["reason"] == "dirty_worktree_outside_branch_diff"
    assert payload["worktree_dirty"] is True
    assert payload["ignored_dirty_path_count"] == 1
    assert payload["ignored_dirty_paths"] == ["docs.md"]


def test_inspect_workflow_status_keeps_quoted_dirty_branch_paths_related(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "base.txt", "base\n", "initial")
    _git(repo, "checkout", "-b", "feature/quoted-path")
    reviewed_head = _commit_file(repo, "src/a b.txt", "feature\n", "feature work")
    record_review_anchor(
        state_dir=state_dir,
        review_cwd=repo,
        lane="review_t3",
        base="main",
        reviewed_head=reviewed_head,
        review_scope={
            "base": "main",
            "reviewed_head": reviewed_head,
            "merge_base": _git(repo, "merge-base", "main", "HEAD"),
        },
    )
    (repo / "src" / "a b.txt").write_text("feature\ndirty\n", encoding="utf-8")

    payload = inspect_workflow_status(
        state_dir=state_dir,
        review_cwd=repo,
        base="main",
    )

    assert payload["recommendation"] == "review-followup"
    assert payload["reason"] == "dirty_worktree_after_review"
    assert payload["top_paths"][0].startswith("src/a b.txt ")


def test_inspect_workflow_status_recommends_full_review_when_merge_base_is_unresolvable(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    reviewed_head = _commit_file(repo, "app.txt", "one\n", "initial")
    record_review_anchor(
        state_dir=state_dir,
        review_cwd=repo,
        lane="review_t3",
        base="missing-base",
        reviewed_head=reviewed_head,
        review_scope={
            "base": "missing-base",
            "reviewed_head": reviewed_head,
            "merge_base": reviewed_head,
        },
    )

    payload = inspect_workflow_status(
        state_dir=state_dir,
        review_cwd=repo,
        base="missing-base",
    )

    assert payload["recommendation"] == "full-review"
    assert payload["reason"] == "base_merge_base_unresolvable"


def test_inspect_workflow_status_honors_requested_base_over_stored_base(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    base_at_review = _commit_file(repo, "base.txt", "one\n", "initial")
    _git(repo, "checkout", "-b", "feature/test")
    reviewed_head = _commit_file(repo, "feature.txt", "feat\n", "feature work")
    record_review_anchor(
        state_dir=state_dir,
        review_cwd=repo,
        lane="review_t3",
        base="main",
        reviewed_head=reviewed_head,
        review_scope={
            "base": "main",
            "reviewed_head": reviewed_head,
            "merge_base": base_at_review,
        },
    )
    _git(repo, "checkout", "-b", "release/x")
    _git(repo, "checkout", "feature/test")

    payload = inspect_workflow_status(
        state_dir=state_dir,
        review_cwd=repo,
        base="release/x",
    )

    assert payload["recommendation"] == "full-review"
    assert payload["reason"] == "base_merge_base_changed"


def test_inspect_workflow_status_uses_effective_base_when_requested_base_matches_metadata(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    old_main = _commit_file(repo, "base.txt", "old\n", "old main")
    new_main = _commit_file(repo, "base.txt", "old\nnew\n", "new upstream main")
    _git(repo, "update-ref", "refs/remotes/origin/main", new_main)
    _git(repo, "checkout", "-b", "feature/effective-base")
    _git(repo, "branch", "-f", "main", old_main)
    reviewed_head = _commit_file(repo, "feature.txt", "feat\n", "feature work")
    record_review_anchor(
        state_dir=state_dir,
        review_cwd=repo,
        lane="review_t3",
        base="origin/main",
        reviewed_head=reviewed_head,
        review_scope={
            "base": "origin/main",
            "requested_base": "main",
            "reviewed_head": reviewed_head,
            "merge_base": new_main,
        },
    )

    payload = inspect_workflow_status(
        state_dir=state_dir,
        review_cwd=repo,
        base="main",
    )

    assert payload["recommendation"] == "none"
    assert payload["current_merge_base"] == new_main


def test_inspect_workflow_status_uses_branch_base_for_followup_anchor_context(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    branch_base = _commit_file(repo, "base.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/followup-only")
    since_head = _commit_file(repo, "feature.txt", "feat\n", "reviewed")
    final_head = _commit_file(repo, "feature.txt", "feat\nfix\n", "fix")
    record_review_anchor(
        state_dir=state_dir,
        review_cwd=repo,
        lane="review-followup",
        base="main",
        reviewed_head=final_head,
        review_scope={
            "commit": since_head,
            "commit_end": final_head,
            "base": since_head,
            "branch_base": "main",
            "merge_base": branch_base,
        },
    )

    payload = inspect_workflow_status(
        state_dir=state_dir,
        review_cwd=repo,
        base="main",
    )

    assert payload["recommendation"] == "none"
    assert payload["current_merge_base"] == branch_base


def test_workflow_state_path_disambiguates_slug_colliding_branch_names(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    head = _commit_file(repo, "app.txt", "one\n", "initial")

    feature_slash = workflow_state_path(
        state_dir=state_dir,
        review_cwd=repo,
        branch="feature/a",
        head=head,
    )
    feature_dash = workflow_state_path(
        state_dir=state_dir,
        review_cwd=repo,
        branch="feature-a",
        head=head,
    )

    assert feature_slash != feature_dash


def test_branch_token_for_detached_head_includes_head_sha() -> None:
    assert branch_token(None, "abc123456789") == "detached-abc123456789"
    assert branch_token(None, "def987654321") == "detached-def987654321"


def test_inspect_workflow_status_preserves_detached_head_state_after_new_commit(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    reviewed_head = _commit_file(repo, "app.txt", "one\n", "initial")
    _git(repo, "checkout", reviewed_head)
    record_review_anchor(
        state_dir=state_dir,
        review_cwd=repo,
        lane="review_t3",
        base="main",
        reviewed_head=reviewed_head,
        review_scope={"base": "main", "reviewed_head": reviewed_head},
    )
    _commit_file(repo, "app.txt", "one\ntwo\n", "detached follow-up")

    payload = inspect_workflow_status(
        state_dir=state_dir,
        review_cwd=repo,
        base="main",
    )

    assert payload["recommendation"] == "review-followup"
    assert payload["last_reviewed_head"] == reviewed_head


def test_inspect_workflow_status_does_not_reuse_unrelated_detached_head_state(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    base_head = _commit_file(repo, "app.txt", "one\n", "initial")
    _git(repo, "checkout", base_head)
    detached_reviewed_head = _commit_file(
        repo, "app.txt", "one\ndetached\n", "detached review"
    )
    record_review_anchor(
        state_dir=state_dir,
        review_cwd=repo,
        lane="review_t3",
        base="main",
        reviewed_head=detached_reviewed_head,
        review_scope={"base": "main", "reviewed_head": detached_reviewed_head},
    )
    _git(repo, "checkout", "main")
    unrelated_head = _commit_file(repo, "main.txt", "main\n", "main follow-up")
    _git(repo, "checkout", unrelated_head)

    payload = inspect_workflow_status(
        state_dir=state_dir,
        review_cwd=repo,
        base="main",
    )

    assert payload["recommendation"] == "full-review"
    assert payload["reason"] == "no_review_anchor"


def test_record_review_anchor_uses_branch_scoped_lock(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    reviewed_head = _commit_file(repo, "app.txt", "one\n", "initial")
    entered: list[str] = []

    class _Lock:
        def __enter__(self) -> None:
            entered.append("enter")

        def __exit__(self, exc_type, exc, tb) -> None:
            entered.append("exit")

    monkeypatch.setattr(
        workflow_state_module, "workflow_state_lock", lambda **kwargs: _Lock()
    )

    record_review_anchor(
        state_dir=state_dir,
        review_cwd=repo,
        lane="review_t3",
        base="main",
        reviewed_head=reviewed_head,
        review_scope={"base": "main", "reviewed_head": reviewed_head},
    )

    assert entered == ["enter", "exit"]


def test_review_state_status_does_not_route_coherence_to_review_deslop(
    monkeypatch, tmp_path: Path
) -> None:
    emitted: list[dict[str, object]] = []

    monkeypatch.setattr(review_state, "resolve_repo_root", lambda cd: tmp_path)
    monkeypatch.setattr(
        review_state,
        "inspect_workflow_status",
        lambda **kwargs: {
            "status": "ok",
            "recommendation": "coherence-review",
        },
    )
    monkeypatch.setattr(
        review_state, "emit_toon", lambda payload: emitted.append(payload)
    )
    monkeypatch.setattr(sys, "argv", ["review.py", "--status", "--base", "main"])

    exit_code = review.main()

    assert exit_code == 0
    assert "lane" not in emitted[0]["Action"]
    assert "review.py" in str(emitted[0]["Action"]["cmd"])
    assert "--mode normal" in str(emitted[0]["Action"]["cmd"])
    assert "review-deslop" not in str(emitted[0]["Action"])


def test_review_state_status_routes_stage_full_review_lane(
    monkeypatch, tmp_path: Path
) -> None:
    emitted: list[dict[str, object]] = []
    state_dir = tmp_path / "state"

    monkeypatch.setattr(review_state, "resolve_repo_root", lambda cd: tmp_path)
    monkeypatch.setattr(
        review_state,
        "inspect_workflow_status",
        lambda **kwargs: {
            "status": "ok",
            "recommendation": "coherence-review",
            "recommended_lane": "review_t3",
        },
    )
    _use_default_state_dir(monkeypatch, state_dir)
    monkeypatch.setattr(
        review_state, "emit_toon", lambda payload: emitted.append(payload)
    )
    monkeypatch.setattr(sys, "argv", ["review.py", "--status", "--base", "main"])

    exit_code = review.main()

    assert exit_code == 0
    assert "lane" not in emitted[0]["Action"]
    assert "review.py" in str(emitted[0]["Action"]["cmd"])
    assert "--mode deep" in str(emitted[0]["Action"]["cmd"])
    assert "review_t3.py" not in str(emitted[0]["Action"])
    assert "review-deslop" not in str(emitted[0]["Action"])


def test_review_state_status_adds_followup_action(monkeypatch, tmp_path: Path) -> None:
    emitted: list[dict[str, object]] = []
    state_dir = tmp_path / "state"

    monkeypatch.setattr(review_state, "resolve_repo_root", lambda cd: tmp_path)
    monkeypatch.setattr(
        review_state,
        "inspect_workflow_status",
        lambda **kwargs: {
            "status": "ok",
            "recommendation": "review-followup",
            "last_reviewed_head": "abc123",
        },
    )
    _use_default_state_dir(monkeypatch, state_dir)
    monkeypatch.setattr(
        review_state, "emit_toon", lambda payload: emitted.append(payload)
    )
    monkeypatch.setattr(sys, "argv", ["review.py", "--status", "--base", "main"])

    exit_code = review.main()

    assert exit_code == 0
    assert "lane" not in emitted[0]["Action"]
    cmd = str(emitted[0]["Action"]["cmd"])
    assert "review_followup.py" in cmd
    assert "--base main" in cmd
    assert "--since abc123" in cmd
    assert "--note-file .review-suite/fix-note.md" in cmd
    assert "--state-dir" not in cmd
    assert "--cd" not in cmd


def test_review_state_status_routes_dirty_followup_to_commit_instruction(
    monkeypatch, tmp_path: Path
) -> None:
    emitted: list[dict[str, object]] = []
    state_dir = tmp_path / "state"

    monkeypatch.setattr(review_state, "resolve_repo_root", lambda cd: tmp_path)
    monkeypatch.setattr(
        review_state,
        "inspect_workflow_status",
        lambda **kwargs: {
            "status": "ok",
            "recommendation": "review-followup",
            "last_reviewed_head": "abc123",
            "worktree_dirty": True,
        },
    )
    _use_default_state_dir(monkeypatch, state_dir)
    monkeypatch.setattr(
        review_state, "emit_toon", lambda payload: emitted.append(payload)
    )
    monkeypatch.setattr(sys, "argv", ["review.py", "--status", "--base", "main"])

    exit_code = review.main()

    assert exit_code == 0
    assert "lane" not in emitted[0]["Action"]
    assert "cmd" not in emitted[0]["Action"]
    assert "Commit intended follow-up changes" in str(emitted[0]["Action"]["note"])


def test_review_state_status_ignores_legacy_pending_grade_for_caller(
    monkeypatch, tmp_path: Path
) -> None:
    emitted: list[dict[str, object]] = []
    state_dir = tmp_path / "state"

    monkeypatch.setattr(review_state, "resolve_repo_root", lambda cd: tmp_path)
    monkeypatch.setattr(
        review_state,
        "inspect_workflow_status",
        lambda **kwargs: {
            "status": "ok",
            "recommendation": "review-followup",
            "last_reviewed_head": "abc123",
        },
    )
    _use_default_state_dir(monkeypatch, state_dir)
    monkeypatch.setattr(
        review_state, "emit_toon", lambda payload: emitted.append(payload)
    )
    monkeypatch.setattr(sys, "argv", ["review.py", "--status", "--base", "main"])

    exit_code = review.main()

    assert exit_code == 0
    assert emitted[0]["recommendation"] == "review-followup"
    assert "lane" not in emitted[0]["Action"]
    assert "round_id" not in emitted[0]["Action"]
    assert emitted[0]["Action"]["cwd"] == str(tmp_path)
    assert "review_followup.py" in str(emitted[0]["Action"]["cmd"])
    assert "review_suite_arena.py" not in str(emitted[0]["Action"])
