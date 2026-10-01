from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from review_suite_core.orchestrator_state import (
    STAGE_CREATED,
    STAGE_FIX_PENDING,
    STAGE_REVIEW_GREEN,
    STAGE_RETRY_REQUESTED,
    abort_cycle,
    create_cycle,
    dismiss_recovery,
    mark_blocked,
    mark_crashed,
    mark_decision_pending,
    mark_deslop_closed,
    mark_deslop_done,
    mark_deslop_failed,
    mark_fix_detected,
    mark_followup_review_pending,
    mark_retry_requested,
    mark_running,
    mark_review_step_pending,
    mark_review_step_running,
    mark_review_step_retry,
    no_work_stage_is_idle,
    record_clean_decision,
    record_contract_conflict,
    record_contract_replan,
    record_convergence_decision,
    record_findings_decision,
    record_followup_clean,
    record_github_result,
)
from review_suite_core.orchestrator_store import (
    load_cycle_by_key,
    reserve_cycle_successor,
    save_cycle,
)


def _cycle(tmp_path: Path, *, mode: str = "normal", restart_token: str | None = None) -> dict[str, object]:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    return create_cycle(
        cwd=repo,
        base="main",
        branch="feature/orchestrator",
        head="head-1",
        merge_base="base-1",
        requested_mode=mode,
        effective_mode=mode,
        selection="auto",
        effective_selection="stable",
        restart_token=restart_token,
    )


@pytest.mark.parametrize("intervening_decision", [None, "RESLICE"])
def test_contract_replan_reservation_preserves_convergence_under_late_writes(
    tmp_path: Path, intervening_decision: str | None,
) -> None:
    state_dir = tmp_path / "state"
    source = record_findings_decision(
        mark_decision_pending(_cycle(tmp_path), round_id="old-round", lane="review_t1"),
        round_id="old-round", lane="review_t1", reviewed_head="head-1",
    )
    stale = save_cycle(state_dir, source)
    replanned = abort_cycle(record_contract_replan(stale, conflict="acceptance"), reason="approved")
    replanned["superseded_by"] = {"kind": "contract-replan", "reason": "approved"}
    successor = _cycle(tmp_path, restart_token="contract-replan")
    if intervening_decision:
        current = record_convergence_decision(
            record_contract_conflict(stale, conflict="scope"), decision=intervening_decision,
        )
        saved = save_cycle(state_dir, current)
        with pytest.raises(ValueError, match="contract replan requires active findings"):
            reserve_cycle_successor(state_dir, source=replanned, successor=successor)
        assert load_cycle_by_key(state_dir, stale["cycle_key"]) == saved
        assert load_cycle_by_key(state_dir, successor["cycle_key"]) is None
    else:
        reserve_cycle_successor(state_dir, source=replanned, successor=successor)
        saved = save_cycle(state_dir, stale)
        assert saved["convergence"] == replanned["convergence"]
        assert saved["convergence"]["decision"] == "REPLAN"
        assert saved["superseded_by"]["review"]
        assert saved["active_findings"] == stale["active_findings"]


def test_create_cycle_is_compact_json_state_keyed_by_normalized_inputs(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    state = create_cycle(
        cwd=repo,
        base="main",
        branch="feature/orchestrator",
        head="head-1",
        merge_base="base-1",
        requested_mode="normal",
        effective_mode="normal",
        selection="auto",
        effective_selection="stable",
    )
    same_state = create_cycle(
        cwd=repo / ".",
        base="main",
        branch="feature/orchestrator",
        head="head-1",
        merge_base="base-1",
        requested_mode="normal",
        effective_mode="normal",
        selection="auto",
        effective_selection="stable",
    )
    fast = create_cycle(
        cwd=repo,
        base="main",
        branch="HEAD",
        head="head-1",
        merge_base="base-1",
        requested_mode="fast",
        effective_mode="fast",
        selection="stable",
    )
    skipped = create_cycle(
        cwd=repo,
        base="main",
        branch="feature/no-deslop",
        head="head-1",
        merge_base="base-1",
        requested_mode="normal",
        effective_mode="normal",
        selection="stable",
        deslop_enabled=False,
    )

    assert json.loads(json.dumps(state)) == state
    assert state["cycle_key"] == same_state["cycle_key"]
    assert state["mode"] == {"requested": "normal", "effective": "normal"}
    assert state["selection"] == {"requested": "auto", "effective": "stable"}
    assert state["deslop"] == {"tracked": True, "status": "tracked"}
    assert state["validation"]["review_green"] == "unknown"
    assert state["validation"]["full_suite"] == "unknown"
    assert fast["identity"]["branch"] is None
    assert fast["deslop"] == {
        "tracked": False,
        "status": "skipped",
        "source": "profile",
    }
    assert skipped["deslop"] == {
        "tracked": False,
        "status": "skipped",
        "source": "profile",
    }


def test_create_cycle_preserves_exact_optional_review_brief(tmp_path: Path) -> None:
    brief = "# Goal\n\nKeep the provider-neutral contract.\n"

    briefed = create_cycle(
        cwd=tmp_path / "repo",
        base="main",
        branch="feature/brief",
        head="head-1",
        merge_base="base-1",
        requested_mode="fast",
        review_brief=brief,
    )
    briefless = create_cycle(
        cwd=tmp_path / "repo",
        base="main",
        branch="feature/brief",
        head="head-1",
        merge_base="base-1",
        requested_mode="fast",
    )

    assert briefed["review_brief"] == brief
    assert briefless["review_brief"] is None


def test_mark_deslop_closed_requires_completion_and_explicit_drift_dismissal(
    tmp_path: Path,
) -> None:
    tracked = _cycle(tmp_path)
    with pytest.raises(ValueError, match="only close after"):
        mark_deslop_closed(tracked, reason="not completed")

    done = mark_deslop_done(
        tracked,
        command="review-deslop",
        conformance="CONFORMS",
        reviewed_head="head-1",
    )
    closed = mark_deslop_closed(done)
    closed_again = mark_deslop_closed(closed)

    assert closed["deslop"]["status"] == "closed"
    assert closed_again == closed
    assert tracked["deslop"] == {"tracked": True, "status": "tracked"}

    drifted = mark_deslop_done(
        tracked,
        command="review-deslop",
        conformance="MATERIALLY_DRIFTED",
        reviewed_head="head-1",
        decision="findings",
        findings="Reviewer claims the implementation violates scope.",
    )
    with pytest.raises(ValueError, match="materially drifted"):
        mark_deslop_closed(drifted)
    with pytest.raises(ValueError, match="dismissal reason is required"):
        mark_deslop_closed(drifted, reason="  ")

    dismissed = mark_deslop_closed(drifted, reason="  The contract permits this.  ")
    assert dismissed["deslop"] == {
        **drifted["deslop"],
        "tracked": False,
        "status": "closed",
        "dismissal_reason": "The contract permits this.",
    }
    assert {**dismissed, "deslop": drifted["deslop"]} == drifted
    assert mark_deslop_closed(dismissed, reason="another reason") == dismissed


def test_failed_closure_cannot_be_skipped(tmp_path: Path) -> None:
    failed = mark_deslop_failed(
        _cycle(tmp_path), command="review-deslop", returncode=2, reason="deslop failed"
    )

    assert failed["stage"] == STAGE_RETRY_REQUESTED
    assert failed["pending_action"] == {"kind": "run-deslop"}
    with pytest.raises(ValueError, match="only close after"):
        mark_deslop_closed(failed, reason="not completed")


def test_wait_states_are_idle_and_transitions_are_idempotent(tmp_path: Path) -> None:
    state = _cycle(tmp_path)
    original = json.loads(json.dumps(state))

    running = mark_running(state, round_id="round-1", lane="review_t1")
    running_again = mark_running(running, round_id="round-1", lane="review_t1")
    pending = mark_decision_pending(
        running_again,
        round_id="round-1",
        lane="review_t1",
        pending_action={
            "kind": "reprint-decision",
            "round_id": "round-1",
            "commands": ["clean", "findings"],
        },
    )
    findings = record_findings_decision(pending, round_id="round-1", lane="review_t1")
    findings_again = record_findings_decision(
        findings, round_id="round-1", lane="review_t1"
    )
    blocked = mark_blocked(
        findings_again, reason="reviewer unavailable", round_id="round-1"
    )
    crashed = mark_crashed(findings_again, reason="transport error", round_id="round-1")

    assert state == original
    assert no_work_stage_is_idle(running)
    assert no_work_stage_is_idle(pending)
    assert no_work_stage_is_idle(findings)
    assert no_work_stage_is_idle(blocked)
    assert no_work_stage_is_idle(crashed)
    assert len(running_again["rounds"]) == 1
    assert pending["pending_action"] == {
        "kind": "reprint-decision",
        "round_id": "round-1",
        "commands": ["clean", "findings"],
    }
    assert len(findings_again["decisions"]) == 1
    assert findings_again["stage"] == STAGE_FIX_PENDING
    assert findings_again["pending_action"] is None

    retry = mark_retry_requested(crashed, reason="operator retry")
    dismissed = dismiss_recovery(crashed, reason="ignore stale crash")
    aborted = abort_cycle(findings_again, reason="operator abort")

    assert retry["recovery"]["status"] == "retry-requested"
    assert retry["pending_action"] is None
    assert dismissed["recovery"]["status"] == "dismissed"
    assert dismissed["pending_action"] is None
    assert aborted["recovery"]["status"] == "aborted"
    assert aborted["pending_action"] is None


def test_clean_profile_steps_record_progress_until_final_review_green(
    tmp_path: Path,
) -> None:
    state = _cycle(tmp_path)
    state["review_plan"] = {
        "steps": [
            {
                "name": "broad-discovery",
                "count": 1,
                "model": "gpt-5.4",
                "reasoning_effort": "medium",
            },
            {
                "name": "precision-signoff",
                "count": 1,
                "model": "gpt-5.5",
                "reasoning_effort": "medium",
            },
        ]
    }
    first_pending = mark_review_step_pending(
        state,
        round_id="round-1",
        lane="review_t1",
        step_index=0,
        step_name="broad-discovery",
        reviewed_head="head-1",
    )

    assert first_pending["review_progress"]["current_step"] == {
        "index": 0,
        "name": "broad-discovery",
        "round_id": "round-1",
        "lane": "review_t1",
    }
    assert first_pending["rounds"][0]["profile_step"] == {
        "index": 0,
        "name": "broad-discovery",
    }

    queued = record_clean_decision(
        first_pending, round_id="round-1", lane="review_t1", reviewed_head="head-1"
    )

    assert queued["stage"] == "created"
    assert queued["pending_action"] == {
        "kind": "run-review-step",
        "step_index": 1,
        "step": "precision-signoff",
    }
    assert queued["review_progress"]["current_step"] is None
    assert queued["review_progress"]["next_step_index"] == 1
    assert queued["review_progress"]["completed_steps"] == [
        {
            "index": 0,
            "name": "broad-discovery",
            "round_id": "round-1",
            "lane": "review_t1",
            "reviewed_head": "head-1",
        }
    ]

    second_pending = mark_review_step_pending(
        queued,
        round_id="round-2",
        lane="review_t1",
        step_index=1,
        step_name="precision-signoff",
        reviewed_head="head-1",
    )
    green = record_clean_decision(
        second_pending, round_id="round-2", lane="review_t1", reviewed_head="head-1"
    )

    assert green["stage"] == STAGE_REVIEW_GREEN
    assert green["pending_action"] is None
    assert green["review_progress"]["next_step_index"] == 2
    assert [
        item["round_id"] for item in green["review_progress"]["completed_steps"]
    ] == ["round-1", "round-2"]


def test_clean_discovery_skips_remaining_discovery_loops_to_signoff(
    tmp_path: Path,
) -> None:
    state = _cycle(tmp_path)
    state["review_plan"] = {
        "steps": [
            {
                "name": "broad-discovery-1",
                "count": 1,
                "model": "gpt-5.4",
                "reasoning_effort": "medium",
            },
            {
                "name": "broad-discovery-2",
                "count": 1,
                "model": "gpt-5.4",
                "reasoning_effort": "medium",
            },
            {
                "name": "broad-discovery-3",
                "count": 1,
                "model": "gpt-5.4",
                "reasoning_effort": "medium",
            },
            {
                "name": "precision-signoff",
                "count": 1,
                "model": "gpt-5.5",
                "reasoning_effort": "medium",
            },
        ]
    }
    pending = mark_review_step_pending(
        state,
        round_id="round-1",
        lane="review_t1",
        step_index=0,
        step_name="broad-discovery-1",
        reviewed_head="head-1",
    )

    clean = record_clean_decision(
        pending, round_id="round-1", lane="review_t1", reviewed_head="head-1"
    )

    assert clean["stage"] == STAGE_CREATED
    assert clean["pending_action"] == {
        "kind": "run-review-step",
        "step_index": 3,
        "step": "precision-signoff",
    }
    assert clean["review_progress"]["next_step_index"] == 3
    assert [item["name"] for item in clean["review_progress"]["completed_steps"]] == [
        "broad-discovery-1"
    ]


def test_clean_deep_discovery_skips_remaining_deep_discovery_to_deep_signoff(
    tmp_path: Path,
) -> None:
    state = _cycle(tmp_path, mode="deep")
    state["review_plan"] = {
        "steps": [
            {
                "name": "broad-discovery-1",
                "count": 1,
                "model": "gpt-5.4",
                "reasoning_effort": "medium",
            },
            {
                "name": "precision-signoff",
                "count": 1,
                "model": "gpt-5.5",
                "reasoning_effort": "medium",
            },
            {
                "name": "deep-discovery-1",
                "count": 1,
                "model": "gpt-5.4",
                "reasoning_effort": "xhigh",
            },
            {
                "name": "deep-discovery-2",
                "count": 1,
                "model": "gpt-5.4",
                "reasoning_effort": "xhigh",
            },
            {
                "name": "deep-signoff",
                "count": 1,
                "model": "gpt-5.5",
                "reasoning_effort": "xhigh",
            },
        ]
    }
    pending = mark_review_step_pending(
        state,
        round_id="round-1",
        lane="review_t3",
        step_index=2,
        step_name="deep-discovery-1",
        reviewed_head="head-1",
    )

    clean = record_clean_decision(
        pending, round_id="round-1", lane="review_t3", reviewed_head="head-1"
    )

    assert clean["stage"] == STAGE_CREATED
    assert clean["pending_action"] == {
        "kind": "run-review-step",
        "step_index": 4,
        "step": "deep-signoff",
    }
    assert clean["review_progress"]["next_step_index"] == 4
    assert [item["name"] for item in clean["review_progress"]["completed_steps"]] == [
        "deep-discovery-1"
    ]


def test_clean_arena_round_skips_remaining_arena_rounds_to_next_stage(
    tmp_path: Path,
) -> None:
    state = _cycle(tmp_path)
    state["review_plan"] = {
        "steps": [
            {
                "name": "arena-discovery-1",
                "kind": "arena",
                "lane": "review_t1",
                "task_class": "phase_review",
            },
            {
                "name": "arena-discovery-2",
                "kind": "arena",
                "lane": "review_t1",
                "task_class": "phase_review",
            },
            {
                "name": "arena-pr-review",
                "kind": "arena",
                "lane": "review_t3",
                "task_class": "pr_review",
            },
            {
                "name": "broad-discovery",
                "count": 1,
                "model": "gpt-5.4",
                "reasoning_effort": "medium",
            },
            {
                "name": "precision-signoff",
                "count": 1,
                "model": "gpt-5.5",
                "reasoning_effort": "medium",
            },
        ]
    }
    pending = mark_review_step_pending(
        state,
        round_id="round-1",
        lane="review_t1",
        step_index=0,
        step_name="arena-discovery-1",
        reviewed_head="head-1",
        grading_required=True,
        arena_round=True,
    )

    clean = record_clean_decision(
        pending, round_id="round-1", lane="review_t1", reviewed_head="head-1"
    )

    assert clean["pending_action"] == {
        "kind": "run-review-step",
        "step_index": 2,
        "step": "arena-pr-review",
        "step_kind": "arena",
    }
    assert clean["review_progress"]["next_step_index"] == 2


def test_clean_arena_round_skips_terminal_arena_rounds(tmp_path: Path) -> None:
    state = _cycle(tmp_path)
    state["review_plan"] = {
        "steps": [
            {
                "name": f"arena-discovery-{index}",
                "kind": "arena",
                "lane": "review_t1",
                "task_class": "phase_review",
            }
            for index in (1, 2)
        ]
    }
    pending = mark_review_step_pending(
        state,
        round_id="round-1",
        lane="review_t1",
        step_index=0,
        step_name="arena-discovery-1",
        reviewed_head="head-1",
        grading_required=True,
        arena_round=True,
    )

    clean = record_clean_decision(
        pending, round_id="round-1", lane="review_t1", reviewed_head="head-1"
    )

    assert clean["stage"] == STAGE_REVIEW_GREEN
    assert clean["review_progress"]["next_step_index"] == 2


def test_followup_clean_completes_source_profile_step_and_continues(
    tmp_path: Path,
) -> None:
    state = _cycle(tmp_path, mode="deep")
    state["review_plan"] = {
        "steps": [
            {
                "name": "broad-discovery",
                "count": 1,
                "model": "gpt-5.4",
                "reasoning_effort": "medium",
            },
            {
                "name": "precision-signoff",
                "count": 1,
                "model": "gpt-5.5",
                "reasoning_effort": "medium",
            },
        ]
    }
    pending = mark_review_step_pending(
        state,
        round_id="round-1",
        lane="review_t1",
        step_index=0,
        step_name="broad-discovery",
        reviewed_head="head-1",
    )
    findings = record_findings_decision(
        pending, round_id="round-1", lane="review_t1", reviewed_head="head-1"
    )
    fixed = mark_fix_detected(findings, head="head-2")
    followup_pending = mark_followup_review_pending(
        fixed,
        round_id="followup-1",
        reviewed_head="head-2",
        source_round_id="round-1",
    )
    clean = record_followup_clean(
        followup_pending, round_id="followup-1", reviewed_head="head-2"
    )

    assert clean["stage"] == STAGE_CREATED
    assert clean["active_findings"] is None
    assert clean["validation"]["review_green"] == "unknown"
    assert clean["pending_action"] == {
        "kind": "run-review-step",
        "step_index": 1,
        "step": "precision-signoff",
    }
    assert clean["review_progress"]["completed_steps"] == [
        {
            "index": 0,
            "name": "broad-discovery",
            "round_id": "round-1",
            "lane": "review_t1",
            "reviewed_head": "head-2",
        }
    ]


def test_followup_clean_after_discovery_findings_keeps_discovery_budget(
    tmp_path: Path,
) -> None:
    state = _cycle(tmp_path, mode="deep")
    state["review_plan"] = {
        "steps": [
            {
                "name": "broad-discovery-1",
                "count": 1,
                "model": "gpt-5.4",
                "reasoning_effort": "medium",
            },
            {
                "name": "broad-discovery-2",
                "count": 1,
                "model": "gpt-5.4",
                "reasoning_effort": "medium",
            },
            {
                "name": "precision-signoff",
                "count": 1,
                "model": "gpt-5.5",
                "reasoning_effort": "medium",
            },
        ]
    }
    pending = mark_review_step_pending(
        state,
        round_id="round-1",
        lane="review_t1",
        step_index=0,
        step_name="broad-discovery-1",
        reviewed_head="head-1",
    )
    findings = record_findings_decision(
        pending, round_id="round-1", lane="review_t1", reviewed_head="head-1"
    )
    fixed = mark_fix_detected(findings, head="head-2")
    followup_pending = mark_followup_review_pending(
        fixed,
        round_id="followup-1",
        reviewed_head="head-2",
        source_round_id="round-1",
    )

    clean = record_followup_clean(
        followup_pending, round_id="followup-1", reviewed_head="head-2"
    )

    assert clean["pending_action"] == {
        "kind": "run-review-step",
        "step_index": 1,
        "step": "broad-discovery-2",
    }
    assert clean["review_progress"]["next_step_index"] == 1


def test_followup_clean_reruns_sticky_signoff_step(tmp_path: Path) -> None:
    state = _cycle(tmp_path, mode="deep")
    state["review_plan"] = {
        "steps": [
            {
                "name": "precision-signoff",
                "count": 1,
                "model": "gpt-5.5",
                "reasoning_effort": "medium",
                "rerun_on_findings": True,
            },
        ]
    }
    pending = mark_review_step_pending(
        state,
        round_id="signoff-1",
        lane="review_t1",
        step_index=0,
        step_name="precision-signoff",
        reviewed_head="head-1",
    )
    findings = record_findings_decision(
        pending, round_id="signoff-1", lane="review_t1", reviewed_head="head-1"
    )
    fixed = mark_fix_detected(findings, head="head-2")
    followup_pending = mark_followup_review_pending(
        fixed,
        round_id="followup-1",
        reviewed_head="head-2",
        source_round_id="signoff-1",
    )
    clean = record_followup_clean(
        followup_pending, round_id="followup-1", reviewed_head="head-2"
    )

    assert clean["stage"] == STAGE_CREATED
    assert clean["active_findings"] is None
    assert clean["validation"]["review_green"] == "unknown"
    assert clean["pending_action"] == {
        "kind": "run-review-step",
        "step_index": 0,
        "step": "precision-signoff",
    }
    assert clean["review_progress"]["next_step_index"] == 0
    assert clean["review_progress"]["completed_steps"] == []

    rerun_pending = mark_review_step_pending(
        clean,
        round_id="signoff-2",
        lane="review_t1",
        step_index=0,
        step_name="precision-signoff",
        reviewed_head="head-2",
    )
    green = record_clean_decision(
        rerun_pending, round_id="signoff-2", lane="review_t1", reviewed_head="head-2"
    )

    assert green["stage"] == STAGE_REVIEW_GREEN
    assert green["review_progress"]["completed_steps"] == [
        {
            "index": 0,
            "name": "precision-signoff",
            "round_id": "signoff-2",
            "lane": "review_t1",
            "reviewed_head": "head-2",
        }
    ]


def test_fast_terminal_findings_get_one_verification_round(tmp_path: Path) -> None:
    state = _cycle(tmp_path)
    state["review_plan"] = {
        "steps": [
            {
                "name": "fast-signoff",
                "count": 1,
                "model": "gpt-5.5",
                "reasoning_effort": "medium",
                "max_review_rounds": 2,
            },
        ]
    }
    pending = mark_review_step_pending(
        state,
        round_id="signoff-1",
        lane="review_t1",
        step_index=0,
        step_name="fast-signoff",
        reviewed_head="head-1",
    )
    findings = record_findings_decision(
        pending, round_id="signoff-1", lane="review_t1", reviewed_head="head-1"
    )

    fixed = mark_fix_detected(findings, head="head-2")

    assert fixed["stage"] == STAGE_CREATED
    assert fixed["active_findings"] is None
    assert fixed["validation"]["review_green"] == "unknown"
    assert fixed["pending_action"] == {
        "kind": "run-review-step",
        "step_index": 0,
        "step": "fast-signoff",
        "fix_verification": {
            "source_round_id": "signoff-1",
            "source_lane": "review_t1",
            "findings_reviewed_head": "head-1",
            "fix_head": "head-2",
        },
    }
    assert fixed["review_progress"]["next_step_index"] == 0
    assert fixed["review_progress"]["completed_steps"] == []
    assert fixed["review_heads"]["last_fix_head"] == "head-2"


def test_distinct_accepted_finding_trees_require_one_use_caller_decision(
    tmp_path: Path,
) -> None:
    state = _cycle(tmp_path)
    for round_id, head, tree in (
        ("round-1", "head-1", "tree-1"),
        ("arena-variant", "restacked-head-1", "tree-1"),
        ("round-2", "head-2", "tree-2"),
        ("round-3", "head-3", "tree-3"),
    ):
        if round_id == "round-3":
            clean = record_followup_clean(state, round_id=round_id, reviewed_head=head)
            assert clean["convergence"]["status"] == "ACTIVE"
            assert len(clean["convergence"]["accepted_findings_heads"]) == 2
        state = record_findings_decision(
            state,
            round_id=round_id,
            lane="review_t1",
            reviewed_head=head,
            reviewed_tree=tree,
        )

    assert state["stage"] == "decision-required"
    assert state["convergence"]["status"] == "DECISION_REQUIRED"
    assert state["convergence"]["reason"] == "budget_exhausted"
    assert len(state["convergence"]["accepted_findings_heads"]) == 3
    assert mark_fix_detected(state, head="head-4") == state

    continued = record_convergence_decision(state, decision="CONTINUE")
    assert continued["stage"] == STAGE_FIX_PENDING
    assert continued["convergence"]["continue_used"] is True

    exhausted = record_findings_decision(
        continued,
        round_id="round-4",
        lane="review_t1",
        reviewed_head="head-4",
        reviewed_tree="tree-4",
    )
    assert exhausted["stage"] == "decision-required"
    with pytest.raises(ValueError, match="available once"):
        record_convergence_decision(exhausted, decision="CONTINUE")
    resliced = record_convergence_decision(exhausted, decision="RESLICE")
    assert resliced["stage"] == "convergence-decided"
    assert resliced["convergence"]["status"] == "DECIDED"
    assert resliced["convergence"]["decision"] == "RESLICE"


@pytest.mark.parametrize("running", [False, True])
def test_continue_preserves_existing_reviewer_work(
    tmp_path: Path, running: bool
) -> None:
    state = _cycle(tmp_path)
    for index in range(3):
        state = record_findings_decision(
            state,
            round_id=f"round-{index}",
            lane="review_t1",
            reviewed_head=f"head-{index}",
        )
    # The old fix transition dispatched another round while convergence was stopped.
    state["active_findings"] = None
    mark_step = mark_review_step_running if running else mark_review_step_pending
    state = mark_step(
        state,
        round_id="round-4",
        lane="review_t1",
        step_index=0,
        step_name="precision-signoff",
        reviewed_head="head-4",
        grading_required=True,
        arena_round=True,
    )
    continued = record_convergence_decision(state, decision="CONTINUE")
    assert continued["stage"] == state["stage"]
    assert continued["pending_action"] == state["pending_action"]
    assert continued["rounds"] == state["rounds"]
    assert continued["decisions"] == state["decisions"]
    assert record_convergence_decision(continued, decision="CONTINUE") == continued


def test_blocked_retries_do_not_consume_accepted_findings_budget(
    tmp_path: Path,
) -> None:
    state = _cycle(tmp_path, mode="fast")
    state["review_plan"] = {
        "steps": [
            {
                "name": "fast-signoff",
                "count": 2,
                "model": "gpt-5.6-sol",
                "reasoning_effort": "medium",
                "max_review_rounds": 2,
            },
        ]
    }
    first_pending = mark_review_step_pending(
        state,
        round_id="signoff-1",
        lane="review_t1",
        step_index=0,
        step_name="fast-signoff",
        reviewed_head="head-1",
    )
    first_retry = mark_review_step_retry(
        first_pending, step_index=0, step_name="fast-signoff"
    )
    second_pending = mark_review_step_pending(
        first_retry,
        round_id="signoff-2",
        lane="review_t1",
        step_index=0,
        step_name="fast-signoff",
        reviewed_head="head-1",
    )

    retried = mark_review_step_retry(
        second_pending, step_index=0, step_name="fast-signoff"
    )

    assert retried["stage"] == STAGE_CREATED
    assert retried["pending_action"]["kind"] == "run-review-step"
    assert retried["convergence"]["accepted_findings_heads"] == []


def test_contract_conflict_requires_an_immediate_durable_decision(
    tmp_path: Path,
) -> None:
    findings = record_findings_decision(
        _cycle(tmp_path),
        round_id="round-1",
        lane="review_t1",
        reviewed_head="head-1",
    )

    conflict = record_contract_conflict(findings, conflict="scope")

    assert conflict["stage"] == "decision-required"
    assert conflict["convergence"]["reason"] == "contract_conflict"
    assert conflict["convergence"]["conflict"] == "scope"


def test_non_deep_discovery_findings_advance_without_extra_discovery(
    tmp_path: Path,
) -> None:
    state = _cycle(tmp_path)
    state["review_plan"] = {
        "steps": [
            {
                "name": "broad-discovery",
                "count": 1,
                "model": "gpt-5.4",
                "reasoning_effort": "medium",
            },
            {
                "name": "precision-signoff",
                "count": 1,
                "model": "gpt-5.5",
                "reasoning_effort": "medium",
            },
        ]
    }
    pending = mark_review_step_pending(
        state,
        round_id="round-1",
        lane="review_t1",
        step_index=0,
        step_name="broad-discovery",
        reviewed_head="head-1",
    )
    findings = record_findings_decision(
        pending, round_id="round-1", lane="review_t1", reviewed_head="head-1"
    )

    fixed = mark_fix_detected(findings, head="head-2")

    assert fixed["stage"] == STAGE_CREATED
    assert fixed["active_findings"] is None
    assert fixed["validation"]["review_green"] == "unknown"
    assert fixed["pending_action"] == {
        "kind": "run-review-step",
        "step_index": 1,
        "step": "precision-signoff",
        "fix_verification": {
            "source_round_id": "round-1",
            "source_lane": "review_t1",
            "findings_reviewed_head": "head-1",
            "fix_head": "head-2",
        },
    }
    assert fixed["review_progress"]["next_step_index"] == 1
    assert fixed["review_progress"]["completed_steps"] == [
        {
            "index": 0,
            "name": "broad-discovery",
            "round_id": "round-1",
            "lane": "review_t1",
            "reviewed_head": "head-1",
        }
    ]
    assert fixed["review_heads"]["last_fix_head"] == "head-2"


def test_non_deep_discovery_findings_consume_one_discovery_budget(
    tmp_path: Path,
) -> None:
    state = _cycle(tmp_path)
    state["review_plan"] = {
        "steps": [
            {
                "name": "broad-discovery-1",
                "count": 1,
                "model": "gpt-5.4",
                "reasoning_effort": "medium",
            },
            {
                "name": "broad-discovery-2",
                "count": 1,
                "model": "gpt-5.4",
                "reasoning_effort": "medium",
            },
            {
                "name": "precision-signoff",
                "count": 1,
                "model": "gpt-5.5",
                "reasoning_effort": "medium",
            },
        ]
    }
    pending = mark_review_step_pending(
        state,
        round_id="round-1",
        lane="review_t1",
        step_index=0,
        step_name="broad-discovery-1",
        reviewed_head="head-1",
    )
    findings = record_findings_decision(
        pending, round_id="round-1", lane="review_t1", reviewed_head="head-1"
    )

    fixed = mark_fix_detected(findings, head="head-2")

    assert fixed["stage"] == STAGE_CREATED
    assert fixed["pending_action"] == {
        "kind": "run-review-step",
        "step_index": 1,
        "step": "broad-discovery-2",
        "fix_verification": {
            "source_round_id": "round-1",
            "source_lane": "review_t1",
            "findings_reviewed_head": "head-1",
            "fix_head": "head-2",
        },
    }
    assert fixed["review_progress"]["completed_steps"] == [
        {
            "index": 0,
            "name": "broad-discovery-1",
            "round_id": "round-1",
            "lane": "review_t1",
            "reviewed_head": "head-1",
        }
    ]


def test_non_deep_arena_findings_consume_one_arena_budget(tmp_path: Path) -> None:
    state = _cycle(tmp_path)
    state["review_plan"] = {
        "steps": [
            {
                "name": "arena-discovery",
                "kind": "arena",
                "lane": "review_t3",
                "task_class": "phase_review",
            },
            {
                "name": "arena-discovery-2",
                "kind": "arena",
                "lane": "review_t3",
                "task_class": "phase_review",
            },
            {
                "name": "precision-signoff",
                "count": 1,
                "model": "gpt-5.5",
                "reasoning_effort": "medium",
            },
        ]
    }
    pending = mark_review_step_pending(
        state,
        round_id="arena-1",
        lane="review_t3",
        step_index=0,
        step_name="arena-discovery",
        reviewed_head="head-1",
        grading_required=True,
        arena_round=True,
    )
    findings = record_findings_decision(
        pending, round_id="arena-1", lane="review_t3", reviewed_head="head-1"
    )

    fixed = mark_fix_detected(findings, head="head-2")

    assert fixed["stage"] == STAGE_CREATED
    assert fixed["pending_action"] == {
        "kind": "run-review-step",
        "step_index": 1,
        "step": "arena-discovery-2",
        "step_kind": "arena",
        "fix_verification": {
            "source_round_id": "arena-1",
            "source_lane": "review_t3",
            "findings_reviewed_head": "head-1",
            "fix_head": "head-2",
        },
    }
    assert fixed["review_progress"]["completed_steps"] == [
        {
            "index": 0,
            "name": "arena-discovery",
            "round_id": "arena-1",
            "lane": "review_t3",
            "reviewed_head": "head-1",
            "arena_round": True,
        }
    ]


def test_non_deep_signoff_findings_rerun_until_clean(tmp_path: Path) -> None:
    state = _cycle(tmp_path)
    state["review_plan"] = {
        "steps": [
            {
                "name": "precision-signoff",
                "count": 1,
                "model": "gpt-5.5",
                "reasoning_effort": "medium",
                "rerun_on_findings": True,
            },
            {
                "name": "post-check",
                "count": 1,
                "model": "gpt-5.5",
                "reasoning_effort": "medium",
            },
        ]
    }
    pending = mark_review_step_pending(
        state,
        round_id="signoff-1",
        lane="review_t1",
        step_index=0,
        step_name="precision-signoff",
        reviewed_head="head-1",
    )
    findings = record_findings_decision(
        pending, round_id="signoff-1", lane="review_t1", reviewed_head="head-1"
    )

    fixed = mark_fix_detected(findings, head="head-2")

    assert fixed["stage"] == STAGE_CREATED
    assert fixed["pending_action"] == {
        "kind": "run-review-step",
        "step_index": 0,
        "step": "precision-signoff",
        "fix_verification": {
            "source_round_id": "signoff-1",
            "source_lane": "review_t1",
            "findings_reviewed_head": "head-1",
            "fix_head": "head-2",
        },
    }
    assert fixed["review_progress"]["next_step_index"] == 0
    assert fixed["review_progress"]["completed_steps"] == []


def test_post_findings_discovery_rerun_clean_skips_remaining_discovery_to_signoff(
    tmp_path: Path,
) -> None:
    state = _cycle(tmp_path)
    state["review_plan"] = {
        "steps": [
            {
                "name": "broad-discovery-1",
                "count": 1,
                "model": "gpt-5.4",
                "reasoning_effort": "medium",
            },
            {
                "name": "broad-discovery-2",
                "count": 1,
                "model": "gpt-5.4",
                "reasoning_effort": "medium",
            },
            {
                "name": "broad-discovery-3",
                "count": 1,
                "model": "gpt-5.4",
                "reasoning_effort": "medium",
            },
            {
                "name": "precision-signoff",
                "count": 1,
                "model": "gpt-5.5",
                "reasoning_effort": "medium",
            },
        ]
    }
    pending = mark_review_step_pending(
        state,
        round_id="round-1",
        lane="review_t1",
        step_index=0,
        step_name="broad-discovery-1",
        reviewed_head="head-1",
    )
    findings = record_findings_decision(
        pending, round_id="round-1", lane="review_t1", reviewed_head="head-1"
    )
    fixed = json.loads(json.dumps(findings))
    fixed["stage"] = STAGE_CREATED
    fixed["active_findings"] = None
    fixed["validation"]["review_green"] = "unknown"
    fixed["review_progress"]["next_step_index"] = 1
    fixed["review_progress"]["next_step_name"] = "broad-discovery-2"
    fixed["review_progress"]["current_step"] = None
    fixed["pending_action"] = {
        "kind": "run-review-step",
        "step_index": 1,
        "step": "broad-discovery-2",
        "fix_verification": {
            "source_round_id": "round-1",
            "source_lane": "review_t1",
            "findings_reviewed_head": "head-1",
            "fix_head": "head-2",
        },
    }
    running_rerun = mark_review_step_pending(
        fixed,
        round_id="round-2",
        lane="review_t1",
        step_index=1,
        step_name="broad-discovery-2",
        reviewed_head="head-2",
        post_findings_rerun=True,
    )

    clean = record_clean_decision(
        running_rerun, round_id="round-2", lane="review_t1", reviewed_head="head-2"
    )

    assert clean["stage"] == STAGE_CREATED
    assert clean["pending_action"] == {
        "kind": "run-review-step",
        "step_index": 3,
        "step": "precision-signoff",
    }
    assert clean["review_progress"]["completed_steps"] == [
        {
            "index": 1,
            "name": "broad-discovery-2",
            "round_id": "round-2",
            "lane": "review_t1",
            "reviewed_head": "head-2",
        }
    ]


def test_github_findings_followup_reruns_last_local_signoff_step(
    tmp_path: Path,
) -> None:
    state = _cycle(tmp_path, mode="deep")
    state["review_plan"] = {
        "steps": [
            {
                "name": "precision-signoff",
                "count": 1,
                "model": "gpt-5.5",
                "reasoning_effort": "medium",
                "rerun_on_findings": True,
            },
        ]
    }
    pending = mark_review_step_pending(
        state,
        round_id="signoff-1",
        lane="review_t1",
        step_index=0,
        step_name="precision-signoff",
        reviewed_head="head-1",
    )
    green = record_clean_decision(
        pending, round_id="signoff-1", lane="review_t1", reviewed_head="head-1"
    )

    github_findings = record_github_result(
        green,
        result="findings",
        note="GitHub found a boundary case.",
        reviewed_head="head-1",
    )

    assert github_findings["stage"] == STAGE_FIX_PENDING
    assert github_findings["validation"]["review_green"] == "unknown"
    assert github_findings["github_review"]["status"] == "findings"
    assert github_findings["active_findings"] == {
        "round_id": "github-review-1",
        "lane": "review-github",
        "reviewed_head": "head-1",
        "status": STAGE_FIX_PENDING,
        "profile_round_id": "signoff-1",
        "rerun_profile_round": True,
        "note": "GitHub found a boundary case.",
    }

    fixed = mark_fix_detected(github_findings, head="head-2")
    followup_pending = mark_followup_review_pending(
        fixed,
        round_id="followup-1",
        reviewed_head="head-2",
        source_round_id="github-review-1",
    )
    clean = record_followup_clean(
        followup_pending, round_id="followup-1", reviewed_head="head-2"
    )

    assert clean["stage"] == STAGE_CREATED
    assert clean["active_findings"] is None
    assert clean["pending_action"] == {
        "kind": "run-review-step",
        "step_index": 0,
        "step": "precision-signoff",
    }
    assert clean["review_progress"]["completed_steps"] == []


def test_github_result_defaults_to_latest_local_reviewed_head(tmp_path: Path) -> None:
    state = _cycle(tmp_path)
    state["review_plan"] = {
        "steps": [
            {
                "name": "precision-signoff",
                "count": 1,
                "model": "gpt-5.5",
                "reasoning_effort": "medium",
                "rerun_on_findings": True,
            },
        ]
    }
    pending = mark_review_step_pending(
        state,
        round_id="signoff-1",
        lane="review_t1",
        step_index=0,
        step_name="precision-signoff",
        reviewed_head="head-2",
    )
    green = record_clean_decision(
        pending, round_id="signoff-1", lane="review_t1", reviewed_head="head-2"
    )

    github_findings = record_github_result(green, result="findings")

    assert github_findings["github_review"]["reviewed_head"] == "head-2"
    assert github_findings["active_findings"]["reviewed_head"] == "head-2"


def test_github_findings_force_signoff_rerun_for_non_sticky_profile(
    tmp_path: Path,
) -> None:
    state = _cycle(tmp_path, mode="deep")
    state["review_plan"] = {
        "steps": [
            {
                "name": "custom-signoff",
                "count": 1,
                "model": "gpt-5.5",
                "reasoning_effort": "medium",
            },
        ]
    }
    pending = mark_review_step_pending(
        state,
        round_id="signoff-1",
        lane="review_t1",
        step_index=0,
        step_name="custom-signoff",
        reviewed_head="head-1",
    )
    green = record_clean_decision(
        pending, round_id="signoff-1", lane="review_t1", reviewed_head="head-1"
    )
    github_findings = record_github_result(
        green, result="findings", reviewed_head="head-1"
    )
    fixed = mark_fix_detected(github_findings, head="head-2")
    followup_pending = mark_followup_review_pending(
        fixed,
        round_id="followup-1",
        reviewed_head="head-2",
        source_round_id="github-review-1",
    )

    clean = record_followup_clean(
        followup_pending, round_id="followup-1", reviewed_head="head-2"
    )

    assert clean["stage"] == STAGE_CREATED
    assert clean["pending_action"] == {
        "kind": "run-review-step",
        "step_index": 0,
        "step": "custom-signoff",
    }
    assert clean["review_progress"]["completed_steps"] == []


def test_github_findings_anchor_to_latest_completed_step_not_older_sticky_step(
    tmp_path: Path,
) -> None:
    state = _cycle(tmp_path, mode="deep")
    state["review_plan"] = {
        "steps": [
            {
                "name": "early-sticky",
                "count": 1,
                "model": "gpt-5.5",
                "reasoning_effort": "medium",
                "rerun_on_findings": True,
            },
            {
                "name": "final-signoff",
                "count": 1,
                "model": "gpt-5.5",
                "reasoning_effort": "medium",
            },
        ]
    }
    first_pending = mark_review_step_pending(
        state,
        round_id="early-1",
        lane="review_t1",
        step_index=0,
        step_name="early-sticky",
        reviewed_head="head-1",
    )
    queued = record_clean_decision(
        first_pending, round_id="early-1", lane="review_t1", reviewed_head="head-1"
    )
    final_pending = mark_review_step_pending(
        queued,
        round_id="final-1",
        lane="review_t1",
        step_index=1,
        step_name="final-signoff",
        reviewed_head="head-1",
    )
    green = record_clean_decision(
        final_pending, round_id="final-1", lane="review_t1", reviewed_head="head-1"
    )

    github_findings = record_github_result(
        green, result="findings", reviewed_head="head-1"
    )

    assert github_findings["active_findings"]["profile_round_id"] == "final-1"
    fixed = mark_fix_detected(github_findings, head="head-2")
    followup_pending = mark_followup_review_pending(
        fixed,
        round_id="followup-1",
        reviewed_head="head-2",
        source_round_id="github-review-1",
    )
    clean = record_followup_clean(
        followup_pending, round_id="followup-1", reviewed_head="head-2"
    )

    assert clean["pending_action"] == {
        "kind": "run-review-step",
        "step_index": 1,
        "step": "final-signoff",
    }
    assert clean["review_progress"]["completed_steps"] == [
        {
            "index": 0,
            "name": "early-sticky",
            "round_id": "early-1",
            "lane": "review_t1",
            "reviewed_head": "head-1",
        }
    ]
