from __future__ import annotations

import json
from argparse import ArgumentError, ArgumentParser
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

from typing import Any

from .round_store import load_round, unique_round_state_dirs
from .cycle_store import UnknownReviewIdError, load_cycle_by_public_id
from .state_locations import read_state_dir


STAGE_REVIEW_GREEN = "review-green"
STAGE_LOCAL_GREEN_HANDOFF = "local-green-handoff"
DECISION_CLEAN = "clean"
DECISION_FINDINGS = "findings"
DECISION_COMMANDS = {DECISION_CLEAN, DECISION_FINDINGS}
CONVERGENCE_DEFAULTS = {
    "status": "ACTIVE",
    "accepted_findings_heads": [],
    "continue_used": False,
    "decisions": [],
}
GITHUB_RESULT_CLEAN = "clean"
GITHUB_RESULT_WAIVED = "waived"
DESLOP_STATUS_DONE = "done"
DESLOP_STATUS_CLOSED = "closed"
VALIDATION_READY_STATUSES = {"passed", "waived"}
HEAD_CHANGED_AFTER_GREEN_REVIEW_LADDER = "head_changed_after_review"
HEAD_CHANGED_AFTER_GREEN_REVIEW_NOTE = (
    "The PR patch changed. Continue the same review id to review only what changed "
    "and check concrete interactions with base changes. Keep required current-head checks."
)
CHANGED_SINCE_REVIEW_PATH_LIMIT = 20
PATCH_RECEIPT_VERSION = 1


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _base_drift_equivalent_head(state: dict[str, Any], head: str) -> str:
    # Historical receipts always name the commit actually inspected.
    return head


def _review_ladder_heads(state: dict[str, Any]) -> list[str]:
    review_heads = dict(state.get("review_heads") or {})
    heads = [str(review_heads[key]) for key in ("last_followup_head", "last_reviewed_head") if review_heads.get(key)]
    if not heads:
        head = str(review_heads.get("head") or dict(state.get("identity") or {}).get("head") or "")
        if head:
            heads.append(head)
    return heads


def _review_ladder_head(state: dict[str, Any], *, current_head: str = "") -> str:
    heads = _review_ladder_heads(state)
    if current_head:
        for head in heads:
            if head == current_head:
                return head
    return heads[0] if heads else ""


def _github_review_required(state: dict[str, Any]) -> bool:
    github_status = str(
        dict(state.get("github_review") or {}).get("status") or "unknown"
    ).strip()
    return not (_effective_mode(state) == "fast" and github_status == "unknown")


def _github_review_matches_head(state: dict[str, Any], comparison_head: str) -> bool:
    github_review = dict(state.get("github_review") or {})
    if str(github_review.get("status") or "").strip() not in {
        GITHUB_RESULT_CLEAN,
        GITHUB_RESULT_WAIVED,
    }:
        return False
    reviewed_head = _base_drift_equivalent_head(
        state, str(github_review.get("reviewed_head") or "").strip()
    )
    return bool(reviewed_head and comparison_head and reviewed_head == comparison_head)


def validation_blockers(
    state: dict[str, Any], *, current_head: str | None = None
) -> list[str]:
    validation = dict(state.get("validation") or {})
    note = _optional_text(validation.get("note"))
    blockers: list[str] = []
    identity = dict(state.get("identity") or {})
    checked_head = validation.get("head") or dict(state.get("review_patch") or {}).get("reviewed_head") or identity.get("head")
    comparison_head = identity.get("head") if current_head is None else current_head
    if current_head is None and state.get("review_patch") and identity.get("cwd"):
        from .paths import cwd_path_from_normalized
        from .workflow_state import current_head as read_head

        try:
            comparison_head = read_head(cwd_path_from_normalized(identity["cwd"]))
        except (OSError, ValueError):
            pass
    for key in ("full_suite", "ci"):
        value = str(validation.get(key) or "unknown").strip() or "unknown"
        if comparison_head and checked_head and comparison_head != checked_head:
            value = "pending"
        if value not in VALIDATION_READY_STATUSES:
            blockers.append(f"{key}:{value}")
        elif value == "waived" and not note:
            blockers.append(f"{key}:waived_without_note")
    return blockers


def _deslop_closed_or_untracked(state: dict[str, Any]) -> bool:
    deslop = dict(state.get("deslop") or {})
    return (
        not bool(deslop.get("tracked"))
        or str(deslop.get("status") or "").strip() == DESLOP_STATUS_CLOSED
    )


def review_ladder_summary(
    state: dict[str, Any], *, current_head: str | None = None, recorded: bool = False,
) -> dict[str, Any]:
    if recorded:
        accepted = recorded_acceptance(state)
        current_head = (
            accepted["accepted_head"] if accepted else (state.get("identity") or {}).get("head")
        )
    else:
        accepted = None

    stage = str(state.get("stage") or "").strip()
    original_head = str(dict(state.get("review_heads") or {}).get("last_reviewed_head") or dict(state.get("identity") or {}).get("head") or "")
    summary: dict[str, Any] = {"done": False, "review_ladder": "pending", "reviewed_head": original_head}
    if current_head:
        summary["current_head"] = current_head
    if stage not in {STAGE_REVIEW_GREEN, STAGE_LOCAL_GREEN_HANDOFF}:
        return summary

    comparison_head = str(current_head or "").strip()
    if not recorded:
        from .patch_identity import acceptance

        accepted = acceptance(state, comparison_head) if comparison_head else None
    validation_head = comparison_head if recorded else None
    if accepted and convergence_summary(state).get("status") == "ACTIVE" and not state.get("active_findings"):
        original = accepted["reviewed_head"]
        # Patch acceptance binds local signoff independently of unfinished external gates.
        # Keep every historical receipt on the commit that was actually inspected.
        github_required = _github_review_required(state)
        github_review = dict(state.get("github_review") or {})
        github_terminal = github_review.get("status") in {
            GITHUB_RESULT_CLEAN, GITHUB_RESULT_WAIVED,
        }
        github_head = str(github_review.get("reviewed_head") or "")
        github_ready = github_terminal and bool(
            accepted if github_head == comparison_head else (
                recorded_acceptance(state, github_head) if recorded else acceptance(state, github_head)
            )
        )
        if recorded and github_terminal and accepted == state.get("accepted_review"):
            github_ready = github_ready or (
                bool(github_head) and github_head == state.get("accepted_review_github_head")
            )
        equivalent_github = github_ready and not _github_review_matches_head(state, comparison_head)
        if (comparison_head != original or equivalent_github) and (not github_terminal or github_ready):
            base_interaction = bound_interaction(state, accepted, recorded=recorded)
            ready = (
                (not github_required or github_ready)
                and _deslop_closed_or_untracked(state)
                and not validation_blockers(state, current_head=validation_head)
                and not base_interaction["check"]
            )
            return {**summary, "done": ready, "review_ladder": "complete" if ready else "pending",
                    "reviewed_head": original, "current_head": comparison_head,
                    "accepted_head": comparison_head, "acceptance": accepted,
                    "required_checks": "current_head_required",
                    "validation_head": dict(state.get("validation") or {}).get("head") or original,
                    "base_interaction": base_interaction}
    if comparison_head == original_head and state.get("review_patch") and not accepted:
        receipt = dict(state["review_patch"])
        if recorded:
            live_base = (state.get("identity") or {}).get("merge_base")
            if live_base != receipt["reviewed_base"]:
                return {**summary, "review_ladder": "invalidated"}
        else:
            from .workflow_state import merge_base
            try:
                live_base = merge_base(Path(receipt["repo"]), receipt["base"], comparison_head)
                if live_base != receipt["reviewed_base"]:
                    return {**summary, "review_ladder": "invalidated"}
            except (OSError, ValueError):
                return summary
    reviewed_head = _review_ladder_head(state, current_head=comparison_head)
    comparison_head = comparison_head or reviewed_head
    if reviewed_head and comparison_head and reviewed_head != comparison_head:
        summary.update(
            {
                "review_ladder": "invalidated",
                "reviewed_head": reviewed_head,
                "current_head": comparison_head,
            }
        )
        return summary

    github_review = dict(state.get("github_review") or {})
    github_status = str(github_review.get("status") or "").strip()
    github_head = _base_drift_equivalent_head(
        state, str(github_review.get("reviewed_head") or "").strip()
    )
    if (
        github_status in {GITHUB_RESULT_CLEAN, GITHUB_RESULT_WAIVED}
        and github_head
        and comparison_head
        and github_head != comparison_head
    ):
        summary.update(
            {
                "review_ladder": "invalidated",
                "reviewed_head": github_head,
                "current_head": comparison_head,
            }
        )
        return summary

    github_required = _github_review_required(state)
    if not github_required and _deslop_closed_or_untracked(state):
        summary.update({"done": True, "review_ladder": "complete", "accepted_head": comparison_head})
        return summary

    github_ready = _github_review_matches_head(state, comparison_head)
    if github_ready and not validation_blockers(state, current_head=validation_head) and _deslop_closed_or_untracked(state):
        summary.update({"done": True, "review_ladder": "complete", "accepted_head": comparison_head})
    return summary


def _changed_since_review_paths(
    state: dict[str, Any],
    *,
    reviewed_head: str,
    current_head: str,
) -> list[str] | None:
    from .paths import cwd_path_from_normalized
    from .workflow_state import diff_paths_between

    cwd = str(dict(state.get("identity") or {}).get("cwd") or "").strip()
    if not cwd:
        return None
    try:
        return sorted(
            diff_paths_between(
                cwd_path_from_normalized(cwd), reviewed_head, current_head
            )
        )
    except OSError, ValueError:
        return None


def green_review_head_change_summary(
    state: dict[str, Any],
    *,
    current_head: str | None = None,
    summary: dict[str, Any] | None = None,
    include_paths: bool = True,
) -> dict[str, Any] | None:
    base_summary = dict(
        summary or review_ladder_summary(state, current_head=current_head)
    )
    if base_summary.get("review_ladder") != "invalidated":
        return None
    if str(state.get("stage") or "").strip() not in {
        STAGE_REVIEW_GREEN,
        STAGE_LOCAL_GREEN_HANDOFF,
    }:
        return None
    if not _deslop_closed_or_untracked(state):
        return None

    reviewed_head = str(base_summary.get("reviewed_head") or "").strip()
    current_head_value = str(
        base_summary.get("current_head") or current_head or ""
    ).strip()
    if (
        not reviewed_head
        or not current_head_value
        or reviewed_head == current_head_value
    ):
        return None

    github_review = dict(state.get("github_review") or {})
    github_status = str(github_review.get("status") or "unknown").strip() or "unknown"
    github_reviewed_head = _base_drift_equivalent_head(
        state, str(github_review.get("reviewed_head") or "").strip()
    )
    github_ready = (
        github_status in {GITHUB_RESULT_CLEAN, GITHUB_RESULT_WAIVED}
        and github_reviewed_head == reviewed_head
    )
    github_optional = _effective_mode(state) == "fast" and github_status == "unknown"
    if not (github_ready or github_optional):
        return None

    changed_summary = {
        **base_summary,
        "done": False,
        "review_ladder": HEAD_CHANGED_AFTER_GREEN_REVIEW_LADDER,
        "head_changed_after_review": True,
        "note": HEAD_CHANGED_AFTER_GREEN_REVIEW_NOTE,
    }
    paths = (
        _changed_since_review_paths(
            state, reviewed_head=reviewed_head, current_head=current_head_value
        ) if include_paths else None
    )
    if paths is not None:
        changed_summary["changed_since_review"] = paths[
            :CHANGED_SINCE_REVIEW_PATH_LIMIT
        ]
        if len(paths) > CHANGED_SINCE_REVIEW_PATH_LIMIT:
            changed_summary["changed_since_review_count"] = len(paths)
    return changed_summary


def _convergence(state: dict[str, Any]) -> dict[str, Any]:
    return state.setdefault("convergence", deepcopy(CONVERGENCE_DEFAULTS))


CONVERGENCE_DECISIONS = {"CONTINUE", "REPLAN", "RESLICE"}


DEFAULT_ACCEPTED_FINDINGS_LIMIT = 3


def _compact(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in payload.items() if value not in (None, "", [], {})
    }


def _effective_findings_heads(state: dict[str, Any]) -> list[dict[str, Any]]:
    corrected = {
        item["accepted_findings_index"]
        for item in (
            state.get("decision_corrections", [])
            + state.get("finding_dispositions", [])
        )
    }
    return [
        item
        for index, item in enumerate(
            dict(state.get("convergence") or {}).get("accepted_findings_heads") or []
        )
        if index not in corrected
    ]


def convergence_summary(state: dict[str, Any]) -> dict[str, Any]:
    convergence = dict(state.get("convergence") or {})
    return _compact(
        {
            "status": convergence.get("status") or "ACTIVE",
            "reason": convergence.get("reason"),
            "conflict": convergence.get("conflict"),
            "decision": convergence.get("decision"),
            "accepted_findings_heads": len(_effective_findings_heads(state)),
            "accepted_findings_limit": DEFAULT_ACCEPTED_FINDINGS_LIMIT,
            "recommendations": sorted(
                CONVERGENCE_DECISIONS
                - ({"CONTINUE"} if convergence.get("continue_used") else set())
            ),
        }
    )


def convergence_restart_decision(state: dict[str, Any]) -> str | None:
    convergence = state.get("convergence") or {}
    if state.get("stage") != "convergence-decided" or convergence.get("status") != "DECIDED":
        return None
    decision = convergence.get("decision")
    if decision == "RESLICE" or (
        decision == "REPLAN"
        and convergence.get("reason") == "budget_exhausted"
        and state.get("active_findings")
    ):
        return decision
    return None


def _effective_mode(state: dict[str, Any]) -> str:
    mode = dict(state.get("mode") or {})
    return str(mode.get("effective") or mode.get("requested") or "").strip()


def recorded_acceptance(
    state: dict[str, Any], head: str | None = None
) -> dict[str, Any] | None:
    identity = state.get("identity") or {}
    receipt = state.get("review_patch") or {}
    reviewed_head = receipt.get("reviewed_head")
    reviewed_base = receipt.get("reviewed_base")
    if (
        not reviewed_head or not reviewed_base or reviewed_head == reviewed_base
        or receipt.get("base") != identity.get("base")
        or (receipt.get("patch") or {}).get("version") != PATCH_RECEIPT_VERSION
    ):
        return None
    saved = state.get("accepted_review") or {}
    if saved and all(saved.get(key) == value for key, value in receipt.items()):
        if head is None or saved["accepted_head"] == head:
            return saved

    recorded_head = identity.get("head")
    # Previous versions saved a verified binding when recording a base check.
    for key in ("base_interaction_pending", "base_interaction_receipt"):
        binding = (state.get(key) or {}).get("binding") or {}
        if (
            binding.get("patch") == receipt["patch"]
            and binding.get("head") and binding.get("base")
            and (
                binding["head"] == head if head is not None
                else recorded_head in {reviewed_head, binding["head"]}
            )
        ):
            return {**receipt, "accepted_head": binding["head"], "accepted_base": binding["base"]}

    if (head or recorded_head) == reviewed_head and (
        recorded_head != reviewed_head or identity.get("merge_base") == reviewed_base
    ):
        return {**receipt, "accepted_head": reviewed_head, "accepted_base": reviewed_base}
    return None


def _no_base_drift() -> dict[str, Any]:
    return {"check": False, "reason": "no_base_drift", "paths": [], "line_overlap_paths": []}


def bound_interaction(
    state: dict[str, Any], accepted: dict[str, Any], *, recorded: bool = False,
) -> dict[str, Any]:
    if accepted["reviewed_base"] == accepted["accepted_base"]:
        result = _no_base_drift()
    elif recorded:
        # Older carry-forward records may lack scope. A matching receipt still
        # clears that binding. Otherwise a check is required before signoff.
        saved_scope = state.get("accepted_review_scope") if state.get("accepted_review") == accepted else None
        result = dict(saved_scope) if saved_scope else {"check": True}
    else:
        from .patch_identity import _interaction_scope

        result = _interaction_scope(accepted)
    binding = {
        "head": accepted["accepted_head"],
        "base": accepted["accepted_base"],
        "patch": accepted["patch"],
    }
    receipt = dict(state.get("base_interaction_receipt") or {})
    if receipt.get("binding") == binding:
        result.update(check=False, status=receipt["status"])
        if receipt["status"] == "overridden":
            result["override"] = {
                "reason": receipt["reason"],
                "head": binding["head"],
                "base": binding["base"],
            }
        else:
            result["receipt"] = receipt
    else:
        result["status"] = "outstanding" if result["check"] else "not_required"
    return result


def _public_status_action(action: dict[str, object]) -> dict[str, object]:
    hidden = {"lane", "round_id"}
    return {
        key: value
        for key, value in action.items()
        if key not in hidden and value not in (None, "", [], {})
    }


def _public_status_payload(payload: dict[str, object]) -> dict[str, object]:
    public: dict[str, object] = {}
    for key in (
        "review",
        "status",
        "done",
        "review_ladder",
        "next_action",
        "recommendation",
        "reason",
        "progress",
        "convergence",
        "base_interaction",
    ):
        value = payload.get(key)
        if value not in (None, "", [], {}):
            public[key] = value

    if payload.get("pending_round_head_matches_current") is False:
        reviewed_head = str(payload.get("pending_round_reviewed_head") or "").strip()
        current_head = str(payload.get("pending_round_current_head") or "").strip()
        if reviewed_head:
            public["reviewed_head"] = reviewed_head
        if current_head:
            public["current_head"] = current_head
        note = str(payload.get("note") or "").strip()
        if note:
            public["note"] = note

    if payload.get("head_changed_after_review"):
        for key in (
            "head_changed_after_review",
            "reviewed_head",
            "current_head",
            "changed_since_review",
            "changed_since_review_count",
            "note",
        ):
            value = payload.get(key)
            if value not in (None, "", [], {}):
                public[key] = value

    action = payload.get("Action")
    if isinstance(action, dict) and action:
        public["Action"] = _public_status_action(action)
    elif str(payload.get("note") or "").strip():
        public["note"] = str(payload.get("note") or "").strip()

    return public or payload


def json_status_payload(payload: dict[str, object]) -> dict[str, object]:
    public = _public_status_payload(payload)
    public.pop("Action", None)
    for key in ("base", "branch", "head", "mode", "reviewed_head", "accepted_head", "current_head", "acceptance", "required_checks", "validation_head", "base_interaction"):
        value = payload.get(key)
        if value not in (None, "", [], {}):
            public[key] = value
    for key in ("steps", "rounds", "round_history", "started_at", "round_started_at"):
        if key in payload:
            public[key] = payload[key]
    return public


def route_convergence(
    payload: dict[str, object], convergence: dict[str, object], *, state: dict[str, Any],
) -> None:
    """Report a non-active convergence checkpoint as the cycle's status."""
    if convergence.get("status") == "DECISION_REQUIRED":
        payload["status"] = "decision_required"
        payload["next_action"] = "caller_decision"
    else:
        payload["status"] = str(convergence.get("decision") or "decided").lower()
        payload["next_action"] = "restart_brief" if convergence_restart_decision(state) else "none"


def _orchestrator_review_state_dir(state_dir: Path) -> Path:
    return state_dir / "orchestrator" / "review-rounds"


def _round_state_dir_candidates(
    state_dir: Path, round_record: dict[str, object]
) -> list[Path]:
    candidates: list[Path] = []
    round_state_dir = str(round_record.get("round_state_dir") or "").strip()
    if round_state_dir:
        candidates.append(Path(round_state_dir))
    candidates.extend([_orchestrator_review_state_dir(state_dir), state_dir])
    return unique_round_state_dirs(candidates)


def _fallback_round_payload(round_record: dict[str, object]) -> dict[str, object]:
    payload = dict(round_record)
    runs: list[dict[str, object]] = []
    for raw_run in list(payload.get("runs") or []):
        if not isinstance(raw_run, dict):
            continue
        run = dict(raw_run)
        if (
            not str(run.get("review_status") or "").strip()
            and str(run.get("status") or "").strip()
        ):
            run["review_status"] = run.get("status")
        if (
            not str(run.get("reviewer_output") or "").strip()
            and str(run.get("summary") or "").strip()
        ):
            run["reviewer_output"] = run.get("summary")
        if (
            not str(run.get("reviewer_output_ref") or "").strip()
            and str(run.get("ref") or "").strip()
        ):
            run["reviewer_output_ref"] = run.get("ref")
        runs.append(run)
    payload["runs"] = runs
    return payload


def _load_output_round_payload(
    state_dir: Path, round_record: dict[str, object]
) -> dict[str, object]:
    round_id = str(round_record.get("round_id") or "").strip()
    if not round_id:
        return _fallback_round_payload(round_record)
    for candidate in _round_state_dir_candidates(state_dir, round_record):
        try:
            payload = load_round(candidate, round_id)
            payload.setdefault("_round_state_dir", str(candidate))
            return payload
        except ValueError:
            continue
    return _fallback_round_payload(round_record)


def _round_blocked(round_record: dict[str, object]) -> bool:
    if bool(round_record.get("review_blocked")):
        return True
    for run in list(round_record.get("runs") or []):
        if not isinstance(run, dict):
            continue
        if bool(run.get("blocked")) or bool(run.get("grade_blocked")):
            return True
    return False


def _round_terminal_command(round_record: dict[str, object]) -> str | None:
    if (
        bool(round_record.get("grading_required"))
        and bool(round_record.get("needs_grade"))
        and not bool(round_record.get("graded"))
    ):
        return None
    commands: list[str] = []
    for run in list(round_record.get("runs") or []):
        if not isinstance(run, dict):
            continue
        status = str(run.get("review_status") or run.get("status") or "").strip()
        if status and status != "completed":
            return None
        command = str(run.get("terminal_command") or "").strip().lower()
        if command not in DECISION_COMMANDS:
            return None
        commands.append(command)
    if not commands:
        return None
    return "findings" if "findings" in commands else "clean"


def review_status_details(
    state: dict[str, object], *, state_dir: Path
) -> dict[str, object]:
    progress = dict(state.get("review_progress") or {})
    current_index = dict(progress.get("current_step") or {}).get("index")
    if current_index is None:
        current_index = progress.get("next_step_index")
    if current_index is None:
        current_index = dict(state.get("pending_action") or {}).get("step_index")
    completed = {
        item.get("index")
        for item in list(progress.get("completed_steps") or [])
        if isinstance(item, dict)
    }
    steps = [
        {
            "index": index,
            "name": step.get("name"),
            "state": (
                "current"
                if index == current_index
                else "done"
                if index in completed
                else "pending"
            ),
        }
        for index, step in enumerate(
            dict(state.get("review_plan") or {}).get("steps") or []
        )
        if isinstance(step, dict)
    ]
    rounds = [
        item for item in list(state.get("rounds") or []) if isinstance(item, dict)
    ]
    step_names = {}
    for record in rounds:
        profile = dict(record.get("profile_step") or {})
        step_names[record.get("round_id")] = profile.get("name") or step_names.get(
            record.get("source_round_id")
        )
    # Caller overrides retain the original round and decision audit records.
    decisions = {
        item.get("round_id"): item.get("command")
        for collection in ("decisions", "rounds", "decision_corrections", "finding_dispositions")
        for item in list(state.get(collection) or [])
        if isinstance(item, dict) and item.get("command") in DECISION_COMMANDS
    }
    history = []
    started_at = None
    for number, record in enumerate(rounds[-20:], start=max(1, len(rounds) - 19)):
        output = _load_output_round_payload(state_dir, record)
        if number == 1:
            started_at = output.get("review_started_at")
        outcome = decisions.get(record.get("round_id"))
        if outcome not in DECISION_COMMANDS:
            blocked = (
                record.get("status") == "blocked"
                or output.get("status") == "blocked"
                or _round_blocked(record)
                or _round_blocked(output)
            )
            outcome = "blocked" if blocked else _round_terminal_command(output)
        history.append(
            {
                "number": number,
                "round_id": record.get("round_id"),
                "lane": record.get("lane"),
                "step": step_names.get(record.get("round_id")),
                "status": record.get("status") or output.get("status"),
                "outcome": outcome,
                # Finding counts are not recorded; reviewer prose is not a count.
                "findings": None,
                "started_at": output.get("review_started_at"),
                "finished_at": output.get("review_completed_at"),
            }
        )
    if len(rounds) > 20:
        started_at = _load_output_round_payload(state_dir, rounds[0]).get(
            "review_started_at"
        )
    return {
        "steps": steps,
        "rounds": len(rounds),
        "round_history": history,
        "started_at": started_at,
        "round_started_at": history[-1]["started_at"] if history else None,
    }


def _short_sha(value: object) -> str:
    text = str(value or "").strip()
    if len(text) > 12:
        return text[:12]
    return text


def _mode_label(state: dict[str, Any]) -> str | None:
    mode = dict(state.get("mode") or {})
    value = str(mode.get("effective") or mode.get("requested") or "").strip()
    return value or None


def _progress_label(state: dict[str, Any]) -> str | None:
    steps = [
        item
        for item in list(dict(state.get("review_plan") or {}).get("steps") or [])
        if isinstance(item, dict)
    ]
    if not steps:
        return None
    completed = [
        item
        for item in list(
            dict(state.get("review_progress") or {}).get("completed_steps") or []
        )
        if isinstance(item, dict)
    ]
    return f"{len(completed)}/{len(steps)}"


def _current_label(state: dict[str, Any]) -> str | None:
    pending = dict(state.get("pending_action") or {})
    if pending:
        kind = str(pending.get("kind") or "").strip()
        step = str(pending.get("step") or pending.get("lane") or "").strip()
        if kind and step:
            return f"{kind}:{step}"
        return kind or None
    current = dict(dict(state.get("review_progress") or {}).get("current_step") or {})
    if current:
        return str(current.get("name") or current.get("step") or "").strip() or None
    return None


def _validation_summary(state: dict[str, Any]) -> dict[str, str]:
    validation = dict(state.get("validation") or {})
    return {
        key: value
        for key in ("focused", "full_suite", "ci", "review_green", "note")
        if (value := str(validation.get(key) or "").strip()) and value != "unknown"
    }


def _public_convergence(state: dict[str, Any]) -> dict[str, Any] | None:
    summary = convergence_summary(state)
    if summary.get("status") == "ACTIVE" and not summary.get("accepted_findings_heads"):
        return None
    return summary


def cycle_status_fields(state: dict[str, Any], *, state_dir: Path) -> dict[str, Any]:
    identity = state.get("identity") or {}
    brief = "available" if state.get("review_brief") else "unavailable"
    payload = {
        "review": state.get("public_id"),
        "status": state.get("stage") or "unknown",
        "review_brief": brief,
        "design_conformance_context": brief,
        **review_status_details(state, state_dir=state_dir),
    }
    fields = {
        "mode": _mode_label(state),
        "cwd": identity.get("cwd"),
        "base": identity.get("base"),
        "branch": identity.get("branch"),
        "head": _short_sha(identity.get("head")),
        "merge_base": _short_sha(identity.get("merge_base")),
        "progress": _progress_label(state),
        "current": _current_label(state),
    }
    deslop = state.get("deslop") or {}
    fields.update(
        deslop=deslop.get("status"),
        conformance=deslop.get("conformance"),
        closure_dismissal_reason=deslop.get("dismissal_reason"),
        validation=_validation_summary(state),
        convergence=_public_convergence(state),
    )
    github_status = (state.get("github_review") or {}).get("status")
    if github_status != "unknown":
        fields["github_review"] = github_status
    payload.update({key: value for key, value in fields.items() if value})
    return payload


def public_cycle_status(payload: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    public = json_status_payload(payload)
    if successor := str((state.get("superseded_by") or {}).get("review") or "").strip():
        public["superseded_by"] = {"review": successor}
    if predecessor := str((state.get("restart") or {}).get("supersedes") or "").strip():
        public["supersedes"] = {"review": predecessor}
    if head := str((state.get("identity") or {}).get("head") or "").strip():
        public["head"] = head
    for key in (
        "reviewed_head", "accepted_head", "current_head", "acceptance",
        "required_checks", "validation_head", "base_interaction",
    ):
        if key in payload:
            public[key] = payload[key]
    convergence = convergence_summary(state)
    if convergence["status"] != "ACTIVE":
        route_convergence(public, convergence, state=state)
    return public


def _recorded_next_action(state: dict[str, Any], summary: dict[str, Any]) -> str:
    if summary["done"]:
        return "none"
    if summary["review_ladder"] == "invalidated":
        return "rerun_review"
    if summary["review_ladder"] == HEAD_CHANGED_AFTER_GREEN_REVIEW_LADDER:
        return "continue"

    deslop = state.get("deslop") or {}
    if deslop.get("tracked"):
        if deslop.get("status") == DESLOP_STATUS_DONE:
            return "continue" if deslop.get("conformance") == "MATERIALLY_DRIFTED" else "deslop_done"
        if deslop.get("status") in {"tracked", "failed"}:
            return "continue"
    if state.get("stage") not in {STAGE_REVIEW_GREEN, STAGE_LOCAL_GREEN_HANDOFF}:
        return "continue"

    github = state.get("github_review") or {}
    github_status = github.get("status") or "unknown"
    if github_status in {GITHUB_RESULT_CLEAN, GITHUB_RESULT_WAIVED} and github.get("reviewed_head"):
        blockers = validation_blockers(state, current_head=summary.get("current_head", ""))
        return "validation" if blockers else "none"
    if _mode_label(state) == "fast" and github_status == "unknown":
        return "none"
    return "github_review"


def recorded_cycle_status(state: dict[str, Any], *, state_dir: Path) -> dict[str, Any]:
    summary = review_ladder_summary(state, recorded=True)
    summary = green_review_head_change_summary(state, summary=summary, include_paths=False) or summary
    payload = cycle_status_fields(state, state_dir=state_dir)
    payload.update(summary)
    payload["next_action"] = _recorded_next_action(state, summary)
    if summary["done"]:
        payload["status"] = "done"
    elif summary["review_ladder"] == "invalidated":
        payload["status"] = "stale"
    elif summary["review_ladder"] == HEAD_CHANGED_AFTER_GREEN_REVIEW_LADDER:
        payload["status"] = HEAD_CHANGED_AFTER_GREEN_REVIEW_LADDER

    if "base_interaction" not in summary:
        if accepted := recorded_acceptance(state):
            payload["base_interaction"] = bound_interaction(state, accepted, recorded=True)
        elif not state.get("review_patch"):
            identity = state.get("identity") or {}
            review_heads = state.get("review_heads") or {}
            head = identity.get("head")
            reviewed = review_heads.get("last_reviewed_head") or review_heads.get("last_followup_head") or head
            base = identity.get("merge_base")
            # An unfinished cycle may only have its original head/base observation.
            if head and head == reviewed and base and base != head and identity.get("cwd"):
                payload["base_interaction"] = {**_no_base_drift(), "status": "not_required"}
    return payload


def add_status_arguments(parser: ArgumentParser) -> None:
    parser.add_argument("--id")
    parser.add_argument(
        "--show-status", action="store_true",
        help="Print the recorded status for one review id.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print sanitized JSON for --status or --show-status.",
    )


def load_status_cycle(
    public_id: str, *, locate: Callable[[], Path] = read_state_dir,
) -> tuple[Path, dict[str, Any]]:
    state_dir = locate()
    try:
        return state_dir, load_cycle_by_public_id(state_dir, public_id)
    except UnknownReviewIdError:
        # A writer may have moved legacy state after the first lookup picked it.
        moved = locate()
        if moved == state_dir:
            raise
        return moved, load_cycle_by_public_id(moved, public_id)


def try_json_status(argv: list[str]) -> int | None:
    """Serve the polling command without loading the orchestration CLI."""
    parser = ArgumentParser(add_help=False, allow_abbrev=False, exit_on_error=False)
    add_status_arguments(parser)
    try:
        args, extras = parser.parse_known_args(argv)
    except ArgumentError:
        return None  # Let the full parser render the usual usage error.
    if extras or not (args.show_status and args.json):
        return None
    try:
        if not args.id:
            raise ValueError("--show-status requires --id")
        state_dir, state = load_status_cycle(args.id)
        payload = recorded_cycle_status(state, state_dir=state_dir)
    except (OSError, ValueError) as exc:
        status = "usage_error" if isinstance(exc, ValueError) else "error"
        print(json.dumps({"error": str(exc), "status": status}))
        return 2 if status == "usage_error" else 1
    print(json.dumps(public_cycle_status(payload, state), sort_keys=True))
    return 0
