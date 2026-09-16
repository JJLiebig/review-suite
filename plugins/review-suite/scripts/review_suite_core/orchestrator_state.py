from __future__ import annotations

import json
from copy import deepcopy
from hashlib import blake2s
from pathlib import Path
from typing import Any

from .orchestrator_profiles import SUPPORTED_MODES, SUPPORTED_SELECTIONS
from .paths import cwd_path_from_normalized, normalize_cwd


ORCHESTRATOR_STATE_SCHEMA_VERSION = 1

STAGE_CREATED = "created"
STAGE_RUNNING = "running"
STAGE_DECISION_PENDING = "decision-pending"
STAGE_FIX_PENDING = "fix-pending"
STAGE_FOLLOWUP_PENDING = "followup-pending"
STAGE_REVIEW_GREEN = "review-green"
STAGE_LOCAL_GREEN_HANDOFF = "local-green-handoff"
STAGE_BLOCKED = "blocked"
STAGE_CRASHED = "crashed"
STAGE_RETRY_REQUESTED = "retry-requested"
STAGE_DISMISSED = "dismissed"
STAGE_ABORTED = "aborted"

DECISION_CLEAN = "clean"
DECISION_FINDINGS = "findings"
DECISION_COMMANDS = {DECISION_CLEAN, DECISION_FINDINGS}

CONVERGENCE_DECISIONS = {"CONTINUE", "REPLAN", "RESLICE"}
CONTRACT_CONFLICTS = set(
    "goal acceptance scope stop-condition owner authorized-behavior unit-boundary".split()
)
DEFAULT_ACCEPTED_FINDINGS_LIMIT = 3
CONVERGENCE_DEFAULTS = {
    "status": "ACTIVE",
    "accepted_findings_heads": [],
    "continue_used": False,
    "decisions": [],
}

GITHUB_RESULT_CLEAN = "clean"
GITHUB_RESULT_FINDINGS = "findings"
GITHUB_RESULT_WAIVED = "waived"
GITHUB_RESULT_COMMANDS = {
    GITHUB_RESULT_CLEAN,
    GITHUB_RESULT_FINDINGS,
    GITHUB_RESULT_WAIVED,
}

DESLOP_STATUS_TRACKED = "tracked"
DESLOP_STATUS_DONE = "done"
DESLOP_STATUS_FAILED = "failed"
DESLOP_STATUS_CLOSED = "closed"
DESLOP_STATUS_SKIPPED = "skipped"
CONFORMANCE_VERDICTS = {"CONFORMS", "MATERIALLY_DRIFTED", "NOT_APPLICABLE"}
NO_WORK_STAGES = {
    STAGE_DECISION_PENDING,
    STAGE_FIX_PENDING,
    STAGE_BLOCKED,
    STAGE_CRASHED,
    STAGE_RUNNING,
}
CLI_VALIDATION_STATUSES = ("passed", "failed", "pending", "waived")
FIXES_VALIDATED_NOTE = "Fixed and validated; no repeat review required."
FIX_REVIEW_POLICY = (
    "After fixing accepted findings, run relevant validation. Do not rerun review when every fix is either "
    "docs/test-only, preserving intended behavior and test coverage, or a low-blast-radius P2 or lower, "
    "outside high-stakes or business-critical behavior such as security, billing, data integrity, or concurrency. "
    "Otherwise, rerun on the same review id. Do not rerun solely for a clean verdict, a changed commit hash, "
    "or extra confidence. Commit/amend fixes before recording completion or repeating review."
)
VALIDATION_STATUSES = {"unknown", *CLI_VALIDATION_STATUSES}
VALIDATION_READY_STATUSES = {"passed", "waived"}
HEAD_CHANGED_AFTER_GREEN_REVIEW_LADDER = "head_changed_after_review"
HEAD_CHANGED_AFTER_GREEN_REVIEW_NOTE = (
    "The review remains green after test-only fixes. If the changes since the reviewed HEAD "
    "only correct stale tests to match behavior that was already reviewed, do not rerun the "
    "review. Run the affected tests and required validation, then proceed. Rerun the review "
    "only if production code or intended behavior changed."
)
CHANGED_SINCE_REVIEW_PATH_LIMIT = 20


def _required_text(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _base_drift_equivalent_head(state: dict[str, Any], head: str) -> str:
    drift = dict(state.get("base_drift") or {})
    if not bool(drift.get("patch_equivalent")):
        return head
    reviewed_head = str(drift.get("reviewed_head") or "").strip()
    equivalent_head = str(drift.get("equivalent_reviewed_head") or "").strip()
    if head and reviewed_head == head and equivalent_head:
        return equivalent_head
    return head


def _review_ladder_heads(state: dict[str, Any]) -> list[str]:
    review_heads = dict(state.get("review_heads") or {})
    heads: list[str] = []
    for key in (
        "last_followup_head",
        "last_reviewed_head",
        "last_fix_head",
        "head",
    ):
        value = str(review_heads.get(key) or "").strip()
        if value:
            heads.append(_base_drift_equivalent_head(state, value))
    identity_head = _base_drift_equivalent_head(
        state, str(dict(state.get("identity") or {}).get("head") or "").strip()
    )
    if identity_head:
        heads.append(identity_head)
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


def _validation_ready(state: dict[str, Any]) -> bool:
    return not validation_blockers(state)


def validation_blockers(state: dict[str, Any]) -> list[str]:
    validation = dict(state.get("validation") or {})
    note = _optional_text(validation.get("note"))
    blockers: list[str] = []
    for key in ("full_suite", "ci"):
        value = str(validation.get(key) or "unknown").strip() or "unknown"
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
    state: dict[str, Any], *, current_head: str | None = None
) -> dict[str, Any]:
    stage = str(state.get("stage") or "").strip()
    summary: dict[str, Any] = {"done": False, "review_ladder": "pending"}
    if stage not in {STAGE_REVIEW_GREEN, STAGE_LOCAL_GREEN_HANDOFF}:
        return summary

    comparison_head = str(current_head or "").strip()
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
        summary.update({"done": True, "review_ladder": "complete"})
        return summary

    github_ready = _github_review_matches_head(state, comparison_head)
    if github_ready and _validation_ready(state) and _deslop_closed_or_untracked(state):
        summary.update({"done": True, "review_ladder": "complete"})
    return summary


def _changed_since_review_paths(
    state: dict[str, Any],
    *,
    reviewed_head: str,
    current_head: str,
) -> list[str] | None:
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
    paths = _changed_since_review_paths(
        state, reviewed_head=reviewed_head, current_head=current_head_value
    )
    if paths is not None:
        changed_summary["changed_since_review"] = paths[
            :CHANGED_SINCE_REVIEW_PATH_LIMIT
        ]
        if len(paths) > CHANGED_SINCE_REVIEW_PATH_LIMIT:
            changed_summary["changed_since_review_count"] = len(paths)
    return changed_summary


def _normalize_mode(mode: str, *, field: str) -> str:
    value = _required_text(mode, field=field)
    if value not in SUPPORTED_MODES:
        raise ValueError(f"{field} must be one of: {', '.join(SUPPORTED_MODES)}")
    return value


def _normalize_selection(selection: str, *, field: str) -> str:
    value = _required_text(selection, field=field)
    if value not in SUPPORTED_SELECTIONS:
        raise ValueError(f"{field} must be one of: {', '.join(SUPPORTED_SELECTIONS)}")
    return value


def _normalize_branch(branch: str | None) -> str | None:
    value = _optional_text(branch)
    if value == "HEAD":
        return None
    return value


def normalize_cycle_identity(
    *,
    cwd: str | Path,
    base: str,
    branch: str | None,
    head: str,
    merge_base: str,
) -> dict[str, str | None]:
    return {
        "cwd": normalize_cwd(str(cwd)),
        "base": _required_text(base, field="base"),
        "branch": _normalize_branch(branch),
        "head": _required_text(head, field="head"),
        "merge_base": _required_text(merge_base, field="merge_base"),
    }


def cycle_key(
    *,
    cwd: str | Path,
    base: str,
    branch: str | None,
    head: str,
    merge_base: str,
    restart_token: str | None = None,
) -> str:
    identity = normalize_cycle_identity(
        cwd=cwd, base=base, branch=branch, head=head, merge_base=merge_base
    )
    token = _optional_text(restart_token)
    material_payload: dict[str, Any] = (
        identity if token is None else {"identity": identity, "restart_token": token}
    )
    material = json.dumps(material_payload, sort_keys=True, separators=(",", ":"))
    return f"orc-{blake2s(material.encode('utf-8'), digest_size=10).hexdigest()}"


def create_cycle(
    *,
    cwd: str | Path,
    base: str,
    branch: str | None,
    head: str,
    merge_base: str,
    requested_mode: str,
    effective_mode: str | None = None,
    selection: str = "auto",
    effective_selection: str | None = None,
    deslop_enabled: bool | None = None,
    deslop_skip_source: str | None = None,
    cycle_token: str | None = None,
    restart_token: str | None = None,
    review_brief: str | None = None,
) -> dict[str, Any]:
    identity = normalize_cycle_identity(
        cwd=cwd, base=base, branch=branch, head=head, merge_base=merge_base
    )
    requested = _normalize_mode(requested_mode, field="requested_mode")
    effective = _normalize_mode(effective_mode or requested, field="effective_mode")
    requested_selection = _normalize_selection(selection, field="selection")
    resolved_selection = _normalize_selection(
        effective_selection or requested_selection, field="effective_selection"
    )
    deslop_tracked = bool(deslop_enabled) if deslop_enabled is not None else True
    # Fast mode currently excludes cleanup/deslop; run correctness review only.
    if effective == "fast":
        deslop_tracked = False
    if deslop_tracked:
        deslop_status = DESLOP_STATUS_TRACKED
        deslop_skip = None
    else:
        deslop_status = DESLOP_STATUS_SKIPPED
        deslop_skip = _optional_text(deslop_skip_source) or "profile"
    fresh_token = _optional_text(cycle_token)
    restart = _optional_text(restart_token)
    if fresh_token is not None and restart is not None:
        raise ValueError("cycle_token cannot be combined with restart_token")
    key_token = restart or fresh_token
    if review_brief is not None and not review_brief.strip():
        raise ValueError("review_brief cannot be blank")
    state = {
        "schema_version": ORCHESTRATOR_STATE_SCHEMA_VERSION,
        "cycle_key": cycle_key(
            cwd=cwd,
            base=base,
            branch=branch,
            head=head,
            merge_base=merge_base,
            restart_token=key_token,
        ),
        "identity": identity,
        "mode": {
            "requested": requested,
            "effective": effective,
        },
        "selection": {
            "requested": requested_selection,
            "effective": resolved_selection,
        },
        "review_brief": review_brief,
        "stage": STAGE_CREATED,
        "pending_action": None,
        "deslop": {
            "tracked": deslop_tracked,
            "status": deslop_status,
        },
        "review_heads": {
            "head": identity["head"],
            "merge_base": identity["merge_base"],
            "last_reviewed_head": None,
            "last_fix_head": None,
            "last_followup_head": None,
        },
        "validation": {
            "review_green": "unknown",
            "focused": "unknown",
            "full_suite": "unknown",
            "ci": "unknown",
        },
        "github_review": {
            "status": "unknown",
        },
        "recovery": {
            "status": "none",
            "retry_count": 0,
        },
        "review_progress": {
            "next_step_index": 0,
            "current_step": None,
            "completed_steps": [],
        },
        "rounds": [],
        "decisions": [],
        "active_findings": None,
        "convergence": deepcopy(CONVERGENCE_DEFAULTS),
    }
    if restart is not None:
        state["restart"] = {"token": restart}
    if fresh_token is not None:
        state["fresh"] = {"token": fresh_token}
    if deslop_skip is not None:
        state["deslop"]["source"] = deslop_skip
    return state


def _copy_state(state: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(state)


def _compact(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in payload.items() if value not in (None, "", [], {})
    }


def _state_head(state: dict[str, Any]) -> str:
    return _required_text(
        dict(state.get("identity") or {}).get("head"), field="state.identity.head"
    )


def _last_reviewed_head(state: dict[str, Any]) -> str | None:
    review_heads = dict(state.get("review_heads") or {})
    for key in (
        "last_reviewed_head",
        "last_followup_head",
        "head",
    ):
        value = _optional_text(review_heads.get(key))
        if value:
            return value
    return None


def _round_kind(lane: str) -> str:
    if lane == "review-followup":
        return "followup"
    return "review"


def _upsert_round(
    state: dict[str, Any],
    *,
    round_id: str,
    lane: str,
    status: str,
    reviewed_head: str | None = None,
    command: str | None = None,
    source_round_id: str | None = None,
) -> dict[str, Any]:
    resolved_round_id = _required_text(round_id, field="round_id")
    resolved_lane = _required_text(lane, field="lane")
    payload = _compact(
        {
            "round_id": resolved_round_id,
            "lane": resolved_lane,
            "kind": _round_kind(resolved_lane),
            "status": status,
            "reviewed_head": reviewed_head,
            "command": command,
            "source_round_id": source_round_id,
        }
    )
    rounds = state.setdefault("rounds", [])
    for item in rounds:
        if isinstance(item, dict) and item.get("round_id") == resolved_round_id:
            item.update(payload)
            return item
    rounds.append(payload)
    return payload


def _upsert_decision(
    state: dict[str, Any],
    *,
    round_id: str,
    lane: str,
    command: str,
    reviewed_head: str,
) -> None:
    if command not in DECISION_COMMANDS:
        raise ValueError(
            f"decision command must be one of: {', '.join(sorted(DECISION_COMMANDS))}"
        )
    payload = _compact(
        {
            "round_id": _required_text(round_id, field="round_id"),
            "lane": _required_text(lane, field="lane"),
            "command": command,
            "reviewed_head": reviewed_head,
        }
    )
    decisions = state.setdefault("decisions", [])
    for item in decisions:
        if not isinstance(item, dict) or item.get("round_id") != payload["round_id"]:
            continue
        existing = str(item.get("command") or "")
        if existing != command:
            raise ValueError(
                f"round {payload['round_id']} already has decision command {existing}"
            )
        item.update(payload)
        return
    decisions.append(payload)


def _set_stage(
    state: dict[str, Any], stage: str, pending_action: dict[str, Any] | None = None
) -> None:
    state["stage"] = stage
    state["pending_action"] = _compact(deepcopy(pending_action or {})) or None


def _convergence(state: dict[str, Any]) -> dict[str, Any]:
    return state.setdefault("convergence", deepcopy(CONVERGENCE_DEFAULTS))


def convergence_summary(state: dict[str, Any]) -> dict[str, Any]:
    convergence = dict(state.get("convergence") or {})
    return _compact(
        {
            "status": convergence.get("status") or "ACTIVE",
            "reason": convergence.get("reason"),
            "conflict": convergence.get("conflict"),
            "decision": convergence.get("decision"),
            "accepted_findings_heads": len(
                convergence.get("accepted_findings_heads") or []
            ),
            "accepted_findings_limit": DEFAULT_ACCEPTED_FINDINGS_LIMIT,
            "recommendations": sorted(
                CONVERGENCE_DECISIONS
                - ({"CONTINUE"} if convergence.get("continue_used") else set())
            ),
        }
    )


def _require_convergence_decision_inplace(
    state: dict[str, Any], *, reason: str, conflict: str | None = None
) -> None:
    convergence = _convergence(state)
    if convergence.get("status") == "DECISION_REQUIRED":
        return
    convergence.update(
        status="DECISION_REQUIRED", reason=reason, conflict=conflict, decision=None
    )
    _set_stage(
        state,
        "decision-required",
        {"kind": "convergence-decision", "reason": reason},
    )


def _record_accepted_findings_head_inplace(
    state: dict[str, Any], *, reviewed_head: str, reviewed_tree: str | None
) -> None:
    convergence = _convergence(state)
    material_id = _optional_text(reviewed_tree) or _required_text(
        reviewed_head, field="reviewed_head"
    )
    heads = convergence["accepted_findings_heads"]
    if any(
        isinstance(item, dict) and item.get("tree") == material_id for item in heads
    ):
        return
    heads.append({"head": reviewed_head, "tree": material_id})
    continued_head = bool(convergence.pop("continue_pending", False))
    if continued_head or len(heads) >= DEFAULT_ACCEPTED_FINDINGS_LIMIT:
        _require_convergence_decision_inplace(state, reason="budget_exhausted")


def record_contract_conflict(state: dict[str, Any], *, conflict: str) -> dict[str, Any]:
    resolved = _required_text(conflict, field="conflict")
    if state.get("stage") != STAGE_FIX_PENDING or not isinstance(
        state.get("active_findings"), dict
    ):
        raise ValueError("--contract-conflict requires findings awaiting a fix")
    next_state = _copy_state(state)
    _require_convergence_decision_inplace(
        next_state, reason="contract_conflict", conflict=resolved
    )
    return next_state


def record_convergence_decision(
    state: dict[str, Any], *, decision: str
) -> dict[str, Any]:
    resolved = _required_text(decision, field="decision").upper()
    if resolved not in CONVERGENCE_DECISIONS:
        raise ValueError(
            f"convergence decision must be one of: {', '.join(sorted(CONVERGENCE_DECISIONS))}"
        )
    next_state = _copy_state(state)
    convergence = _convergence(next_state)
    if (resolved, convergence.get("status"), convergence.get("continue_pending")) == (
        "CONTINUE",
        "ACTIVE",
        True,
    ):
        if (
            next_state.get("stage") == STAGE_FIX_PENDING
            and next_state.get("active_findings") is None
        ):
            current = _review_progress(next_state).get("current_step") or {}
            for round_record in next_state.get("rounds", []):
                if (
                    round_record.get("round_id") == current.get("round_id")
                    and round_record.get("kind") == "review"
                    and round_record.get("status") == STAGE_DECISION_PENDING
                    and not round_record.get("command")
                ):
                    # Earlier CONTINUE versions erased this completed round's action.
                    profile = _profile_step_for_round(
                        next_state, round_record["round_id"]
                    )
                    return mark_review_step_pending(
                        next_state,
                        round_id=round_record["round_id"],
                        lane=round_record["lane"],
                        step_index=profile["index"],
                        step_name=profile["name"],
                        reviewed_head=round_record["reviewed_head"],
                        grading_required=bool(profile.get("grading_required")),
                        arena_round=bool(profile.get("arena_round")),
                        post_findings_rerun=bool(profile.get("post_findings_rerun")),
                    )
        return next_state
    if (
        convergence.get("status") == "DECIDED"
        and convergence.get("decision") == resolved
    ):
        return next_state
    if convergence.get("status") != "DECISION_REQUIRED":
        raise ValueError("no convergence decision is required")
    if resolved == "CONTINUE" and convergence.get("continue_used"):
        raise ValueError("CONTINUE is available once per review cycle")
    convergence["decisions"].append(
        {"decision": resolved, "reason": convergence.get("reason")}
    )
    convergence["decision"] = resolved
    if resolved == "CONTINUE":
        convergence.update(status="ACTIVE", continue_used=True, continue_pending=True)
        if next_state.get("stage") == "decision-required" and isinstance(
            next_state.get("active_findings"), dict
        ):
            _set_stage(next_state, STAGE_FIX_PENDING)
    else:
        convergence["status"] = "DECIDED"
        _set_stage(next_state, "convergence-decided")
    return next_state


def _nonnegative_int(value: Any, *, field: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if number < 0:
        raise ValueError(f"{field} must be >= 0")
    return number


def _review_plan_steps(state: dict[str, Any]) -> list[dict[str, Any]]:
    plan = state.get("review_plan")
    if not isinstance(plan, dict):
        return []
    return [
        dict(item) for item in list(plan.get("steps") or []) if isinstance(item, dict)
    ]


def _review_progress(state: dict[str, Any]) -> dict[str, Any]:
    progress = dict(state.get("review_progress") or {})
    completed = [
        dict(item)
        for item in list(progress.get("completed_steps") or [])
        if isinstance(item, dict)
    ]
    current = progress.get("current_step")
    return {
        "next_step_index": _nonnegative_int(
            progress.get("next_step_index", len(completed)),
            field="review_progress.next_step_index",
        ),
        "current_step": dict(current) if isinstance(current, dict) else None,
        "completed_steps": completed,
        **(
            {"next_step_name": str(progress.get("next_step_name"))}
            if str(progress.get("next_step_name") or "").strip()
            else {}
        ),
    }


def _review_step_name(state: dict[str, Any], step_index: int) -> str | None:
    steps = _review_plan_steps(state)
    if step_index < 0 or step_index >= len(steps):
        return None
    return _optional_text(steps[step_index].get("name"))


def _review_step_rerun_on_findings(state: dict[str, Any], step_index: int) -> bool:
    steps = _review_plan_steps(state)
    if step_index < 0 or step_index >= len(steps):
        return False
    return bool(steps[step_index].get("rerun_on_findings"))


def _review_step_max_review_rounds(
    state: dict[str, Any], step_index: int
) -> int | None:
    steps = _review_plan_steps(state)
    if step_index < 0 or step_index >= len(steps):
        return None
    value = steps[step_index].get("max_review_rounds")
    if value is None:
        return None
    rounds = _nonnegative_int(value, field="review_plan.steps[].max_review_rounds")
    if rounds == 0:
        raise ValueError("review_plan.steps[].max_review_rounds must be > 0")
    return rounds


def _effective_mode(state: dict[str, Any]) -> str:
    mode = dict(state.get("mode") or {})
    return str(mode.get("effective") or mode.get("requested") or "").strip()


def _findings_use_followup(state: dict[str, Any]) -> bool:
    return _effective_mode(state) == "deep"


def _profile_step_kind(step: dict[str, Any]) -> str:
    return str(step.get("kind") or "review").strip() or "review"


def _next_profile_step_action(state: dict[str, Any]) -> dict[str, Any]:
    progress = _review_progress(state)
    step_index = int(progress["next_step_index"])
    steps = _review_plan_steps(state)
    step = steps[step_index] if 0 <= step_index < len(steps) else {}
    action = {
        "kind": "run-review-step",
        "step_index": step_index,
        "step": progress.get("next_step_name"),
    }
    step_kind = _profile_step_kind(step)
    if step_kind != "review":
        action["step_kind"] = step_kind
    return action


def _rewind_profile_step_action(
    state: dict[str, Any], profile_step: dict[str, Any]
) -> dict[str, Any]:
    index = _nonnegative_int(profile_step.get("index"), field="profile_step.index")
    completed = [
        item
        for item in _review_progress(state)["completed_steps"]
        if int(item.get("index") or 0) < index
    ]
    _set_review_progress(
        state, next_step_index=index, current_step=None, completed_steps=completed
    )
    return _next_profile_step_action(state)


def _set_review_progress(
    state: dict[str, Any],
    *,
    next_step_index: int,
    current_step: dict[str, Any] | None,
    completed_steps: list[dict[str, Any]] | None = None,
) -> None:
    progress = _review_progress(state)
    index = _nonnegative_int(next_step_index, field="review_progress.next_step_index")
    progress["next_step_index"] = index
    progress["current_step"] = deepcopy(current_step) if current_step else None
    if completed_steps is not None:
        progress["completed_steps"] = deepcopy(completed_steps)
    next_step_name = _review_step_name(state, index)
    if next_step_name:
        progress["next_step_name"] = next_step_name
    else:
        progress.pop("next_step_name", None)
    state["review_progress"] = progress


def next_review_profile_step(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    progress = _review_progress(state)
    step_index = int(progress["next_step_index"])
    steps = _review_plan_steps(state)
    if step_index >= len(steps):
        raise ValueError("no review profile steps remain")
    return step_index, dict(steps[step_index])


def review_profile_has_next_step(state: dict[str, Any]) -> bool:
    progress = _review_progress(state)
    return int(progress["next_step_index"]) < len(_review_plan_steps(state))


def mark_review_step_pending(
    state: dict[str, Any],
    *,
    round_id: str,
    lane: str,
    step_index: int,
    step_name: str,
    reviewed_head: str | None = None,
    grading_required: bool = False,
    arena_round: bool = False,
    post_findings_rerun: bool = False,
    fix_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    index = _nonnegative_int(step_index, field="step_index")
    name = _required_text(step_name, field="step_name")
    action = {
        "kind": "decision",
        "round_id": round_id,
        "lane": lane,
        "step_index": index,
        "step": name,
    }
    if grading_required:
        action["grading_required"] = True
    if arena_round:
        action["arena_round"] = True
    if post_findings_rerun:
        action["post_findings_rerun"] = True
    if isinstance(fix_verification, dict) and fix_verification:
        action["fix_verification"] = _compact(deepcopy(fix_verification))
    next_state = mark_decision_pending(
        state,
        round_id=round_id,
        lane=lane,
        reviewed_head=reviewed_head,
        pending_action=action,
    )
    profile_step = {
        "index": index,
        "name": name,
        "round_id": _required_text(round_id, field="round_id"),
        "lane": _required_text(lane, field="lane"),
    }
    if _review_step_rerun_on_findings(next_state, index):
        profile_step["rerun_on_findings"] = True
    max_review_rounds = _review_step_max_review_rounds(next_state, index)
    if max_review_rounds is not None:
        profile_step["max_review_rounds"] = max_review_rounds
    if grading_required:
        profile_step["grading_required"] = True
    if arena_round:
        profile_step["arena_round"] = True
    if post_findings_rerun:
        profile_step["post_findings_rerun"] = True
    for item in list(next_state.get("rounds") or []):
        if isinstance(item, dict) and item.get("round_id") == round_id:
            item["profile_step"] = {
                key: profile_step[key]
                for key in (
                    "index",
                    "name",
                    "rerun_on_findings",
                    "max_review_rounds",
                    "grading_required",
                    "arena_round",
                    "post_findings_rerun",
                )
                if key in profile_step
            }
            if grading_required:
                item["grading_required"] = True
            if arena_round:
                item["arena_round"] = True
            break
    _set_review_progress(next_state, next_step_index=index, current_step=profile_step)
    return next_state


def _profile_step_for_round(
    state: dict[str, Any], round_id: str
) -> dict[str, Any] | None:
    progress = _review_progress(state)
    current = progress.get("current_step")
    if isinstance(current, dict) and current.get("round_id") == round_id:
        return dict(current)
    pending = dict(state.get("pending_action") or {})
    if (
        pending.get("round_id") == round_id
        and "step_index" in pending
        and pending.get("step")
    ):
        payload = {
            "index": pending.get("step_index"),
            "name": pending.get("step"),
            "round_id": round_id,
            "lane": pending.get("lane"),
        }
        if pending.get("arena_round"):
            payload["arena_round"] = True
        if pending.get("grading_required"):
            payload["grading_required"] = True
        if pending.get("post_findings_rerun"):
            payload["post_findings_rerun"] = True
        max_review_rounds = _review_step_max_review_rounds(state, int(payload["index"]))
        if max_review_rounds is not None:
            payload["max_review_rounds"] = max_review_rounds
        return payload
    for item in list(state.get("rounds") or []):
        if not isinstance(item, dict) or item.get("round_id") != round_id:
            continue
        profile_step = item.get("profile_step")
        if isinstance(profile_step, dict):
            payload = {
                "index": profile_step.get("index"),
                "name": profile_step.get("name"),
                "round_id": round_id,
                "lane": item.get("lane"),
            }
            if bool(profile_step.get("rerun_on_findings")):
                payload["rerun_on_findings"] = True
            if profile_step.get("max_review_rounds") is not None:
                payload["max_review_rounds"] = profile_step.get("max_review_rounds")
            if bool(profile_step.get("arena_round")):
                payload["arena_round"] = True
            if bool(profile_step.get("grading_required")):
                payload["grading_required"] = True
            if bool(profile_step.get("post_findings_rerun")):
                payload["post_findings_rerun"] = True
            if bool(item.get("arena_round")):
                payload["arena_round"] = True
            if bool(item.get("grading_required")):
                payload["grading_required"] = True
            return payload
    return None


def _profile_round_id_for_findings(
    state: dict[str, Any], active: dict[str, Any]
) -> str | None:
    candidates = (
        active.get("profile_round_id"),
        active.get("previous_round_id"),
        active.get("round_id"),
    )
    for candidate in candidates:
        round_id = _optional_text(candidate)
        if round_id and _profile_step_for_round(state, round_id):
            return round_id
    return None


def _complete_profile_step_from_metadata(
    state: dict[str, Any],
    *,
    profile_step: dict[str, Any],
    round_id: str,
    lane: str,
    reviewed_head: str,
) -> bool:
    index = _nonnegative_int(profile_step.get("index"), field="profile_step.index")
    name = _required_text(profile_step.get("name"), field="profile_step.name")
    progress = _review_progress(state)
    completed = list(progress["completed_steps"])
    step_kind = _profile_step_kind(profile_step)
    completed_item = _compact(
        {
            "index": index,
            "name": name,
            "round_id": _required_text(round_id, field="round_id"),
            "lane": _required_text(lane, field="lane"),
            "kind": step_kind if step_kind != "review" else None,
            "reviewed_head": reviewed_head,
            "arena_round": True if profile_step.get("arena_round") else None,
        }
    )
    for item in completed:
        if item.get("round_id") == round_id:
            item.update(completed_item)
            break
    else:
        completed.append(completed_item)
    _set_review_progress(
        state,
        next_step_index=max(int(progress["next_step_index"]), index + 1),
        current_step=None,
        completed_steps=completed,
    )
    return True


def _complete_profile_step(
    state: dict[str, Any], *, round_id: str, lane: str, reviewed_head: str
) -> bool:
    profile_step = _profile_step_for_round(state, round_id)
    if not profile_step:
        return False
    return _complete_profile_step_from_metadata(
        state,
        profile_step=profile_step,
        round_id=round_id,
        lane=lane,
        reviewed_head=reviewed_head,
    )


def _profile_step_is_discovery(profile_step: dict[str, Any]) -> bool:
    name = str(profile_step.get("name") or "").strip()
    return "discovery" in name and not bool(profile_step.get("arena_round"))


def _advance_after_clean_discovery_or_arena(
    state: dict[str, Any], profile_step: dict[str, Any]
) -> None:
    arena_round = bool(profile_step.get("arena_round"))
    if not arena_round and not _profile_step_is_discovery(profile_step):
        return
    index = _nonnegative_int(profile_step.get("index"), field="profile_step.index")
    steps = _review_plan_steps(state)
    arena_stage = tuple(
        steps[index].get(key) for key in ("lane", "task_class", "rating_pool_id")
    )
    for next_index in range(index + 1, len(steps)):
        next_step = steps[next_index]
        next_arena_stage = tuple(
            next_step.get(key) for key in ("lane", "task_class", "rating_pool_id")
        )
        if arena_round and (
            _profile_step_kind(next_step) != "arena" or next_arena_stage != arena_stage
        ):
            progress = _review_progress(state)
            if int(progress["next_step_index"]) < next_index:
                _set_review_progress(
                    state, next_step_index=next_index, current_step=None
                )
            return
        name = str(steps[next_index].get("name") or "").strip()
        if not arena_round and "signoff" in name:
            progress = _review_progress(state)
            if int(progress["next_step_index"]) < next_index:
                _set_review_progress(
                    state, next_step_index=next_index, current_step=None
                )
            return
    if arena_round:
        _set_review_progress(state, next_step_index=len(steps), current_step=None)


def _profile_step_reruns_after_findings(
    state: dict[str, Any], profile_step: dict[str, Any]
) -> bool:
    if bool(profile_step.get("rerun_on_findings")):
        return True
    try:
        index = _nonnegative_int(profile_step.get("index"), field="profile_step.index")
    except ValueError:
        return False
    return _review_step_rerun_on_findings(state, index)


def _profile_step_has_fixed_findings_budget(
    state: dict[str, Any], profile_step: dict[str, Any]
) -> bool:
    try:
        index = _nonnegative_int(profile_step.get("index"), field="profile_step.index")
    except ValueError:
        return False
    if index + 1 >= len(_review_plan_steps(state)):
        return False
    if bool(profile_step.get("rerun_on_findings")) or _review_step_rerun_on_findings(
        state, index
    ):
        return False
    return _profile_step_is_discovery(profile_step) or bool(
        profile_step.get("arena_round")
    )


def _profile_step_max_review_rounds(
    state: dict[str, Any], profile_step: dict[str, Any]
) -> int | None:
    if bool(profile_step.get("rerun_on_findings")):
        return None
    try:
        index = _nonnegative_int(profile_step.get("index"), field="profile_step.index")
    except ValueError:
        return None
    if _review_step_rerun_on_findings(state, index):
        return None
    value = profile_step.get("max_review_rounds")
    if value is None:
        return _review_step_max_review_rounds(state, index)
    rounds = _nonnegative_int(value, field="profile_step.max_review_rounds")
    if rounds == 0:
        raise ValueError("profile_step.max_review_rounds must be > 0")
    return rounds


def _findings_fix_context(active: dict[str, Any]) -> dict[str, Any]:
    return _compact(
        {
            "source_round_id": active.get("round_id"),
            "source_lane": active.get("lane"),
            "findings_reviewed_head": active.get("reviewed_head"),
            "fix_head": active.get("fix_head"),
            "github_note": active.get("note"),
        }
    )


def _mark_profile_fix_review_needed_inplace(
    state: dict[str, Any], active: dict[str, Any]
) -> bool:
    profile_round_id = _profile_round_id_for_findings(state, active)
    profile_step = (
        _profile_step_for_round(state, profile_round_id) if profile_round_id else None
    )
    if not profile_step:
        return False
    state["active_findings"] = None
    if _profile_step_has_fixed_findings_budget(state, profile_step):
        _complete_profile_step_from_metadata(
            state,
            profile_step=profile_step,
            round_id=_required_text(
                active.get("round_id"), field="active_findings.round_id"
            ),
            lane=_required_text(active.get("lane"), field="active_findings.lane"),
            reviewed_head=_required_text(
                active.get("reviewed_head"), field="active_findings.reviewed_head"
            ),
        )
        action = _next_profile_step_action(state)
    else:
        action = _rewind_profile_step_action(state, profile_step)
    action["fix_verification"] = _findings_fix_context(active)
    _set_review_green(state, "unknown")
    _set_stage(state, STAGE_CREATED, action)
    return True


def _last_completed_profile_round_id(state: dict[str, Any]) -> str | None:
    for item in reversed(_review_progress(state)["completed_steps"]):
        if not isinstance(item, dict):
            continue
        round_id = _optional_text(item.get("round_id"))
        if not round_id:
            continue
        profile_step = _profile_step_for_round(state, round_id)
        if profile_step:
            return round_id
    return None


def mark_latest_profile_step_rerun_needed(
    state: dict[str, Any], *, head: str
) -> dict[str, Any]:
    next_state = _copy_state(state)
    profile_round_id = _last_completed_profile_round_id(next_state)
    profile_step = (
        _profile_step_for_round(next_state, profile_round_id)
        if profile_round_id
        else None
    )
    if not profile_step:
        raise ValueError("cannot rerun review signoff without a completed profile step")
    review_heads = next_state.setdefault("review_heads", {})
    review_heads["last_fix_head"] = _required_text(head, field="head")
    validation = next_state.setdefault("validation", {})
    for key in ("focused", "full_suite", "ci"):
        validation[key] = "unknown"
    validation.pop("note", None)
    action = _rewind_profile_step_action(next_state, profile_step)
    _set_review_green(next_state, "unknown")
    _set_stage(next_state, STAGE_CREATED, action)
    return next_state


def _next_github_round_id(state: dict[str, Any]) -> str:
    count = sum(
        1
        for item in list(state.get("rounds") or [])
        if isinstance(item, dict) and str(item.get("lane") or "") == "review-github"
    )
    return f"github-review-{count + 1}"


def deslop_is_ready(state: dict[str, Any]) -> bool:
    deslop = dict(state.get("deslop") or {})
    if not bool(deslop.get("tracked")):
        return True
    return deslop.get("status") == DESLOP_STATUS_TRACKED and not deslop_should_run(
        state
    )


def deslop_should_run(state: dict[str, Any]) -> bool:
    deslop = dict(state.get("deslop") or {})
    if not bool(deslop.get("tracked")):
        return False
    status = str(deslop.get("status") or "")
    if status == DESLOP_STATUS_FAILED:
        return (
            state.get("stage") == STAGE_RETRY_REQUESTED
            and dict(state.get("pending_action") or {}).get("kind") == "run-deslop"
        )
    steps = _review_plan_steps(state)
    return (
        status == DESLOP_STATUS_TRACKED
        and state.get("stage") == STAGE_CREATED
        and bool(steps)
        and int(_review_progress(state)["next_step_index"]) == len(steps) - 1
        and _profile_step_kind(steps[-1]) == "review"
    )


def mark_deslop_done(
    state: dict[str, Any],
    *,
    command: str,
    conformance: str,
    reviewed_head: str,
    decision: str | None = None,
    findings: str | None = None,
) -> dict[str, Any]:
    next_state = _copy_state(state)
    verdict = _required_text(conformance, field="conformance")
    if verdict not in CONFORMANCE_VERDICTS:
        raise ValueError("invalid deslop conformance verdict")
    prior = dict(next_state.get("deslop") or {})
    next_state["deslop"] = {
        **prior,
        "tracked": True,
        "status": DESLOP_STATUS_DONE,
        "command": _required_text(command, field="command"),
        "returncode": 0,
        "conformance": verdict,
        "reviewed_head": _required_text(reviewed_head, field="reviewed_head"),
    }
    next_state["deslop"].pop("findings", None)
    if decision is not None:
        parsed_decision = _required_text(decision, field="decision")
        if parsed_decision not in {"clean", "findings"}:
            raise ValueError("invalid deslop review decision")
        next_state["deslop"]["decision"] = parsed_decision
        if parsed_decision == "findings" and _optional_text(findings):
            next_state["deslop"]["findings"] = _optional_text(findings)
    recovery = dict(next_state.get("recovery") or {})
    next_state["recovery"] = {
        "status": "none",
        "retry_count": int(recovery.get("retry_count") or 0),
    }
    action = _next_profile_step_action(next_state)
    if context := dict(next_state.get("pending_action") or {}).get("fix_verification"):
        action["fix_verification"] = context
    _set_stage(next_state, STAGE_CREATED, action)
    return next_state


def mark_deslop_closed(
    state: dict[str, Any], *, reason: str | None = None, fixes_applied: bool = False
) -> dict[str, Any]:
    next_state = _copy_state(state)
    deslop = dict(next_state.get("deslop") or {})
    if not bool(deslop.get("tracked")):
        return next_state
    if str(deslop.get("status") or "") != DESLOP_STATUS_DONE:
        raise ValueError("cleanup can only close after the bounded closure pass")
    dismissal_reason = (
        _required_text(reason, field="closure dismissal reason")
        if reason is not None
        else None
    )
    if (
        deslop.get("conformance") == "MATERIALLY_DRIFTED"
        and not dismissal_reason
        and not fixes_applied
    ):
        raise ValueError(
            "materially drifted closure requires fixes or explicit dismissal; "
            "use --deslop-done --reason '<why the findings are dismissed>'"
        )
    next_state["deslop"] = {
        **deslop,
        "tracked": False,
        "status": DESLOP_STATUS_CLOSED,
    }
    if dismissal_reason:
        next_state["deslop"]["dismissal_reason"] = dismissal_reason
    return next_state


def mark_deslop_failed(
    state: dict[str, Any], *, command: str, returncode: int | None, reason: str
) -> dict[str, Any]:
    next_state = _copy_state(state)
    next_state["deslop"] = {
        **dict(next_state.get("deslop") or {}),
        "tracked": True,
        "status": DESLOP_STATUS_FAILED,
        "command": _required_text(command, field="command"),
        "returncode": returncode,
    }
    recovery = dict(next_state.get("recovery") or {})
    next_state["recovery"] = {
        **recovery,
        "status": STAGE_RETRY_REQUESTED,
        "reason": _required_text(reason, field="reason"),
        "retry_count": int(recovery.get("retry_count") or 0) + 1,
    }
    action = {"kind": "run-deslop"}
    if context := dict(next_state.get("pending_action") or {}).get("fix_verification"):
        action["fix_verification"] = context
    _set_stage(next_state, STAGE_RETRY_REQUESTED, action)
    return next_state


def _set_review_green(state: dict[str, Any], value: str) -> None:
    if value not in VALIDATION_STATUSES:
        raise ValueError(
            f"review_green must be one of: {', '.join(sorted(VALIDATION_STATUSES))}"
        )
    state.setdefault("validation", {})["review_green"] = value


def mark_running(
    state: dict[str, Any],
    *,
    round_id: str,
    lane: str,
    reviewed_head: str | None = None,
) -> dict[str, Any]:
    next_state = _copy_state(state)
    head = reviewed_head or _state_head(next_state)
    _upsert_round(
        next_state,
        round_id=round_id,
        lane=lane,
        status=STAGE_RUNNING,
        reviewed_head=head,
    )
    _set_stage(next_state, STAGE_RUNNING)
    return next_state


def mark_review_step_running(
    state: dict[str, Any],
    *,
    round_id: str,
    lane: str,
    step_index: int,
    step_name: str,
    reviewed_head: str | None = None,
    round_state_dir: str | None = None,
    grading_required: bool = False,
    arena_round: bool = False,
    post_findings_rerun: bool = False,
    fix_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    index = _nonnegative_int(step_index, field="step_index")
    name = _required_text(step_name, field="step_name")
    next_state = mark_running(
        state, round_id=round_id, lane=lane, reviewed_head=reviewed_head
    )
    action = {
        "kind": "collect-review-step",
        "round_id": round_id,
        "lane": lane,
        "step_index": index,
        "step": name,
        "round_state_dir": _optional_text(round_state_dir),
    }
    if grading_required:
        action["grading_required"] = True
    if arena_round:
        action["arena_round"] = True
    if post_findings_rerun:
        action["post_findings_rerun"] = True
    if isinstance(fix_verification, dict) and fix_verification:
        action["fix_verification"] = _compact(deepcopy(fix_verification))
    _set_stage(next_state, STAGE_RUNNING, action)
    profile_step = {
        "index": index,
        "name": name,
        "round_id": _required_text(round_id, field="round_id"),
        "lane": _required_text(lane, field="lane"),
    }
    if _review_step_rerun_on_findings(next_state, index):
        profile_step["rerun_on_findings"] = True
    max_review_rounds = _review_step_max_review_rounds(next_state, index)
    if max_review_rounds is not None:
        profile_step["max_review_rounds"] = max_review_rounds
    if grading_required:
        profile_step["grading_required"] = True
    if arena_round:
        profile_step["arena_round"] = True
    if post_findings_rerun:
        profile_step["post_findings_rerun"] = True
    for item in list(next_state.get("rounds") or []):
        if isinstance(item, dict) and item.get("round_id") == round_id:
            item["profile_step"] = {
                key: profile_step[key]
                for key in (
                    "index",
                    "name",
                    "rerun_on_findings",
                    "max_review_rounds",
                    "grading_required",
                    "arena_round",
                    "post_findings_rerun",
                )
                if key in profile_step
            }
            if action.get("round_state_dir"):
                item["round_state_dir"] = action["round_state_dir"]
            if grading_required:
                item["grading_required"] = True
            if arena_round:
                item["arena_round"] = True
            break
    _set_review_progress(next_state, next_step_index=index, current_step=profile_step)
    return next_state


def mark_decision_pending(
    state: dict[str, Any],
    *,
    round_id: str,
    lane: str,
    reviewed_head: str | None = None,
    pending_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    next_state = _copy_state(state)
    head = reviewed_head or _state_head(next_state)
    _upsert_round(
        next_state,
        round_id=round_id,
        lane=lane,
        status=STAGE_DECISION_PENDING,
        reviewed_head=head,
    )
    next_state.setdefault("review_heads", {})["last_reviewed_head"] = head
    _set_stage(next_state, STAGE_DECISION_PENDING, pending_action)
    return next_state


def record_findings_decision(
    state: dict[str, Any],
    *,
    round_id: str,
    lane: str,
    reviewed_head: str | None = None,
    reviewed_tree: str | None = None,
) -> dict[str, Any]:
    next_state = _copy_state(state)
    head = reviewed_head or _state_head(next_state)
    profile_step = _profile_step_for_round(next_state, round_id)
    _upsert_decision(
        next_state,
        round_id=round_id,
        lane=lane,
        command=DECISION_FINDINGS,
        reviewed_head=head,
    )
    _upsert_round(
        next_state,
        round_id=round_id,
        lane=lane,
        status="decided",
        reviewed_head=head,
        command=DECISION_FINDINGS,
    )
    next_state.setdefault("review_heads", {})["last_reviewed_head"] = head
    next_state["active_findings"] = _compact(
        {
            "round_id": _required_text(round_id, field="round_id"),
            "lane": _required_text(lane, field="lane"),
            "reviewed_head": head,
            "status": STAGE_FIX_PENDING,
            "profile_round_id": round_id if profile_step else None,
        }
    )
    _set_review_green(next_state, "unknown")
    _set_stage(next_state, STAGE_FIX_PENDING)
    _record_accepted_findings_head_inplace(
        next_state, reviewed_head=head, reviewed_tree=reviewed_tree
    )
    return next_state


def _active_findings(state: dict[str, Any]) -> dict[str, Any]:
    active = state.get("active_findings")
    if not isinstance(active, dict):
        raise ValueError("no active findings are waiting for a transition")
    return active


def record_fixes_validated(
    state: dict[str, Any], *, head: str, note: str
) -> dict[str, Any]:
    """Close accepted findings by caller validation without inventing a reviewer verdict."""
    if state.get("stage") != STAGE_FIX_PENDING:
        raise ValueError("--fixes-validated requires accepted findings awaiting fixes")
    next_state = _copy_state(state)
    active = _active_findings(next_state)
    note = _required_text(note, field="fix eligibility and validation note")
    resolution = {
        "source_round_id": active["round_id"],
        "findings_head": active["reviewed_head"],
        "head": head,
        "note": note,
    }
    next_state.setdefault("validated_fixes", []).append(resolution)
    next_state["active_findings"] = None
    next_state["validation"] = {
        "focused": "passed",
        "full_suite": "unknown",
        "ci": "unknown",
        "review_green": "unknown",
    }
    next_state.setdefault("review_heads", {}).update(
        last_fix_head=head,
        last_reviewed_head=head,
    )
    _convergence(next_state).pop("continue_pending", None)
    profile_round_id = _profile_round_id_for_findings(next_state, active)
    if profile_round_id:
        profile = _profile_step_for_round(next_state, profile_round_id)
        if profile:
            _complete_profile_step_from_metadata(
                next_state,
                profile_step=profile,
                round_id=profile_round_id,
                lane=profile["lane"],
                reviewed_head=active["reviewed_head"],
            )
    if active.get("lane") == "review-github":
        next_state["github_review"] = {
            "status": GITHUB_RESULT_WAIVED,
            "reviewed_head": head,
            "note": f"{FIXES_VALIDATED_NOTE} {note}",
        }
    if review_profile_has_next_step(next_state):
        _set_stage(next_state, STAGE_CREATED, _next_profile_step_action(next_state))
    else:
        _set_review_green(next_state, "passed")
        _set_stage(next_state, STAGE_REVIEW_GREEN)
    return next_state


def mark_fix_detected(
    state: dict[str, Any],
    *,
    head: str,
) -> dict[str, Any]:
    next_state = _copy_state(state)
    if _convergence(next_state).get("status") == "DECISION_REQUIRED":
        return next_state
    active = _active_findings(next_state)
    fix_head = _required_text(head, field="head")
    active["fix_head"] = fix_head
    next_state.setdefault("review_heads", {})["last_fix_head"] = fix_head
    if _findings_use_followup(next_state):
        active["status"] = STAGE_FOLLOWUP_PENDING
        _set_stage(
            next_state,
            STAGE_FOLLOWUP_PENDING,
            {
                "kind": "run-followup",
                "source_round_id": active.get("round_id"),
                "since_head": active.get("reviewed_head"),
                "head": fix_head,
            },
        )
        return next_state
    if _mark_profile_fix_review_needed_inplace(next_state, active):
        return next_state
    active["status"] = STAGE_FOLLOWUP_PENDING
    _set_stage(
        next_state,
        STAGE_FOLLOWUP_PENDING,
        {
            "kind": "run-followup",
            "source_round_id": active.get("round_id"),
            "since_head": active.get("reviewed_head"),
            "head": fix_head,
        },
    )
    return next_state


def mark_followup_review_pending(
    state: dict[str, Any],
    *,
    round_id: str,
    reviewed_head: str,
    source_round_id: str,
) -> dict[str, Any]:
    next_state = mark_decision_pending(
        state,
        round_id=round_id,
        lane="review-followup",
        reviewed_head=reviewed_head,
        pending_action={
            "kind": "decision",
            "round_id": round_id,
            "lane": "review-followup",
            "source_round_id": source_round_id,
        },
    )
    active = _active_findings(next_state)
    active["followup_round_id"] = _required_text(round_id, field="round_id")
    active["followup_head"] = _required_text(reviewed_head, field="reviewed_head")
    active["status"] = STAGE_DECISION_PENDING
    for item in list(next_state.get("rounds") or []):
        if isinstance(item, dict) and item.get("round_id") == round_id:
            item["source_round_id"] = _required_text(
                source_round_id, field="source_round_id"
            )
            break
    next_state.setdefault("review_heads", {})["last_followup_head"] = reviewed_head
    return next_state


def record_followup_clean(
    state: dict[str, Any],
    *,
    round_id: str,
    reviewed_head: str | None = None,
) -> dict[str, Any]:
    next_state = _copy_state(state)
    active = _active_findings(next_state)
    head = reviewed_head or active.get("fix_head") or _state_head(next_state)
    _upsert_decision(
        next_state,
        round_id=round_id,
        lane="review-followup",
        command=DECISION_CLEAN,
        reviewed_head=head,
    )
    _upsert_round(
        next_state,
        round_id=round_id,
        lane="review-followup",
        status="decided",
        reviewed_head=head,
        command=DECISION_CLEAN,
        source_round_id=str(active.get("round_id") or ""),
    )
    active["followup_round_id"] = _required_text(round_id, field="round_id")
    active["followup_head"] = head
    next_state.setdefault("review_heads", {})["last_followup_head"] = head
    _convergence(next_state).pop("continue_pending", None)
    next_state["active_findings"] = None
    profile_round_id = _profile_round_id_for_findings(next_state, active)
    profile_step = (
        _profile_step_for_round(next_state, profile_round_id)
        if profile_round_id
        else None
    )
    if profile_step and (
        bool(active.get("rerun_profile_round"))
        or _profile_step_reruns_after_findings(next_state, profile_step)
    ):
        _set_review_green(next_state, "unknown")
        _set_stage(
            next_state,
            STAGE_CREATED,
            _rewind_profile_step_action(next_state, profile_step),
        )
        return next_state
    completed_profile_step = False
    if profile_round_id:
        profile_step = (
            profile_step or _profile_step_for_round(next_state, profile_round_id) or {}
        )
        completed_profile_step = _complete_profile_step(
            next_state,
            round_id=profile_round_id,
            lane=_required_text(profile_step.get("lane"), field="profile_step.lane"),
            reviewed_head=head,
        )
    if completed_profile_step and review_profile_has_next_step(next_state):
        _set_review_green(next_state, "unknown")
        _set_stage(next_state, STAGE_CREATED, _next_profile_step_action(next_state))
        return next_state
    _set_review_green(next_state, "passed")
    _set_stage(next_state, STAGE_REVIEW_GREEN)
    return next_state


def record_followup_findings(
    state: dict[str, Any],
    *,
    round_id: str,
    reviewed_head: str | None = None,
    reviewed_tree: str | None = None,
) -> dict[str, Any]:
    next_state = _copy_state(state)
    active = _active_findings(next_state)
    head = reviewed_head or active.get("fix_head") or _state_head(next_state)
    _upsert_decision(
        next_state,
        round_id=round_id,
        lane="review-followup",
        command=DECISION_FINDINGS,
        reviewed_head=head,
    )
    _upsert_round(
        next_state,
        round_id=round_id,
        lane="review-followup",
        status="decided",
        reviewed_head=head,
        command=DECISION_FINDINGS,
        source_round_id=str(active.get("round_id") or ""),
    )
    next_state.setdefault("review_heads", {})["last_followup_head"] = head
    profile_round_id = active.get("profile_round_id") or _profile_round_id_for_findings(
        next_state, active
    )
    next_state["active_findings"] = _compact(
        {
            "round_id": _required_text(round_id, field="round_id"),
            "lane": "review-followup",
            "reviewed_head": head,
            "status": STAGE_FIX_PENDING,
            "previous_round_id": active.get("round_id"),
            "profile_round_id": profile_round_id,
        }
    )
    _set_review_green(next_state, "unknown")
    _set_stage(next_state, STAGE_FIX_PENDING)
    _record_accepted_findings_head_inplace(
        next_state, reviewed_head=head, reviewed_tree=reviewed_tree
    )
    return next_state


def record_clean_decision(
    state: dict[str, Any],
    *,
    round_id: str,
    lane: str,
    reviewed_head: str | None = None,
) -> dict[str, Any]:
    next_state = _copy_state(state)
    resolved_lane = _required_text(lane, field="lane")
    head = reviewed_head or _state_head(next_state)
    active = next_state.get("active_findings")
    if isinstance(active, dict):
        raise ValueError("findings require a clean follow-up before advancing")

    _upsert_decision(
        next_state,
        round_id=round_id,
        lane=resolved_lane,
        command=DECISION_CLEAN,
        reviewed_head=head,
    )
    _upsert_round(
        next_state,
        round_id=round_id,
        lane=resolved_lane,
        status="decided",
        reviewed_head=head,
        command=DECISION_CLEAN,
    )
    next_state.setdefault("review_heads", {})["last_reviewed_head"] = head
    _convergence(next_state).pop("continue_pending", None)
    completed_profile_step = False
    profile_step = _profile_step_for_round(next_state, round_id)
    if profile_step:
        completed_profile_step = _complete_profile_step_from_metadata(
            next_state,
            profile_step=profile_step,
            round_id=round_id,
            lane=resolved_lane,
            reviewed_head=head,
        )
        _advance_after_clean_discovery_or_arena(next_state, profile_step)
    if completed_profile_step and review_profile_has_next_step(next_state):
        _set_review_green(next_state, "unknown")
        _set_stage(next_state, STAGE_CREATED, _next_profile_step_action(next_state))
        return next_state
    _set_review_green(next_state, "passed")
    _set_stage(next_state, STAGE_REVIEW_GREEN)
    return next_state


def can_advance_or_anchor(state: dict[str, Any]) -> bool:
    return (
        state.get("active_findings") is None
        and dict(state.get("validation") or {}).get("review_green") == "passed"
    )


def record_github_result(
    state: dict[str, Any],
    *,
    result: str,
    note: str | None = None,
    reviewed_head: str | None = None,
    reviewed_tree: str | None = None,
) -> dict[str, Any]:
    resolved_result = _required_text(result, field="github_result")
    if resolved_result not in GITHUB_RESULT_COMMANDS:
        raise ValueError(
            f"github_result must be one of: {', '.join(sorted(GITHUB_RESULT_COMMANDS))}"
        )
    if resolved_result == GITHUB_RESULT_WAIVED and not _optional_text(note):
        raise ValueError("--github-note is required when --github-result waived")
    if not can_advance_or_anchor(state):
        raise ValueError("--github-result requires local green review state")

    next_state = _copy_state(state)
    head = reviewed_head or _last_reviewed_head(next_state) or _state_head(next_state)
    next_state["github_review"] = _compact(
        {
            "status": resolved_result,
            "reviewed_head": head,
            "note": note,
        }
    )
    if resolved_result != GITHUB_RESULT_FINDINGS:
        return next_state

    profile_round_id = _last_completed_profile_round_id(next_state)
    if not profile_round_id:
        raise ValueError(
            "--github-result findings requires a completed local signoff step"
        )
    round_id = _next_github_round_id(next_state)
    _upsert_round(
        next_state,
        round_id=round_id,
        lane="review-github",
        status="decided",
        reviewed_head=head,
        command=DECISION_FINDINGS,
    )
    _upsert_decision(
        next_state,
        round_id=round_id,
        lane="review-github",
        command=DECISION_FINDINGS,
        reviewed_head=head,
    )
    next_state["active_findings"] = _compact(
        {
            "round_id": round_id,
            "lane": "review-github",
            "reviewed_head": head,
            "status": STAGE_FIX_PENDING,
            "profile_round_id": profile_round_id,
            "rerun_profile_round": True,
            "note": note,
        }
    )
    _set_review_green(next_state, "unknown")
    _set_stage(next_state, STAGE_FIX_PENDING)
    _record_accepted_findings_head_inplace(
        next_state, reviewed_head=head, reviewed_tree=reviewed_tree
    )
    return next_state


def mark_blocked(
    state: dict[str, Any],
    *,
    reason: str,
    round_id: str | None = None,
) -> dict[str, Any]:
    next_state = _copy_state(state)
    next_state["recovery"] = {
        **dict(next_state.get("recovery") or {}),
        "status": STAGE_BLOCKED,
        "reason": _required_text(reason, field="reason"),
    }
    if round_id:
        next_state["recovery"]["round_id"] = round_id
    _set_stage(next_state, STAGE_BLOCKED)
    return next_state


def mark_arena_recovery_requested(
    state: dict[str, Any],
    *,
    reason: str,
    round_id: str,
    lane: str,
    step_index: int,
    step_name: str,
    round_state_dir: str | None = None,
    post_findings_rerun: bool = False,
    fix_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    index = _nonnegative_int(step_index, field="step_index")
    name = _required_text(step_name, field="step_name")
    next_state = _copy_state(state)
    recovery = dict(next_state.get("recovery") or {})
    next_state["recovery"] = {
        **recovery,
        "status": STAGE_RETRY_REQUESTED,
        "reason": _required_text(reason, field="reason"),
        "round_id": _required_text(round_id, field="round_id"),
        "retry_count": int(recovery.get("retry_count") or 0) + 1,
    }
    action = {
        "kind": "arena-blocked",
        "round_id": round_id,
        "lane": lane,
        "step_index": index,
        "step": name,
        "round_state_dir": _optional_text(round_state_dir),
    }
    if post_findings_rerun:
        action["post_findings_rerun"] = True
    if isinstance(fix_verification, dict) and fix_verification:
        action["fix_verification"] = _compact(deepcopy(fix_verification))
    _set_stage(next_state, STAGE_RETRY_REQUESTED, action)
    return next_state


def mark_recovery_resolved(state: dict[str, Any]) -> dict[str, Any]:
    next_state = _copy_state(state)
    recovery = dict(next_state.get("recovery") or {})
    next_state["recovery"] = {
        "status": "none",
        "retry_count": int(recovery.get("retry_count") or 0),
    }
    return next_state


def mark_review_step_retry(
    state: dict[str, Any],
    *,
    step_index: int,
    step_name: str,
    post_findings_rerun: bool = False,
    fix_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    index = _nonnegative_int(step_index, field="step_index")
    name = _required_text(step_name, field="step_name")
    next_state = mark_recovery_resolved(state)
    _set_review_progress(next_state, next_step_index=index, current_step=None)
    action = {"kind": "run-review-step", "step_index": index, "step": name}
    if post_findings_rerun:
        action["post_findings_rerun"] = True
    if isinstance(fix_verification, dict) and fix_verification:
        action["fix_verification"] = _compact(deepcopy(fix_verification))
    _set_stage(next_state, STAGE_CREATED, action)
    return next_state


def mark_crashed(
    state: dict[str, Any],
    *,
    reason: str,
    round_id: str | None = None,
) -> dict[str, Any]:
    next_state = _copy_state(state)
    next_state["recovery"] = {
        **dict(next_state.get("recovery") or {}),
        "status": STAGE_CRASHED,
        "reason": _required_text(reason, field="reason"),
    }
    if round_id:
        next_state["recovery"]["round_id"] = round_id
    _set_stage(next_state, STAGE_CRASHED)
    return next_state


def mark_retry_requested(
    state: dict[str, Any], *, reason: str | None = None
) -> dict[str, Any]:
    next_state = _copy_state(state)
    recovery = dict(next_state.get("recovery") or {})
    recovery["status"] = STAGE_RETRY_REQUESTED
    recovery["retry_count"] = int(recovery.get("retry_count") or 0) + 1
    if reason:
        recovery["reason"] = reason
    next_state["recovery"] = recovery
    _set_stage(next_state, STAGE_RETRY_REQUESTED)
    return next_state


def dismiss_recovery(
    state: dict[str, Any], *, reason: str | None = None
) -> dict[str, Any]:
    next_state = _copy_state(state)
    recovery = dict(next_state.get("recovery") or {})
    recovery["status"] = STAGE_DISMISSED
    if reason:
        recovery["reason"] = reason
    next_state["recovery"] = recovery
    _set_stage(next_state, STAGE_DISMISSED)
    return next_state


def abort_cycle(state: dict[str, Any], *, reason: str) -> dict[str, Any]:
    next_state = _copy_state(state)
    next_state["recovery"] = {
        **dict(next_state.get("recovery") or {}),
        "status": STAGE_ABORTED,
        "reason": _required_text(reason, field="reason"),
    }
    _set_stage(next_state, STAGE_ABORTED)
    return next_state


def _set_validation_status(state: dict[str, Any], key: str, value: str | None) -> None:
    if value is None:
        return
    if value not in VALIDATION_STATUSES:
        raise ValueError(
            f"{key} must be one of: {', '.join(sorted(VALIDATION_STATUSES))}"
        )
    state.setdefault("validation", {})[key] = value


def record_validation_statuses(
    state: dict[str, Any],
    *,
    focused: str | None = None,
    full_suite: str | None = None,
    ci: str | None = None,
    validation_note: str | None = None,
) -> dict[str, Any]:
    note = _optional_text(validation_note)
    if "waived" in (full_suite, ci) and not note:
        raise ValueError(
            "--validation-note is required when --full-suite or --ci is waived"
        )
    if note and "waived" not in (full_suite, ci):
        raise ValueError(
            "--validation-note requires --full-suite waived or --ci waived"
        )
    next_state = _copy_state(state)
    _set_validation_status(next_state, "focused", focused)
    _set_validation_status(next_state, "full_suite", full_suite)
    _set_validation_status(next_state, "ci", ci)
    validation = next_state.setdefault("validation", {})
    if note:
        validation["note"] = note
    elif not any(validation.get(key) == "waived" for key in ("full_suite", "ci")):
        validation.pop("note", None)
    return next_state


def mark_local_green_handoff(
    state: dict[str, Any],
    *,
    focused: str | None = None,
    full_suite: str | None = None,
    ci: str | None = None,
    validation_note: str | None = None,
) -> dict[str, Any]:
    if not can_advance_or_anchor(state):
        raise ValueError(
            "local-green handoff requires review_green without unresolved findings"
        )
    next_state = record_validation_statuses(
        state,
        focused=focused,
        full_suite=full_suite,
        ci=ci,
        validation_note=validation_note,
    )
    _set_stage(next_state, STAGE_LOCAL_GREEN_HANDOFF)
    return next_state


def no_work_stage_is_idle(state: dict[str, Any]) -> bool:
    if state.get("stage") == STAGE_DECISION_PENDING:
        return True
    return state.get("stage") in NO_WORK_STAGES and state.get("pending_action") is None
