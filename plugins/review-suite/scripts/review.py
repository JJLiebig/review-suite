#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

_previous_dont_write_bytecode = sys.dont_write_bytecode
sys.dont_write_bytecode = True

from review_suite_runtime_bootstrap import (
    bootstrap_from_installed_cache,
    launcher_script_path,
)

bootstrap_from_installed_cache(__file__)
sys.dont_write_bytecode = _previous_dont_write_bytecode

from review_suite_core import (
    AxiArgumentParser,
    current_branch,
    current_head,
    cwd_path_from_normalized,
    default_opencode_reasoning_effort,
    EFFECTIVE_BASE_METADATA_KEYS,
    effective_base_ref,
    emit_error,
    emit_toon,
    format_command,
    has_worktree_changes,
    is_ancestor,
    merge_base,
    merge_base_drift_scope,
    normalize_cwd,
    OPENCODE_MODEL_PREFIX,
    parse_model_label,
    resolve_ref,
    resolve_repo_root,
    SUPPORTED_REASONING_EFFORTS,
    validate_opencode_reasoning_effort,
    write_text,
)
from review_suite_core.config import default_state_dir, load_config
from review_suite_core.orchestrator_profiles import (
    MODE_STRICTNESS_ORDER,
    RESTART_TARGET_MODES,
    SUPPORTED_MODES,
    resolve_orchestrator_profile,
)
from review_suite_core.review_backend import split_review_backend_model
from review_suite_core.review_branch_status import cmd_status as cmd_branch_status
from review_suite_core.orchestrator_runner import run_one_expensive_step
from review_suite_core.orchestrator_state import (
    DECISION_CLEAN,
    DECISION_FINDINGS,
    CONVERGENCE_DECISIONS,
    CONTRACT_CONFLICTS,
    GITHUB_RESULT_CLEAN,
    GITHUB_RESULT_FINDINGS,
    GITHUB_RESULT_WAIVED,
    GITHUB_RESULT_COMMANDS,
    CLI_VALIDATION_STATUSES,
    STAGE_BLOCKED,
    STAGE_CRASHED,
    STAGE_CREATED,
    STAGE_DECISION_PENDING,
    STAGE_FIX_PENDING,
    STAGE_FOLLOWUP_PENDING,
    STAGE_RUNNING,
    STAGE_LOCAL_GREEN_HANDOFF,
    STAGE_RETRY_REQUESTED,
    STAGE_REVIEW_GREEN,
    STAGE_ABORTED,
    DESLOP_STATUS_DONE,
    DESLOP_STATUS_CLOSED,
    DESLOP_STATUS_SKIPPED,
    create_cycle,
    deslop_is_ready,
    mark_deslop_closed,
    mark_fix_detected,
    mark_latest_profile_step_rerun_needed,
    record_clean_decision,
    record_findings_decision,
    record_followup_clean,
    record_followup_findings,
    record_github_result,
    record_validation_statuses,
    validation_blockers,
    review_ladder_summary,
    green_review_head_change_summary,
    HEAD_CHANGED_AFTER_GREEN_REVIEW_LADDER,
    mark_review_step_retry,
    abort_cycle,
    convergence_summary,
    record_contract_conflict,
    record_convergence_decision,
)
from review_suite_core.orchestrator_store import (
    cycles_dir,
    load_cycle_by_key,
    load_cycle_by_public_id,
    orchestrator_store_lock,
    reserve_cycle_successor,
    save_cycle,
)
from review_suite_local import (
    PUBLIC_REVIEWER_LABELS,
    grade_rank_placeholders,
    latest_rerolled_round_payload,
    load_round,
    print_reviewer_output_section,
    round_has_live_reviewer_process,
    round_needs_caller_grade,
    unique_round_state_dirs,
)


FOLLOWUP_LANE = "review-followup"
ARENA_REROLL_SLOTS = set(PUBLIC_REVIEWER_LABELS)
DECISION_COMMANDS = {DECISION_CLEAN, DECISION_FINDINGS}
NO_DECISION_PENDING_MESSAGE = "no decision is pending for this review cycle"
BASE_DRIFT_PATH_SAMPLE_LIMIT = 20
CONTINUATION_REDIRECT_STAGES = {
    STAGE_CREATED,
    STAGE_RUNNING,
    STAGE_DECISION_PENDING,
    STAGE_FIX_PENDING,
    STAGE_FOLLOWUP_PENDING,
    STAGE_RETRY_REQUESTED,
    STAGE_BLOCKED,
    STAGE_CRASHED,
    "decision-required",
}
REPO_NO_ARENA_CONFIG = "review-suite.no-arena"


def build_parser() -> argparse.ArgumentParser:
    parser = AxiArgumentParser(description="Run the review-suite orchestrator shell.")
    parser.add_argument("--id")
    parser.add_argument("--mode", choices=SUPPORTED_MODES)
    parser.add_argument(
        "--model",
        help="Override the review model for this cycle. provider/model values select the OpenCode backend automatically.",
    )
    parser.add_argument(
        "--reasoning",
        choices=tuple(sorted(SUPPORTED_REASONING_EFFORTS)),
        help="Override the review reasoning effort for this cycle.",
    )
    parser.add_argument("--restart-mode", choices=RESTART_TARGET_MODES)
    parser.add_argument(
        "--restart-brief",
        help="Replace a mistaken frozen review brief and start a fresh same-mode cycle.",
    )
    parser.add_argument(
        "--contract-conflict", choices=tuple(sorted(CONTRACT_CONFLICTS))
    )
    parser.add_argument(
        "--convergence-decision",
        choices=tuple(sorted(decision.lower() for decision in CONVERGENCE_DECISIONS)),
    )
    parser.add_argument("--reason", help="Reason for restart or closure dismissal")
    parser.add_argument("--cd")
    parser.add_argument(
        "--no-arena",
        action="store_true",
        help="Skip Arena when creating this review cycle.",
    )
    arena_preference = parser.add_mutually_exclusive_group()
    arena_preference.add_argument(
        "--set-no-arena",
        action="store_true",
        help="Disable Arena for future reviews in this repository.",
    )
    arena_preference.add_argument(
        "--clear-no-arena",
        action="store_true",
        help="Return this repository to the global Arena setting.",
    )
    parser.add_argument("--base", help="Override the detected default branch ref.")
    parser.add_argument(
        "--review-brief",
        help="Optional Markdown goal and constraints, frozen for this review cycle.",
    )
    parser.add_argument("--decision", choices=(DECISION_CLEAN, DECISION_FINDINGS))
    parser.add_argument("--github-review", action="store_true")
    parser.add_argument("--github-force", action="store_true")
    parser.add_argument(
        "--github-result", choices=tuple(sorted(GITHUB_RESULT_COMMANDS))
    )
    parser.add_argument("--github-note")
    parser.add_argument("--focused-validation", choices=CLI_VALIDATION_STATUSES)
    parser.add_argument("--full-suite", choices=CLI_VALIDATION_STATUSES)
    parser.add_argument("--ci", choices=CLI_VALIDATION_STATUSES)
    parser.add_argument("--validation-note")
    parser.add_argument("--deslop-done", action="store_true")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Inspect branch routing when no review id exists.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full branch routing snapshot for --status.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print sanitized JSON for --status.",
    )
    parser.add_argument(
        "--skip-deslop",
        "--no-deslop",
        dest="skip_deslop",
        action="store_true",
        help="Skip cleanup before final signoff when creating a review cycle.",
    )
    parser.add_argument(
        "--show-findings",
        action="store_true",
        help="Print stored reviewer output for --id without running review.",
    )
    parser.add_argument(
        "--show-status",
        action="store_true",
        help="Print compact status for --id without running review.",
    )
    parser.add_argument("--wsl", action="store_true")
    return parser


def _help_command() -> str:
    return format_command([sys.executable, str(_launcher_script_path()), "--help"])


def _launcher_script_path(name: str = "review.py") -> Path:
    return launcher_script_path(__file__, name)


class NoDecisionPendingError(ValueError):
    def __init__(self, state: dict[str, Any]) -> None:
        super().__init__(NO_DECISION_PENDING_MESSAGE)
        self.state = state


class SuccessorAdvanceBusyError(RuntimeError):
    def __init__(self, state: dict[str, Any]) -> None:
        super().__init__("successor review is already advancing")
        self.state = state


def _runtime_uses_wsl(state: dict[str, Any]) -> bool:
    return bool(
        dict(state.get("runtime") or {}).get("allow_unsafe_windows_wsl_fallback")
    )


def _review_command(public_id: str, *extra: str, state_dir: Path | None = None) -> str:
    command = [sys.executable, str(_launcher_script_path()), "--id", public_id, *extra]
    return format_command(command)


def _github_review_action_command(public_id: str, *, state_dir: Path) -> str:
    return _review_command(public_id, "--github-review", state_dir=state_dir)


def _github_result_command(
    public_id: str, result: str, *extra: str, state_dir: Path
) -> str:
    return _review_command(
        public_id, "--github-result", result, *extra, state_dir=state_dir
    )


def _deslop_done_command(public_id: str, *, state_dir: Path) -> str:
    return _review_command(public_id, "--deslop-done", state_dir=state_dir)


def _arena_grade_command(
    state: dict[str, Any],
    *,
    state_dir: Path,
) -> str | None:
    pending_payload = _pending_grade_payload(state, state_dir=state_dir)
    if pending_payload is None:
        return None
    round_id = str(pending_payload.get("round_id") or "").strip()
    branch = str(dict(state.get("identity") or {}).get("branch") or "").strip()
    task_id = (
        str(pending_payload.get("task_id_hint") or "").strip()
        or branch
        or str(state.get("public_id") or "").strip()
    )
    rating_pool_id = str(pending_payload.get("rating_pool_id") or "").strip()
    grade_state_dir = Path(str(pending_payload.get("_round_state_dir") or state_dir))
    command = [
        sys.executable,
        str(_launcher_script_path("review_suite_arena.py")),
        "grade",
        "--round-id",
        round_id,
        "--task-id",
        task_id,
        "--rating-pool-id",
        rating_pool_id or "RATING_POOL_ID",
    ]
    for rank_group in grade_rank_placeholders(pending_payload):
        command.extend(["--rank", rank_group])
    command.extend(["--basis", "BASIS", "--state-dir", str(grade_state_dir)])
    return format_command(command)


def _arena_reroll_argv(
    state: dict[str, Any], *, state_dir: Path, round_id: str, slot: str
) -> list[str]:
    identity = dict(state.get("identity") or {})
    command = [
        sys.executable,
        str(_launcher_script_path("review_suite_arena.py")),
        "reroll-slot",
        "--round-id",
        round_id,
        "--slot",
        slot,
        "--state-dir",
        str(state_dir),
        "--sqlite-path",
        str(Path.home() / ".codex" / "state_5.sqlite"),
    ]
    cwd = str(identity.get("cwd") or "").strip()
    if cwd:
        command.extend(["--cd", str(cwd_path_from_normalized(cwd))])
    base = str(identity.get("base") or "").strip()
    if base:
        command.extend(["--base", base])
    if _runtime_uses_wsl(state):
        command.append("--wsl")
    return command


def _arena_dismiss_argv(*, state_dir: Path, round_id: str) -> list[str]:
    return [
        sys.executable,
        str(_launcher_script_path("review_suite_arena.py")),
        "dismiss-round",
        "--round-id",
        round_id,
        "--state-dir",
        str(state_dir),
        "--reason",
        "orchestrator_arena_blocked",
    ]


def _arena_resume_argv(*, state_dir: Path, round_id: str) -> list[str]:
    return [
        sys.executable,
        str(_launcher_script_path("review_suite_arena.py")),
        "resume-round",
        "--round-id",
        round_id,
        "--state-dir",
        str(state_dir),
        "--sqlite-path",
        str(Path.home() / ".codex" / "state_5.sqlite"),
    ]


def _arena_run_round_argv(
    state: dict[str, Any], payload: dict[str, Any], *, state_dir: Path, round_id: str
) -> list[str]:
    identity = dict(state.get("identity") or {})
    review_scope = dict(payload.get("review_scope") or {})
    base = str(review_scope.get("base") or identity.get("base") or "").strip()
    command = [
        sys.executable,
        str(_launcher_script_path("review_suite_arena.py")),
        "run-round",
        "--round-id",
        round_id,
        "--base",
        base or "main",
        "--state-dir",
        str(state_dir),
        "--sqlite-path",
        str(Path.home() / ".codex" / "state_5.sqlite"),
    ]
    cwd = str(identity.get("cwd") or payload.get("review_cwd") or "").strip()
    if cwd:
        command.extend(["--cd", str(cwd_path_from_normalized(cwd))])
    if _runtime_uses_wsl(state):
        command.append("--wsl")
    return command


def _round_blocked_slots(payload: dict[str, Any]) -> list[str]:
    slots: list[str] = []
    for run in list(payload.get("runs") or []):
        if not isinstance(run, dict):
            continue
        if bool(run.get("blocked")) or bool(run.get("grade_blocked")):
            slot = str(run.get("slot") or "").strip()
            if slot:
                slots.append(slot)
    return slots


def _round_is_arena_recovery(
    pending: dict[str, Any], round_record: dict[str, Any], payload: dict[str, Any]
) -> bool:
    profile_step = dict(round_record.get("profile_step") or {})
    return bool(
        pending.get("arena_round")
        or round_record.get("arena_round")
        or profile_step.get("arena_round")
        or payload.get("arena_round")
    )


def _round_requires_arena_grade(
    pending: dict[str, Any], round_record: dict[str, Any], payload: dict[str, Any]
) -> bool:
    return _round_is_arena_recovery(pending, round_record, payload) and bool(
        pending.get("grading_required")
        or round_record.get("grading_required")
        or payload.get("grading_required")
    )


def _recovery_note_subject(
    pending: dict[str, Any], round_record: dict[str, Any], payload: dict[str, Any]
) -> str:
    return (
        "Arena"
        if _round_is_arena_recovery(pending, round_record, payload)
        else "Review"
    )


def _arena_recovery_plan(
    state: dict[str, Any], *, state_dir: Path
) -> dict[str, Any] | None:
    pending = dict(state.get("pending_action") or {})
    if (
        state.get("stage") != STAGE_RETRY_REQUESTED
        or str(pending.get("kind") or "") != "arena-blocked"
    ):
        return None
    round_id = str(pending.get("round_id") or "").strip()
    if not round_id:
        return {"operation": "none", "note": "Arena recovery is missing round id."}
    round_record = _round_by_id(state, round_id)
    pending_state_dir = str(pending.get("round_state_dir") or "").strip()
    extra_state_dirs = [Path(pending_state_dir)] if pending_state_dir else None
    payload = _load_output_round_payload(
        state_dir, round_record, extra_state_dirs=extra_state_dirs
    )
    round_id, payload, round_state_dir = latest_rerolled_round_payload(
        round_id=round_id,
        payload=payload,
        search_dirs=_round_state_dir_candidates(
            state_dir, round_record, extra_state_dirs=extra_state_dirs
        ),
    )
    subject = _recovery_note_subject(pending, round_record, payload)
    status = str(payload.get("status") or "").strip()
    if status == "dismissed":
        return {
            "operation": "none",
            "note": f"{subject} round dismissed; rerun this review id.",
        }
    if status == "sampled":
        return {
            "operation": "run",
            "round_id": round_id,
            "payload": payload,
            "round_state_dir": round_state_dir,
            "note": f"{subject} replacement round is sampled; rerun this review id to continue backend recovery.",
        }
    if status == "running":
        return {
            "operation": "resume",
            "round_id": round_id,
            "round_state_dir": round_state_dir,
            "note": f"{subject} replacement round is running; rerun this review id to continue backend recovery.",
        }
    if status and status not in {"completed", "dismissed"}:
        return {
            "operation": "dismiss",
            "round_id": round_id,
            "round_state_dir": round_state_dir,
            "note": f"{subject} replacement round is {status}; rerun this review id to continue backend recovery.",
        }
    slots = _round_blocked_slots(payload)
    if slots:
        if not _round_is_arena_recovery(pending, round_record, payload):
            return {
                "operation": "none",
                "note": f"{subject} round blocked; rerun this review id to retry the review step.",
            }
        rerollable_slots = [slot for slot in slots if slot in ARENA_REROLL_SLOTS]
        if len(rerollable_slots) != len(slots):
            return {
                "operation": "dismiss",
                "round_id": round_id,
                "round_state_dir": round_state_dir,
                "note": f"Blocked {subject.lower()} slot(s) cannot be rerolled safely; rerun this review id to continue backend recovery.",
            }
        return {
            "operation": "reroll",
            "round_id": round_id,
            "round_state_dir": round_state_dir,
            "slot": rerollable_slots[0],
            "note": f"Blocked {subject.lower()} slot {rerollable_slots[0]} can be rerolled; rerun this review id to continue backend recovery.",
        }
    return {
        "operation": "none",
        "note": f"{subject} recovery is ready; rerun this review id.",
    }


def _arena_recovery_action(
    state: dict[str, Any], *, state_dir: Path, public_id: str
) -> dict[str, Any] | None:
    plan = _arena_recovery_plan(state, state_dir=state_dir)
    if plan is None:
        return None
    return {
        "cmd": _review_command(public_id, state_dir=state_dir),
        "note": str(
            plan.get("note") or "Arena recovery is ready; rerun this review id."
        ),
    }


def _arena_recovery_backend_argv(
    state: dict[str, Any], *, state_dir: Path
) -> list[str] | None:
    plan = _arena_recovery_plan(state, state_dir=state_dir)
    if plan is None:
        return None
    operation = str(plan.get("operation") or "")
    round_id = str(plan.get("round_id") or "").strip()
    round_state_dir = plan.get("round_state_dir")
    if operation == "run":
        payload = dict(plan.get("payload") or {})
        return _arena_run_round_argv(
            state, payload, state_dir=round_state_dir, round_id=round_id
        )
    if operation == "resume":
        return _arena_resume_argv(state_dir=round_state_dir, round_id=round_id)
    if operation == "dismiss":
        return _arena_dismiss_argv(state_dir=round_state_dir, round_id=round_id)
    if operation == "reroll":
        return _arena_reroll_argv(
            state,
            state_dir=round_state_dir,
            round_id=round_id,
            slot=str(plan.get("slot") or ""),
        )
    return None


def _run_arena_recovery_backend_once(state: dict[str, Any], *, state_dir: Path) -> bool:
    command = _arena_recovery_backend_argv(state, state_dir=state_dir)
    if command is None:
        return False
    proc = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stderr = str(proc.stderr or "")
    if stderr:
        sys.stderr.write(stderr)
    if proc.returncode != 0:
        details = stderr.strip() or str(proc.stdout or "").strip()
        suffix = f": {details}" if details else ""
        raise ValueError(
            f"arena backend recovery command failed with exit code {proc.returncode}: {format_command(command)}{suffix}"
        )
    return True


def _blocked_decision_recovery_state(
    state: dict[str, Any], *, state_dir: Path
) -> dict[str, Any]:
    pending = dict(state.get("pending_action") or {})
    if str(pending.get("kind") or "") != "decision":
        return state
    round_id = str(pending.get("round_id") or "").strip()
    if not round_id:
        return state
    round_record = _round_by_id(state, round_id)
    payload = _load_output_round_payload(state_dir, round_record)
    if not (
        bool(round_record.get("review_blocked"))
        or bool(payload.get("review_blocked"))
        or _round_blocked_slots(payload)
    ):
        return state
    profile_step = dict(round_record.get("profile_step") or {})
    step_index = profile_step.get("index")
    if step_index is None:
        step_index = pending.get("step_index")
    step_name = str(
        profile_step.get("name") or pending.get("step") or "review-recovery"
    )
    fix_verification = pending.get("fix_verification")
    fix_context = (
        fix_verification
        if isinstance(fix_verification, dict) and fix_verification
        else None
    )
    if not _round_is_arena_recovery(pending, round_record, payload):
        return mark_review_step_retry(
            state,
            step_index=int(step_index if step_index is not None else 0),
            step_name=step_name,
            post_findings_rerun=bool(pending.get("post_findings_rerun")),
            fix_verification=fix_context,
        )
    recovery_state = dict(state)
    recovery_state["stage"] = STAGE_RETRY_REQUESTED
    action = {
        "kind": "arena-blocked",
        "round_id": round_id,
        "lane": str(pending.get("lane") or round_record.get("lane") or "review_t1"),
        "step_index": int(step_index if step_index is not None else 0),
        "step": step_name,
    }
    if _round_is_arena_recovery(pending, round_record, payload):
        action["arena_round"] = True
    if _round_requires_arena_grade(pending, round_record, payload):
        action["grading_required"] = True
    if bool(pending.get("post_findings_rerun")):
        action["post_findings_rerun"] = True
    if fix_context:
        action["fix_verification"] = deepcopy(fix_context)
    recovery_state["pending_action"] = action
    return recovery_state


def _blocked_decision_action(
    state: dict[str, Any], *, state_dir: Path, public_id: str
) -> dict[str, Any] | None:
    recovery_state = _blocked_decision_recovery_state(state, state_dir=state_dir)
    if recovery_state is state:
        return None
    pending = dict(recovery_state.get("pending_action") or {})
    if str(pending.get("kind") or "") == "run-review-step":
        step = str(pending.get("step") or "review").strip() or "review"
        return {
            "cmd": _review_command(public_id, state_dir=state_dir),
            "note": f"Review round blocked; rerun this review id to retry {step}.",
        }
    return _arena_recovery_action(
        recovery_state, state_dir=state_dir, public_id=public_id
    )


def _pending_grade_payload(
    state: dict[str, Any], *, state_dir: Path
) -> dict[str, Any] | None:
    pending = dict(state.get("pending_action") or {})
    round_id = str(pending.get("round_id") or "").strip()
    if str(pending.get("kind") or "") != "decision" or not round_id:
        return None
    round_record = _round_record_by_id(state, round_id)
    if not bool(round_record.get("grading_required")):
        return None
    payload = _load_output_round_payload(state_dir, round_record)
    if not bool(round_record.get("arena_round") or payload.get("arena_round")):
        return None
    if not round_needs_caller_grade(payload):
        return None
    return payload


def _configured_selection(config: dict[str, Any]) -> str:
    return str(((config.get("orchestrator") or {}).get("selection") or "auto")).strip()


def _git_config(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), "config", "--local", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _repo_has_no_arena(repo: Path) -> bool:
    proc = _git_config(repo, "--type=bool", "--get", REPO_NO_ARENA_CONFIG)
    if proc.returncode == 1:
        return False
    if proc.returncode != 0:
        raise ValueError(
            (
                proc.stderr or proc.stdout or "could not read repository Arena setting"
            ).strip()
        )
    return proc.stdout.strip() == "true"


def _set_repo_no_arena(repo: Path, *, disabled: bool) -> None:
    if disabled:
        proc = _git_config(repo, REPO_NO_ARENA_CONFIG, "true")
    else:
        current = _git_config(repo, "--get-all", REPO_NO_ARENA_CONFIG)
        if current.returncode == 1:
            return
        if current.returncode != 0:
            raise ValueError(
                (
                    current.stderr
                    or current.stdout
                    or "could not read repository Arena setting"
                ).strip()
            )
        proc = _git_config(repo, "--unset-all", REPO_NO_ARENA_CONFIG)
    if proc.returncode != 0:
        raise ValueError(
            (
                proc.stderr
                or proc.stdout
                or "could not update repository Arena setting"
            ).strip()
        )


def _configure_repo_arena(args: argparse.Namespace) -> int | None:
    if not (args.set_no_arena or args.clear_no_arena):
        return None
    allowed = {"cd", "set_no_arena", "clear_no_arena"}
    conflicts = [
        name
        for name, value in vars(args).items()
        if name not in allowed and value not in (None, False)
    ]
    if conflicts:
        raise ValueError(
            "repository Arena configuration cannot be combined with "
            + ", ".join(f"--{name.replace('_', '-')}" for name in conflicts)
        )
    repo = resolve_repo_root(args.cd)
    disabled = bool(args.set_no_arena)
    _set_repo_no_arena(repo, disabled=disabled)
    emit_toon(
        {
            "status": "configured",
            "cwd": normalize_cwd(str(repo)),
            "arena": "disabled" if disabled else "global-default",
        }
    )
    return 0


def _path_key(path: Path) -> str:
    key = str(path.resolve(strict=False))
    return key.lower() if sys.platform == "win32" else key


def _append_unique_path(paths: list[Path], path: Path | None) -> None:
    if path is None:
        return
    key = _path_key(path)
    if all(_path_key(item) != key for item in paths):
        paths.append(path)


def _cycle_candidates(initial_state_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    default_dir = Path(default_state_dir())
    _append_unique_path(candidates, initial_state_dir)
    _append_unique_path(candidates, default_dir)
    return candidates


def _load_cycle_and_state_dir(
    initial_state_dir: Path, public_id: str
) -> tuple[Path, dict[str, Any]]:
    for candidate in _cycle_candidates(initial_state_dir):
        try:
            return candidate, load_cycle_by_public_id(candidate, public_id)
        except ValueError:
            continue
    raise ValueError(f"unknown review cycle id: {public_id}")


def _reject_id_creation_args(args: argparse.Namespace, state: dict[str, Any]) -> None:
    sent = [
        name
        for name in ("mode", "cd", "base", "review_brief", "model", "reasoning")
        if getattr(args, name) is not None
    ]
    if args.no_arena:
        sent.append("no_arena")
    if not sent:
        return
    mode = dict(state.get("mode") or {})
    identity = dict(state.get("identity") or {})
    context = []
    locked_mode = str(mode.get("effective") or mode.get("requested") or "").strip()
    if locked_mode:
        context.append(f"mode {locked_mode}")
    cwd = str(identity.get("cwd") or "").strip()
    if cwd:
        context.append(cwd)
    suffix = f"; this id is locked to {' at '.join(context)}" if context else ""
    raise ValueError(
        f"--id already selects review context; remove --{', --'.join(sent)}{suffix}"
    )


def _reject_review_brief_replacement(
    state: dict[str, Any], review_brief: str | None
) -> None:
    if review_brief is not None and review_brief != state.get("review_brief"):
        public_id = str(state.get("public_id") or "<id>").strip()
        raise ValueError(
            "review brief is frozen for this cycle; use "
            f"--id {public_id} --restart-brief <MARKDOWN> --reason <WHY> to replace it"
        )


def _restart_reason(args: argparse.Namespace, *, option: str = "--restart-mode") -> str:
    reason = str(args.reason or "").strip()
    if not reason or reason == "REASON":
        raise ValueError(f"--reason is required for {option}")
    return reason


def _mode_rank(mode: str) -> int:
    if mode not in MODE_STRICTNESS_ORDER:
        allowed = ", ".join(MODE_STRICTNESS_ORDER)
        raise ValueError(
            f"review cycle mode {mode} cannot be restarted; supported modes: {allowed}"
        )
    return MODE_STRICTNESS_ORDER[mode]


def _validate_restart_mode(state: dict[str, Any], target_mode: str) -> str:
    current_mode = _current_mode(state)
    current_rank = _mode_rank(current_mode)
    target_rank = _mode_rank(target_mode)
    if target_rank <= current_rank:
        raise ValueError(
            f"--restart-mode must increase strictness from {current_mode}; requested {target_mode}"
        )
    return current_mode


def _current_mode(state: dict[str, Any]) -> str:
    mode = dict(state.get("mode") or {})
    current_mode = str(mode.get("effective") or mode.get("requested") or "").strip()
    _mode_rank(current_mode)
    return current_mode


def _normalized_branch(branch: str | None) -> str | None:
    value = str(branch or "").strip()
    return None if value in {"", "HEAD"} else value


def _round_by_id(state: dict[str, Any], round_id: str) -> dict[str, Any]:
    for item in list(state.get("rounds") or []):
        if isinstance(item, dict) and item.get("round_id") == round_id:
            return dict(item)
    return {}


def _pending_decision(state: dict[str, Any]) -> tuple[str, str]:
    pending = dict(state.get("pending_action") or {})
    round_id = str(pending.get("round_id") or "").strip()
    lane = str(pending.get("lane") or "").strip()
    if round_id and lane:
        return round_id, lane
    for item in reversed(list(state.get("rounds") or [])):
        if not isinstance(item, dict) or item.get("status") != STAGE_DECISION_PENDING:
            continue
        round_id = str(item.get("round_id") or "").strip()
        lane = str(item.get("lane") or "").strip()
        if round_id and lane:
            return round_id, lane
    raise ValueError("no decision is pending for this review cycle")


def _round_terminal_command(payload: dict[str, Any]) -> str | None:
    command = str(payload.get("terminal_command") or "").strip().lower()
    if command in DECISION_COMMANDS:
        return command
    commands: list[str] = []
    for run in list(payload.get("runs") or []):
        if not isinstance(run, dict):
            continue
        if bool(run.get("blocked")) or bool(run.get("grade_blocked")):
            return None
        status = str(run.get("review_status") or run.get("status") or "").strip()
        if status and status != "completed":
            return None
        command = str(run.get("terminal_command") or "").strip().lower()
        if command not in DECISION_COMMANDS:
            return None
        commands.append(command)
    if not commands:
        return None
    return DECISION_FINDINGS if DECISION_FINDINGS in commands else DECISION_CLEAN


def _auto_decision_command(state: dict[str, Any], *, state_dir: Path) -> str | None:
    pending = dict(state.get("pending_action") or {})
    if str(pending.get("kind") or "") != "decision":
        return None
    round_id = str(pending.get("round_id") or "").strip()
    if not round_id or _pending_grade_payload(state, state_dir=state_dir) is not None:
        return None
    round_record = _round_by_id(state, round_id)
    payload = _load_output_round_payload(state_dir, round_record)
    if bool(round_record.get("review_blocked")) or bool(payload.get("review_blocked")):
        return None
    if _round_blocked_slots(round_record) or _round_blocked_slots(payload):
        return None
    status_values = {
        str(round_record.get("review_status") or "").strip(),
        str(round_record.get("status") or "").strip(),
        str(payload.get("review_status") or "").strip(),
        str(payload.get("status") or "").strip(),
    }
    if not (status_values & {"completed", "decision-pending"}):
        return None
    return _round_terminal_command(payload)


def _with_fix_action(state: dict[str, Any]) -> dict[str, Any]:
    next_state = dict(state)
    active = dict(next_state.get("active_findings") or {})
    next_state["pending_action"] = {
        "kind": "fix-findings",
        "round_id": active.get("round_id"),
    }
    return next_state


def _orchestrator_review_state_dir(state_dir: Path) -> Path:
    return state_dir / "orchestrator" / "review-rounds"


def _round_state_dir_candidates(
    state_dir: Path,
    round_record: dict[str, Any],
    *,
    extra_state_dirs: list[Path] | None = None,
) -> list[Path]:
    candidates: list[Path] = []
    candidates.extend(extra_state_dirs or [])
    round_state_dir = str(round_record.get("round_state_dir") or "").strip()
    if round_state_dir:
        candidates.append(Path(round_state_dir))
    candidates.extend([_orchestrator_review_state_dir(state_dir), state_dir])
    return unique_round_state_dirs(candidates)


def _task_class_for_lane(lane: str) -> str:
    return {
        "review_t1": "phase_review",
        "review_t3": "pr_review",
        FOLLOWUP_LANE: "phase_review",
    }.get(lane, lane)


def _fallback_round_payload(round_record: dict[str, Any]) -> dict[str, Any]:
    lane = str(round_record.get("lane") or "").strip()
    payload = dict(round_record)
    payload["task_class"] = str(payload.get("task_class") or _task_class_for_lane(lane))
    payload["status"] = str(
        payload.get("status") or payload.get("review_status") or "unknown"
    )
    runs = []
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


def _round_record_by_id(state: dict[str, Any], round_id: str) -> dict[str, Any]:
    for item in list(state.get("rounds") or []):
        if isinstance(item, dict) and str(item.get("round_id") or "") == round_id:
            return dict(item)
    return {}


def _load_output_round_payload(
    state_dir: Path,
    round_record: dict[str, Any],
    *,
    extra_state_dirs: list[Path] | None = None,
) -> dict[str, Any]:
    round_id = str(round_record.get("round_id") or "").strip()
    if not round_id:
        return _fallback_round_payload(round_record)
    for candidate in _round_state_dir_candidates(
        state_dir, round_record, extra_state_dirs=extra_state_dirs
    ):
        try:
            payload = load_round(candidate, round_id)
            payload.setdefault("_round_state_dir", str(candidate))
            return payload
        except ValueError:
            continue
    return _fallback_round_payload(round_record)


def _payload_has_reviewer_output(payload: dict[str, Any]) -> bool:
    for run in list(payload.get("runs") or []):
        if not isinstance(run, dict):
            continue
        for key in ("reviewer_output", "status_summary", "summary"):
            if str(run.get(key) or "").strip():
                return True
    return False


def _output_round_candidates(state: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_round_id(round_id: str) -> None:
        value = str(round_id or "").strip()
        if not value or value in seen:
            return
        seen.add(value)
        record = _round_record_by_id(state, value)
        if record:
            candidates.append(record)
        else:
            candidates.append({"round_id": value})

    pending = dict(state.get("pending_action") or {})
    if str(pending.get("kind") or "") in {"collect-review-step", "decision"}:
        add_round_id(str(pending.get("round_id") or ""))
    active = state.get("active_findings")
    if isinstance(active, dict):
        add_round_id(str(active.get("round_id") or ""))
    for item in reversed(list(state.get("rounds") or [])):
        if isinstance(item, dict):
            add_round_id(str(item.get("round_id") or ""))
    return candidates


def _show_findings(state: dict[str, Any], *, state_dir: Path) -> int:
    public_id = str(state.get("public_id") or "").strip()
    closure_findings = str(
        dict(state.get("deslop") or {}).get("findings") or ""
    ).strip()
    if closure_findings and _deslop_is_open(state):
        write_text(f"review: {public_id}")
        write_text("")
        write_text("Output:")
        write_text(closure_findings)
        return 0
    candidates = _output_round_candidates(state)
    if not candidates:
        write_text(f"review: {public_id}")
        write_text("no stored reviewer output found")
        return 0
    selected = candidates[0]
    payload = _load_output_round_payload(state_dir, selected)
    for candidate in candidates:
        candidate_payload = _load_output_round_payload(state_dir, candidate)
        if _payload_has_reviewer_output(candidate_payload):
            selected = candidate
            payload = candidate_payload
            break
    write_text(f"review: {public_id}")
    write_text("")
    if not print_reviewer_output_section(
        [run for run in list(payload.get("runs") or []) if isinstance(run, dict)]
    ):
        write_text("no stored reviewer output found for this round")
    return 0


def _short_sha(value: object) -> str:
    text = str(value or "").strip()
    if len(text) > 12:
        return text[:12]
    return text


def _mode_label(state: dict[str, Any]) -> str | None:
    mode = dict(state.get("mode") or {})
    value = str(mode.get("effective") or mode.get("requested") or "").strip()
    return value or None


def _cwd_label(identity: dict[str, Any]) -> str:
    cwd = str(identity.get("cwd") or "").strip()
    if not cwd:
        return ""
    try:
        return str(cwd_path_from_normalized(cwd))
    except ValueError:
        return cwd


def _live_worktree_status(identity: dict[str, Any]) -> dict[str, Any] | str:
    cwd = str(identity.get("cwd") or "").strip()
    if not cwd:
        return "unknown"
    try:
        review_cwd = cwd_path_from_normalized(cwd)
        payload: dict[str, Any] = {
            "head": _short_sha(current_head(review_cwd)),
            "dirty": has_worktree_changes(review_cwd),
        }
        if branch := current_branch(review_cwd):
            payload["branch"] = branch
        return payload
    except OSError, ValueError:
        return "unavailable"


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


def _review_runtime_label(state: dict[str, Any], *, state_dir: Path) -> str | None:
    pending = dict(state.get("pending_action") or {})
    if str(pending.get("kind") or "") != "collect-review-step":
        return None
    round_record = _round_record_by_id(
        state, str(pending.get("round_id") or "").strip()
    )
    payload = _load_output_round_payload(state_dir, round_record)
    if str(payload.get("status") or "") != "running" or not payload.get("runs"):
        return None
    return (
        "active" if round_has_live_reviewer_process(payload) else "collection_pending"
    )


def _next_action_label(summary: dict[str, Any], action: dict[str, Any] | None) -> str:
    if bool(summary.get("done")):
        return "none"
    if summary.get("review_ladder") == "invalidated":
        return (
            "continue"
            if _action_recovers_pending_github_head_change(action)
            else "rerun_review"
        )
    if not action:
        if summary.get("review_ladder") == HEAD_CHANGED_AFTER_GREEN_REVIEW_LADDER:
            return "inspect_changed_since_review"
        return "none"
    command = str(action.get("cmd") or "")
    if "--github-review" in command:
        return "github_review"
    if "--full-suite" in command or "--ci" in command:
        return "validation"
    if "--deslop-done" in command:
        return "deslop_done"
    return "continue"


def _action_recovers_pending_github_head_change(action: dict[str, Any] | None) -> bool:
    note = str(dict(action or {}).get("note") or "")
    return "before a terminal GitHub result" in note


def _add_review_ladder_fields(
    payload: dict[str, Any],
    state: dict[str, Any],
    action: dict[str, Any] | None,
    *,
    current_head_value: str | None = None,
) -> dict[str, Any]:
    summary = review_ladder_summary(
        state, current_head=current_head_value or _identity_head(state)
    )
    if summary.get(
        "review_ladder"
    ) == "invalidated" and _action_recovers_pending_github_head_change(action):
        summary = {**summary, "review_ladder": "pending"}
    else:
        summary = green_review_head_change_summary(state, summary=summary) or summary
    payload.update(summary)
    payload["next_action"] = _next_action_label(summary, action)
    if summary.get("review_ladder") == "invalidated":
        payload["status"] = "stale"
    elif summary.get("review_ladder") == HEAD_CHANGED_AFTER_GREEN_REVIEW_LADDER:
        payload["status"] = HEAD_CHANGED_AFTER_GREEN_REVIEW_LADDER
    elif bool(summary.get("done")):
        payload["status"] = "done"
    return summary


def _show_status(state: dict[str, Any], *, state_dir: Path) -> int:
    identity = dict(state.get("identity") or {})
    payload: dict[str, Any] = {
        "review": state.get("public_id"),
        "status": state.get("stage") or "unknown",
        "review_brief": "available" if state.get("review_brief") else "unavailable",
        "design_conformance_context": "available"
        if state.get("review_brief")
        else "unavailable",
    }
    if mode := _mode_label(state):
        payload["mode"] = mode
    if cwd := _cwd_label(identity):
        payload["cwd"] = cwd
    for key in ("base", "branch"):
        value = str(identity.get(key) or "").strip()
        if value:
            payload[key] = value
    if head := _short_sha(identity.get("head")):
        payload["head"] = head
    if merge_base_value := _short_sha(identity.get("merge_base")):
        payload["merge_base"] = merge_base_value
    payload["worktree"] = _live_worktree_status(identity)
    if progress := _progress_label(state):
        payload["progress"] = progress
    if current := _current_label(state):
        payload["current"] = current
    if runtime := _review_runtime_label(state, state_dir=state_dir):
        payload["runtime"] = runtime
    payload["rounds"] = len(
        [item for item in list(state.get("rounds") or []) if isinstance(item, dict)]
    )
    deslop = dict(state.get("deslop") or {})
    if deslop_status := str(deslop.get("status") or "").strip():
        payload["deslop"] = deslop_status
    if conformance := str(deslop.get("conformance") or "").strip():
        payload["conformance"] = conformance
    if reason := deslop.get("dismissal_reason"):
        payload["closure_dismissal_reason"] = reason
    if validation := _validation_summary(state):
        payload["validation"] = validation
    github_review = dict(state.get("github_review") or {})
    github_status = str(github_review.get("status") or "").strip()
    if github_status and github_status != "unknown":
        payload["github_review"] = github_status
    if convergence := _public_convergence(state):
        payload["convergence"] = convergence
    action = _action_payload(state, state_dir=state_dir)
    summary = _add_review_ladder_fields(payload, state, action)
    if action:
        if summary.get("review_ladder") != "invalidated":
            payload["Action"] = action
    emit_toon(payload)
    return 0


def _identity_head(state: dict[str, Any]) -> str | None:
    identity = dict(state.get("identity") or {})
    cwd = str(identity.get("cwd") or "").strip()
    if not cwd:
        return None
    try:
        return current_head(cwd_path_from_normalized(cwd))
    except OSError, ValueError:
        return None


def _base_drift_reviewed_head(state: dict[str, Any]) -> str | None:
    if _deslop_is_open(state):
        return str(dict(state.get("deslop") or {}).get("reviewed_head") or "") or None
    active = dict(state.get("active_findings") or {})
    reviewed_head = str(active.get("reviewed_head") or "").strip()
    if reviewed_head:
        return reviewed_head
    pending = dict(state.get("pending_action") or {})
    pending_round = _round_by_id(state, str(pending.get("round_id") or "").strip())
    reviewed_head = str(pending_round.get("reviewed_head") or "").strip()
    if reviewed_head:
        return reviewed_head
    review_heads = dict(state.get("review_heads") or {})
    for key in ("last_reviewed_head", "head"):
        reviewed_head = str(review_heads.get(key) or "").strip()
        if reviewed_head:
            return reviewed_head
    identity = dict(state.get("identity") or {})
    reviewed_head = str(identity.get("head") or "").strip()
    return reviewed_head or None


def _allowed_base_drift(
    *,
    review_root: Path,
    recorded_merge_base: str,
    current_merge_base: str,
    reviewed_head: str,
    current_head_value: str,
) -> dict[str, Any] | None:
    if (
        not recorded_merge_base
        or not current_merge_base
        or recorded_merge_base == current_merge_base
    ):
        return None
    drift = merge_base_drift_scope(
        review_cwd=review_root,
        recorded_merge_base=recorded_merge_base,
        current_merge_base=current_merge_base,
        reviewed_head=reviewed_head,
    )
    if list(drift.get("overlapping_paths") or []):
        return None
    base_changed_paths = list(drift.get("base_changed_paths") or [])
    payload = {
        "status": "ignored_no_path_overlap",
        "recorded_merge_base": recorded_merge_base,
        "current_merge_base": current_merge_base,
        "reviewed_head": reviewed_head,
        "current_head": current_head_value,
        "base_changed_path_count": len(base_changed_paths),
        "base_changed_paths": base_changed_paths[:BASE_DRIFT_PATH_SAMPLE_LIMIT],
        "overlap_paths": [],
        "patch_equivalent": bool(drift.get("patch_equivalent")),
    }
    if payload["patch_equivalent"]:
        payload["equivalent_reviewed_head"] = current_head_value
        return payload
    parent_head = _first_parent(review_root, current_head_value)
    if not parent_head:
        return payload
    parent_drift = merge_base_drift_scope(
        review_cwd=review_root,
        recorded_merge_base=recorded_merge_base,
        current_merge_base=current_merge_base,
        reviewed_head=reviewed_head,
        current_head=parent_head,
    )
    if not list(parent_drift.get("overlapping_paths") or []) and bool(
        parent_drift.get("patch_equivalent")
    ):
        payload["equivalent_reviewed_head"] = parent_head
    return payload


def _first_parent(review_root: Path, head: str) -> str | None:
    try:
        return resolve_ref(review_root, f"{head}^")
    except ValueError:
        return None


def _current_cycle_identity_if_compatible(
    state: dict[str, Any],
) -> dict[str, Any] | None:
    identity = dict(state.get("identity") or {})
    cwd = str(identity.get("cwd") or "").strip()
    base = str(identity.get("base") or "").strip()
    if not cwd or not base:
        return None
    try:
        review_root = cwd_path_from_normalized(cwd)
        expected_branch = _normalized_branch(str(identity.get("branch") or ""))
        branch = _normalized_branch(current_branch(review_root))
        if branch != expected_branch:
            return None
        if has_worktree_changes(review_root):
            return None
        head = current_head(review_root)
        expected_merge_base = str(identity.get("merge_base") or "").strip()
        current_merge_base = merge_base(review_root, base, "HEAD")
        base_drift = None
        if expected_merge_base and current_merge_base != expected_merge_base:
            reviewed_head = _base_drift_reviewed_head(state)
            if not reviewed_head:
                return None
            base_drift = _allowed_base_drift(
                review_root=review_root,
                recorded_merge_base=expected_merge_base,
                current_merge_base=current_merge_base,
                reviewed_head=reviewed_head,
                current_head_value=head,
            )
            if base_drift is None:
                return None
            if (
                state.get("stage") in {STAGE_DECISION_PENDING, STAGE_FIX_PENDING}
                and not str(base_drift.get("equivalent_reviewed_head") or "").strip()
            ):
                return None
        return {
            "head": head,
            "merge_base": current_merge_base,
            "base_drift": base_drift,
        }
    except AttributeError, OSError, ValueError:
        return None


def _fix_pending_head_change_identity_if_compatible(
    state: dict[str, Any],
) -> dict[str, Any] | None:
    active = dict(state.get("active_findings") or {})
    reviewed_head = str(active.get("reviewed_head") or "").strip()
    identity = dict(state.get("identity") or {})
    cwd = str(identity.get("cwd") or "").strip()
    base = str(identity.get("base") or "").strip()
    if not reviewed_head or not cwd or not base:
        return None
    try:
        review_root = cwd_path_from_normalized(cwd)
        expected_branch = _normalized_branch(str(identity.get("branch") or ""))
        if _normalized_branch(current_branch(review_root)) != expected_branch:
            return None
        if has_worktree_changes(review_root):
            return None
        head = current_head(review_root)
        if head == reviewed_head:
            return None
        return {
            "head": head,
            "merge_base": merge_base(review_root, base, "HEAD"),
            "base_drift": None,
        }
    except AttributeError, OSError, ValueError:
        return None


def _reviewed_tree(state: dict[str, Any], head: str | None) -> str | None:
    try:
        cwd = str(dict(state.get("identity") or {}).get("cwd") or "").strip()
        if not cwd or not head:
            return None
        return resolve_ref(cwd_path_from_normalized(cwd), f"{head}^{{tree}}")
    except OSError, ValueError:
        return None


def _is_material_fix_head(
    state: dict[str, Any], *, head: str, reviewed_head: str
) -> bool:
    head_tree = _reviewed_tree(state, head)
    reviewed_tree = _reviewed_tree(state, reviewed_head)
    return head != reviewed_head and (
        not head_tree or not reviewed_tree or head_tree != reviewed_tree
    )


def _apply_decision_to_ready_state(
    state: dict[str, Any],
    decision: str,
    *,
    state_dir: Path,
    require_grade: bool = True,
) -> dict[str, Any]:
    if state.get("stage") != STAGE_DECISION_PENDING:
        raise ValueError(NO_DECISION_PENDING_MESSAGE)
    round_id, lane = _pending_decision(state)
    if require_grade and _pending_grade_payload(state, state_dir=state_dir) is not None:
        raise ValueError(
            "grade the arena round before recording a clean/findings decision"
        )
    round_payload = _round_by_id(state, round_id)
    blocked_slots = _round_blocked_slots(round_payload)
    if bool(round_payload.get("review_blocked")) or blocked_slots:
        slot_text = (
            f" Blocked slots: {', '.join(blocked_slots)}." if blocked_slots else ""
        )
        raise ValueError(
            f"cannot record a {decision} decision for blocked review round {round_id}.{slot_text} "
            "Rerun or recover the review round before recording a decision."
        )
    reviewed_head = str(round_payload.get("reviewed_head") or "").strip() or None
    reviewed_tree = _reviewed_tree(state, reviewed_head)
    if decision == DECISION_CLEAN:
        if lane == FOLLOWUP_LANE:
            return record_followup_clean(
                state, round_id=round_id, reviewed_head=reviewed_head
            )
        next_state = record_clean_decision(
            state, round_id=round_id, lane=lane, reviewed_head=reviewed_head
        )
        return next_state
    if decision == DECISION_FINDINGS:
        if lane == FOLLOWUP_LANE:
            return _with_fix_action(
                record_followup_findings(
                    state,
                    round_id=round_id,
                    reviewed_head=reviewed_head,
                    reviewed_tree=reviewed_tree,
                )
            )
        next_state = _with_fix_action(
            record_findings_decision(
                state,
                round_id=round_id,
                lane=lane,
                reviewed_head=reviewed_head,
                reviewed_tree=reviewed_tree,
            )
        )
        return next_state
    raise ValueError(f"unsupported decision: {decision}")


def _auto_record_pending_decision_fix(
    state: dict[str, Any],
    *,
    current_head_value: str,
    state_dir: Path,
) -> dict[str, Any]:
    round_id, lane = _pending_decision(state)
    round_payload = _round_by_id(state, round_id)
    if bool(round_payload.get("review_blocked")) or _round_blocked_slots(round_payload):
        return state
    reviewed_head = str(round_payload.get("reviewed_head") or "").strip()
    if not reviewed_head or reviewed_head == current_head_value:
        return state
    findings = _apply_decision_to_ready_state(
        state, DECISION_FINDINGS, state_dir=state_dir, require_grade=False
    )
    return mark_fix_detected(findings, head=current_head_value)


def _auto_record_structured_decision(
    state: dict[str, Any], *, state_dir: Path
) -> dict[str, Any]:
    command = _auto_decision_command(state, state_dir=state_dir)
    if command is None:
        return state
    return _apply_decision_to_ready_state(state, command, state_dir=state_dir)


def _resume_progress(
    state: dict[str, Any], *, state_dir: Path | None = None
) -> dict[str, Any]:
    stage = state.get("stage")
    if stage == STAGE_CREATED and dict(state.get("deslop") or {}).get("status") in {
        DESLOP_STATUS_DONE,
        DESLOP_STATUS_CLOSED,
    }:
        identity = _current_cycle_identity_if_compatible(state)
        if identity and identity["head"] != dict(state.get("identity") or {}).get(
            "head"
        ):
            cleanup_head = str(
                dict(state.get("deslop") or {}).get("reviewed_head") or ""
            )
            fixes_applied = _is_material_fix_head(
                state, head=identity["head"], reviewed_head=cleanup_head
            ) and not bool(
                dict(identity.get("base_drift") or {}).get("patch_equivalent")
            )
            state = _with_current_identity(
                state,
                head=identity["head"],
                merge_base_head=identity["merge_base"],
                base_drift=identity.get("base_drift"),
            )
            state = _with_equivalent_base_drift_review_head(
                state, identity.get("base_drift")
            )
            state["validation"] = {
                key: "unknown"
                for key in ("focused", "full_suite", "ci", "review_green")
            }
            if _deslop_is_open(state) and fixes_applied:
                return mark_deslop_closed(state, fixes_applied=True)
            return state
        return state
    if stage in {STAGE_CREATED, STAGE_FOLLOWUP_PENDING}:
        return state
    if stage == STAGE_DECISION_PENDING:
        if (
            state_dir is not None
            and _pending_grade_payload(state, state_dir=state_dir) is not None
        ):
            return state
        try:
            identity = _current_cycle_identity_if_compatible(state)
        except OSError, ValueError:
            return state
        if identity:
            head = str(identity.get("head") or "").strip()
            base_drift = (
                identity.get("base_drift")
                if isinstance(identity.get("base_drift"), dict)
                else None
            )
            state = _with_current_identity(
                state,
                head=head,
                merge_base_head=str(identity.get("merge_base") or "").strip(),
                base_drift=base_drift,
            )
            state = _with_equivalent_base_drift_review_head(state, base_drift)
            if state_dir is None:
                return state
            if bool(dict(base_drift or {}).get("patch_equivalent")):
                return state
            return _auto_record_pending_decision_fix(
                state, current_head_value=head, state_dir=state_dir
            )
        return state
    if stage in {STAGE_REVIEW_GREEN, STAGE_LOCAL_GREEN_HANDOFF}:
        identity = _github_pending_head_change_identity(state)
        if not identity:
            return state
        head = str(identity.get("head") or "").strip()
        base_drift = (
            identity.get("base_drift")
            if isinstance(identity.get("base_drift"), dict)
            else None
        )
        next_state = _with_current_identity(
            state,
            head=head,
            merge_base_head=str(identity.get("merge_base") or "").strip(),
            base_drift=base_drift,
        )
        next_state = _with_equivalent_base_drift_review_head(next_state, base_drift)
        next_state["github_review"] = {"status": "unknown"}
        return mark_latest_profile_step_rerun_needed(next_state, head=head)
    if stage != STAGE_FIX_PENDING:
        return state

    active = dict(state.get("active_findings") or {})
    reviewed_head = str(active.get("reviewed_head") or "").strip()
    try:
        identity = _current_cycle_identity_if_compatible(state)
    except OSError, ValueError:
        identity = None
    if not identity:
        identity = _fix_pending_head_change_identity_if_compatible(state)
    head = ""
    if identity:
        head = str(identity.get("head") or "").strip()
        base_drift = (
            identity.get("base_drift")
            if isinstance(identity.get("base_drift"), dict)
            else None
        )
        state = _with_current_identity(
            state,
            head=head,
            merge_base_head=str(identity.get("merge_base") or "").strip(),
            base_drift=base_drift,
        )
        state = _with_equivalent_base_drift_review_head(state, base_drift)
    base_drift = dict(base_drift or {}) if identity else {}
    active = dict(state.get("active_findings") or {})
    reviewed_head = str(active.get("reviewed_head") or "").strip()
    if (
        head
        and reviewed_head
        and _is_material_fix_head(state, head=head, reviewed_head=reviewed_head)
        and not bool(base_drift.get("patch_equivalent"))
    ):
        return mark_fix_detected(state, head=head)
    return _with_fix_action(state)


def _apply_profile_resolution(state: dict[str, Any], resolution: Any) -> dict[str, Any]:
    state["selection"]["reason"] = resolution.selection_reason

    def step_payload(step: Any) -> dict[str, Any]:
        payload = {
            "kind": step.kind,
            "name": step.name,
        }
        if step.kind == "arena":
            payload["lane"] = step.lane
            payload["task_class"] = step.task_class
            payload["rating_pool_id"] = step.rating_pool_id
            if step.reporting_pool:
                payload["reporting_pool"] = True
            payload["variant_groups"] = [list(group) for group in step.variant_groups]
            if step.variant_ids:
                payload["variant_ids"] = list(step.variant_ids)
            return payload
        payload.update(
            {
                "count": step.count,
                "model": step.model,
                "reasoning_effort": step.reasoning_effort,
                "service_tier": step.service_tier,
            }
        )
        if step.rerun_on_findings:
            payload["rerun_on_findings"] = True
        if step.max_review_rounds is not None:
            payload["max_review_rounds"] = step.max_review_rounds
        return payload

    state["review_plan"] = {
        "steps": [step_payload(step) for step in resolution.steps],
    }
    return state


def _with_current_identity(
    state: dict[str, Any],
    *,
    head: str,
    merge_base_head: str,
    base_drift: dict[str, Any] | None = None,
) -> dict[str, Any]:
    next_state = dict(state)
    identity = dict(next_state.get("identity") or {})
    identity["head"] = head
    identity["merge_base"] = merge_base_head
    next_state["identity"] = identity
    review_heads = dict(next_state.get("review_heads") or {})
    review_heads["head"] = head
    review_heads["merge_base"] = merge_base_head
    next_state["review_heads"] = review_heads
    if base_drift is not None:
        next_state["base_drift"] = base_drift
    return next_state


def _with_equivalent_base_drift_review_head(
    state: dict[str, Any],
    base_drift: dict[str, Any] | None,
) -> dict[str, Any]:
    drift = dict(base_drift or {})
    old_head = str(drift.get("reviewed_head") or "").strip()
    new_head = str(drift.get("equivalent_reviewed_head") or "").strip()
    if not old_head or not new_head or old_head == new_head:
        return state
    next_state = dict(state)
    deslop = dict(next_state.get("deslop") or {})
    if deslop.get("reviewed_head") == old_head:
        next_state["deslop"] = {**deslop, "reviewed_head": new_head}
    rounds = []
    for item in list(next_state.get("rounds") or []):
        if not isinstance(item, dict):
            rounds.append(item)
            continue
        next_item = dict(item)
        if str(next_item.get("reviewed_head") or "").strip() == old_head:
            next_item["reviewed_head"] = new_head
        rounds.append(next_item)
    next_state["rounds"] = rounds
    active = dict(next_state.get("active_findings") or {})
    if str(active.get("reviewed_head") or "").strip() == old_head:
        active["reviewed_head"] = new_head
        next_state["active_findings"] = active
    decisions = []
    for item in list(next_state.get("decisions") or []):
        if not isinstance(item, dict):
            decisions.append(item)
            continue
        next_item = dict(item)
        if str(next_item.get("reviewed_head") or "").strip() == old_head:
            next_item["reviewed_head"] = new_head
        decisions.append(next_item)
    if decisions:
        next_state["decisions"] = decisions
    review_heads = dict(next_state.get("review_heads") or {})
    for key in ("last_reviewed_head", "last_fix_head", "last_followup_head", "head"):
        if str(review_heads.get(key) or "").strip() == old_head:
            review_heads[key] = new_head
    next_state["review_heads"] = review_heads
    progress = dict(next_state.get("review_progress") or {})
    completed = []
    for item in list(progress.get("completed_steps") or []):
        if not isinstance(item, dict):
            completed.append(item)
            continue
        next_item = dict(item)
        if str(next_item.get("reviewed_head") or "").strip() == old_head:
            next_item["reviewed_head"] = new_head
        completed.append(next_item)
    if completed:
        progress["completed_steps"] = completed
        next_state["review_progress"] = progress
    pending = dict(next_state.get("pending_action") or {})
    fix_verification = dict(pending.get("fix_verification") or {})
    if str(fix_verification.get("findings_reviewed_head") or "").strip() == old_head:
        fix_verification["findings_reviewed_head"] = new_head
        pending["fix_verification"] = fix_verification
        next_state["pending_action"] = pending
    return next_state


def _cycle_cli_skips_deslop(state: dict[str, Any]) -> bool:
    deslop = dict(state.get("deslop") or {})
    if str(deslop.get("status") or "").strip() != DESLOP_STATUS_SKIPPED:
        return False
    source = str(deslop.get("source") or "cli").strip()
    return source == "cli"


def _deslop_is_done_or_skipped(state: dict[str, Any]) -> bool:
    deslop = dict(state.get("deslop") or {})
    status = str(deslop.get("status") or "").strip()
    return status == DESLOP_STATUS_DONE or (
        not bool(deslop.get("tracked")) and status == DESLOP_STATUS_SKIPPED
    )


def _green_cycle_needs_current_head_signoff(
    state: dict[str, Any], *, head: str
) -> bool:
    if state.get("stage") not in {STAGE_REVIEW_GREEN, STAGE_LOCAL_GREEN_HANDOFF}:
        return False
    github_status = _github_review_status(state)
    if (
        _mode_label(state) == "fast" and github_status == "unknown"
    ) or github_status in {GITHUB_RESULT_CLEAN, GITHUB_RESULT_WAIVED}:
        return False
    summary = review_ladder_summary(state, current_head=head)
    return summary.get("review_ladder") == "invalidated"


def _continuation_head_match_kind(
    state: dict[str, Any], *, review_root: Path, head: str
) -> str | None:
    identity = dict(state.get("identity") or {})
    recorded_head = str(identity.get("head") or "").strip()
    if not recorded_head:
        return None
    if recorded_head == head:
        return "exact"
    stage = str(state.get("stage") or "")
    if stage == STAGE_CREATED:
        if dict(state.get("deslop") or {}).get("status") in {
            DESLOP_STATUS_DONE,
            DESLOP_STATUS_CLOSED,
        }:
            return "changed"
        if not _deslop_is_done_or_skipped(state):
            return None
        try:
            if is_ancestor(review_root, recorded_head, head) or is_ancestor(
                review_root, head, recorded_head
            ):
                return None
            return "amended"
        except ValueError:
            return None
    if stage in {
        STAGE_DECISION_PENDING,
        STAGE_FIX_PENDING,
        STAGE_FOLLOWUP_PENDING,
        "decision-required",
    }:
        return "changed"
    if _green_cycle_needs_current_head_signoff(state, head=head):
        return "changed"
    return None


def _compatible_continuation_cycle(
    *,
    state_dir: Path,
    review_root: Path,
    base: str,
    branch: str | None,
    head: str,
    merge_base_head: str,
    effective_mode: str,
    arena_enabled: bool,
    review_brief: str | None = None,
    skip_deslop: bool = False,
) -> dict[str, Any] | None:
    normalized_cwd = normalize_cwd(str(review_root))
    normalized_branch = _normalized_branch(branch)
    candidates: list[tuple[float, str, dict[str, Any], dict[str, Any] | None]] = []
    directory = cycles_dir(state_dir)
    if not directory.exists():
        return None
    for path in directory.glob("*.json"):
        try:
            state = load_cycle_by_key(state_dir, path.stem)
        except ValueError:
            continue
        if not isinstance(state, dict):
            continue
        state_stage = str(state.get("stage") or "")
        if state_stage not in CONTINUATION_REDIRECT_STAGES and not (
            _green_cycle_needs_current_head_signoff(state, head=head)
        ):
            continue
        if isinstance(state.get("superseded_by"), dict):
            continue
        identity = dict(state.get("identity") or {})
        if str(identity.get("cwd") or "") != normalized_cwd:
            continue
        if str(identity.get("base") or "").strip() != str(base or "").strip():
            continue
        if _normalized_branch(str(identity.get("branch") or "")) != normalized_branch:
            continue
        recorded_merge_base = str(identity.get("merge_base") or "").strip()
        base_drift = None
        if recorded_merge_base != str(merge_base_head or "").strip():
            reviewed_head = _base_drift_reviewed_head(state)
            if not reviewed_head:
                continue
            try:
                base_drift = _allowed_base_drift(
                    review_root=review_root,
                    recorded_merge_base=recorded_merge_base,
                    current_merge_base=str(merge_base_head or "").strip(),
                    reviewed_head=reviewed_head,
                    current_head_value=head,
                )
            except OSError, ValueError:
                continue
            if base_drift is None:
                continue
            if state_stage == STAGE_CREATED and not bool(
                base_drift.get("patch_equivalent")
            ):
                continue
            if (
                state_stage in {STAGE_DECISION_PENDING, STAGE_FIX_PENDING}
                and not str(base_drift.get("equivalent_reviewed_head") or "").strip()
            ):
                continue
        mode = dict(state.get("mode") or {})
        state_mode = str(mode.get("effective") or mode.get("requested") or "").strip()
        decision_pending = state_stage == "decision-required"
        if state_mode != effective_mode and not decision_pending:
            continue
        state_arena_enabled = any(
            isinstance(step, dict) and step.get("kind") == "arena"
            for step in dict(state.get("review_plan") or {}).get("steps", [])
        )
        if state_arena_enabled != arena_enabled and not decision_pending:
            continue
        if _cycle_cli_skips_deslop(state) != bool(skip_deslop) and not decision_pending:
            continue
        match_kind = _continuation_head_match_kind(
            state, review_root=review_root, head=head
        )
        if match_kind is None:
            continue
        candidates.append((path.stat().st_mtime, match_kind, state, base_drift))
    if not candidates:
        return None
    exact_candidates = [
        candidate for candidate in candidates if candidate[1] == "exact"
    ]
    selected_candidates = exact_candidates or candidates
    if len(selected_candidates) > 1:
        public_ids = sorted(
            str(candidate[2].get("public_id") or candidate[2].get("cycle_key") or "")
            for candidate in selected_candidates
        )
        raise ValueError(
            "multiple active review cycles match this repo/base/branch/merge-base; "
            f"rerun with --id for one of: {', '.join(public_ids)}"
        )
    selected_base_drift = selected_candidates[0][3]
    selected_state = selected_candidates[0][2]
    _reject_review_brief_replacement(selected_state, review_brief)
    if selected_state.get("stage") == STAGE_CREATED and dict(
        selected_state.get("deslop") or {}
    ).get("status") in {DESLOP_STATUS_DONE, DESLOP_STATUS_CLOSED}:
        return _resume_progress(selected_state, state_dir=state_dir)
    resumed = _with_current_identity(
        selected_state,
        head=head,
        merge_base_head=merge_base_head,
        base_drift=selected_base_drift,
    )
    if str(resumed.get("stage") or "") in {
        STAGE_REVIEW_GREEN,
        STAGE_LOCAL_GREEN_HANDOFF,
    }:
        if bool(dict(selected_base_drift or {}).get("patch_equivalent")):
            return _with_equivalent_base_drift_review_head(resumed, selected_base_drift)
        return mark_latest_profile_step_rerun_needed(resumed, head=head)
    return resumed


def _requested_model_override(args: argparse.Namespace) -> dict[str, str]:
    model = str(getattr(args, "model", "") or "").strip()
    if model and "/" in model and not model.startswith(OPENCODE_MODEL_PREFIX):
        model = f"{OPENCODE_MODEL_PREFIX}{model}"
    if model:
        split_review_backend_model(model)
    reasoning = str(getattr(args, "reasoning", "") or "").strip()
    override: dict[str, str] = {}
    if model:
        override["model"] = model
    if reasoning:
        override["reasoning"] = reasoning
    return override


def _config_with_model_override(
    config: dict[str, Any], override: dict[str, str]
) -> dict[str, Any]:
    model = str(override.get("model") or "").strip()
    reasoning = str(override.get("reasoning") or "").strip()
    if not model and not reasoning:
        return config
    merged = deepcopy(config)
    orchestrator = merged.setdefault("orchestrator", {})
    defaults = dict(orchestrator.get("stable_defaults") or {})
    for ref, group in (
        ("signoff_normal_model", "normal"),
        ("signoff_deep_model", "deep"),
    ):
        existing_model, existing_effort, existing_tier = parse_model_label(
            defaults.get(ref), field=ref
        )
        chosen_model = model or existing_model
        effort = reasoning or existing_effort
        if chosen_model.startswith(OPENCODE_MODEL_PREFIX):
            opencode_name = chosen_model[len(OPENCODE_MODEL_PREFIX) :]
            if reasoning:
                effort = validate_opencode_reasoning_effort(opencode_name, reasoning)
            else:
                try:
                    effort = validate_opencode_reasoning_effort(opencode_name, effort)
                except ValueError:
                    effort = default_opencode_reasoning_effort(opencode_name)
        tier = None if chosen_model.startswith(OPENCODE_MODEL_PREFIX) else existing_tier
        defaults[ref] = "-".join([chosen_model, effort] + ([tier] if tier else []))
        section = merged.get(group)
        if isinstance(section, dict):
            section["model"] = chosen_model
            if reasoning:
                section["reasoning"] = reasoning
            if chosen_model.startswith(OPENCODE_MODEL_PREFIX):
                section.pop("service_tier", None)
    orchestrator["stable_defaults"] = defaults
    return merged


def _config_without_arena(config: dict[str, Any], *, disabled: bool) -> dict[str, Any]:
    if not disabled:
        return config
    merged = deepcopy(config)
    merged["arena"]["enabled"] = False
    return merged


def _record_model_override(state: dict[str, Any], override: dict[str, str]) -> None:
    if override:
        state["model_override"] = dict(override)


def _reject_model_override_on_existing(
    override: dict[str, str], state: dict[str, Any]
) -> None:
    if not override:
        return
    public_id = str(state.get("public_id") or "").strip()
    raise ValueError(
        "--model/--reasoning apply when a review is created; "
        f"{f'review {public_id} already' if public_id else 'a review already'} "
        "has a frozen plan. Remove the override to continue it, or start a new "
        "review once this one closes."
    )


def _create_or_resume_cycle(
    *, args: argparse.Namespace, state_dir: Path
) -> dict[str, Any]:
    mode = str(args.mode or "normal")
    review_root = resolve_repo_root(args.cd)
    head = current_head(review_root)
    branch = current_branch(review_root)
    base_info = effective_base_ref(review_root, args.base)
    base = str(base_info["base"])
    merge_base_head = merge_base(review_root, base, "HEAD")
    model_override = _requested_model_override(args)
    config = _config_with_model_override(load_config(state_dir), model_override)
    arena_disabled = bool(args.no_arena) or _repo_has_no_arena(review_root)
    config = _config_without_arena(config, disabled=arena_disabled)
    resolution = resolve_orchestrator_profile(
        config, mode=mode, selection=_configured_selection(config)
    )
    skip_deslop = bool(args.skip_deslop)
    continuation = _compatible_continuation_cycle(
        state_dir=state_dir,
        review_root=review_root,
        base=base,
        branch=branch,
        head=head,
        merge_base_head=merge_base_head,
        effective_mode=resolution.effective_mode,
        arena_enabled=any(step.kind == "arena" for step in resolution.steps),
        review_brief=args.review_brief,
        skip_deslop=skip_deslop,
    )
    if continuation is not None:
        _reject_model_override_on_existing(model_override, continuation)
        return _apply_runtime_options(continuation, args)
    state = create_cycle(
        cwd=review_root,
        base=base,
        branch=branch,
        head=head,
        merge_base=merge_base_head,
        requested_mode=resolution.requested_mode,
        effective_mode=resolution.effective_mode,
        selection=resolution.requested_selection,
        effective_selection=resolution.effective_selection,
        deslop_enabled=not skip_deslop,
        deslop_skip_source="cli" if skip_deslop else None,
        cycle_token=",".join(
            token
            for token, enabled in (
                ("skip-deslop", skip_deslop),
                ("no-arena", arena_disabled),
            )
            if enabled
        )
        or None,
        review_brief=args.review_brief,
    )
    if str(base_info["requested_base"]) != base:
        state["identity"]["requested_base"] = str(base_info["requested_base"])
    for key in EFFECTIVE_BASE_METADATA_KEYS:
        if key in base_info:
            state["identity"][key] = base_info[key]
    state = _apply_runtime_options(state, args)
    existing = load_cycle_by_key(state_dir, str(state["cycle_key"]))
    if existing is not None:
        _reject_model_override_on_existing(model_override, existing)
        _reject_review_brief_replacement(existing, args.review_brief)
        return _apply_runtime_options(existing, args)
    state = _apply_profile_resolution(state, resolution)
    if arena_disabled:
        state["arena_disabled"] = "cli" if args.no_arena else "repository"
    _record_model_override(state, model_override)
    return state


def _apply_runtime_options(
    state: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    if not bool(getattr(args, "wsl", False)):
        return state
    next_state = dict(state)
    runtime = dict(next_state.get("runtime") or {})
    runtime["allow_unsafe_windows_wsl_fallback"] = True
    next_state["runtime"] = runtime
    return next_state


def _copy_runtime_options(
    target: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    runtime = dict(source.get("runtime") or {})
    if not runtime:
        return target
    next_target = dict(target)
    next_target["runtime"] = runtime
    return next_target


def _current_restart_identity(
    state: dict[str, Any], *, require_exact: bool = True
) -> tuple[Path, str, str | None, str, str]:
    identity = dict(state.get("identity") or {})
    cwd = str(identity.get("cwd") or "").strip()
    if not cwd:
        raise ValueError("review cycle is missing cwd and cannot be restarted")
    base = str(identity.get("base") or "").strip()
    if not base:
        raise ValueError("review cycle is missing base and cannot be restarted")
    review_root = cwd_path_from_normalized(cwd)
    old_branch = _normalized_branch(str(identity.get("branch") or ""))
    branch = _normalized_branch(current_branch(review_root))
    if branch != old_branch:
        raise ValueError(
            f"cannot restart review cycle on branch {branch or 'HEAD'}; expected {old_branch or 'HEAD'}"
        )
    if has_worktree_changes(review_root):
        raise ValueError(
            "cannot restart review cycle with a dirty worktree; commit or stash changes, then rerun"
        )
    head = current_head(review_root)
    expected_head = str(identity.get("head") or "").strip()
    if require_exact and head != expected_head:
        raise ValueError(
            "cannot restart review cycle after HEAD changed; start a new review instead"
        )
    merge_base_head = merge_base(review_root, base, "HEAD")
    expected_merge_base = str(identity.get("merge_base") or "").strip()
    if require_exact and merge_base_head != expected_merge_base:
        raise ValueError(
            "cannot restart review cycle after merge-base changed; start a new review instead"
        )
    return review_root, base, branch, head, merge_base_head


def _create_successor_cycle(
    *,
    state: dict[str, Any],
    state_dir: Path,
    target_mode: str,
    reason: str,
    kind: str,
    require_exact_identity: bool,
    review_brief: str | None,
) -> tuple[dict[str, Any], bool]:
    if isinstance(state.get("superseded_by"), dict):
        replacement = str(
            dict(state.get("superseded_by") or {}).get("review") or ""
        ).strip()
        raise ValueError(
            f"review cycle is already superseded{f' by {replacement}' if replacement else ''}"
        )
    current_mode = _current_mode(state)
    review_root, base, branch, head, merge_base_head = _current_restart_identity(
        state, require_exact=require_exact_identity
    )
    model_override = dict(state.get("model_override") or {})
    config = _config_with_model_override(load_config(state_dir), model_override)
    arena_disabled = state.get("arena_disabled") == "cli" or _repo_has_no_arena(
        review_root
    )
    config = _config_without_arena(config, disabled=arena_disabled)
    selection = str(
        dict(state.get("selection") or {}).get("requested")
        or _configured_selection(config)
    ).strip()
    resolution = resolve_orchestrator_profile(
        config, mode=target_mode, selection=selection
    )
    source_deslop = dict(state.get("deslop") or {})
    source_skipped_deslop = _cycle_cli_skips_deslop(state)
    deslop_skip_source = (
        str(source_deslop.get("source") or "cli").strip()
        if source_skipped_deslop
        else None
    )
    restart_token = (
        f"{state.get('cycle_key')}:{target_mode}"
        if kind == "mode-restart"
        else f"{state.get('cycle_key')}:{kind}"
    )
    replacement = create_cycle(
        cwd=review_root,
        base=base,
        branch=branch,
        head=head,
        merge_base=merge_base_head,
        requested_mode=resolution.requested_mode,
        effective_mode=resolution.effective_mode,
        selection=resolution.requested_selection,
        effective_selection=resolution.effective_selection,
        deslop_enabled=not source_skipped_deslop,
        deslop_skip_source=deslop_skip_source,
        restart_token=restart_token,
        review_brief=review_brief,
    )
    existing = load_cycle_by_key(state_dir, str(replacement["cycle_key"]))
    if existing is not None:
        if existing.get("review_brief") != review_brief:
            raise ValueError("review cycle already has a different successor brief")
        return _copy_runtime_options(existing, state), True
    replacement = _copy_runtime_options(replacement, state)
    if arena_disabled:
        replacement["arena_disabled"] = (
            "cli" if state.get("arena_disabled") == "cli" else "repository"
        )
    _record_model_override(replacement, model_override)
    replacement["restart"].update(
        {
            "supersedes": str(state.get("public_id") or ""),
            "supersedes_cycle_key": str(state.get("cycle_key") or ""),
            "from_mode": current_mode,
            "reason": reason,
            "kind": kind,
        }
    )
    return _apply_profile_resolution(replacement, resolution), False


def _start_successor_cycle(
    state: dict[str, Any],
    *,
    state_dir: Path,
    target_mode: str,
    reason: str,
    kind: str,
    require_exact_identity: bool,
    review_brief: str | None,
) -> dict[str, Any]:
    replacement, _ = _create_successor_cycle(
        state=state,
        state_dir=state_dir,
        target_mode=target_mode,
        reason=reason,
        kind=kind,
        require_exact_identity=require_exact_identity,
        review_brief=review_brief,
    )
    superseded = abort_cycle(state, reason=reason)
    superseded["superseded_by"] = {
        "mode": target_mode,
        "reason": reason,
        "kind": kind,
    }
    saved_replacement, _ = reserve_cycle_successor(
        state_dir,
        source=superseded,
        successor=replacement,
    )
    return _resume_reserved_successor(saved_replacement, state_dir=state_dir)


def _resume_reserved_successor(
    state: dict[str, Any], *, state_dir: Path
) -> dict[str, Any]:
    if state.get("stage") not in {STAGE_CREATED, STAGE_RUNNING}:
        return state
    cycle_key = str(state.get("cycle_key") or "").strip()
    try:
        with orchestrator_store_lock(
            state_dir=state_dir,
            name=f"successor-{cycle_key}",
            timeout_seconds=1,
        ):
            current = load_cycle_by_key(state_dir, cycle_key) or state
            if current.get("stage") not in {STAGE_CREATED, STAGE_RUNNING}:
                return current
            advanced = _advance_without_decision(current, state_dir=state_dir)
            return save_cycle(state_dir, advanced)
    except TimeoutError:
        raise SuccessorAdvanceBusyError(
            load_cycle_by_key(state_dir, cycle_key) or state
        ) from None


def _is_successor_cycle(state: dict[str, Any]) -> bool:
    restart = dict(state.get("restart") or {})
    return bool(str(restart.get("token") or "").strip())


def _successor_needs_locked_resume(state: dict[str, Any]) -> bool:
    return _is_successor_cycle(state) and state.get("stage") in {
        STAGE_CREATED,
        STAGE_RUNNING,
    }


def _restart_cycle(
    state: dict[str, Any], *, state_dir: Path, target_mode: str, reason: str
) -> dict[str, Any]:
    _validate_restart_mode(state, target_mode)
    return _start_successor_cycle(
        state,
        state_dir=state_dir,
        target_mode=target_mode,
        reason=reason,
        kind="mode-restart",
        require_exact_identity=True,
        review_brief=state.get("review_brief"),
    )


def _restart_review_brief(
    state: dict[str, Any], *, state_dir: Path, review_brief: str, reason: str
) -> dict[str, Any]:
    if not review_brief.strip():
        raise ValueError("--restart-brief cannot be blank")
    current_brief = state.get("review_brief")
    if not isinstance(current_brief, str) or not current_brief.strip():
        raise ValueError("review cycle has no frozen review brief to replace")
    if review_brief == current_brief:
        raise ValueError("--restart-brief must differ from the frozen review brief")

    convergence = convergence_summary(state)
    if (
        state.get("stage")
        not in {
            STAGE_CREATED,
            STAGE_DECISION_PENDING,
            STAGE_REVIEW_GREEN,
            STAGE_LOCAL_GREEN_HANDOFF,
        }
        or state.get("active_findings") is not None
        or convergence.get("status") != "ACTIVE"
        or int(convergence.get("accepted_findings_heads") or 0) != 0
    ):
        raise ValueError(
            "cannot replace review brief after findings or convergence activity; "
            "follow the existing review action"
        )

    review_root, _, _, head, merge_base_head = _current_restart_identity(
        state, require_exact=False
    )
    identity = dict(state.get("identity") or {})
    expected_head = str(identity.get("head") or "").strip()
    expected_merge_base = str(identity.get("merge_base") or "").strip()
    if merge_base_head != expected_merge_base:
        raise ValueError(
            "cannot replace review brief after merge-base changed; start a new review instead"
        )
    if head != expected_head and not is_ancestor(review_root, expected_head, head):
        raise ValueError(
            "cannot replace review brief after review lineage changed; start a new review instead"
        )

    return _start_successor_cycle(
        state,
        state_dir=state_dir,
        target_mode=_current_mode(state),
        reason=reason,
        kind="review-brief-restart",
        require_exact_identity=False,
        review_brief=review_brief,
    )


def _advance_without_decision(
    state: dict[str, Any], *, state_dir: Path
) -> dict[str, Any]:
    ready_state = state
    ran_expensive_step = False

    def persist_running(next_state: dict[str, Any]) -> dict[str, Any]:
        saved = save_cycle(state_dir, next_state)
        return saved

    for _ in range(6):
        if deslop_is_ready(ready_state) or _deslop_is_open(ready_state):
            resumed = _resume_progress(ready_state, state_dir=state_dir)
            resumed = _blocked_decision_recovery_state(resumed, state_dir=state_dir)
        else:
            resumed = ready_state
        decided = (
            _auto_record_structured_decision(resumed, state_dir=state_dir)
            if deslop_is_ready(resumed)
            else resumed
        )
        if decided != resumed:
            ready_state = decided
            continue
        if ran_expensive_step:
            return resumed
        result = run_one_expensive_step(
            resumed, state_dir=state_dir, persist_state=persist_running
        )
        if result.ran_step:
            ran_expensive_step = True
            ready_state = result.state
            continue
        if _run_arena_recovery_backend_once(resumed, state_dir=state_dir):
            ready_state = resumed
            continue
        return _resume_progress(resumed, state_dir=state_dir)
    return ready_state


def _apply_decision(
    state: dict[str, Any], decision: str, *, state_dir: Path
) -> dict[str, Any]:
    ready_state = (
        _resume_progress(state, state_dir=state_dir)
        if deslop_is_ready(state)
        else state
    )
    if not deslop_is_ready(ready_state):
        ready_state = run_one_expensive_step(ready_state, state_dir=state_dir).state
        if not deslop_is_ready(ready_state):
            return ready_state
        ready_state = _resume_progress(ready_state, state_dir=state_dir)
    try:
        return _apply_decision_to_ready_state(
            ready_state, decision, state_dir=state_dir
        )
    except ValueError as exc:
        if (
            str(exc) == NO_DECISION_PENDING_MESSAGE
            and ready_state.get("stage") != STAGE_DECISION_PENDING
        ):
            raise NoDecisionPendingError(ready_state) from exc
        raise


def _has_validation_status(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, name) is not None
        for name in ("focused_validation", "full_suite", "ci")
    )


def _record_validation_status(
    state: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    return record_validation_statuses(
        state,
        focused=args.focused_validation,
        full_suite=args.full_suite,
        ci=args.ci,
        validation_note=args.validation_note,
    )


def _record_github_result(
    state: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    _require_local_green_for_github_review(state)
    reviewed_head = str(
        dict(state.get("review_heads") or {}).get("last_reviewed_head")
        or dict(state.get("identity") or {}).get("head")
        or ""
    ).strip()
    return record_github_result(
        state,
        result=str(args.github_result),
        note=args.github_note,
        reviewed_head=reviewed_head or None,
        reviewed_tree=_reviewed_tree(state, reviewed_head),
    )


def _last_decision_is_clean_followup(state: dict[str, Any]) -> bool:
    for item in reversed(list(state.get("decisions") or [])):
        if not isinstance(item, dict):
            continue
        return (
            str(item.get("lane") or "") == FOLLOWUP_LANE
            and str(item.get("command") or "") == DECISION_CLEAN
        )
    return False


def _review_step_label(state: dict[str, Any], pending: dict[str, Any]) -> str:
    step = str(pending.get("step") or "").strip() or "next review step"
    try:
        position = int(pending.get("step_index")) + 1
    except TypeError, ValueError:
        position = None
    steps = list(dict(state.get("review_plan") or {}).get("steps") or [])
    if position is not None and steps:
        return f"review step {position}/{len(steps)} {step}"
    return f"review step {step}"


def _continuation_note(state: dict[str, Any]) -> str | None:
    pending = dict(state.get("pending_action") or {})
    kind = str(pending.get("kind") or "").strip()
    if kind == "run-review-step":
        if not _last_decision_is_clean_followup(state):
            return None
        label = _review_step_label(state, pending)
        return f"Clean follow-up is not final signoff; run {label} before treating the review as green."
    return None


def _github_handoff_action(state: dict[str, Any], *, state_dir: Path) -> dict[str, Any]:
    public_id = str(state.get("public_id") or "").strip()
    action: dict[str, Any] = {
        "cmd": _github_review_action_command(public_id, state_dir=state_dir),
        "after": "PR create/update",
        "result": {
            GITHUB_RESULT_CLEAN: _github_result_command(
                public_id, GITHUB_RESULT_CLEAN, state_dir=state_dir
            ),
            GITHUB_RESULT_FINDINGS: _github_result_command(
                public_id, GITHUB_RESULT_FINDINGS, state_dir=state_dir
            ),
            GITHUB_RESULT_WAIVED: _github_result_command(
                public_id,
                GITHUB_RESULT_WAIVED,
                "--github-note",
                "REASON",
                state_dir=state_dir,
            ),
        },
    }
    github_review = dict(state.get("github_review") or {})
    if str(github_review.get("status") or "") == GITHUB_RESULT_FINDINGS:
        action["note"] = (
            "GitHub findings were fixed and locally signed off; request GitHub review again."
        )
    blockers = validation_blockers(state)
    if blockers:
        action["blocked_by"] = blockers
    return action


def _validation_status_command(
    public_id: str, blockers: list[str], *, state_dir: Path
) -> str:
    args: list[str] = []
    for blocker in blockers:
        key, value = blocker.split(":", 1)
        if key == "full_suite":
            args.extend(
                [
                    "--full-suite",
                    "waived" if value == "waived_without_note" else "FULL_SUITE_STATUS",
                ]
            )
        if key == "ci":
            args.extend(
                ["--ci", "waived" if value == "waived_without_note" else "CI_STATUS"]
            )
    if any(blocker.endswith(":waived_without_note") for blocker in blockers):
        args.extend(["--validation-note", "WAIVER_REASON"])
    return _review_command(public_id, *args, state_dir=state_dir)


def _validation_blocker_action(
    public_id: str, blockers: list[str], *, state_dir: Path
) -> dict[str, Any]:
    return {
        "cmd": _validation_status_command(public_id, blockers, state_dir=state_dir),
        "blocked_by": blockers,
        "note": 'GitHub result is recorded; replace status placeholders with passed, or waived and append --validation-note "reason", before PR-final or merge-ready.',
    }


def _github_review_is_terminal(state: dict[str, Any]) -> bool:
    github_review = dict(state.get("github_review") or {})
    status = str(github_review.get("status") or "").strip()
    if status not in {GITHUB_RESULT_CLEAN, GITHUB_RESULT_WAIVED}:
        return False
    reviewed_head = str(github_review.get("reviewed_head") or "").strip()
    if not reviewed_head:
        return False
    current_head_value = _identity_head(state)
    if not current_head_value:
        current_head_value = str(
            dict(state.get("review_heads") or {}).get("last_reviewed_head") or ""
        ).strip()
    if not current_head_value:
        current_head_value = str(
            dict(state.get("identity") or {}).get("head") or ""
        ).strip()
    if not current_head_value:
        return False
    if reviewed_head == current_head_value:
        return True
    summary = review_ladder_summary(state, current_head=current_head_value)
    return summary.get("review_ladder") != "invalidated"


def _github_review_status(state: dict[str, Any]) -> str:
    return (
        str(dict(state.get("github_review") or {}).get("status") or "unknown").strip()
        or "unknown"
    )


def _github_pending_head_change_identity(
    state: dict[str, Any],
) -> dict[str, Any] | None:
    if state.get("stage") not in {STAGE_REVIEW_GREEN, STAGE_LOCAL_GREEN_HANDOFF}:
        return None
    try:
        identity = _current_cycle_identity_if_compatible(state)
    except OSError, ValueError:
        return None
    if not identity:
        try:
            _, _, _, head, current_merge_base = _current_restart_identity(
                state, require_exact=False
            )
        except AttributeError, OSError, ValueError:
            return None
        identity = {
            "head": head,
            "merge_base": current_merge_base,
            "base_drift": None,
        }
    head = str(identity.get("head") or "").strip()
    current_merge_base = str(identity.get("merge_base") or "").strip()
    deslop = dict(state.get("deslop") or {})
    signoff_head = str(
        dict(state.get("review_heads") or {}).get("last_reviewed_head")
        or dict(state.get("identity") or {}).get("head")
        or ""
    ).strip()
    terminal = _github_review_status(state) in {"clean", "waived"}
    if (
        str(deslop.get("status") or "").strip() != DESLOP_STATUS_SKIPPED or terminal
    ) and (
        signoff_head != head
        or str(dict(state.get("identity") or {}).get("merge_base") or "").strip()
        != current_merge_base
    ):
        return identity
    if bool(dict(identity.get("base_drift") or {}).get("patch_equivalent")):
        return None
    summary = review_ladder_summary(state, current_head=head)
    if summary.get("review_ladder") != "invalidated":
        return None
    reviewed_head = str(summary.get("reviewed_head") or "").strip()
    current_head_value = str(summary.get("current_head") or "").strip()
    if (
        not reviewed_head
        or not current_head_value
        or reviewed_head == current_head_value
    ):
        return None
    return identity


def _github_terminal_action(
    state: dict[str, Any], public_id: str, *, state_dir: Path
) -> dict[str, Any] | None:
    blockers = validation_blockers(state)
    if blockers:
        return _validation_blocker_action(public_id, blockers, state_dir=state_dir)
    return None


def _deslop_is_open(state: dict[str, Any]) -> bool:
    deslop = dict(state.get("deslop") or {})
    return (
        bool(deslop.get("tracked"))
        and str(deslop.get("status") or "").strip() == DESLOP_STATUS_DONE
    )


def _with_deslop_done_action(
    state: dict[str, Any],
    action: dict[str, Any] | None,
    public_id: str,
    *,
    state_dir: Path,
) -> dict[str, Any] | None:
    if not _deslop_is_open(state):
        return action
    deslop = dict(state.get("deslop") or {})
    if deslop.get("conformance") == "MATERIALLY_DRIFTED":
        return {
            "choices": {
                "revise": _review_command(public_id, state_dir=state_dir),
                "dismiss": _review_command(
                    public_id,
                    "--deslop-done",
                    "--reason",
                    "<why the findings are dismissed>",
                    state_dir=state_dir,
                ),
            },
            "note": "Revise valid findings and commit before rerunning, or dismiss them with a reason.",
        }
    findings = deslop.get("decision") == "findings"
    return {
        "cmd": _deslop_done_command(public_id, state_dir=state_dir),
        "note": (
            "Address or dismiss the closure findings, then acknowledge the exact-head result."
            if findings
            else "Acknowledge the exact-head closure before continuing."
        ),
    }


def _public_convergence(state: dict[str, Any]) -> dict[str, Any] | None:
    summary = convergence_summary(state)
    if summary.get("status") == "ACTIVE" and not summary.get("accepted_findings_heads"):
        return None
    return summary


def _action_payload(state: dict[str, Any], *, state_dir: Path) -> dict[str, Any] | None:
    public_id = str(state.get("public_id") or "").strip()
    action: dict[str, Any] | None
    stage = state.get("stage")
    if stage in {"decision-required", "convergence-decided"}:
        convergence = convergence_summary(state)
        if convergence.get("status") != "DECISION_REQUIRED":
            return None
        return {
            "choices": {
                decision: _review_command(
                    public_id,
                    "--convergence-decision",
                    decision.lower(),
                    state_dir=state_dir,
                )
                for decision in convergence["recommendations"]
            },
            "note": "The caller must commit one decision; Review Suite will not change scope automatically.",
        }
    if stage == STAGE_ABORTED and isinstance(state.get("superseded_by"), dict):
        replacement = dict(state.get("superseded_by") or {})
        replacement_id = str(replacement.get("review") or "").strip()
        if replacement_id:
            action = {
                "cmd": _review_command(replacement_id, state_dir=state_dir),
                "note": f"Review {public_id} was superseded by {replacement_id}.",
            }
            return _with_deslop_done_action(
                state, action, public_id, state_dir=state_dir
            )
    arena_recovery = _arena_recovery_action(
        state, state_dir=state_dir, public_id=public_id
    )
    if arena_recovery:
        return _with_deslop_done_action(
            state, arena_recovery, public_id, state_dir=state_dir
        )
    if stage == STAGE_DECISION_PENDING:
        blocked_action = _blocked_decision_action(
            state, state_dir=state_dir, public_id=public_id
        )
        if blocked_action:
            return _with_deslop_done_action(
                state, blocked_action, public_id, state_dir=state_dir
            )
        grade = _arena_grade_command(state, state_dir=state_dir)
        if grade:
            action = {
                "cmd": grade,
                "note": "Grade the arena round, then rerun this review id to continue.",
                "next": _review_command(public_id, state_dir=state_dir),
            }
            return _with_deslop_done_action(
                state, action, public_id, state_dir=state_dir
            )
        auto_decision = _auto_decision_command(state, state_dir=state_dir)
        if auto_decision:
            action = {
                "cmd": _review_command(public_id, state_dir=state_dir),
                "note": f"Structured {auto_decision} verdict is ready; rerun this review id to record it and continue.",
                "override": {
                    DECISION_CLEAN: _review_command(
                        public_id, "--decision", DECISION_CLEAN, state_dir=state_dir
                    ),
                    DECISION_FINDINGS: _review_command(
                        public_id, "--decision", DECISION_FINDINGS, state_dir=state_dir
                    ),
                },
            }
            return _with_deslop_done_action(
                state, action, public_id, state_dir=state_dir
            )
        action = {
            "choices": {
                decision: _review_command(
                    public_id, "--decision", decision, state_dir=state_dir
                )
                for decision in (DECISION_CLEAN, DECISION_FINDINGS)
            },
            "note": "Classify the reviewer output, then record clean or findings.",
        }
        return _with_deslop_done_action(state, action, public_id, state_dir=state_dir)
    if stage == STAGE_FIX_PENDING:
        note = "Commit/amend valid fixes, then rerun this command."
        if str(state.get("review_brief") or "").strip():
            note += (
                " If a finding conflicts with the frozen contract, rerun this review "
                "id with --contract-conflict <dimension> instead."
            )
        action = {
            "cmd": _review_command(public_id, state_dir=state_dir),
            "note": note,
        }
        return _with_deslop_done_action(state, action, public_id, state_dir=state_dir)
    if stage == STAGE_CREATED:
        action = {"cmd": _review_command(public_id, state_dir=state_dir)}
        note = _continuation_note(state)
        if note:
            action["note"] = note
        return _with_deslop_done_action(state, action, public_id, state_dir=state_dir)
    if stage in {STAGE_REVIEW_GREEN, STAGE_LOCAL_GREEN_HANDOFF}:
        deslop = dict(state.get("deslop") or {})
        if bool(deslop.get("tracked")) and str(deslop.get("status") or "") in {
            "tracked",
            "failed",
        }:
            return {
                "cmd": _review_command(public_id, state_dir=state_dir),
                "note": "Run the bounded exact-head closure before handoff.",
            }
        summary = review_ladder_summary(state, current_head=_identity_head(state))
        if summary.get("review_ladder") == "invalidated":
            if green_review_head_change_summary(state, summary=summary):
                action = _github_terminal_action(state, public_id, state_dir=state_dir)
            elif _github_pending_head_change_identity(state):
                action = {
                    "cmd": _review_command(public_id, state_dir=state_dir),
                    "note": "PR head changed before a terminal GitHub result; rerun local signoff on this review id before GitHub review.",
                }
            else:
                action = None
        elif _github_review_is_terminal(state):
            action = _github_terminal_action(state, public_id, state_dir=state_dir)
        elif _mode_label(state) == "fast" and _github_review_status(state) == "unknown":
            action = None
        else:
            action = _github_handoff_action(state, state_dir=state_dir)
        return _with_deslop_done_action(state, action, public_id, state_dir=state_dir)
    return _with_deslop_done_action(
        state,
        {"cmd": _review_command(public_id, state_dir=state_dir)},
        public_id,
        state_dir=state_dir,
    )


def _render(state: dict[str, Any], *, state_dir: Path) -> None:
    payload: dict[str, Any] = {
        "review": state.get("public_id"),
        "review_brief": "available" if state.get("review_brief") else "unavailable",
    }
    action = _action_payload(state, state_dir=state_dir)
    summary = _add_review_ladder_fields(payload, state, action)
    if action:
        if summary.get("review_ladder") != "invalidated":
            payload["Action"] = action
    github_review = dict(state.get("github_review") or {})
    github_status = str(github_review.get("status") or "").strip()
    if github_status and github_status != "unknown":
        payload["github_review"] = github_status
    if conformance := str(
        dict(state.get("deslop") or {}).get("conformance") or ""
    ).strip():
        payload["conformance"] = conformance
    if reason := dict(state.get("deslop") or {}).get("dismissal_reason"):
        payload["closure_dismissal_reason"] = reason
    if convergence := _public_convergence(state):
        payload["convergence"] = convergence
    if validation := _validation_summary(state):
        payload["validation"] = validation
    emit_toon(payload)


def _render_stale_decision_recovery(
    state: dict[str, Any], *, state_dir: Path, decision: str
) -> None:
    action = _action_payload(state, state_dir=state_dir)
    note = "The review already advanced past that decision."
    if action:
        note += " Continue with Action.cmd; do not add --decision unless Action.override asks for it."
    else:
        note += " No further action is pending."
    payload: dict[str, Any] = {
        "review": state.get("public_id"),
        "status": "decision_not_pending",
        "decision": decision,
        "note": note,
    }
    if action:
        payload["Action"] = action
    emit_toon(payload)


def _require_local_green_for_github_review(state: dict[str, Any]) -> None:
    if state.get("stage") not in {STAGE_REVIEW_GREEN, STAGE_LOCAL_GREEN_HANDOFF}:
        raise ValueError("--github-review requires local green review state")
    if bool(dict(state.get("deslop") or {}).get("tracked")):
        raise ValueError("--github-review requires completed exact-head closure")


def _github_review_subprocess_command(
    state: dict[str, Any], *, state_dir: Path, force: bool
) -> list[str]:
    identity = dict(state.get("identity") or {})
    cwd = str(identity.get("cwd") or "").strip()
    if not cwd:
        raise ValueError("review cycle is missing cwd for --github-review")
    command = [
        sys.executable,
        str(_launcher_script_path("review_github.py")),
        "run",
        "--cd",
        str(cwd_path_from_normalized(cwd)),
        "--state-dir",
        str(state_dir),
    ]
    if force:
        command.append("--force")
    return command


def _run_github_review(state: dict[str, Any], *, state_dir: Path, force: bool) -> int:
    _require_local_green_for_github_review(state)
    command = _github_review_subprocess_command(state, state_dir=state_dir, force=force)
    return subprocess.run(command, check=False).returncode


def main() -> int:
    parser = build_parser()
    try:
        args = parser.parse_args()
        arena_configuration = _configure_repo_arena(args)
        if arena_configuration is not None:
            return arena_configuration
        state_dir = Path(default_state_dir()).resolve(strict=False)
        has_validation_status = _has_validation_status(args)
        has_restart_brief = args.restart_brief is not None
        if args.json and not args.status:
            raise ValueError("--json requires --status")
        if args.json and args.verbose:
            raise ValueError("--json cannot be combined with --verbose")
        if args.verbose and not args.status:
            raise ValueError("--verbose requires --status")
        if args.validation_note and not has_validation_status:
            raise ValueError(
                "--validation-note requires --full-suite waived or --ci waived"
            )
        if args.status:
            if args.id:
                raise ValueError(
                    "--status cannot be combined with --id; use --id <id> --show-status"
                )
            if (
                args.mode
                or args.restart_mode
                or has_restart_brief
                or args.contract_conflict
                or args.convergence_decision
                or args.reason
                or args.decision
                or args.github_review
                or args.github_force
                or args.github_result
                or args.github_note
                or args.focused_validation
                or args.full_suite
                or args.ci
                or args.validation_note
                or args.deslop_done
                or args.skip_deslop
                or args.review_brief
                or args.model
                or args.reasoning
                or args.no_arena
                or args.show_findings
                or args.show_status
            ):
                raise ValueError(
                    "--status cannot be combined with review creation, id actions, validation, or restart flags"
                )
            return cmd_branch_status(args)
        if args.restart_mode and not args.id:
            raise ValueError("--restart-mode requires --id")
        if has_restart_brief and not args.id:
            raise ValueError("--restart-brief requires --id")
        convergence_action = args.contract_conflict or args.convergence_decision
        if convergence_action and not args.id:
            raise ValueError("convergence actions require --id")
        convergence_conflict = args.contract_conflict and args.convergence_decision
        other_action = (
            any(
                getattr(args, name)
                for name in (
                    "restart_mode",
                    "reason",
                    "decision",
                    "github_review",
                    "github_force",
                    "github_result",
                    "github_note",
                    "deslop_done",
                    "show_findings",
                    "show_status",
                    "wsl",
                )
            )
            or has_restart_brief
            or has_validation_status
        )
        if convergence_action and (convergence_conflict or other_action):
            raise ValueError(
                "convergence actions cannot be combined with another id action"
            )
        if args.skip_deslop and args.id:
            raise ValueError(
                "--skip-deslop/--no-deslop can only be used when creating a review cycle"
            )
        if args.reason and not (
            args.restart_mode or has_restart_brief or args.deslop_done
        ):
            raise ValueError(
                "--reason requires --restart-mode, --restart-brief, or --deslop-done"
            )
        if args.deslop_done and not args.id:
            raise ValueError("--deslop-done requires --id")
        if args.deslop_done and (
            args.restart_mode
            or has_restart_brief
            or args.decision
            or args.github_review
            or args.github_force
            or args.github_result
            or args.github_note
            or has_validation_status
            or args.show_findings
            or args.show_status
        ):
            raise ValueError(
                "--deslop-done cannot be combined with restart, decisions, GitHub review, GitHub results, "
                "GitHub notes, validation status flags, show-findings, or show-status"
            )
        if args.restart_mode and (
            has_restart_brief
            or args.decision
            or args.github_review
            or args.github_force
            or args.github_result
            or has_validation_status
        ):
            raise ValueError(
                "--restart-mode cannot be combined with restart-brief, decisions, GitHub review, GitHub results, or validation status flags"
            )
        if has_restart_brief and (
            args.decision
            or args.github_review
            or args.github_force
            or args.github_result
            or has_validation_status
        ):
            raise ValueError(
                "--restart-brief cannot be combined with decisions, GitHub review, GitHub results, or validation status flags"
            )
        if args.decision and not args.id:
            raise ValueError("--decision requires --id")
        if args.github_force and not args.github_review:
            raise ValueError("--github-force requires --github-review")
        if args.github_review and not args.id:
            raise ValueError("--github-review requires --id")
        if args.github_review and (
            args.decision
            or args.github_result
            or args.github_note
            or has_validation_status
        ):
            raise ValueError(
                "--github-review cannot be combined with decisions, GitHub results, GitHub notes, or validation status flags"
            )
        if args.github_result and not args.id:
            raise ValueError("--github-result requires --id")
        if args.github_result and (args.decision or has_validation_status):
            raise ValueError(
                "--github-result cannot be combined with decisions or validation status flags"
            )
        if args.github_note and not args.github_result:
            raise ValueError("--github-note requires --github-result")
        if has_validation_status and not args.id:
            raise ValueError("validation status flags require --id")
        if args.show_findings and not args.id:
            raise ValueError("--show-findings requires --id")
        if args.show_status and not args.id:
            raise ValueError("--show-status requires --id")
        if args.show_findings and args.show_status:
            raise ValueError("--show-findings cannot be combined with --show-status")
        if args.show_findings and (
            args.restart_mode
            or has_restart_brief
            or args.decision
            or args.github_review
            or args.github_force
            or args.github_result
            or args.github_note
            or has_validation_status
        ):
            raise ValueError(
                "--show-findings cannot be combined with decisions, GitHub review, GitHub results, validation status flags, or restart"
            )
        if args.show_status and (
            args.restart_mode
            or has_restart_brief
            or args.decision
            or args.github_review
            or args.github_force
            or args.github_result
            or args.github_note
            or has_validation_status
        ):
            raise ValueError(
                "--show-status cannot be combined with decisions, GitHub review, GitHub results, validation status flags, or restart"
            )
        if args.id:
            state_dir, state = _load_cycle_and_state_dir(state_dir, str(args.id))
            state = _apply_runtime_options(state, args)
            _reject_id_creation_args(args, state)
            if args.show_findings:
                return _show_findings(state, state_dir=state_dir)
            if args.show_status:
                return _show_status(state, state_dir=state_dir)
            if args.contract_conflict:
                state = record_contract_conflict(
                    state, conflict=str(args.contract_conflict)
                )
                saved = save_cycle(state_dir, state)
                _render(saved, state_dir=state_dir)
                return 0
            if args.convergence_decision:
                state = record_convergence_decision(
                    state, decision=str(args.convergence_decision)
                )
                if str(args.convergence_decision).upper() == "CONTINUE":
                    state = _resume_progress(state, state_dir=state_dir)
                saved = save_cycle(state_dir, state)
                _render(saved, state_dir=state_dir)
                return 0
            if args.deslop_done:
                resumed = _resume_progress(state, state_dir=state_dir)
                if resumed != state:
                    saved = save_cycle(state_dir, resumed)
                    _render(saved, state_dir=state_dir)
                    return 0
                if _current_cycle_identity_if_compatible(state) is None:
                    raise ValueError(
                        "--deslop-done requires the exact clean branch, HEAD, and merge-base"
                    )
                state = mark_deslop_closed(state, reason=args.reason)
                saved = save_cycle(state_dir, state)
                _render(saved, state_dir=state_dir)
                return 0
            if args.restart_mode:
                state = _restart_cycle(
                    state,
                    state_dir=state_dir,
                    target_mode=str(args.restart_mode),
                    reason=_restart_reason(args),
                )
                _render(state, state_dir=state_dir)
                return 0
            if has_restart_brief:
                state = _restart_review_brief(
                    state,
                    state_dir=state_dir,
                    review_brief=str(args.restart_brief),
                    reason=_restart_reason(args, option="--restart-brief"),
                )
                _render(state, state_dir=state_dir)
                return 0
            if args.github_review:
                resumed = _resume_progress(state, state_dir=state_dir)
                if resumed != state:
                    saved = save_cycle(state_dir, resumed)
                    _render(saved, state_dir=state_dir)
                    return 0
                saved = save_cycle(state_dir, state)
                return _run_github_review(
                    saved, state_dir=state_dir, force=bool(args.github_force)
                )
            if args.github_result:
                resumed = _resume_progress(state, state_dir=state_dir)
                if resumed != state:
                    saved = save_cycle(state_dir, resumed)
                    _render(saved, state_dir=state_dir)
                    return 0
                state = _record_github_result(state, args)
                saved = save_cycle(state_dir, state)
                _render(saved, state_dir=state_dir)
                return 0
            if args.decision:
                if has_validation_status:
                    state = _record_validation_status(state, args)
                try:
                    state = _apply_decision(
                        state, str(args.decision), state_dir=state_dir
                    )
                except NoDecisionPendingError as exc:
                    recovery_state = exc.state
                    should_save = recovery_state != state
                    if has_validation_status:
                        recovery_state = _record_validation_status(recovery_state, args)
                        should_save = True
                    if should_save:
                        recovery_state = save_cycle(state_dir, recovery_state)
                    _render_stale_decision_recovery(
                        recovery_state, state_dir=state_dir, decision=str(args.decision)
                    )
                    return 0
            if has_validation_status and not args.decision:
                resumed = (
                    _resume_progress(state, state_dir=state_dir)
                    if _github_review_status(state) in {"clean", "waived"}
                    else state
                )
                if resumed != state:
                    saved = save_cycle(state_dir, resumed)
                    _render(saved, state_dir=state_dir)
                    return 0
                state = _record_validation_status(state, args)
            elif not args.decision:
                state = (
                    _resume_reserved_successor(state, state_dir=state_dir)
                    if _successor_needs_locked_resume(state)
                    else _advance_without_decision(state, state_dir=state_dir)
                )
        else:
            state = _create_or_resume_cycle(args=args, state_dir=state_dir)
            state = _advance_without_decision(state, state_dir=state_dir)
        saved = save_cycle(state_dir, state)
        _render(saved, state_dir=state_dir)
        return 0
    except SuccessorAdvanceBusyError as exc:
        emit_toon(
            {
                "review": str(exc.state.get("public_id") or ""),
                "status": "busy",
                "done": False,
                "review_ladder": "pending",
                "next_action": "wait",
                "note": "The successor review is already advancing; retry this id after the active command finishes.",
            }
        )
        return 0
    except ValueError as exc:
        return emit_error(str(exc), status="usage_error", help_items=[_help_command()])
    except OSError as exc:
        return emit_error(str(exc), status="error")


if __name__ == "__main__":
    raise SystemExit(main())
