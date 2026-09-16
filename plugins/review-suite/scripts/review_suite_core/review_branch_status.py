from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from review_suite_runtime_bootstrap import launcher_script_path
from review_suite_core import (
    effective_base_ref,
    emit_toon,
    format_command,
    inspect_workflow_status,
    resolve_repo_root,
)
from review_suite_core.config import default_state_dir
from review_suite_core.orchestrator_profiles import MODE_STRICTNESS_ORDER
from review_suite_core.orchestrator_state import (
    HEAD_CHANGED_AFTER_GREEN_REVIEW_LADDER,
    FIX_REVIEW_POLICY,
    convergence_summary,
    green_review_head_change_summary,
    review_ladder_summary,
    validation_blockers,
)
from review_suite_local import (
    grade_rank_placeholders,
    load_round,
    normalize_review_cwd_value,
    round_needs_caller_grade,
    unique_round_state_dirs,
)

ORCHESTRATOR_HIDDEN_STAGES = {"aborted", "dismissed"}
ORCHESTRATOR_HEAD_CURRENT_STAGES = {"review-green", "local-green-handoff"}
DECISION_COMMANDS = {"clean", "findings"}


def _script_path(name: str) -> str:
    return launcher_script_path(
        Path(__file__).resolve().parents[1] / "review.py", name
    ).as_posix()


def _wsl_unc_cd(cd: str | None) -> bool:
    normalized = str(cd or "").strip().replace("\\", "/").lower()
    return normalized.startswith("//wsl.localhost/") or normalized.startswith("//wsl$/")


def _status_path_for_wsl_check(cd: str | None) -> str | None:
    if str(cd or "").strip():
        return str(cd)
    try:
        return str(Path.cwd())
    except OSError:
        return None


def _reject_status_unc_wsl(path: str | None) -> None:
    if sys.platform != "win32" or not _wsl_unc_cd(path):
        return
    raise ValueError(
        "review.py --status does not launch Codex, so --wsl is not useful here. "
        "Run review.py --status from native WSL with the Linux repo path instead of a Windows UNC path; Windows git over //wsl.localhost can hang."
    )


def _review_mode_for_lane(lane: str) -> str:
    return "deep" if lane == "review_t3" else "normal"


def _start_review_command(*, review_cwd: Path, base: str, mode: str) -> str:
    return format_command(
        [
            sys.executable,
            _script_path("review.py"),
            "--mode",
            mode,
            "--cd",
            str(review_cwd),
            "--base",
            base,
        ]
    )


def _with_action_context(
    action: dict[str, object], *, review_cwd: Path, round_id: str | None = None
) -> dict[str, object]:
    action["cwd"] = str(review_cwd)
    if round_id:
        action["round_id"] = round_id
    return action


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


def _json_status_payload(payload: dict[str, object]) -> dict[str, object]:
    public = _public_status_payload(payload)
    public.pop("Action", None)
    for key in ("base", "branch", "head", "mode"):
        value = payload.get(key)
        if value not in (None, "", [], {}):
            public[key] = value
    return public


def _review_command(public_id: str, *, extra: tuple[str, ...] = ()) -> str:
    return format_command(
        [
            sys.executable,
            _script_path("review.py"),
            "--id",
            public_id,
            *extra,
        ]
    )


def _orchestrator_validation_status_command(public_id: str, blockers: list[str]) -> str:
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
    return _review_command(public_id, extra=tuple(args))


def _orchestrator_validation_blocker_action(
    public_id: str, blockers: list[str]
) -> dict[str, object]:
    return {
        "cmd": _orchestrator_validation_status_command(public_id, blockers),
        "blocked_by": blockers,
        "note": 'GitHub result is recorded; replace status placeholders with passed, or waived and append --validation-note "reason", before PR-final or merge-ready.',
    }


def _restart_deep_action(
    state: dict[str, object], public_id: str
) -> dict[str, object] | None:
    stage = str(state.get("stage") or "").strip()
    if stage in ORCHESTRATOR_HIDDEN_STAGES or isinstance(
        state.get("superseded_by"), dict
    ):
        return None
    mode = str(
        dict(state.get("mode") or {}).get("effective")
        or dict(state.get("mode") or {}).get("requested")
        or ""
    ).strip()
    if (
        mode not in MODE_STRICTNESS_ORDER
        or MODE_STRICTNESS_ORDER[mode] >= MODE_STRICTNESS_ORDER["deep"]
    ):
        return None
    return {
        "cmd": _review_command(
            public_id, extra=("--restart-mode", "deep", "--reason", "REASON")
        ),
        "mode": "deep",
        "note": "Use only for explicit escalation; replace REASON.",
    }


def _orchestrator_cycles(state_dir: Path) -> list[dict[str, object]]:
    cycles: list[dict[str, object]] = []
    for path in sorted((state_dir / "orchestrator" / "cycles").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except OSError, json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            try:
                payload["_state_file_mtime"] = path.stat().st_mtime
            except OSError:
                payload["_state_file_mtime"] = 0.0
            cycles.append(payload)
    return cycles


def _review_step_position(
    state: dict[str, object], step_index: int
) -> tuple[int | None, int | None]:
    steps = [
        item
        for item in list(dict(state.get("review_plan") or {}).get("steps") or [])
        if isinstance(item, dict)
    ]
    review_indices = [
        index
        for index, item in enumerate(steps)
        if str(item.get("kind") or "review").strip() == "review"
    ]
    if step_index not in review_indices:
        return None, None
    return review_indices.index(step_index) + 1, len(review_indices)


def _orchestrator_progress_label(state: dict[str, object]) -> str | None:
    candidates: list[dict[str, object]] = []
    pending = dict(state.get("pending_action") or {})
    if pending:
        candidates.append(pending)
    current_step = dict(
        dict(state.get("review_progress") or {}).get("current_step") or {}
    )
    if current_step:
        candidates.append(current_step)
    for item in candidates:
        try:
            step_index = int(
                item.get("step_index")
                if item.get("step_index") is not None
                else item.get("index")
            )
        except TypeError, ValueError:
            continue
        step_name = str(item.get("step") or item.get("name") or "").strip()
        lane = str(item.get("lane") or "review_t1").strip()
        if lane != "review_t1" or not step_name:
            continue
        position, total = _review_step_position(state, step_index)
        if position and total:
            return f"review {position}/{total} {step_name}"
    return None


def _round_by_id(state: dict[str, object], round_id: str) -> dict[str, object]:
    for item in list(state.get("rounds") or []):
        if isinstance(item, dict) and str(item.get("round_id") or "") == round_id:
            return dict(item)
    return {}


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


def _pending_grade_payload(
    state: dict[str, object], *, state_dir: Path
) -> dict[str, object] | None:
    pending = dict(state.get("pending_action") or {})
    round_id = str(pending.get("round_id") or "").strip()
    if str(pending.get("kind") or "") != "decision" or not round_id:
        return None
    round_record = _round_by_id(state, round_id)
    if not bool(round_record.get("grading_required")):
        return None
    payload = _load_output_round_payload(state_dir, round_record)
    if not bool(round_record.get("arena_round") or payload.get("arena_round")):
        return None
    if not round_needs_caller_grade(payload):
        return None
    return payload


def _arena_grade_command(
    state: dict[str, object],
    *,
    state_dir: Path,
) -> str | None:
    pending_payload = _pending_grade_payload(state, state_dir=state_dir)
    if pending_payload is None:
        return None
    round_id = str(pending_payload.get("round_id") or "").strip()
    task_id = str(pending_payload.get("task_id_hint") or "").strip()
    if not task_id:
        task_id = str(
            dict(state.get("identity") or {}).get("branch")
            or state.get("public_id")
            or ""
        ).strip()
    grade_state_dir = Path(str(pending_payload.get("_round_state_dir") or state_dir))
    command = [
        sys.executable,
        _script_path("review_suite_arena.py"),
        "grade",
        "--round-id",
        round_id,
        "--task-id",
        task_id,
        "--rating-pool-id",
        str(pending_payload.get("rating_pool_id") or "").strip() or "RATING_POOL_ID",
    ]
    for rank_group in grade_rank_placeholders(pending_payload):
        command.extend(["--rank", rank_group])
    command.extend(["--basis", "BASIS", "--state-dir", str(grade_state_dir)])
    return format_command(command)


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


def _auto_decision_command(state: dict[str, object], *, state_dir: Path) -> str | None:
    pending = dict(state.get("pending_action") or {})
    if str(pending.get("kind") or "") != "decision":
        return None
    round_id = str(pending.get("round_id") or "").strip()
    if not round_id:
        return None
    if _pending_grade_payload(state, state_dir=state_dir) is not None:
        return None
    round_record = _round_by_id(state, round_id)
    if not round_record:
        return None
    round_payload = _load_output_round_payload(state_dir, round_record)
    if _round_blocked(round_record) or _round_blocked(round_payload):
        return None
    return _round_terminal_command(round_payload)


def _orchestrator_mode_label(state: dict[str, object]) -> str:
    mode = dict(state.get("mode") or {})
    return str(mode.get("effective") or mode.get("requested") or "").strip()


def _orchestrator_github_review_status(state: dict[str, object]) -> str:
    return (
        str(dict(state.get("github_review") or {}).get("status") or "unknown").strip()
        or "unknown"
    )


def _orchestrator_terminal_review_head(state: dict[str, object]) -> str:
    review_heads = dict(state.get("review_heads") or {})
    for key in (
        "last_reviewed_head",
        "last_followup_head",
        "last_fix_head",
        "head",
    ):
        value = str(review_heads.get(key) or "").strip()
        if value:
            return value
    return str(dict(state.get("identity") or {}).get("head") or "").strip()


def _orchestrator_github_review_is_terminal(
    state: dict[str, object], *, current_head: str | None = None
) -> bool:
    github_review = dict(state.get("github_review") or {})
    if str(github_review.get("status") or "").strip() not in {"clean", "waived"}:
        return False
    reviewed_head = str(github_review.get("reviewed_head") or "").strip()
    drift = dict(state.get("base_drift") or {})
    if (
        bool(drift.get("patch_equivalent"))
        and reviewed_head == str(drift.get("reviewed_head") or "").strip()
    ):
        reviewed_head = (
            str(drift.get("equivalent_reviewed_head") or "").strip() or reviewed_head
        )
    if not reviewed_head:
        return False
    comparison_head = str(
        current_head or ""
    ).strip() or _orchestrator_terminal_review_head(state)
    return bool(comparison_head) and reviewed_head == comparison_head


def _orchestrator_action(
    state: dict[str, object],
    public_id: str,
    *,
    state_dir: Path,
    current_head: str | None = None,
) -> dict[str, object] | None:
    stage = str(state.get("stage") or "").strip()
    if convergence_summary(state).get("status") != "ACTIVE":
        return None
    superseded_by = state.get("superseded_by")
    if stage == "aborted" and isinstance(superseded_by, dict):
        replacement = str(superseded_by.get("review") or "").strip()
        if replacement:
            return {
                "cmd": _review_command(replacement),
                "note": f"Review {public_id} was superseded by {replacement}.",
            }
    if stage == "decision-pending":
        grade = _arena_grade_command(state, state_dir=state_dir)
        if grade:
            action = {
                "cmd": grade,
                "note": "Grade the arena round, then rerun this review id to continue.",
                "next": _review_command(public_id),
            }
        else:
            auto_decision = _auto_decision_command(state, state_dir=state_dir)
            if auto_decision:
                action: dict[str, object] = {
                    "cmd": _review_command(public_id),
                    "note": f"Structured {auto_decision} verdict is ready; rerun this review id to record it and continue.",
                    "override": {
                        "clean": _review_command(
                            public_id, extra=("--decision", "clean")
                        ),
                        "findings": _review_command(
                            public_id, extra=("--decision", "findings")
                        ),
                    },
                }
            else:
                action = {
                    "choices": {
                        decision: _review_command(
                            public_id, extra=("--decision", decision)
                        )
                        for decision in ("clean", "findings")
                    },
                    "note": "Classify the reviewer output, then record clean or findings.",
                }
    elif stage == "fix-pending":
        note = FIX_REVIEW_POLICY
        if str(state.get("review_brief") or "").strip():
            note += (
                " If a finding conflicts with the frozen contract, rerun this review "
                "id with --contract-conflict <dimension> instead."
            )
        action = {
            "choices": {
                "fixes_validated": _review_command(
                    public_id, extra=("--fixes-validated", "ELIGIBILITY_AND_VALIDATION")
                ),
                "repeat_review": _review_command(public_id),
            },
            "note": note,
        }
    elif stage in {"review-green", "local-green-handoff"}:
        summary = review_ladder_summary(state, current_head=current_head)
        if summary.get("review_ladder") == "invalidated":
            if (
                green_review_head_change_summary(
                    state, current_head=current_head, summary=summary
                )
                is None
            ):
                return None
            blockers = validation_blockers(state)
            return (
                _orchestrator_validation_blocker_action(public_id, blockers)
                if blockers
                else None
            )
        if _orchestrator_github_review_is_terminal(state, current_head=current_head):
            blockers = validation_blockers(state)
            if not blockers:
                return None
            action = _orchestrator_validation_blocker_action(public_id, blockers)
        elif (
            _orchestrator_mode_label(state) == "fast"
            and _orchestrator_github_review_status(state) == "unknown"
        ):
            return None
        else:
            action = {
                "cmd": _review_command(public_id, extra=("--github-review",)),
                "after": "PR create/update",
            }
            blockers = validation_blockers(state)
            if blockers:
                action["blocked_by"] = blockers
    else:
        action = {"cmd": _review_command(public_id)}
    restart = _restart_deep_action(state, public_id)
    if restart:
        action["restart"] = restart
    return action


def _orchestrator_review_head(state: dict[str, object]) -> str:
    review_heads = dict(state.get("review_heads") or {})
    for key in (
        "last_followup_head",
        "last_reviewed_head",
        "last_fix_head",
        "head",
    ):
        value = str(review_heads.get(key) or "").strip()
        if value:
            return value
    return str(dict(state.get("identity") or {}).get("head") or "").strip()


def _orchestrator_cycle_is_current(
    state: dict[str, object], current_payload: dict[str, object]
) -> bool:
    if str(state.get("stage") or "") not in ORCHESTRATOR_HEAD_CURRENT_STAGES:
        return True
    current_head = str(current_payload.get("head") or "").strip()
    if not current_head:
        return True
    summary = review_ladder_summary(state, current_head=current_head)
    return (
        summary.get("review_ladder") != "invalidated"
        or green_review_head_change_summary(
            state,
            current_head=current_head,
            summary=summary,
        )
        is not None
    )


def _orchestrator_status_override(
    *,
    state_dir: Path,
    review_cwd: Path,
    base: str,
    current_payload: dict[str, object],
) -> dict[str, object] | None:
    normalized_cwd = str(normalize_review_cwd_value(review_cwd) or "")
    branch = str(current_payload.get("branch") or "").strip()
    candidates: list[dict[str, object]] = []
    for state in _orchestrator_cycles(state_dir):
        identity = dict(state.get("identity") or {})
        if str(identity.get("cwd") or "") != normalized_cwd:
            continue
        if str(identity.get("base") or "") != str(base or "").strip():
            continue
        cycle_branch = str(identity.get("branch") or "").strip()
        if branch and cycle_branch and cycle_branch != branch:
            continue
        if str(state.get("stage") or "") in ORCHESTRATOR_HIDDEN_STAGES:
            continue
        if not _orchestrator_cycle_is_current(state, current_payload):
            continue
        if not str(state.get("public_id") or "").strip():
            continue
        candidates.append(state)
    if not candidates:
        return None
    state = sorted(
        candidates,
        key=lambda item: (
            float(item.get("_state_file_mtime") or 0.0),
            str(
                item.get("updated_at")
                or item.get("created_at")
                or item.get("public_id")
                or ""
            ),
        ),
    )[-1]
    public_id = str(state.get("public_id") or "").strip()
    payload: dict[str, object] = {
        "review": public_id,
        "base": current_payload.get("base"),
        "branch": current_payload.get("branch"),
        "head": current_payload.get("head"),
        "mode": _orchestrator_mode_label(state),
    }
    convergence = convergence_summary(state)
    if convergence.get("status") != "ACTIVE" or convergence.get(
        "accepted_findings_heads"
    ):
        payload["convergence"] = convergence
    current_head = str(current_payload.get("head") or "").strip()
    action = _orchestrator_action(
        state, public_id, state_dir=state_dir, current_head=current_head
    )
    summary = review_ladder_summary(state, current_head=current_head)
    summary = (
        green_review_head_change_summary(
            state, current_head=current_head, summary=summary
        )
        or summary
    )
    payload.update(summary)
    if convergence.get("status") != "ACTIVE":
        if convergence.get("status") == "DECISION_REQUIRED":
            payload["status"] = "decision_required"
            payload["next_action"] = "caller_decision"
        else:
            payload["status"] = str(convergence.get("decision") or "decided").lower()
            payload["next_action"] = "none"
    elif summary.get("review_ladder") == "invalidated":
        payload["status"] = "stale"
        payload["next_action"] = "rerun_review"
    elif summary.get("review_ladder") == HEAD_CHANGED_AFTER_GREEN_REVIEW_LADDER:
        payload["status"] = HEAD_CHANGED_AFTER_GREEN_REVIEW_LADDER
        payload["next_action"] = (
            "validation" if action is not None else "inspect_changed_since_review"
        )
    elif bool(summary.get("done")):
        payload["status"] = "done"
        payload["next_action"] = "none"
    elif action is not None:
        command = str(action.get("cmd") or "")
        if "--github-review" in command:
            payload["next_action"] = "github_review"
        elif "--full-suite" in command or "--ci" in command:
            payload["next_action"] = "validation"
        else:
            payload["next_action"] = "continue"
    else:
        payload["next_action"] = "none"
    if action is not None:
        if summary.get("review_ladder") != "invalidated":
            payload["Action"] = action
    progress = _orchestrator_progress_label(state)
    if progress:
        payload["progress"] = progress
    return payload


def _status_action(
    payload: dict[str, object], *, review_cwd: Path, base: str, state_dir: Path
) -> dict[str, object] | None:
    recommendation = str(payload.get("recommendation") or "")
    if recommendation == "review-followup":
        since_head = str(payload.get("last_reviewed_head") or "").strip()
        if not since_head:
            return None
        if bool(payload.get("worktree_dirty")):
            return _with_action_context(
                {
                    "lane": "commit-or-stash",
                    "note": (
                        "Commit intended follow-up changes or stash unrelated dirty files, "
                        "then rerun review.py --status to get the review-followup command."
                    ),
                },
                review_cwd=review_cwd,
            )
        command = [
            sys.executable,
            _script_path("review_followup.py"),
            "--base",
            base,
            "--since",
            since_head,
        ]
        command.extend(
            [
                "--note-file",
                ".review-suite/fix-note.md",
            ]
        )
        action = {
            "lane": "review-followup",
            "cmd": format_command(command),
        }
        return _with_action_context(action, review_cwd=review_cwd)
    recommended_lane = str(payload.get("recommended_lane") or "").strip()
    if recommendation in {"coherence-review", "full-review"} and recommended_lane in {
        "review_t1",
        "review_t3",
    }:
        return _with_action_context(
            {
                "lane": recommended_lane,
                "cmd": _start_review_command(
                    review_cwd=review_cwd,
                    base=base,
                    mode=_review_mode_for_lane(recommended_lane),
                ),
            },
            review_cwd=review_cwd,
        )
    if recommendation == "coherence-review":
        return _with_action_context(
            {
                "lane": "coherence-review",
                "cmd": _start_review_command(
                    review_cwd=review_cwd, base=base, mode="normal"
                ),
            },
            review_cwd=review_cwd,
        )
    if recommendation == "full-review":
        return _with_action_context(
            {
                "lane": "full-review",
                "cmd": _start_review_command(
                    review_cwd=review_cwd, base=base, mode="normal"
                ),
            },
            review_cwd=review_cwd,
        )
    return None


def cmd_status(args: argparse.Namespace) -> int:
    _reject_status_unc_wsl(_status_path_for_wsl_check(getattr(args, "cd", None)))
    review_cwd = resolve_repo_root(args.cd)
    state_dir = Path(default_state_dir()).resolve(strict=False)
    base = str(effective_base_ref(review_cwd, args.base)["base"])
    payload = inspect_workflow_status(
        state_dir=state_dir,
        review_cwd=review_cwd,
        base=base,
    )
    orchestrator_override = _orchestrator_status_override(
        state_dir=state_dir,
        review_cwd=review_cwd,
        base=base,
        current_payload=payload,
    )
    if orchestrator_override is not None:
        payload = orchestrator_override
    action = _status_action(
        payload, review_cwd=review_cwd, base=base, state_dir=state_dir
    )
    if action is not None:
        payload["Action"] = action
    if bool(args.json):
        print(json.dumps(_json_status_payload(payload), sort_keys=True))
    else:
        emit_toon(payload if bool(args.verbose) else _public_status_payload(payload))
    return 0
