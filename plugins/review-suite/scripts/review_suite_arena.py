#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

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
    DEFAULT_PROGRESS_INTERVAL_SECONDS,
    effective_execution_cwd,
    emit_error,
    emit_toon,
    format_command,
    record_review_anchor,
    resolve_repo_root,
    utc_now_iso,
    validate_codex_runtime,
    write_text,
)
from review_suite_core.config import default_state_dir
from review_suite_core.orchestrator_store import orchestrator_store_lock
from review_suite_local import (
    OPERATIONAL_STATE_FILENAME,
    RUN_LOG_FILENAME,
    TASK_CLASSES,
    PUBLIC_REVIEWER_LABELS,
    aggregate_records,
    append_record_if_new,
    build_phase_instructions,
    build_pr_instructions,
    build_local_review_request,
    build_record_from_grade,
    build_reroll_slot_payload,
    compact_benchmark_record,
    compact_round_files,
    collect_round_results,
    default_roster_path,
    default_rubric_path,
    ensure_clean_git_worktree,
    enrich_record_repo_names,
    find_blocking_rounds_for_caller,
    find_pending_rounds_for_caller,
    grade_rank_placeholders,
    guard_no_stage_step_down,
    includes_deep_review_effort,
    load_operational_state,
    load_roster,
    normalize_record_review_cwd_value,
    load_round,
    load_rubric,
    normalize_review_cwd_value,
    iter_round_payloads,
    print_deep_review_wait_note,
    promote,
    public_round_payload,
    public_round_result,
    public_reviewer_label,
    read_jsonl,
    round_needs_caller_grade,
    resolve_caller_id,
    run_round,
    select_pair,
    make_round_id,
    state_lock,
    ungraded_round_exposure_records,
    payload_has_blocked_runs,
    public_task_name,
    round_has_live_reviewer_process,
    output_isatty,
    print_reviewer_output_section,
    reviewer_output_heading,
    usable_output_slots,
    write_json,
    write_jsonl,
    write_reports,
    write_round,
    final_display_body,
)
from review_costs import (
    DEFAULT_COST_REPORT_FILENAME,
    REVIEW_COSTS_LOCK_NAME,
    REVIEW_COSTS_LOCK_TIMEOUT_SECONDS,
    apply_review_cost_delta_best_effort,
    collect_review_cost_rows,
    read_review_cost_row_for_cwd,
    read_review_cost_row_cache,
    update_review_cost_row_cache,
    write_review_cost_report,
)
from review_state_prune import prune_review_state


PUBLIC_ARENA_TASK_CLASS_ALIASES = {
    "review_t1": "phase_review",
    "review_t3": "pr_review",
}


class BlockingRoundError(ValueError):
    def __init__(self, message: str, *, action_payload: dict[str, object]) -> None:
        super().__init__(message)
        self.action_payload = action_payload


def _blocking_round_error(
    *,
    payload: dict[str, object],
    action: str,
    state_dir: Path | None = None,
) -> BlockingRoundError:
    round_id = str(payload.get("round_id") or "")
    status = str(payload.get("status") or "unknown")
    if round_needs_caller_grade(payload):
        message = f"pending round blocks {action}: {round_id}"
        action_payload: dict[str, object] = {
            "cmd": _grade_command(
                round_id=round_id,
                rating_pool_id=str(payload.get("rating_pool_id") or "").strip()
                or "RATING_POOL_ID",
                rank_groups=grade_rank_placeholders(payload),
                state_dir=state_dir,
            ),
            "dismiss_cmd": _dismiss_round_command(round_id=round_id),
        }
    else:
        message = f"pending round blocks {action}: {round_id} ({status})"
        action_payload = {
            "dismiss_cmd": _dismiss_round_command(round_id=round_id),
        }
    return BlockingRoundError(message, action_payload=action_payload)


def _raise_if_blocking_round_exists(
    *,
    action: str,
    ignore_pending_grades: bool,
    state_dir: Path,
    caller_id: str | None,
    review_cwd: Path,
    roster_path: Path,
    rubric_path: Path,
) -> None:
    if ignore_pending_grades:
        return
    blocking = find_blocking_rounds_for_caller(
        state_dir=state_dir, caller_id=caller_id, review_cwd=review_cwd
    )
    if not blocking:
        return
    latest = blocking[-1]
    raise _blocking_round_error(
        payload=latest,
        action=action,
        state_dir=state_dir,
    )


def _resolve_review_cwd(cd: str | None) -> Path:
    return resolve_repo_root(cd)


def _summary_equivalence_view(summary: dict[str, object]) -> dict[str, object]:
    comparable = copy.deepcopy(summary)
    comparable.pop("generated_at", None)
    return comparable


def build_parser() -> argparse.ArgumentParser:
    parser = AxiArgumentParser(
        description="Review Suite paired local review orchestration."
    )
    sub = parser.add_subparsers(
        dest="command", required=True, parser_class=AxiArgumentParser
    )
    task_class_choices = tuple(TASK_CLASSES) + tuple(PUBLIC_ARENA_TASK_CLASS_ALIASES)

    run = sub.add_parser("run")
    run.add_argument("--task-class", required=True, choices=task_class_choices)
    run.add_argument("--cd")
    run.add_argument("--task-id")
    run.add_argument("--base", required=True)
    run.add_argument("--seed", type=int)
    run.add_argument("--roster", default=str(default_roster_path()))
    run.add_argument("--rubric", default=str(default_rubric_path()))
    run.add_argument("--state-dir", default=str(default_state_dir()))
    run.add_argument(
        "--sqlite-path", default=str(Path.home() / ".codex" / "state_5.sqlite")
    )
    run.add_argument("--wsl", action="store_true")
    run.add_argument("--caller-id")
    run.add_argument("--ignore-pending-grades", action="store_true")
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the round selection only. Does not launch reviewers, write state, or run preflight guards (pending grades, worktree, runtime).",
    )
    run.add_argument("--rating-pool-id", help="rating pool/epoch")
    run.add_argument(
        "--rank",
        action="append",
        dest="rank_groups",
        help="repeat best to worst; comma-separate ties",
    )
    run.add_argument("--basis", help="grade basis")
    run.add_argument("--note")

    sample = sub.add_parser("sample")
    sample.add_argument("--task-class", required=True, choices=task_class_choices)
    sample.add_argument("--seed", type=int)
    sample.add_argument("--roster", default=str(default_roster_path()))
    sample.add_argument("--state-dir", default=str(default_state_dir()))
    sample.add_argument("--caller-id")
    sample.add_argument("--ignore-pending-grades", action="store_true")
    sample.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the round selection only. Does not write state or run preflight guards.",
    )
    sample.add_argument("--exclude-variant-id", action="append", default=[])

    run = sub.add_parser("run-round")
    run.add_argument("--round-id", required=True)
    run.add_argument("--cd")
    run.add_argument("--roster", default=str(default_roster_path()))
    run.add_argument("--state-dir", default=str(default_state_dir()))
    run.add_argument(
        "--sqlite-path", default=str(Path.home() / ".codex" / "state_5.sqlite")
    )
    run.add_argument("--base", required=True)
    run.add_argument("--wsl", action="store_true")

    resume = sub.add_parser("resume-round")
    resume.add_argument("--round-id", required=True)
    resume.add_argument("--cd")
    resume.add_argument("--roster", default=str(default_roster_path()))
    resume.add_argument("--state-dir", default=str(default_state_dir()))
    resume.add_argument(
        "--sqlite-path", default=str(Path.home() / ".codex" / "state_5.sqlite")
    )

    show = sub.add_parser("show-round")
    show.add_argument("--round-id", required=True)
    show.add_argument("--state-dir", default=str(default_state_dir()))
    show.add_argument("--json", action="store_true")

    show_last = sub.add_parser("show-last", aliases=["show_last"])
    show_last.add_argument("--cd")
    show_last.add_argument("--task", choices=["review_t1", "review_t3"])
    show_last.add_argument("--state-dir", default=str(default_state_dir()))
    show_last.add_argument("--json", action="store_true")

    costs = sub.add_parser("costs")
    costs.add_argument("--cd")
    costs.add_argument(
        "--all",
        action="store_true",
        help="rebuild all cached rows from canonical history, including implementation estimates",
    )
    costs.add_argument("--state-dir", default=str(default_state_dir()))
    costs.add_argument("--codex-home")
    costs.add_argument("--output")
    costs.add_argument("--json", action="store_true")

    grade = sub.add_parser("grade")
    grade.add_argument("--round-id")
    grade.add_argument("--task-id")
    grade.add_argument("--rating-pool-id", required=True, help="rating pool/epoch")
    grade.add_argument(
        "--rank",
        action="append",
        dest="rank_groups",
        required=True,
        help="repeat best to worst; comma-separate ties",
    )
    grade.add_argument("--basis", required=True, help="grade basis")
    grade.add_argument("--note")
    grade.add_argument("--roster", default=str(default_roster_path()))
    grade.add_argument("--rubric", default=str(default_rubric_path()))
    grade.add_argument("--state-dir", default=str(default_state_dir()))
    grade.add_argument("--caller-id")
    dismiss = sub.add_parser("dismiss-round")
    dismiss.add_argument("--round-id", required=True)
    dismiss.add_argument("--state-dir", default=str(default_state_dir()))
    dismiss.add_argument("--reason", default="manual_dismiss")
    dismiss.add_argument("--caller-id")

    report = sub.add_parser("report")
    report.add_argument("--roster", default=str(default_roster_path()))
    report.add_argument("--state-dir", default=str(default_state_dir()))
    report.add_argument("--round-id", help=argparse.SUPPRESS)

    refresh = sub.add_parser("refresh")
    refresh.add_argument("--roster", default=str(default_roster_path()))
    refresh.add_argument("--state-dir", default=str(default_state_dir()))

    promote_parser = sub.add_parser("promote")
    promote_parser.add_argument("--roster", default=str(default_roster_path()))
    promote_parser.add_argument("--state-dir", default=str(default_state_dir()))
    promote_parser.add_argument("--apply", action="store_true")

    compact = sub.add_parser("compact-runs")
    compact.add_argument("--roster", default=str(default_roster_path()))
    compact.add_argument("--state-dir", default=str(default_state_dir()))
    compact.add_argument("--backup-suffix", default=".bak")
    compact.add_argument("--apply", action="store_true")

    compact_rounds = sub.add_parser("compact-rounds")
    compact_rounds.add_argument("--state-dir", default=str(default_state_dir()))
    compact_rounds.add_argument("--apply", action="store_true")

    prune_state = sub.add_parser("prune-state")
    prune_state.add_argument("--state-dir", default=str(default_state_dir()))
    prune_state.add_argument("--older-than-days", type=int, default=14)
    prune_state.add_argument("--apply", action="store_true")

    reroll = sub.add_parser("reroll-slot")
    reroll.add_argument("--round-id", required=True)
    reroll.add_argument("--slot", required=True, choices=PUBLIC_REVIEWER_LABELS)
    reroll.add_argument("--cd")
    reroll.add_argument("--base")
    reroll.add_argument("--seed", type=int)
    reroll.add_argument("--roster", default=str(default_roster_path()))
    reroll.add_argument("--rubric", default=str(default_rubric_path()))
    reroll.add_argument("--state-dir", default=str(default_state_dir()))
    reroll.add_argument(
        "--sqlite-path", default=str(Path.home() / ".codex" / "state_5.sqlite")
    )
    reroll.add_argument("--wsl", action="store_true")

    return parser


def _select_round_payload(
    *,
    roster: dict[str, object],
    task_class: str,
    review_cwd: Path,
    state_dir: Path,
    seed: int | None,
    caller_id: str | None,
    caller_id_source: str | None,
    excluded_variant_ids: set[str] | None = None,
    cleanup_stale_rounds: bool = True,
) -> dict[str, object]:
    records = read_jsonl(
        state_dir / RUN_LOG_FILENAME
    ) + ungraded_round_exposure_records(state_dir, cleanup=cleanup_stale_rounds)
    operational_state = load_operational_state(state_dir / OPERATIONAL_STATE_FILENAME)
    return select_pair(
        roster=roster,
        operational_state=operational_state,
        records=records,
        task_class=task_class,
        review_cwd=review_cwd,
        seed=seed,
        caller_id=caller_id,
        caller_id_source=caller_id_source,
        excluded_variant_ids=excluded_variant_ids or set(),
    )


def _dry_run_round_payload(
    payload: dict[str, object], *, task_name: str
) -> dict[str, object]:
    return {
        "status": "dry_run",
        "task": task_name,
        "round_id": str(payload.get("round_id") or ""),
        "selection_mode": payload.get("selection_mode"),
        "selection_pairing": payload.get("selection_pairing"),
        "reviewers": [
            {
                "slot": str(run.get("slot") or ""),
                "variant_id": run.get("variant_id"),
                "model": run.get("model"),
                "reasoning_effort": run.get("reasoning_effort"),
            }
            for run in list(payload.get("runs") or [])
            if isinstance(run, dict)
        ],
    }


def cmd_sample(args: argparse.Namespace) -> int:
    task_class = _normalize_arena_task_class(str(args.task_class))
    caller_id, caller_id_source = resolve_caller_id(args.caller_id)
    review_cwd = _resolve_review_cwd(getattr(args, "cd", None))
    roster = load_roster(Path(args.roster))
    state_dir = Path(args.state_dir)
    dry_run = bool(getattr(args, "dry_run", False))
    if not dry_run:
        _raise_if_blocking_round_exists(
            action="sampling another round",
            ignore_pending_grades=bool(args.ignore_pending_grades),
            state_dir=state_dir,
            caller_id=caller_id,
            review_cwd=review_cwd,
            roster_path=Path(args.roster),
            rubric_path=Path(getattr(args, "rubric", default_rubric_path())),
        )
    payload = _select_round_payload(
        roster=roster,
        task_class=task_class,
        review_cwd=review_cwd,
        state_dir=state_dir,
        seed=args.seed,
        caller_id=caller_id,
        caller_id_source=caller_id_source,
        excluded_variant_ids=set(args.exclude_variant_id),
        cleanup_stale_rounds=not dry_run,
    )
    if dry_run:
        emit_toon(
            _dry_run_round_payload(
                payload, task_name=_public_local_task_name(task_class)
            )
        )
        return 0
    write_round(state_dir, payload)
    emit_toon(
        public_round_payload(payload, task_name=_public_local_task_name(task_class))
    )
    return 0


def _current_branch_name(review_cwd: Path) -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=str(review_cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    branch = proc.stdout.strip()
    if not branch or branch == "HEAD":
        return None
    return branch


def _validate_benchmarked_review_runtime(
    *,
    review_cwd: Path,
    allow_unsafe_windows_wsl_fallback: bool,
) -> None:
    codex_executable = shutil.which("codex") or shutil.which("codex.cmd") or "codex"
    validate_codex_runtime(
        tool_name="review-suite",
        codex_executable=codex_executable,
        review_root=review_cwd,
        allow_unsafe_windows_wsl_fallback=allow_unsafe_windows_wsl_fallback,
    )


def _print_findings(result: dict[str, object]) -> bool:
    return print_reviewer_output_section(
        [run for run in list(result.get("runs") or []) if isinstance(run, dict)]
    )


def _ensure_no_pending_grades(
    *,
    ignore_pending_grades: bool,
    state_dir: Path,
    caller_id: str | None,
    review_cwd: Path,
    roster_path: Path,
    rubric_path: Path,
) -> None:
    _raise_if_blocking_round_exists(
        action="starting another round",
        ignore_pending_grades=ignore_pending_grades,
        state_dir=state_dir,
        caller_id=caller_id,
        review_cwd=review_cwd,
        roster_path=roster_path,
        rubric_path=rubric_path,
    )


def _script_command() -> str:
    return format_command([sys.executable, launcher_script_path(__file__).as_posix()])


def _grade_command(
    *,
    round_id: str | None = None,
    task_id: str | None = None,
    rating_pool_id: str = "RATING_POOL_ID",
    rank_groups: list[str] | None = None,
    basis: str = "BASIS",
    state_dir: Path | None = None,
) -> str:
    parts = [
        sys.executable,
        launcher_script_path(__file__).as_posix(),
        "grade",
    ]
    if round_id:
        parts.extend(["--round-id", round_id])
    if task_id:
        parts.extend(["--task-id", task_id])
    parts.extend(["--rating-pool-id", rating_pool_id])
    for rank_group in rank_groups or ["FIRST[,TIED]", "NEXT"]:
        parts.extend(["--rank", rank_group])
    parts.extend(["--basis", basis])
    if state_dir is not None:
        parts.extend(["--state-dir", str(state_dir)])
    return format_command(parts)


def _grade_command_for_payload(
    payload: dict[str, object],
    *,
    state_dir: Path,
) -> str:
    round_id = str(payload.get("round_id") or "").strip() or None
    task_id = (
        str(payload.get("task_id_hint") or payload.get("graded_task_id") or "").strip()
        or None
    )
    return _grade_command(
        round_id=round_id,
        task_id=task_id,
        rating_pool_id=str(payload.get("rating_pool_id") or "").strip()
        or "RATING_POOL_ID",
        rank_groups=grade_rank_placeholders(payload),
        state_dir=state_dir,
    )


def _show_round_command(*, round_id: str) -> str:
    return format_command(
        [
            sys.executable,
            launcher_script_path(__file__).as_posix(),
            "show-round",
            "--round-id",
            round_id,
        ]
    )


def _dismiss_round_command(
    *,
    round_id: str,
) -> str:
    return format_command(
        [
            sys.executable,
            launcher_script_path(__file__).as_posix(),
            "dismiss-round",
            "--round-id",
            round_id,
        ]
    )


def _reroll_command(
    *,
    round_id: str,
    slot: str,
    review_cwd: Path,
    base: str | None,
    roster_path: Path,
    rubric_path: Path,
    state_dir: Path,
    sqlite_path: Path,
    allow_unsafe_windows_wsl_fallback: bool,
) -> str:
    parts = [
        sys.executable,
        launcher_script_path(__file__).as_posix(),
        "reroll-slot",
        "--round-id",
        round_id,
        "--slot",
        slot,
    ]
    if allow_unsafe_windows_wsl_fallback:
        parts.append("--wsl")
    return format_command(parts)


def _blocked_slots(round_result: dict[str, object]) -> list[str]:
    blocked: list[str] = []
    for run in round_result.get("runs", []):
        if run.get("blocked"):
            blocked.append(str(run["slot"]))
    return blocked


def _newly_usable_output_slots(
    *, previous_payload: dict[str, object], completed_payload: dict[str, object]
) -> set[str]:
    previous_visible = usable_output_slots(previous_payload)
    return usable_output_slots(completed_payload) - previous_visible


def _visible_completed_output_slots(
    *,
    previous_payload: dict[str, object],
    completed_payload: dict[str, object],
    show_all_when_gradeable: bool,
) -> set[str] | None:
    if show_all_when_gradeable:
        if payload_has_blocked_runs(completed_payload):
            return usable_output_slots(completed_payload)
        return None
    return _newly_usable_output_slots(
        previous_payload=previous_payload, completed_payload=completed_payload
    )


def _reroll_rows(
    *,
    round_result: dict[str, object],
    round_id: str,
    review_cwd: Path,
    base: str | None,
    roster_path: Path,
    rubric_path: Path,
    state_dir: Path,
    sqlite_path: Path,
    allow_unsafe_windows_wsl_fallback: bool,
) -> list[dict[str, str]]:
    return [
        {
            "slot": slot,
            "command": _reroll_command(
                round_id=round_id,
                slot=slot,
                review_cwd=review_cwd,
                base=base,
                roster_path=roster_path,
                rubric_path=rubric_path,
                state_dir=state_dir,
                sqlite_path=sqlite_path,
                allow_unsafe_windows_wsl_fallback=allow_unsafe_windows_wsl_fallback,
            ),
        }
        for slot in _blocked_slots(round_result)
    ]


def _has_direct_grade_inputs(
    *,
    task_id: str | None,
    rating_pool_id: str | None,
    rank_groups: list[str] | None,
    basis: str | None,
) -> bool:
    return bool(
        str(task_id or "").strip()
        and str(rating_pool_id or "").strip()
        and rank_groups
        and str(basis or "").strip()
    )


def _resolve_pending_round_for_direct_grade(
    *,
    state_dir: Path,
    caller_id: str | None,
    review_cwd: Path,
    task_class: str,
) -> dict[str, object] | None:
    pending = [
        payload
        for payload in find_pending_rounds_for_caller(
            state_dir=state_dir, caller_id=caller_id, review_cwd=review_cwd
        )
        if str(payload.get("task_class") or "") == task_class
    ]
    if len(pending) == 1:
        return pending[-1]
    if len(pending) > 1:
        raise ValueError(
            f"multiple pending {task_class} rounds found for this caller in this repo; use review_suite_arena.py grade --round-id ... to choose one explicitly"
        )
    review_cwd_normalized = normalize_review_cwd_value(review_cwd)
    repo_pending = [
        payload
        for payload in iter_round_payloads(state_dir)
        if str(payload.get("task_class") or "") == task_class
        and normalize_record_review_cwd_value(payload) == review_cwd_normalized
        and round_needs_caller_grade(payload)
    ]
    if len(repo_pending) == 1:
        return repo_pending[0]
    if len(repo_pending) > 1:
        raise ValueError(
            f"multiple pending {task_class} rounds found in this repo; use review_suite_arena.py grade --round-id ... to choose one explicitly"
        )
    return None


def _completed_round_payload(
    *,
    round_result: dict[str, object],
    grade_command: str | None = None,
    reroll_rows: list[dict[str, str]] | None = None,
    status: str = "completed_ungraded",
    grade: dict[str, object] | None = None,
    manual: bool | None = None,
) -> dict[str, object]:
    actions: list[dict[str, object]] = []
    if grade_command:
        actions.append({"kind": "grade", "cmd": grade_command})
    if reroll_rows:
        actions.extend(
            {"kind": "reroll", "slot": row["slot"], "cmd": row["command"]}
            for row in reroll_rows
        )
    if (
        status == "completed_ungraded"
        and grade_command
        and not bool(round_result.get("blocked"))
        and not reroll_rows
        and grade is None
        and not manual
    ):
        return {"Action": {"cmd": grade_command}}
    if manual and not grade_command and not reroll_rows and grade is None:
        note = (
            "manual review blocked; read Output"
            if round_result.get("blocked")
            else "manual review complete"
        )
        return {"Action": {"note": note}}
    payload: dict[str, object] = {
        "status": status,
        "blocked": bool(round_result.get("blocked")),
        "runs": round_result["runs"],
    }
    if actions:
        payload["actions"] = actions
    if grade is not None:
        payload["grade"] = grade
    if manual:
        payload["manual"] = True
    return payload


def _public_local_task_name(task_class: str) -> str:
    return public_task_name(task_class)


def _output_isatty() -> bool:
    return output_isatty()


def _print_round_banner(*, task_name: str, round_id: str) -> None:
    print(f"[review-suite] round {task_name} {round_id}", file=sys.stderr, flush=True)


def _orchestrator_banner_task_name(
    *,
    step_name: str | None = None,
    step_position: int | None = None,
    step_total: int | None = None,
) -> str:
    name = str(step_name or "").strip()
    if step_position and step_total and step_position > 0 and step_total > 0 and name:
        return f"review {step_position}/{step_total} {name}"
    return "review"


def _public_summary(summary: dict[str, object]) -> dict[str, object]:
    public = copy.deepcopy(summary)
    task_classes = dict(public.get("task_classes") or {})
    public["task_classes"] = {
        _public_local_task_name(task): value for task, value in task_classes.items()
    }
    return public


def _normalize_arena_task_class(task_class: str) -> str:
    return PUBLIC_ARENA_TASK_CLASS_ALIASES.get(task_class, task_class)


def _public_operational_state(state: dict[str, object]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    task_classes = dict(state.get("task_classes") or {})
    for task in TASK_CLASSES:
        slot = dict(task_classes.get(task) or {})
        champions = [
            str(value)
            for value in list(slot.get("champion_variant_ids") or [])
            if str(value).strip()
        ]
        probation = [
            str(value)
            for value in list(slot.get("probation_variant_ids") or [])
            if str(value).strip()
        ]
        stable = [
            str(value)
            for value in list(slot.get("stable_variant_ids") or [])
            if str(value).strip()
        ]
        rows.append(
            {
                "task": _public_local_task_name(task),
                "champions": ",".join(champions) if champions else None,
                "probation": ",".join(probation) if probation else None,
                "stable": ",".join(stable) if stable else None,
                "cooldowns": len(dict(slot.get("cooldowns") or {})),
            }
        )
    return {
        "status": "ok",
        "tasks": rows,
    }


def _print_next_steps(
    *,
    round_id: str,
    task_id: str,
    round_result: dict[str, object],
    review_cwd: Path,
    base: str | None,
    roster_path: Path,
    rubric_path: Path,
    state_dir: Path,
    sqlite_path: Path,
    allow_unsafe_windows_wsl_fallback: bool,
) -> None:
    blocked_slots = _blocked_slots(round_result)
    if blocked_slots:
        for slot in blocked_slots:
            print(f"Reroll {slot}:", file=sys.stderr, flush=True)
            print(
                _reroll_command(
                    round_id=round_id,
                    slot=slot,
                    review_cwd=review_cwd,
                    base=base,
                    roster_path=roster_path,
                    rubric_path=rubric_path,
                    state_dir=state_dir,
                    sqlite_path=sqlite_path,
                    allow_unsafe_windows_wsl_fallback=allow_unsafe_windows_wsl_fallback,
                ),
                file=sys.stderr,
                flush=True,
            )
        return
    print("Grade:", file=sys.stderr, flush=True)
    print(
        _grade_command(rank_groups=grade_rank_placeholders(round_result)),
        file=sys.stderr,
        flush=True,
    )


def _review_output_refs(runs: list[dict[str, object]]) -> list[str]:
    refs: list[str] = []
    for run in runs:
        ref = str(run.get("reviewer_output_ref") or "").strip()
        if ref:
            refs.append(ref)
    return refs


def _is_orchestrated_round_payload(payload: dict[str, object]) -> bool:
    return bool(payload.get("arena_round")) and bool(
        str(payload.get("orchestrator_step") or "").strip()
    )


def _round_is_graded(payload: dict[str, object]) -> bool:
    return bool(str(payload.get("graded_at") or "").strip())


def _record_standalone_review_anchor_for_round(
    *,
    state_dir: Path,
    review_cwd: Path,
    lane: str,
    base: str | None,
    review_scope: dict[str, object],
    round_payload: dict[str, object],
    completed_payload: dict[str, object],
    round_id: str,
    task_id: str,
) -> bool:
    anchor_payload = {**round_payload, **completed_payload}
    if _is_orchestrated_round_payload(anchor_payload):
        return False
    try:
        record_review_anchor(
            state_dir=state_dir,
            review_cwd=review_cwd,
            lane=lane,
            base=base,
            review_scope=review_scope,
            round_id=round_id,
            task_id=task_id,
            output_refs=_review_output_refs(list(completed_payload.get("runs") or [])),
        )
    except Exception as exc:  # pragma: no cover - warning path only
        _record_anchor_warning(exc)
        return False
    return True


def _orchestrator_review_state_dir(state_dir: Path) -> Path:
    return state_dir / "orchestrator" / "review-rounds"


def _orchestrator_roster_from_round(payload: dict[str, object]) -> dict[str, object]:
    variants: dict[str, dict[str, object]] = {}
    task_class = (
        str(payload.get("task_class") or "phase_review").strip() or "phase_review"
    )
    for run in list(payload.get("runs") or []):
        if not isinstance(run, dict):
            continue
        variant_id = str(run.get("variant_id") or "").strip()
        model = str(run.get("model") or "").strip()
        reasoning_effort = str(run.get("reasoning_effort") or "").strip()
        if not variant_id or not model or not reasoning_effort:
            continue
        variant = {
            "id": variant_id,
            "state": "active",
            "model": model,
            "reasoning_effort": reasoning_effort,
            "task_classes": [task_class],
        }
        service_tier = str(run.get("service_tier") or "").strip()
        if service_tier:
            variant["service_tier"] = service_tier
        variants[variant_id] = variant
    if not variants:
        raise ValueError(
            f"round {payload.get('round_id')} is missing reviewer model metadata"
        )
    return {"settings": {}, "variants": list(variants.values())}


def _completed_orchestrator_review_result(
    *,
    completed: dict[str, object],
    lane: str,
    step_name: str,
    review_cwd: Path,
    state_dir: Path,
    round_state_dir: Path,
    task_id: str | None,
    grading_required: bool,
    step_position: int | None = None,
    step_total: int | None = None,
) -> dict[str, object]:
    completed["task_id_hint"] = task_id or str(completed.get("round_id") or "")
    completed["grading_required"] = bool(grading_required)
    completed["public_task"] = lane
    completed["orchestrator_step"] = step_name
    if step_position is not None and step_total is not None:
        completed["orchestrator_step_position"] = step_position
        completed["orchestrator_step_total"] = step_total
    write_round(round_state_dir, completed)
    result = public_round_result(completed)
    _print_findings(completed)
    apply_review_cost_delta_best_effort(
        state_dir=state_dir, review_cwd=review_cwd, record=completed
    )
    output_refs = _review_output_refs(
        [run for run in list(completed.get("runs") or []) if isinstance(run, dict)]
    )
    review_scope = dict(completed.get("review_scope") or {})
    return {
        "round_id": str(completed.get("round_id") or ""),
        "lane": lane,
        "kind": "review",
        "status": result.get("status"),
        "blocked": bool(result.get("blocked")),
        "reviewed_head": str(
            review_scope.get("reviewed_head") or review_scope.get("commit_end") or ""
        ),
        "output_refs": output_refs,
        "runs": list(result.get("runs") or []),
        "round_state_dir": str(round_state_dir),
        "grading_required": bool(grading_required),
        "arena_round": bool(completed.get("arena_round")),
        "needs_grade": bool(grading_required or completed.get("arena_round"))
        and bool(round_needs_caller_grade(completed)),
        "graded": _round_is_graded(completed),
    }


def _orchestrator_review_slots(count: int) -> list[str]:
    labels = list(PUBLIC_REVIEWER_LABELS)
    return [
        labels[index]
        if index < len(labels)
        else public_reviewer_label(f"reviewer_{index + 1}")
        for index in range(count)
    ]


def _orchestrator_phase_review_request(
    *, review_cwd: Path, base: str, custom_instructions: str | None = None
):
    return _orchestrator_review_request(
        review_cwd=review_cwd,
        base=base,
        task_class="phase_review",
        custom_instructions=custom_instructions,
    )


def _orchestrator_review_request(
    *,
    review_cwd: Path,
    base: str,
    task_class: str,
    custom_instructions: str | None = None,
):
    instruction_builders = {
        "phase_review": build_phase_instructions,
        "pr_review": build_pr_instructions,
    }
    instruction_builder = instruction_builders.get(task_class)
    if instruction_builder is None:
        raise ValueError(f"unsupported orchestrator arena task_class: {task_class}")
    request = build_local_review_request(
        review_cwd=review_cwd,
        base=base,
        commit_values=None,
        instruction_builder=instruction_builder,
        custom_instructions=custom_instructions or "",
    )
    if not request.prompt.strip():
        raise ValueError(
            f"orchestrator review requires a non-empty {task_class} prompt"
        )
    return request


def run_orchestrator_review_step(
    *,
    lane: str,
    step_name: str,
    reviewer_count: int,
    model: str,
    reasoning_effort: str,
    service_tier: str | None,
    review_cwd: Path,
    state_dir: Path,
    sqlite_path: Path,
    review_scope: dict[str, object],
    task_id: str | None,
    progress_interval_seconds: int,
    allow_unsafe_windows_wsl_fallback: bool,
    grading_required: bool = False,
    step_position: int | None = None,
    step_total: int | None = None,
    on_round_started: Callable[[dict[str, object]], None] | None = None,
    custom_instructions: str | None = None,
) -> dict[str, object]:
    if reviewer_count <= 0:
        raise ValueError("reviewer_count must be > 0")
    base = str(review_scope.get("base") or "").strip()
    if not base:
        raise ValueError("review_scope.base is required")
    request = _orchestrator_phase_review_request(
        review_cwd=review_cwd,
        base=base,
        custom_instructions=custom_instructions,
    )
    return _run_orchestrator_manual_review_step(
        lane=lane,
        step_name=step_name,
        reviewer_count=reviewer_count,
        model=model,
        reasoning_effort=reasoning_effort,
        service_tier=service_tier,
        review_cwd=review_cwd,
        state_dir=state_dir,
        sqlite_path=sqlite_path,
        review_scope=request.review_scope,
        prompt=request.prompt,
        task_id=task_id,
        progress_interval_seconds=progress_interval_seconds,
        allow_unsafe_windows_wsl_fallback=allow_unsafe_windows_wsl_fallback,
        grading_required=grading_required,
        step_position=step_position,
        step_total=step_total,
        on_round_started=on_round_started,
    )


def run_orchestrated_arena_round(
    *,
    lane: str,
    task_class: str,
    step_name: str,
    rating_pool_id: str | None = None,
    reporting_pool: bool = False,
    variant_groups: list[list[str]] | None = None,
    variant_ids: list[str] | None = None,
    review_cwd: Path,
    state_dir: Path,
    sqlite_path: Path,
    review_scope: dict[str, object],
    task_id: str | None,
    progress_interval_seconds: int,
    allow_unsafe_windows_wsl_fallback: bool,
    step_position: int | None = None,
    step_total: int | None = None,
    on_round_started: Callable[[dict[str, object]], None] | None = None,
    custom_instructions: str | None = None,
) -> dict[str, object]:
    base = str(review_scope.get("base") or "").strip()
    if not base:
        raise ValueError("review_scope.base is required")
    request = _orchestrator_review_request(
        review_cwd=review_cwd,
        base=base,
        task_class=task_class,
        custom_instructions=custom_instructions,
    )
    _validate_benchmarked_review_runtime(
        review_cwd=review_cwd,
        allow_unsafe_windows_wsl_fallback=allow_unsafe_windows_wsl_fallback,
    )
    roster_path = default_roster_path()
    rubric_path = default_rubric_path()
    roster = load_roster(roster_path)
    records = read_jsonl(
        state_dir / RUN_LOG_FILENAME
    ) + ungraded_round_exposure_records(state_dir)
    operational_state = load_operational_state(state_dir / OPERATIONAL_STATE_FILENAME)
    payload = select_pair(
        roster=roster,
        operational_state=operational_state,
        records=records,
        task_class=task_class,
        review_cwd=review_cwd,
        seed=None,
        caller_id=None,
        caller_id_source=None,
        excluded_variant_ids=set(),
        rating_pool_id=rating_pool_id,
        variant_groups=variant_groups,
        variant_ids=variant_ids,
    )
    branch_default = task_id or _current_branch_name(review_cwd) or payload["round_id"]
    payload["task_class"] = task_class
    payload["task_id_hint"] = branch_default
    payload["roster_path"] = str(roster_path)
    payload["rubric_path"] = str(rubric_path)
    payload["public_task"] = lane
    payload["orchestrator_step"] = step_name
    payload["reporting_pool"] = reporting_pool
    payload["arena_round"] = True
    payload["grading_required"] = True
    payload.setdefault("sampled_at", utc_now_iso())
    payload["review_cwd_normalized"] = normalize_review_cwd_value(review_cwd)
    payload["review_cwd"] = str(review_cwd)
    payload["review_scope"] = dict(request.review_scope)
    payload["requested_prompt"] = request.prompt
    payload["allow_unsafe_windows_wsl_fallback"] = allow_unsafe_windows_wsl_fallback
    payload["progress_interval_seconds"] = progress_interval_seconds
    if step_position is not None and step_total is not None:
        payload["orchestrator_step_position"] = step_position
        payload["orchestrator_step_total"] = step_total
    write_round(state_dir, payload)
    if on_round_started is not None:
        on_round_started(
            {
                "round_id": str(payload.get("round_id") or ""),
                "round_state_dir": str(state_dir),
                "reviewed_head": str(
                    request.review_scope.get("reviewed_head")
                    or request.review_scope.get("commit_end")
                    or ""
                ),
            }
        )
    _print_round_banner(
        task_name=_orchestrator_banner_task_name(
            step_name=step_name,
            step_position=step_position,
            step_total=step_total,
        ),
        round_id=str(payload["round_id"]),
    )
    if includes_deep_review_effort(list(payload.get("runs") or [])):
        print_deep_review_wait_note()
    completed = run_round(
        round_payload=payload,
        roster=roster,
        state_dir=state_dir,
        review_cwd=review_cwd,
        prompt=request.prompt,
        review_scope=request.review_scope,
        sqlite_path=sqlite_path,
        progress_interval_seconds=progress_interval_seconds,
        allow_unsafe_windows_wsl_fallback=allow_unsafe_windows_wsl_fallback,
    )
    completed["task_id_hint"] = branch_default
    completed["task_class"] = task_class
    completed["arena_round"] = True
    completed["roster_path"] = str(roster_path)
    completed["rubric_path"] = str(rubric_path)
    completed["grading_required"] = True
    return _completed_orchestrator_review_result(
        completed=completed,
        lane=lane,
        step_name=step_name,
        review_cwd=review_cwd,
        state_dir=state_dir,
        round_state_dir=state_dir,
        task_id=branch_default,
        grading_required=True,
        step_position=step_position,
        step_total=step_total,
    )


def run_orchestrator_arena_step(**kwargs) -> dict[str, object]:
    return run_orchestrated_arena_round(**kwargs)


def _run_orchestrator_manual_review_step(
    *,
    lane: str,
    step_name: str,
    reviewer_count: int,
    model: str,
    reasoning_effort: str,
    service_tier: str | None,
    review_cwd: Path,
    state_dir: Path,
    sqlite_path: Path,
    review_scope: dict[str, object],
    prompt: str,
    task_id: str | None,
    progress_interval_seconds: int,
    allow_unsafe_windows_wsl_fallback: bool,
    grading_required: bool = False,
    step_position: int | None = None,
    step_total: int | None = None,
    on_round_started: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    if reviewer_count <= 0:
        raise ValueError("reviewer_count must be > 0")
    if not prompt.strip():
        raise ValueError("orchestrator review requires a non-empty prompt")
    variant_id = "-".join(
        part for part in [model, reasoning_effort, service_tier] if part
    )
    variant = {
        "id": variant_id,
        "state": "active",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "task_classes": ["phase_review"],
    }
    if service_tier:
        variant["service_tier"] = service_tier
    round_state_dir = _orchestrator_review_state_dir(state_dir)
    round_id = make_round_id("phase_review", review_cwd=review_cwd)
    runs = []
    for slot in _orchestrator_review_slots(reviewer_count):
        run = {
            "slot": slot,
            "variant_id": variant_id,
            "model": model,
            "reasoning_effort": reasoning_effort,
        }
        if service_tier:
            run["service_tier"] = service_tier
        runs.append(run)
    payload = {
        "round_id": round_id,
        "task_class": "phase_review",
        "public_task": lane,
        "orchestrator_step": step_name,
        "grading_required": bool(grading_required),
        "sampled_at": utc_now_iso(),
        "review_cwd_normalized": normalize_review_cwd_value(review_cwd),
        "review_cwd": str(review_cwd),
        "review_scope": dict(review_scope),
        "requested_prompt": prompt,
        "allow_unsafe_windows_wsl_fallback": allow_unsafe_windows_wsl_fallback,
        "progress_interval_seconds": progress_interval_seconds,
        "status": "sampled",
        "runs": runs,
    }
    if step_position is not None and step_total is not None:
        payload["orchestrator_step_position"] = step_position
        payload["orchestrator_step_total"] = step_total
    write_round(round_state_dir, payload)
    if on_round_started is not None:
        on_round_started(
            {
                "round_id": round_id,
                "round_state_dir": str(round_state_dir),
                "reviewed_head": str(
                    review_scope.get("reviewed_head")
                    or review_scope.get("commit_end")
                    or ""
                ),
            }
        )
    _print_round_banner(
        task_name=_orchestrator_banner_task_name(
            step_name=step_name,
            step_position=step_position,
            step_total=step_total,
        ),
        round_id=round_id,
    )
    if includes_deep_review_effort(list(payload.get("runs") or [])):
        print_deep_review_wait_note()
    completed = run_round(
        round_payload=payload,
        roster={"settings": {}, "variants": [variant]},
        state_dir=round_state_dir,
        review_cwd=review_cwd,
        prompt=prompt,
        review_scope=review_scope,
        sqlite_path=sqlite_path,
        progress_interval_seconds=progress_interval_seconds,
        allow_unsafe_windows_wsl_fallback=allow_unsafe_windows_wsl_fallback,
    )
    return _completed_orchestrator_review_result(
        completed=completed,
        lane=lane,
        step_name=step_name,
        review_cwd=review_cwd,
        state_dir=state_dir,
        round_state_dir=round_state_dir,
        task_id=task_id,
        grading_required=grading_required,
        step_position=step_position,
        step_total=step_total,
    )


def resume_orchestrator_review_step(
    *,
    round_id: str,
    lane: str,
    step_name: str,
    review_cwd: Path,
    state_dir: Path,
    round_state_dir: Path | None,
    sqlite_path: Path,
    task_id: str | None,
    progress_interval_seconds: int,
    grading_required: bool = False,
) -> dict[str, object]:
    resolved_round_state_dir = round_state_dir or _orchestrator_review_state_dir(
        state_dir
    )
    payload = load_round(resolved_round_state_dir, round_id)
    status = str(payload.get("status") or "").strip()
    resolved_review_cwd = Path(str(payload.get("review_cwd") or review_cwd))
    if status == "completed":
        completed = payload
    elif status == "running":
        completed = collect_round_results(
            round_payload=payload,
            roster=_orchestrator_roster_from_round(payload),
            state_dir=resolved_round_state_dir,
            review_cwd=resolved_review_cwd,
            sqlite_path=sqlite_path,
            progress_interval_seconds=progress_interval_seconds,
            wait=True,
        )
    elif status in {"sampled", "failed"}:
        completed = run_round(
            round_payload=payload,
            roster=_orchestrator_roster_from_round(payload),
            state_dir=resolved_round_state_dir,
            review_cwd=resolved_review_cwd,
            prompt=str(payload.get("requested_prompt") or ""),
            review_scope=dict(payload.get("review_scope") or {}),
            sqlite_path=sqlite_path,
            progress_interval_seconds=int(
                payload.get("progress_interval_seconds") or progress_interval_seconds
            ),
            allow_unsafe_windows_wsl_fallback=bool(
                payload.get("allow_unsafe_windows_wsl_fallback")
            ),
        )
    else:
        raise ValueError(
            f"round {round_id} is {status or 'missing status'}, not resumable"
        )
    return _completed_orchestrator_review_result(
        completed=completed,
        lane=lane,
        step_name=step_name,
        review_cwd=resolved_review_cwd,
        state_dir=state_dir,
        round_state_dir=resolved_round_state_dir,
        task_id=task_id,
        grading_required=grading_required or bool(payload.get("grading_required")),
        step_position=(
            payload.get("orchestrator_step_position")
            if isinstance(payload.get("orchestrator_step_position"), int)
            else None
        ),
        step_total=(
            payload.get("orchestrator_step_total")
            if isinstance(payload.get("orchestrator_step_total"), int)
            else None
        ),
    )


def run_orchestrator_followup_review_step(
    *,
    model: str,
    reasoning_effort: str,
    service_tier: str | None,
    review_cwd: Path,
    state_dir: Path,
    sqlite_path: Path,
    review_scope: dict[str, object],
    prompt: str,
    task_id: str | None,
    progress_interval_seconds: int,
    allow_unsafe_windows_wsl_fallback: bool,
    on_round_started: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    result = _run_orchestrator_manual_review_step(
        lane="review-followup",
        step_name="followup",
        reviewer_count=1,
        model=model,
        reasoning_effort=reasoning_effort,
        service_tier=service_tier,
        review_cwd=review_cwd,
        state_dir=state_dir,
        sqlite_path=sqlite_path,
        review_scope=review_scope,
        prompt=prompt,
        task_id=task_id,
        progress_interval_seconds=progress_interval_seconds,
        allow_unsafe_windows_wsl_fallback=allow_unsafe_windows_wsl_fallback,
        grading_required=False,
        on_round_started=on_round_started,
    )
    result["kind"] = "followup"
    return result


def _record_anchor_warning(exc: Exception) -> None:
    print(
        f"[review-suite] WARNING: failed to record workflow anchor: {exc}",
        file=sys.stderr,
        flush=True,
    )


def run_benchmarked_round(
    *,
    task_class: str,
    review_cwd: Path,
    roster_path: Path,
    rubric_path: Path,
    state_dir: Path,
    sqlite_path: Path,
    seed: int | None,
    progress_interval_seconds: int,
    allow_unsafe_windows_wsl_fallback: bool,
    review_scope: dict[str, object],
    prompt: str,
    caller_id: str | None,
    caller_id_source: str | None,
    ignore_pending_grades: bool,
    task_id: str | None,
    rating_pool_id: str | None,
    rank_groups: list[str] | None,
    basis: str | None,
    note: str | None,
    public_task_name: str | None = None,
    allow_stage_step_down: bool = False,
    dry_run: bool = False,
) -> int:
    public_task = str(public_task_name or _public_local_task_name(task_class))
    direct_grade_requested = _has_direct_grade_inputs(
        task_id=task_id,
        rating_pool_id=rating_pool_id,
        rank_groups=rank_groups,
        basis=basis,
    )
    if any((rating_pool_id, rank_groups, basis)) and not direct_grade_requested:
        raise ValueError(
            "direct grading requires --task-id, --rating-pool-id, --rank, and --basis"
        )
    if dry_run and direct_grade_requested:
        raise ValueError("--dry-run cannot be combined with direct grading inputs")
    if dry_run:
        roster = load_roster(roster_path)
        payload = _select_round_payload(
            roster=roster,
            task_class=task_class,
            review_cwd=review_cwd,
            state_dir=state_dir,
            seed=seed,
            caller_id=caller_id,
            caller_id_source=caller_id_source,
            cleanup_stale_rounds=False,
        )
        emit_toon(_dry_run_round_payload(payload, task_name=public_task))
        return 0
    if direct_grade_requested:
        pending_round = _resolve_pending_round_for_direct_grade(
            state_dir=state_dir,
            caller_id=caller_id,
            review_cwd=review_cwd,
            task_class=task_class,
        )
        if pending_round is not None:
            roster = load_roster(roster_path)
            rubric = load_rubric(rubric_path)
            result = _record_grade_result(
                roster=roster,
                rubric=rubric,
                state_dir=state_dir,
                round_id=str(pending_round["round_id"]),
                task_id=str(task_id),
                rating_pool_id=str(rating_pool_id),
                rank_groups=list(rank_groups or []),
                basis=str(basis),
                note=note,
                caller_id=caller_id,
                caller_id_source=caller_id_source,
            )
            result = dict(result)
            result["task"] = public_task
            emit_toon(result)
            return 0
    _ensure_no_pending_grades(
        ignore_pending_grades=ignore_pending_grades,
        state_dir=state_dir,
        caller_id=caller_id,
        review_cwd=review_cwd,
        roster_path=roster_path,
        rubric_path=rubric_path,
    )
    if not allow_stage_step_down:
        guard_no_stage_step_down(
            lane=public_task,
            review_cwd=review_cwd,
            base=str(review_scope.get("base") or ""),
            state_dir=state_dir,
            review_scope=review_scope,
        )
    if review_scope.get("base"):
        ensure_clean_git_worktree(review_cwd, review_scope=review_scope)
    _validate_benchmarked_review_runtime(
        review_cwd=review_cwd,
        allow_unsafe_windows_wsl_fallback=allow_unsafe_windows_wsl_fallback,
    )
    roster = load_roster(roster_path)
    rubric = load_rubric(rubric_path)
    payload = _select_round_payload(
        roster=roster,
        task_class=task_class,
        review_cwd=review_cwd,
        state_dir=state_dir,
        seed=seed,
        caller_id=caller_id,
        caller_id_source=caller_id_source,
    )
    payload["roster_path"] = str(roster_path)
    payload["rubric_path"] = str(rubric_path)
    write_round(state_dir, payload)
    _print_round_banner(task_name=public_task, round_id=str(payload["round_id"]))
    if includes_deep_review_effort(list(payload.get("runs") or [])):
        print_deep_review_wait_note()
    completed = run_round(
        round_payload=payload,
        roster=roster,
        state_dir=state_dir,
        review_cwd=review_cwd,
        prompt=prompt,
        review_scope=review_scope,
        sqlite_path=sqlite_path,
        progress_interval_seconds=progress_interval_seconds,
        allow_unsafe_windows_wsl_fallback=allow_unsafe_windows_wsl_fallback,
    )
    round_result = public_round_result(
        completed,
        output_slots=_newly_usable_output_slots(
            previous_payload=payload, completed_payload=completed
        ),
    )
    _print_findings(completed)

    branch_default = task_id or _current_branch_name(review_cwd) or payload["round_id"]
    completed["task_id_hint"] = branch_default
    completed["roster_path"] = str(roster_path)
    completed["rubric_path"] = str(rubric_path)
    write_round(state_dir, completed)
    apply_review_cost_delta_best_effort(
        state_dir=state_dir, review_cwd=review_cwd, record=completed
    )
    if not bool(round_result.get("blocked")):
        _record_standalone_review_anchor_for_round(
            state_dir=state_dir,
            review_cwd=review_cwd,
            lane=public_task,
            base=str(review_scope.get("base") or "") or None,
            review_scope=review_scope,
            round_payload=payload,
            completed_payload=completed,
            round_id=str(payload["round_id"]),
            task_id=branch_default,
        )
    interactive_output = _output_isatty()
    if interactive_output:
        _print_next_steps(
            round_id=payload["round_id"],
            task_id=branch_default,
            round_result=round_result,
            review_cwd=review_cwd,
            base=str(review_scope.get("base") or "") or None,
            roster_path=roster_path,
            rubric_path=rubric_path,
            state_dir=state_dir,
            sqlite_path=sqlite_path,
            allow_unsafe_windows_wsl_fallback=allow_unsafe_windows_wsl_fallback,
        )
    if not direct_grade_requested:
        if not interactive_output:
            grade_command = _grade_command_for_payload(completed, state_dir=state_dir)
            reroll_rows = _reroll_rows(
                round_result=round_result,
                round_id=payload["round_id"],
                review_cwd=review_cwd,
                base=str(review_scope.get("base") or "") or None,
                roster_path=roster_path,
                rubric_path=rubric_path,
                state_dir=state_dir,
                sqlite_path=sqlite_path,
                allow_unsafe_windows_wsl_fallback=allow_unsafe_windows_wsl_fallback,
            )
            if round_result.get("blocked"):
                grade_command = None
            emit_toon(
                _completed_round_payload(
                    round_result=round_result,
                    grade_command=grade_command,
                    reroll_rows=reroll_rows,
                )
            )
        return 0

    grade_result = _record_grade_result(
        roster=roster,
        rubric=rubric,
        state_dir=state_dir,
        round_id=payload["round_id"],
        task_id=str(task_id),
        rating_pool_id=str(rating_pool_id),
        rank_groups=list(rank_groups or []),
        basis=str(basis),
        note=note,
        caller_id=caller_id,
        caller_id_source=caller_id_source,
    )
    emit_toon(
        _completed_round_payload(
            round_result=round_result,
            status="graded",
            grade=grade_result,
        )
    )
    return 0


def _record_grade_result(
    *,
    roster: dict[str, object],
    rubric: dict[str, object],
    state_dir: Path,
    round_id: str,
    task_id: str,
    rating_pool_id: str,
    rank_groups: list[str],
    basis: str,
    note: str | None,
    caller_id: str | None,
    caller_id_source: str | None,
) -> dict[str, object]:
    round_payload = load_round(state_dir, round_id)
    record = build_record_from_grade(
        round_payload=round_payload,
        roster=roster,
        rubric=rubric,
        task_id=task_id,
        rating_pool_id=rating_pool_id,
        rank_groups=rank_groups,
        basis=basis,
        shared_note=note,
    )
    if not append_record_if_new(state_dir, record):
        result: dict[str, object] = {
            "status": "ok",
            "duplicate": True,
            "recorded": False,
            "round_id": round_id,
        }
    else:
        result = {
            "status": "ok",
            "recorded": True,
            "duplicate": False,
            "round_id": round_id,
        }
    round_payload["grade_recorded"] = bool(result["recorded"] or result["duplicate"])
    round_payload["grade_refresh_pending"] = True
    round_payload["graded_by_caller_id"] = caller_id
    round_payload["graded_by_caller_id_source"] = caller_id_source
    round_payload["graded_task_id"] = task_id
    write_round(state_dir, round_payload)
    _refresh_state_and_reports(state_dir=state_dir, roster=roster)
    round_payload["graded_at"] = str(record["recorded_at"])
    round_payload.pop("grade_refresh_pending", None)
    write_round(state_dir, round_payload)
    return result


def _refresh_state_and_reports(*, state_dir: Path, roster: dict[str, object]) -> None:
    with (
        state_lock(state_dir, "runs"),
        state_lock(state_dir, "operational-state"),
        state_lock(state_dir, "reports"),
    ):
        _refresh_state_and_reports_locked(
            state_dir=state_dir,
            roster=roster,
            records=_read_enriched_run_records(state_dir),
        )


def _read_enriched_run_records(state_dir: Path) -> list[dict[str, object]]:
    records, changed = enrich_record_repo_names(
        state_dir, read_jsonl(state_dir / RUN_LOG_FILENAME)
    )
    if changed:
        write_jsonl(state_dir / RUN_LOG_FILENAME, records)
    return records


def _refresh_state_and_reports_locked(
    *,
    state_dir: Path,
    roster: dict[str, object],
    records: list[dict[str, object]],
) -> None:
    operational_state = load_operational_state(state_dir / OPERATIONAL_STATE_FILENAME)
    summary = aggregate_records(
        roster=roster,
        records=records,
        operational_state=operational_state,
    )
    next_state = promote(roster, summary, operational_state)
    write_json(state_dir / OPERATIONAL_STATE_FILENAME, next_state)
    refreshed_summary = aggregate_records(
        roster=roster,
        records=records,
        operational_state=next_state,
    )
    write_reports(state_dir, refreshed_summary)


def cmd_run(args: argparse.Namespace) -> int:
    task_class = _normalize_arena_task_class(str(args.task_class))
    review_cwd = _resolve_review_cwd(args.cd)
    caller_id, caller_id_source = resolve_caller_id(args.caller_id)
    return run_benchmarked_round(
        task_class=task_class,
        review_cwd=review_cwd,
        roster_path=Path(args.roster),
        rubric_path=Path(args.rubric),
        state_dir=Path(args.state_dir),
        sqlite_path=Path(args.sqlite_path),
        seed=args.seed,
        progress_interval_seconds=DEFAULT_PROGRESS_INTERVAL_SECONDS,
        allow_unsafe_windows_wsl_fallback=bool(args.wsl),
        review_scope={"base": args.base},
        prompt="",
        caller_id=caller_id,
        caller_id_source=caller_id_source,
        ignore_pending_grades=bool(args.ignore_pending_grades),
        task_id=args.task_id,
        rating_pool_id=args.rating_pool_id,
        rank_groups=args.rank_groups,
        basis=args.basis,
        note=args.note,
        public_task_name=_public_local_task_name(task_class),
        dry_run=bool(getattr(args, "dry_run", False)),
    )


def cmd_run_round(args: argparse.Namespace) -> int:
    review_cwd = _resolve_review_cwd(args.cd)
    print(
        f"[review-suite] input_repo={review_cwd} effective_cwd={effective_execution_cwd(review_cwd, bool(args.wsl))} cwd={Path.cwd()}",
        file=sys.stderr,
        flush=True,
    )
    roster = load_roster(Path(args.roster))
    state_dir = Path(args.state_dir)
    payload = load_round(state_dir, args.round_id)
    review_scope = copy.deepcopy(payload.get("review_scope") or {})
    if args.base and not str(review_scope.get("base") or "").strip():
        review_scope["base"] = args.base
    completed = run_round(
        round_payload=payload,
        roster=roster,
        state_dir=state_dir,
        review_cwd=review_cwd,
        prompt=str(payload.get("requested_prompt") or ""),
        review_scope=review_scope,
        sqlite_path=Path(args.sqlite_path),
        progress_interval_seconds=DEFAULT_PROGRESS_INTERVAL_SECONDS,
        allow_unsafe_windows_wsl_fallback=bool(args.wsl),
    )
    completed["task_id_hint"] = _current_branch_name(review_cwd) or str(
        payload["round_id"]
    )
    write_round(state_dir, completed)
    apply_review_cost_delta_best_effort(
        state_dir=state_dir, review_cwd=review_cwd, record=completed
    )
    result = public_round_result(
        completed,
        output_slots=_visible_completed_output_slots(
            previous_payload=payload,
            completed_payload=completed,
            show_all_when_gradeable=False,
        ),
    )
    _print_findings(completed)
    if not bool(result.get("blocked")):
        _record_standalone_review_anchor_for_round(
            state_dir=state_dir,
            review_cwd=review_cwd,
            lane=_public_local_task_name(
                str(payload.get("task_class") or completed.get("task_class") or "")
            ),
            base=str(review_scope.get("base") or "") or None,
            review_scope=review_scope,
            round_payload=payload,
            completed_payload=completed,
            round_id=str(completed.get("round_id") or payload["round_id"]),
            task_id=str(completed["task_id_hint"]),
        )
    if not _output_isatty():
        emit_toon(
            _completed_round_payload(
                round_result=result,
                grade_command=None
                if result.get("blocked")
                else _grade_command_for_payload(completed, state_dir=state_dir),
            )
        )
    return 0


def cmd_reroll_slot(args: argparse.Namespace) -> int:
    roster = load_roster(Path(args.roster))
    state_dir = Path(args.state_dir)
    records = read_jsonl(
        state_dir / RUN_LOG_FILENAME
    ) + ungraded_round_exposure_records(state_dir)
    original = load_round(state_dir, args.round_id)
    payload = build_reroll_slot_payload(
        round_payload=original,
        roster=roster,
        operational_state=load_operational_state(
            state_dir / OPERATIONAL_STATE_FILENAME
        ),
        records=records,
        slot=args.slot,
        seed=args.seed,
    )
    review_cwd = (
        Path(args.cd).resolve()
        if args.cd
        else Path(str(original["review_cwd"])).resolve()
    )
    review_scope = copy.deepcopy(original.get("review_scope") or {})
    if args.base:
        review_scope["base"] = args.base
    payload["review_scope"] = review_scope
    payload["requested_prompt"] = str(original.get("requested_prompt") or "")
    payload["review_cwd"] = str(review_cwd)
    payload["review_cwd_normalized"] = (
        normalize_review_cwd_value(review_cwd)
        if args.cd
        else str(
            original.get("review_cwd_normalized")
            or normalize_review_cwd_value(review_cwd)
        )
    )
    for key in (
        "allow_unsafe_windows_wsl_fallback",
        "progress_interval_seconds",
        "public_task",
        "orchestrator_step",
        "orchestrator_step_position",
        "orchestrator_step_total",
        "arena_round",
        "grading_required",
        "roster_path",
        "rubric_path",
        "rating_pool_id",
        "reporting_pool",
        "schedule_index",
        "schedule_length",
    ):
        if key in original:
            payload[key] = copy.deepcopy(original[key])
    if _output_isatty():
        _print_round_banner(
            task_name=_public_local_task_name(
                str(original.get("task_class") or payload.get("task_class") or "")
            ),
            round_id=str(payload["round_id"]),
        )
    print(f"[review-suite] reroll {args.slot}", file=sys.stderr, flush=True)
    write_round(state_dir, payload)
    completed = run_round(
        round_payload=payload,
        roster=roster,
        state_dir=state_dir,
        review_cwd=review_cwd,
        prompt=str(original.get("requested_prompt") or ""),
        review_scope=review_scope,
        sqlite_path=Path(args.sqlite_path),
        progress_interval_seconds=DEFAULT_PROGRESS_INTERVAL_SECONDS,
        allow_unsafe_windows_wsl_fallback=bool(args.wsl),
    )
    result = public_round_result(
        completed,
        output_slots=_visible_completed_output_slots(
            previous_payload=payload,
            completed_payload=completed,
            show_all_when_gradeable=True,
        ),
    )
    _print_findings(completed)
    if _output_isatty():
        _print_next_steps(
            round_id=payload["round_id"],
            task_id=_current_branch_name(review_cwd) or payload["round_id"],
            round_result=result,
            review_cwd=review_cwd,
            base=str(review_scope.get("base") or "") or None,
            roster_path=Path(args.roster),
            rubric_path=Path(args.rubric),
            state_dir=state_dir,
            sqlite_path=Path(args.sqlite_path),
            allow_unsafe_windows_wsl_fallback=bool(args.wsl),
        )
    completed["task_id_hint"] = _current_branch_name(review_cwd) or payload["round_id"]
    if not bool(result.get("blocked")):
        _record_standalone_review_anchor_for_round(
            state_dir=state_dir,
            review_cwd=review_cwd,
            lane=_public_local_task_name(
                str(original.get("task_class") or payload.get("task_class") or "")
            ),
            base=str(review_scope.get("base") or "") or None,
            review_scope=review_scope,
            round_payload=payload,
            completed_payload=completed,
            round_id=str(payload["round_id"]),
            task_id=str(completed["task_id_hint"]),
        )
    write_round(state_dir, completed)
    apply_review_cost_delta_best_effort(
        state_dir=state_dir, review_cwd=review_cwd, record=completed
    )
    if not _output_isatty():
        emit_toon(
            _completed_round_payload(
                round_result=result,
                grade_command=None
                if result.get("blocked")
                else _grade_command_for_payload(completed, state_dir=state_dir),
            )
        )
    return 0


def cmd_resume_round(args: argparse.Namespace) -> int:
    if args.cd:
        print(
            f"[review-suite] ignoring --cd={Path(args.cd).resolve()} on resume-round; stored round review_cwd is authoritative",
            file=sys.stderr,
            flush=True,
        )
    roster = load_roster(Path(args.roster))
    state_dir = Path(args.state_dir)
    payload = load_round(state_dir, args.round_id)
    if payload.get("status") == "completed":
        _print_findings(payload)
        result = public_round_result(payload)
        emit_toon(
            _completed_round_payload(
                round_result=result,
                grade_command=(
                    _grade_command_for_payload(payload, state_dir=state_dir)
                    if round_needs_caller_grade(payload) and not result.get("blocked")
                    else None
                ),
            )
        )
        return 0
    if payload.get("status") != "running":
        raise ValueError(
            f"round {args.round_id} is {payload.get('status')}, not running"
        )
    review_cwd = Path(str(payload["review_cwd"]))
    completed = collect_round_results(
        round_payload=payload,
        roster=roster,
        state_dir=state_dir,
        review_cwd=review_cwd,
        sqlite_path=Path(args.sqlite_path),
        progress_interval_seconds=DEFAULT_PROGRESS_INTERVAL_SECONDS,
        wait=True,
    )
    result = public_round_result(
        completed,
        output_slots=_visible_completed_output_slots(
            previous_payload=payload,
            completed_payload=completed,
            show_all_when_gradeable=True,
        ),
    )
    _print_findings(completed)
    completed["task_id_hint"] = _current_branch_name(review_cwd) or str(
        payload["round_id"]
    )
    apply_review_cost_delta_best_effort(
        state_dir=state_dir, review_cwd=review_cwd, record=completed
    )
    if not bool(result.get("blocked")):
        review_scope = dict(
            completed.get("review_scope") or payload.get("review_scope") or {}
        )
        _record_standalone_review_anchor_for_round(
            state_dir=state_dir,
            review_cwd=review_cwd,
            lane=_public_local_task_name(
                str(payload.get("task_class") or completed.get("task_class") or "")
            ),
            base=str(review_scope.get("base") or "") or None,
            review_scope=review_scope,
            round_payload=payload,
            completed_payload=completed,
            round_id=str(completed.get("round_id") or payload["round_id"]),
            task_id=str(completed["task_id_hint"]),
        )
    if not _output_isatty():
        emit_toon(
            _completed_round_payload(
                round_result=result,
                grade_command=None
                if result.get("blocked")
                else _grade_command_for_payload(completed, state_dir=state_dir),
            )
        )
    return 0


def _round_output_payload(payload: dict[str, object]) -> dict[str, object]:
    return {
        "round_id": payload.get("round_id"),
        "task": public_task_name(str(payload.get("task_class") or "")),
        "task_class": payload.get("task_class"),
        "status": payload.get("status"),
        "recorded_at": payload.get("recorded_at"),
        "review_cwd": payload.get("review_cwd") or payload.get("review_cwd_normalized"),
        "graded_at": payload.get("graded_at"),
        "runs": [
            {
                "slot": run.get("slot"),
                "variant_id": run.get("variant_id"),
                "model": run.get("model"),
                "reasoning_effort": run.get("reasoning_effort"),
                "review_status": run.get("review_status"),
                "grade_blocked": run.get("grade_blocked"),
                "grade_block_reason": run.get("grade_block_reason"),
                "reviewer_output_ref": run.get("reviewer_output_ref"),
                "reviewer_output": final_display_body(run),
            }
            for run in list(payload.get("runs") or [])
            if isinstance(run, dict)
        ],
    }


def _print_round_outputs(payload: dict[str, object]) -> None:
    write_text(f"round_id: {payload.get('round_id')}")
    write_text(f"task: {public_task_name(str(payload.get('task_class') or ''))}")
    write_text(f"task_class: {payload.get('task_class')}")
    write_text(f"status: {payload.get('status')}")

    recorded_at = str(payload.get("recorded_at") or "").strip()
    if recorded_at:
        write_text(f"recorded_at: {recorded_at}")
    review_cwd = str(
        payload.get("review_cwd") or payload.get("review_cwd_normalized") or ""
    ).strip()
    if review_cwd:
        write_text(f"review_cwd: {review_cwd}")
    graded_at = str(payload.get("graded_at") or "").strip()
    if graded_at:
        write_text(f"graded_at: {graded_at}")
    write_text("")
    runs = [run for run in list(payload.get("runs") or []) if isinstance(run, dict)]
    if not runs:
        write_text("(no runs recorded)")
        return
    for run in runs:
        write_text(reviewer_output_heading(run))
        write_text(final_display_body(run))
        write_text("")


def _round_output_sort_key(payload: dict[str, object]) -> str:
    return str(
        payload.get("review_completed_at")
        or payload.get("recorded_at")
        or payload.get("sampled_at")
        or payload.get("round_id")
        or ""
    )


def _recoverable_round_state_dirs(state_dir: Path) -> list[Path]:
    candidates = [state_dir, _orchestrator_review_state_dir(state_dir)]
    seen: set[str] = set()
    result: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        key = str(resolved).lower() if sys.platform == "win32" else str(resolved)
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _iter_recoverable_round_outputs(state_dir: Path) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for round_state_dir in _recoverable_round_state_dirs(state_dir):
        for payload in iter_round_payloads(round_state_dir):
            task_class = str(payload.get("task_class") or "")
            if task_class in {"phase_review", "pr_review"}:
                payloads.append(payload)
    return payloads


def _last_round_outputs(
    *,
    state_dir: Path,
    review_cwd: Path | None,
    task: str | None,
) -> list[dict[str, object]]:
    normalized_cwd = (
        normalize_review_cwd_value(review_cwd) if review_cwd is not None else None
    )
    latest_by_task: dict[str, dict[str, object]] = {}
    for payload in _iter_recoverable_round_outputs(state_dir):
        public_task = public_task_name(str(payload.get("task_class") or ""))
        if task and public_task != task:
            continue
        if normalized_cwd:
            payload_cwd = str(normalize_record_review_cwd_value(payload) or "")
            if payload_cwd != normalized_cwd:
                continue
        previous = latest_by_task.get(public_task)
        if previous is None or _round_output_sort_key(payload) > _round_output_sort_key(
            previous
        ):
            latest_by_task[public_task] = payload
    return [latest_by_task[key] for key in sorted(latest_by_task)]


def cmd_show_round(args: argparse.Namespace) -> int:
    state_dir = Path(args.state_dir)
    payload: dict[str, object] | None = None
    load_error: ValueError | None = None
    for round_state_dir in _recoverable_round_state_dirs(state_dir):
        try:
            payload = load_round(round_state_dir, args.round_id)
            break
        except ValueError as exc:
            load_error = exc
    if payload is None:
        raise load_error or ValueError(f"unknown round: {args.round_id}")
    if bool(getattr(args, "json", False)):
        write_text(
            json.dumps(_round_output_payload(payload), indent=2, ensure_ascii=False)
        )
        return 0
    _print_round_outputs(payload)
    return 0


def cmd_show_last(args: argparse.Namespace) -> int:
    review_cwd = resolve_repo_root(args.cd) if getattr(args, "cd", None) else None
    payloads = _last_round_outputs(
        state_dir=Path(args.state_dir),
        review_cwd=review_cwd,
        task=getattr(args, "task", None),
    )
    if bool(getattr(args, "json", False)):
        write_text(
            json.dumps(
                [_round_output_payload(payload) for payload in payloads],
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    if not payloads:
        target = f" for {review_cwd}" if review_cwd is not None else ""
        task = f" ({args.task})" if getattr(args, "task", None) else ""
        write_text(f"no stored local review outputs found{target}{task}")
        return 0
    for idx, payload in enumerate(payloads):
        if idx:
            write_text("=" * 72)
        _print_round_outputs(payload)
    return 0


def _cost_row_payload(row) -> dict[str, object]:
    payload = {
        "repo": row.repo,
        "folder": row.folder,
        "branch": row.branch,
        "pr": row.pr_number,
        "worker_model": row.worker_model,
        "implementation_tokens": row.implementation_tokens,
        "implementation_cost_usd": row.implementation_cost_usd,
        "latest_review": row.latest_review,
        "t1_sessions": row.lane_sessions.get("review_t1", 0),
        "t3_sessions": row.lane_sessions.get("review_t3", 0),
        "review_seconds": int(round(row.review_seconds)),
        "tokens": row.tokens,
        "cost_usd": row.cost_usd,
    }
    followup_sessions = row.lane_sessions.get("review_followup", 0)
    if followup_sessions:
        payload["fu_sessions"] = followup_sessions
    return payload


def cmd_costs(args: argparse.Namespace) -> int:
    state_dir = Path(args.state_dir)
    with orchestrator_store_lock(
        state_dir=state_dir,
        name=REVIEW_COSTS_LOCK_NAME,
        timeout_seconds=REVIEW_COSTS_LOCK_TIMEOUT_SECONDS,
    ):
        return _cmd_costs(args, state_dir=state_dir)


def _cmd_costs(args: argparse.Namespace, *, state_dir: Path) -> int:
    include_all = bool(getattr(args, "all", False))
    review_cwd = (
        None
        if include_all
        else (resolve_repo_root(args.cd) if getattr(args, "cd", None) else None)
    )
    if review_cwd is None and not include_all:
        review_cwd = resolve_repo_root(None)
    if include_all:
        rows = collect_review_cost_rows(
            state_dir=state_dir,
            review_cwd=None,
            include_all=True,
            codex_home=Path(args.codex_home)
            if getattr(args, "codex_home", None)
            else None,
        )
        update_review_cost_row_cache(state_dir=state_dir, rows=rows)
    else:
        row = read_review_cost_row_for_cwd(state_dir=state_dir, review_cwd=review_cwd)
        if row is None:
            raise ValueError(
                f"no cached cost row for {review_cwd}; run costs --all to rebuild it"
            )
        if not row.implementation_costs_refreshed:
            raise ValueError(
                f"cached implementation costs are stale for {review_cwd}; run costs --all to rebuild them"
            )
        rows = [row]
    output_path = (
        Path(args.output)
        if getattr(args, "output", None)
        else state_dir / DEFAULT_COST_REPORT_FILENAME
    )
    report_rows = read_review_cost_row_cache(state_dir) or rows
    write_review_cost_report(rows=report_rows, output_path=output_path)
    scoped_totals = {
        "tokens": sum(row.tokens for row in rows),
        "cost_usd": round(sum(row.cost_usd for row in rows), 6),
        "implementation_tokens": sum(row.implementation_tokens for row in rows),
        "implementation_cost_usd": round(
            sum(row.implementation_cost_usd for row in rows), 6
        ),
    }
    payload = {
        "status": "ok",
        "rows": len(rows),
        "report_rows": len(report_rows),
        "report": str(output_path),
        "total_tokens": scoped_totals["tokens"],
        "total_cost_usd": scoped_totals["cost_usd"],
        "total_implementation_tokens": scoped_totals["implementation_tokens"],
        "total_implementation_cost_usd": scoped_totals["implementation_cost_usd"],
        "costs": [_cost_row_payload(row) for row in rows],
    }
    if len(report_rows) != len(rows):
        payload["report_total_tokens"] = sum(row.tokens for row in report_rows)
        payload["report_total_cost_usd"] = round(
            sum(row.cost_usd for row in report_rows), 6
        )
        payload["report_total_implementation_tokens"] = sum(
            row.implementation_tokens for row in report_rows
        )
        payload["report_total_implementation_cost_usd"] = round(
            sum(row.implementation_cost_usd for row in report_rows),
            6,
        )
    if bool(getattr(args, "json", False)):
        print(json.dumps(payload, indent=2))
    else:
        emit_toon(payload)
    return 0


def cmd_grade(args: argparse.Namespace) -> int:
    roster = load_roster(Path(args.roster))
    rubric = load_rubric(Path(args.rubric))
    state_dir = Path(args.state_dir)
    caller_id, caller_id_source = resolve_caller_id(args.caller_id)
    round_id = str(args.round_id or "").strip()
    if not round_id:
        review_cwd = resolve_repo_root(None)
        blocking = [
            payload
            for payload in find_blocking_rounds_for_caller(
                state_dir=state_dir, caller_id=caller_id, review_cwd=review_cwd
            )
            if round_needs_caller_grade(payload)
        ]
        if not blocking:
            raise ValueError(
                "no completed ungraded round found for this caller/worktree; pass --round-id to grade a specific round"
            )
        round_id = str(blocking[-1].get("round_id") or "").strip()
    round_payload = load_round(state_dir, round_id)
    task_id = str(
        args.task_id
        or round_payload.get("task_id_hint")
        or round_payload.get("graded_task_id")
        or round_id
    ).strip()
    result = _record_grade_result(
        roster=roster,
        rubric=rubric,
        state_dir=state_dir,
        round_id=round_id,
        task_id=task_id,
        rating_pool_id=args.rating_pool_id,
        rank_groups=args.rank_groups,
        basis=args.basis,
        note=args.note,
        caller_id=caller_id,
        caller_id_source=caller_id_source,
    )
    if not _output_isatty():
        emit_toon(result)
    return 0


def cmd_dismiss_round(args: argparse.Namespace) -> int:
    state_dir = Path(args.state_dir)
    payload = load_round(state_dir, args.round_id)
    caller_id, caller_id_source = resolve_caller_id(args.caller_id)
    previous_status = str(payload.get("status") or "unknown")
    if str(payload.get("graded_at") or "").strip():
        raise ValueError(
            f"round {args.round_id} is already graded and cannot be dismissed"
        )
    if previous_status == "dismissed":
        emit_toon(
            {
                "status": "ok",
                "dismissed": True,
                "already_dismissed": True,
                "round_id": args.round_id,
                "reason": str(payload.get("dismissed_reason") or ""),
            }
        )
        return 0
    if round_has_live_reviewer_process(payload):
        raise ValueError(
            f"round {args.round_id} still has live reviewer processes; refusing dismissal"
        )
    dismissed_at = utc_now_iso()
    payload["status"] = "dismissed"
    payload["dismissed_at"] = dismissed_at
    payload["dismissed_reason"] = str(args.reason or "").strip() or "manual_dismiss"
    payload.setdefault("dismissed_previous_status", previous_status)
    if caller_id:
        payload["dismissed_by_caller_id"] = caller_id
        payload["dismissed_by_caller_id_source"] = caller_id_source
    write_round(state_dir, payload)
    emit_toon(
        {
            "status": "ok",
            "dismissed": True,
            "round_id": args.round_id,
            "previous_status": previous_status,
            "reason": payload["dismissed_reason"],
        }
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    if getattr(args, "round_id", None):
        return cmd_show_round(args)
    roster = load_roster(Path(args.roster))
    state_dir = Path(args.state_dir)
    with (
        state_lock(state_dir, "runs"),
        state_lock(state_dir, "operational-state"),
        state_lock(state_dir, "reports"),
    ):
        records = _read_enriched_run_records(state_dir)
        summary = aggregate_records(
            roster=roster,
            records=records,
            operational_state=load_operational_state(
                state_dir / OPERATIONAL_STATE_FILENAME
            ),
        )
        write_reports(state_dir, summary)
    emit_toon(_public_summary(summary))
    return 0


def cmd_refresh(args: argparse.Namespace) -> int:
    roster = load_roster(Path(args.roster))
    state_dir = Path(args.state_dir)
    _refresh_state_and_reports(state_dir=state_dir, roster=roster)
    emit_toon(
        {
            "status": "ok",
            "refreshed": True,
            "summary": str(state_dir / "summary.json"),
            "leaderboard": str(state_dir / "leaderboard.md"),
            "state": str(state_dir / OPERATIONAL_STATE_FILENAME),
        }
    )
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    roster = load_roster(Path(args.roster))
    state_dir = Path(args.state_dir)
    with (
        state_lock(state_dir, "runs"),
        state_lock(state_dir, "operational-state"),
        state_lock(state_dir, "reports"),
    ):
        records = read_jsonl(state_dir / RUN_LOG_FILENAME)
        operational_state = load_operational_state(
            state_dir / OPERATIONAL_STATE_FILENAME
        )
        summary = aggregate_records(
            roster=roster,
            records=records,
            operational_state=operational_state,
        )
        next_state = promote(roster, summary, operational_state)
        if args.apply:
            write_json(state_dir / OPERATIONAL_STATE_FILENAME, next_state)
    emit_toon(_public_operational_state(next_state))
    return 0


def cmd_compact_runs(args: argparse.Namespace) -> int:
    roster = load_roster(Path(args.roster))
    state_dir = Path(args.state_dir)
    run_log_path = state_dir / RUN_LOG_FILENAME
    with (
        state_lock(state_dir, "runs"),
        state_lock(state_dir, "operational-state"),
        state_lock(state_dir, "reports"),
    ):
        records = read_jsonl(run_log_path)
        compacted_records = [compact_benchmark_record(record) for record in records]
        operational_state = load_operational_state(
            state_dir / OPERATIONAL_STATE_FILENAME
        )
        summary_before = aggregate_records(
            roster=roster, records=records, operational_state=operational_state
        )
        summary_after = aggregate_records(
            roster=roster,
            records=compacted_records,
            operational_state=operational_state,
        )
        if _summary_equivalence_view(summary_before) != _summary_equivalence_view(
            summary_after
        ):
            raise ValueError(
                "compacted runs would change aggregate output; aborting migration"
            )
        before_bytes = run_log_path.stat().st_size if run_log_path.exists() else 0
        compacted_text = ""
        if compacted_records:
            compacted_text = (
                "\n".join(
                    json.dumps(record, sort_keys=True) for record in compacted_records
                )
                + "\n"
            )
        after_bytes = len(compacted_text.encode("utf-8"))
        backup_path = None
        report_refreshed = False
        if args.apply:
            if run_log_path.exists():
                backup_path = run_log_path.with_name(
                    f"{run_log_path.name}{args.backup_suffix}"
                )
                shutil.copy2(run_log_path, backup_path)
            write_jsonl(run_log_path, compacted_records)
            write_reports(state_dir, summary_after)
            report_refreshed = True
    payload = {
        "status": "ok",
        "verified": True,
        "applied": bool(args.apply),
        "records": len(records),
        "before_b": before_bytes,
        "after_b": after_bytes,
        "saved_b": before_bytes - after_bytes,
        "reduction_pct": round(
            (((before_bytes - after_bytes) / before_bytes) * 100.0), 3
        )
        if before_bytes
        else 0.0,
        "runs": str(run_log_path),
        "backup": str(backup_path) if backup_path else None,
        "reports": report_refreshed,
    }
    emit_toon(payload)
    return 0


def cmd_compact_rounds(args: argparse.Namespace) -> int:
    result = compact_round_files(Path(args.state_dir), apply=bool(args.apply))
    result.update(
        {
            "status": "ok",
            "applied": bool(args.apply),
            "reduction_pct": round((result["saved_b"] / result["before_b"]) * 100.0, 3)
            if result["before_b"]
            else 0.0,
        }
    )
    emit_toon(result)
    return 0


def cmd_prune_state(args: argparse.Namespace) -> int:
    result = prune_review_state(
        Path(args.state_dir),
        apply=bool(args.apply),
        older_than_days=int(args.older_than_days),
    )
    result["status"] = "ok"
    emit_toon(result)
    return 0


def main() -> int:
    parser = build_parser()
    try:
        args = parser.parse_args()
        if args.command == "run":
            return cmd_run(args)
        if args.command == "sample":
            return cmd_sample(args)
        if args.command == "run-round":
            return cmd_run_round(args)
        if args.command == "resume-round":
            return cmd_resume_round(args)
        if args.command == "show-round":
            return cmd_show_round(args)
        if args.command in {"show-last", "show_last"}:
            return cmd_show_last(args)
        if args.command == "costs":
            return cmd_costs(args)
        if args.command == "reroll-slot":
            return cmd_reroll_slot(args)
        if args.command == "grade":
            return cmd_grade(args)
        if args.command == "dismiss-round":
            return cmd_dismiss_round(args)
        if args.command == "report":
            return cmd_report(args)
        if args.command == "refresh":
            return cmd_refresh(args)
        if args.command == "promote":
            return cmd_promote(args)
        if args.command == "compact-runs":
            return cmd_compact_runs(args)
        if args.command == "compact-rounds":
            return cmd_compact_rounds(args)
        if args.command == "prune-state":
            return cmd_prune_state(args)
    except BlockingRoundError as exc:
        return emit_error(
            str(exc),
            status="usage_error",
            extra={"Action": exc.action_payload},
            help_items=[f"{_script_command()} --help"],
        )
    except ValueError as exc:
        return emit_error(
            str(exc),
            status="usage_error",
            help_items=[f"{_script_command()} --help"],
        )
    except OSError as exc:
        return emit_error(
            str(exc),
            status="error",
            help_items=[f"{_script_command()} --help"],
        )
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
