from __future__ import annotations

import ctypes
import html
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import blake2s
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

from review_suite_runtime_bootstrap import launcher_script_path
from rollout_capture import (
    DEFAULT_SQLITE_STATE_PATH,
    REVIEW_SUBAGENT_SOURCE,
    enrich_thread_record,
    find_review_child_thread,
    find_thread_by_id,
    rollout_activity_summary,
)
from review_suite_core import (
    core_usage_tokens,
    EFFECTIVE_BASE_METADATA_KEYS,
    current_head,
    effective_base_ref,
    format_command,
    has_committed_diff,
    has_worktree_changes,
    inspect_workflow_status,
    is_deep_reasoning_effort,
    launch_captured_child_process,
    meaningful_worktree_status_entries,
    merge_base,
    normalize_cwd,
    normalize_service_tier,
    OPENCODE_MODEL_PREFIX,
    parse_opencode_review_metadata,
    prepare_codex_review_launch,
    price_usage_tokens,
    run_total_tokens,
    terminate_process_tree,
    utc_now,
    utc_now_iso,
    validated_linear_review_range,
    write_text,
)

TASK_CLASSES = ("phase_review", "pr_review")
PAIR_SELECTION_MODES = ("legacy", "slight_bias", "true_scramble")
LOCAL_REVIEW_LANE_STAGE_RANK = {
    "review_t1": 1,
    "review_t3": 3,
}
STALE_REVIEW_STATE_TTL_SECONDS = 24 * 60 * 60
RUN_LOG_FILENAME = "runs.jsonl"
SUMMARY_FILENAME = "summary.json"
ROSTER_FILENAME = "roster.json"
RUBRIC_FILENAME = "rubric_v2.json"
OPERATIONAL_STATE_FILENAME = "operational_state.json"
ROUNDS_DIRNAME = "rounds"
ORCHESTRATOR_ROUND_STATE_DIR = Path("orchestrator") / "review-rounds"
GRADE_BASIS_VALUES = (
    "valid_findings_vs_none",
    "more_valid_findings",
    "better_finding_validity",
    "better_bug_coverage",
    "false_positive_loss",
    "hallucinated_finding_loss",
    "fringe_finding_loss",
    "scope_bloat_loss",
    "tie_clean",
    "tie_both_useful",
)
TERMINAL_REVIEW_COMMANDS = {"clean", "findings"}
TERMINAL_REVIEW_RESULT_PREFIX = "Review result:"
LOW_QUALITY_LOSS_REASON_BASES = (
    "false_positive_loss",
    "hallucinated_finding_loss",
    "fringe_finding_loss",
    "scope_bloat_loss",
    "better_finding_validity",
)
LOW_QUALITY_LOSS_BASES = set(LOW_QUALITY_LOSS_REASON_BASES)
MISSED_BUG_LOSS_BASES = {
    "valid_findings_vs_none",
    "more_valid_findings",
    "better_bug_coverage",
}
VALID_FINDING_WIN_BASES = {
    "valid_findings_vs_none",
    "more_valid_findings",
    "better_finding_validity",
    "better_bug_coverage",
}
BUG_OPPORTUNITY_BASES = VALID_FINDING_WIN_BASES | {"tie_both_useful"}
ALL_PARTICIPANTS_FOUND_BASES = {
    "more_valid_findings",
    "better_bug_coverage",
    "tie_both_useful",
}
PUBLIC_REVIEWER_LABELS = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot")
CAPACITY_COOLDOWN_SECONDS = (30 * 60, 2 * 60 * 60, 6 * 60 * 60, 12 * 60 * 60)
CAPACITY_RETRY_DELAY_SECONDS = 10
CAPACITY_RETRY_MAX_ATTEMPTS = 1
MULTI_REVIEW_DISPATCH_STAGGER_SECONDS = 5.0
VARIANT_STATES = {"active", "disabled", "retired"}
REVIEW_STALL_WARNING_SECONDS = 10 * 60
REVIEW_INACTIVITY_DEADLINE_SECONDS = 30 * 60
REVIEW_ABSOLUTE_DEADLINE_SECONDS = 2 * 60 * 60
TRANSPORT_STALL_GRACE_SECONDS = 3 * 60
TRANSPORT_RECONNECT_PATTERNS = (
    "stream disconnected",
    "ERROR: Reconnecting...",
    "falling back to HTTP",
)
CALLER_ID_ENV_KEYS = (
    ("PWF_SUBAGENT_ID", "pwf_subagent_id"),
    ("CODEX_THREAD_ID", "codex_thread_id"),
)

_PROCESS_SYNCHRONIZE = 0x00100000
_WAIT_TIMEOUT = 0x00000102
_ERROR_ACCESS_DENIED = 5

if os.name == "nt":
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _open_process = _kernel32.OpenProcess
    _open_process.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
    _open_process.restype = ctypes.c_void_p
    _wait_for_single_object = _kernel32.WaitForSingleObject
    _wait_for_single_object.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
    _wait_for_single_object.restype = ctypes.c_ulong
    _close_handle = _kernel32.CloseHandle
    _close_handle.argtypes = (ctypes.c_void_p,)
    _close_handle.restype = ctypes.c_int


@dataclass(frozen=True)
class LocalReviewRequest:
    review_scope: dict[str, Any]
    prompt: str
    target_label: str


def pending_launch_ready(
    *,
    dispatch_stagger_seconds: float,
    last_pending_launch_at: float | None,
    now: float,
) -> bool:
    if dispatch_stagger_seconds <= 0:
        return True
    if last_pending_launch_at is None:
        return True
    return (now - last_pending_launch_at) >= dispatch_stagger_seconds


def includes_deep_review_effort(items: list[dict[str, Any]]) -> bool:
    return any(
        is_deep_reasoning_effort(str(item.get("reasoning_effort") or ""))
        for item in items
    )


def print_deep_review_wait_note() -> None:
    print(
        "[review-suite] this step includes high-reasoning reviewers; reviews can take a long time. Wait for wrapper output.",
        file=sys.stderr,
        flush=True,
    )


def load_custom_instructions(
    *, instructions: str | None, instructions_file: str | None
) -> str | None:
    if instructions is not None and instructions_file is not None:
        raise ValueError("use either --instructions or --instructions-file")
    if instructions_file is not None:
        try:
            payload = Path(instructions_file).read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(str(exc)) from exc
    elif instructions is not None:
        payload = instructions
    else:
        return None
    if not payload.strip():
        raise ValueError("custom instructions must not be empty")
    return payload


def normalize_commit_spec(
    commit_values: list[str] | None,
) -> tuple[str | None, str | None]:
    if not commit_values:
        return None, None
    if len(commit_values) == 1:
        return commit_values[0], None
    if len(commit_values) == 2:
        return commit_values[0], commit_values[1]
    raise ValueError("--commit accepts one sha or two shas")


def build_phase_instructions(target_label: str) -> str:
    return (
        "Review this implementation slice for correctness and regression risk.\n"
        f"The review target is {target_label}.\n"
        f"{build_correctness_review_contract()}"
    )


def build_pr_instructions(target_label: str) -> str:
    return (
        "Review this PR-ready branch diff for correctness and regression risk.\n"
        f"The review target is {target_label}.\n"
        f"{build_correctness_review_contract()}"
    )


def terminal_review_result_instruction() -> str:
    return (
        f"End with exactly one final machine-readable line: `{TERMINAL_REVIEW_RESULT_PREFIX} clean` "
        f"if there are no valid findings, or `{TERMINAL_REVIEW_RESULT_PREFIX} findings` if you reported one or more valid findings."
    )


def build_correctness_review_contract() -> str:
    return (
        "Reviewer output is advisory risk input, not authoritative product direction.\n"
        "Return every concrete finding you can support; do not stop after the first issue.\n"
        "For each finding, include severity, file path and line number when available, violated invariant or owner/source of truth when clear, and fix suggestion.\n"
        "A finding is only valid when it identifies a concrete correctness, regression, integration, security, accessibility, or maintainability risk against stated requirements, docs, code invariants, or explicit contracts.\n"
        "Do not treat UX preference, product-scope speculation, backwards-compat speculation, or alternative product direction as a blocking finding.\n"
        "Do not assume backwards compatibility, legacy behavior, broad fallback behavior, or support for unsupported inputs unless the task, docs, or diff explicitly requires it.\n"
        "If a concern depends on ambiguous product scope or conflicts with explicit user/product direction, report it under `Scope questions / suggestions (non-findings)` with the tradeoff to escalate instead of assigning severity.\n"
        "Do not recommend code changes that reverse explicit product intent.\n"
        "When discussing validation, distinguish focused seam validation from full-suite/CI validation: focused review-relevant checks can be enough to launch the next review round before slow full-suite/CI completes; full-suite/CI remains a merge-readiness requirement.\n"
        "Record full-suite/CI validation status when relevant as pending, passed, failed, or explicitly waived with a reason; do not call a PR final or merge-ready while that status is unknown, and investigate/fix relevant failures first.\n"
        "When applicable, flag correctness-relevant risks from oversized or hard-to-stage diffs, external integration surface breaks, missing regression or integration coverage, and unbounded agent-context injection.\n"
        "Skip style-only comments. If there are no issues, say 'No findings.'\n"
        "Do not suggest localized guards when the evidence points to broader ownership, fallback, retry, lifecycle, concurrency, or persistence issues.\n"
        f"{terminal_review_result_instruction()}"
    )


def _base_review_scope(*, review_cwd: Path, base: str) -> tuple[dict[str, Any], str]:
    base_info = effective_base_ref(review_cwd, base)
    effective_base = str(base_info["base"])
    requested_base = str(base_info["requested_base"])
    merge_base_head = merge_base(review_cwd, effective_base, "HEAD")
    reviewed_head = current_head(review_cwd)
    if requested_base != effective_base:
        target_label = f"base `{effective_base}` (requested `{requested_base}`)"
    else:
        target_label = f"base `{effective_base}`"
    review_scope = {
        "base": effective_base,
        "merge_base": merge_base_head,
        "reviewed_head": reviewed_head,
        "target_label": target_label,
    }
    if requested_base != effective_base:
        review_scope["requested_base"] = requested_base
    for key in EFFECTIVE_BASE_METADATA_KEYS:
        if key in base_info:
            review_scope[key] = base_info[key]
    return review_scope, target_label


def _ensure_base_review_has_committed_diff_or_clean_worktree(
    *, review_cwd: Path, base: str, merge_base: str
) -> None:
    if not has_committed_diff(review_cwd, merge_base, "HEAD") and has_worktree_changes(
        review_cwd
    ):
        raise ValueError(
            f"base review found no committed diff against `{base}`, but the worktree has uncommitted changes. "
            "Commit the intended review changes or stash unrelated worktree changes, then rerun the emitted review.py command."
        )


def _combined_review_instructions(
    *, standard_instructions: str, custom_instructions: str | None
) -> str:
    instruction_text = standard_instructions.strip()
    if not instruction_text:
        raise ValueError("manual review mode requires built-in instructions")
    custom_instruction_text = (
        "" if custom_instructions is None else custom_instructions.strip()
    )
    if not custom_instruction_text:
        return instruction_text
    terminal_instruction = terminal_review_result_instruction()
    if instruction_text.endswith(terminal_instruction):
        base_instruction_text = instruction_text[: -len(terminal_instruction)].rstrip()
        return f"{base_instruction_text}\n\nAdditional review instructions:\n{custom_instruction_text}\n\n{terminal_instruction}"
    return f"{instruction_text}\n\nAdditional review instructions:\n{custom_instruction_text}"


def build_local_review_request(
    *,
    review_cwd: Path,
    base: str,
    commit_values: list[str] | None,
    instruction_builder: Callable[[str], str],
    custom_instructions: str | None,
) -> LocalReviewRequest:
    commit, commit_end = normalize_commit_spec(commit_values)
    if commit:
        if base != "main":
            raise ValueError("use either --base or --commit")
        if commit_end:
            range_info = validated_linear_review_range(
                review_cwd,
                commit,
                commit_end,
                label="native commit-range review",
            )
            ensure_clean_git_worktree(review_cwd)
            target_label = f"commit range `{commit}..{commit_end}`"
            return LocalReviewRequest(
                review_scope={
                    "base": commit,
                    "commit": commit,
                    "commit_end": commit_end,
                    "reviewed_head": range_info["head"],
                    "target_label": target_label,
                },
                prompt=_combined_review_instructions(
                    standard_instructions=instruction_builder(target_label),
                    custom_instructions=custom_instructions,
                ),
                target_label=target_label,
            )
        target_label = f"commit `{commit}`"
        return LocalReviewRequest(
            review_scope={"commit": commit, "target_label": target_label},
            prompt=_combined_review_instructions(
                standard_instructions=instruction_builder(target_label),
                custom_instructions=custom_instructions,
            ),
            target_label=target_label,
        )
    review_scope, target_label = _base_review_scope(
        review_cwd=review_cwd, base=str(base)
    )
    _ensure_base_review_has_committed_diff_or_clean_worktree(
        review_cwd=review_cwd,
        base=str(review_scope["base"]),
        merge_base=str(review_scope["merge_base"]),
    )
    ensure_clean_git_worktree(review_cwd)
    prompt = _combined_review_instructions(
        standard_instructions=instruction_builder(target_label),
        custom_instructions=custom_instructions,
    )
    return LocalReviewRequest(
        review_scope=review_scope,
        target_label=target_label,
        prompt=prompt,
    )


def _is_branch_review_scope(review_scope: dict[str, Any]) -> bool:
    if not str(review_scope.get("base") or "").strip():
        return False
    if str(review_scope.get("commit") or "").strip():
        return False
    if str(review_scope.get("commit_end") or "").strip():
        return False
    return True


def _review_status_command(*, review_cwd: Path, base: str) -> str:
    return format_command(
        [
            sys.executable,
            launcher_script_path(__file__, "review.py").as_posix(),
            "--status",
            "--cd",
            str(review_cwd),
            "--base",
            str(base),
        ]
    )


def guard_no_stage_step_down(
    *,
    lane: str,
    review_cwd: Path,
    base: str,
    state_dir: Path,
    review_scope: dict[str, Any],
) -> None:
    if not _is_branch_review_scope(review_scope):
        return
    lane_rank = LOCAL_REVIEW_LANE_STAGE_RANK.get(lane)
    if lane_rank is None:
        return
    try:
        status = inspect_workflow_status(
            state_dir=state_dir,
            review_cwd=review_cwd,
            base=str(base),
        )
    except ValueError:
        return
    current_stage_lane = str(
        status.get("current_stage_lane")
        or status.get("recommended_lane")
        or status.get("last_reviewed_lane")
        or ""
    ).strip()
    current_stage_rank = LOCAL_REVIEW_LANE_STAGE_RANK.get(current_stage_lane)
    if current_stage_rank is None or lane_rank >= current_stage_rank:
        return
    command = _review_status_command(
        review_cwd=review_cwd,
        base=str(base),
    )
    raise ValueError(
        f"{lane} would step down from the current review stage {current_stage_lane}. "
        "Review-suite lanes are monotonic for a branch: once a branch reaches a higher tier, do not rerun lower tiers after amended commits. "
        f"Run {command} and follow its action. Do not invent a lower-tier final-head requirement. "
        "Only pass --allow-stage-step-down when the user/operator explicitly requests an exceptional lower-tier rerun."
    )


def script_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_roster_path() -> Path:
    return script_root() / "references" / ROSTER_FILENAME


def default_rubric_path() -> Path:
    return script_root() / "references" / RUBRIC_FILENAME


def default_state_dir() -> Path:
    return ensure_state_dir()


def ensure_state_dir() -> Path:
    canonical = Path.home() / ".codex" / "state" / "review-suite"
    canonical.mkdir(parents=True, exist_ok=True)
    return canonical


def default_operational_state() -> dict[str, Any]:
    return {
        "generated_at": utc_now_iso(),
        "task_classes": {
            task_class: {
                "champion_variant_id": None,
                "champion_variant_ids": [],
                "probation_variant_ids": [],
                "stable_variant_ids": [],
                "cooldowns": {},
                "cooldown_failures": {},
            }
            for task_class in TASK_CLASSES
        },
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _json_bytes(payload: dict[str, Any]) -> int:
    return len((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = ""
    if rows:
        text = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    _atomic_write_text(path, text)


@contextmanager
def state_lock(
    state_dir: Path, name: str, *, timeout_seconds: int = 30, poll_seconds: float = 0.1
):
    locks_dir = state_dir / ".locks"
    locks_dir.mkdir(parents=True, exist_ok=True)
    lock_path = locks_dir / f"{name}.lock"
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            lock_path.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for lock: {lock_path}")
            time.sleep(poll_seconds)
    try:
        yield
    finally:
        try:
            lock_path.rmdir()
        except OSError:
            pass


def load_roster(path: Path) -> dict[str, Any]:
    roster = read_json(path)
    ids = set()
    for variant in roster.get("variants", []):
        variant_id = variant["id"]
        if variant_id in ids:
            raise ValueError(f"duplicate variant id: {variant_id}")
        ids.add(variant_id)
        state = variant.get("state", "active")
        if state not in VARIANT_STATES:
            raise ValueError(f"variant {variant_id} has invalid state: {state}")
        for task_class in variant.get("task_classes", []):
            if task_class not in TASK_CLASSES:
                raise ValueError(
                    f"variant {variant_id} has invalid task_class: {task_class}"
                )
    return roster


def load_rubric(path: Path) -> dict[str, Any]:
    rubric = read_json(path)
    if rubric.get("version") != "placement_v1":
        raise ValueError("rubric version must be placement_v1")
    expected = list(GRADE_BASIS_VALUES)
    if rubric.get("basis") != expected:
        raise ValueError(f"rubric basis must be {expected}")
    return rubric


def load_operational_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return default_operational_state()
    payload = read_json(path)
    payload.setdefault("task_classes", {})
    for task_class in TASK_CLASSES:
        payload["task_classes"].setdefault(
            task_class,
            {
                "champion_variant_id": None,
                "champion_variant_ids": [],
                "probation_variant_ids": [],
                "stable_variant_ids": [],
                "cooldowns": {},
                "cooldown_failures": {},
            },
        )
        payload["task_classes"][task_class].setdefault("cooldowns", {})
        cooldown_failures = payload["task_classes"][task_class].setdefault(
            "cooldown_failures", {}
        )
        if not cooldown_failures:
            for variant_id, entry in dict(
                payload["task_classes"][task_class].get("cooldowns") or {}
            ).items():
                cooldown_failures[str(variant_id)] = {
                    "failure_count": int((entry or {}).get("failure_count", 1) or 1),
                    "last_reason": str((entry or {}).get("last_reason") or ""),
                    "last_triggered_at": str(
                        (entry or {}).get("last_triggered_at") or ""
                    ),
                }
        payload["task_classes"][task_class].setdefault("champion_variant_ids", [])
        payload["task_classes"][task_class].setdefault("probation_variant_ids", [])
    _prune_expired_cooldowns(payload)
    return payload


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _payload_reference_time(payload: dict[str, Any]) -> datetime | None:
    timestamps = [
        _parse_timestamp(str(payload.get(key) or ""))
        for key in (
            "review_completed_at",
            "review_started_at",
            "sampled_at",
            "recorded_at",
            "round_started_at",
        )
    ]
    timestamps = [
        timestamp
        if timestamp.tzinfo is not None
        else timestamp.replace(tzinfo=timezone.utc)
        for timestamp in timestamps
        if timestamp is not None
    ]
    if timestamps:
        return max(timestamps)
    file_path = str(payload.get("_round_file_path") or "").strip()
    if file_path:
        try:
            return datetime.fromtimestamp(
                Path(file_path).stat().st_mtime, tz=timezone.utc
            )
        except OSError:
            pass
    return None


def format_cooldown_until_for_display(value: object) -> str:
    timestamp = _parse_timestamp(str(value or ""))
    if timestamp is None:
        return str(value or "")
    return timestamp.astimezone().isoformat(timespec="seconds")


def _prune_expired_cooldowns(
    payload: dict[str, Any], *, now: datetime | None = None
) -> None:
    now = now or utc_now()
    for task_class in TASK_CLASSES:
        slot = payload["task_classes"].setdefault(task_class, {})
        cooldowns = dict(slot.get("cooldowns") or {})
        active: dict[str, Any] = {}
        for variant_id, entry in cooldowns.items():
            until = _parse_timestamp(str((entry or {}).get("until") or ""))
            if until and until > now:
                active[str(variant_id)] = {
                    "until": until.isoformat().replace("+00:00", "Z"),
                    "failure_count": int((entry or {}).get("failure_count", 1) or 1),
                    "last_reason": str(
                        (entry or {}).get("last_reason") or "selected_model_at_capacity"
                    ),
                    "last_triggered_at": str(
                        (entry or {}).get("last_triggered_at") or ""
                    ),
                }
        slot["cooldowns"] = active


def _active_cooldowns(
    operational_state: dict[str, Any], task_class: str
) -> dict[str, dict[str, Any]]:
    _prune_expired_cooldowns(operational_state)
    return dict(operational_state["task_classes"][task_class].get("cooldowns") or {})


def _capacity_cooldown_seconds(failure_count: int) -> int:
    idx = max(0, min(int(failure_count) - 1, len(CAPACITY_COOLDOWN_SECONDS) - 1))
    return CAPACITY_COOLDOWN_SECONDS[idx]


COOLDOWN_BLOCK_REASONS = {
    "selected_model_at_capacity",
    "selected_model_unavailable",
    "opencode_review_failed",
    "review_timed_out",
    "review_transport_stalled",
}
MARKED_COOLDOWN_BLOCK_REASONS = {
    "review_interrupted",
    "missing_reviewer_output",
    "reviewer_process_exited",
}
OPENCODE_ERROR_BLOCK_REASONS = {
    "capacity": "selected_model_at_capacity",
    "unavailable": "selected_model_unavailable",
    "failed": "opencode_review_failed",
    "adapter": "review_tooling_failure",
}
OPENCODE_ERROR_REVIEW_STATUSES = {
    "capacity": "interrupted_capacity",
    "unavailable": "opencode_unavailable",
    "failed": "opencode_failed",
    "adapter": "tooling_failure",
}
OPENCODE_ERROR_STATUS_SUMMARIES = {
    "capacity": "selected_model_at_capacity",
    "unavailable": "OpenCode model is not available.",
    "failed": "OpenCode review failed before a usable result was captured.",
    "adapter": "OpenCode adapter failed before a review was launched.",
}


def _cooldown_unavailability_message(
    *, task_class: str, cooling: dict[str, dict[str, Any]], needed: int
) -> str:
    details = ", ".join(
        f"{variant_id} until {format_cooldown_until_for_display(entry.get('until'))}"
        for variant_id, entry in sorted(cooling.items())
    )
    return f"need at least {needed} active non-cooling variants for {task_class}; cooling down: {details}"


def variant_index(roster: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {variant["id"]: variant for variant in roster["variants"]}


def variant_is_arena_eligible(
    variant: dict[str, Any], task_class: str | None = None
) -> bool:
    return (
        variant.get("state", "active") == "active"
        and variant.get("arena_eligible", True) is True
        and (task_class is None or task_class in variant.get("task_classes", []))
    )


def eligible_variants(roster: dict[str, Any], task_class: str) -> list[dict[str, Any]]:
    return [
        variant
        for variant in roster["variants"]
        if variant_is_arena_eligible(variant, task_class)
    ]


def task_class_config(roster: dict[str, Any], task_class: str) -> dict[str, Any]:
    config = (roster.get("task_classes") or {}).get(task_class) or {}
    return config if isinstance(config, dict) else {}


def configured_pair_selection_mode(roster: dict[str, Any], task_class: str) -> str:
    task_config = task_class_config(roster, task_class)
    raw_mode = task_config.get(
        "selection_mode", (roster.get("settings") or {}).get("selection_mode", "legacy")
    )
    mode = str(raw_mode or "legacy").strip()
    if mode == "scramble":
        return "legacy"
    if mode not in PAIR_SELECTION_MODES:
        allowed = ", ".join(PAIR_SELECTION_MODES)
        raise ValueError(
            f"unknown selection_mode for {task_class}: {mode!r}; expected one of: {allowed}"
        )
    return mode


def summarize_counts(records: list[dict[str, Any]], task_class: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        if record.get("task_class") != task_class:
            continue
        for run in record.get("runs", []):
            variant_id = run["variant_id"]
            counts[variant_id] = counts.get(variant_id, 0) + 1
    return counts


def initial_weight(
    variant: dict[str, Any],
    sample_count: int,
    settings: dict[str, Any],
    *,
    relative_target: float | None = None,
) -> float:
    target = float(bootstrap_target_samples(variant, settings))
    if relative_target is not None:
        target = max(target, float(relative_target))
    if sample_count >= target:
        return 1.0
    boost = float(settings.get("bootstrap_weight_boost", 2.0))
    remaining = target - sample_count
    return 1.0 + (remaining / max(target, 1)) * boost


def bootstrap_target_samples(variant: dict[str, Any], settings: dict[str, Any]) -> int:
    return int(
        variant.get(
            "bootstrap_target_samples",
            settings.get("default_bootstrap_target_samples", 8),
        )
    )


def relative_underuse_target(
    variants: list[dict[str, Any]],
    sample_counts: dict[str, int],
    settings: dict[str, Any],
) -> float | None:
    ratio = float(settings.get("relative_underuse_ratio", 0.0) or 0.0)
    if ratio <= 0.0:
        return None
    established_counts = [
        sample_counts.get(variant["id"], 0)
        for variant in variants
        if sample_counts.get(variant["id"], 0)
        >= bootstrap_target_samples(variant, settings)
    ]
    if not established_counts:
        return None
    return float(statistics.median(established_counts)) * ratio


def effective_underuse_target(
    variant: dict[str, Any],
    variants: list[dict[str, Any]],
    sample_counts: dict[str, int],
    settings: dict[str, Any],
) -> float:
    target = float(bootstrap_target_samples(variant, settings))
    relative_target = relative_underuse_target(variants, sample_counts, settings)
    if relative_target is not None:
        target = max(target, float(relative_target))
    return target


def is_under_sampled(
    variant: dict[str, Any],
    sample_counts: dict[str, int],
    settings: dict[str, Any],
    variants: list[dict[str, Any]],
) -> bool:
    return sample_counts.get(variant["id"], 0) < effective_underuse_target(
        variant, variants, sample_counts, settings
    )


def probation_min_samples(settings: dict[str, Any]) -> int:
    return int(
        settings.get("probation_min_samples", settings.get("promotion_min_samples", 20))
    )


def probation_max_elo(settings: dict[str, Any]) -> float:
    return float(settings.get("probation_max_elo", 1500.0))


def _acting_champion_variants(
    *,
    roster: dict[str, Any],
    operational_state: dict[str, Any],
    records: list[dict[str, Any]],
    task_class: str,
    available_variants: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    settings = roster.get("settings", {})
    min_samples = int(settings.get("promotion_min_samples", 20))
    probation_ids = set(
        str(item)
        for item in (
            operational_state["task_classes"][task_class].get("probation_variant_ids")
            or []
        )
    )
    indexed = variant_index(roster)
    available_ids = {variant["id"] for variant in available_variants}
    leaderboard = aggregate_records(
        roster=roster, records=records, operational_state=operational_state
    )["task_classes"][task_class]["leaderboard"]
    acting: list[dict[str, Any]] = []
    for row in leaderboard:
        variant_id = str(row.get("variant_id") or "")
        if (
            not variant_id
            or variant_id not in available_ids
            or variant_id in probation_ids
        ):
            continue
        if int(row.get("sample_count", 0) or 0) < min_samples:
            continue
        variant = indexed.get(variant_id)
        if variant is not None:
            acting.append(variant)
    return acting


def _seed_with_offset(seed: int | None, offset: int) -> int | None:
    if seed is None:
        return None
    return int(seed) + offset


def pick_weighted_without_replacement(
    variants: list[dict[str, Any]],
    sample_counts: dict[str, int],
    settings: dict[str, Any],
    count: int,
    seed: int | None,
    *,
    relative_target: float | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    available = variants[:]
    chosen: list[dict[str, Any]] = []
    for _ in range(count):
        weights = [
            initial_weight(
                variant,
                sample_counts.get(variant["id"], 0),
                settings,
                relative_target=relative_target,
            )
            for variant in available
        ]
        total = sum(weights)
        roll = rng.uniform(0.0, total)
        upto = 0.0
        pick = 0
        for idx, weight in enumerate(weights):
            upto += weight
            if roll <= upto:
                pick = idx
                break
        chosen.append(available.pop(pick))
    return chosen


def pick_custom_weighted_without_replacement(
    variants: list[dict[str, Any]],
    weights_by_variant_id: dict[str, float],
    count: int,
    seed: int | None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    available = variants[:]
    chosen: list[dict[str, Any]] = []
    for _ in range(count):
        weights = [
            max(float(weights_by_variant_id.get(variant["id"], 1.0)), 0.01)
            for variant in available
        ]
        total = sum(weights)
        roll = rng.uniform(0.0, total)
        upto = 0.0
        pick = 0
        for idx, weight in enumerate(weights):
            upto += weight
            if roll <= upto:
                pick = idx
                break
        chosen.append(available.pop(pick))
    return chosen


def compute_cost_usd(variant: dict[str, Any], usage: dict[str, Any]) -> float | None:
    cost = price_usage_tokens(dict(variant.get("pricing") or {}), usage)
    return round(cost, 6) if cost is not None else None


def variant_service_tier(variant: dict[str, Any]) -> str | None:
    service_tier = normalize_service_tier(variant.get("service_tier"))
    if not service_tier:
        return None
    if "supported_service_tiers" not in variant:
        return service_tier
    supported = {
        normalize_service_tier(item)
        for item in list(variant.get("supported_service_tiers") or [])
    }
    supported.discard(None)
    if service_tier not in supported:
        raise ValueError(
            f"variant {variant.get('id') or variant.get('model') or '<unknown>'} does not support service_tier={service_tier}"
        )
    return service_tier


def format_decimal(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.1f}"


def format_signed_decimal(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):+.1f}"


def format_markdown_inline(value: Any) -> str:
    return (
        html.escape(re.sub(r"[\r\n]+", " ", str(value)), quote=False)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("`", "\\`")
    )


def format_compact_tokens(value: float | None) -> str:
    if value is None:
        return "-"
    value = float(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}m"
    if value >= 100_000:
        return f"{value / 1_000:.0f}k"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.0f}"


def format_cost_cents(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100.0:.2f}c"


def total_usage_tokens(usage: dict[str, Any]) -> int:
    # Shared comparison total: uncached input (cache writes included) + output
    # (including reasoning). Keep this aligned with review_costs and Arena.
    return core_usage_tokens(usage)


def rounds_dir(state_dir: Path) -> Path:
    return state_dir / ROUNDS_DIRNAME


def round_path(state_dir: Path, round_id: str) -> Path:
    return rounds_dir(state_dir) / f"{round_id}.json"


def _round_prompt_is_needed(payload: dict[str, Any]) -> bool:
    return str(payload.get("status") or "") == "completed"


def compact_round_payload_for_storage(payload: dict[str, Any]) -> dict[str, Any]:
    if str(payload.get("status") or "") not in {"completed", "dismissed"}:
        return deepcopy(payload)
    compacted = deepcopy(payload)
    compacted.pop("_round_file_path", None)
    if not _round_prompt_is_needed(compacted):
        compacted.pop("requested_prompt", None)
    for run in list(compacted.get("runs") or []):
        if not isinstance(run, dict):
            continue
        classification = _classification_for_run(run)
        run["review_status"] = classification["review_status"]
        run["status_summary"] = classification["status_summary"]
        run["grade_blocked"] = classification["grade_blocked"]
        run["grade_block_reason"] = classification["grade_block_reason"]
        for key in (
            "command",
            "stdout",
            "stderr",
            "stdout_path",
            "stderr_path",
            "final_message_path",
            "stderr_progress_offset",
            "_rollout_activity_epoch",
        ):
            run.pop(key, None)
    return compacted


def _compact_round_file(path: Path, *, apply: bool) -> tuple[int, int, bool] | None:
    try:
        before = path.stat().st_size
        payload = read_json(path)
    except Exception:
        return None
    compacted = compact_round_payload_for_storage(payload)
    changed = compacted != payload
    after = _json_bytes(compacted) if changed else before
    if changed and apply:
        cleanup_succeeded = True
        for run in list(payload.get("runs") or []):
            if isinstance(run, dict):
                cleanup_succeeded = _cleanup_run_artifacts(run) and cleanup_succeeded
        if not cleanup_succeeded:
            write_json(path, payload)
            raise OSError(f"reviewer artifact cleanup incomplete: {path.name}")
        write_json(path, compacted)
    return before, after, changed


def compact_round_files(state_dir: Path, *, apply: bool = False) -> dict[str, Any]:
    dirs = unique_round_state_dirs(
        [state_dir, state_dir / ORCHESTRATOR_ROUND_STATE_DIR]
    )
    result = {"checked": 0, "changed": 0, "before_b": 0, "after_b": 0}
    for round_state_dir in dirs:
        directory = rounds_dir(round_state_dir)
        if not directory.exists():
            continue
        with state_lock(round_state_dir, "compact-rounds"):
            for path in sorted(directory.glob("*.json")):
                if apply:
                    with state_lock(round_state_dir, f"round-{path.stem}"):
                        stats = _compact_round_file(path, apply=True)
                else:
                    stats = _compact_round_file(path, apply=False)
                if stats is None:
                    continue
                before, after, changed = stats
                result["checked"] += 1
                result["before_b"] += before
                result["after_b"] += after
                if not changed:
                    continue
                result["changed"] += 1
    result["saved_b"] = result["before_b"] - result["after_b"]
    return result


def normalize_review_cwd_value(review_cwd: Path | str | None) -> str | None:
    if review_cwd is None:
        return None
    try:
        return normalize_cwd(str(review_cwd))
    except Exception:
        return None


def normalize_record_review_cwd_value(payload: dict[str, Any]) -> str | None:
    return normalize_review_cwd_value(
        payload.get("review_cwd_normalized")
    ) or normalize_review_cwd_value(payload.get("review_cwd"))


def resolve_caller_id(
    explicit_caller_id: str | None = None,
) -> tuple[str | None, str | None]:
    explicit = (explicit_caller_id or "").strip()
    if explicit:
        return explicit, "arg"
    subagent_id = (os.environ.get("PWF_SUBAGENT_ID") or "").strip()
    thread_id = (os.environ.get("CODEX_THREAD_ID") or "").strip()
    if subagent_id and thread_id:
        return f"{thread_id}:{subagent_id}", "codex_thread_id+pwf_subagent_id"
    for env_name, source in CALLER_ID_ENV_KEYS:
        value = (os.environ.get(env_name) or "").strip()
        if value:
            return value, source
    return None, None


def _round_has_recorded_grade(payload: dict[str, Any]) -> bool:
    return bool(str(payload.get("graded_at") or "").strip())


def round_needs_caller_grade(payload: dict[str, Any]) -> bool:
    if str(payload.get("status") or "") != "completed":
        return False
    if _round_has_recorded_grade(payload):
        return False
    if payload_has_blocked_runs(payload):
        return False
    return True


def iter_round_payloads(state_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    directory = rounds_dir(state_dir)
    if not directory.exists():
        return entries
    for path in sorted(directory.glob("*.json")):
        try:
            payload = read_json(path)
        except Exception:
            continue
        payload.setdefault("_round_file_path", str(path))
        entries.append(payload)
    return entries


def _round_state_dir_key(path: Path) -> str:
    resolved = path.resolve(strict=False)
    return str(resolved).lower() if sys.platform == "win32" else str(resolved)


def unique_round_state_dirs(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = _round_state_dir_key(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _rerolled_round_sort_key(payload: dict[str, Any]) -> tuple[str, str]:
    return (
        str(
            payload.get("sampled_at")
            or payload.get("started_at")
            or payload.get("completed_at")
            or ""
        ),
        str(payload.get("round_id") or ""),
    )


def latest_rerolled_round_payload(
    *,
    round_id: str,
    payload: dict[str, Any],
    search_dirs: list[Path],
) -> tuple[str, dict[str, Any], Path]:
    current_id = round_id
    current_payload = dict(payload)
    current_state_dir = (
        unique_round_state_dirs(search_dirs)[0] if search_dirs else Path(".")
    )
    seen = {current_id}
    while True:
        replacements: list[tuple[Path, dict[str, Any]]] = []
        for candidate_dir in unique_round_state_dirs([current_state_dir, *search_dirs]):
            replacements.extend(
                (candidate_dir, dict(candidate_payload))
                for candidate_payload in iter_round_payloads(candidate_dir)
                if str(candidate_payload.get("rerolled_from_round_id") or "").strip()
                == current_id
            )
        if not replacements:
            return current_id, current_payload, current_state_dir
        current_state_dir, current_payload = max(
            replacements, key=lambda item: _rerolled_round_sort_key(item[1])
        )
        recorded_state_dir = str(current_payload.get("round_state_dir") or "").strip()
        if recorded_state_dir:
            current_state_dir = Path(recorded_state_dir)
        next_id = str(current_payload.get("round_id") or "").strip()
        if not next_id or next_id in seen:
            return next_id or current_id, current_payload, current_state_dir
        current_id = next_id
        seen.add(current_id)


def ungraded_round_exposure_records(
    state_dir: Path, *, cleanup: bool = True
) -> list[dict[str, Any]]:
    """Return synthetic exposure records for ungraded rounds.

    Stale ungraded rounds are always excluded. When cleanup is true they are
    also dismissed on disk first; when false the call is read-only. Always
    excluding stale rounds keeps the read-only preview consistent with a real
    run even if dismissal fails (for example on an OSError).
    """
    if cleanup:
        cleanup_stale_ungraded_rounds(state_dir)
    records: list[dict[str, Any]] = []
    payloads = iter_round_payloads(state_dir)
    replaced_round_ids = {
        str(payload.get("rerolled_from_round_id") or "") for payload in payloads
    }
    for payload in payloads:
        if str(payload.get("round_id") or "") in replaced_round_ids:
            continue
        if str(payload.get("status") or "") not in {"sampled", "running", "completed"}:
            continue
        if _round_has_recorded_grade(payload):
            continue
        if round_is_stale_ungraded(payload):
            continue
        task_class = str(payload.get("task_class") or "")
        runs = [
            {"variant_id": str(run.get("variant_id") or "")}
            for run in list(payload.get("runs") or [])
            if str(run.get("variant_id") or "")
        ]
        if task_class and runs:
            record = {"task_class": task_class, "runs": runs}
            if payload.get("rating_pool_id"):
                record["rating_pool_id"] = payload["rating_pool_id"]
            records.append(record)
    return records


def round_is_stale_ungraded(
    payload: dict[str, Any],
    *,
    now: datetime | None = None,
    stale_seconds: int = STALE_REVIEW_STATE_TTL_SECONDS,
) -> bool:
    if str(payload.get("status") or "") not in {
        "sampled",
        "running",
        "completed",
        "cleanup_pending",
    }:
        return False
    if _round_has_recorded_grade(payload):
        return False
    if round_has_live_reviewer_process(payload):
        return False
    reference_time = _payload_reference_time(payload)
    if reference_time is None:
        return False
    return ((now or utc_now()) - reference_time).total_seconds() >= stale_seconds


def cleanup_stale_ungraded_rounds(
    state_dir: Path,
    *,
    stale_seconds: int = STALE_REVIEW_STATE_TTL_SECONDS,
) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    now = utc_now()
    dismissed_at = now.isoformat().replace("+00:00", "Z")
    reason = f"auto_stale_ungraded_round_{max(1, stale_seconds // 3600)}h"
    for payload in iter_round_payloads(state_dir):
        if not round_is_stale_ungraded(payload, now=now, stale_seconds=stale_seconds):
            continue
        previous_status = str(payload.get("status") or "unknown")
        payload["status"] = "dismissed"
        payload["dismissed_at"] = dismissed_at
        payload["dismissed_reason"] = reason
        payload["dismissed_previous_status"] = str(
            payload.get("dismissed_previous_status") or previous_status
        )
        payload.pop("_round_file_path", None)
        try:
            write_round(state_dir, payload)
        except OSError:
            continue
        cleaned.append(
            {
                "round_id": str(payload.get("round_id") or ""),
                "previous_status": previous_status,
                "reason": reason,
            }
        )
    return cleaned


def find_pending_rounds_for_caller(
    *, state_dir: Path, caller_id: str | None, review_cwd: Path | str | None
) -> list[dict[str, Any]]:
    cleanup_stale_ungraded_rounds(state_dir)
    normalized_review_cwd = normalize_review_cwd_value(review_cwd)
    caller = (caller_id or "").strip()
    if not caller or not normalized_review_cwd:
        return []
    pending: list[dict[str, Any]] = []
    for payload in iter_round_payloads(state_dir):
        if str(payload.get("caller_id") or "").strip() != caller:
            continue
        if normalize_record_review_cwd_value(payload) != normalized_review_cwd:
            continue
        if not round_needs_caller_grade(payload):
            continue
        pending.append(payload)
    pending.sort(
        key=lambda item: str(
            item.get("review_completed_at")
            or item.get("sampled_at")
            or item.get("round_id")
            or ""
        )
    )
    return pending


def find_blocking_rounds_for_caller(
    *, state_dir: Path, caller_id: str | None, review_cwd: Path | str | None
) -> list[dict[str, Any]]:
    cleanup_stale_ungraded_rounds(state_dir)
    normalized_review_cwd = normalize_review_cwd_value(review_cwd)
    caller = (caller_id or "").strip()
    if not caller or not normalized_review_cwd:
        return []
    blocking: list[dict[str, Any]] = []
    for payload in iter_round_payloads(state_dir):
        if str(payload.get("caller_id") or "").strip() != caller:
            continue
        if normalize_record_review_cwd_value(payload) != normalized_review_cwd:
            continue
        status = str(payload.get("status") or "")
        if (
            round_needs_caller_grade(payload)
            or status in {"sampled", "running"}
            or round_has_live_reviewer_process(payload)
        ):
            blocking.append(payload)
    blocking.sort(
        key=lambda item: str(
            item.get("review_completed_at")
            or item.get("sampled_at")
            or item.get("round_id")
            or ""
        )
    )
    return blocking


def repo_token(review_cwd: Path | None) -> str | None:
    if review_cwd is None:
        return None
    normalized = str(review_cwd.resolve())
    name = re.sub(r"[^a-z0-9]+", "-", review_cwd.name.lower()).strip("-") or "repo"
    digest = blake2s(normalized.encode("utf-8"), digest_size=3).hexdigest()
    return f"{name}-{digest}"


def make_round_id(task_class: str, *, review_cwd: Path | None = None) -> str:
    token = repo_token(review_cwd)
    prefix = f"{task_class}-{token}" if token else task_class
    return f"{prefix}-{utc_now().strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"


def public_reviewer_label(slot: str) -> str:
    if slot in PUBLIC_REVIEWER_LABELS:
        return slot
    if slot.startswith("reviewer_"):
        try:
            idx = int(slot.split("_", 1)[1]) - 1
        except ValueError:
            return slot
        if 0 <= idx < len(PUBLIC_REVIEWER_LABELS):
            return PUBLIC_REVIEWER_LABELS[idx]
    return slot


def grade_rank_placeholders(payload: dict[str, Any]) -> list[str]:
    count = max(2, len(list(payload.get("runs") or [])))
    return [f"RANK_{index}[,TIED]" for index in range(1, count + 1)]


def _elapsed_seconds_for_runs(runs: list[dict[str, Any]]) -> int:
    elapsed = 0
    for run in runs:
        started_at = run.get("started_at")
        if not started_at:
            continue
        value = int(
            (
                utc_now()
                - datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
            ).total_seconds()
        )
        elapsed = max(elapsed, value)
    return elapsed


def _running_status_line(runs: list[dict[str, Any]]) -> str:
    labels = [public_reviewer_label(str(run["slot"])) for run in runs]
    elapsed = _elapsed_seconds_for_runs(runs)
    return f"Running: {elapsed}s {', '.join(labels)}"


def _heartbeat_status_line(runs: list[dict[str, Any]]) -> str:
    labels = [public_reviewer_label(str(run["slot"])) for run in runs]
    minutes = max(1, _elapsed_seconds_for_runs(runs) // 60)
    return f"OK {minutes}m: {','.join(labels)}"


def _progress_status_line(runs: list[dict[str, Any]]) -> str:
    if progress_output_isatty():
        return _running_status_line(runs)
    return _heartbeat_status_line(runs)


def _run_started_at_datetime(run: dict[str, Any]) -> datetime | None:
    started_at = run.get("started_at")
    if not started_at:
        return None
    try:
        return datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
    except ValueError:
        return None


def _path_stat(path_text: str) -> os.stat_result | None:
    if not path_text.strip():
        return None
    path = Path(path_text)
    try:
        return path.stat()
    except OSError:
        return None


def _artifact_activity_epoch(run: dict[str, Any]) -> float | None:
    times: list[float] = []
    for key in ("stdout_path", "stderr_path", "final_message_path"):
        stat = _path_stat(str(run.get(key) or ""))
        if stat is not None:
            times.append(float(stat.st_mtime))
    rollout_activity = run.get("_rollout_activity_epoch")
    if isinstance(rollout_activity, (int, float)):
        times.append(float(rollout_activity))
    return max(times) if times else None


def _stdout_has_content(run: dict[str, Any]) -> bool:
    for key in ("stdout_path", "final_message_path"):
        stat = _path_stat(str(run.get(key) or ""))
        if stat is not None and int(stat.st_size) > 0:
            return True
    return False


def _stderr_text_for_run(run: dict[str, Any]) -> str:
    path_text = str(run.get("stderr_path") or "").strip()
    if not path_text:
        return ""
    path = Path(path_text)
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _transport_stall_signal(stderr_text: str) -> str | None:
    lowered = stderr_text.lower()
    if "falling back to http" in lowered:
        return "http_fallback_no_output"
    if "reconnecting... 5/5" in lowered or "retrying sampling request (5/5" in lowered:
        return "reconnect_exhausted_no_output"
    return None


def _transport_stalled(
    run: dict[str, Any], *, now_epoch: float | None = None
) -> str | None:
    if _stdout_has_content(run):
        return None
    stderr_text = _stderr_text_for_run(run)
    signal = _transport_stall_signal(stderr_text)
    if signal is None:
        return None
    last_activity = _artifact_activity_epoch(run)
    if last_activity is None:
        return None
    now = time.time() if now_epoch is None else now_epoch
    if now - last_activity < TRANSPORT_STALL_GRACE_SECONDS:
        return None
    return signal


def _transport_hung_after_output(
    run: dict[str, Any], *, now_epoch: float | None = None
) -> str | None:
    if not _stdout_has_content(run):
        return None
    last_activity = _artifact_activity_epoch(run)
    if last_activity is None:
        return None
    now = time.time() if now_epoch is None else now_epoch
    if now - last_activity < TRANSPORT_STALL_GRACE_SECONDS:
        return None
    return "output_captured_process_still_running"


_terminate_process_tree = terminate_process_tree


def _transport_event_lines(
    stderr_text: str, *, start_offset: int
) -> tuple[list[str], int]:
    offset = max(0, int(start_offset or 0))
    chunk = stderr_text[offset:]
    lines = [
        line.strip()
        for line in chunk.splitlines()
        if line.strip()
        and any(pattern in line for pattern in TRANSPORT_RECONNECT_PATTERNS)
    ]
    return lines, len(stderr_text)


def _print_transport_events(active_runs: list[dict[str, Any]]) -> bool:
    printed = False
    for run in active_runs:
        slot = str(run.get("slot") or "")
        if not slot:
            continue
        stderr_text = _stderr_text_for_run(run)
        start_offset = int(run.get("stderr_progress_offset", 0) or 0)
        lines, next_offset = _transport_event_lines(
            stderr_text, start_offset=start_offset
        )
        run["stderr_progress_offset"] = next_offset
        label = public_reviewer_label(slot)
        for line in lines:
            print(
                f"[review-suite] {label} transport: {line}", file=sys.stderr, flush=True
            )
            printed = True
    return printed


def _live_review_thread(
    *,
    run: dict[str, Any],
    variant: dict[str, Any],
    sqlite_path: Path,
    review_cwd: Path,
) -> dict[str, Any] | None:
    stderr_path_text = str(run.get("stderr_path") or "").strip()
    stderr_path = Path(stderr_path_text) if stderr_path_text else None
    stderr_text = (
        stderr_path.read_text(encoding="utf-8", errors="replace")
        if stderr_path is not None and stderr_path.is_file()
        else ""
    )
    session_id = extract_session_id(stderr_text)
    if not session_id:
        return None
    candidate = find_thread_by_id(sqlite_path=sqlite_path, thread_id=session_id)
    if candidate and str(candidate.get("source") or "") == REVIEW_SUBAGENT_SOURCE:
        return candidate
    return find_review_child_thread(
        sqlite_path=sqlite_path, parent_thread_id=session_id
    )


def _reviewer_deadline_reason(
    *,
    run: dict[str, Any],
    variant: dict[str, Any],
    sqlite_path: Path,
    review_cwd: Path,
    now: datetime | None = None,
) -> str | None:
    current = utc_now() if now is None else now
    started = _run_started_at_datetime(run)
    if (
        started
        and (current - started).total_seconds() >= REVIEW_ABSOLUTE_DEADLINE_SECONDS
    ):
        return "absolute_deadline"
    thread = _live_review_thread(
        run=run,
        variant=variant,
        sqlite_path=sqlite_path,
        review_cwd=review_cwd,
    )
    rollout_path = Path(str((thread or {}).get("rollout_path") or ""))
    activity = rollout_activity_summary(rollout_path) if rollout_path.is_file() else {}
    rollout_activity = activity.get("last_meaningful_at")
    if rollout_activity is not None:
        run["_rollout_activity_epoch"] = rollout_activity.timestamp()
    last_activity = rollout_activity or started
    if (
        last_activity
        and (current - last_activity).total_seconds()
        >= REVIEW_INACTIVITY_DEADLINE_SECONDS
    ):
        return "inactivity_deadline"
    return None


def _print_stall_warnings(
    *,
    active_runs: list[dict[str, Any]],
    indexed: dict[str, dict[str, Any]],
    sqlite_path: Path,
    review_cwd: Path,
    warned_slots: set[str],
) -> None:
    now = utc_now()
    for run in active_runs:
        slot = str(run.get("slot") or "")
        if not slot or slot in warned_slots:
            continue
        variant = dict(
            run.get("variant") or indexed.get(str(run.get("variant_id") or ""), {})
        )
        if not variant:
            continue
        thread = _live_review_thread(
            run=run,
            variant=variant,
            sqlite_path=sqlite_path,
            review_cwd=review_cwd,
        )
        rollout_path_text = str((thread or {}).get("rollout_path") or "")
        if not rollout_path_text:
            continue
        rollout_path = Path(rollout_path_text)
        if not rollout_path.is_file():
            continue
        activity = rollout_activity_summary(rollout_path)
        last_meaningful_at = activity.get(
            "last_meaningful_at"
        ) or _run_started_at_datetime(run)
        if last_meaningful_at is None:
            continue
        idle_seconds = int((now - last_meaningful_at).total_seconds())
        if idle_seconds < REVIEW_STALL_WARNING_SECONDS:
            continue
        label = public_reviewer_label(slot)
        print(
            f"[review-suite] possible stall: {label} idle {idle_seconds // 60}m; wrapper will keep waiting.",
            file=sys.stderr,
            flush=True,
        )
        warned_slots.add(slot)


def reviewer_completion_status(run: dict[str, Any]) -> str:
    block = str(run.get("grade_block_reason") or "").strip()
    if block:
        return block
    status = str(run.get("review_status") or "").strip()
    return status or "unknown"


def output_isatty() -> bool:
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def progress_output_isatty() -> bool:
    return bool(getattr(sys.stderr, "isatty", lambda: False)())


def reviewer_output_heading(run: dict[str, Any]) -> str:
    label = public_reviewer_label(str(run.get("slot") or "reviewer"))
    status = str(run.get("review_status") or "unknown")
    if status == "completed":
        return f"{label}:"
    return f"{label} [{status}]:"


def _print_live_completed_run(run: dict[str, Any]) -> None:
    write_text(
        f"{reviewer_output_heading(run)} {reviewer_completion_status(run)}",
        stream=sys.stderr,
    )


def final_display_body(run: dict[str, Any]) -> str:
    reviewer_output = str(run.get("reviewer_output") or "").strip()
    if reviewer_output:
        return reviewer_output
    status_summary = str(run.get("status_summary") or "").strip()
    if status_summary:
        return status_summary
    status = str(run.get("review_status") or "").strip()
    return f"({status or 'no output'})"


def print_reviewer_output_section(runs: list[dict[str, Any]]) -> bool:
    if not runs:
        return False
    write_text("Output:")
    for index, raw_run in enumerate(runs):
        run = _finalized_run_summary(raw_run)
        if index:
            write_text("")
        write_text(reviewer_output_heading(run))
        write_text(final_display_body(run))
    return True


def review_label(task_class: str) -> str:
    public = public_task_name(task_class)
    if public != task_class:
        return public
    if task_class == "phase_review":
        return "phase"
    if task_class == "pr_review":
        return "pr"
    return str(task_class)


def public_task_name(task_class: str) -> str:
    if task_class == "phase_review":
        return "review_t1"
    if task_class == "pr_review":
        return "review_t3"
    return str(task_class)


def _repo_name_from_remote_url(remote_url: str) -> str | None:
    trimmed = str(remote_url or "").strip().rstrip("\\/")
    if not trimmed:
        return None
    name = re.split(r"[:\\/]+", trimmed)[-1]
    if name.endswith(".git"):
        name = name[:-4]
    return name or None


def _is_wsl_unc_path(path_text: str) -> bool:
    normalized = str(path_text or "").strip().replace("\\", "/").lower()
    return normalized.startswith("//wsl.localhost/") or normalized.startswith("//wsl$/")


def repo_name_from_review_cwd_value(review_cwd: str) -> str:
    trimmed = str(review_cwd or "").strip().rstrip("\\/")
    if not trimmed:
        return "-"
    if not _is_wsl_unc_path(trimmed):
        try:
            path = Path(trimmed)
            if path.exists():
                proc = subprocess.run(
                    ["git", "-C", str(path), "config", "--get", "remote.origin.url"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                    timeout=5,
                )
                if proc.returncode == 0:
                    name = _repo_name_from_remote_url(proc.stdout)
                    if name:
                        return name
        except OSError:
            pass
        except subprocess.SubprocessError:
            pass
    return re.split(r"[\\/]+", trimmed)[-1] or "-"


def repo_name_from_round_payload(
    payload: dict[str, Any], cwd_cache: dict[str, str] | None = None
) -> str:
    explicit = str(payload.get("repo_name") or "").strip()
    if explicit:
        return explicit
    review_cwd = str(
        payload.get("review_cwd") or payload.get("review_cwd_normalized") or ""
    ).strip()
    cache_key = review_cwd.rstrip("\\/")
    if cwd_cache is not None and cache_key:
        cached = cwd_cache.get(cache_key)
        if cached is not None:
            return cached
    name = repo_name_from_review_cwd_value(review_cwd)
    if cwd_cache is not None and cache_key:
        cwd_cache[cache_key] = name
    return name


def round_repo_name(
    state_dir: Path,
    round_id: str,
    cache: dict[str, str],
    cwd_cache: dict[str, str] | None = None,
) -> str:
    cached = cache.get(round_id)
    if cached is not None:
        return cached
    name = "-"
    try:
        payload = read_json(round_path(state_dir, round_id))
        name = repo_name_from_round_payload(payload, cwd_cache)
    except Exception:
        pass
    cache[round_id] = name
    return name


def record_repo_name(
    state_dir: Path,
    record: dict[str, Any],
    cache: dict[str, str],
    cwd_cache: dict[str, str] | None = None,
) -> str:
    explicit = str(record.get("repo_name") or "").strip()
    if explicit:
        return explicit
    return round_repo_name(
        state_dir, str(record.get("round_id") or ""), cache, cwd_cache
    )


def enrich_record_repo_names(
    state_dir: Path, records: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], bool]:
    cache: dict[str, str] = {}
    cwd_cache: dict[str, str] = {}
    changed = False
    enriched: list[dict[str, Any]] = []
    for record in records:
        if str(record.get("repo_name") or "").strip():
            enriched.append(record)
            continue
        repo_name = record_repo_name(state_dir, record, cache, cwd_cache)
        if repo_name == "-":
            enriched.append(record)
            continue
        next_record = deepcopy(record)
        next_record["repo_name"] = repo_name
        enriched.append(next_record)
        changed = True
    return enriched, changed


def public_round_payload(
    payload: dict[str, Any], *, task_name: str | None = None
) -> dict[str, Any]:
    return {
        "round_id": payload["round_id"],
        "task": task_name or public_task_name(str(payload["task_class"])),
        "status": payload.get("status"),
        "sampled_at": payload.get("sampled_at"),
        "reviewers": [
            public_reviewer_label(run["slot"]) for run in payload.get("runs", [])
        ],
    }


def usable_output_slots(payload: dict[str, Any]) -> set[str]:
    visible: set[str] = set()
    for raw_run in payload.get("runs", []):
        if not isinstance(raw_run, dict):
            continue
        slot = str(raw_run.get("slot") or "")
        if not slot:
            continue
        classification = _classification_for_run(raw_run)
        if classification["grade_blocked"]:
            continue
        if classification["review_status"] != "completed":
            continue
        if not str(raw_run.get("reviewer_output") or "").strip():
            continue
        visible.add(slot)
    return visible


def payload_has_blocked_runs(payload: dict[str, Any]) -> bool:
    for raw_run in payload.get("runs", []):
        if not isinstance(raw_run, dict):
            continue
        if _classification_for_run(raw_run)["grade_blocked"]:
            return True
    return False


def _finalized_run_summary(run: dict[str, Any]) -> dict[str, Any]:
    classification = _classification_for_run(run)
    normalized = deepcopy(run)
    normalized["review_status"] = classification["review_status"]
    normalized["status_summary"] = classification["status_summary"]
    normalized["grade_blocked"] = classification["grade_blocked"]
    normalized["grade_block_reason"] = classification["grade_block_reason"]
    return normalized


def _run_is_finalized(run: dict[str, Any]) -> bool:
    if run.get("review_status"):
        return True
    if run.get("returncode") is not None:
        return True
    if run.get("reviewer_output"):
        return True
    return False


def public_round_result(
    payload: dict[str, Any], *, output_slots: set[str] | None = None
) -> dict[str, Any]:
    run_summaries = []
    for run in payload.get("runs", []):
        classification = _classification_for_run(run)
        row = {
            "slot": public_reviewer_label(run["slot"]),
            "status": classification["review_status"],
            "summary": classification["status_summary"],
            "ref": run.get("reviewer_output_ref"),
            "blocked": classification["grade_blocked"],
            "block": classification["grade_block_reason"],
        }
        run_summaries.append(row)
    result = {
        "round_id": payload["round_id"],
        "task": public_task_name(str(payload.get("task_class") or "")),
        "status": payload["status"],
        "blocked": any(bool(run.get("blocked")) for run in run_summaries),
        "runs": run_summaries,
    }
    if payload.get("cooldown_updates"):
        result["cooldowns"] = [
            {
                "variant": str(update.get("variant_id") or ""),
                "until": format_cooldown_until_for_display(
                    update.get("cooldown_until")
                ),
                "reason": update.get("reason"),
                "failures": update.get("failure_count"),
            }
            for update in list(payload.get("cooldown_updates") or [])
        ]
    return result


def _report_settings_snapshot(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "elo_k_factor": float(settings.get("elo_k_factor", 24.0)),
        "promotion_min_samples": int(settings.get("promotion_min_samples", 20)),
        "promotion_min_elo": float(settings.get("promotion_min_elo", 1550.0)),
        "promotion_champion_group_window": float(
            settings.get(
                "promotion_champion_group_window",
                settings.get("promotion_min_elo_lead", 25.0),
            )
        ),
    }


def _probation_variant_ids(
    *,
    leaderboard: list[dict[str, Any]],
    champion_ids: set[str],
    settings: dict[str, Any],
) -> list[str]:
    min_samples = probation_min_samples(settings)
    max_elo = probation_max_elo(settings)
    probation: list[str] = []
    for row in leaderboard:
        variant_id = str(row["variant_id"])
        if variant_id in champion_ids:
            continue
        if int(row.get("sample_count", 0) or 0) < min_samples:
            continue
        if float(row.get("elo", 0.0) or 0.0) >= max_elo:
            continue
        if int(row.get("loss_count", 0) or 0) <= int(row.get("win_count", 0) or 0):
            continue
        probation.append(variant_id)
    return probation


def _configured_variant_group(
    *,
    roster: dict[str, Any],
    operational_state: dict[str, Any],
    records: list[dict[str, Any]],
    task_class: str,
    rating_pool_id: str,
    variant_groups: list[list[str]],
    excluded_variant_ids: set[str],
) -> tuple[list[dict[str, Any]], int]:
    if not rating_pool_id or not variant_groups:
        raise ValueError("configured arena selection requires a rating pool and groups")
    schedule_index = sum(
        1
        for record in records
        if record.get("task_class") == task_class
        and record.get("rating_pool_id") == rating_pool_id
    ) % len(variant_groups)
    variant_ids = [str(value).strip() for value in variant_groups[schedule_index]]
    if len(variant_ids) < 2 or any(not value for value in variant_ids):
        raise ValueError(
            "configured arena groups must contain at least two variant ids"
        )
    if len(variant_ids) != len(set(variant_ids)):
        raise ValueError("configured arena groups cannot repeat a variant")
    if len(variant_ids) > len(PUBLIC_REVIEWER_LABELS):
        raise ValueError(
            f"configured arena group exceeds {len(PUBLIC_REVIEWER_LABELS)} reviewer slots"
        )
    indexed = variant_index(roster)
    unknown = [variant_id for variant_id in variant_ids if variant_id not in indexed]
    if unknown:
        raise ValueError(f"unknown configured arena variants: {', '.join(unknown)}")
    unavailable = [
        variant_id
        for variant_id in variant_ids
        if not variant_is_arena_eligible(indexed[variant_id], task_class)
        or variant_id in excluded_variant_ids
    ]
    cooling = _active_cooldowns(operational_state, task_class)
    unavailable.extend(
        variant_id
        for variant_id in variant_ids
        if variant_id in cooling and variant_id not in unavailable
    )
    if unavailable:
        raise ValueError(
            f"configured arena variants are unavailable for {task_class}: "
            + ", ".join(unavailable)
        )
    return [indexed[variant_id] for variant_id in variant_ids], schedule_index


def _balanced_configured_variant_group(
    *,
    roster: dict[str, Any],
    operational_state: dict[str, Any],
    records: list[dict[str, Any]],
    task_class: str,
    rating_pool_id: str,
    variant_ids: list[str],
    group_size: int,
    excluded_variant_ids: set[str],
) -> list[dict[str, Any]]:
    settings = roster.get("settings", {})
    if group_size > len(PUBLIC_REVIEWER_LABELS):
        raise ValueError(
            f"configured arena group exceeds {len(PUBLIC_REVIEWER_LABELS)} reviewer slots"
        )
    indexed = variant_index(roster)
    if len(variant_ids) != len(set(variant_ids)):
        raise ValueError("configured arena candidate ids cannot repeat a variant")
    unknown = [variant_id for variant_id in variant_ids if variant_id not in indexed]
    if unknown:
        raise ValueError(f"unknown configured arena variants: {', '.join(unknown)}")
    ineligible = [
        variant_id
        for variant_id in variant_ids
        if not variant_is_arena_eligible(indexed[variant_id], task_class)
    ]
    if ineligible:
        raise ValueError(
            f"configured arena variants are unavailable for {task_class}: "
            + ", ".join(ineligible)
        )
    cooling = _active_cooldowns(operational_state, task_class)
    available_ids = [
        variant_id
        for variant_id in variant_ids
        if variant_id not in excluded_variant_ids and variant_id not in cooling
    ]
    if len(available_ids) < group_size:
        raise ValueError(
            f"configured arena needs {group_size} available variants for {task_class}"
        )

    pool_records = [
        record
        for record in records
        if record.get("task_class") == task_class
        and record.get("rating_pool_id") == rating_pool_id
    ]
    sample_counts = summarize_counts(pool_records, task_class)
    pair_counts: dict[tuple[str, str], int] = {}
    for record in pool_records:
        participants = [
            str(run.get("variant_id") or "")
            for run in record.get("runs", [])
            if str(run.get("variant_id") or "") in available_ids
        ]
        for left, right in combinations(sorted(set(participants)), 2):
            pair = (left, right)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    order = {variant_id: index for index, variant_id in enumerate(available_ids)}

    def meetings(left: str, right: str) -> int:
        return pair_counts.get(tuple(sorted((left, right))), 0)

    available_variants = [indexed[variant_id] for variant_id in available_ids]
    under_sampled_ids = [
        variant["id"]
        for variant in available_variants
        if is_under_sampled(variant, sample_counts, settings, available_variants)
    ]
    under_sampled_id_set = set(under_sampled_ids)
    established_ids = [
        variant_id
        for variant_id in available_ids
        if variant_id not in under_sampled_id_set
    ]
    selected: list[str] = []

    def select_from(candidates: list[str], count: int) -> None:
        for _ in range(count):
            remaining = [item for item in candidates if item not in selected]
            if not remaining:
                return
            selected.append(
                min(
                    remaining,
                    key=lambda item: (
                        sum(meetings(item, chosen) for chosen in selected),
                        sample_counts.get(item, 0),
                        order[item],
                    ),
                )
            )

    established_slots = min(len(established_ids), group_size // 2)
    select_from(
        under_sampled_ids,
        min(len(under_sampled_ids), group_size - established_slots),
    )
    select_from(established_ids, min(len(established_ids), group_size - len(selected)))
    select_from(available_ids, group_size - len(selected))
    return [indexed[variant_id] for variant_id in selected]


def select_pair(
    *,
    roster: dict[str, Any],
    operational_state: dict[str, Any],
    records: list[dict[str, Any]],
    task_class: str,
    review_cwd: Path | None,
    seed: int | None,
    caller_id: str | None = None,
    caller_id_source: str | None = None,
    excluded_variant_ids: set[str] | None = None,
    rating_pool_id: str | None = None,
    variant_groups: list[list[str]] | None = None,
    variant_ids: list[str] | None = None,
) -> dict[str, Any]:
    settings = roster.get("settings", {})
    excluded_variant_ids = excluded_variant_ids or set()
    schedule_index: int | None = None
    if variant_groups is not None:
        pool_id = str(rating_pool_id or "").strip()
        pool_record_count = sum(
            1
            for record in records
            if record.get("task_class") == task_class
            and record.get("rating_pool_id") == pool_id
        )
        if variant_ids and pool_record_count >= len(variant_groups):
            selected = _balanced_configured_variant_group(
                roster=roster,
                operational_state=operational_state,
                records=records,
                task_class=task_class,
                rating_pool_id=pool_id,
                variant_ids=variant_ids,
                group_size=len(variant_groups[0]),
                excluded_variant_ids=excluded_variant_ids,
            )
            selection_pairing = "configured_balanced"
        else:
            selected, schedule_index = _configured_variant_group(
                roster=roster,
                operational_state=operational_state,
                records=records,
                task_class=task_class,
                rating_pool_id=pool_id,
                variant_groups=variant_groups,
                excluded_variant_ids=excluded_variant_ids,
            )
            selection_pairing = "configured_schedule"
        selection_mode = "configured"
    else:
        all_variants = [
            variant
            for variant in eligible_variants(roster, task_class)
            if variant["id"] not in excluded_variant_ids
        ]
        cooling = _active_cooldowns(operational_state, task_class)
        variants = [variant for variant in all_variants if variant["id"] not in cooling]
        if len(variants) < 2:
            if len(all_variants) >= 2 and cooling:
                raise ValueError(
                    _cooldown_unavailability_message(
                        task_class=task_class, cooling=cooling, needed=2
                    )
                )
            raise ValueError(f"need at least two active variants for {task_class}")
        counts = summarize_counts(records, task_class)
        state = operational_state["task_classes"][task_class]
        available_ids = {variant["id"] for variant in variants}
        probation_ids: list[str] = [
            variant_id
            for variant_id in list(state.get("probation_variant_ids") or [])
            if variant_id in available_ids
        ]
        selection_mode = configured_pair_selection_mode(roster, task_class)
        if selection_mode == "legacy":
            selected, selection_pairing = _select_scramble_pair(
                variants=variants,
                reference_variants=all_variants,
                probation_ids=probation_ids,
                counts=counts,
                settings=settings,
                seed=seed,
            )
        elif selection_mode == "true_scramble":
            selected, selection_pairing = _select_true_scramble_pair(
                variants=variants, seed=seed
            )
        else:
            leaderboard = aggregate_records(
                roster=roster,
                records=records,
                operational_state=operational_state,
            )["task_classes"][task_class]["leaderboard"]
            selected, selection_pairing = _select_slight_bias_pair(
                variants=variants,
                leaderboard=leaderboard,
                settings=settings,
                seed=seed,
            )

    payload = {
        "round_id": make_round_id(task_class, review_cwd=review_cwd),
        "task_class": task_class,
        "selection_mode": selection_mode,
        "selection_pairing": selection_pairing,
        "selection_champion_variant_ids": [],
        "selection_anchor_kind": None,
        "selection_fallback_reason": None,
        "sampled_at": utc_now_iso(),
        "caller_id": caller_id,
        "caller_id_source": caller_id_source,
        "review_cwd_normalized": normalize_review_cwd_value(review_cwd),
        "excluded_variant_ids": sorted(excluded_variant_ids),
        "status": "sampled",
        "runs": [
            {
                "slot": PUBLIC_REVIEWER_LABELS[idx],
                "variant_id": variant["id"],
                "model": variant["model"],
                "reasoning_effort": variant["reasoning_effort"],
            }
            for idx, variant in enumerate(selected)
        ],
    }
    if variant_groups is not None:
        payload.update(
            {
                "rating_pool_id": str(rating_pool_id).strip(),
                "rating_pool_variant_ids": list(
                    variant_ids
                    or dict.fromkeys(
                        variant_id for group in variant_groups for variant_id in group
                    )
                ),
                "schedule_length": len(variant_groups),
            }
        )
        if schedule_index is not None:
            payload["schedule_index"] = schedule_index
    return payload


def _select_scramble_pair(
    *,
    variants: list[dict[str, Any]],
    reference_variants: list[dict[str, Any]],
    probation_ids: list[str],
    counts: dict[str, int],
    settings: dict[str, Any],
    seed: int | None,
) -> tuple[list[dict[str, Any]], str]:
    underuse_target = relative_underuse_target(reference_variants, counts, settings)
    probation_id_set = set(probation_ids)
    under_sampled = [
        variant
        for variant in variants
        if variant["id"] not in probation_id_set
        and is_under_sampled(variant, counts, settings, reference_variants)
    ]
    if under_sampled:
        lowest_exposure = min(counts.get(variant["id"], 0) for variant in under_sampled)
        under_sampled = [
            variant
            for variant in under_sampled
            if counts.get(variant["id"], 0) == lowest_exposure
        ]
    if not under_sampled:
        preferred_variants = [
            variant for variant in variants if variant["id"] not in probation_id_set
        ]
        probation_variants = [
            variant for variant in variants if variant["id"] in probation_id_set
        ]
        if preferred_variants:
            anchor = pick_weighted_without_replacement(
                preferred_variants,
                counts,
                settings,
                1,
                _seed_with_offset(seed, 101),
                relative_target=underuse_target,
            )[0]
            remaining_preferred = [
                variant
                for variant in preferred_variants
                if variant["id"] != anchor["id"]
            ]
            probation_probe_ratio = float(
                settings.get("scramble_probation_probe_ratio", 0.15)
            )
            want_probation_probe = bool(probation_variants) and (
                not remaining_preferred
                or (
                    probation_probe_ratio > 0.0
                    and random.Random(_seed_with_offset(seed, 100)).random()
                    < probation_probe_ratio
                )
            )
            if want_probation_probe:
                probe = pick_weighted_without_replacement(
                    probation_variants,
                    counts,
                    settings,
                    1,
                    _seed_with_offset(seed, 102),
                    relative_target=underuse_target,
                )[0]
                return [anchor, probe], "scramble_weighted_vs_probation"
            if remaining_preferred:
                second = pick_weighted_without_replacement(
                    remaining_preferred,
                    counts,
                    settings,
                    1,
                    _seed_with_offset(seed, 102),
                    relative_target=underuse_target,
                )[0]
                return [anchor, second], "scramble_weighted"
        highest_sample_count = max(counts.get(variant["id"], 0) for variant in variants)
        anchor_pool = [
            variant
            for variant in variants
            if counts.get(variant["id"], 0) == highest_sample_count
        ]
        anchor = pick_weighted_without_replacement(
            anchor_pool,
            counts,
            settings,
            1,
            _seed_with_offset(seed, 103),
            relative_target=underuse_target,
        )[0]
        remaining_variants = [
            variant for variant in variants if variant["id"] != anchor["id"]
        ]
        second = pick_weighted_without_replacement(
            remaining_variants,
            counts,
            settings,
            1,
            _seed_with_offset(seed, 104),
            relative_target=underuse_target,
        )[0]
        return [anchor, second], "scramble_weighted_best_available"
    exploration = pick_weighted_without_replacement(
        under_sampled,
        counts,
        settings,
        1,
        _seed_with_offset(seed, 1),
        relative_target=underuse_target,
    )[0]
    rest_pool = [
        variant
        for variant in variants
        if variant["id"] != exploration["id"]
        and not is_under_sampled(variant, counts, settings, reference_variants)
        and variant["id"] not in probation_id_set
    ]
    pairing = "scramble_exploration_vs_rest"
    if not rest_pool:
        degraded_pool = [
            variant
            for variant in variants
            if variant["id"] != exploration["id"]
            and variant["id"] not in probation_id_set
        ]
        if not degraded_pool:
            degraded_pool = [
                variant for variant in variants if variant["id"] != exploration["id"]
            ]
        if not degraded_pool:
            raise ValueError(
                "scramble sampling requires at least two eligible variants"
            )
        highest_sample_count = max(
            counts.get(variant["id"], 0) for variant in degraded_pool
        )
        rest_pool = [
            variant
            for variant in degraded_pool
            if counts.get(variant["id"], 0) == highest_sample_count
        ]
        pairing = "scramble_exploration_vs_best_available"
    stability = pick_weighted_without_replacement(
        rest_pool,
        counts,
        settings,
        1,
        _seed_with_offset(seed, 2),
        relative_target=underuse_target,
    )[0]
    return [exploration, stability], pairing


def _select_true_scramble_pair(
    *,
    variants: list[dict[str, Any]],
    seed: int | None,
) -> tuple[list[dict[str, Any]], str]:
    return random.Random(seed).sample(variants, 2), "true_scramble_random"


def _select_slight_bias_pair(
    *,
    variants: list[dict[str, Any]],
    leaderboard: list[dict[str, Any]],
    settings: dict[str, Any],
    seed: int | None,
) -> tuple[list[dict[str, Any]], str]:
    ratings = {
        str(row.get("variant_id")): float(row.get("elo", 1500.0) or 1500.0)
        for row in leaderboard
    }
    active_ratings = [ratings.get(variant["id"], 1500.0) for variant in variants]
    center = statistics.mean(active_ratings) if active_ratings else 1500.0
    strength = max(float(settings.get("slight_bias_elo_weight", 0.15) or 0.0), 0.0)
    scale = max(float(settings.get("slight_bias_elo_scale", 400.0) or 400.0), 1.0)
    weights = {}
    for variant in variants:
        normalized_delta = max(
            min((ratings.get(variant["id"], 1500.0) - center) / scale, 1.0), -1.0
        )
        weights[variant["id"]] = 1.0 + (strength * normalized_delta)
    selected = pick_custom_weighted_without_replacement(variants, weights, 2, seed)
    return selected, "slight_bias_elo_weighted"


def _pick_weighted_pool(pool_weights: list[tuple[str, float]], seed: int | None) -> str:
    rng = random.Random(seed)
    total = sum(weight for _, weight in pool_weights)
    threshold = rng.random() * total
    running = 0.0
    for name, weight in pool_weights:
        running += weight
        if threshold <= running:
            return name
    return pool_weights[-1][0]


def _select_champion_pair(
    *,
    variants: list[dict[str, Any]],
    reference_variants: list[dict[str, Any]],
    indexed: dict[str, dict[str, Any]],
    champion_ids: list[str],
    probation_ids: list[str],
    counts: dict[str, int],
    settings: dict[str, Any],
    seed: int | None,
    anchor_override: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    champion_pool = [
        indexed[variant_id] for variant_id in champion_ids if variant_id in indexed
    ]
    probation_id_set = set(probation_ids)
    underuse_target = relative_underuse_target(reference_variants, counts, settings)
    if anchor_override is not None:
        anchor = anchor_override
    else:
        anchor = pick_weighted_without_replacement(
            champion_pool,
            counts,
            settings,
            1,
            _seed_with_offset(seed, 10),
            relative_target=underuse_target,
        )[0]
    new_challenger_pool = [
        variant
        for variant in variants
        if variant["id"] != anchor["id"]
        and variant["id"] not in champion_ids
        and variant["id"] not in probation_id_set
        and is_under_sampled(variant, counts, settings, reference_variants)
    ]
    contender_pool = [
        variant
        for variant in variants
        if variant["id"] != anchor["id"]
        and variant["id"] not in champion_ids
        and variant["id"] not in probation_id_set
        and variant["id"] not in {candidate["id"] for candidate in new_challenger_pool}
    ]
    probation_pool = [
        variant
        for variant in variants
        if variant["id"] != anchor["id"]
        and variant["id"] not in champion_ids
        and variant["id"] in probation_id_set
    ]
    champion_opponent_pool = (
        [variant for variant in champion_pool if variant["id"] != anchor["id"]]
        if anchor_override is None
        else []
    )
    pools: list[tuple[str, list[dict[str, Any]], float]] = []
    if champion_opponent_pool:
        pools.append(("champion_vs_champion", champion_opponent_pool, 0.50))
    if new_challenger_pool:
        pools.append(("champion_vs_new_challenger", new_challenger_pool, 0.20))
    if contender_pool:
        pools.append(("champion_vs_contender", contender_pool, 0.20))
    if probation_pool:
        pools.append(("champion_vs_probation", probation_pool, 0.10))
    if not pools:
        fallback = [variant for variant in variants if variant["id"] != anchor["id"]]
        opponent = pick_weighted_without_replacement(
            fallback,
            counts,
            settings,
            1,
            _seed_with_offset(seed, 11),
            relative_target=underuse_target,
        )[0]
        return [anchor, opponent], "champion_vs_fallback"
    chosen_name = _pick_weighted_pool(
        [(name, weight) for name, _, weight in pools], _seed_with_offset(seed, 11)
    )
    chosen_pool = next(pool for name, pool, _ in pools if name == chosen_name)
    opponent = pick_weighted_without_replacement(
        chosen_pool,
        counts,
        settings,
        1,
        _seed_with_offset(seed, 12),
        relative_target=underuse_target,
    )[0]
    return [anchor, opponent], chosen_name


def _reroll_candidate_variants(
    *,
    roster: dict[str, Any],
    operational_state: dict[str, Any],
    records: list[dict[str, Any]],
    round_payload: dict[str, Any],
    slot: str,
    excluded_variant_ids: set[str],
) -> list[dict[str, Any]]:
    settings = roster.get("settings", {})
    task_class = str(round_payload["task_class"])
    all_variants = [
        variant
        for variant in eligible_variants(roster, task_class)
        if variant["id"] not in excluded_variant_ids
    ]
    cooling = _active_cooldowns(operational_state, task_class)
    variants = [variant for variant in all_variants if variant["id"] not in cooling]
    counts = summarize_counts(records, task_class)
    probation_ids = set(
        str(item)
        for item in (
            operational_state["task_classes"][task_class].get("probation_variant_ids")
            or []
        )
    )
    if round_payload.get("selection_mode") == "champion":
        champion_ids = set(
            str(item)
            for item in (round_payload.get("selection_champion_variant_ids") or [])
        )
        pairing = str(round_payload.get("selection_pairing") or "")
        if slot == "alpha" or pairing == "champion_vs_champion":
            pool = [variant for variant in variants if variant["id"] in champion_ids]
            if pool:
                return pool
            acting_champions = _acting_champion_variants(
                roster=roster,
                operational_state=operational_state,
                records=records,
                task_class=task_class,
                available_variants=variants,
            )
            if acting_champions:
                return acting_champions
            raise ValueError(
                f"no compatible champion or acting-champion reroll variants remain for {task_class} ({pairing}, {slot})"
            )
        if pairing == "champion_vs_new_challenger":
            pool = [
                variant
                for variant in variants
                if variant["id"] not in champion_ids
                and variant["id"] not in probation_ids
                and is_under_sampled(variant, counts, settings, all_variants)
            ]
            if pool:
                return pool
            fallback = [
                variant
                for variant in variants
                if variant["id"] not in champion_ids
                and variant["id"] not in probation_ids
            ]
            if fallback:
                return fallback
        if pairing == "champion_vs_contender":
            pool = [
                variant
                for variant in variants
                if variant["id"] not in champion_ids
                and variant["id"] not in probation_ids
                and not is_under_sampled(variant, counts, settings, all_variants)
            ]
            if pool:
                return pool
            fallback = [
                variant
                for variant in variants
                if variant["id"] not in champion_ids
                and variant["id"] not in probation_ids
            ]
            if fallback:
                return fallback
        if pairing == "champion_vs_probation":
            if slot == "alpha":
                pool = [
                    variant for variant in variants if variant["id"] in champion_ids
                ]
            else:
                pool = [
                    variant
                    for variant in variants
                    if variant["id"] not in champion_ids
                    and variant["id"] in probation_ids
                ]
            if pool:
                return pool
            if slot != "alpha":
                fallback = [
                    variant for variant in variants if variant["id"] not in champion_ids
                ]
                if fallback:
                    return fallback
    if round_payload.get("selection_mode") in {"scramble", "legacy"}:
        pairing = str(round_payload.get("selection_pairing") or "")
        if pairing == "scramble_weighted_vs_probation":
            if slot == "alpha":
                pool = [
                    variant
                    for variant in variants
                    if variant["id"] not in probation_ids
                ]
                if pool:
                    return pool
            if slot == "bravo":
                pool = [
                    variant for variant in variants if variant["id"] in probation_ids
                ]
                if pool:
                    return pool
        if pairing == "scramble_weighted":
            pool = [
                variant for variant in variants if variant["id"] not in probation_ids
            ]
            if pool:
                return pool
        if pairing == "scramble_weighted_best_available" and slot == "alpha":
            if variants:
                highest_sample_count = max(
                    counts.get(variant["id"], 0) for variant in variants
                )
                pool = [
                    variant
                    for variant in variants
                    if counts.get(variant["id"], 0) == highest_sample_count
                ]
                if pool:
                    return pool
            raise ValueError(
                f"no compatible best-available reroll variants remain for {task_class} ({pairing}, {slot})"
            )
        if pairing == "scramble_exploration_vs_best_available" and slot == "bravo":
            preferred = [
                variant for variant in variants if variant["id"] not in probation_ids
            ]
            if preferred:
                highest_sample_count = max(
                    counts.get(variant["id"], 0) for variant in preferred
                )
                pool = [
                    variant
                    for variant in preferred
                    if counts.get(variant["id"], 0) == highest_sample_count
                ]
                if pool:
                    return pool
            if variants:
                highest_sample_count = max(
                    counts.get(variant["id"], 0) for variant in variants
                )
                pool = [
                    variant
                    for variant in variants
                    if counts.get(variant["id"], 0) == highest_sample_count
                ]
                if pool:
                    return pool
        if slot == "alpha":
            pool = [
                variant
                for variant in variants
                if variant["id"] not in probation_ids
                and is_under_sampled(variant, counts, settings, all_variants)
            ]
            if pool:
                return pool
            pool = [
                variant for variant in variants if variant["id"] not in probation_ids
            ]
            if pool:
                return pool
        if slot == "bravo":
            if pairing == "scramble_exploration_vs_best_available":
                preferred = [
                    variant
                    for variant in variants
                    if variant["id"] not in probation_ids
                ]
                if preferred:
                    highest_sample_count = max(
                        counts.get(variant["id"], 0) for variant in preferred
                    )
                    pool = [
                        variant
                        for variant in preferred
                        if counts.get(variant["id"], 0) == highest_sample_count
                    ]
                    if pool:
                        return pool
            pool = [
                variant
                for variant in variants
                if not is_under_sampled(variant, counts, settings, all_variants)
                and variant["id"] not in probation_ids
            ]
            if pool:
                return pool
            pool = [
                variant
                for variant in variants
                if not is_under_sampled(variant, counts, settings, all_variants)
            ]
            if pool:
                return pool
    return variants


def select_replacement_variant(
    *,
    roster: dict[str, Any],
    operational_state: dict[str, Any],
    records: list[dict[str, Any]],
    task_class: str,
    excluded_variant_ids: set[str],
    seed: int | None,
) -> dict[str, Any]:
    settings = roster.get("settings", {})
    all_variants = [
        variant
        for variant in eligible_variants(roster, task_class)
        if variant["id"] not in excluded_variant_ids
    ]
    cooling = _active_cooldowns(operational_state, task_class)
    variants = [variant for variant in all_variants if variant["id"] not in cooling]
    if not variants:
        if all_variants and cooling:
            raise ValueError(
                _cooldown_unavailability_message(
                    task_class=task_class, cooling=cooling, needed=1
                )
            )
        raise ValueError(f"no eligible replacement variants remain for {task_class}")
    counts = summarize_counts(records, task_class)
    return pick_weighted_without_replacement(
        variants,
        counts,
        settings,
        1,
        seed,
        relative_target=relative_underuse_target(all_variants, counts, settings),
    )[0]


def build_reroll_slot_payload(
    *,
    round_payload: dict[str, Any],
    roster: dict[str, Any],
    operational_state: dict[str, Any],
    records: list[dict[str, Any]],
    slot: str,
    seed: int | None,
) -> dict[str, Any]:
    if round_payload.get("status") != "completed":
        raise ValueError(
            f"round {round_payload['round_id']} must be completed before rerolling a slot"
        )
    slot = str(slot).strip().lower()
    indexed_runs = {str(run["slot"]): run for run in round_payload.get("runs", [])}
    if slot not in indexed_runs:
        raise ValueError(f"unknown slot for round {round_payload['round_id']}: {slot}")
    replacement_source = indexed_runs[slot]
    prior_excluded_variant_ids = {
        str(variant_id) for variant_id in round_payload.get("excluded_variant_ids", [])
    }
    unsupported_variant_ids: set[str] = set()
    block_reason = replacement_source.get("grade_block_reason")
    if block_reason == "selected_model_not_supported":
        unsupported_variant_ids = {
            str(variant["id"])
            for variant in roster.get("variants", [])
            if variant.get("model") == replacement_source.get("model")
        }
    if round_payload.get("selection_mode") == "configured":
        replacement_variant = {
            "id": replacement_source["variant_id"],
            "model": replacement_source["model"],
            "reasoning_effort": replacement_source["reasoning_effort"],
        }
        excluded_variant_ids = set(prior_excluded_variant_ids)
        if unsupported_variant_ids or block_reason == "review_timed_out":
            rating_pool_variant_ids = {
                str(variant_id)
                for variant_id in round_payload.get("rating_pool_variant_ids", [])
            }
            if not rating_pool_variant_ids:
                raise ValueError(
                    "configured unsupported-model reroll requires rating-pool variant ids"
                )
            excluded_variant_ids.update(
                str(run["variant_id"]) for run in round_payload.get("runs", [])
            )
            excluded_variant_ids.update(unsupported_variant_ids)
            excluded_variant_ids.update(
                str(variant["id"])
                for variant in roster.get("variants", [])
                if str(variant["id"]) not in rating_pool_variant_ids
            )
            replacement_variant = select_replacement_variant(
                roster=roster,
                operational_state=operational_state,
                records=records,
                task_class=str(round_payload["task_class"]),
                excluded_variant_ids=excluded_variant_ids,
                seed=seed,
            )
        runs = [
            {
                "slot": slot,
                "variant_id": replacement_variant["id"],
                "model": replacement_variant["model"],
                "reasoning_effort": replacement_variant["reasoning_effort"],
                "rerolled_from_round_id": round_payload["round_id"],
                "rerolled_from_variant_id": replacement_source["variant_id"],
            }
            if str(run["slot"]) == slot
            else _finalized_run_summary(run)
            for run in round_payload.get("runs", [])
        ]
        return {
            "round_id": make_round_id(
                str(round_payload["task_class"]),
                review_cwd=Path(str(round_payload.get("review_cwd")))
                if round_payload.get("review_cwd")
                else None,
            ),
            "task_class": round_payload["task_class"],
            "selection_mode": "configured",
            "selection_pairing": round_payload.get("selection_pairing"),
            "rating_pool_id": round_payload.get("rating_pool_id"),
            "rating_pool_variant_ids": list(
                round_payload.get("rating_pool_variant_ids", [])
            ),
            "schedule_index": round_payload.get("schedule_index"),
            "schedule_length": round_payload.get("schedule_length"),
            "sampled_at": utc_now_iso(),
            "caller_id": round_payload.get("caller_id"),
            "caller_id_source": round_payload.get("caller_id_source"),
            "review_cwd_normalized": round_payload.get("review_cwd_normalized"),
            "excluded_variant_ids": sorted(excluded_variant_ids),
            "status": "sampled",
            "runs": runs,
            "rerolled_from_round_id": round_payload["round_id"],
            "rerolled_slot": slot,
        }
    if slot not in {"alpha", "bravo"}:
        raise ValueError("reroll-slot currently supports alpha or bravo only")
    survivor_slot = "bravo" if slot == "alpha" else "alpha"
    survivor = indexed_runs.get(survivor_slot)
    if survivor is None:
        raise ValueError(
            f"round {round_payload['round_id']} is missing {survivor_slot}"
        )
    excluded_variant_ids = prior_excluded_variant_ids | unsupported_variant_ids
    if not bool(replacement_source.get("grade_blocked")):
        excluded_variant_ids.add(str(replacement_source["variant_id"]))
    if survivor.get("variant_id"):
        excluded_variant_ids.add(str(survivor["variant_id"]))
    replacement_candidates = _reroll_candidate_variants(
        roster=roster,
        operational_state=operational_state,
        records=records,
        round_payload=round_payload,
        slot=slot,
        excluded_variant_ids=excluded_variant_ids,
    )
    if not replacement_candidates:
        raise ValueError(
            f"no eligible replacement variants remain for {round_payload['task_class']}"
        )
    counts = summarize_counts(records, str(round_payload["task_class"]))
    settings = roster.get("settings", {})
    reference_variants = [
        variant
        for variant in eligible_variants(roster, str(round_payload["task_class"]))
        if variant["id"] not in excluded_variant_ids
    ]
    champion_id_set = set(
        str(item)
        for item in (round_payload.get("selection_champion_variant_ids") or [])
    )
    if (
        round_payload.get("selection_mode") == "champion"
        and (
            slot == "alpha"
            or str(round_payload.get("selection_pairing") or "")
            == "champion_vs_champion"
        )
        and replacement_candidates
        and all(
            variant["id"] not in champion_id_set for variant in replacement_candidates
        )
    ):
        replacement_variant = replacement_candidates[0]
    else:
        replacement_variant = pick_weighted_without_replacement(
            replacement_candidates,
            counts,
            settings,
            1,
            seed,
            relative_target=relative_underuse_target(
                reference_variants, counts, settings
            ),
        )[0]
    survivor_run = _finalized_run_summary(survivor)
    replacement_run = {
        "slot": slot,
        "variant_id": replacement_variant["id"],
        "model": replacement_variant["model"],
        "reasoning_effort": replacement_variant["reasoning_effort"],
        "rerolled_from_round_id": round_payload["round_id"],
        "rerolled_from_variant_id": replacement_source["variant_id"],
    }
    runs_by_slot = {
        slot: replacement_run,
        survivor_slot: survivor_run,
    }
    return {
        "round_id": make_round_id(
            str(round_payload["task_class"]),
            review_cwd=Path(str(round_payload.get("review_cwd")))
            if round_payload.get("review_cwd")
            else None,
        ),
        "task_class": round_payload["task_class"],
        "selection_mode": round_payload.get("selection_mode"),
        "selection_pairing": round_payload.get("selection_pairing"),
        "selection_champion_variant_ids": list(
            round_payload.get("selection_champion_variant_ids") or []
        ),
        "selection_anchor_kind": (
            (
                "champion"
                if replacement_variant["id"] in champion_id_set
                else "acting_champion"
            )
            if round_payload.get("selection_mode") == "champion" and slot == "alpha"
            else round_payload.get("selection_anchor_kind")
        ),
        "selection_fallback_reason": (
            (
                None
                if replacement_variant["id"] in champion_id_set
                else "champion_pool_unavailable"
            )
            if round_payload.get("selection_mode") == "champion" and slot == "alpha"
            else round_payload.get("selection_fallback_reason")
        ),
        "sampled_at": utc_now_iso(),
        "caller_id": round_payload.get("caller_id"),
        "caller_id_source": round_payload.get("caller_id_source"),
        "review_cwd_normalized": round_payload.get("review_cwd_normalized"),
        "excluded_variant_ids": sorted(excluded_variant_ids),
        "status": "sampled",
        "runs": [runs_by_slot["alpha"], runs_by_slot["bravo"]],
        "rerolled_from_round_id": round_payload["round_id"],
        "rerolled_slot": slot,
    }


def write_round(state_dir: Path, payload: dict[str, Any]) -> Path:
    path = round_path(state_dir, payload["round_id"])
    with state_lock(state_dir, f"round-{payload['round_id']}"):
        if str(payload.get("status") or "") == "dismissed":
            retry_payload = deepcopy(payload)
            retry_payload["status"] = "cleanup_pending"
            write_json(path, retry_payload)
            cleanup_succeeded = True
            for run in list(retry_payload.get("runs") or []):
                if isinstance(run, dict):
                    cleanup_succeeded = (
                        _cleanup_run_artifacts(run) and cleanup_succeeded
                    )
            if not cleanup_succeeded:
                write_json(path, retry_payload)
                raise OSError("reviewer artifact cleanup incomplete")
        write_json(path, compact_round_payload_for_storage(payload))
    return path


def load_round(state_dir: Path, round_id: str) -> dict[str, Any]:
    path = round_path(state_dir, round_id)
    if not path.exists():
        raise ValueError(f"unknown round_id: {round_id}")
    return read_json(path)


def append_record_if_new(state_dir: Path, record: dict[str, Any]) -> bool:
    with state_lock(state_dir, "runs"):
        record = compact_benchmark_record(record)
        existing_records = read_jsonl(state_dir / RUN_LOG_FILENAME)
        identity = record_identity_key(record)
        if any(
            record_identity_key(existing) == identity for existing in existing_records
        ):
            return False
        append_jsonl(state_dir / RUN_LOG_FILENAME, record)
        return True


def normalize_grade_basis(value: str, rubric: dict[str, Any]) -> str:
    basis = str(value or "").strip().lower()
    allowed = set(str(item) for item in list(rubric.get("basis") or GRADE_BASIS_VALUES))
    if basis not in allowed:
        raise ValueError(f"basis must be one of: {', '.join(sorted(allowed))}")
    return basis


def ensure_clean_git_worktree(
    review_cwd: Path, *, review_scope: dict[str, Any] | None = None
) -> None:
    try:
        dirty_entries = meaningful_worktree_status_entries(review_cwd)
    except ValueError as exc:
        raise ValueError(
            f"review-suite requires a git repo with committed changes ready for review: {exc}"
        ) from exc
    if not dirty_entries:
        return
    raise ValueError(
        "review-suite requires a clean worktree. Commit intended review changes or stash unrelated worktree changes, then rerun."
    )


def extract_session_id(text: str) -> str | None:
    marker = "session id:"
    for line in text.splitlines():
        if marker not in line.lower():
            continue
        _, _, tail = line.partition(":")
        candidate = tail.strip()
        if candidate:
            return candidate
    return None


def _started_at_epoch_seconds(value: Any) -> int | None:
    if not value:
        return None
    try:
        return int(
            datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
        )
    except ValueError:
        return None


def _normalize_review_text(value: str) -> str:
    return " ".join((value or "").strip().split()).lower()


def _capacity_interruption_detected(*, stderr_text: str, reviewer_output: str) -> bool:
    haystack = f"{stderr_text}\n{reviewer_output}".lower()
    return "selected model is at capacity" in haystack


def _review_interrupted_detected(*, stderr_text: str, reviewer_output: str) -> bool:
    output_first_line = _first_nonempty_line(reviewer_output)
    if output_first_line:
        return _normalize_review_text(output_first_line).startswith(
            "review was interrupted"
        )
    return "review was interrupted" in (stderr_text or "").lower()


def _tooling_failure_detected(*, stderr_text: str, reviewer_output: str) -> bool:
    output_haystack = (reviewer_output or "").lower()
    stderr_haystack = (stderr_text or "").lower()
    output_markers = (
        "i could not inspect the changes because shell access failed",
        "without a readable diff",
        "cannot identify verifiable, actionable defects, so no findings are reported",
    )
    if any(marker in output_haystack for marker in output_markers):
        return True
    if not output_haystack.strip():
        return "windows sandbox: setup refresh failed" in stderr_haystack
    return False


def _first_nonempty_line(value: str) -> str | None:
    for line in (value or "").splitlines():
        text = line.strip()
        if text:
            return text
    return None


def _review_output_summary(reviewer_output: str) -> str | None:
    first_line = _first_nonempty_line(reviewer_output)
    if not first_line:
        return None
    if len(first_line) <= 220:
        return first_line
    return f"{first_line[:217]}..."


def terminal_review_command(reviewer_output: str) -> str | None:
    lines = [
        line.strip() for line in str(reviewer_output or "").splitlines() if line.strip()
    ]
    if not lines:
        return None
    final_line = lines[-1]
    result_lines = [
        line
        for line in lines
        if line.lower().startswith(TERMINAL_REVIEW_RESULT_PREFIX.lower())
    ]
    if result_lines != [final_line]:
        return None
    match = re.fullmatch(
        rf"{re.escape(TERMINAL_REVIEW_RESULT_PREFIX)}\s*(clean|findings)",
        final_line,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    command = match.group(1).lower()
    return command if command in TERMINAL_REVIEW_COMMANDS else None


def _classify_review_result(
    *,
    reviewer_output: str,
    stderr_text: str,
    session_id: str | None,
    thread_id: str | None,
    rollout_error: str = "",
) -> dict[str, Any]:
    output = (reviewer_output or "").strip()
    authentication_failed = (
        "access token could not be refreshed" in (rollout_error or "").lower()
    )
    unsupported_model = (
        "model is not supported when using codex with a chatgpt account"
        in (rollout_error or "").lower()
    )
    capacity = _capacity_interruption_detected(
        stderr_text=stderr_text, reviewer_output=output
    )
    interrupted = _review_interrupted_detected(
        stderr_text=stderr_text, reviewer_output=output
    )
    if authentication_failed and (not output or interrupted):
        return {
            "review_status": "authentication_failed",
            "status_summary": "Codex authentication failed: access token could not be refreshed. Log out and sign in again, then resume this review.",
            "grade_blocked": True,
            "grade_block_reason": "codex_authentication_failed",
        }
    if unsupported_model and (not output or interrupted):
        return {
            "review_status": "unsupported_model",
            "status_summary": "Selected model is not supported by Codex for this ChatGPT account.",
            "grade_blocked": True,
            "grade_block_reason": "selected_model_not_supported",
        }
    if capacity and (
        not output or interrupted or "selected model is at capacity" in output.lower()
    ):
        return {
            "review_status": "interrupted_capacity",
            "status_summary": "selected_model_at_capacity",
            "grade_blocked": True,
            "grade_block_reason": "selected_model_at_capacity",
        }
    opencode_metadata = parse_opencode_review_metadata(stderr_text)
    opencode_error = str(opencode_metadata.get("error_class") or "").strip().lower()
    if (not output or interrupted) and opencode_error:
        reason = OPENCODE_ERROR_BLOCK_REASONS.get(opencode_error)
        review_status = OPENCODE_ERROR_REVIEW_STATUSES.get(opencode_error)
        if reason and review_status:
            if opencode_error == "capacity":
                return {
                    "review_status": review_status,
                    "status_summary": "selected_model_at_capacity",
                    "grade_blocked": True,
                    "grade_block_reason": reason,
                }
            detail = " / ".join(
                part
                for part in (
                    str(opencode_metadata.get("error_name") or "").strip(),
                    str(opencode_metadata.get("error_message") or "").strip(),
                )
                if part
            )
            summary = OPENCODE_ERROR_STATUS_SUMMARIES.get(
                opencode_error, "OpenCode review failed before a usable result."
            )
            if detail:
                summary = f"{summary} ({detail})"
            return {
                "review_status": review_status,
                "status_summary": summary,
                "grade_blocked": True,
                "grade_block_reason": reason,
            }
    if _tooling_failure_detected(stderr_text=stderr_text, reviewer_output=output):
        return {
            "review_status": "tooling_failure",
            "status_summary": "review_tooling_failure",
            "grade_blocked": True,
            "grade_block_reason": "review_tooling_failure",
        }
    if interrupted:
        return {
            "review_status": "interrupted",
            "status_summary": "Review was interrupted before a usable result was captured. Do not grade this against the model.",
            "grade_blocked": True,
            "grade_block_reason": "review_interrupted",
        }
    if output:
        terminal_command = terminal_review_command(output)
        return {
            "review_status": "completed",
            "status_summary": _review_output_summary(output) or "Review completed.",
            "grade_blocked": False,
            "grade_block_reason": None,
            "terminal_command": terminal_command,
        }
    if session_id or thread_id:
        return {
            "review_status": "completed_no_output",
            "status_summary": "Reviewer session finished, but no review text was captured.",
            "grade_blocked": True,
            "grade_block_reason": "missing_reviewer_output",
        }
    return {
        "review_status": "process_died",
        "status_summary": "Reviewer process exited without a captured result.",
        "grade_blocked": True,
        "grade_block_reason": "reviewer_process_exited",
    }


def classify_review_capture(
    *,
    reviewer_output: str,
    stderr_text: str,
    session_id: str | None,
    thread_id: str | None,
    rollout_error: str = "",
    timed_out: bool = False,
    transport_stalled: bool = False,
) -> dict[str, Any]:
    if transport_stalled:
        return {
            "review_status": "transport_stalled",
            "status_summary": "Reviewer transport stalled after reconnect exhaustion before a usable result was captured.",
            "grade_blocked": True,
            "grade_block_reason": "review_transport_stalled",
        }
    if timed_out:
        return {
            "review_status": "timeout",
            "status_summary": (
                "Reviewer timed out after partial output was captured. Do not treat this as a completed review."
                if str(reviewer_output or "").strip()
                else "Reviewer timed out before a usable result was captured."
            ),
            "grade_blocked": True,
            "grade_block_reason": "review_timed_out",
        }
    return _classify_review_result(
        reviewer_output=reviewer_output,
        stderr_text=stderr_text,
        session_id=session_id,
        thread_id=thread_id,
        rollout_error=rollout_error,
    )


def _classification_for_run(run: dict[str, Any]) -> dict[str, Any]:
    existing_status = run.get("review_status")
    if existing_status:
        if existing_status in {
            "completed",
            "completed_findings",
            "completed_no_findings",
        } and _tooling_failure_detected(
            stderr_text=str(run.get("stderr") or ""),
            reviewer_output=str(run.get("reviewer_output") or ""),
        ):
            return _classify_review_result(
                reviewer_output=str(run.get("reviewer_output") or ""),
                stderr_text=str(run.get("stderr") or ""),
                session_id=str(run.get("session_id") or "") or None,
                thread_id=str(run.get("thread_id") or "") or None,
            )
        if existing_status in {
            "completed",
            "completed_findings",
            "completed_no_findings",
        }:
            return {
                "review_status": "completed",
                "status_summary": run.get("status_summary")
                or _review_output_summary(str(run.get("reviewer_output") or ""))
                or "Review completed.",
                "grade_blocked": False,
                "grade_block_reason": None,
                "terminal_command": run.get("terminal_command")
                or terminal_review_command(str(run.get("reviewer_output") or "")),
            }
        return {
            "review_status": existing_status,
            "status_summary": run.get("status_summary"),
            "grade_blocked": bool(run.get("grade_blocked")),
            "grade_block_reason": run.get("grade_block_reason"),
        }
    return _classify_review_result(
        reviewer_output=str(run.get("reviewer_output") or ""),
        stderr_text=str(run.get("stderr") or ""),
        session_id=str(run.get("session_id") or "") or None,
        thread_id=str(run.get("thread_id") or "") or None,
    )


def _read_review_artifacts(
    *,
    stdout_path: Path,
    stderr_path: Path,
    final_message_path: Path | None = None,
) -> tuple[str, str]:
    stderr_text = (
        stderr_path.read_text(encoding="utf-8", errors="replace").strip()
        if stderr_path.exists()
        else ""
    )
    reviewer_output = (
        stdout_path.read_text(encoding="utf-8", errors="replace").strip()
        if stdout_path.exists()
        else ""
    )
    if final_message_path is not None and final_message_path.exists():
        final_message = final_message_path.read_text(
            encoding="utf-8", errors="replace"
        ).strip()
        if final_message:
            reviewer_output = final_message
    return stderr_text, reviewer_output


def collect_completed_review_capture(
    *,
    slot: str,
    variant_id: str,
    variant: dict[str, Any],
    title: str,
    command: list[str] | None,
    stdout_path: Path,
    stderr_path: Path,
    started_at: str | None,
    sqlite_path: Path,
    review_cwd: Path,
    final_message_path: Path | None = None,
    timed_out: bool = False,
    transport_stalled: bool = False,
) -> dict[str, Any]:
    stderr_text, reviewer_output = _read_review_artifacts(
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        final_message_path=final_message_path,
    )
    metadata = parse_opencode_review_metadata(stderr_text)
    opencode_backend = (
        str(variant.get("model") or "").strip().startswith(OPENCODE_MODEL_PREFIX)
    )
    session_id = extract_session_id(stderr_text)
    thread = None
    if not opencode_backend and session_id:
        for attempt in range(6):
            candidate = find_thread_by_id(sqlite_path=sqlite_path, thread_id=session_id)
            if (
                candidate
                and str(candidate.get("source") or "") == REVIEW_SUBAGENT_SOURCE
            ):
                thread = candidate
            else:
                thread = find_review_child_thread(
                    sqlite_path=sqlite_path, parent_thread_id=session_id
                )
            if thread:
                break
            if attempt < 5:
                time.sleep(0.5)
    enriched = enrich_thread_record(thread) if thread else {}
    if not reviewer_output and enriched.get("reviewer_output"):
        reviewer_output = enriched["reviewer_output"]
    if opencode_backend:
        metadata_usage = metadata.get("usage")
        usage = dict(metadata_usage) if isinstance(metadata_usage, dict) else {}
        captured_cost = metadata.get("cost_usd")
        cost_usd = (
            float(captured_cost)
            if isinstance(captured_cost, (int, float))
            and not isinstance(captured_cost, bool)
            else None
        )
        session_id = str(metadata.get("session_id") or "").strip() or session_id
        thread_id = None
        rollout_path = None
        tokens_used = (
            int(usage["total_tokens"])
            if isinstance(usage.get("total_tokens"), int)
            else total_usage_tokens(usage)
        )
    else:
        usage = dict(enriched.get("usage") or {})
        cost_usd = compute_cost_usd(variant, usage)
        thread_id = enriched.get("id")
        rollout_path = enriched.get("rollout_path")
        tokens_used = enriched.get("tokens_used")
    classification = classify_review_capture(
        reviewer_output=reviewer_output,
        stderr_text=stderr_text,
        session_id=session_id,
        thread_id=str(thread_id or "") or None,
        rollout_error=str(enriched.get("task_error") or ""),
        timed_out=timed_out,
        transport_stalled=transport_stalled,
    )
    elapsed_seconds = None
    if started_at:
        elapsed_seconds = round(
            (
                utc_now()
                - datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
            ).total_seconds(),
            3,
        )
    return {
        "slot": slot,
        "variant_id": variant_id,
        "service_tier": variant_service_tier(variant),
        "title": title,
        "command": command,
        "returncode": 0 if session_id or reviewer_output else None,
        "elapsed_seconds": elapsed_seconds,
        "session_id": session_id,
        "thread_id": thread_id,
        "rollout_path": rollout_path,
        "tokens_used": tokens_used,
        "usage": usage,
        "cost_usd": cost_usd,
        "reviewer_output": reviewer_output,
        "stderr": stderr_text,
        "review_status": classification["review_status"],
        "status_summary": classification["status_summary"],
        "grade_blocked": classification["grade_blocked"],
        "grade_block_reason": classification["grade_block_reason"],
        "terminal_command": classification.get("terminal_command"),
        "reviewer_output_ref": (
            f"rollout://{thread_id}/{variant_id}" if thread_id else None
        ),
        "cooldown_eligible": bool(opencode_backend and classification["grade_blocked"]),
    }


def reject_duplicate_review_references(runs: list[dict[str, Any]]) -> None:
    duplicate_indexes: set[int] = set()
    for key in ("thread_id", "rollout_path", "reviewer_output_ref"):
        indexes_by_value: dict[str, list[int]] = {}
        for index, run in enumerate(runs):
            value = str(run.get(key) or "").strip()
            if value:
                indexes_by_value.setdefault(value, []).append(index)
        for indexes in indexes_by_value.values():
            if len(indexes) > 1:
                duplicate_indexes.update(indexes)
    for index in duplicate_indexes:
        run = runs[index]
        run.update(
            {
                "review_status": "duplicate_output",
                "status_summary": "Reviewer output duplicated another reviewer's rollout reference and was rejected.",
                "grade_blocked": True,
                "grade_block_reason": "duplicate_reviewer_output",
                "terminal_command": None,
            }
        )


def _load_live_run_artifacts(item: dict[str, Any]) -> tuple[str, str]:
    stderr_path = Path(str(item["stderr_path"]))
    stdout_path = Path(str(item["stdout_path"]))
    final_path_text = str(item.get("final_message_path") or "").strip()
    return _read_review_artifacts(
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        final_message_path=Path(final_path_text) if final_path_text else None,
    )


def _summarize_live_run(item: dict[str, Any]) -> dict[str, Any]:
    stderr_text, reviewer_output = _load_live_run_artifacts(item)
    summary = _classify_review_result(
        reviewer_output=reviewer_output,
        stderr_text=stderr_text,
        session_id=None,
        thread_id=None,
    )
    summary["stderr"] = stderr_text
    summary["reviewer_output"] = reviewer_output
    return summary


def _apply_capacity_cooldowns(
    *, state_dir: Path, round_payload: dict[str, Any]
) -> list[dict[str, Any]]:
    updates: list[dict[str, Any]] = []
    with state_lock(state_dir, "operational-state"):
        state_path = state_dir / OPERATIONAL_STATE_FILENAME
        operational_state = load_operational_state(state_path)
        task_state = operational_state["task_classes"][str(round_payload["task_class"])]
        cooldowns = dict(task_state.get("cooldowns") or {})
        cooldown_failures = dict(task_state.get("cooldown_failures") or {})
        changed = False
        now = utc_now()
        now_iso = now.isoformat().replace("+00:00", "Z")
        triggered_variants: set[str] = set()
        completed_variants: set[str] = set()
        for run in round_payload.get("runs", []):
            classification = _classification_for_run(run)
            variant_id = str(run.get("variant_id") or "")
            if not variant_id:
                continue
            block_reason = str(classification.get("grade_block_reason") or "")
            if block_reason in COOLDOWN_BLOCK_REASONS or (
                bool(run.get("cooldown_eligible"))
                and block_reason in MARKED_COOLDOWN_BLOCK_REASONS
            ):
                triggered_variants.add(variant_id)
                current = (
                    cooldown_failures.get(variant_id) or cooldowns.get(variant_id) or {}
                )
                failure_count = int(current.get("failure_count", 0) or 0) + 1
                until_iso = (
                    (now + timedelta(seconds=_capacity_cooldown_seconds(failure_count)))
                    .isoformat()
                    .replace("+00:00", "Z")
                )
                cooldowns[variant_id] = {
                    "until": until_iso,
                    "failure_count": failure_count,
                    "last_reason": block_reason,
                    "last_triggered_at": now_iso,
                }
                cooldown_failures[variant_id] = {
                    "failure_count": failure_count,
                    "last_reason": block_reason,
                    "last_triggered_at": now_iso,
                }
                updates.append(
                    {
                        "variant_id": variant_id,
                        "reason": block_reason,
                        "failure_count": failure_count,
                        "until": until_iso,
                    }
                )
                changed = True
                continue
            if classification["review_status"] == "completed":
                completed_variants.add(variant_id)
        for variant_id in completed_variants - triggered_variants:
            if variant_id in cooldowns:
                cooldowns.pop(variant_id, None)
                changed = True
            if variant_id in cooldown_failures:
                cooldown_failures.pop(variant_id, None)
                changed = True
        if not changed:
            return updates
        task_state["cooldowns"] = cooldowns
        task_state["cooldown_failures"] = cooldown_failures
        operational_state["generated_at"] = utc_now_iso()
        _prune_expired_cooldowns(operational_state)
        write_json(state_path, operational_state)
    return updates


def _review_run_title(
    *, round_id: str, slot: str, variant_id: str, capacity_retry_attempts: int = 0
) -> str:
    title = f"review-suite::{round_id}::{slot}::{variant_id}"
    if capacity_retry_attempts <= 0:
        return title
    return f"{title}::retry{capacity_retry_attempts}"


def _launch_reviewer_process(
    *,
    round_payload: dict[str, Any],
    run: dict[str, Any],
    variant: dict[str, Any],
    review_cwd: Path,
    prompt: str,
    review_scope: dict[str, Any],
    allow_unsafe_windows_wsl_fallback: bool,
) -> dict[str, Any]:
    retry_attempts = int(run.get("capacity_retry_attempts", 0) or 0)
    title = _review_run_title(
        round_id=str(round_payload["round_id"]),
        slot=str(run["slot"]),
        variant_id=str(run["variant_id"]),
        capacity_retry_attempts=retry_attempts,
    )
    base_ref = str(review_scope.get("base") or "").strip()
    commit_ref = str(review_scope.get("commit") or "").strip()
    service_tier = variant_service_tier(variant)
    launch = prepare_codex_review_launch(
        tool_name="review-suite",
        model=variant["model"],
        reasoning_effort=variant["reasoning_effort"],
        service_tier=service_tier,
        title=title,
        review_root=review_cwd,
        base=base_ref or None,
        commit=None if base_ref else commit_ref or None,
        commit_end=str(review_scope.get("commit_end") or "").strip() or None,
        prompt=prompt,
        output_prefix=f"review-suite-{run['slot']}-message-",
        allow_unsafe_windows_wsl_fallback=allow_unsafe_windows_wsl_fallback,
    )
    child = launch_captured_child_process(
        command=launch.command,
        cwd=launch.cwd,
        env=launch.env,
        stdin_text=launch.stdin_text,
        stdout_prefix=f"review-suite-{run['slot']}-",
    )
    updates = {
        "title": title,
        "command": launch.command,
        "service_tier": service_tier,
        "effective_reasoning_effort": launch.effective_reasoning_effort,
        "pid": child.process.pid,
        "started_at": utc_now_iso(),
        "stdout_path": str(child.stdout_path),
        "stderr_path": str(child.stderr_path),
    }
    if launch.final_message_path is not None:
        updates["final_message_path"] = str(launch.final_message_path)
    run.update(updates)
    return run


def _cleanup_run_artifacts(run: dict[str, Any]) -> bool:
    succeeded = True
    for path_key in ("stdout_path", "stderr_path", "final_message_path"):
        path_value = str(run.get(path_key) or "").strip()
        if not path_value:
            continue
        try:
            Path(path_value).unlink(missing_ok=True)
        except OSError:
            succeeded = False
        else:
            run.pop(path_key, None)
    return succeeded


def _persist_cleanup_failure(state_dir: Path, payload: dict[str, Any]) -> None:
    write_round(state_dir, payload)
    raise OSError("reviewer artifact cleanup incomplete")


def _strip_live_run_transient_fields(run: dict[str, Any]) -> dict[str, Any]:
    cleaned = deepcopy(run)
    for key in (
        "pid",
        "stdout_path",
        "stderr_path",
        "final_message_path",
        "_rollout_activity_epoch",
    ):
        cleaned.pop(key, None)
    return cleaned


def _collect_completed_run_from_artifacts(
    *,
    item: dict[str, Any],
    indexed: dict[str, dict[str, Any]],
    sqlite_path: Path,
    review_cwd: Path,
    transport_stalled: bool = False,
    timed_out: bool = False,
) -> dict[str, Any]:
    variant_id = str(item["variant_id"])
    variant = dict(indexed[variant_id])
    if item.get("effective_reasoning_effort"):
        variant["effective_reasoning_effort"] = item["effective_reasoning_effort"]
    return collect_completed_review_capture(
        slot=str(item["slot"]),
        variant_id=variant_id,
        variant=variant,
        title=str(item["title"]),
        command=list(item.get("command") or []),
        stdout_path=Path(str(item["stdout_path"])),
        stderr_path=Path(str(item["stderr_path"])),
        started_at=str(item.get("started_at") or "") or None,
        sqlite_path=sqlite_path,
        review_cwd=review_cwd,
        final_message_path=Path(str(item["final_message_path"]))
        if item.get("final_message_path")
        else None,
        transport_stalled=transport_stalled,
        timed_out=timed_out,
    )


def _maybe_retry_capacity_run(
    *,
    round_payload: dict[str, Any],
    run: dict[str, Any],
    indexed: dict[str, dict[str, Any]],
    state_dir: Path,
    review_cwd: Path,
) -> bool:
    summary = (
        {}
        if (cleanup_pending := bool(run.get("capacity_cleanup_pending")))
        else _summarize_live_run(run)
    )
    if summary and summary.get("grade_block_reason") != "selected_model_at_capacity":
        return False
    retry_attempts = int(run.get("capacity_retry_attempts", 0) or 0)
    if not cleanup_pending and retry_attempts >= CAPACITY_RETRY_MAX_ATTEMPTS:
        return False
    next_attempt = retry_attempts if cleanup_pending else retry_attempts + 1
    print(
        f"[review-suite] {public_reviewer_label(str(run['slot']))} hit capacity; retrying in {CAPACITY_RETRY_DELAY_SECONDS}s "
        f"(attempt {next_attempt}/{CAPACITY_RETRY_MAX_ATTEMPTS})",
        file=sys.stderr,
        flush=True,
    )
    run.update(capacity_retry_attempts=next_attempt, capacity_cleanup_pending=True)
    write_round(state_dir, round_payload)
    if _cleanup_run_artifacts(run) is False:
        _persist_cleanup_failure(state_dir, round_payload)
    run.pop("capacity_cleanup_pending", None)
    for transient_key in (
        "pid",
        "session_id",
        "thread_id",
        "rollout_path",
        "tokens_used",
        "usage",
        "reviewer_output",
        "stderr",
        "review_status",
        "status_summary",
        "grade_blocked",
        "grade_block_reason",
        "reviewer_output_ref",
        "returncode",
        "elapsed_seconds",
        "service_tier",
    ):
        run.pop(transient_key, None)
    time.sleep(CAPACITY_RETRY_DELAY_SECONDS)
    _launch_reviewer_process(
        round_payload=round_payload,
        run=run,
        variant=indexed[str(run["variant_id"])],
        review_cwd=review_cwd,
        prompt=str(round_payload.get("requested_prompt") or ""),
        review_scope=deepcopy(round_payload.get("review_scope") or {}),
        allow_unsafe_windows_wsl_fallback=bool(
            round_payload.get("allow_unsafe_windows_wsl_fallback")
        ),
    )
    write_round(state_dir, round_payload)
    return True


def run_round(
    *,
    round_payload: dict[str, Any],
    roster: dict[str, Any],
    state_dir: Path,
    review_cwd: Path,
    prompt: str,
    review_scope: dict[str, Any],
    sqlite_path: Path = DEFAULT_SQLITE_STATE_PATH,
    progress_interval_seconds: int = 30,
    allow_unsafe_windows_wsl_fallback: bool = False,
) -> dict[str, Any]:
    launched = launch_round(
        round_payload=round_payload,
        roster=roster,
        state_dir=state_dir,
        review_cwd=review_cwd,
        prompt=prompt,
        review_scope=review_scope,
        progress_interval_seconds=progress_interval_seconds,
        allow_unsafe_windows_wsl_fallback=allow_unsafe_windows_wsl_fallback,
    )
    return collect_round_results(
        round_payload=launched,
        roster=roster,
        state_dir=state_dir,
        review_cwd=review_cwd,
        sqlite_path=sqlite_path,
        progress_interval_seconds=progress_interval_seconds,
        wait=True,
    )


def launch_round(
    *,
    round_payload: dict[str, Any],
    roster: dict[str, Any],
    state_dir: Path,
    review_cwd: Path,
    prompt: str,
    review_scope: dict[str, Any],
    progress_interval_seconds: int = 30,
    allow_unsafe_windows_wsl_fallback: bool = False,
) -> dict[str, Any]:
    indexed = variant_index(roster)
    review_cwd = review_cwd.resolve()
    if round_payload.get("status") not in {"sampled", "failed"}:
        raise ValueError(
            f"round {round_payload['round_id']} is already {round_payload.get('status')}"
        )
    if review_scope.get("base"):
        ensure_clean_git_worktree(review_cwd, review_scope=review_scope)

    review_started_at = utc_now_iso()
    running_payload = deepcopy(round_payload)
    running_payload["status"] = "running"
    running_payload["review_cwd"] = str(review_cwd)
    running_payload["review_started_at"] = review_started_at
    running_payload["review_scope"] = review_scope
    running_payload["allow_unsafe_windows_wsl_fallback"] = (
        allow_unsafe_windows_wsl_fallback
    )
    running_payload["progress_interval_seconds"] = progress_interval_seconds
    launched_count = 0
    for run in round_payload["runs"]:
        if _run_is_finalized(run):
            continue
        if launched_count:
            time.sleep(MULTI_REVIEW_DISPATCH_STAGGER_SECONDS)
        variant = indexed[run["variant_id"]]
        _launch_reviewer_process(
            round_payload=round_payload,
            run=run,
            variant=variant,
            review_cwd=review_cwd,
            prompt=prompt,
            review_scope=review_scope,
            allow_unsafe_windows_wsl_fallback=allow_unsafe_windows_wsl_fallback,
        )
        launched_count += 1
        running_run = deepcopy(run)
        running_payload["runs"] = [
            running_run
            if existing["slot"] == running_run["slot"]
            else deepcopy(existing)
            for existing in running_payload["runs"]
        ]
        write_round(state_dir, running_payload)
    if prompt:
        running_payload["requested_prompt"] = prompt
    if not any(
        run.get("pid") for run in running_payload["runs"] if not _run_is_finalized(run)
    ):
        running_payload["status"] = "completed"
        running_payload["review_completed_at"] = utc_now_iso()
    write_round(state_dir, running_payload)
    return running_payload


def _process_is_running(pid: int | None) -> bool:
    try:
        normalized_pid = int(pid) if pid is not None else 0
    except TypeError, ValueError, OverflowError:
        return False
    if normalized_pid <= 0 or normalized_pid > 0xFFFFFFFF:
        return False
    if os.name == "nt":
        handle = _open_process(_PROCESS_SYNCHRONIZE, False, normalized_pid)
        if not handle:
            return ctypes.get_last_error() == _ERROR_ACCESS_DENIED
        try:
            return _wait_for_single_object(handle, 0) == _WAIT_TIMEOUT
        finally:
            _close_handle(handle)
    try:
        with open(
            f"/proc/{normalized_pid}/status", encoding="utf-8", errors="replace"
        ) as handle:
            for line in handle:
                if not line.startswith("State:"):
                    continue
                if "\tZ" in line or "zombie" in line.lower():
                    return False
                break
    except FileNotFoundError:
        return False
    except OSError:
        pass
    try:
        os.kill(normalized_pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return False
    except PermissionError:
        return True
    return True


def round_has_live_reviewer_process(round_payload: dict[str, Any]) -> bool:
    for run in list(round_payload.get("runs") or []):
        if _run_is_finalized(run):
            continue
        if _process_is_running(run.get("pid")):
            return True
    return False


def _reviewer_wait_line(round_payload: dict[str, Any]) -> str:
    count = len(
        [run for run in list(round_payload.get("runs") or []) if isinstance(run, dict)]
    )
    label = "reviewer" if count == 1 else "reviewers"
    return f"[review-suite] waiting for {count} {label}; wrapper is active as long as output streams, do not stop it prematurely"


def collect_round_results(
    *,
    round_payload: dict[str, Any],
    roster: dict[str, Any],
    state_dir: Path,
    review_cwd: Path,
    sqlite_path: Path = DEFAULT_SQLITE_STATE_PATH,
    progress_interval_seconds: int = 30,
    wait: bool = True,
) -> dict[str, Any]:
    indexed = variant_index(roster)
    if round_payload.get("status") != "running":
        raise ValueError(f"round {round_payload['round_id']} is not running")

    last_progress = time.monotonic()
    announced_terminal_states: set[str] = set()
    stall_warned_slots: set[str] = set()
    live_completion_statuses: dict[str, str] = {}
    print(_reviewer_wait_line(round_payload), file=sys.stderr, flush=True)
    while True:
        alive = [
            run for run in round_payload["runs"] if _process_is_running(run.get("pid"))
        ]
        restarted_capacity_run = False
        for item in round_payload["runs"]:
            slot = str(item["slot"])
            if _run_is_finalized(item):
                announced_terminal_states.add(slot)
                continue
            if not item.get("capacity_cleanup_pending") and (
                not item.get("stderr_path") or not item.get("stdout_path")
            ):
                announced_terminal_states.add(slot)
                continue
            if any(alive_item["slot"] == slot for alive_item in alive):
                continue
            if slot in announced_terminal_states:
                continue
            if _maybe_retry_capacity_run(
                round_payload=round_payload,
                run=item,
                indexed=indexed,
                state_dir=state_dir,
                review_cwd=review_cwd,
            ):
                announced_terminal_states.discard(slot)
                last_progress = time.monotonic()
                restarted_capacity_run = True
                break
            terminal_summary = _collect_completed_run_from_artifacts(
                item=item,
                indexed=indexed,
                sqlite_path=sqlite_path,
                review_cwd=review_cwd,
                transport_stalled=bool(item.get("transport_stalled")),
                timed_out=bool(item.get("timed_out")),
            )
            item.update(terminal_summary)
            write_round(state_dir, round_payload)
            _print_live_completed_run(terminal_summary)
            live_completion_statuses[slot] = reviewer_completion_status(
                terminal_summary
            )
            announced_terminal_states.add(slot)
        if restarted_capacity_run:
            continue
        if not alive:
            break
        if not wait:
            return round_payload
        now = time.monotonic()
        if now - last_progress >= progress_interval_seconds:
            _print_transport_events(alive)
            for item in list(alive):
                variant = dict(indexed[str(item["variant_id"])])
                deadline_reason = _reviewer_deadline_reason(
                    run=item,
                    variant=variant,
                    sqlite_path=sqlite_path,
                    review_cwd=review_cwd,
                )
                if deadline_reason is not None:
                    label = public_reviewer_label(str(item.get("slot") or "reviewer"))
                    print(
                        f"[review-suite] {label} reached {deadline_reason}; stopping reviewer.",
                        file=sys.stderr,
                        flush=True,
                    )
                    item["timed_out"] = True
                    _terminate_process_tree(item.get("pid"))
                    continue
                stall_reason = _transport_stalled(item)
                if stall_reason is None:
                    hang_reason = _transport_hung_after_output(item)
                    if hang_reason is None:
                        continue
                    label = public_reviewer_label(str(item.get("slot") or "reviewer"))
                    print(
                        f"[review-suite] {label} transport hung after output ({hang_reason}); stopping process and preserving captured output.",
                        file=sys.stderr,
                        flush=True,
                    )
                    _terminate_process_tree(item.get("pid"))
                else:
                    label = public_reviewer_label(str(item.get("slot") or "reviewer"))
                    print(
                        f"[review-suite] {label} transport stalled ({stall_reason}); stopping this reviewer and preserving the other reviewer output.",
                        file=sys.stderr,
                        flush=True,
                    )
                    item["transport_stalled"] = True
                    _terminate_process_tree(item.get("pid"))
            _print_stall_warnings(
                active_runs=alive,
                indexed=indexed,
                sqlite_path=sqlite_path,
                review_cwd=review_cwd,
                warned_slots=stall_warned_slots,
            )
            print(_progress_status_line(alive), file=sys.stderr, flush=True)
            last_progress = now
        time.sleep(1.0)

    completed_runs: list[dict[str, Any]] = []
    for item in round_payload["runs"]:
        if _run_is_finalized(item):
            completed_runs.append(
                _strip_live_run_transient_fields(_finalized_run_summary(item))
            )
            if _cleanup_run_artifacts(item) is False:
                _persist_cleanup_failure(state_dir, round_payload)
            continue
        if not item.get("stderr_path") or not item.get("stdout_path"):
            completed_runs.append(
                _strip_live_run_transient_fields(_finalized_run_summary(item))
            )
            continue
        captured = _collect_completed_run_from_artifacts(
            item=item,
            indexed=indexed,
            sqlite_path=sqlite_path,
            review_cwd=review_cwd,
            transport_stalled=bool(item.get("transport_stalled")),
            timed_out=bool(item.get("timed_out")),
        )
        item.update(captured)
        completed_runs.append(captured)
        if _cleanup_run_artifacts(item) is False:
            _persist_cleanup_failure(state_dir, round_payload)
    reject_duplicate_review_references(completed_runs)
    round_payload = deepcopy(round_payload)
    round_payload["status"] = "completed"
    round_payload["review_completed_at"] = utc_now_iso()
    round_payload["live_completion_statuses"] = {
        str(run["slot"]): reviewer_completion_status(run) for run in completed_runs
    }
    round_payload["runs"] = completed_runs
    cooldown_updates = _apply_capacity_cooldowns(
        state_dir=state_dir, round_payload=round_payload
    )
    if cooldown_updates:
        round_payload["cooldown_updates"] = cooldown_updates
        for update in cooldown_updates:
            print(
                f"[review-suite] cooling {update['variant_id']} for {round_payload['task_class']} until {format_cooldown_until_for_display(update['until'])} after {update['reason']} (failure_count={update['failure_count']})",
                file=sys.stderr,
                flush=True,
            )
    write_round(state_dir, round_payload)
    return round_payload


def record_identity_key(record: dict[str, Any]) -> str:
    identity_runs: list[dict[str, Any]] = []
    for run in sorted(record.get("runs", []), key=lambda item: item["variant_id"]):
        identity_runs.append(
            {
                "variant_id": run["variant_id"],
                "reviewer_output": run.get("reviewer_output"),
                "reviewer_output_ref": run.get("reviewer_output_ref"),
                "grader_notes": run.get("grader_notes"),
                "elapsed_seconds": run.get("elapsed_seconds"),
            }
        )
    return json.dumps(
        {
            "task_class": record.get("task_class"),
            "task_id": record.get("task_id"),
            "selection_mode": record.get("selection_mode"),
            "rating_pool_id": record.get("rating_pool_id"),
            "placement_v1": record.get("placement_v1"),
            "runs": identity_runs,
        },
        sort_keys=True,
    )


def compact_benchmark_run(run: dict[str, Any]) -> dict[str, Any]:
    compacted: dict[str, Any] = {
        "variant_id": run["variant_id"],
        "service_tier": run.get("service_tier"),
        "elapsed_seconds": run.get("elapsed_seconds"),
        "usage": deepcopy(run.get("usage", {})),
        "cost_usd": run.get("cost_usd"),
    }
    if isinstance(run.get("tokens_used"), int) and not isinstance(
        run.get("tokens_used"), bool
    ):
        compacted["tokens_used"] = int(run["tokens_used"])
    if not compacted["service_tier"]:
        compacted.pop("service_tier", None)
    reviewer_output = run.get("reviewer_output")
    if reviewer_output:
        compacted["reviewer_output"] = reviewer_output
    reviewer_output_ref = run.get("reviewer_output_ref")
    if reviewer_output_ref:
        compacted["reviewer_output_ref"] = reviewer_output_ref
    grader_notes = run.get("grader_notes")
    if grader_notes:
        compacted["grader_notes"] = grader_notes
    return compacted


def compact_benchmark_record(record: dict[str, Any]) -> dict[str, Any]:
    if placement_record(record) is None:
        return deepcopy(record)
    compacted = {
        "recorded_at": record.get("recorded_at"),
        "round_id": record.get("round_id"),
        "task_class": record.get("task_class"),
        "task_id": record.get("task_id"),
        "selection_mode": record.get("selection_mode"),
        "rating_pool_id": record["rating_pool_id"],
        "placement_v1": deepcopy(record["placement_v1"]),
        "runs": [compact_benchmark_run(run) for run in record.get("runs", [])],
    }
    if record.get("reporting_pool"):
        compacted["reporting_pool"] = True
    repo_name = str(record.get("repo_name") or "").strip()
    if repo_name:
        compacted["repo_name"] = repo_name
    return compacted


def placement_record(
    record: dict[str, Any], *, expected_variants: set[str] | None = None
) -> tuple[str, list[list[str]], str] | None:
    if "placement_v1" not in record:
        return None
    placement = record["placement_v1"]
    if not isinstance(placement, dict):
        raise ValueError("placement_v1 must be an object")
    rating_pool_id = str(record.get("rating_pool_id") or "").strip()
    if not rating_pool_id:
        raise ValueError("rating_pool_id is required")
    basis = str(placement.get("basis") or "").strip()
    if basis not in GRADE_BASIS_VALUES:
        raise ValueError(f"invalid placement basis: {basis or '<empty>'}")
    raw_groups = placement.get("groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError("placement_v1.groups must be a non-empty array")
    groups: list[list[str]] = []
    seen: set[str] = set()
    for raw_group in raw_groups:
        if not isinstance(raw_group, list) or not raw_group:
            raise ValueError("each placement group must be a non-empty array")
        group = [str(value).strip() for value in raw_group]
        if any(not value for value in group):
            raise ValueError("placement groups cannot contain empty variant ids")
        duplicate = next((value for value in group if value in seen), None)
        if duplicate:
            raise ValueError(f"duplicate placement variant: {duplicate}")
        if len(group) != len(set(group)):
            raise ValueError("placement groups cannot contain duplicate variants")
        groups.append(group)
        seen.update(group)
    if len(seen) < 2:
        raise ValueError("placement requires at least two distinct variants")
    if len(groups) == 1 and not basis.startswith("tie_"):
        raise ValueError("one placement group requires a tie basis")
    if len(groups) > 1 and basis.startswith("tie_"):
        raise ValueError(f"{basis} requires all variants in one placement group")
    if expected_variants is not None and seen != expected_variants:
        missing = sorted(expected_variants - seen)
        unknown = sorted(seen - expected_variants)
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown: {', '.join(unknown)}")
        raise ValueError(
            "placement must rank every run exactly once; " + "; ".join(details)
        )
    return rating_pool_id, groups, basis


def placement_groups_from_ranks(
    runs: list[dict[str, Any]], rank_groups: list[str], basis: str
) -> list[list[str]]:
    aliases: dict[str, str] = {}
    variants: set[str] = set()
    for run in runs:
        slot = str(run.get("slot") or "").strip()
        variant_id = str(run.get("variant_id") or "").strip()
        if not slot or not variant_id:
            raise ValueError("every run must have a slot and variant_id")
        if variant_id in variants:
            raise ValueError(f"duplicate run variant: {variant_id}")
        variants.add(variant_id)
        for alias in (slot, variant_id):
            if alias in aliases and aliases[alias] != variant_id:
                raise ValueError(f"ambiguous rank entry: {alias}")
            aliases[alias] = variant_id
    groups: list[list[str]] = []
    for raw_group in rank_groups:
        tokens = [value.strip() for value in str(raw_group).split(",")]
        if any(not token for token in tokens):
            raise ValueError("rank groups cannot contain empty entries")
        unknown = next((token for token in tokens if token not in aliases), None)
        if unknown:
            raise ValueError(f"unknown rank entry: {unknown}")
        groups.append([aliases[token] for token in tokens])
    placement_record(
        {
            "rating_pool_id": "validation",
            "placement_v1": {"groups": groups, "basis": basis},
        },
        expected_variants=variants,
    )
    return groups


def build_record_from_grade(
    *,
    round_payload: dict[str, Any],
    roster: dict[str, Any],
    rubric: dict[str, Any],
    task_id: str,
    rating_pool_id: str,
    rank_groups: list[str],
    basis: str,
    shared_note: str | None,
) -> dict[str, Any]:
    if round_payload.get("status") != "completed":
        raise ValueError(f"round {round_payload['round_id']} is not completed yet")
    blocked_runs = []
    for run in round_payload.get("runs", []):
        classification = _classification_for_run(run)
        if classification["grade_blocked"]:
            blocked_runs.append(
                f"{public_reviewer_label(run['slot'])}: {classification['grade_block_reason'] or classification['review_status'] or 'unknown'}"
            )
    if blocked_runs:
        raise ValueError(
            "round contains interrupted or incomplete reviewers and should not be graded against model quality: "
            + "; ".join(blocked_runs)
        )
    basis = normalize_grade_basis(basis, rubric)
    indexed = variant_index(roster)
    runs = [deepcopy(run) for run in round_payload.get("runs", [])]
    groups = placement_groups_from_ranks(runs, rank_groups, basis)
    if not str(rating_pool_id or "").strip():
        raise ValueError("rating_pool_id is required")
    configured_pool_id = str(round_payload.get("rating_pool_id") or "").strip()
    if configured_pool_id and str(rating_pool_id).strip() != configured_pool_id:
        raise ValueError(
            f"rating_pool_id must match the configured pool: {configured_pool_id}"
        )
    for run in runs:
        variant_id = str(run["variant_id"])
        if variant_id not in indexed:
            raise ValueError(f"unknown roster variant: {variant_id}")
        run["grader_notes"] = shared_note or str(run["slot"])
        if run.get("cost_usd") is None:
            run["cost_usd"] = compute_cost_usd(
                indexed[variant_id], run.get("usage", {})
            )
    recorded_at = utc_now_iso()
    record = {
        "recorded_at": recorded_at,
        "round_id": round_payload["round_id"],
        "task_class": round_payload["task_class"],
        "task_id": task_id,
        "selection_mode": round_payload["selection_mode"],
        "rating_pool_id": str(rating_pool_id).strip(),
        "placement_v1": {"groups": groups, "basis": basis},
        "runs": runs,
    }
    if round_payload.get("reporting_pool"):
        record["reporting_pool"] = True
    repo_name = repo_name_from_round_payload(round_payload)
    if repo_name != "-":
        record["repo_name"] = repo_name
    return compact_benchmark_record(record)


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def update_elo(
    ratings: dict[str, float], groups: list[list[str]], k_factor: float
) -> dict[str, float]:
    ranks = {
        variant_id: rank for rank, group in enumerate(groups) for variant_id in group
    }
    if len(ratings) < 2 or set(ratings) != set(ranks):
        raise ValueError("Elo field must match a placement of at least two variants")
    weight = k_factor / (len(ratings) - 1)
    return {
        variant_id: rating
        + weight
        * sum(
            (
                0.5
                if ranks[variant_id] == ranks[opponent]
                else 1.0
                if ranks[variant_id] < ranks[opponent]
                else 0.0
            )
            - expected_score(rating, ratings[opponent])
            for opponent in ratings
            if opponent != variant_id
        )
        for variant_id, rating in ratings.items()
    }


def percentage(numerator: int | float, denominator: int | float) -> float | None:
    if float(denominator) <= 0.0:
        return None
    return round((float(numerator) / float(denominator)) * 100.0, 3)


def aggregate_records(
    roster: dict[str, Any],
    records: list[dict[str, Any]],
    operational_state: dict[str, Any],
    *,
    _include_rating_pools: bool = True,
) -> dict[str, Any]:
    settings = roster.get("settings", {})
    k_factor = float(settings.get("elo_k_factor", 24))
    summary: dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "report_settings": _report_settings_snapshot(settings),
        "task_classes": {},
    }
    for task_class in TASK_CLASSES:
        active = eligible_variants(roster, task_class)
        ratings = {variant["id"]: 1500.0 for variant in active}
        metrics: dict[str, dict[str, Any]] = {
            variant["id"]: {
                "sample_count": 0,
                "win_count": 0,
                "tie_count": 0,
                "loss_count": 0,
                "finding_opportunity_count": 0,
                "valid_finding_count": 0,
                "missed_bug_loss_count": 0,
                "low_quality_loss_count": 0,
                "low_quality_loss_reasons": {
                    basis: 0 for basis in LOW_QUALITY_LOSS_REASON_BASES
                },
                "elapsed_values": [],
                "cost_values": [],
                "total_token_values": [],
            }
            for variant in active
        }
        placed: list[tuple[dict[str, Any], tuple[str, list[list[str]], str]]] = []
        for record in (row for row in records if row.get("task_class") == task_class):
            record_runs = list(record.get("runs") or [])
            variants = {str(run.get("variant_id") or "") for run in record_runs}
            if len(variants) != len(record_runs):
                raise ValueError("placement record runs must have distinct variants")
            placement = placement_record(record, expected_variants=variants)
            if placement is not None:
                placed.append((record, placement))
        reporting = [item for item in placed if item[0].get("reporting_pool")]
        rating_pool_id = (reporting or placed)[-1][1][0] if placed else None
        recent_rounds: list[dict[str, Any]] = []
        indexed = variant_index(roster)
        for record, (pool_id, groups, basis) in placed:
            if pool_id != rating_pool_id:
                continue
            record_runs = list(record["runs"])
            ranks = {
                variant_id: rank
                for rank, group in enumerate(groups)
                for variant_id in group
            }
            before = {
                variant_id: ratings.get(variant_id, 1500.0) for variant_id in ranks
            }
            after = update_elo(before, groups, k_factor)
            ratings.update(after)
            participants: list[dict[str, Any]] = []
            for run in record_runs:
                variant_id = run["variant_id"]
                usage = run.get("usage", {})
                cost_usd = run.get("cost_usd")
                if cost_usd is None and variant_id in indexed:
                    cost_usd = compute_cost_usd(indexed[variant_id], usage)
                rank = ranks[variant_id]
                participants.append(
                    {
                        "variant_id": variant_id,
                        "rank": rank + 1,
                        "tied": len(groups[rank]) > 1,
                        "elo_before": before[variant_id],
                        "elo_after": after[variant_id],
                        "elo_delta": after[variant_id] - before[variant_id],
                        "elapsed_seconds": run.get("elapsed_seconds"),
                        "usage": deepcopy(usage),
                        "cost_usd": cost_usd,
                    }
                )
                if variant_id not in metrics:
                    continue
                bucket = metrics[variant_id]
                bucket["sample_count"] += 1
                if len(groups) == 1 or (rank == 0 and len(groups[0]) > 1):
                    bucket["tie_count"] += 1
                    outcome = "tie"
                elif rank == 0:
                    bucket["win_count"] += 1
                    outcome = "win"
                else:
                    bucket["loss_count"] += 1
                    outcome = "loss"
                if basis in BUG_OPPORTUNITY_BASES:
                    bucket["finding_opportunity_count"] += 1
                if (
                    outcome == "win" and basis in VALID_FINDING_WIN_BASES
                ) or basis in ALL_PARTICIPANTS_FOUND_BASES:
                    bucket["valid_finding_count"] += 1
                if outcome == "loss" and basis in MISSED_BUG_LOSS_BASES:
                    bucket["missed_bug_loss_count"] += 1
                if outcome == "loss" and basis in LOW_QUALITY_LOSS_BASES:
                    bucket["low_quality_loss_count"] += 1
                    bucket["low_quality_loss_reasons"][basis] += 1
                elapsed_seconds = run.get("elapsed_seconds")
                if isinstance(elapsed_seconds, (int, float)):
                    bucket["elapsed_values"].append(float(elapsed_seconds))
                total_tokens = run_total_tokens(run)
                if total_tokens > 0:
                    bucket["total_token_values"].append(total_tokens)
                if cost_usd is not None:
                    bucket["cost_values"].append(float(cost_usd))
            recent_rounds.append(
                {
                    "recorded_at": record.get("recorded_at"),
                    "review": review_label(task_class),
                    "repo": record.get("repo_name") or "-",
                    "task_id": record.get("task_id"),
                    "selection_mode": record.get("selection_mode"),
                    "rating_pool_id": pool_id,
                    "groups": deepcopy(groups),
                    "basis": basis,
                    "participants": participants,
                }
            )
        leaderboard = []
        for variant_id, bucket in metrics.items():
            sample_count = bucket["sample_count"]
            leaderboard.append(
                {
                    "variant_id": variant_id,
                    "variant_label": variant_id,
                    "sample_count": sample_count,
                    "elo": round(ratings[variant_id], 2),
                    "win_count": bucket["win_count"],
                    "tie_count": bucket["tie_count"],
                    "loss_count": bucket["loss_count"],
                    "wtl": f"{bucket['win_count']}/{bucket['tie_count']}/{bucket['loss_count']}",
                    "finding_opportunity_count": bucket["finding_opportunity_count"],
                    "valid_finding_count": bucket["valid_finding_count"],
                    "valid_finding_rate": percentage(
                        bucket["valid_finding_count"],
                        bucket["finding_opportunity_count"],
                    ),
                    "missed_bug_loss_count": bucket["missed_bug_loss_count"],
                    "missed_bug_loss_rate": percentage(
                        bucket["missed_bug_loss_count"],
                        bucket["finding_opportunity_count"],
                    ),
                    "low_quality_loss_count": bucket["low_quality_loss_count"],
                    "low_quality_loss_rate": percentage(
                        bucket["low_quality_loss_count"], sample_count
                    ),
                    "low_quality_loss_reasons": dict(
                        bucket["low_quality_loss_reasons"]
                    ),
                    "median_elapsed_seconds": round(
                        statistics.median(bucket["elapsed_values"]), 3
                    )
                    if bucket["elapsed_values"]
                    else None,
                    "median_total_tokens": round(
                        statistics.median(bucket["total_token_values"]), 1
                    )
                    if bucket["total_token_values"]
                    else None,
                    "median_cost_usd": round(
                        statistics.median(bucket["cost_values"]), 6
                    )
                    if bucket["cost_values"]
                    else None,
                }
            )
        leaderboard.sort(
            key=lambda row: (
                row["elo"],
                row["valid_finding_rate"] or 0.0,
                row["sample_count"],
            ),
            reverse=True,
        )
        task_summary = {
            "operational": operational_state["task_classes"][task_class],
            "rating_pool_id": rating_pool_id,
            "leaderboard": leaderboard,
            "recent_rounds": recent_rounds[-50:],
        }
        if _include_rating_pools:
            pool_ids = list(dict.fromkeys(pool_id for _, (pool_id, _, _) in placed))
            if rating_pool_id in pool_ids:
                pool_ids.remove(rating_pool_id)
                pool_ids.insert(0, rating_pool_id)
            task_summary["rating_pools"] = []
            for pool_id in pool_ids:
                if pool_id == rating_pool_id:
                    pool_summary = task_summary
                else:
                    pool_records = [
                        record
                        for record, placement in placed
                        if placement[0] == pool_id
                    ]
                    pool_summary = aggregate_records(
                        roster,
                        pool_records,
                        operational_state,
                        _include_rating_pools=False,
                    )["task_classes"][task_class]
                task_summary["rating_pools"].append(
                    {
                        "rating_pool_id": pool_id,
                        "reporting_pool": pool_id == rating_pool_id,
                        "leaderboard": pool_summary["leaderboard"],
                        "recent_rounds": pool_summary["recent_rounds"],
                    }
                )
        summary["task_classes"][task_class] = task_summary
    return summary


def write_reports(state_dir: Path, summary: dict[str, Any]) -> None:
    write_json(state_dir / SUMMARY_FILENAME, summary)
    report_settings = dict(summary.get("report_settings") or {})
    champion_min_samples = int(report_settings.get("promotion_min_samples", 20))
    champion_min_elo = float(report_settings.get("promotion_min_elo", 1550.0))
    champion_group_window = float(
        report_settings.get("promotion_champion_group_window", 25.0)
    )
    lines = ["# Review Arena Leaderboard", ""]
    for task_class in TASK_CLASSES:
        task = summary["task_classes"][task_class]
        op = task["operational"]
        champion_ids = list(op.get("champion_variant_ids") or [])
        lines.append(f"## {public_task_name(task_class)}")
        lines.append("")
        cooldowns = op.get("cooldowns") or {}
        if cooldowns:
            joined = "; ".join(
                f"`{variant_id}` until `{format_cooldown_until_for_display(entry.get('until'))}` (failures={entry.get('failure_count', 1)})"
                for variant_id, entry in sorted(cooldowns.items())
            )
            lines.append(f"- Cooldowns: {joined}")
        rating_pools = list(task.get("rating_pools") or [task])
        for pool in rating_pools:
            pool_id = pool.get("rating_pool_id") or "none"
            lines.append(f"### {format_markdown_inline(pool_id)}")
            lines.append("")
            lines.append(f"- Rating pool: {format_markdown_inline(pool_id)}")
            if pool.get("reporting_pool", True):
                lines.append(
                    f"- Champion: `{', '.join(champion_ids) if champion_ids else 'none'}`"
                )
            lines.append("")
            lines.append(
                "| model | elo | samples | W/T/L | found/opp | found % | missed % | low-quality % | sec | tok/job | cost/job |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for row in pool["leaderboard"]:
                if row["sample_count"] == 0:
                    continue
                lines.append(
                    f"| {format_markdown_inline(row['variant_label'])} | {format_decimal(row['elo'])} | {row['sample_count']} | {row['wtl']} | "
                    f"{row['valid_finding_count']}/{row['finding_opportunity_count']} | {format_decimal(row['valid_finding_rate'])} | "
                    f"{format_decimal(row['missed_bug_loss_rate'])} | {format_decimal(row['low_quality_loss_rate'])} | {format_decimal(row['median_elapsed_seconds'])} | "
                    f"{format_compact_tokens(row['median_total_tokens'])} | {format_cost_cents(row['median_cost_usd'])} |"
                )
            lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        f"- Champion gate: `sample_count >= {champion_min_samples}` and `elo >= {format_decimal(champion_min_elo)}`. Champion group membership is `<= {format_decimal(champion_group_window)}` Elo behind the top eligible model."
    )
    lines.append(
        "- `found %` is valid findings over bug-present opportunities, including all participants on `tie_both_useful` and multi-finding coverage wins. `missed %` is missed-bug losses over bug-present opportunities. `low-quality %` is validity, false-positive, hallucinated, fringe, or scope-bloat losses over all samples."
    )
    lines.append(
        "- `tok/job` is uncached input (cache writes included) plus output, with reasoning folded into output; cache reads are excluded. `cost/job` is the authoritative per-run cost."
    )
    lines.append("")
    lines.append("## round history")
    lines.append("")
    lines.append("| review | repo | rating pool | placements and Elo |")
    lines.append("|---|---|---|---|")
    history = sorted(
        (
            row
            for task in summary["task_classes"].values()
            for row in task.get("recent_rounds", [])
        ),
        key=lambda row: str(row.get("recorded_at") or ""),
        reverse=True,
    )[:10]
    for row in history:
        participants = {
            participant["variant_id"]: participant
            for participant in row["participants"]
        }
        placements = []
        for rank, group in enumerate(row["groups"], start=1):
            placements.append(
                f"{rank}. "
                + " = ".join(
                    f"{format_markdown_inline(variant_id)} {format_decimal(participants[variant_id]['elo_before'])} -> {format_decimal(participants[variant_id]['elo_after'])} ({format_signed_decimal(participants[variant_id]['elo_delta'])})"
                    for variant_id in group
                )
            )
        lines.append(
            f"| {format_markdown_inline(row['review'])} | {format_markdown_inline(row['repo'])} | {format_markdown_inline(row['rating_pool_id'])} | {'; '.join(placements)} |"
        )
    lines.append("")
    _atomic_write_text(state_dir / "leaderboard.md", "\n".join(lines) + "\n")


def promote(
    roster: dict[str, Any], summary: dict[str, Any], operational_state: dict[str, Any]
) -> dict[str, Any]:
    settings = roster["settings"]
    min_samples = int(settings["promotion_min_samples"])
    min_elo = float(settings.get("promotion_min_elo", 1550.0))
    champion_group_window = float(
        settings.get(
            "promotion_champion_group_window",
            settings.get("promotion_min_elo_lead", 25.0),
        )
    )
    state = deepcopy(operational_state)
    state["generated_at"] = utc_now_iso()
    for task_class in TASK_CLASSES:
        leaderboard = summary["task_classes"][task_class]["leaderboard"]
        slot = state["task_classes"][task_class]
        slot["champion_variant_id"] = None
        slot["champion_variant_ids"] = []
        slot["probation_variant_ids"] = []
        if not leaderboard:
            continue
        eligible = [
            row
            for row in leaderboard
            if row["sample_count"] >= min_samples and float(row["elo"]) >= min_elo
        ]
        champion_id_set: set[str] = set()
        if not eligible:
            slot["probation_variant_ids"] = _probation_variant_ids(
                leaderboard=leaderboard,
                champion_ids=champion_id_set,
                settings=settings,
            )
            continue
        top_eligible = eligible[0]
        champion_group = [
            row
            for row in eligible
            if (float(top_eligible["elo"]) - float(row["elo"])) <= champion_group_window
        ]
        champion_ids = [row["variant_id"] for row in champion_group]
        champion_id_set = set(champion_ids)
        slot["champion_variant_id"] = top_eligible["variant_id"]
        slot["champion_variant_ids"] = champion_ids
        slot["probation_variant_ids"] = _probation_variant_ids(
            leaderboard=leaderboard,
            champion_ids=champion_id_set,
            settings=settings,
        )
    return state
