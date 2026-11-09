#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import ExitStack
import sys
from pathlib import Path

_previous_dont_write_bytecode = sys.dont_write_bytecode
sys.dont_write_bytecode = True

from review_suite_runtime_bootstrap import (
    bootstrap_runtime,
    launcher_script_path,
)

bootstrap_runtime(__file__)
sys.dont_write_bytecode = _previous_dont_write_bytecode

from review_suite_core import (
    AxiArgumentParser,
    DEFAULT_PROGRESS_INTERVAL_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    EFFECTIVE_BASE_METADATA_KEYS,
    classify_delta_recommendation,
    current_head,
    diff_stats,
    effective_base_ref,
    emit_error,
    emit_result,
    format_command,
    has_committed_diff,
    inspect_workflow_status,
    is_ancestor,
    lens_model_config,
    merge_base,
    record_review_anchor,
    resolve_ref,
    resolve_repo_root,
    run_codex_review,
    validated_linear_review_range,
)
from review_suite_local import (
    build_correctness_review_contract,
    default_state_dir,
    ensure_clean_git_worktree,
)


def build_parser() -> argparse.ArgumentParser:
    parser = AxiArgumentParser(
        description="Review the interdiff after a reviewer-driven fix pass."
    )
    parser.add_argument("--cd")
    parser.add_argument("--base", help="Override the detected default branch ref.")
    parser.add_argument("--since")
    parser.add_argument("--reviewed-base", help="Immutable merge-base of the previously reviewed PR patch.")
    parser.add_argument("--note")
    parser.add_argument("--note-file")
    parser.add_argument(
        "--state-dir", default=None, help=argparse.SUPPRESS
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--wsl", action="store_true")
    return parser


def _help_command() -> str:
    return format_command(
        [sys.executable, str(launcher_script_path(__file__)), "--help"]
    )


def load_followup_note(
    *, note: str | None, note_file: str | None, review_root: Path | None = None
) -> str:
    if note is not None and note_file is not None:
        raise ValueError("use either --note or --note-file")
    if note_file is not None:
        try:
            path = Path(note_file)
            if not path.is_absolute() and review_root is not None:
                path = review_root / path
            payload = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(str(exc)) from exc
    elif note is not None:
        payload = note
    else:
        raise ValueError("review-followup requires --note or --note-file")
    text = payload.strip()
    if not text:
        raise ValueError("follow-up note must not be empty")
    return text


def _followup_guard_error(*, recommendation: str, note: str) -> str:
    if recommendation == "coherence-review":
        return f"{note} Run review.py --status and use coherence/reset or a full review, or pass --force to override."
    if recommendation == "full-review":
        return f"{note} Run review.py --status and use the appropriate full-diff review lane, or pass --force to override."
    return note


def resolve_since_head(
    *,
    explicit_since: str | None,
    state_dir: Path,
    review_cwd: Path,
    base: str,
    force: bool,
) -> str:
    if explicit_since:
        since_head = str(explicit_since).strip()
        if not since_head:
            raise ValueError("--since must not be empty")
        if force:
            return since_head
        status = inspect_workflow_status(
            state_dir=state_dir, review_cwd=review_cwd, base=base
        )
        if str(status.get("reason") or "") in {
            "branch_review_pressure_exceeded",
            "followup_cycle_limit_exceeded",
        }:
            raise ValueError(
                _followup_guard_error(
                    recommendation=str(status.get("recommendation") or ""),
                    note=str(
                        status.get("note")
                        or "follow-up review is not appropriate for this branch state."
                    ),
                )
            )
        resolved_since = resolve_ref(review_cwd, since_head)
        if not is_ancestor(review_cwd, resolved_since, "HEAD"):
            raise ValueError(
                "--since must resolve to an ancestor of HEAD for a non-forced follow-up review. "
                "Use review.py --status to choose the right lane, or pass --force to override."
            )
        decision = classify_delta_recommendation(
            diff_stats(review_cwd, resolved_since, "HEAD")
        )
        if str(decision.get("recommendation") or "") != "review-followup":
            raise ValueError(
                _followup_guard_error(
                    recommendation=str(decision.get("recommendation") or ""),
                    note=str(
                        decision.get("note")
                        or "follow-up review is not appropriate for this delta."
                    ),
                )
            )
        return resolved_since
    status = inspect_workflow_status(
        state_dir=state_dir, review_cwd=review_cwd, base=base
    )
    since_head = str(status.get("last_reviewed_head") or "").strip()
    if not since_head:
        raise ValueError(
            "review-followup requires --since or an existing recorded review anchor for this branch"
        )
    recommendation = str(status.get("recommendation") or "")
    if recommendation == "none":
        raise ValueError(
            "current HEAD already matches the latest recorded review anchor"
        )
    if recommendation != "review-followup" and not force:
        raise ValueError(
            _followup_guard_error(
                recommendation=recommendation,
                note=str(
                    status.get("note")
                    or "follow-up review is not appropriate for this branch state."
                ),
            )
        )
    return since_head


def build_followup_prompt(
    *, since_head: str, head: str, note: str, target_label: str | None = None
) -> str:
    resolved_target_label = target_label or f"interdiff `{since_head}..{head}`"
    return (
        "Review this follow-up diff for correctness and regression risk.\n"
        f"The review target is {resolved_target_label}.\n"
        "This is a fix pass after earlier review feedback. Focus on whether the review target resolves the underlying problem and whether adjacent invariants still hold.\n\n"
        "Fixer root-cause note:\n"
        f"{note}\n\n"
        f"{build_correctness_review_contract()}"
    )


def _record_anchor_warning(exc: Exception) -> None:
    print(
        f"[review-suite] WARNING: failed to record workflow anchor: {exc}",
        file=sys.stderr,
        flush=True,
    )


from review_suite_core.state_paths import state_directory


def main() -> int:
    parser = build_parser()
    scope = ExitStack()
    try:
        args = parser.parse_args()
        scope.enter_context(state_directory(args.state_dir))
        if args.state_dir is None:
            args.state_dir = str(default_state_dir())
        review_root = resolve_repo_root(args.cd)
        state_dir = Path(args.state_dir)
        note_text = load_followup_note(
            note=args.note, note_file=args.note_file, review_root=review_root
        )
        base_info = effective_base_ref(review_root, args.base)
        branch_base = str(base_info["base"])
        requested_base = str(base_info["requested_base"])
        since_head = resolve_since_head(
            explicit_since=args.since,
            state_dir=state_dir,
            review_cwd=review_root,
            base=branch_base,
            force=bool(args.force or args.reviewed_base),
        )
        head = current_head(review_root)
        if since_head == head:
            raise ValueError(
                "current HEAD already matches the requested follow-up anchor"
            )
        if not args.reviewed_base:
            validated_linear_review_range(
                review_root, since_head, head, label="native follow-up review"
            )
        if not args.reviewed_base and not has_committed_diff(review_root, since_head, head):
            raise ValueError(
                f"follow-up review found no committed diff between `{since_head}` and `{head}`. "
                "Commit the intended fix or use the appropriate full review lane."
            )
        ensure_clean_git_worktree(review_root)
        prompt = build_followup_prompt(
            since_head=since_head,
            head=head,
            note=note_text,
        )
        if args.reviewed_base:
            from review_suite_core.patch_identity import delta_prompt
            reviewed_base = resolve_ref(review_root, args.reviewed_base)
            since_head = resolve_ref(review_root, since_head)
            prompt = delta_prompt(reviewed_base, since_head, merge_base(review_root, branch_base), head) + "\n" + note_text
        model_config = lens_model_config("review-followup", state_dir=state_dir)
        result = run_codex_review(
            tool_name="review-followup",
            prompt=prompt,
            model=model_config.model,
            reasoning_effort=model_config.reasoning_effort,
            service_tier=model_config.service_tier,
            title="review-followup",
            review_root=review_root,
            base=None if args.reviewed_base else since_head,
            progress_interval_seconds=DEFAULT_PROGRESS_INTERVAL_SECONDS,
            timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
            allow_unsafe_windows_wsl_fallback=bool(args.wsl),
        )
        if int(result["returncode"]) == 0:
            try:
                output_ref = (
                    f"session:{result['session_id']}"
                    if result.get("session_id")
                    else None
                )
                branch_scope = {
                    "branch_base": branch_base,
                    "merge_base": merge_base(review_root, branch_base),
                }
                if requested_base != branch_base:
                    branch_scope["requested_base"] = requested_base
                for key in EFFECTIVE_BASE_METADATA_KEYS:
                    if key in base_info:
                        branch_scope[key] = base_info[key]
                record_review_anchor(
                    state_dir=state_dir,
                    review_cwd=review_root,
                    lane="review-followup",
                    base=branch_base,
                    review_scope=(
                        {
                            "commit": since_head,
                            "commit_end": head,
                            "base": since_head,
                            "target_label": f"interdiff `{since_head}..{head}`",
                            **branch_scope,
                        }
                    ),
                    reviewed_head=head,
                    output_refs=[output_ref] if output_ref else [],
                    note=note_text,
                )
            except Exception as exc:  # pragma: no cover - warning path only
                _record_anchor_warning(exc)
        return emit_result(
            tool_name="review-followup",
            result=result,
        )
    except ValueError as exc:
        return emit_error(
            str(exc),
            status="usage_error",
            help_items=[_help_command()],
        )
    finally:
        scope.close()


if __name__ == "__main__":
    raise SystemExit(main())
