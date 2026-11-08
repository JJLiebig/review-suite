#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import os
import select
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
    emit_error,
    emit_result,
    format_command,
    lens_model_config,
    resolve_cd_path,
    resolve_repo_root,
    run_codex,
)

_STD_INPUT_HANDLE = -10
_FILE_TYPE_DISK = 0x0001
_FILE_TYPE_PIPE = 0x0003


def build_parser() -> argparse.ArgumentParser:
    parser = AxiArgumentParser(description="Review Plan structural review wrapper.")
    parser.add_argument("--input-file")
    parser.add_argument("--input-text")
    parser.add_argument("--cd")
    parser.add_argument("--skip-git-repo-check", action="store_true")
    parser.add_argument("--wsl", action="store_true")
    return parser


def _help_command() -> str:
    return format_command(
        [sys.executable, str(launcher_script_path(__file__)), "--help"]
    )


def resolve_review_root(args: argparse.Namespace) -> Path:
    skip_repo_check = bool(args.skip_git_repo_check)
    if args.cd:
        if skip_repo_check:
            return resolve_cd_path(args.cd)
        return resolve_repo_root(args.cd)
    if args.input_file:
        parent = Path(args.input_file).resolve().parent
        if skip_repo_check:
            return parent
        try:
            return resolve_repo_root(parent)
        except ValueError:
            try:
                return resolve_repo_root(None)
            except ValueError:
                return parent
    if skip_repo_check:
        return Path.cwd().resolve(strict=False)
    return resolve_repo_root(None)


def _stdin_has_data() -> bool:
    if sys.stdin.isatty():
        return False
    if os.name != "nt":
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        return bool(ready)
    handle = ctypes.windll.kernel32.GetStdHandle(_STD_INPUT_HANDLE)
    if handle in (0, -1):
        return False
    file_type = ctypes.windll.kernel32.GetFileType(handle)
    if file_type == _FILE_TYPE_DISK:
        return True
    if file_type != _FILE_TYPE_PIPE:
        return False
    available = ctypes.c_ulong()
    ok = ctypes.windll.kernel32.PeekNamedPipe(
        handle, None, 0, None, ctypes.byref(available), None
    )
    if ok == 0:
        return False
    return bool(available.value)


def load_plan_input(args: argparse.Namespace) -> tuple[str, str]:
    if args.input_file and args.input_text:
        raise ValueError("use either --input-file or --input-text")
    if args.input_file:
        path = Path(args.input_file)
        return path.read_text(encoding="utf-8"), f"file:{path.resolve()}"
    if args.input_text:
        return args.input_text, "inline-text"
    default_plan_path = Path.cwd() / "task_plan.md"
    if default_plan_path.exists():
        return default_plan_path.read_text(
            encoding="utf-8"
        ), f"default:{default_plan_path.resolve()}"
    if _stdin_has_data():
        text = sys.stdin.read()
        if text.strip():
            return text, "stdin"
    raise ValueError(
        "review-plan requires task_plan.md, --input-file, --input-text, or stdin content"
    )


def build_prompt(plan_text: str) -> str:
    return (
        "Review this implementation plan against the repository.\n\n"
        "Before recommending a solution shape:\n"
        "- identify the root cause, canonical owner, and invariants that must hold\n"
        "- find materially distinct credible solution shapes supported by the repository\n"
        "- compare those shapes; if one solution is clearly the only credible one, say so "
        "instead of inventing alternatives\n"
        "- recommend the best shape in this order: contract correctness; canonical "
        "ownership; fewer durable concepts and less coupling; justified blast radius and "
        "validation; then diff minimality only as a tie-breaker\n\n"
        "End with exactly one verdict: PROCEED, REVISE, or RETHINK.\n"
        "Support it with concise repository evidence and actionable plan changes.\n"
        "Use PROCEED when the proposed shape is best, REVISE when bounded plan edits are "
        "enough, and RETHINK when the root cause, owner, or solution shape should change.\n"
        "Do not give style-only feedback or treat RETHINK as scope authority.\n\n"
        "=== BEGIN PLAN ===\n"
        f"{plan_text.strip()}\n"
        "=== END PLAN ==="
    )


def main() -> int:
    parser = build_parser()
    try:
        args = parser.parse_args()
        review_root = resolve_review_root(args)
        plan_text, _input_source = load_plan_input(args)
        model_config = lens_model_config("review-plan")
        result = run_codex(
            tool_name="review-plan",
            prompt=build_prompt(plan_text),
            model=model_config.model,
            reasoning_effort=model_config.reasoning_effort,
            service_tier=model_config.service_tier,
            review_root=review_root,
            progress_interval_seconds=DEFAULT_PROGRESS_INTERVAL_SECONDS,
            timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
            allow_unsafe_windows_wsl_fallback=bool(args.wsl),
            skip_git_repo_check=bool(args.skip_git_repo_check),
        )
        return emit_result(
            tool_name="review-plan",
            result=result,
        )
    except ValueError as exc:
        return emit_error(
            str(exc),
            status="usage_error",
            help_items=[_help_command()],
        )


if __name__ == "__main__":
    raise SystemExit(main())
