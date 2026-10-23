#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

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
    DEFAULT_TIMEOUT_SECONDS,
    effective_base_ref,
    emit_error,
    emit_result,
    format_command,
    lens_model_config,
    resolve_repo_root,
    resolve_ref,
    run_codex,
    validated_linear_review_range,
    write_text,
)
from review_suite_local import ensure_clean_git_worktree
from review_suite_core.config import override_model_config
from review_suite_core.model_labels import SUPPORTED_REASONING_EFFORTS

STATIC_CLEANUP_LIMIT = 20
STATIC_CLEANUP_MIN_CONFIDENCE = 90
VULTURE_OUTPUT_RE = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+): (?P<message>.+) "
    r"\((?P<confidence>\d+)% confidence(?:, (?P<size>\d+) lines?)?\)$"
)
DIFF_FILE_RE = re.compile(r"^\+\+\+ b/(?P<path>.+)$")
DIFF_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,(?P<count>\d+))? @@")
UNUSABLE_REVIEW_MARKERS = (
    "could not inspect",
    "couldn't inspect",
    "couldn't perform the review",
    "cannot inspect",
    "local process execution is blocked",
    "powershell and node repl failed",
    "windows sandbox failed",
    "spawn setup refresh",
    "no mcp workspace resource",
)


@dataclass(frozen=True)
class StaticCleanupSuggestion:
    path: str
    line: int
    message: str
    confidence: int


@dataclass(frozen=True)
class StaticCleanupScan:
    process: subprocess.Popen[str]
    changed_lines: dict[str, set[int]]
    stdout_path: Path
    stderr_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = AxiArgumentParser(
        description="Review Deslop post-implementation review wrapper."
    )
    parser.add_argument("--cd")
    parser.add_argument("--base", help="Override the detected default branch ref.")
    parser.add_argument("--commit", nargs="+")
    parser.add_argument("--focus")
    parser.add_argument("--review-brief", help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument(
        "--reasoning", choices=sorted(SUPPORTED_REASONING_EFFORTS), help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--conformance-only", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--wsl", action="store_true")
    parser.add_argument("--output-only", action="store_true", help=argparse.SUPPRESS)
    return parser


def _help_command() -> str:
    return format_command(
        [sys.executable, str(launcher_script_path(__file__)), "--help"]
    )


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


def build_prompt(
    *,
    base: str | None,
    commit: str | None,
    commit_end: str | None,
    focus: str | None,
    review_brief: str | None = None,
    conformance_only: bool = False,
) -> str:
    focus_block = (
        f"\nPay extra attention to this focus area:\n- {focus.strip()}\n"
        if focus
        else ""
    )
    if commit and commit_end:
        target_block = (
            f"Review the commit range `{commit}..{commit_end}` in the current repository.\n"
            "Inspect that range and adjacent touched paths, not the whole branch diff.\n\n"
        )
    elif commit:
        target_block = (
            f"Review commit `{commit}` in the current repository.\n"
            "Inspect that commit and adjacent touched paths, not the whole branch diff.\n\n"
        )
    else:
        target_block = (
            f"Review the current repository changes against base branch `{base}`.\n\n"
        )
    conformance = (
        "Compare the implementation with this frozen review brief:\n"
        f"<review_brief>\n{review_brief.strip()}\n</review_brief>\n"
        "Report CONFORMS unless the implementation materially changes the brief's goal or constraints; otherwise report MATERIALLY_DRIFTED.\n\n"
        if review_brief
        else "No frozen review brief is available; report NOT_APPLICABLE for conformance.\n\n"
    )
    cleanup = (
        "Skip cleanup; this is the post-edit conformance rerun.\n"
        if conformance_only
        else (
            "For cleanup, review over-engineering only, not correctness. "
            "One line per finding: `<file>:L<line>: <tag>: <what to cut>. <replacement>.` "
            "Tags: `delete` dead or speculative code; `stdlib` reinvented standard library; "
            "`native` platform replacements; `yagni` unused abstractions or flexibility; "
            "`shrink` same behavior in fewer lines.\n"
        )
    )
    return (
        target_block
        + conformance
        + cleanup
        + "Do not redesign ownership, add abstractions, broaden scope, or change behavior.\n"
        + focus_block
        + "\nBegin with exactly `Conformance: CONFORMS`, `Conformance: MATERIALLY_DRIFTED`, or `Conformance: NOT_APPLICABLE`.\n"
        + "Report only concrete cleanup findings. Skip style-only comments.\n"
        + "Always finish with exactly `Review decision: clean` or `Review decision: findings`."
    )


def _git_output(
    review_root: Path, args: list[str], *, timeout_seconds: int = 20
) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=review_root,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return ""
    if proc.returncode != 0:
        return ""
    return str(proc.stdout or "")


def _git_lines(
    review_root: Path, args: list[str], *, timeout_seconds: int = 20
) -> list[str]:
    return [
        line.strip()
        for line in _git_output(
            review_root, args, timeout_seconds=timeout_seconds
        ).splitlines()
        if line.strip()
    ]


def _changed_python_lines(*, review_root: Path, diff_range: str) -> dict[str, set[int]]:
    diff = _git_output(
        review_root,
        ["diff", "--unified=0", "--find-renames", diff_range, "--", "*.py"],
    )
    changed: dict[str, set[int]] = {}
    current_path: str | None = None
    for line in diff.splitlines():
        file_match = DIFF_FILE_RE.match(line)
        if file_match:
            path = _normalize_repo_path(file_match.group("path"))
            current_path = path if path.endswith(".py") else None
            if current_path:
                changed.setdefault(current_path, set())
            continue
        if not current_path:
            continue
        hunk_match = DIFF_HUNK_RE.match(line)
        if not hunk_match:
            continue
        start = int(hunk_match.group("start"))
        count = int(hunk_match.group("count") or "1")
        if count > 0:
            changed[current_path].update(range(start, start + count))
    return {path: lines for path, lines in changed.items() if lines}


def _normalize_repo_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _module_is_available(command: list[str]) -> bool:
    try:
        proc = subprocess.run(
            [*command, "--version"],
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return False
    return proc.returncode == 0


def _ensure_vulture_command() -> list[str] | None:
    local_command = [sys.executable, "-m", "vulture"]
    if _module_is_available(local_command):
        return local_command
    path_command = shutil.which("vulture")
    if path_command and _module_is_available([path_command]):
        return [path_command]
    return None


def _start_static_cleanup_scan(
    *,
    review_root: Path,
    base: str | None,
    commit: str | None,
    commit_end: str | None,
) -> StaticCleanupScan | None:
    if commit and not commit_end:
        return None
    if commit_end and resolve_ref(review_root, commit_end) != resolve_ref(
        review_root, "HEAD"
    ):
        return None
    diff_range = (
        f"{commit}..{commit_end}" if commit else f"{base}...HEAD" if base else None
    )
    if not diff_range:
        return None
    changed_lines = _changed_python_lines(
        review_root=review_root, diff_range=diff_range
    )
    if not changed_lines:
        return None
    command = _ensure_vulture_command()
    if not command:
        return None
    stdout_handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="review-deslop-vulture-stdout-",
        suffix=".txt",
        delete=False,
    )
    stderr_handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="review-deslop-vulture-stderr-",
        suffix=".txt",
        delete=False,
    )
    try:
        process = subprocess.Popen(
            [
                *command,
                *sorted(changed_lines),
                "--min-confidence",
                str(STATIC_CLEANUP_MIN_CONFIDENCE),
            ],
            cwd=review_root,
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
    except OSError, subprocess.SubprocessError:
        stdout_handle.close()
        stderr_handle.close()
        _unlink_quietly(Path(stdout_handle.name))
        _unlink_quietly(Path(stderr_handle.name))
        return None
    finally:
        stdout_handle.close()
        stderr_handle.close()
    return StaticCleanupScan(
        process=process,
        changed_lines=changed_lines,
        stdout_path=Path(stdout_handle.name),
        stderr_path=Path(stderr_handle.name),
    )


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _stop_static_cleanup_scan(scan: StaticCleanupScan) -> None:
    try:
        if scan.process.poll() is None:
            scan.process.terminate()
            try:
                scan.process.wait(timeout=1)
            except OSError, subprocess.TimeoutExpired:
                scan.process.kill()
    finally:
        _unlink_quietly(scan.stdout_path)
        _unlink_quietly(scan.stderr_path)


def _collect_static_cleanup_scan(
    scan: StaticCleanupScan | None,
) -> list[StaticCleanupSuggestion]:
    if not scan:
        return []
    if scan.process.poll() is None:
        _stop_static_cleanup_scan(scan)
        return []
    try:
        scan.process.wait(timeout=1)
        output = "\n".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in (scan.stdout_path, scan.stderr_path)
            if path.exists()
        )
    except OSError, subprocess.SubprocessError:
        return []
    finally:
        _unlink_quietly(scan.stdout_path)
        _unlink_quietly(scan.stderr_path)
    return _parse_static_cleanup_output(output, scan.changed_lines)


def _parse_static_cleanup_output(
    output: str, changed_lines: dict[str, set[int]]
) -> list[StaticCleanupSuggestion]:
    suggestions: list[StaticCleanupSuggestion] = []
    for line in output.splitlines():
        match = VULTURE_OUTPUT_RE.match(line.strip())
        if not match:
            continue
        path = _normalize_repo_path(match.group("path"))
        line_number = int(match.group("line"))
        if line_number not in changed_lines.get(path, set()):
            continue
        confidence = int(match.group("confidence"))
        message = match.group("message").strip()
        if confidence >= 100 or (
            confidence >= 90 and "unused import" in message.lower()
        ):
            suggestions.append(
                StaticCleanupSuggestion(
                    path=path,
                    line=line_number,
                    message=message,
                    confidence=confidence,
                )
            )
        if len(suggestions) >= STATIC_CLEANUP_LIMIT:
            break
    return suggestions


def _render_static_cleanup_items(
    suggestions: list[StaticCleanupSuggestion],
) -> list[str]:
    return [
        f"Low - {suggestion.path}:{suggestion.line} - {suggestion.message}. Fix: {_static_cleanup_fix_suggestion(suggestion.message)}"
        for suggestion in suggestions[:STATIC_CLEANUP_LIMIT]
    ]


def _static_cleanup_fix_suggestion(message: str) -> str:
    lowered = message.lower()
    if "unused import" in lowered:
        return "Remove the unused import."
    if "unreachable code" in lowered:
        return "Remove the unreachable code."
    if "unused variable" in lowered:
        return "Remove the unused variable or mark it intentionally unused."
    return "Remove the unused code if it is not required."


def _render_static_cleanup_section(suggestions: list[StaticCleanupSuggestion]) -> str:
    items = _render_static_cleanup_items(suggestions)
    if not items:
        return ""
    return "Static cleanup suggestions:\n" + "\n".join(f"- {item}" for item in items)


def _with_static_cleanup_output(
    result: dict[str, object], suggestions: list[StaticCleanupSuggestion]
) -> dict[str, object]:
    section = _render_static_cleanup_section(suggestions)
    if not section or _result_returncode(result) != 0:
        return result
    body = str(result.get("final_message") or "").strip()
    if not body:
        return result
    protocol_decision = _deslop_protocol_decision(result)
    conformance = ""
    if protocol_decision:
        conformance, _, body = body.partition("\n")
        body = body.rpartition("\n")[0].strip()
    elif _deslop_output_clean(result) and "conformance:" not in body.lower():
        body = "No reviewer findings."
    prefix = f"{conformance}\n\n" if conformance else ""
    body = f"{prefix}{section}\n\nDeslop Results:\n{body}\nReview decision: findings"
    return {**result, "final_message": body}


def _review_output_text(result: dict[str, object]) -> str:
    return "\n".join(
        str(result.get(key) or "") for key in ("final_message", "stdout", "stderr")
    )


def _deslop_output_unusable(result: dict[str, object]) -> bool:
    text = (
        _review_output_text(result)
        .lower()
        .replace(chr(0x2019), "'")
        .replace(chr(0xFFFD), "'")
    )
    return any(marker in text for marker in UNUSABLE_REVIEW_MARKERS)


def _deslop_protocol_decision(result: dict[str, object]) -> str | None:
    lines = [
        line.strip()
        for line in str(result.get("final_message") or "").splitlines()
        if line.strip()
    ]
    conformance_lines = [line for line in lines if line.startswith("Conformance: ")]
    decision_lines = [line for line in lines if line.startswith("Review decision: ")]
    if lines[:1] != conformance_lines or len(decision_lines) != 1:
        return None
    if conformance_lines[0].removeprefix("Conformance: ") not in {
        "CONFORMS",
        "MATERIALLY_DRIFTED",
        "NOT_APPLICABLE",
    }:
        return None
    decision = decision_lines[0].removeprefix("Review decision: ")
    if decision not in {"clean", "findings"} or lines[-1] != decision_lines[0]:
        return None
    return decision


def _deslop_output_clean(result: dict[str, object]) -> bool:
    text = (
        _review_output_text(result).replace(chr(0x2019), "'").replace(chr(0xFFFD), "'")
    )
    compact = " ".join(text.lower().split()).rstrip(".")
    return _deslop_protocol_decision(result) == "clean" or compact in {
        "no findings",
        "no concrete findings",
    }


def _with_effective_returncode(result: dict[str, object]) -> dict[str, object]:
    if result.get("timed_out"):
        return {**result, "returncode": int(result.get("returncode") or 1)}
    if _deslop_protocol_decision(result):
        return {**result, "returncode": 0}
    if _deslop_output_unusable(result):
        return {**result, "returncode": int(result.get("returncode") or 1)}
    if _deslop_output_clean(result):
        return {**result, "returncode": 0}
    return result


def _result_returncode(result: dict[str, object]) -> int:
    value = result.get("returncode")
    if value is None:
        return 1
    return int(value)


def emit_output_only(*, tool_name: str, result: dict[str, object]) -> int:
    result = _with_effective_returncode(result)
    returncode = _result_returncode(result)
    body = str(result.get("final_message") or "").strip()
    if not body and returncode != 0:
        body = (
            f"{tool_name} run timed out"
            if result.get("timed_out")
            else f"{tool_name} run failed"
        )
    if body:
        write_text(body)
    return returncode


def main() -> int:
    parser = build_parser()
    try:
        args = parser.parse_args()
        commit, commit_end = normalize_commit_spec(args.commit)
        if commit and args.base is not None:
            raise ValueError("use either --base or --commit")
        review_root = resolve_repo_root(args.cd)
        model_config = override_model_config(
            lens_model_config("review-deslop"),
            {"model": args.model, "reasoning": args.reasoning},
        )
        if commit and commit_end:
            validated_linear_review_range(
                review_root,
                commit,
                commit_end,
                label="commit-range deslop review",
            )
            ensure_clean_git_worktree(review_root)
            prompt_base = None
        elif commit:
            prompt_base = None
        else:
            prompt_base = str(effective_base_ref(review_root, args.base)["base"])
            ensure_clean_git_worktree(review_root)
        static_scan = (
            None
            if args.conformance_only
            else _start_static_cleanup_scan(
                review_root=review_root,
                base=prompt_base,
                commit=commit,
                commit_end=commit_end,
            )
        )
        prompt = build_prompt(
            base=prompt_base,
            commit=commit,
            commit_end=commit_end,
            focus=args.focus,
            review_brief=args.review_brief,
            conformance_only=bool(args.conformance_only),
        )
        try:
            result = run_codex(
                tool_name="review-deslop",
                prompt=prompt,
                model=model_config.model,
                reasoning_effort=model_config.reasoning_effort,
                service_tier=model_config.service_tier,
                review_root=review_root,
                progress_interval_seconds=DEFAULT_PROGRESS_INTERVAL_SECONDS,
                timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
                allow_unsafe_windows_wsl_fallback=bool(args.wsl),
            )
            result = _with_effective_returncode(result)
            if static_scan is not None:
                static_suggestions = _collect_static_cleanup_scan(static_scan)
                static_scan = None
                result = _with_static_cleanup_output(result, static_suggestions)
        finally:
            if static_scan is not None:
                _stop_static_cleanup_scan(static_scan)
        if args.output_only:
            return emit_output_only(tool_name="review-deslop", result=result)
        return emit_result(
            tool_name="review-deslop",
            result=_with_effective_returncode(result),
        )
    except ValueError as exc:
        return emit_error(
            str(exc),
            status="usage_error",
            help_items=[_help_command()],
        )


if __name__ == "__main__":
    raise SystemExit(main())
