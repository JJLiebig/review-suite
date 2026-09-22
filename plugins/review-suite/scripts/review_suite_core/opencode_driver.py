from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


TERMINAL_REVIEW_RESULT_PREFIX = "Review result:"
REVIEW_METADATA_PREFIX = "[review-suite] opencode-metadata: "
REVIEW_EXPORT_TIMEOUT_SECONDS = 120

OPENCODE_ERROR_CLASS_CAPACITY = "capacity"
OPENCODE_ERROR_CLASS_UNAVAILABLE = "unavailable"
OPENCODE_ERROR_CLASS_FAILED = "failed"
OPENCODE_ERROR_CLASS_ADAPTER = "adapter"
OPENCODE_ERROR_MESSAGE_LIMIT = 500

_CAPACITY_ERROR_MARKERS = (
    "at capacity",
    "capacity",
    "rate limit",
    "rate-limit",
    "rate_limit",
    "ratelimit",
    "too many requests",
    "quota",
    "usage limit",
    "resource_exhausted",
    "resource exhausted",
    "overloaded",
    "throttl",
    "limit reached",
    "limit exceeded",
)
_UNAVAILABLE_ERROR_MARKERS = (
    "model not found",
    "unknown model",
    "invalid model",
    "no such model",
    "model unavailable",
    "model is not available",
    "model is unavailable",
    "unsupported model",
    "unsupported",
    "not supported",
    "does not exist",
    "unauthorized",
    "forbidden",
    "authentication",
    "invalid api key",
    "api key",
    "access denied",
    "no access",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one isolated OpenCode review.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant")
    parser.add_argument("--dir", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--base")
    parser.add_argument("--commit")
    parser.add_argument("--commit-end")
    return parser


def _git_diff_command(args: argparse.Namespace) -> list[str]:
    common = ["git"]
    if args.commit_end:
        if not args.base or args.commit:
            raise ValueError("commit range requires --base and --commit-end")
        return [
            *common,
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            f"{args.base}..{args.commit_end}",
        ]
    if bool(args.base) == bool(args.commit):
        raise ValueError("review requires exactly one of --base or --commit")
    if args.base:
        return [
            *common,
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            f"{args.base}...HEAD",
        ]
    return [
        *common,
        "show",
        "--first-parent",
        "--format=fuller",
        "--no-ext-diff",
        "--no-textconv",
        str(args.commit),
    ]


def _write_target_patch(args: argparse.Namespace, review_root: Path) -> Path:
    proc = subprocess.run(
        _git_diff_command(args),
        cwd=review_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            proc.stderr.strip() or "failed to produce review target diff"
        )
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="review-suite-opencode-target-",
        suffix=".patch",
        delete=False,
    )
    try:
        handle.write(proc.stdout)
        return Path(handle.name)
    finally:
        handle.close()


def _int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except TypeError, ValueError:
        return 0


def _new_usage_totals() -> dict[str, Any]:
    return {
        "input": 0,
        "output": 0,
        "reasoning": 0,
        "cache_read": 0,
        "cache_write": 0,
        "total": 0,
        "cost": 0.0,
    }


def _add_usage_totals(
    totals: dict[str, Any], tokens: Any, cost: Any
) -> tuple[bool, bool]:
    saw_tokens = False
    if isinstance(tokens, dict):
        cache = tokens.get("cache")
        if not isinstance(cache, dict):
            cache = {}
        totals["input"] += _int_value(tokens.get("input"))
        totals["output"] += _int_value(tokens.get("output"))
        totals["reasoning"] += _int_value(tokens.get("reasoning"))
        totals["cache_read"] += _int_value(cache.get("read"))
        totals["cache_write"] += _int_value(cache.get("write"))
        totals["total"] += _int_value(tokens.get("total"))
        saw_tokens = True
    saw_cost = False
    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
        totals["cost"] += float(cost)
        saw_cost = True
    return saw_tokens, saw_cost


def _usage_totals_to_usage(totals: dict[str, Any]) -> dict[str, int]:
    cache_read = int(totals["cache_read"])
    cache_write = int(totals["cache_write"])
    reasoning = int(totals["reasoning"])
    usage = {
        "input_tokens": int(totals["input"]) + cache_read + cache_write,
        "cached_input_tokens": cache_read,
        "output_tokens": int(totals["output"]),
    }
    if cache_write:
        usage["cache_write_tokens"] = cache_write
    if reasoning:
        usage["reasoning_output_tokens"] = reasoning
    if int(totals["total"]):
        usage["total_tokens"] = int(totals["total"])
    return usage


def _error_event_fields(event: dict[str, Any]) -> dict[str, str]:
    error = event.get("error")
    name = ""
    message = ""
    if isinstance(error, dict):
        name = str(error.get("name") or "").strip()
        data = error.get("data")
        if isinstance(data, dict):
            message = str(data.get("message") or "").strip()
        if not message:
            message = str(error.get("message") or "").strip()
    elif error is not None:
        message = str(error).strip()
    if not name and not message:
        message = str(event.get("message") or "").strip()
    return {"name": name, "message": message}


def _parse_event_stream(stdout: str) -> dict[str, Any]:
    session_id: str | None = None
    current_parts: list[str] = []
    completed_messages: list[str] = []
    error_events: list[dict[str, str]] = []
    totals = _new_usage_totals()
    saw_tokens = False
    saw_cost = False
    for raw_line in stdout.splitlines():
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        candidate_session = str(event.get("sessionID") or "").strip()
        if candidate_session:
            session_id = candidate_session
        event_type = str(event.get("type") or "")
        if event_type == "error":
            fields = _error_event_fields(event)
            if fields["name"] or fields["message"]:
                error_events.append(fields)
            continue
        if event_type == "step_start":
            current_parts = []
            continue
        if event_type == "text":
            part = event.get("part")
            text = str(part.get("text") or "") if isinstance(part, dict) else ""
            if text:
                current_parts.append(text)
            continue
        if event_type == "step_finish":
            part = event.get("part")
            if isinstance(part, dict):
                tokens_seen, cost_seen = _add_usage_totals(
                    totals, part.get("tokens"), part.get("cost")
                )
                saw_tokens = saw_tokens or tokens_seen
                saw_cost = saw_cost or cost_seen
            if current_parts:
                completed_messages.append("\n".join(current_parts).strip())
                current_parts = []
    if current_parts:
        completed_messages.append("\n".join(current_parts).strip())

    terminal_messages = [
        text
        for text in completed_messages
        if TERMINAL_REVIEW_RESULT_PREFIX.lower() in text.lower()
    ]
    return {
        "session_id": session_id,
        "reviewer_output": terminal_messages[-1] if terminal_messages else None,
        "usage": _usage_totals_to_usage(totals) if saw_tokens else {},
        "cost_usd": round(totals["cost"], 9) if saw_cost else None,
        "errors": error_events,
    }


def _usage_from_export(payload: Any) -> tuple[dict[str, int], float | None]:
    if not isinstance(payload, dict):
        return {}, None
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return {}, None
    totals = _new_usage_totals()
    saw_tokens = False
    saw_cost = False
    for message in messages:
        if not isinstance(message, dict) or message.get("type") not in {
            "assistant",
            "compaction",
        }:
            continue
        tokens_seen, cost_seen = _add_usage_totals(
            totals, message.get("tokens"), message.get("cost")
        )
        saw_tokens = saw_tokens or tokens_seen
        saw_cost = saw_cost or cost_seen
    if not (saw_tokens or saw_cost):
        return {}, None
    return (
        _usage_totals_to_usage(totals) if saw_tokens else {},
        round(totals["cost"], 9) if saw_cost else None,
    )


def _format_metadata_line(
    session_id: str | None,
    usage: dict[str, int],
    cost_usd: float | None,
    *,
    error_class: str | None = None,
    error_name: str | None = None,
    error_message: str | None = None,
) -> str:
    payload: dict[str, Any] = {"session_id": session_id, "usage": usage}
    if cost_usd is not None:
        payload["cost_usd"] = cost_usd
    if error_class:
        payload["error_class"] = error_class
    if error_name:
        payload["error_name"] = error_name
    if error_message:
        payload["error_message"] = error_message
    return REVIEW_METADATA_PREFIX + json.dumps(payload, separators=(",", ":")) + "\n"


def parse_opencode_review_metadata(text: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for line in str(text or "").splitlines():
        if not line.startswith(REVIEW_METADATA_PREFIX):
            continue
        raw_payload = line[len(REVIEW_METADATA_PREFIX) :].strip()
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            metadata = payload
    return metadata


def classify_opencode_error(
    *,
    returncode: int,
    errors: list[dict[str, Any]] | None = None,
    stderr_text: str = "",
    reviewer_output: str = "",
) -> str | None:
    if str(reviewer_output or "").strip():
        return None
    parts: list[str] = []
    for error in errors or []:
        if isinstance(error, dict):
            parts.append(str(error.get("name") or ""))
            parts.append(str(error.get("message") or ""))
    parts.append(str(stderr_text or ""))
    haystack = " ".join(parts).lower()
    if any(marker in haystack for marker in _CAPACITY_ERROR_MARKERS) or re.search(
        r"\b429\b", haystack
    ):
        return OPENCODE_ERROR_CLASS_CAPACITY
    if any(marker in haystack for marker in _UNAVAILABLE_ERROR_MARKERS) or re.search(
        r"\b(?:401|403)\b", haystack
    ):
        return OPENCODE_ERROR_CLASS_UNAVAILABLE
    if returncode != 0 or errors:
        return OPENCODE_ERROR_CLASS_FAILED
    return None


def _assistant_text_candidates(value: Any) -> list[str]:
    candidates: list[str] = []
    if isinstance(value, list):
        for item in value:
            candidates.extend(_assistant_text_candidates(item))
        return candidates
    if not isinstance(value, dict):
        return candidates

    content = value.get("content")
    if value.get("type") == "assistant" and isinstance(content, list):
        text = "\n".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict)
            and str(part.get("type") or "") == "text"
            and str(part.get("text") or "")
        ).strip()
        if text and (
            TERMINAL_REVIEW_RESULT_PREFIX.lower() in text.lower()
            or value.get("finish") == "stop"
        ):
            candidates.append(text)
    for nested in value.values():
        if isinstance(nested, dict | list):
            candidates.extend(_assistant_text_candidates(nested))
    return candidates


def _fetch_export_payload(
    opencode: str, session_id: str, review_root: Path
) -> dict[str, Any] | None:
    try:
        proc = subprocess.run(
            [opencode, "session", "export", "--standalone", session_id],
            cwd=review_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=REVIEW_EXPORT_TIMEOUT_SECONDS,
        )
    except OSError, subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _exported_review_text(
    opencode: str, session_id: str, review_root: Path
) -> str | None:
    payload = _fetch_export_payload(opencode, session_id, review_root)
    if payload is None:
        return None
    candidates = _assistant_text_candidates(payload)
    return candidates[-1] if candidates else None


def _opencode_run_command(
    opencode: str,
    args: argparse.Namespace,
    patch_path: Path,
) -> list[str]:
    model = args.model + (f"#{args.variant}" if args.variant else "")
    command = [
        opencode,
        "run",
        "--standalone",
        "--format",
        "json",
        "--model",
        model,
        "--agent",
        "review-suite",
        "--title",
        args.title,
    ]
    command.extend(["--file", str(patch_path)])
    return command


def _truncate(value: str, limit: int = 4000) -> str:
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]..."


def _emit_adapter_error(message: str) -> None:
    print(message, file=sys.stderr)
    sys.stderr.write(
        _format_metadata_line(
            None,
            {},
            None,
            error_class=OPENCODE_ERROR_CLASS_ADAPTER,
            error_message=_truncate(message, OPENCODE_ERROR_MESSAGE_LIMIT) or None,
        )
    )


def main() -> int:
    args = build_parser().parse_args()
    review_root = Path(args.dir).resolve()
    prompt = sys.stdin.read()
    if not prompt.strip():
        _emit_adapter_error("OpenCode review prompt is empty")
        return 2

    opencode = (
        shutil.which("opencode")
        or shutil.which("opencode.exe")
        or shutil.which("opencode.cmd")
    )
    if not opencode:
        _emit_adapter_error("OpenCode CLI was not found on PATH")
        return 127

    patch_path: Path | None = None
    try:
        patch_path = _write_target_patch(args, review_root)
        command = _opencode_run_command(opencode, args, patch_path)
        proc = subprocess.run(
            command,
            cwd=review_root,
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if proc.stderr:
            sys.stderr.write(proc.stderr)
            if not proc.stderr.endswith("\n"):
                sys.stderr.write("\n")
        stream = _parse_event_stream(proc.stdout)
        session_id = stream["session_id"]
        reviewer_output = stream["reviewer_output"]
        usage = stream["usage"]
        cost_usd = stream["cost_usd"]
        errors = list(stream.get("errors") or [])
        export_payload = (
            _fetch_export_payload(opencode, session_id, review_root)
            if session_id
            else None
        )
        if not reviewer_output and export_payload is not None:
            candidates = _assistant_text_candidates(export_payload)
            reviewer_output = candidates[-1] if candidates else None
        export_usage, export_cost = _usage_from_export(export_payload)
        if export_usage:
            usage = export_usage
        if export_cost is not None:
            cost_usd = export_cost
        error_class = classify_opencode_error(
            returncode=proc.returncode,
            errors=errors,
            stderr_text=proc.stderr,
            reviewer_output=reviewer_output or "",
        )
        if session_id or usage or error_class:
            primary_error = errors[0] if errors else {}
            error_message = _truncate(
                str(primary_error.get("message") or ""), OPENCODE_ERROR_MESSAGE_LIMIT
            )
            sys.stderr.write(
                _format_metadata_line(
                    session_id,
                    usage,
                    cost_usd,
                    error_class=error_class,
                    error_name=str(primary_error.get("name") or "") or None,
                    error_message=error_message or None,
                )
            )
        if proc.returncode != 0 and not reviewer_output:
            if proc.stdout:
                print(f"[opencode stdout]\n{_truncate(proc.stdout)}", file=sys.stderr)
            return proc.returncode
        if not reviewer_output:
            if proc.stdout:
                print(f"[opencode stdout]\n{_truncate(proc.stdout)}", file=sys.stderr)
            print(
                "OpenCode completed without a usable reviewer response", file=sys.stderr
            )
            return 2
        sys.stdout.write(reviewer_output.strip() + "\n")
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        _emit_adapter_error(f"OpenCode review adapter failed: {exc}")
        return 2
    finally:
        if patch_path is not None:
            try:
                patch_path.unlink(missing_ok=True)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
