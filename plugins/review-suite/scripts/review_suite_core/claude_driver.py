from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from review_suite_core.claude_runtime import CLAUDE_REVIEW_SYSTEM_PROMPT
from review_suite_core.opencode_driver import (
    _write_target_patch,
    classify_opencode_error,
)


REVIEW_METADATA_PREFIX = "[review-suite] claude-metadata: "


def _metadata(**fields: Any) -> None:
    sys.stderr.write(
        REVIEW_METADATA_PREFIX + json.dumps(fields, separators=(",", ":")) + "\n"
    )


def _number(value: Any) -> int:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else 0
    )


def _field(usage: dict[str, Any], snake: str, camel: str) -> int:
    return _number(usage.get(snake, usage.get(camel)))


def parse_claude_result(stdout: str, requested_model: str) -> dict[str, Any]:
    payload = json.loads(stdout)
    if not isinstance(payload, dict):
        raise ValueError("Claude returned a non-object result")
    model_usage = payload.get("modelUsage", payload.get("model_usage"))
    if not isinstance(model_usage, dict) or not model_usage:
        raise ValueError("Claude result did not report the actual model")
    actual_models = sorted(
        model for model in model_usage if model.startswith("claude-")
    )
    if not actual_models:
        raise ValueError("Claude result did not report a Claude model")
    usage = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "cache_write_tokens": 0,
        "output_tokens": 0,
    }
    for model in actual_models:
        item = model_usage[model]
        if not isinstance(item, dict):
            continue
        usage["input_tokens"] += _field(item, "input_tokens", "inputTokens")
        usage["cached_input_tokens"] += _field(
            item, "cache_read_input_tokens", "cacheReadInputTokens"
        )
        usage["cache_write_tokens"] += _field(
            item, "cache_creation_input_tokens", "cacheCreationInputTokens"
        )
        usage["output_tokens"] += _field(item, "output_tokens", "outputTokens")
    usage["input_tokens"] += usage["cached_input_tokens"] + usage["cache_write_tokens"]
    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    result = str(payload.get("result") or "").strip()
    return {
        "session_id": str(payload.get("session_id") or "").strip() or None,
        "actual_model": actual_models[0] if len(actual_models) == 1 else None,
        "actual_models": actual_models,
        "usage": usage,
        "result": result,
        "error": bool(payload.get("is_error"))
        or str(payload.get("subtype") or "") != "success",
        "model_mismatch": any(model != requested_model for model in actual_models),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one read-only Claude Code review."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", required=True)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--base")
    parser.add_argument("--commit")
    parser.add_argument("--commit-end")
    args = parser.parse_args()
    review_root = Path(args.dir).resolve()
    prompt = sys.stdin.read().strip()
    if not prompt:
        _metadata(
            error_class="adapter",
            error_message="Claude review prompt is empty",
            usage={},
        )
        return 2
    claude = (
        shutil.which("claude")
        or shutil.which("claude.exe")
        or shutil.which("claude.cmd")
    )
    if not claude:
        _metadata(
            error_class="adapter",
            error_message="Claude Code CLI was not found on PATH",
            usage={},
        )
        return 127
    try:
        auth = subprocess.run(
            [claude, "--safe-mode", "--restricted", "auth", "status"],
            cwd=review_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        status = json.loads(auth.stdout) if auth.returncode == 0 else {}
        if not isinstance(status, dict) or not (
            status.get("loggedIn")
            and status.get("authMethod") == "claude.ai"
            and status.get("apiProvider") == "firstParty"
        ):
            _metadata(
                error_class="unavailable",
                error_message="Claude Code subscription login is required",
                usage={},
            )
            return 2
        with tempfile.TemporaryDirectory(prefix="review-suite-claude-") as patch_dir:
            patch_path = _write_target_patch(args, review_root, Path(patch_dir))
            command = [
                claude,
                "-p",
                "--output-format",
                "json",
                "--model",
                args.model,
                "--effort",
                args.effort,
                "--safe-mode",
                "--restricted",
                "--add-dir",
                patch_dir,
                "--strict-mcp-config",
                "--no-session-persistence",
                "--tools",
                "Read,Glob,Grep",
                "--allowedTools",
                "Read,Glob,Grep",
                "--permission-mode",
                "dontAsk",
                "--permission-prompts",
                "none",
                "--append-system-prompt",
                CLAUDE_REVIEW_SYSTEM_PROMPT,
            ]
            proc = subprocess.run(
                command,
                cwd=review_root,
                input=f"{prompt}\n\nTarget patch file: {patch_path}. Read it first.\n",
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        if proc.stderr:
            sys.stderr.write(proc.stderr.rstrip() + "\n")
        try:
            result = parse_claude_result(proc.stdout, args.model)
        except (json.JSONDecodeError, ValueError) as exc:
            error_class = classify_opencode_error(
                returncode=proc.returncode or 2, stderr_text=proc.stderr or proc.stdout
            )
            _metadata(
                error_class=error_class or "adapter", error_message=str(exc), usage={}
            )
            return proc.returncode or 2
        error_message = (
            f"Claude used {', '.join(result['actual_models'])} instead of {args.model}"
            if result["model_mismatch"]
            else result["result"][:500]
            if result["error"]
            else ""
        )
        error_class = (
            "unavailable"
            if result["model_mismatch"]
            else classify_opencode_error(
                returncode=proc.returncode or 2,
                stderr_text=proc.stderr + " " + error_message,
            )
            if result["error"] or proc.returncode
            else None
        )
        _metadata(
            session_id=result["session_id"],
            usage=result["usage"],
            actual_model=result["actual_model"],
            actual_models=result["actual_models"],
            error_class=error_class,
            error_message=error_message or None,
        )
        if error_class or not result["result"]:
            return proc.returncode or 2
        sys.stdout.write(result["result"] + "\n")
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        _metadata(error_class="adapter", error_message=str(exc)[:500], usage={})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
