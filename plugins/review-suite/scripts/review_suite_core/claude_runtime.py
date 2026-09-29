from __future__ import annotations

import os
import sys
from pathlib import Path

from .lens_runtime import CodexReviewLaunch
from .opencode_runtime import OPENCODE_REVIEW_SYSTEM_PROMPT, _opencode_review_prompt
from .workflow_state import validated_linear_review_range


CLAUDE_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
CLAUDE_AUTH_ENV = {
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST",
    "CLAUDE_CODE_EFFORT_LEVEL",
    "CLAUDE_CODE_DISABLE_THINKING",
    "CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING",
    "MAX_THINKING_TOKENS",
}


def prepare_claude_review_launch(
    *,
    tool_name: str,
    model: str,
    reasoning_effort: str,
    service_tier: str | None = None,
    title: str,
    review_root: Path,
    base: str | None = None,
    commit: str | None = None,
    commit_end: str | None = None,
    prompt: str = "",
    output_prefix: str | None = None,
    allow_unsafe_windows_wsl_fallback: bool,
) -> CodexReviewLaunch:
    del title, output_prefix, allow_unsafe_windows_wsl_fallback
    if service_tier:
        raise ValueError("service_tier is only supported by the Codex review backend")
    if model not in {"claude-opus-5-5", "claude-sonnet-5-5"}:
        raise ValueError(f"unsupported Claude review model: {model}")
    effort = str(reasoning_effort or "").strip().lower()
    if effort not in CLAUDE_EFFORTS:
        raise ValueError(
            f"Claude thinking level must be one of: {', '.join(sorted(CLAUDE_EFFORTS))}"
        )
    if base and commit_end:
        validated_linear_review_range(
            review_root, base, commit_end, label="Claude commit-range review launch"
        )
    stdin_text = _opencode_review_prompt(
        tool_name=tool_name,
        prompt=prompt,
        base=base,
        commit=commit,
        commit_end=commit_end,
    )
    env = {
        key: value for key, value in os.environ.items() if key not in CLAUDE_AUTH_ENV
    }
    env["PYTHONIOENCODING"] = "utf-8"
    command = [
        sys.executable,
        str(Path(__file__).with_name("claude_driver.py")),
        "--model",
        model,
        "--effort",
        effort,
        "--dir",
        str(review_root),
    ]
    if base:
        command.extend(["--base", base])
    if commit:
        command.extend(["--commit", commit])
    if commit_end:
        command.extend(["--commit-end", commit_end])
    return CodexReviewLaunch(
        command=command,
        stdin_text=stdin_text,
        final_message_path=None,
        cwd=review_root.resolve(),
        env=env,
        effective_reasoning_effort=effort,
    )


CLAUDE_REVIEW_SYSTEM_PROMPT = OPENCODE_REVIEW_SYSTEM_PROMPT
