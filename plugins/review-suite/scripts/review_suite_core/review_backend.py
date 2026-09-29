from __future__ import annotations

from pathlib import Path

from .lens_runtime import (
    CodexReviewLaunch,
    prepare_codex_review_launch as _prepare_codex_review_launch,
)
from .opencode_runtime import prepare_opencode_review_launch
from .claude_runtime import prepare_claude_review_launch


OPENCODE_MODEL_PREFIX = "opencode::"
CLAUDE_MODEL_PREFIX = "claude::"


def split_review_backend_model(model: str) -> tuple[str, str]:
    model_name = str(model or "").strip()
    if not model_name:
        raise ValueError("review model is required")
    if model_name.startswith(OPENCODE_MODEL_PREFIX):
        resolved = model_name[len(OPENCODE_MODEL_PREFIX) :].strip()
        provider, separator, provider_model = resolved.partition("/")
        if not separator or not provider.strip() or not provider_model.strip():
            raise ValueError("OpenCode review models must use opencode::provider/model")
        return "opencode", resolved
    if model_name.startswith(CLAUDE_MODEL_PREFIX):
        resolved = model_name[len(CLAUDE_MODEL_PREFIX) :].strip()
        if resolved not in {"claude-opus-5-5", "claude-sonnet-5-5"}:
            raise ValueError(
                "Claude review model must be claude::claude-opus-5-5 or claude::claude-sonnet-5-5"
            )
        return "claude", resolved
    return "codex", model_name


def prepare_review_launch(
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
    backend, resolved_model = split_review_backend_model(model)
    kwargs = {
        "tool_name": tool_name,
        "model": resolved_model,
        "reasoning_effort": reasoning_effort,
        "service_tier": service_tier,
        "title": title,
        "review_root": review_root,
        "base": base,
        "commit": commit,
        "commit_end": commit_end,
        "prompt": prompt,
        "output_prefix": output_prefix,
        "allow_unsafe_windows_wsl_fallback": allow_unsafe_windows_wsl_fallback,
    }
    if backend == "opencode":
        return prepare_opencode_review_launch(**kwargs)
    if backend == "claude":
        return prepare_claude_review_launch(**kwargs)
    return _prepare_codex_review_launch(**kwargs)
