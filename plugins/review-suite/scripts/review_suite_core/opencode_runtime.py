from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

from .lens_runtime import (
    CodexReviewLaunch,
    codex_review_prompt_instructions,
    should_apply_technical_review_charter,
)
from .workflow_state import validated_linear_review_range


OPENCODE_REVIEW_AGENT = "review-suite"
OPENCODE_DEFAULT_REASONING_EFFORT = "high"
OPENCODE_MODEL_REASONING_EFFORTS = {
    "opencode-go/deepseek-v4.1-flash": ("low", "high", "max"),
    "opencode-go/glm-5.3-flash": ("low", "high", "max"),
}
OPENCODE_REVIEW_SYSTEM_PROMPT = (
    "Act as a code reviewer for a proposed change made by another engineer. "
    "Report only discrete, actionable defects introduced or exposed by the target change that materially affect correctness, "
    "security, performance, integration behavior, accessibility, or maintainability. Do not report pre-existing problems, "
    "style nits, speculative breakage, intentional product changes, or concerns that depend on unstated assumptions. "
    "Before reporting a finding, inspect enough surrounding code and repository instructions to verify the affected path and concrete failure scenario. "
    "Return every qualifying finding rather than stopping after the first. Keep each finding concise, matter-of-fact, and specific about the conditions in which it occurs. "
    "Prefer the smallest changed line range that identifies the defect. Do not modify files. "
    "Write in the language of the caller's instructions and never translate machine-readable result lines. "
    "Follow the caller's review-output contract exactly; it overrides any default formatting preference in this prompt."
)


def opencode_reasoning_efforts(model: str) -> tuple[str, ...] | None:
    return OPENCODE_MODEL_REASONING_EFFORTS.get(str(model or "").strip())


def validate_opencode_reasoning_effort(model: str, effort: str) -> str:
    supported = opencode_reasoning_efforts(model)
    value = str(effort or "").strip().lower()
    if supported is None:
        return value
    if not value:
        raise ValueError(f"thinking level is required for {model}")
    if value not in supported:
        raise ValueError(
            f"thinking level '{effort}' is unsupported for {model}; "
            f"supported thinking levels: {', '.join(supported)}"
        )
    return value


def default_opencode_reasoning_effort(model: str) -> str:
    supported = opencode_reasoning_efforts(model)
    if not supported:
        return ""
    if OPENCODE_DEFAULT_REASONING_EFFORT in supported:
        return OPENCODE_DEFAULT_REASONING_EFFORT
    return supported[0]


def opencode_reasoning_variant(model: str, effort: str) -> str | None:
    supported = opencode_reasoning_efforts(model)
    if not supported:
        return None
    value = str(effort or "").strip().lower()
    if value in supported:
        return value
    return default_opencode_reasoning_effort(model)


def _allow_gitless_review_config() -> dict[str, object]:
    return {
        "$schema": "https://opencode.ai/config.json",
        "share": "disabled",
        "agent": {
            OPENCODE_REVIEW_AGENT: {
                "description": "Read-only Review Suite code reviewer",
                "mode": "primary",
                "prompt": OPENCODE_REVIEW_SYSTEM_PROMPT,
                "permission": {
                    "*": "deny",
                    "read": {
                        "*": "allow",
                        "*.env": "deny",
                        "*.env.*": "deny",
                        "*.env.example": "allow",
                    },
                    "glob": "allow",
                    "grep": "allow",
                    "list": "allow",
                    "lsp": "allow",
                    "edit": "deny",
                    "bash": "deny",
                    "task": "deny",
                    "external_directory": "deny",
                    "todowrite": "deny",
                    "webfetch": "deny",
                    "websearch": "deny",
                    "skill": "deny",
                    "question": "deny",
                    "doom_loop": "deny",
                },
            }
        },
    }


def _opencode_output_config(model: str) -> dict[str, object]:
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    try:
        catalog = json.loads(
            (cache / "opencode" / "models.json").read_text(encoding="utf-8")
        )
        provider_id, model_id = model.split("/", 1)
        provider = catalog[provider_id]
        info = provider["models"][model_id]
        maximum = info["limit"]["output"]
        npm = info.get("provider", {}).get("npm") or provider.get("npm")
        if not isinstance(maximum, int) or maximum <= 0:
            raise ValueError("no advertised output maximum")
        if npm in {"@ai-sdk/openai-compatible", "@ai-sdk/anthropic"}:
            body = {"max_tokens": maximum}
        elif npm == "@ai-sdk/openai":
            body = {"max_output_tokens": maximum}
        elif npm == "@ai-sdk/google":
            body = {"generationConfig": {"maxOutputTokens": maximum}}
        else:
            raise ValueError(f"unsupported model adapter: {npm}")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        warnings.warn(
            f"Cannot determine the output maximum for {model}; using the provider default ({exc}).",
            stacklevel=2,
        )
        return {}
    return {provider_id: {"models": {model_id: {"body": body}}}}


def opencode_review_env(model: str = "") -> dict[str, str]:
    env = os.environ.copy()
    config = _allow_gitless_review_config()
    if model:
        config["providers"] = _opencode_output_config(model)
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config, separators=(",", ":"))
    env["OPENCODE_DISABLE_AUTOUPDATE"] = "true"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _opencode_review_prompt(
    *,
    tool_name: str,
    prompt: str,
    base: str | None,
    commit: str | None,
    commit_end: str | None,
) -> str:
    base_ref = str(base or "").strip()
    commit_ref = str(commit or "").strip()
    commit_end_ref = str(commit_end or "").strip()
    if commit_end_ref:
        if not base_ref or commit_ref:
            raise ValueError(
                "OpenCode commit-range review requires base and commit_end"
            )
        target = f"review commit range `{base_ref}..{commit_end_ref}`"
    elif bool(base_ref) == bool(commit_ref):
        raise ValueError("OpenCode review requires exactly one of base or commit")
    elif base_ref:
        target = f"compare the current checkout against base ref `{base_ref}`"
    else:
        target = f"review the changes introduced by commit `{commit_ref}`"

    prompt_text = prompt.strip()
    review_prompt = (
        codex_review_prompt_instructions(prompt_text)
        if should_apply_technical_review_charter(tool_name)
        else prompt_text
    )
    message = (
        "You are running a focused code review. Do not modify files.\n"
        f"Review target: {target}.\n"
        "Review Suite attaches the exact target diff as a patch file. Treat that patch as the authoritative review boundary, "
        "then use repository read/search tools only for surrounding context and applicable project instructions.\n"
    )
    if review_prompt:
        message += f"\nReview instructions:\n{review_prompt}\n"
    return message


def _driver_path() -> Path:
    return Path(__file__).with_name("opencode_driver.py")


def prepare_opencode_review_launch(
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
    del output_prefix, allow_unsafe_windows_wsl_fallback
    if service_tier:
        raise ValueError("service_tier is only supported by the Codex review backend")

    model_name = str(model or "").strip()
    if not model_name or "/" not in model_name:
        raise ValueError("OpenCode review model must use provider/model")
    variant = opencode_reasoning_variant(model_name, reasoning_effort)

    base_ref = str(base or "").strip()
    commit_ref = str(commit or "").strip()
    commit_end_ref = str(commit_end or "").strip()
    if base_ref and commit_end_ref:
        validated_linear_review_range(
            review_root,
            base_ref,
            commit_end_ref,
            label="OpenCode commit-range review launch",
        )

    stdin_text = _opencode_review_prompt(
        tool_name=tool_name,
        prompt=prompt,
        base=base,
        commit=commit,
        commit_end=commit_end,
    )
    command = [
        sys.executable,
        str(_driver_path()),
        "--model",
        model_name,
        "--dir",
        str(review_root),
        "--title",
        title,
    ]
    if variant:
        command.extend(["--variant", variant])
    if base_ref:
        command.extend(["--base", base_ref])
    if commit_ref:
        command.extend(["--commit", commit_ref])
    if commit_end_ref:
        command.extend(["--commit-end", commit_end_ref])

    return CodexReviewLaunch(
        command=command,
        stdin_text=stdin_text,
        final_message_path=None,
        cwd=review_root.resolve(),
        env=opencode_review_env(model_name),
        effective_reasoning_effort=variant or "provider-default",
    )
