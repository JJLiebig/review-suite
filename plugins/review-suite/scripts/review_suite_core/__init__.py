"""Shared review helpers, loaded when a caller uses them."""

from importlib import import_module


_MODULE_EXPORTS = {
    "axi_output": (
        "AxiArgumentParser",
        "emit_error",
        "emit_toon",
        "format_command",
        "toon_dumps",
        "write_text",
    ),
    "codex_runtime": (
        "effective_execution_cwd",
        "is_windows_wsl_unc_path",
        "running_in_wsl",
        "use_unsafe_windows_wsl_fallback",
        "validate_codex_runtime",
        "wrapper_launch_cwd",
    ),
    "config": (
        "LensModelConfig",
        "lens_model_config",
        "load_config",
    ),
    "costing": (
        "core_usage_tokens",
        "normalize_usage_tokens",
        "price_usage_tokens",
        "run_total_tokens",
    ),
    "opencode_driver": (
        "parse_opencode_review_metadata",
        "parse_review_backend_metadata",
    ),
    "opencode_runtime": (
        "default_opencode_reasoning_effort",
        "opencode_reasoning_efforts",
        "validate_opencode_reasoning_effort",
    ),
    "lens_runtime": (
        "DEFAULT_PROGRESS_INTERVAL_SECONDS",
        "DEFAULT_TIMEOUT_SECONDS",
        "WRAPPER_SESSION_LOG_FILENAME",
        "CodexReviewLaunch",
        "codex_review_stdin_text",
        "emit_result",
        "normalize_service_tier",
        "progress_heartbeat_line",
        "record_wrapper_session",
        "run_codex",
        "run_codex_review",
    ),
    "model_labels": (
        "SUPPORTED_REASONING_EFFORTS",
        "SUPPORTED_SERVICE_TIERS",
        "codex_reasoning_effort",
        "is_deep_reasoning_effort",
        "parse_model_label",
    ),
    "paths": (
        "cwd_path_from_normalized",
        "normalize_cwd",
        "resolve_cd_path",
        "resolve_repo_root",
        "utc_now",
        "utc_now_iso",
    ),
    "process_runtime": (
        "CapturedChildProcess",
        "CapturedChildWaitResult",
        "launch_captured_child_process",
        "terminate_process_tree",
        "wait_for_captured_child_process",
    ),
    "review_backend": (
        "CLAUDE_MODEL_PREFIX",
        "OPENCODE_MODEL_PREFIX",
        "prepare_review_launch",
    ),
    "workflow_state": (
        "EFFECTIVE_BASE_METADATA_KEYS",
        "anchor_updates_branch_state",
        "branch_diff_paths",
        "classify_delta_recommendation",
        "current_branch",
        "current_head",
        "diff_artifact",
        "diff_stats",
        "dirty_worktree_scope",
        "effective_base_ref",
        "has_committed_diff",
        "has_worktree_changes",
        "inspect_workflow_status",
        "is_ancestor",
        "meaningful_worktree_status_entries",
        "merge_base",
        "merge_base_drift_scope",
        "record_review_anchor",
        "resolve_ref",
        "resolve_reviewed_head",
        "validated_linear_review_range",
        "worktree_diff_artifact",
        "worktree_diff_stats",
        "worktree_status_entries",
        "workflow_state_path",
    ),
}
_EXPORTS = {
    name: (module, name)
    for module, names in _MODULE_EXPORTS.items()
    for name in names
}
_EXPORTS["prepare_codex_review_launch"] = ("review_backend", "prepare_review_launch")
__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    try:
        module, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(f".{module}", __name__), attribute)
    globals()[name] = value
    return value
