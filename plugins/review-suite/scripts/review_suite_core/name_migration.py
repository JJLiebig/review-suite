from __future__ import annotations

import re
from typing import Any


KEY_NAMES = {
    "phase_review": "base_review",
    "pr_review": "deep_review",
    "review_t1": "review_base",
    "review_t3": "review_deep",
    "arena_phase": "arena_base",
    "discovery_phase": "discovery_base",
    "normal_signoff": "base_signoff",
    "signoff_normal_model": "signoff_base_model",
    "normal_arena_loops": "base_arena_loops",
    "t1_sessions": "base_sessions",
    "t3_sessions": "deep_sessions",
}
NAME_FIELDS = {
    "task_class",
    "task_classes",
    "taskName",
    "task",
    "public_task",
    "lane",
    "source_lane",
    "last_reviewed_lane",
    "tool_name",
    "pool",
    "model_ref",
    "loop_ref",
    "subtract_loop_ref",
}
STEP_FIELDS = {
    "name",
    "step",
    "step_name",
    "orchestrator_step",
    "next_step_name",
    "profile_step",
}
STEP_NAMES = {
    "precision-signoff": "base-signoff",
    "fast-signoff": "base-signoff",
    "arena-phase-review": "arena-base-review",
    "arena-pr-review": "arena-deep-review",
}


def _step_name(value: str) -> str:
    for old, new in STEP_NAMES.items():
        if value == old or re.fullmatch(re.escape(old) + r"-\d+", value):
            return new + value[len(old) :]
    return value


def rename_payload(
    value: Any,
    *,
    deep_rounds: set[str],
    field: str = "",
    path: tuple[str, ...] = (),
    settings: bool = False,
) -> Any:
    if isinstance(value, list):
        return [
            rename_payload(
                item, deep_rounds=deep_rounds, field=field, path=path, settings=settings
            )
            for item in value
        ]
    if isinstance(value, str):
        if field in NAME_FIELDS:
            return KEY_NAMES.get(value, value)
        if field in STEP_FIELDS:
            return _step_name(value)
        if value == "normal" and (
            field in {"mode", "requested_mode", "effective_mode"}
            or path[-2:-1] == ("mode",)
            and field in {"requested", "effective"}
        ):
            return "base"
        return value
    if not isinstance(value, dict):
        return value

    result: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"review_brief", "reviewer_output", "requested_prompt"}:
            result[key] = item
            continue
        new_key = KEY_NAMES.get(key, key)
        if (
            key == "normal"
            and settings
            and (not path or len(path) >= 2 and path[-2] == "profiles")
        ):
            new_key = "base"
        renamed = rename_payload(
            item,
            deep_rounds=deep_rounds,
            field=key,
            path=(*path, key),
            settings=settings,
        )
        if new_key in result and result[new_key] != renamed:
            raise ValueError(
                f"conflicting saved values for {'.'.join((*path, new_key))}"
            )
        result[new_key] = renamed

    # Old fixed-model deep signoffs were mislabeled as phase reviews.
    is_deep = any(value.get(key) == "deep-signoff" for key in STEP_FIELDS)
    profile_step = value.get("profile_step")
    is_deep = (
        is_deep
        or isinstance(profile_step, dict)
        and profile_step.get("name") == "deep-signoff"
    )
    is_deep = is_deep or any(
        value.get(key) in deep_rounds
        for key in ("round_id", "source_round_id", "origin_round_id")
    )
    if is_deep:
        if "task_class" in result or "round_id" in result:
            result["task_class"] = "deep_review"
        for key in ("lane", "public_task", "taskName", "source_lane"):
            if result.get(key) == "review_base":
                result[key] = "review_deep"
    if (
        value.get("name") in {"precision-signoff", "fast-signoff", "deep-signoff"}
        or value.get("kind") == "review"
        and "name" in value
        and "count" in value
    ):
        result["lane"] = "review_deep" if is_deep else "review_base"
        result["task_class"] = "deep_review" if is_deep else "base_review"
    return result
