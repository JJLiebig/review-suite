from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .model_labels import (
    SUPPORTED_REASONING_EFFORTS,
    SUPPORTED_SERVICE_TIERS,
    parse_model_label,
    supported_reasoning_efforts_text,
)

SUPPORTED_MODES = ("fast", "base", "deep")
SUPPORTED_SELECTIONS = ("stable", "auto")
# Retain the removed mode's rank only so persisted brief cycles can restart upward.
MODE_STRICTNESS_ORDER = {"fast": 0, "brief": 1, "base": 2, "deep": 3}
RESTART_TARGET_MODES = ("base", "deep")
SUPPORTED_SELECTION_REASONS = (
    "explicit_stable",
    "auto_stable_profile",
)
SUPPORTED_STEP_KINDS = ("review", "arena")
SUPPORTED_ARENA_LANES = ("review_base", "review_deep")
SUPPORTED_ARENA_TASK_CLASSES = ("base_review", "deep_review")
ARENA_LANES_BY_TASK_CLASS = {
    "base_review": "review_base",
    "deep_review": "review_deep",
}


@dataclass(frozen=True)
class OrchestratorProfileStep:
    kind: str
    name: str
    count: int | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    service_tier: str | None = None
    lane: str | None = None
    task_class: str | None = None
    rating_pool_id: str | None = None
    reporting_pool: bool = False
    variant_groups: tuple[tuple[str, ...], ...] = ()
    variant_ids: tuple[str, ...] = ()
    rerun_on_findings: bool = False
    max_review_rounds: int | None = None


@dataclass(frozen=True)
class OrchestratorProfile:
    mode: str
    profile: str
    steps: tuple[OrchestratorProfileStep, ...]


@dataclass(frozen=True)
class OrchestratorProfileResolution:
    requested_mode: str
    effective_mode: str
    requested_selection: str
    effective_selection: str
    selection_reason: str
    profile: OrchestratorProfile

    @property
    def steps(self) -> tuple[OrchestratorProfileStep, ...]:
        return self.profile.steps


def _non_empty_text(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _positive_int(value: Any, *, field: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if number <= 0:
        raise ValueError(f"{field} must be > 0")
    return number


def _non_negative_int(value: Any, *, field: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if number < 0:
        raise ValueError(f"{field} must be >= 0")
    return number


def _optional_positive_int(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field=field)


def _optional_bool(value: Any, *, field: str) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be true or false")
    return value


def _parse_model_label(value: Any, *, field: str) -> tuple[str, str, str | None]:
    return parse_model_label(value, field=field)


def _model_from_step(
    raw_step: dict[str, Any],
    *,
    stable_defaults: dict[str, Any],
    field: str,
) -> tuple[str, str, str | None]:
    model_ref = str(raw_step.get("model_ref") or "").strip()
    if model_ref:
        if "model" in raw_step or "reasoning_effort" in raw_step:
            raise ValueError(
                f"{field} must use either model_ref or model/reasoning_effort"
            )
        return _parse_model_label(
            stable_defaults.get(model_ref),
            field=f"orchestrator.stable_defaults.{model_ref}",
        )
    model = _non_empty_text(raw_step.get("model"), field=f"{field}.model")
    effort = _non_empty_text(
        raw_step.get("reasoning_effort"), field=f"{field}.reasoning_effort"
    )
    if effort not in SUPPORTED_REASONING_EFFORTS:
        raise ValueError(
            f"{field}.reasoning_effort must be one of: {supported_reasoning_efforts_text()}"
        )
    return model, effort, None


def _service_tier(
    raw_step: dict[str, Any], *, default: str | None, field: str
) -> str | None:
    service_tier = str(raw_step.get("service_tier") or default or "").strip() or None
    if service_tier and service_tier not in SUPPORTED_SERVICE_TIERS:
        raise ValueError(
            f"{field}.service_tier must be one of: {', '.join(sorted(SUPPORTED_SERVICE_TIERS))}"
        )
    return service_tier


def _normalize_mode(mode: str) -> str:
    normalized = str(mode or "").strip()
    if normalized not in SUPPORTED_MODES:
        raise ValueError(
            f"orchestrator mode must be one of: {', '.join(SUPPORTED_MODES)}"
        )
    return normalized


def _normalize_selection(selection: str) -> str:
    normalized = str(selection or "").strip()
    if normalized not in SUPPORTED_SELECTIONS:
        raise ValueError(
            f"orchestrator selection must be one of: {', '.join(SUPPORTED_SELECTIONS)}"
        )
    return normalized


def _normalize_step_kind(value: Any, *, field: str) -> str:
    kind = str(value or "review").strip() or "review"
    if kind not in SUPPORTED_STEP_KINDS:
        raise ValueError(
            f"{field}.kind must be one of: {', '.join(SUPPORTED_STEP_KINDS)}"
        )
    return kind


def _normalize_arena_lane(value: Any, *, field: str) -> str:
    lane = _non_empty_text(value, field=f"{field}.lane")
    if lane not in SUPPORTED_ARENA_LANES:
        raise ValueError(
            f"{field}.lane must be one of: {', '.join(SUPPORTED_ARENA_LANES)}"
        )
    return lane


def _normalize_arena_task_class(value: Any, *, field: str) -> str:
    task_class = _non_empty_text(value, field=f"{field}.task_class")
    if task_class not in SUPPORTED_ARENA_TASK_CLASSES:
        raise ValueError(
            f"{field}.task_class must be one of: {', '.join(SUPPORTED_ARENA_TASK_CLASSES)}"
        )
    return task_class


def _normalize_arena_pair(raw_step: dict[str, Any], *, field: str) -> tuple[str, str]:
    lane = _normalize_arena_lane(raw_step.get("lane"), field=field)
    task_class = _normalize_arena_task_class(raw_step.get("task_class"), field=field)
    expected_lane = ARENA_LANES_BY_TASK_CLASS[task_class]
    if lane != expected_lane:
        raise ValueError(
            f"{field}.lane must be {expected_lane} for task_class {task_class}"
        )
    return lane, task_class


def _normalize_arena_pool(
    raw_step: dict[str, Any], *, config: dict[str, Any], field: str
) -> tuple[str, tuple[tuple[str, ...], ...], tuple[str, ...], bool]:
    pool_name = _non_empty_text(raw_step.get("pool"), field=f"{field}.pool")
    arena = config.get("arena") or {}
    pools = arena.get("pools") if isinstance(arena, dict) else None
    pool = pools.get(pool_name) if isinstance(pools, dict) else None
    if not isinstance(pool, dict):
        raise ValueError(f"{field}.pool references unknown arena pool: {pool_name}")
    rating_pool_id = _non_empty_text(
        pool.get("rating_pool_id"), field=f"arena.pools.{pool_name}.rating_pool_id"
    )
    raw_groups = pool.get("variant_groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError(
            f"arena.pools.{pool_name}.variant_groups must be a non-empty array"
        )
    groups: list[tuple[str, ...]] = []
    group_size: int | None = None
    for index, raw_group in enumerate(raw_groups):
        group_field = f"arena.pools.{pool_name}.variant_groups[{index}]"
        if not isinstance(raw_group, list) or len(raw_group) < 2:
            raise ValueError(f"{group_field} must contain at least two variants")
        group = tuple(_non_empty_text(value, field=group_field) for value in raw_group)
        if len(group) != len(set(group)):
            raise ValueError(f"{group_field} cannot contain duplicate variants")
        if group_size is None:
            group_size = len(group)
        elif len(group) != group_size:
            raise ValueError(
                f"arena.pools.{pool_name}.variant_groups must use one group size"
            )
        groups.append(group)
    raw_variant_ids = pool.get("variant_ids")
    variant_ids: tuple[str, ...] = ()
    if raw_variant_ids is not None:
        candidate_field = f"arena.pools.{pool_name}.variant_ids"
        if not isinstance(raw_variant_ids, list) or len(raw_variant_ids) < int(
            group_size or 2
        ):
            raise ValueError(
                f"{candidate_field} must contain at least {group_size or 2} variants"
            )
        variant_ids = tuple(
            _non_empty_text(value, field=candidate_field) for value in raw_variant_ids
        )
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError(f"{candidate_field} cannot contain duplicate variants")
        missing = sorted(
            {value for group in groups for value in group} - set(variant_ids)
        )
        if missing:
            raise ValueError(
                f"{candidate_field} must include every scheduled variant: {', '.join(missing)}"
            )
    reporting = _optional_bool(
        pool.get("reporting"), field=f"arena.pools.{pool_name}.reporting"
    )
    return rating_pool_id, tuple(groups), variant_ids, reporting


def _raw_loop_ref(raw_step: Any) -> str:
    if not isinstance(raw_step, dict):
        return ""
    return str(raw_step.get("loop_ref") or "").strip()


def _config_ref(config: dict[str, Any], ref: str, *, field: str) -> Any:
    current: Any = config
    for part in ref.split("."):
        key = part.strip()
        if not key or not isinstance(current, dict) or key not in current:
            raise ValueError(f"{field} references unknown config value {ref}")
        current = current[key]
    return current


def _config_bool_ref(config: dict[str, Any], ref: str, *, field: str) -> bool:
    value = _config_ref(config, ref, field=field)
    if not isinstance(value, bool):
        raise ValueError(f"{field} must reference a true or false config value")
    return value


def _step_enabled(
    raw_step: dict[str, Any], *, config: dict[str, Any], field: str
) -> bool:
    enabled_ref = str(raw_step.get("enabled_ref") or "").strip()
    if not enabled_ref:
        return True
    return _config_bool_ref(config, enabled_ref, field=f"{field}.enabled_ref")


def _loop_count(
    raw_step: dict[str, Any],
    *,
    config: dict[str, Any],
    stable_defaults: dict[str, Any],
    loop_ref: str,
    field: str,
) -> int:
    if not _step_enabled(raw_step, config=config, field=field):
        return 0
    allow_zero = (
        _normalize_step_kind(raw_step.get("kind"), field=field) == "arena"
        or "enabled_ref" in raw_step
    )
    if allow_zero:
        loops = _non_negative_int(
            stable_defaults.get(loop_ref),
            field=f"orchestrator.stable_defaults.{loop_ref}",
        )
    else:
        loops = _positive_int(
            stable_defaults.get(loop_ref),
            field=f"orchestrator.stable_defaults.{loop_ref}",
        )

    subtract_ref = str(raw_step.get("subtract_loop_ref") or "").strip()
    if subtract_ref:
        subtract_if = str(raw_step.get("subtract_if") or "").strip()
        if not subtract_if or _config_bool_ref(
            config, subtract_if, field=f"{field}.subtract_if"
        ):
            loops -= _non_negative_int(
                stable_defaults.get(subtract_ref),
                field=f"orchestrator.stable_defaults.{subtract_ref}",
            )
        min_loops = _positive_int(
            raw_step.get("min_loops", 1), field=f"{field}.min_loops"
        )
        loops = max(min_loops, loops)
    return loops


def _normalize_step(
    raw_step: Any,
    *,
    field: str,
    config: dict[str, Any],
    stable_defaults: dict[str, Any],
    name_suffix: str = "",
) -> OrchestratorProfileStep:
    if not isinstance(raw_step, dict):
        raise ValueError(f"{field} must be an object")
    kind = _normalize_step_kind(raw_step.get("kind"), field=field)
    name = (
        f"{_non_empty_text(raw_step.get('name'), field=f'{field}.name')}{name_suffix}"
    )
    if kind == "arena":
        lane, task_class = _normalize_arena_pair(raw_step, field=field)
        rating_pool_id, variant_groups, variant_ids, reporting_pool = (
            _normalize_arena_pool(raw_step, config=config, field=field)
        )
        return OrchestratorProfileStep(
            kind=kind,
            name=name,
            lane=lane,
            task_class=task_class,
            rating_pool_id=rating_pool_id,
            reporting_pool=reporting_pool,
            variant_groups=variant_groups,
            variant_ids=variant_ids,
        )
    model, effort, default_service_tier = _model_from_step(
        raw_step, stable_defaults=stable_defaults, field=field
    )
    rerun_on_findings = _optional_bool(
        raw_step.get("rerun_on_findings"), field=f"{field}.rerun_on_findings"
    )
    max_review_rounds = _optional_positive_int(
        raw_step.get("max_review_rounds"), field=f"{field}.max_review_rounds"
    )
    if rerun_on_findings and max_review_rounds is not None:
        raise ValueError(
            f"{field} cannot combine rerun_on_findings with max_review_rounds"
        )
    task_class = "deep_review" if name == f"deep-signoff{name_suffix}" else "base_review"
    return OrchestratorProfileStep(
        kind=kind,
        name=name,
        lane=ARENA_LANES_BY_TASK_CLASS[task_class],
        task_class=task_class,
        count=_positive_int(raw_step.get("count"), field=f"{field}.count"),
        model=model,
        reasoning_effort=effort,
        service_tier=_service_tier(raw_step, default=default_service_tier, field=field),
        rerun_on_findings=rerun_on_findings,
        max_review_rounds=max_review_rounds,
    )


def _normalize_steps(
    raw_steps: list[Any],
    *,
    field: str,
    config: dict[str, Any],
    stable_defaults: dict[str, Any],
) -> tuple[OrchestratorProfileStep, ...]:
    steps: list[OrchestratorProfileStep] = []
    index = 0
    while index < len(raw_steps):
        loop_ref = _raw_loop_ref(raw_steps[index])
        if not loop_ref:
            if isinstance(raw_steps[index], dict) and not _step_enabled(
                raw_steps[index], config=config, field=f"{field}[{index}]"
            ):
                index += 1
                continue
            steps.append(
                _normalize_step(
                    raw_steps[index],
                    field=f"{field}[{index}]",
                    config=config,
                    stable_defaults=stable_defaults,
                )
            )
            index += 1
            continue
        block: list[tuple[int, Any]] = []
        while index < len(raw_steps) and _raw_loop_ref(raw_steps[index]) == loop_ref:
            block.append((index, raw_steps[index]))
            index += 1
        enabled_block = [
            (raw_index, raw_step)
            for raw_index, raw_step in block
            if not isinstance(raw_step, dict)
            or _step_enabled(raw_step, config=config, field=f"{field}[{raw_index}]")
        ]
        if not enabled_block:
            continue
        loops = _loop_count(
            enabled_block[0][1],
            config=config,
            stable_defaults=stable_defaults,
            loop_ref=loop_ref,
            field=f"{field}[{enabled_block[0][0]}]",
        )
        for loop_index in range(loops):
            suffix = f"-{loop_index + 1}" if loops > 1 else ""
            for raw_index, raw_step in enabled_block:
                steps.append(
                    _normalize_step(
                        raw_step,
                        field=f"{field}[{raw_index}]",
                        config=config,
                        stable_defaults=stable_defaults,
                        name_suffix=suffix,
                    )
                )
    return tuple(steps)


def _normalize_profile(
    raw_profile: Any,
    *,
    mode: str,
    profile: str,
    config: dict[str, Any],
    stable_defaults: dict[str, Any],
) -> OrchestratorProfile:
    if not isinstance(raw_profile, dict):
        raise ValueError(f"orchestrator.profiles.{profile}.{mode} must be an object")
    if "deslop_enabled" in raw_profile:
        raise ValueError(
            f"orchestrator.profiles.{profile}.{mode}.deslop_enabled is obsolete; use --skip-deslop"
        )
    raw_steps = raw_profile.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError(
            f"orchestrator.profiles.{profile}.{mode}.steps must contain at least one step"
        )
    steps = _normalize_steps(
        raw_steps,
        field=f"orchestrator.profiles.{profile}.{mode}.steps",
        config=config,
        stable_defaults=stable_defaults,
    )
    return OrchestratorProfile(
        mode=mode,
        profile=profile,
        steps=steps,
    )


def _validate_calibration_policy(orchestrator: dict[str, Any]) -> None:
    raw_calibration = orchestrator.get("calibration") or {}
    if not isinstance(raw_calibration, dict):
        raise ValueError("orchestrator.calibration must be an object")
    if bool(raw_calibration.get("auto_promotion_enabled", False)):
        raise ValueError(
            "orchestrator.calibration.auto_promotion_enabled must be false"
        )


def _stable_defaults(orchestrator: dict[str, Any]) -> dict[str, Any]:
    raw_defaults = orchestrator.get("stable_defaults") or {}
    if not isinstance(raw_defaults, dict):
        raise ValueError("orchestrator.stable_defaults must be an object")
    return raw_defaults


def load_orchestrator_profiles(
    config: dict[str, Any],
) -> dict[str, dict[str, OrchestratorProfile]]:
    orchestrator = config.get("orchestrator") or {}
    if not isinstance(orchestrator, dict):
        raise ValueError("orchestrator config must be an object")
    _validate_calibration_policy(orchestrator)
    stable_defaults = _stable_defaults(orchestrator)
    raw_profiles = orchestrator.get("profiles") or {}
    if not isinstance(raw_profiles, dict):
        raise ValueError("orchestrator.profiles config must be an object")

    raw_modes = raw_profiles.get("stable") or {}
    if not isinstance(raw_modes, dict):
        raise ValueError("orchestrator.profiles.stable must be an object")
    return {
        "stable": {
            mode: _normalize_profile(
                raw_modes[mode],
                mode=mode,
                profile="stable",
                config=config,
                stable_defaults=stable_defaults,
            )
            for mode in SUPPORTED_MODES
            if mode in raw_modes
        }
    }


def resolve_orchestrator_profile(
    config: dict[str, Any],
    *,
    mode: str,
    selection: str = "auto",
) -> OrchestratorProfileResolution:
    requested_mode = _normalize_mode(mode)
    requested_selection = _normalize_selection(selection)
    profiles = load_orchestrator_profiles(config)

    effective_selection = requested_selection
    selection_reason = f"explicit_{requested_selection}"
    if requested_selection == "auto":
        effective_selection = "stable"
        selection_reason = "auto_stable_profile"

    profile = profiles[effective_selection].get(requested_mode)
    if profile is None:
        raise ValueError(
            f"missing orchestrator {effective_selection} profile for mode {requested_mode}"
        )
    return OrchestratorProfileResolution(
        requested_mode=requested_mode,
        effective_mode=profile.mode,
        requested_selection=requested_selection,
        effective_selection=effective_selection,
        selection_reason=selection_reason,
        profile=profile,
    )
