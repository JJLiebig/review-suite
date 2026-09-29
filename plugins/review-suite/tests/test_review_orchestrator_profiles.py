from __future__ import annotations

import sys
from collections import Counter
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from review_suite_core.config import load_config
from review_suite_core.orchestrator_profiles import (
    SUPPORTED_SELECTION_REASONS,
    OrchestratorProfileStep,
    load_orchestrator_profiles,
    resolve_orchestrator_profile,
)
from review_suite_local import default_roster_path, eligible_variants, load_roster


def _step_summary(
    step: OrchestratorProfileStep,
) -> tuple[str, str, int | None, bool]:
    return (
        step.kind,
        step.name,
        step.count,
        step.rerun_on_findings,
    )


def test_default_stable_profiles_cover_all_modes(tmp_path: Path) -> None:
    config = load_config(tmp_path / "state")

    profiles = load_orchestrator_profiles(config)

    assert set(profiles["stable"]) == {"normal", "deep", "fast"}
    assert config["arena"]["enabled"] is False
    assert config["orchestrator"]["calibration"]["auto_promotion_enabled"] is False
    assert [_step_summary(step) for step in profiles["stable"]["normal"].steps] == [
        ("review", "precision-signoff", 2, True),
    ]
    assert [step.name for step in profiles["stable"]["deep"].steps] == [
        "precision-signoff",
        "deep-signoff",
    ]
    assert profiles["stable"]["normal"].steps[-1].rerun_on_findings is True
    assert profiles["stable"]["deep"].steps[0].rerun_on_findings is True
    assert profiles["stable"]["deep"].steps[-1].rerun_on_findings is True
    assert [_step_summary(step) for step in profiles["stable"]["fast"].steps] == [
        ("review", "fast-signoff", 2, False)
    ]
    assert profiles["stable"]["fast"].steps[0].max_review_rounds == 2
    assert set(profiles) == {"stable"}


def test_stable_profile_rejects_obsolete_deslop_setting(tmp_path: Path) -> None:
    config = deepcopy(load_config(tmp_path / "state"))
    config["orchestrator"]["profiles"]["stable"]["normal"]["deslop_enabled"] = False

    with pytest.raises(ValueError, match=r"deslop_enabled.*--skip-deslop"):
        load_orchestrator_profiles(config)


def test_profile_step_kind_defaults_to_review(tmp_path: Path) -> None:
    config = deepcopy(load_config(tmp_path / "state"))
    config["orchestrator"]["profiles"]["stable"]["normal"]["steps"] = [
        {
            "name": "precision",
            "count": 1,
            "model": "gpt-5.5",
            "reasoning_effort": "medium",
        }
    ]

    profiles = load_orchestrator_profiles(config)

    assert profiles["stable"]["normal"].steps[0].kind == "review"


def test_profile_step_rejects_conflicting_findings_policies(tmp_path: Path) -> None:
    config = deepcopy(load_config(tmp_path / "state"))
    config["orchestrator"]["profiles"]["stable"]["normal"]["steps"] = [
        {
            "name": "precision",
            "count": 1,
            "model": "gpt-5.5",
            "reasoning_effort": "medium",
            "rerun_on_findings": True,
            "max_review_rounds": 2,
        }
    ]

    with pytest.raises(
        ValueError, match="cannot combine rerun_on_findings with max_review_rounds"
    ):
        load_orchestrator_profiles(config)


def test_arena_disabled_omits_all_multi_model_steps(
    tmp_path: Path,
) -> None:
    config = deepcopy(load_config(tmp_path / "state"))
    config["orchestrator"]["stable_defaults"]["normal_arena_loops"] = 2
    config["orchestrator"]["stable_defaults"]["deep_arena_loops"] = 1

    profiles = load_orchestrator_profiles(config)

    assert [step.name for step in profiles["stable"]["normal"].steps] == [
        "precision-signoff",
    ]
    assert all(
        "arena-phase" not in step.name for step in profiles["stable"]["deep"].steps
    )
    assert all("arena-pr" not in step.name for step in profiles["stable"]["deep"].steps)


def test_disabled_first_step_in_loop_block_does_not_drop_enabled_steps(
    tmp_path: Path,
) -> None:
    config = deepcopy(load_config(tmp_path / "state"))
    config["arena"]["enabled"] = False
    config["orchestrator"]["stable_defaults"]["test_loops"] = 2
    config["orchestrator"]["profiles"]["stable"]["normal"]["steps"] = [
        {
            "kind": "arena",
            "name": "arena-phase-review",
            "lane": "review_t1",
            "task_class": "phase_review",
            "loop_ref": "test_loops",
            "enabled_ref": "arena.enabled",
        },
        {
            "name": "broad-discovery",
            "count": 1,
            "model_ref": "signoff_normal_model",
            "loop_ref": "test_loops",
        },
    ]

    profiles = load_orchestrator_profiles(config)

    assert [step.name for step in profiles["stable"]["normal"].steps] == [
        "broad-discovery-1",
        "broad-discovery-2",
    ]


def test_arena_enabled_inserts_only_configured_arena_steps(
    tmp_path: Path,
) -> None:
    config = deepcopy(load_config(tmp_path / "state"))
    config["arena"]["enabled"] = True
    config["orchestrator"]["stable_defaults"].update(
        {
            "normal_arena_loops": 2,
            "deep_arena_loops": 3,
        }
    )

    profiles = load_orchestrator_profiles(config)

    normal = profiles["stable"]["normal"].steps
    assert [(step.kind, step.name, step.lane, step.task_class) for step in normal] == [
        ("arena", "arena-phase-review-1", "review_t1", "phase_review"),
        ("arena", "arena-phase-review-2", "review_t1", "phase_review"),
        ("review", "precision-signoff", None, None),
    ]
    assert [step.reporting_pool for step in normal if step.kind == "arena"] == [
        True,
        True,
    ]

    deep = profiles["stable"]["deep"].steps
    assert [(step.kind, step.name, step.lane, step.task_class) for step in deep] == [
        ("review", "precision-signoff", None, None),
        ("arena", "arena-pr-review-1", "review_t3", "pr_review"),
        ("arena", "arena-pr-review-2", "review_t3", "pr_review"),
        ("arena", "arena-pr-review-3", "review_t3", "pr_review"),
        ("review", "deep-signoff", None, None),
    ]


def test_default_arena_pool_schedules_are_balanced(tmp_path: Path) -> None:
    config = load_config(tmp_path / "state")
    pools = config["arena"]["pools"]
    roster = load_roster(default_roster_path())
    for pool_name, task_class in (
        ("arena_phase", "phase_review"),
        ("arena_deep", "pr_review"),
    ):
        claude_variants = {
            variant["id"]
            for variant in eligible_variants(roster, task_class)
            if variant["model"].startswith("claude::")
        }
        assert claude_variants <= set(pools[pool_name]["variant_ids"])
    assert {pool["rating_pool_id"] for pool in pools.values()} == {
        "discovery-phase-gpt-5.6-v1",
        "discovery-deep-gpt-5.6-v1",
        "arena-phase-gpt-5.6-v1",
        "arena-deep-gpt-5.6-v1",
    }
    assert {name for name, pool in pools.items() if pool.get("reporting")} == {
        "arena_phase",
        "arena_deep",
    }
    for pool_name in ("discovery_phase", "discovery_deep"):
        assert all(
            len(group) == 4 and len(set(group)) == 4
            for group in pools[pool_name]["variant_groups"]
        )
    for pool_name in ("arena_phase", "arena_deep"):
        candidates = set(pools[pool_name]["variant_ids"])
        groups = pools[pool_name]["variant_groups"]
        scheduled = [variant for group in groups for variant in group]
        assert all(len(group) == 4 and len(set(group)) == 4 for group in groups)
        # Candidates may exceed the round-robin schedule; new candidates join the
        # balanced phase without a new schedule. Scheduled variants stay balanced.
        assert set(scheduled) <= candidates
        assert Counter(scheduled) == Counter({variant: 4 for variant in set(scheduled)})
        pairs = Counter(
            pair for group in groups for pair in combinations(sorted(group), 2)
        )
        assert max(pairs.values()) <= 2
        assert len(scheduled) == 4 * len(set(scheduled))


def test_arena_candidate_list_must_cover_bootstrap_schedule(tmp_path: Path) -> None:
    config = deepcopy(load_config(tmp_path / "state"))
    config["arena"]["enabled"] = True
    config["orchestrator"]["stable_defaults"]["normal_arena_loops"] = 1
    phase_pool = config["arena"]["pools"]["arena_phase"]
    phase_pool["variant_ids"].remove(phase_pool["variant_groups"][0][0])

    with pytest.raises(ValueError, match="must include every scheduled variant"):
        load_orchestrator_profiles(config)


def test_arena_steps_reject_mismatched_lane_and_task_class(tmp_path: Path) -> None:
    config = deepcopy(load_config(tmp_path / "state"))
    config["arena"]["enabled"] = True
    config["orchestrator"]["stable_defaults"]["normal_arena_loops"] = 1
    config["orchestrator"]["profiles"]["stable"]["normal"]["steps"][0]["lane"] = (
        "review_t1"
    )
    config["orchestrator"]["profiles"]["stable"]["normal"]["steps"][0]["task_class"] = (
        "pr_review"
    )

    with pytest.raises(
        ValueError, match="lane must be review_t3 for task_class pr_review"
    ):
        load_orchestrator_profiles(config)


@pytest.mark.parametrize("mode", ["normal", "deep", "fast"])
def test_auto_selection_uses_stable_profile_when_configured(
    tmp_path: Path, mode: str
) -> None:
    config = load_config(tmp_path / "state")

    resolution = resolve_orchestrator_profile(config, mode=mode, selection="auto")

    assert resolution.requested_mode == mode
    assert resolution.effective_mode == mode
    assert resolution.requested_selection == "auto"
    assert resolution.effective_selection == "stable"
    assert resolution.selection_reason == "auto_stable_profile"
    assert resolution.steps


def test_brief_mode_is_not_supported(tmp_path: Path) -> None:
    config = load_config(tmp_path / "state")

    with pytest.raises(
        ValueError, match="orchestrator mode must be one of: fast, normal, deep"
    ):
        resolve_orchestrator_profile(config, mode="brief", selection="auto")


def test_explicit_stable_selection_records_reason_and_skips_grading(
    tmp_path: Path,
) -> None:
    config = load_config(tmp_path / "state")

    resolution = resolve_orchestrator_profile(config, mode="normal", selection="stable")

    assert resolution.effective_selection == "stable"
    assert resolution.selection_reason == "explicit_stable"
    assert resolution.selection_reason in SUPPORTED_SELECTION_REASONS


def test_benchmark_selection_is_not_supported(tmp_path: Path) -> None:
    config = load_config(tmp_path / "state")

    with pytest.raises(ValueError, match="orchestrator selection must be one of"):
        resolve_orchestrator_profile(config, mode="normal", selection="benchmark")


def test_auto_selection_requires_stable_profile(tmp_path: Path) -> None:
    config = deepcopy(load_config(tmp_path / "state"))
    del config["orchestrator"]["profiles"]["stable"]["normal"]

    with pytest.raises(
        ValueError, match="missing orchestrator stable profile for mode normal"
    ):
        resolve_orchestrator_profile(config, mode="normal", selection="auto")


def test_calibration_auto_promotion_is_not_enabled(tmp_path: Path) -> None:
    config = deepcopy(load_config(tmp_path / "state"))
    config["orchestrator"]["calibration"]["auto_promotion_enabled"] = True

    with pytest.raises(ValueError, match="auto_promotion_enabled must be false"):
        load_orchestrator_profiles(config)
