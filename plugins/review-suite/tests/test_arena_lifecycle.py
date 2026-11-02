from __future__ import annotations

import json
import sys
import tomllib
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import review_suite_arena
import review_suite_local
from review_suite_core.arena_lifecycle import (
    RETIREMENTS_FILENAME,
    arena_launch_guard,
    resolve_roster_lifecycle,
)
from review_suite_local import (
    aggregate_records,
    build_reroll_slot_payload,
    default_operational_state,
    eligible_variants,
    load_roster,
    select_pair,
)


def variant(model: str, effort: str = "medium", **overrides) -> dict:
    return {
        "id": f"{model}-{effort}",
        "model_id": model,
        "model": model,
        "reasoning_effort": effort,
        "state": "active",
        "task_classes": ["base_review", "deep_review"],
        **overrides,
    }


def roster(generations=("first", "second", "third")) -> dict:
    return {
        "lineages": {"family": list(generations)},
        "settings": {"selection_mode": "true_scramble", "promotion_min_samples": 20},
        "variants": [
            variant(model, effort)
            for model in generations
            for effort in ("medium", "max")
        ],
    }


def write_roster(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def record(ids: list[str]) -> dict:
    return {
        "round_id": "old-round",
        "recorded_at": "2026-10-01T12:00:00Z",
        "task_class": "base_review",
        "rating_pool_id": "unchanged-pool",
        "reporting_pool": True,
        "placement_v1": {"basis": "more_valid_findings", "groups": [[v] for v in ids]},
        "runs": [
            {
                "variant_id": v,
                "cost_usd": 0.5,
                "elapsed_seconds": 10,
                "low_quality": False,
            }
            for v in ids
        ],
    }


def test_policy_uses_declared_release_order_and_retires_every_effort() -> None:
    data = roster(("v9", "v10", "v2"))
    data["variants"].append(variant("unreleased-v100", state="disabled"))
    resolved = resolve_roster_lifecycle(data)
    retired = [v["id"] for v in resolved["variants"] if v["state"] == "retired"]
    assert retired == ["v9-medium", "v9-max"]
    assert len(eligible_variants(resolved, "base_review")) == 4
    assert len(eligible_variants(resolved, "deep_review")) == 4
    assert data["variants"][0]["state"] == "active"


def test_shipped_roster_retires_legacy_models_without_retiring_luna() -> None:
    data = load_roster(SCRIPT_DIR.parent / "references" / "roster.json")
    for task_class in ("base_review", "deep_review"):
        eligible = {v["id"] for v in eligible_variants(data, task_class)}
        assert not any(v.startswith(("gpt-5.6-terra-", "gpt-5.5-", "gpt-5.6-sol-")) for v in eligible)
        assert "gpt-5.6-luna-medium" in eligible
        assert "gpt-6.1-terra-medium" in eligible
    terra = next(v for v in data["variants"] if v["id"] == "gpt-5.6-terra-medium")
    assert terra["lifecycle"]["status"] == "retired"
    assert terra["lifecycle"]["generation"] == "previous"
    legacy = [v for v in data["variants"] if v["model_id"] in {"gpt-5.4", "gpt-5.4-mini"}]
    assert legacy and all(v["lifecycle"]["status"] == "retired" for v in legacy)


@pytest.mark.parametrize("pool_name,task_class", [
    ("discovery_base", "base_review"), ("discovery_deep", "deep_review"),
    ("arena_base", "base_review"), ("arena_deep", "deep_review"),
])
def test_shipped_pools_can_select_four_after_retirement(pool_name: str, task_class: str) -> None:
    references = SCRIPT_DIR.parent / "references"
    pool = tomllib.loads((references / "arena_settings.toml").read_text())["arena"]["pools"][pool_name]
    result = select_pair(
        roster=load_roster(references / "roster.json"),
        operational_state=default_operational_state(), records=[], task_class=task_class,
        review_cwd=None, seed=None, rating_pool_id=pool["rating_pool_id"],
        variant_groups=pool["variant_groups"], variant_ids=pool.get("variant_ids"),
    )
    assert len(result["runs"]) == 4
    assert not any(run["variant_id"].startswith(("gpt-5.6-terra-", "gpt-5.5-", "gpt-5.6-sol-")) for run in result["runs"])


def test_automatic_load_persists_retirement_despite_roster_downgrade(
    tmp_path: Path,
) -> None:
    path = write_roster(tmp_path / "roster.json", roster())
    state = tmp_path / "state"
    first = load_roster(path, state_dir=state)
    retired_at = first["variants"][0]["lifecycle"]["retired_at"]
    replacement = roster(("first", "second"))
    replacement["variants"].append(variant("first", "low"))
    write_roster(path, replacement)
    restored = load_roster(path, state_dir=state)
    assert all(
        v["state"] == "retired"
        for v in restored["variants"]
        if v["model_id"] == "first"
    )
    assert restored["variants"][0]["lifecycle"]["retired_at"] == retired_at
    assert (
        json.loads((state / RETIREMENTS_FILENAME).read_text())["models"]["first"][
            "retired_at"
        ]
        == retired_at
    )


def test_manual_retirement_is_permanent_and_does_not_backfill_predecessor(
    tmp_path: Path,
) -> None:
    data = roster()
    resolved = resolve_roster_lifecycle(
        data,
        state_dir=tmp_path,
        persist=True,
        retire_model_id="second",
        reason="Too costly",
    )
    repeated = resolve_roster_lifecycle(
        data,
        state_dir=tmp_path,
        persist=True,
        retire_model_id="second",
        reason="Different reason",
    )
    assert [v["model_id"] for v in eligible_variants(repeated, "base_review")] == [
        "third",
        "third",
    ]
    assert (
        repeated["_arena_retirements"]["models"]["second"]
        == resolved["_arena_retirements"]["models"]["second"]
    )
    removed = roster(("third",))
    resolve_roster_lifecycle(removed, state_dir=tmp_path, persist=True)
    assert all(
        v["state"] == "retired"
        for v in resolve_roster_lifecycle(data, state_dir=tmp_path)["variants"]
        if v["model_id"] == "second"
    )


def test_variant_tombstone_prevents_renaming_around_retirement(tmp_path: Path) -> None:
    original = roster()
    resolve_roster_lifecycle(original, state_dir=tmp_path, persist=True)
    changed = roster(("new-family-model", "other-model"))
    changed["variants"][0]["id"] = "first-medium"
    resolved = resolve_roster_lifecycle(changed, state_dir=tmp_path)
    assert resolved["variants"][0]["state"] == "retired"


def test_preview_does_not_create_retirement_state(tmp_path: Path) -> None:
    path = write_roster(tmp_path / "roster.json", roster())
    state = tmp_path / "not-created"
    resolved = load_roster(path, state_dir=state, persist_retirements=False)
    assert resolved["variants"][0]["state"] == "retired"
    assert "retired_at" not in resolved["variants"][0]["lifecycle"]
    assert not state.exists()


def test_sample_dry_run_applies_policy_without_writes(
    tmp_path: Path, monkeypatch
) -> None:
    path = write_roster(tmp_path / "roster.json", roster())
    state = tmp_path / "state"
    emitted = []
    monkeypatch.setattr(review_suite_arena, "emit_toon", emitted.append)
    monkeypatch.setattr(review_suite_arena, "_resolve_review_cwd", lambda _: tmp_path)
    args = review_suite_arena.build_parser().parse_args(
        [
            "sample",
            "--task-class",
            "base_review",
            "--roster",
            str(path),
            "--state-dir",
            str(state),
            "--dry-run",
        ]
    )
    assert review_suite_arena.cmd_sample(args) == 0
    assert all(
        not r["variant_id"].startswith("first-") for r in emitted[0]["reviewers"]
    )
    assert not state.exists()


@pytest.mark.parametrize("mode", ["legacy", "true_scramble", "slight_bias"])
def test_all_automatic_selectors_exclude_retired_models(mode: str) -> None:
    data = roster()
    data["settings"]["selection_mode"] = mode
    for seed in range(10):
        result = select_pair(
            roster=data,
            operational_state=default_operational_state(),
            records=[],
            task_class="base_review",
            review_cwd=None,
            seed=seed,
        )
        assert all(not run["variant_id"].startswith("first-") for run in result["runs"])


def test_configured_stale_groups_keep_field_size_and_pool_identity() -> None:
    data = roster()
    ids = [v["id"] for v in data["variants"]]
    result = select_pair(
        roster=data,
        operational_state=default_operational_state(),
        records=[],
        task_class="base_review",
        review_cwd=None,
        seed=None,
        rating_pool_id="unchanged-pool",
        variant_ids=ids,
        variant_groups=[ids[:4]],
    )
    assert result["rating_pool_id"] == "unchanged-pool"
    assert len(result["runs"]) == 4
    assert all(not run["variant_id"].startswith("first-") for run in result["runs"])
    assert result["rating_pool_variant_ids"] == ids[2:]
    data["variants"][2]["state"] = "retired"
    with pytest.raises(ValueError, match="needs 4 available variants"):
        select_pair(
            roster=data,
            operational_state=default_operational_state(),
            records=[],
            task_class="base_review",
            review_cwd=None,
            seed=None,
            rating_pool_id="unchanged-pool",
            variant_ids=ids,
            variant_groups=[ids[:4]],
        )


def test_retired_history_keeps_metrics_and_existing_ratings(tmp_path: Path) -> None:
    data = roster()
    ids = ["first-medium", "second-medium", "third-max"]
    history = [record(ids)]
    before_roster = deepcopy(data)
    before_roster.pop("lineages")
    before = aggregate_records(before_roster, history, default_operational_state())
    retired = resolve_roster_lifecycle(data, state_dir=tmp_path, persist=True)
    after = aggregate_records(retired, history, default_operational_state())
    before_rows = {
        r["variant_id"]: r for r in before["task_classes"]["base_review"]["leaderboard"]
    }
    rows = {
        r["variant_id"]: r for r in after["task_classes"]["base_review"]["leaderboard"]
    }
    for identity in ids:
        for metric in (
            "elo",
            "sample_count",
            "median_cost_usd",
            "median_elapsed_seconds",
            "wtl",
        ):
            assert rows[identity][metric] == before_rows[identity][metric]
    assert rows["first-medium"]["lifecycle"]["status"] == "retired"
    assert rows["first-medium"]["lifecycle"]["retired_at"]
    # Removal from the catalog does not remove the persisted historical row.
    retired["variants"] = [v for v in retired["variants"] if v["model_id"] != "first"]
    removed = aggregate_records(retired, history, default_operational_state())
    assert any(
        r["variant_id"] == "first-medium" and r["lifecycle"]["status"] == "retired"
        for r in removed["task_classes"]["base_review"]["leaderboard"]
    )


def test_disabled_history_remains_available_without_becoming_eligible() -> None:
    data = roster(("first", "second"))
    data["variants"][0]["state"] = "disabled"
    result = aggregate_records(
        data, [record(["first-medium", "second-medium"])], default_operational_state()
    )
    row = next(r for r in result["task_classes"]["base_review"]["leaderboard"] if r["variant_id"] == "first-medium")
    assert row["sample_count"] == 1
    assert row["lifecycle"] == {"status": "disabled", "lineage": "family", "generation": "previous"}
    assert "first-medium" not in {v["id"] for v in eligible_variants(data, "base_review")}
    assert (
        len(result["task_classes"]["base_review"]["recent_rounds"][0]["participants"])
        == 2
    )


def test_prepared_round_rechecks_current_roster_before_any_launch(
    tmp_path: Path, monkeypatch
) -> None:
    path = write_roster(tmp_path / "roster.json", roster(("first", "second")))
    state = tmp_path / "state"
    stale_roster = load_roster(path, state_dir=state)
    write_roster(path, roster())
    payload = {
        "round_id": "queued",
        "selection_mode": "configured",
        "status": "sampled",
        "runs": [{"slot": "alpha", "variant_id": "first-medium"}],
        "roster_path": str(path),
    }
    monkeypatch.setattr(
        review_suite_local,
        "_launch_reviewer_process_unchecked",
        lambda **_: pytest.fail("retired model was launched"),
    )
    with pytest.raises(ValueError, match="retired"):
        review_suite_local.launch_round(
            round_payload=payload,
            roster=stale_roster,
            state_dir=state,
            review_cwd=tmp_path,
            prompt="review",
            review_scope={},
        )


def test_final_dispatch_guard_observes_retirement_after_selection(
    tmp_path: Path,
) -> None:
    path = write_roster(tmp_path / "roster.json", roster(("first", "second")))
    stale = load_roster(path)
    resolve_roster_lifecycle(
        stale, state_dir=tmp_path, persist=True, retire_model_id="second"
    )
    nested = tmp_path / "orchestrator" / "review-rounds"
    with pytest.raises(ValueError, match="permanently retired"):
        with arena_launch_guard(stale, state_dir=nested, variant_id="second-max"):
            pytest.fail("retired reviewer passed the final guard")


def test_retirement_during_dispatch_keeps_other_slots_recoverable(
    tmp_path: Path, monkeypatch
) -> None:
    path = write_roster(tmp_path / "roster.json", roster(("first", "second")))
    data = load_roster(path)
    launched = []

    def fake_launch(**kwargs):
        run = kwargs["run"]
        launched.append(run["variant_id"])
        run.update(
            review_status="completed",
            reviewer_output="Existing successful result",
            grade_blocked=False,
        )
        return run

    monkeypatch.setattr(
        review_suite_local, "_launch_reviewer_process_unchecked", fake_launch
    )
    monkeypatch.setattr(
        review_suite_local.time,
        "sleep",
        lambda _: resolve_roster_lifecycle(
            data, state_dir=tmp_path, persist=True, retire_model_id="second"
        ),
    )
    ids = ["first-medium", "second-medium", "first-max", "second-max"]
    payload = {
        "round_id": "staggered",
        "selection_mode": "configured",
        "status": "sampled",
        "roster_path": str(path),
        "runs": [
            {"slot": slot, "variant_id": identity}
            for slot, identity in zip(("alpha", "bravo", "charlie", "delta"), ids)
        ],
    }
    result = review_suite_local.launch_round(
        round_payload=payload,
        roster=data,
        state_dir=tmp_path,
        review_cwd=tmp_path,
        prompt="review",
        review_scope={},
    )
    assert launched == ["first-medium", "first-max"]
    assert result["status"] == "completed"
    assert result["runs"][1]["grade_block_reason"] == "selected_model_retired"
    assert result["runs"][3]["grade_block_reason"] == "selected_model_retired"
    assert result["runs"][0]["reviewer_output"] == "Existing successful result"
    assert (
        json.loads((tmp_path / "rounds" / "staggered.json").read_text())["status"]
        == "completed"
    )


def test_old_orchestrator_runtime_cannot_revive_a_retired_generation(
    tmp_path: Path, monkeypatch
) -> None:
    old_path = write_roster(tmp_path / "old-roster.json", roster(("first", "second")))
    new_path = write_roster(tmp_path / "new-roster.json", roster())
    stale = load_roster(old_path)
    old_path.unlink()  # The runtime that prepared the queued round was cleaned up.
    monkeypatch.setattr(review_suite_local, "default_roster_path", lambda: new_path)
    monkeypatch.setattr(
        review_suite_local,
        "_launch_reviewer_process_unchecked",
        lambda **_: pytest.fail("retired reviewer launched"),
    )
    payload = {
        "round_id": "queued",
        "arena_round": True,
        "orchestrator_step": "arena-base-review",
        "status": "sampled",
        "roster_path": str(old_path),
        "runs": [{"slot": "alpha", "variant_id": "first-medium"}],
    }
    with pytest.raises(ValueError, match="retired"):
        review_suite_local.launch_round(
            round_payload=payload,
            roster=stale,
            state_dir=tmp_path / "state",
            review_cwd=tmp_path,
            prompt="review",
            review_scope={},
        )


def test_capacity_retry_keeps_existing_attempt_without_new_call(
    tmp_path: Path, monkeypatch
) -> None:
    path = write_roster(tmp_path / "roster.json", roster(("first", "second")))
    stale = load_roster(path)
    resolve_roster_lifecycle(
        stale, state_dir=tmp_path, persist=True, retire_model_id="second"
    )
    payload = {
        "round_id": "running",
        "selection_mode": "configured",
        "roster_path": str(path),
    }
    run = {
        "variant_id": "second-max",
        "slot": "alpha",
        "reviewer_output": "retained output",
    }
    monkeypatch.setattr(
        review_suite_local,
        "_summarize_live_run",
        lambda _: {"grade_block_reason": "selected_model_at_capacity"},
    )
    monkeypatch.setattr(
        review_suite_local,
        "_launch_reviewer_process_unchecked",
        lambda **_: pytest.fail("retired capacity retry launched"),
    )
    assert (
        review_suite_local._maybe_retry_capacity_run(
            round_payload=payload,
            run=run,
            indexed={v["id"]: v for v in stale["variants"]},
            state_dir=tmp_path,
            review_cwd=tmp_path,
        )
        is False
    )
    assert run["reviewer_output"] == "retained output"


def test_retirement_during_retry_delay_finishes_round_for_reroll(
    tmp_path: Path, monkeypatch
) -> None:
    path = write_roster(tmp_path / "roster.json", roster(("first", "second")))
    stale = load_roster(path)
    run = {
        "variant_id": "second-max",
        "slot": "alpha",
        "model": "second",
        "reasoning_effort": "max",
        "stdout_path": str(tmp_path / "out"),
        "stderr_path": str(tmp_path / "err"),
    }
    payload = {
        "round_id": "running",
        "task_class": "base_review",
        "status": "running",
        "selection_mode": "configured",
        "roster_path": str(path),
        "runs": [run],
    }
    monkeypatch.setattr(
        review_suite_local,
        "_summarize_live_run",
        lambda _: {
            "grade_block_reason": "selected_model_at_capacity",
            "reviewer_output": "Existing capacity response",
        },
    )
    monkeypatch.setattr(review_suite_local, "_process_is_running", lambda _: False)
    monkeypatch.setattr(
        review_suite_local,
        "_launch_reviewer_process_unchecked",
        lambda **_: pytest.fail("retired capacity retry launched"),
    )
    monkeypatch.setattr(
        review_suite_local.time,
        "sleep",
        lambda _: resolve_roster_lifecycle(
            stale, state_dir=tmp_path, persist=True, retire_model_id="second"
        ),
    )
    completed = review_suite_local.collect_round_results(
        round_payload=payload, roster=stale, state_dir=tmp_path, review_cwd=tmp_path
    )
    assert completed["status"] == "completed"
    assert completed["runs"][0]["grade_block_reason"] == "selected_model_retired"
    assert completed["runs"][0]["reviewer_output"] == "Existing capacity response"
    assert not completed["runs"][0].get("capacity_cleanup_pending")


def test_configured_reroll_replaces_newly_retired_model() -> None:
    data = roster()
    ids = ["first-medium", "second-medium", "third-medium", "third-max"]
    payload = {
        "round_id": "completed",
        "task_class": "base_review",
        "status": "completed",
        "selection_mode": "configured",
        "rating_pool_id": "unchanged-pool",
        "rating_pool_variant_ids": [v["id"] for v in data["variants"]],
        "runs": [
            {
                "slot": slot,
                "variant_id": identity,
                "model": identity.rsplit("-", 1)[0],
                "reasoning_effort": identity.rsplit("-", 1)[1],
                "reviewer_output": "Existing result",
                "review_status": "completed",
                "grade_blocked": False,
            }
            for slot, identity in zip(("alpha", "bravo", "charlie", "delta"), ids)
        ],
    }
    result = build_reroll_slot_payload(
        round_payload=payload,
        roster=data,
        operational_state=default_operational_state(),
        records=[],
        slot="alpha",
        seed=1,
    )
    assert result["runs"][0]["variant_id"] == "second-max"
    assert result["runs"][1]["reviewer_output"] == "Existing result"


def test_corrupt_ledger_fails_closed(tmp_path: Path) -> None:
    (tmp_path / RETIREMENTS_FILENAME).write_text(
        '{"version": 1, "models": [], "variants": {}}'
    )
    with pytest.raises(ValueError, match="invalid models"):
        resolve_roster_lifecycle(roster(), state_dir=tmp_path, persist=True)


def test_manual_retire_command_refreshes_archive_without_running_models(
    tmp_path: Path, monkeypatch
) -> None:
    path = write_roster(tmp_path / "roster.json", roster(("first", "second")))
    state = tmp_path / "state"
    state.mkdir()
    (state / "runs.jsonl").write_text(
        json.dumps(record(["first-medium", "second-medium"])) + "\n"
    )
    monkeypatch.setattr(review_suite_arena, "emit_toon", lambda _: None)
    args = review_suite_arena.build_parser().parse_args(
        [
            "retire-model",
            "--model-id",
            "first",
            "--reason",
            "No longer useful",
            "--roster",
            str(path),
            "--state-dir",
            str(state),
        ]
    )
    assert review_suite_arena.cmd_retire_model(args) == 0
    summary = json.loads((state / "summary.json").read_text())
    row = next(
        r
        for r in summary["task_classes"]["base_review"]["leaderboard"]
        if r["variant_id"] == "first-medium"
    )
    assert row["lifecycle"]["status"] == "retired"
    assert row["lifecycle"]["reason"] == "No longer useful"
    assert row["sample_count"] == 1
