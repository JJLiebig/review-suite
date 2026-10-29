from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from migrate_review_names import migrate_names, rename_payload


def _save(path: Path, payload: object) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    original = json.dumps(payload, indent=2).encode()
    path.write_bytes(original)
    return original


def test_offline_cutover_preserves_history_and_corrects_known_deep_signoffs(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    state.mkdir()
    settings = state / "settings.toml"
    settings.write_text(
        '[normal]\nmodel="custom"\nreasoning="high"\n'
        "[orchestrator.stable_defaults]\nnormal_arena_loops=7\n"
        '[arena.pools.arena_phase]\nrating_pool_id="arena-phase-gpt-5.6-v1"\n'
    )
    base_id = "phase_review-repo-abcdef-20261001T120000Z-01234567"
    deep_id = "phase_review-repo-abcdef-20261001T120001Z-12345678"
    round_path = state / "orchestrator/review-rounds/rounds" / f"{deep_id}.json"
    prompt = (
        "Review this phase_review. Use --mode normal and keep the original evidence."
    )
    original_round = _save(
        round_path,
        {
            "round_id": deep_id,
            "task_class": "phase_review",
            "public_task": "review_t1",
            "orchestrator_step": "deep-signoff",
            "requested_prompt": prompt,
            "rating_pool_id": "arena-phase-gpt-5.6-v1",
            "runs": [{"cost_usd": 1.25}],
        },
    )
    cycle_path = state / "orchestrator/cycles/cycle.json"
    _save(
        cycle_path,
        {
            "mode": {"requested": "normal", "effective": "normal"},
            "review_plan": {
                "steps": [
                    {"name": "precision-signoff", "kind": "review", "count": 2},
                    {"name": "deep-signoff", "kind": "review", "count": 2},
                ]
            },
            "rounds": [{"round_id": deep_id, "lane": "review_t1"}],
            "pending_action": {"step": "arena-pr-review-2", "lane": "review_t3"},
        },
    )
    run = {
        "round_id": base_id,
        "task_class": "phase_review",
        "recorded_at": "original-time",
        "rating_pool_id": "arena-phase-gpt-5.6-v1",
        "runs": [{"cost_usd": 1.25}],
    }
    runs = state / "runs.jsonl"
    runs.write_text(json.dumps(run) + "\n")
    snapshot = tmp_path / "api/snapshot.json"
    original_snapshot = _save(
        snapshot,
        {
            "recordCount": 1,
            "tasks": {
                "review_t1": {
                    "recentMatches": [{"taskName": "review_t1", "roundId": base_id}]
                },
                "review_t3": {"recentMatches": []},
            },
        },
    )
    originals = {
        path: path.read_bytes()
        for path in (settings, round_path, cycle_path, runs, snapshot)
    }

    assert migrate_names(state, [snapshot]) == {"status": "dry_run", "files": 5}
    assert all(path.read_bytes() == original for path, original in originals.items())
    result = migrate_names(state, [snapshot], apply=True)
    backup = Path(result["backup"])
    manifest = json.loads((backup / "manifest.json").read_text())
    assert all(
        (backup / manifest[str(path)]).read_bytes() == original
        for path, original in originals.items()
    )
    assert (backup / round_path.relative_to(state)).read_bytes() == original_round
    assert (backup / manifest[str(snapshot)]).read_bytes() == original_snapshot

    migrated_round = json.loads(round_path.read_text())
    assert migrated_round == {
        **json.loads(original_round),
        "task_class": "deep_review",
        "public_task": "review_deep",
    }
    migrated_cycle = json.loads(cycle_path.read_text())
    assert migrated_cycle["rounds"][0]["lane"] == "review_deep"
    assert migrated_cycle["review_plan"]["steps"] == [
        {
            "name": "base-signoff",
            "kind": "review",
            "count": 2,
            "lane": "review_base",
            "task_class": "base_review",
        },
        {
            "name": "deep-signoff",
            "kind": "review",
            "count": 2,
            "lane": "review_deep",
            "task_class": "deep_review",
        },
    ]
    assert migrated_cycle["mode"] == {"requested": "base", "effective": "base"}
    assert migrated_cycle["pending_action"] == {
        "step": "arena-deep-review-2",
        "lane": "review_deep",
    }
    assert json.loads(runs.read_text()) == {**run, "task_class": "base_review"}
    migrated_settings = tomllib.loads(settings.read_text())
    assert migrated_settings["base"] == {"model": "custom", "reasoning": "high"}
    assert migrated_settings["orchestrator"]["stable_defaults"]["base_arena_loops"] == 7
    assert (
        migrated_settings["arena"]["pools"]["arena_base"]["rating_pool_id"]
        == "arena-phase-gpt-5.6-v1"
    )
    assert json.loads(snapshot.read_text())["tasks"]["review_base"][
        "recentMatches"
    ] == [
        {"taskName": "review_base", "roundId": base_id},
    ]
    assert migrate_names(state, [snapshot], apply=True) == {
        "status": "unchanged",
        "files": 0,
    }


@pytest.mark.parametrize("problem", ["invalid-json", "conflicting-settings"])
def test_preflight_failure_changes_no_saved_files(tmp_path: Path, problem: str) -> None:
    state = tmp_path / "state"
    state.mkdir()
    settings = state / "settings.toml"
    settings.write_text('[normal]\nreasoning="medium"\n')
    runs = state / "runs.jsonl"
    if problem == "invalid-json":
        runs.write_text('{"task_class":"phase_review"}\ninvalid\n')
    else:
        settings.write_text('[normal]\nreasoning="medium"\n[base]\nreasoning="high"\n')
        runs.write_text('{"task_class":"phase_review"}\n')
    originals = (settings.read_bytes(), runs.read_bytes())
    with pytest.raises(ValueError):
        migrate_names(state, [], apply=True)
    assert (settings.read_bytes(), runs.read_bytes()) == originals
    assert not list(state.glob("base-deep-backup-*"))


def test_deep_classification_uses_stage_evidence_and_keeps_unknown_fields() -> None:
    original = {
        "orchestrator_step": "precision-signoff",
        "task_class": "phase_review",
        "public_task": "review_t1",
        "runs": [{"reasoning_effort": "xhigh"}],
        "future_field": {"text": "normal", "rating_pool_id": "pr_review"},
        "review_brief": {"mode": "normal", "name": "precision-signoff"},
        "reviewer_output": {"task_class": "phase_review"},
    }
    migrated = rename_payload(original, deep_rounds=set())
    assert migrated["task_class"] == "base_review"
    assert migrated["public_task"] == "review_base"
    assert migrated["future_field"] == original["future_field"]
    assert migrated["review_brief"] == original["review_brief"]
    assert migrated["reviewer_output"] == original["reviewer_output"]


@pytest.mark.parametrize("deep_source", ["signoff", "arena"])
def test_nested_followup_chains_keep_the_deep_origin_after_restart(
    tmp_path: Path, deep_source: str
) -> None:
    state = tmp_path / "state"
    round_dir = state / "orchestrator/review-rounds/rounds"
    source_id = "z-original"
    source = {
        "round_id": source_id,
        "orchestrator_step": "deep-signoff"
        if deep_source == "signoff"
        else "arena-pr-review",
        "task_class": "phase_review" if deep_source == "signoff" else "pr_review",
        "public_task": "review_t1" if deep_source == "signoff" else "review_t3",
    }
    _save(round_dir / f"{source_id}.json", source)
    for round_id, parent_id in (
        ("b-followup", source_id),
        ("a-followup", "b-followup"),
    ):
        _save(
            round_dir / f"{round_id}.json",
            {
                "round_id": round_id,
                "orchestrator_step": "followup",
                "task_class": "phase_review",
                "public_task": "review-followup",
                "review_scope": {"source_round_id": parent_id},
            },
        )
    cycle_file = state / "orchestrator/cycles/cycle.json"
    _save(
        cycle_file, {"rounds": [{"round_id": "a-followup", "lane": "review-followup"}]}
    )
    migrate_names(state, [], apply=True)
    for name in ("a-followup", "b-followup"):
        migrated = json.loads((round_dir / f"{name}.json").read_text())
        assert migrated["task_class"] == "deep_review"
        assert migrated["public_task"] == "review-followup"
    assert json.loads(cycle_file.read_text())["rounds"] == [
        {
            "round_id": "a-followup",
            "lane": "review-followup",
            "task_class": "deep_review",
        },
    ]
    assert migrate_names(state, [], apply=True) == {"status": "unchanged", "files": 0}
