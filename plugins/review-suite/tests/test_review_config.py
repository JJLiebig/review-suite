from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from review_suite_core.config import load_config
from review_suite_core.orchestrator_profiles import load_orchestrator_profiles


def test_deep_model_is_reserved_for_final_signoff(tmp_path: Path) -> None:
    (tmp_path / "settings.toml").write_text(
        '[base]\nmodel = "gpt-5.6-sol"\nreasoning = "medium"\n'
        '[deep]\nmodel = "gpt-6-astra"\nreasoning = "xhigh"\n',
        encoding="utf-8",
    )
    steps = load_orchestrator_profiles(load_config(tmp_path))["stable"]["deep"].steps
    signoffs = [step for step in steps if step.kind != "arena"]
    assert [(step.model, step.reasoning_effort) for step in signoffs] == [
        ("gpt-5.6-sol", "medium"),
        ("gpt-6-astra", "xhigh"),
    ]


def test_legacy_migration_preserves_non_model_settings_only(tmp_path: Path) -> None:
    legacy = {
        "lens": {
            "default": {
                "model": "gpt-5.5",
                "reasoning_effort": "xhigh",
                "service_tier": "flex",
            },
            "review-deslop": {"model": "gpt-5.5", "service_tier": "fast"},
        },
        "arena": {
            "enabled": True,
            "pools": {
                "arena_base": {
                    "rating_pool_id": "existing-ratings",
                    "variant_ids": ["gpt-5.5-medium"],
                    "variant_groups": [["gpt-5.5-medium"]],
                }
            },
        },
        "orchestrator": {
            "selection": "stable",
            "stable_defaults": {
                "signoff_base_model": "gpt-5.5-medium",
                "signoff_deep_model": "gpt-5.5-xhigh",
            },
            "profiles": {
                "stable": {
                    "fast": {
                        "steps": [
                            {
                                "name": "custom-signoff",
                                "count": 1,
                                "model": "gpt-5.5",
                                "reasoning_effort": "high",
                                "max_review_rounds": 2,
                            }
                        ]
                    }
                }
            },
        },
        "future_setting": {
            "path": 'C:\\some folder\\"quoted"',
            "enabled": False,
            "ratio": 1.5,
            "tags": ["one", "two"],
            "unused": None,
        },
    }
    original = json.dumps(legacy)
    (tmp_path / "config.json").write_text(original, encoding="utf-8")
    (tmp_path / "runs.jsonl").write_text("historical results", encoding="utf-8")
    config = load_config(tmp_path)
    migrated_text = (tmp_path / "settings.toml").read_text()
    migrated = tomllib.loads(migrated_text)
    assert "gpt-5.5" not in migrated_text
    assert "base" not in migrated or "model" not in migrated["base"]
    assert migrated["arena"]["pools"]["arena_base"] == {
        "rating_pool_id": "existing-ratings"
    }
    assert config["arena"]["enabled"] is True
    assert config["orchestrator"]["selection"] == "stable"
    assert config["future_setting"]["path"] == legacy["future_setting"]["path"]
    assert config["future_setting"]["tags"] == ["one", "two"]
    assert migrated["jobs"]["deslop"]["service_tier"] == "fast"
    assert migrated["base"]["service_tier"] == "flex"
    fast = load_orchestrator_profiles(config)["stable"]["fast"].steps[0]
    assert fast.count == 1
    assert fast.max_review_rounds == 2
    assert (tmp_path / "config.json").read_text() == original
    assert (tmp_path / "runs.jsonl").read_text() == "historical results"
    assert load_config(tmp_path) == config
    assert (tmp_path / "settings.toml").read_text() == migrated_text


def test_existing_toml_is_authoritative_over_legacy_json(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("invalid old JSON", encoding="utf-8")
    override = "[arena]\nenabled = true\n"
    (tmp_path / "settings.toml").write_text(override, encoding="utf-8")
    assert load_config(tmp_path)["arena"]["enabled"] is True
    assert (tmp_path / "settings.toml").read_text() == override


@pytest.mark.parametrize(
    "legacy", ["{broken", "[]", '{"orchestrator": {"selection": "invalid"}}']
)
def test_failed_migration_leaves_legacy_and_no_stub(
    tmp_path: Path, legacy: str
) -> None:
    (tmp_path / "config.json").write_text(legacy, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(tmp_path)
    assert (tmp_path / "config.json").read_text() == legacy
    assert not (tmp_path / "settings.toml").exists()


def test_invalid_user_settings_are_not_overwritten(tmp_path: Path) -> None:
    override = "[broken"
    (tmp_path / "settings.toml").write_text(override, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(tmp_path)
    assert (tmp_path / "settings.toml").read_text() == override


def test_concurrent_first_loads_publish_complete_settings(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    (tmp_path / "config.json").write_text(
        json.dumps({"arena": {"enabled": True}}),
        encoding="utf-8",
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(load_config, [tmp_path] * 16))
    assert all(result == results[0] for result in results)
    assert load_config(tmp_path)["arena"]["enabled"] is True
    assert list(tmp_path.glob("*.tmp")) == []


def test_public_config_rejects_external_arena_publish(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "config.json").write_text(
        json.dumps({"privacy": {"arena_external_publish_enabled": True}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="arena_external_publish_enabled"):
        load_config(state_dir)


def test_public_config_rejects_invalid_orchestrator_selection(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "config.json").write_text(
        json.dumps({"orchestrator": {"selection": "scramble"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="orchestrator.selection"):
        load_config(state_dir)


def test_no_external_arena_publish_endpoint_in_scripts() -> None:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    banned = (
        "arena" + "-web",
        "review" + "arena",
        "review" + "-arena",
        "arena." + "private-host",
        "review" + "arena.",
    )
    hits: list[str] = []
    for path in scripts_dir.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for marker in banned:
            if marker in text:
                hits.append(f"{path.name}:{marker}")

    assert hits == []
