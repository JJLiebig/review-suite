from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import review_suite_local
from review_suite_core.opencode_driver import REVIEW_METADATA_PREFIX
from review_suite_local import (
    _apply_capacity_cooldowns,
    _classify_review_result,
    _launch_reviewer_process,
    _live_review_thread,
    _maybe_retry_capacity_run,
    _heartbeat_status_line,
    _print_live_completed_run,
    _print_stall_warnings,
    _print_transport_events,
    _process_is_running,
    _reviewer_wait_line,
    _reroll_candidate_variants,
    _running_status_line,
    _transport_hung_after_output,
    _transport_stalled,
    LOW_QUALITY_LOSS_REASON_BASES,
    aggregate_records,
    build_record_from_grade,
    compact_benchmark_record,
    compact_benchmark_run,
    compact_round_files,
    cleanup_stale_ungraded_rounds,
    build_reroll_slot_payload,
    classify_review_capture,
    collect_round_results,
    ensure_clean_git_worktree,
    enrich_record_repo_names,
    find_blocking_rounds_for_caller,
    format_cooldown_until_for_display,
    guard_no_stage_step_down,
    normalize_record_review_cwd_value,
    normalize_service_tier,
    output_isatty,
    placement_record,
    public_round_result,
    repo_name_from_round_payload,
    reviewer_completion_status,
    select_pair,
    terminal_review_command,
    update_elo,
    variant_service_tier,
    ungraded_round_exposure_records,
    write_reports,
    write_round,
)


@pytest.mark.skipif(os.name != "nt", reason="Windows-native process check")
def test_process_is_running_uses_no_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        review_suite_local.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("process check started a subprocess"),
    )

    assert _process_is_running(os.getpid()) is True


@pytest.mark.skipif(os.name != "nt", reason="Windows-native process check")
def test_process_is_running_tracks_child_exit() -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
    )
    try:
        assert _process_is_running(child.pid) is True
        child.terminate()
        child.wait(timeout=10)
        assert _process_is_running(child.pid) is False
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


@pytest.mark.parametrize("pid", [None, 0, -1, "not-a-pid"])
def test_process_is_running_rejects_invalid_pid(pid: object) -> None:
    assert _process_is_running(pid) is False  # type: ignore[arg-type]


@pytest.mark.skipif(os.name != "nt", reason="Windows-native process check")
def test_process_is_running_handles_missing_and_native_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _process_is_running(0xFFFFFFFC) is False

    monkeypatch.setattr(review_suite_local, "_open_process", lambda *_args: None)
    monkeypatch.setattr(review_suite_local.ctypes, "get_last_error", lambda: 5)
    assert _process_is_running(123) is True

    closed: list[int] = []
    monkeypatch.setattr(review_suite_local, "_open_process", lambda *_args: 123)
    monkeypatch.setattr(
        review_suite_local, "_wait_for_single_object", lambda *_args: 0xFFFFFFFF
    )
    monkeypatch.setattr(review_suite_local, "_close_handle", closed.append)
    assert _process_is_running(123) is False
    assert closed == [123]


def test_reviewer_wait_line_uses_actual_count() -> None:
    assert _reviewer_wait_line({"runs": [{"slot": "alpha"}]}) == (
        "[review-suite] waiting for 1 reviewer; wrapper is active as long as output streams, do not stop it prematurely"
    )
    assert _reviewer_wait_line({"runs": [{"slot": "alpha"}, {"slot": "bravo"}]}) == (
        "[review-suite] waiting for 2 reviewers; wrapper is active as long as output streams, do not stop it prematurely"
    )


def test_normalize_record_review_cwd_value_matches_wsl_unc_and_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")

    assert normalize_record_review_cwd_value(
        {"review_cwd_normalized": r"\\wsl.localhost\Ubuntu\home\alice\code\repo"}
    ) == (normalize_record_review_cwd_value({"review_cwd": "/home/alice/code/repo"}))


def _variant(
    variant_id: str, *, task_classes: list[str] | None = None
) -> dict[str, object]:
    return {
        "id": variant_id,
        "model": variant_id,
        "reasoning_effort": "medium",
        "state": "active",
        "task_classes": task_classes or ["phase_review"],
    }


def _roster(*variants: dict[str, object]) -> dict[str, object]:
    return {
        "settings": {
            "promotion_min_samples": 20,
            "default_bootstrap_target_samples": 8,
            "bootstrap_weight_boost": 2.0,
            "relative_underuse_ratio": 0.0,
        },
        "variants": list(variants),
    }


def test_service_tier_is_dormant_when_variant_does_not_set_it() -> None:
    assert normalize_service_tier(None) is None

    compacted = compact_benchmark_run(
        {
            "variant_id": "gpt-5.5-medium",
            "elapsed_seconds": 1.0,
            "usage": {},
            "cost_usd": None,
        }
    )

    assert "service_tier" not in compacted


def test_write_round_compacts_finalized_storage(tmp_path: Path) -> None:
    write_round(
        tmp_path,
        {
            "round_id": "round-1",
            "task_class": "pr_review",
            "status": "completed",
            "grading_required": False,
            "requested_prompt": "large prompt",
            "runs": [
                {
                    "variant_id": "gpt-5.5-high",
                    "reviewer_output": "No findings.",
                    "stderr": "large stderr",
                    "command": ["codex", "exec"],
                }
            ],
        },
    )

    payload = json.loads(
        (tmp_path / "rounds" / "round-1.json").read_text(encoding="utf-8")
    )
    assert payload["requested_prompt"] == "large prompt"
    assert payload["runs"][0]["reviewer_output"] == "No findings."
    assert "stderr" not in payload["runs"][0]
    assert "command" not in payload["runs"][0]


def test_write_round_keeps_prompt_for_blocked_reroll(tmp_path: Path) -> None:
    write_round(
        tmp_path,
        {
            "round_id": "round-1",
            "task_class": "pr_review",
            "status": "completed",
            "grading_required": False,
            "requested_prompt": "reroll prompt",
            "runs": [
                {
                    "variant_id": "gpt-5.5-high",
                    "reviewer_output": "",
                    "grade_blocked": True,
                    "stderr": "large stderr",
                }
            ],
        },
    )

    payload = json.loads(
        (tmp_path / "rounds" / "round-1.json").read_text(encoding="utf-8")
    )
    assert payload["requested_prompt"] == "reroll prompt"
    assert "stderr" not in payload["runs"][0]


def test_write_round_preserves_stderr_derived_classification(tmp_path: Path) -> None:
    write_round(
        tmp_path,
        {
            "round_id": "round-1",
            "task_class": "pr_review",
            "status": "completed",
            "requested_prompt": "prompt",
            "runs": [
                {
                    "variant_id": "gpt-5.5-high",
                    "review_status": "completed",
                    "reviewer_output": "",
                    "stderr": "windows sandbox: setup refresh failed",
                }
            ],
        },
    )

    run = json.loads(
        (tmp_path / "rounds" / "round-1.json").read_text(encoding="utf-8")
    )["runs"][0]
    assert run["grade_blocked"] is True
    assert run["grade_block_reason"] == "review_tooling_failure"
    assert "stderr" not in run


def test_compact_round_files_cleans_existing_round_state_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state_dir = tmp_path / "state"
    round_path = state_dir / "rounds" / "old.json"
    orchestrator_path = (
        state_dir / "orchestrator" / "review-rounds" / "rounds" / "old-orc.json"
    )
    live_path = state_dir / "rounds" / "live.json"
    for path in (round_path, orchestrator_path, live_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    round_path.write_text(
        json.dumps(
            {
                "round_id": "old",
                "status": "completed",
                "grading_required": True,
                "requested_prompt": "keep for reroll",
                "runs": [
                    {"variant_id": "a", "reviewer_output": "body", "stderr": "raw"}
                ],
            }
        ),
        encoding="utf-8",
    )
    orchestrator_path.write_text(
        json.dumps(
            {
                "round_id": "old-orc",
                "status": "completed",
                "grading_required": False,
                "requested_prompt": "drop",
                "runs": [
                    {"variant_id": "a", "reviewer_output": "body", "command": ["raw"]}
                ],
            }
        ),
        encoding="utf-8",
    )
    live_path.write_text(
        json.dumps(
            {
                "round_id": "live",
                "status": "running",
                "runs": [{"variant_id": "a", "stderr": "raw"}],
            }
        ),
        encoding="utf-8",
    )

    dry_run = compact_round_files(state_dir)
    assert dry_run["checked"] == 3
    assert dry_run["changed"] == 2
    assert "stderr" in json.loads(round_path.read_text(encoding="utf-8"))["runs"][0]

    locks: list[str] = []

    class Lock:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: object) -> None:
            return None

    def fake_state_lock(_state_dir: Path, name: str, **_kwargs: object) -> Lock:
        locks.append(name)
        return Lock()

    monkeypatch.setattr("review_suite_local.state_lock", fake_state_lock)
    applied = compact_round_files(state_dir, apply=True)

    assert applied["changed"] == 2
    assert "round-old" in locks
    assert "round-old-orc" in locks
    compacted = json.loads(round_path.read_text(encoding="utf-8"))
    assert compacted["requested_prompt"] == "keep for reroll"
    assert "stderr" not in compacted["runs"][0]
    compacted_orchestrator = json.loads(orchestrator_path.read_text(encoding="utf-8"))
    assert compacted_orchestrator["requested_prompt"] == "drop"
    assert "command" not in compacted_orchestrator["runs"][0]
    assert "stderr" in json.loads(live_path.read_text(encoding="utf-8"))["runs"][0]


def test_normalize_service_tier_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="service_tier"):
        normalize_service_tier("priority")


def test_variant_service_tier_rejects_unsupported_config() -> None:
    with pytest.raises(ValueError, match="does not support"):
        variant_service_tier(
            {
                "id": "gpt-5.5-medium",
                "service_tier": "fast",
                "supported_service_tiers": [],
            }
        )


def test_guard_no_stage_step_down_blocks_lower_tier_with_override_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "review_suite_local.inspect_workflow_status",
        lambda **kwargs: {"current_stage_lane": "review_t3"},
    )

    with pytest.raises(
        ValueError, match="Do not invent a lower-tier final-head requirement"
    ) as exc_info:
        guard_no_stage_step_down(
            lane="review_t1",
            review_cwd=tmp_path,
            base="main",
            state_dir=tmp_path / "state",
            review_scope={"base": "main"},
        )
    assert "--allow-stage-step-down" in str(exc_info.value)

    guard_no_stage_step_down(
        lane="review_t3",
        review_cwd=tmp_path,
        base="main",
        state_dir=tmp_path / "state",
        review_scope={"base": "main"},
    )


def test_format_cooldown_until_for_display_uses_local_offset() -> None:
    rendered = format_cooldown_until_for_display("2026-04-13T12:30:00Z")

    assert rendered != "2026-04-13T12:30:00Z"
    assert rendered.endswith("Z") is False
    assert __import__("datetime").datetime.fromisoformat(rendered).tzinfo is not None


def _operational_state(
    *,
    champion_ids: list[str],
    probation_ids: list[str],
    cooling: dict[str, dict[str, object]],
) -> dict[str, object]:
    return {
        "generated_at": "2026-04-12T00:00:00Z",
        "task_classes": {
            "phase_review": {
                "champion_variant_id": champion_ids[0] if champion_ids else None,
                "champion_variant_ids": champion_ids,
                "probation_variant_ids": probation_ids,
                "cooldowns": cooling,
            },
            "pr_review": {
                "champion_variant_id": None,
                "champion_variant_ids": [],
                "probation_variant_ids": [],
                "cooldowns": {},
            },
        },
    }


def _summary_with_leaderboard(*rows: dict[str, object]) -> dict[str, object]:
    return {
        "task_classes": {
            "phase_review": {"leaderboard": list(rows)},
            "pr_review": {"leaderboard": []},
        }
    }


def _record(
    *,
    recorded_at: str,
    round_id: str,
    task_class: str,
    alpha: str,
    beta: str,
    groups: list[list[str]] | None = None,
    basis: str = "valid_findings_vs_none",
    rating_pool_id: str | None = None,
) -> dict[str, object]:
    return {
        "recorded_at": recorded_at,
        "round_id": round_id,
        "task_class": task_class,
        "task_id": f"{task_class}-task",
        "selection_mode": "scramble",
        "rating_pool_id": rating_pool_id or f"{task_class}-v1",
        "placement_v1": {"groups": groups or [[alpha], [beta]], "basis": basis},
        "runs": [
            {
                "variant_id": alpha,
                "elapsed_seconds": 10.0,
                "usage": {},
                "cost_usd": 0.1,
            },
            {
                "variant_id": beta,
                "elapsed_seconds": 12.0,
                "usage": {},
                "cost_usd": 0.2,
            },
        ],
    }


def test_build_record_from_grade_uses_explicit_ordered_groups() -> None:
    round_payload = {
        "round_id": "round-1",
        "task_class": "phase_review",
        "selection_mode": "true_scramble",
        "reporting_pool": True,
        "status": "completed",
        "runs": [
            {
                "slot": slot,
                "variant_id": variant_id,
                "review_status": "completed",
                "grade_blocked": False,
                "usage": {},
            }
            for slot, variant_id in (
                ("alpha", "model-a"),
                ("bravo", "model-b"),
                ("charlie", "model-c"),
            )
        ],
    }
    roster = {
        "variants": [
            {**_variant(variant_id), "pricing": {}}
            for variant_id in ("model-a", "model-b", "model-c")
        ]
    }
    record = build_record_from_grade(
        round_payload=round_payload,
        roster=roster,
        rubric={"basis": list(review_suite_local.GRADE_BASIS_VALUES)},
        task_id="task-1",
        rating_pool_id="arena-phase-v1",
        rank_groups=["alpha", "bravo,model-c"],
        basis="better_bug_coverage",
        shared_note="checked",
    )

    assert record["rating_pool_id"] == "arena-phase-v1"
    assert record["reporting_pool"] is True
    assert record["placement_v1"] == {
        "groups": [["model-a"], ["model-b", "model-c"]],
        "basis": "better_bug_coverage",
    }
    assert "pairwise_outcome" not in record
    assert "grade_schema" not in record
    assert all(run["grader_notes"] == "checked" for run in record["runs"])

    configured = json.loads(json.dumps(round_payload))
    configured["rating_pool_id"] = "discovery-phase-gpt-5.6-v1"
    with pytest.raises(ValueError, match="must match the configured pool"):
        build_record_from_grade(
            round_payload=configured,
            roster=roster,
            rubric={"basis": list(review_suite_local.GRADE_BASIS_VALUES)},
            task_id="task-1",
            rating_pool_id="wrong-pool",
            rank_groups=["alpha", "bravo,model-c"],
            basis="better_bug_coverage",
            shared_note=None,
        )

    for ranks, message in (
        (["alpha", "bravo"], "missing: model-c"),
        (["alpha", "bravo", "model-a"], "duplicate placement variant"),
        (["alpha", "bravo", "unknown"], "unknown rank entry"),
    ):
        with pytest.raises(ValueError, match=message):
            build_record_from_grade(
                round_payload=round_payload,
                roster=roster,
                rubric={"basis": list(review_suite_local.GRADE_BASIS_VALUES)},
                task_id="task-1",
                rating_pool_id="arena-phase-v1",
                rank_groups=ranks,
                basis="better_bug_coverage",
                shared_note=None,
            )

    blocked = json.loads(json.dumps(round_payload))
    blocked["runs"][2].update(
        review_status="timeout",
        grade_blocked=True,
        grade_block_reason="review_timed_out",
    )
    with pytest.raises(ValueError, match="should not be graded"):
        build_record_from_grade(
            round_payload=blocked,
            roster=roster,
            rubric={"basis": list(review_suite_local.GRADE_BASIS_VALUES)},
            task_id="task-1",
            rating_pool_id="arena-phase-v1",
            rank_groups=["alpha", "bravo", "charlie"],
            basis="better_bug_coverage",
            shared_note=None,
        )


def test_update_elo_normalizes_simultaneous_multiplayer_results() -> None:
    ratings = {"strong": 1600.0, "middle": 1500.0, "weak": 1400.0}
    upset = update_elo(ratings, [["weak"], ["middle"], ["strong"]], 24.0)
    expected = update_elo(ratings, [["strong"], ["middle"], ["weak"]], 24.0)

    assert sum(upset.values()) == pytest.approx(sum(ratings.values()))
    assert abs(upset["strong"] - ratings["strong"]) > abs(
        expected["weak"] - ratings["weak"]
    )
    pair = update_elo({"a": 1600.0, "b": 1400.0}, [["a"], ["b"]], 24.0)
    pair_expected = review_suite_local.expected_score(1600.0, 1400.0)
    assert pair == pytest.approx(
        {
            "a": 1600.0 + 24.0 * (1.0 - pair_expected),
            "b": 1400.0 - 24.0 * (1.0 - pair_expected),
        }
    )
    tied = update_elo(
        {"a": 1500.0, "b": 1500.0, "c": 1500.0}, [["a", "b"], ["c"]], 24.0
    )
    assert tied == pytest.approx({"a": 1506.0, "b": 1506.0, "c": 1488.0})


@pytest.mark.parametrize(
    "record,message",
    [
        ({"placement_v1": []}, "placement_v1 must be an object"),
        (
            {"placement_v1": {"groups": [["a"], ["b"]], "basis": "tie_clean"}},
            "rating_pool_id is required",
        ),
        (
            {
                "rating_pool_id": "pool",
                "placement_v1": {"groups": [["a"], ["a"]], "basis": "tie_clean"},
            },
            "duplicate placement variant",
        ),
    ],
)
def test_placement_record_rejects_invalid_schema(
    record: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        placement_record(record)


def _git_repo_with_origin(path: Path, origin: str) -> None:
    path.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", origin],
        cwd=path,
        check=True,
        capture_output=True,
    )


def test_repo_name_from_round_payload_prefers_git_remote_for_opaque_worktree(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "esbzptu4"
    _git_repo_with_origin(repo, "https://github.com/JJLiebig/review-suite.git")

    assert repo_name_from_round_payload({"review_cwd": str(repo)}) == "review-suite"


def test_compact_benchmark_record_preserves_repo_name() -> None:
    record = _record(
        recorded_at="2026-04-12T13:00:00Z",
        round_id="phase_review-esbzptu4-123456-20260412T130000Z-abcdef01",
        task_class="phase_review",
        alpha="gpt-5.4-medium",
        beta="gpt-5.5-medium",
        groups=[["gpt-5.5-medium"], ["gpt-5.4-medium"]],
    )
    record["repo_name"] = "codex-account-switcher"

    assert compact_benchmark_record(record)["repo_name"] == "codex-account-switcher"


def test_enrich_record_repo_names_uses_round_payload_remote_for_old_rows(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    rounds_dir = state_dir / "rounds"
    rounds_dir.mkdir(parents=True)
    repo = tmp_path / "spp_worktrees" / "esbzptu4"
    _git_repo_with_origin(repo, "git@github.com:JJLiebig/codex-account-switcher.git")
    round_id = "phase_review-esbzptu4-123456-20260412T130000Z-abcdef01"
    (rounds_dir / f"{round_id}.json").write_text(
        json.dumps({"review_cwd": str(repo)}),
        encoding="utf-8",
    )
    records = [
        _record(
            recorded_at="2026-04-12T13:00:00Z",
            round_id=round_id,
            task_class="phase_review",
            alpha="gpt-5.4-medium",
            beta="gpt-5.5-medium",
            groups=[["gpt-5.5-medium"], ["gpt-5.4-medium"]],
        )
    ]

    enriched, changed = enrich_record_repo_names(state_dir, records)

    assert changed is True
    assert "repo_name" not in records[0]
    assert enriched[0]["repo_name"] == "codex-account-switcher"


def test_enrich_record_repo_names_caches_review_cwd_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    rounds_dir = state_dir / "rounds"
    rounds_dir.mkdir(parents=True)
    review_cwd = str(tmp_path / "spp_worktrees" / "esbzptu4")
    calls = 0

    def fake_repo_name(review_cwd_value: str) -> str:
        nonlocal calls
        calls += 1
        assert review_cwd_value == review_cwd
        return "review-suite"

    monkeypatch.setattr(
        review_suite_local, "repo_name_from_review_cwd_value", fake_repo_name
    )
    records = []
    for idx in range(2):
        round_id = f"phase_review-esbzptu4-12345{idx}-20260412T130000Z-abcdef0{idx}"
        (rounds_dir / f"{round_id}.json").write_text(
            json.dumps({"review_cwd": review_cwd}),
            encoding="utf-8",
        )
        records.append(
            _record(
                recorded_at=f"2026-04-12T13:0{idx}:00Z",
                round_id=round_id,
                task_class="phase_review",
                alpha="gpt-5.4-medium",
                beta="gpt-5.5-medium",
                groups=[["gpt-5.5-medium"], ["gpt-5.4-medium"]],
            )
        )

    enriched, changed = enrich_record_repo_names(state_dir, records)

    assert changed is True
    assert calls == 1
    assert [record["repo_name"] for record in enriched] == [
        "review-suite",
        "review-suite",
    ]


def test_classify_review_result_prefers_valid_output_over_stale_interruption_marker() -> (
    None
):
    classification = _classify_review_result(
        reviewer_output="No findings.",
        stderr_text="WARN review was interrupted before a usable result was captured.",
        session_id="session-123",
        thread_id="thread-123",
    )

    assert classification["review_status"] == "completed"
    assert classification["grade_blocked"] is False
    assert classification["grade_block_reason"] is None


def test_classify_review_result_prefers_valid_output_over_stale_capacity_marker() -> (
    None
):
    classification = _classify_review_result(
        reviewer_output="No findings.",
        stderr_text="selected model is at capacity",
        session_id="session-123",
        thread_id="thread-123",
    )

    assert classification["review_status"] == "completed"
    assert classification["grade_blocked"] is False
    assert classification["grade_block_reason"] is None


def test_classify_review_result_keeps_capacity_for_interruption_boilerplate() -> None:
    classification = _classify_review_result(
        reviewer_output="Review was interrupted before a usable result was captured.",
        stderr_text="selected model is at capacity",
        session_id="session-123",
        thread_id="thread-123",
    )

    assert classification["review_status"] == "interrupted_capacity"
    assert classification["grade_blocked"] is True
    assert classification["grade_block_reason"] == "selected_model_at_capacity"


def test_classify_review_result_surfaces_authentication_failure() -> None:
    classification = _classify_review_result(
        reviewer_output="",
        stderr_text="",
        session_id="session-123",
        thread_id="thread-123",
        rollout_error="Your access token could not be refreshed. Please log out and sign in again.",
    )

    assert classification["review_status"] == "authentication_failed"
    assert classification["grade_block_reason"] == "codex_authentication_failed"
    assert "Log out and sign in again" in classification["status_summary"]


def _opencode_error_stderr(error_class: str, **extra: object) -> str:
    payload: dict[str, object] = {
        "session_id": "ses_opencode",
        "usage": {},
        "error_class": error_class,
    }
    payload.update(extra)
    return REVIEW_METADATA_PREFIX + json.dumps(payload) + "\n"


def test_classify_review_result_maps_opencode_capacity_error() -> None:
    classification = _classify_review_result(
        reviewer_output="",
        stderr_text=_opencode_error_stderr(
            "capacity",
            error_name="ProviderError",
            error_message="rate limit exceeded",
        ),
        session_id="ses_opencode",
        thread_id=None,
    )

    assert classification["review_status"] == "interrupted_capacity"
    assert classification["grade_blocked"] is True
    assert classification["grade_block_reason"] == "selected_model_at_capacity"


def test_classify_review_result_maps_opencode_unavailable_error() -> None:
    classification = _classify_review_result(
        reviewer_output="",
        stderr_text=_opencode_error_stderr(
            "unavailable", error_message="model not found"
        ),
        session_id="ses_opencode",
        thread_id=None,
    )

    assert classification["review_status"] == "opencode_unavailable"
    assert classification["grade_blocked"] is True
    assert classification["grade_block_reason"] == "selected_model_unavailable"
    assert "model not found" in classification["status_summary"]


def test_classify_review_result_maps_opencode_generic_failure() -> None:
    classification = _classify_review_result(
        reviewer_output="",
        stderr_text=_opencode_error_stderr("failed", error_name="UnknownError"),
        session_id="ses_opencode",
        thread_id=None,
    )

    assert classification["review_status"] == "opencode_failed"
    assert classification["grade_blocked"] is True
    assert classification["grade_block_reason"] == "opencode_review_failed"


def test_classify_review_result_maps_opencode_adapter_error_to_tooling_failure() -> (
    None
):
    classification = _classify_review_result(
        reviewer_output="",
        stderr_text=_opencode_error_stderr(
            "adapter", error_message="OpenCode CLI was not found on PATH"
        ),
        session_id=None,
        thread_id=None,
    )

    assert classification["review_status"] == "tooling_failure"
    assert classification["grade_blocked"] is True
    assert classification["grade_block_reason"] == "review_tooling_failure"


def test_classify_review_result_prefers_output_over_opencode_error() -> None:
    classification = _classify_review_result(
        reviewer_output="No findings.",
        stderr_text=_opencode_error_stderr("failed", error_name="UnknownError"),
        session_id="ses_opencode",
        thread_id=None,
    )

    assert classification["review_status"] == "completed"
    assert classification["grade_blocked"] is False
    assert classification["grade_block_reason"] is None


def test_terminal_review_command_requires_final_machine_line() -> None:
    assert terminal_review_command("No findings.\n\nReview result: clean") == "clean"
    assert terminal_review_command("P1 bug\n\nReview result: findings") == "findings"
    assert terminal_review_command("Review result: clean\n\nAdditional prose") is None
    assert (
        terminal_review_command("Review result: clean\n\nReview result: findings")
        is None
    )
    assert terminal_review_command("No findings.") is None


def test_classify_review_result_preserves_terminal_command() -> None:
    classification = _classify_review_result(
        reviewer_output="No findings.\n\nReview result: clean",
        stderr_text="",
        session_id="session-1",
        thread_id=None,
    )

    assert classification["review_status"] == "completed"
    assert classification["terminal_command"] == "clean"


def test_classify_review_result_blocks_direct_interruption_output() -> None:
    classification = _classify_review_result(
        reviewer_output="Review was interrupted before a usable result was captured.",
        stderr_text="",
        session_id="session-123",
        thread_id="thread-123",
    )

    assert classification["review_status"] == "interrupted"
    assert classification["grade_blocked"] is True
    assert classification["grade_block_reason"] == "review_interrupted"


def test_classify_review_capture_marks_timeout_without_output() -> None:
    classification = classify_review_capture(
        reviewer_output="",
        stderr_text="",
        session_id=None,
        thread_id=None,
        timed_out=True,
    )

    assert classification["review_status"] == "timeout"
    assert classification["grade_blocked"] is True
    assert classification["grade_block_reason"] == "review_timed_out"


def test_classify_review_capture_marks_timeout_even_with_partial_output() -> None:
    classification = classify_review_capture(
        reviewer_output="Partial findings before kill",
        stderr_text="",
        session_id="session-123",
        thread_id="thread-123",
        timed_out=True,
    )

    assert classification["review_status"] == "timeout"
    assert classification["grade_blocked"] is True
    assert classification["grade_block_reason"] == "review_timed_out"


def test_classify_review_capture_marks_transport_stall() -> None:
    classification = classify_review_capture(
        reviewer_output="",
        stderr_text="ERROR: Reconnecting... 5/5\nfalling back to HTTP\n",
        session_id=None,
        thread_id=None,
        transport_stalled=True,
    )

    assert classification["review_status"] == "transport_stalled"
    assert classification["grade_blocked"] is True
    assert classification["grade_block_reason"] == "review_transport_stalled"


def test_apply_capacity_cooldowns_handles_timeout_and_transport_stall(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    round_payload = {
        "task_class": "pr_review",
        "runs": [
            {
                "variant_id": "model-timeout",
                "review_status": "timeout",
                "grade_blocked": True,
                "grade_block_reason": "review_timed_out",
            },
            {
                "variant_id": "model-stall",
                "review_status": "transport_stalled",
                "grade_blocked": True,
                "grade_block_reason": "review_transport_stalled",
            },
            {
                "variant_id": "model-other",
                "review_status": "interrupted",
                "grade_blocked": True,
                "grade_block_reason": "review_interrupted",
            },
        ],
    }

    updates = _apply_capacity_cooldowns(
        state_dir=state_dir, round_payload=round_payload
    )
    state = json.loads(
        (state_dir / "operational_state.json").read_text(encoding="utf-8")
    )
    cooldowns = state["task_classes"]["pr_review"]["cooldowns"]

    assert [update["variant_id"] for update in updates] == [
        "model-timeout",
        "model-stall",
    ]
    assert cooldowns["model-timeout"]["last_reason"] == "review_timed_out"
    assert cooldowns["model-stall"]["last_reason"] == "review_transport_stalled"
    assert "model-other" not in cooldowns


def test_apply_capacity_cooldowns_keeps_cooldown_when_same_variant_also_completed(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    round_payload = {
        "task_class": "pr_review",
        "runs": [
            {
                "variant_id": "model-a",
                "review_status": "transport_stalled",
                "grade_blocked": True,
                "grade_block_reason": "review_transport_stalled",
            },
            {
                "variant_id": "model-a",
                "review_status": "completed",
                "grade_blocked": False,
                "grade_block_reason": None,
            },
        ],
    }

    updates = _apply_capacity_cooldowns(
        state_dir=state_dir, round_payload=round_payload
    )
    state = json.loads(
        (state_dir / "operational_state.json").read_text(encoding="utf-8")
    )
    cooldowns = state["task_classes"]["pr_review"]["cooldowns"]

    assert [update["variant_id"] for update in updates] == ["model-a"]
    assert cooldowns["model-a"]["last_reason"] == "review_transport_stalled"


def test_apply_capacity_cooldowns_cools_opencode_failures(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    round_payload = {
        "task_class": "pr_review",
        "runs": [
            {
                "variant_id": "opencode-unavailable",
                "review_status": "opencode_unavailable",
                "grade_blocked": True,
                "grade_block_reason": "selected_model_unavailable",
            },
            {
                "variant_id": "opencode-failed",
                "review_status": "opencode_failed",
                "grade_blocked": True,
                "grade_block_reason": "opencode_review_failed",
            },
            {
                "variant_id": "opencode-eligible-exit",
                "review_status": "reviewer_process_exited",
                "grade_blocked": True,
                "grade_block_reason": "reviewer_process_exited",
                "cooldown_eligible": True,
            },
            {
                "variant_id": "opencode-ineligible-exit",
                "review_status": "reviewer_process_exited",
                "grade_blocked": True,
                "grade_block_reason": "reviewer_process_exited",
            },
            {
                "variant_id": "opencode-adapter",
                "review_status": "tooling_failure",
                "grade_blocked": True,
                "grade_block_reason": "review_tooling_failure",
                "cooldown_eligible": True,
            },
        ],
    }

    updates = _apply_capacity_cooldowns(
        state_dir=state_dir, round_payload=round_payload
    )
    state = json.loads(
        (state_dir / "operational_state.json").read_text(encoding="utf-8")
    )
    cooldowns = state["task_classes"]["pr_review"]["cooldowns"]

    assert [update["variant_id"] for update in updates] == [
        "opencode-unavailable",
        "opencode-failed",
        "opencode-eligible-exit",
    ]
    assert (
        cooldowns["opencode-unavailable"]["last_reason"] == "selected_model_unavailable"
    )
    assert cooldowns["opencode-failed"]["last_reason"] == "opencode_review_failed"
    assert (
        cooldowns["opencode-eligible-exit"]["last_reason"] == "reviewer_process_exited"
    )
    assert "opencode-ineligible-exit" not in cooldowns
    assert "opencode-adapter" not in cooldowns


def test_apply_capacity_cooldowns_escalates_after_history_expires(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    expired = "2000-01-01T00:00:00Z"
    seed = {
        "generated_at": expired,
        "task_classes": {
            "phase_review": {
                "cooldowns": {},
                "cooldown_failures": {
                    "model-x": {
                        "failure_count": 2,
                        "last_reason": "opencode_review_failed",
                        "last_triggered_at": expired,
                    }
                },
            },
            "pr_review": {"cooldowns": {}, "cooldown_failures": {}},
        },
    }
    (state_dir / "operational_state.json").write_text(
        json.dumps(seed), encoding="utf-8"
    )

    updates = _apply_capacity_cooldowns(
        state_dir=state_dir,
        round_payload={
            "task_class": "phase_review",
            "runs": [
                {
                    "variant_id": "model-x",
                    "review_status": "opencode_failed",
                    "grade_blocked": True,
                    "grade_block_reason": "opencode_review_failed",
                }
            ],
        },
    )

    state = json.loads(
        (state_dir / "operational_state.json").read_text(encoding="utf-8")
    )
    task_state = state["task_classes"]["phase_review"]
    assert task_state["cooldown_failures"]["model-x"]["failure_count"] == 3
    assert updates[0]["failure_count"] == 3
    until = datetime.fromisoformat(
        task_state["cooldowns"]["model-x"]["until"].replace("Z", "+00:00")
    )
    remaining = (until - datetime.now(timezone.utc)).total_seconds()
    assert 5.9 * 3600 < remaining <= 6 * 3600 + 5


def test_apply_capacity_cooldowns_clears_failure_history_on_success(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    expired = "2000-01-01T00:00:00Z"
    seed = {
        "generated_at": expired,
        "task_classes": {
            "phase_review": {
                "cooldowns": {},
                "cooldown_failures": {
                    "model-y": {
                        "failure_count": 2,
                        "last_reason": "opencode_review_failed",
                        "last_triggered_at": expired,
                    }
                },
            },
            "pr_review": {"cooldowns": {}, "cooldown_failures": {}},
        },
    }
    (state_dir / "operational_state.json").write_text(
        json.dumps(seed), encoding="utf-8"
    )

    _apply_capacity_cooldowns(
        state_dir=state_dir,
        round_payload={
            "task_class": "phase_review",
            "runs": [
                {
                    "variant_id": "model-y",
                    "review_status": "completed",
                    "grade_blocked": False,
                    "grade_block_reason": None,
                }
            ],
        },
    )

    state = json.loads(
        (state_dir / "operational_state.json").read_text(encoding="utf-8")
    )
    task_state = state["task_classes"]["phase_review"]
    assert "model-y" not in task_state["cooldown_failures"]


def test_load_operational_state_seeds_cooldown_failure_history(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "operational_state.json"
    state_path.write_text(
        json.dumps(
            {
                "generated_at": "2000-01-01T00:00:00Z",
                "task_classes": {
                    "phase_review": {
                        "cooldowns": {
                            "model-z": {
                                "until": "2999-01-01T00:00:00Z",
                                "failure_count": 4,
                                "last_reason": "opencode_review_failed",
                                "last_triggered_at": "2000-01-01T00:00:00Z",
                            }
                        },
                    },
                    "pr_review": {"cooldowns": {}},
                },
            }
        ),
        encoding="utf-8",
    )

    state = review_suite_local.load_operational_state(state_path)

    failures = state["task_classes"]["phase_review"]["cooldown_failures"]
    assert failures["model-z"]["failure_count"] == 4
    assert failures["model-z"]["last_reason"] == "opencode_review_failed"


def test_ensure_clean_git_worktree_ignores_untracked_review_suite_scratch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "review_suite_local.meaningful_worktree_status_entries", lambda review_cwd: []
    )

    ensure_clean_git_worktree(tmp_path)


def test_ensure_clean_git_worktree_still_blocks_other_untracked_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "review_suite_local.meaningful_worktree_status_entries",
        lambda review_cwd: [{"code": "??", "path": "todo.txt"}],
    )

    with pytest.raises(ValueError, match="clean worktree"):
        ensure_clean_git_worktree(tmp_path)


def test_ensure_clean_git_worktree_blocks_unrelated_dirty_files_for_base_review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "review_suite_local.meaningful_worktree_status_entries",
        lambda review_cwd: [{"code": " M", "path": "docs/notes.md"}],
    )
    with pytest.raises(ValueError, match="clean worktree"):
        ensure_clean_git_worktree(
            tmp_path, review_scope={"base": "main", "merge_base": "base123"}
        )


def test_ensure_clean_git_worktree_blocks_unrelated_dirty_files_for_flagged_commit_review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "review_suite_local.meaningful_worktree_status_entries",
        lambda review_cwd: [{"code": " M", "path": "docs/notes.md"}],
    )
    with pytest.raises(ValueError, match="clean worktree"):
        ensure_clean_git_worktree(
            tmp_path,
            review_scope={
                "base": "main",
                "commit": "head-1",
                "commit_end": "head-2",
                "merge_base": "base123",
            },
        )


def test_running_status_line_compacts_alive_reviewers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "review_suite_local.utc_now",
        lambda: __import__("datetime").datetime.fromisoformat(
            "2026-04-13T12:00:42+00:00"
        ),
    )

    line = _running_status_line(
        [
            {"slot": "Alpha", "started_at": "2026-04-13T12:00:00Z"},
            {"slot": "Bravo", "started_at": "2026-04-13T12:00:20Z"},
        ]
    )

    assert line == "Running: 42s Alpha, Bravo"


def test_heartbeat_status_line_is_minimal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "review_suite_local.utc_now",
        lambda: __import__("datetime").datetime.fromisoformat(
            "2026-04-13T12:02:05+00:00"
        ),
    )

    line = _heartbeat_status_line(
        [
            {"slot": "Alpha", "started_at": "2026-04-13T12:00:00Z"},
            {"slot": "Bravo", "started_at": "2026-04-13T12:00:20Z"},
        ]
    )

    assert line == "OK 2m: Alpha,Bravo"


def test_transport_stalled_requires_reconnect_exhaustion_and_quiet_artifacts(
    tmp_path: Path,
) -> None:
    stdout = tmp_path / "review.stdout"
    stderr = tmp_path / "review.stderr"
    stdout.write_text("", encoding="utf-8")
    stderr.write_text(
        "ERROR: Reconnecting... 5/5\n"
        "2026-04-23T22:59:35Z WARN codex_core::client: falling back to HTTP\n",
        encoding="utf-8",
    )
    old_time = 1000.0
    recent_time = old_time + 10.0
    os_utime = __import__("os").utime
    os_utime(stdout, (old_time, old_time))
    os_utime(stderr, (old_time, old_time))

    assert (
        _transport_stalled(
            {"stdout_path": str(stdout), "stderr_path": str(stderr)},
            now_epoch=old_time + 181.0,
        )
        == "http_fallback_no_output"
    )

    os_utime(stderr, (recent_time, recent_time))
    assert (
        _transport_stalled(
            {"stdout_path": str(stdout), "stderr_path": str(stderr)},
            now_epoch=recent_time + 10.0,
        )
        is None
    )

    stdout.write_text("No findings.\n", encoding="utf-8")
    assert (
        _transport_stalled(
            {"stdout_path": str(stdout), "stderr_path": str(stderr)},
            now_epoch=old_time + 1000.0,
        )
        is None
    )


def test_transport_stall_respects_active_reviewer_rollout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stdout = tmp_path / "review.stdout"
    stderr = tmp_path / "review.stderr"
    rollout = tmp_path / "rollout.jsonl"
    stdout.write_text("", encoding="utf-8")
    stderr.write_text("ERROR: Reconnecting... 5/5\n", encoding="utf-8")
    rollout.write_text("", encoding="utf-8")
    old_time = 1000.0
    __import__("os").utime(stdout, (old_time, old_time))
    __import__("os").utime(stderr, (old_time, old_time))
    run = {
        "stdout_path": str(stdout),
        "stderr_path": str(stderr),
        "started_at": "2026-04-13T12:00:00Z",
    }
    active_at = datetime.fromisoformat("2026-04-13T12:20:00+00:00")
    monkeypatch.setattr(
        "review_suite_local._live_review_thread",
        lambda **_: {"rollout_path": str(rollout)},
    )
    monkeypatch.setattr(
        "review_suite_local.rollout_activity_summary",
        lambda path: {"last_meaningful_at": active_at},
    )

    assert (
        review_suite_local._reviewer_deadline_reason(
            run=run,
            variant={"model": "gpt-test", "reasoning_effort": "xhigh"},
            sqlite_path=tmp_path / "state.sqlite",
            review_cwd=tmp_path,
            now=active_at,
        )
        is None
    )
    assert _transport_stalled(run, now_epoch=active_at.timestamp() + 60.0) is None


def test_transport_hung_after_output_requires_quiet_captured_stdout(
    tmp_path: Path,
) -> None:
    stdout = tmp_path / "review.stdout"
    stderr = tmp_path / "review.stderr"
    stdout.write_text("No findings.\n", encoding="utf-8")
    stderr.write_text("codex\nNo findings.\n", encoding="utf-8")
    old_time = 1000.0
    recent_time = old_time + 10.0
    os_utime = __import__("os").utime
    os_utime(stdout, (old_time, old_time))
    os_utime(stderr, (old_time, old_time))

    assert (
        _transport_hung_after_output(
            {"stdout_path": str(stdout), "stderr_path": str(stderr)},
            now_epoch=old_time + 181.0,
        )
        == "output_captured_process_still_running"
    )

    os_utime(stdout, (recent_time, recent_time))
    assert (
        _transport_hung_after_output(
            {"stdout_path": str(stdout), "stderr_path": str(stderr)},
            now_epoch=recent_time + 10.0,
        )
        is None
    )

    stdout.write_text("", encoding="utf-8")
    assert (
        _transport_hung_after_output(
            {"stdout_path": str(stdout), "stderr_path": str(stderr)},
            now_epoch=old_time + 1000.0,
        )
        is None
    )


def test_print_transport_events_streams_reconnect_lines_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stderr = tmp_path / "review.stderr"
    stderr.write_text(
        "noise\n"
        "2026-04-23T22:34:18Z WARN codex_core::session::turn: stream disconnected - retrying sampling request (1/5 in 201ms)...\n"
        "ERROR: Reconnecting... 2/5\n"
        "2026-04-23T22:59:35Z WARN codex_core::client: falling back to HTTP\n",
        encoding="utf-8",
    )
    run = {"slot": "bravo", "stderr_path": str(stderr)}

    assert _print_transport_events([run]) is True
    assert _print_transport_events([run]) is False

    captured = capsys.readouterr()
    assert "bravo transport: 2026-04-23T22:34:18Z WARN" in captured.err
    assert "bravo transport: ERROR: Reconnecting... 2/5" in captured.err
    assert "bravo transport: 2026-04-23T22:59:35Z WARN" in captured.err
    assert captured.err.count("bravo transport:") == 3


def test_collect_round_results_stops_transport_stalled_live_reviewer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stdout = tmp_path / "review.stdout"
    stderr = tmp_path / "review.stderr"
    stdout.write_text("", encoding="utf-8")
    stderr.write_text(
        "ERROR: Reconnecting... 5/5\nfalling back to HTTP\n", encoding="utf-8"
    )
    old_time = 1000.0
    __import__("os").utime(stdout, (old_time, old_time))
    __import__("os").utime(stderr, (old_time, old_time))
    killed: set[int] = set()

    monkeypatch.setattr("review_suite_local.time.monotonic", lambda: 2000.0)
    monkeypatch.setattr("review_suite_local.time.time", lambda: old_time + 181.0)
    monkeypatch.setattr("review_suite_local.time.sleep", lambda seconds: None)
    monkeypatch.setattr(
        "review_suite_local._process_is_running", lambda pid: int(pid) not in killed
    )
    monkeypatch.setattr(
        "review_suite_local._terminate_process_tree", lambda pid: killed.add(int(pid))
    )
    monkeypatch.setattr(
        "review_suite_local._reviewer_deadline_reason", lambda **_: None
    )
    monkeypatch.setattr(
        "review_suite_local.find_review_child_thread", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "review_suite_local._apply_capacity_cooldowns", lambda **kwargs: []
    )
    monkeypatch.setattr("review_suite_local._cleanup_run_artifacts", lambda run: True)

    result = collect_round_results(
        round_payload={
            "round_id": "round-1",
            "task_class": "phase_review",
            "status": "running",
            "runs": [
                {
                    "slot": "bravo",
                    "variant_id": "model-a",
                    "pid": 123,
                    "title": "review-title",
                    "command": [],
                    "stdout_path": str(stdout),
                    "stderr_path": str(stderr),
                    "started_at": "2026-04-13T12:00:00Z",
                }
            ],
        },
        roster=_roster(_variant("model-a")),
        state_dir=tmp_path,
        review_cwd=tmp_path,
        sqlite_path=tmp_path / "state.sqlite",
        progress_interval_seconds=0,
        wait=True,
    )

    captured = capsys.readouterr()
    assert "bravo transport: ERROR: Reconnecting... 5/5" in captured.err
    assert "bravo transport stalled (http_fallback_no_output)" in captured.err
    assert "Running:" not in captured.err
    assert "OK " in captured.err
    assert ": bravo" in captured.err
    assert result["runs"][0]["review_status"] == "transport_stalled"
    assert result["runs"][0]["grade_block_reason"] == "review_transport_stalled"
    assert killed == {123}


def test_collect_round_results_stops_transport_hung_after_output_as_completed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stdout = tmp_path / "review.stdout"
    stderr = tmp_path / "review.stderr"
    stdout.write_text("No findings.\n", encoding="utf-8")
    stderr.write_text("codex\nNo findings.\n", encoding="utf-8")
    old_time = 1000.0
    __import__("os").utime(stdout, (old_time, old_time))
    __import__("os").utime(stderr, (old_time, old_time))
    killed: set[int] = set()

    monkeypatch.setattr("review_suite_local.time.monotonic", lambda: 2000.0)
    monkeypatch.setattr("review_suite_local.time.time", lambda: old_time + 181.0)
    monkeypatch.setattr("review_suite_local.time.sleep", lambda seconds: None)
    monkeypatch.setattr(
        "review_suite_local._process_is_running", lambda pid: int(pid) not in killed
    )
    monkeypatch.setattr(
        "review_suite_local._terminate_process_tree", lambda pid: killed.add(int(pid))
    )
    monkeypatch.setattr(
        "review_suite_local._reviewer_deadline_reason", lambda **_: None
    )
    monkeypatch.setattr(
        "review_suite_local.find_review_child_thread", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "review_suite_local._apply_capacity_cooldowns", lambda **kwargs: []
    )
    monkeypatch.setattr("review_suite_local._cleanup_run_artifacts", lambda run: True)

    result = collect_round_results(
        round_payload={
            "round_id": "round-1",
            "task_class": "phase_review",
            "status": "running",
            "runs": [
                {
                    "slot": "bravo",
                    "variant_id": "model-a",
                    "pid": 123,
                    "title": "review-title",
                    "command": [],
                    "stdout_path": str(stdout),
                    "stderr_path": str(stderr),
                    "started_at": "2026-04-13T12:00:00Z",
                }
            ],
        },
        roster=_roster(_variant("model-a")),
        state_dir=tmp_path,
        review_cwd=tmp_path,
        sqlite_path=tmp_path / "state.sqlite",
        progress_interval_seconds=0,
        wait=True,
    )

    captured = capsys.readouterr()
    assert (
        "bravo transport hung after output (output_captured_process_still_running)"
        in captured.err
    )
    assert "Running:" not in captured.err
    assert "OK " in captured.err
    assert ": bravo" in captured.err
    assert result["runs"][0]["review_status"] == "completed"
    assert result["runs"][0]["grade_blocked"] is False
    assert killed == {123}


def test_reviewer_deadlines_bound_inactivity_and_total_runtime(tmp_path: Path) -> None:
    variant = {"model": "gpt-test", "reasoning_effort": "xhigh"}
    run = {"started_at": "2026-04-13T12:00:00Z"}

    assert (
        review_suite_local._reviewer_deadline_reason(
            run=run,
            variant=variant,
            sqlite_path=tmp_path / "state.sqlite",
            review_cwd=tmp_path,
            now=datetime.fromisoformat("2026-04-13T12:31:00+00:00"),
        )
        == "inactivity_deadline"
    )
    assert (
        review_suite_local._reviewer_deadline_reason(
            run=run,
            variant=variant,
            sqlite_path=tmp_path / "state.sqlite",
            review_cwd=tmp_path,
            now=datetime.fromisoformat("2026-04-13T14:01:00+00:00"),
        )
        == "absolute_deadline"
    )


def test_duplicate_reviewer_references_block_every_colliding_run() -> None:
    runs = [
        {
            "slot": "alpha",
            "thread_id": "child-alpha",
            "rollout_path": "same.jsonl",
            "reviewer_output_ref": "rollout://child-alpha/model",
            "review_status": "completed",
            "grade_blocked": False,
        },
        {
            "slot": "bravo",
            "thread_id": "child-alpha",
            "rollout_path": "same.jsonl",
            "reviewer_output_ref": "rollout://child-alpha/model",
            "review_status": "completed",
            "grade_blocked": False,
        },
    ]

    review_suite_local.reject_duplicate_review_references(runs)

    assert {run["review_status"] for run in runs} == {"duplicate_output"}
    assert {run["grade_block_reason"] for run in runs} == {"duplicate_reviewer_output"}


def test_print_stall_warnings_flags_heartbeat_only_review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rollout_path = tmp_path / "review.jsonl"
    rollout_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "review_suite_local.utc_now",
        lambda: __import__("datetime").datetime.fromisoformat(
            "2026-04-13T12:20:00+00:00"
        ),
    )
    monkeypatch.setattr(
        "review_suite_local._live_review_thread",
        lambda **_: {"rollout_path": str(rollout_path)},
    )
    monkeypatch.setattr(
        "review_suite_local.rollout_activity_summary",
        lambda path: {
            "last_event_at": __import__("datetime").datetime.fromisoformat(
                "2026-04-13T12:19:00+00:00"
            ),
            "last_meaningful_at": None,
            "last_meaningful_type": None,
        },
    )
    warned: set[str] = set()

    _print_stall_warnings(
        active_runs=[
            {"slot": "bravo", "variant_id": "v1", "started_at": "2026-04-13T12:00:00Z"}
        ],
        indexed={"v1": {"model": "gpt-test", "reasoning_effort": "xhigh"}},
        sqlite_path=tmp_path / "state.sqlite",
        review_cwd=tmp_path,
        warned_slots=warned,
    )

    captured = capsys.readouterr()
    assert "possible stall: bravo idle 20m; wrapper will keep waiting." in captured.err
    assert warned == {"bravo"}


def test_print_stall_warnings_flags_empty_rollout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rollout_path = tmp_path / "empty.jsonl"
    rollout_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "review_suite_local.utc_now",
        lambda: __import__("datetime").datetime.fromisoformat(
            "2026-04-13T12:20:00+00:00"
        ),
    )
    monkeypatch.setattr(
        "review_suite_local._live_review_thread",
        lambda **_: {"rollout_path": str(rollout_path)},
    )
    warned: set[str] = set()

    _print_stall_warnings(
        active_runs=[
            {"slot": "bravo", "variant_id": "v1", "started_at": "2026-04-13T12:00:00Z"}
        ],
        indexed={"v1": {"model": "gpt-test", "reasoning_effort": "xhigh"}},
        sqlite_path=tmp_path / "state.sqlite",
        review_cwd=tmp_path,
        warned_slots=warned,
    )

    captured = capsys.readouterr()
    assert "possible stall: bravo idle 20m; wrapper will keep waiting." in captured.err
    assert warned == {"bravo"}


def test_print_stall_warnings_suppresses_recent_visible_activity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rollout_path = tmp_path / "review.jsonl"
    rollout_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "review_suite_local.utc_now",
        lambda: __import__("datetime").datetime.fromisoformat(
            "2026-04-13T12:20:00+00:00"
        ),
    )
    monkeypatch.setattr(
        "review_suite_local._live_review_thread",
        lambda **_: {"rollout_path": str(rollout_path)},
    )
    monkeypatch.setattr(
        "review_suite_local.rollout_activity_summary",
        lambda path: {
            "last_event_at": __import__("datetime").datetime.fromisoformat(
                "2026-04-13T12:19:00+00:00"
            ),
            "last_meaningful_at": __import__("datetime").datetime.fromisoformat(
                "2026-04-13T12:15:00+00:00"
            ),
            "last_meaningful_type": "reasoning",
        },
    )
    warned: set[str] = set()

    _print_stall_warnings(
        active_runs=[
            {"slot": "bravo", "variant_id": "v1", "started_at": "2026-04-13T12:00:00Z"}
        ],
        indexed={"v1": {"model": "gpt-test", "reasoning_effort": "xhigh"}},
        sqlite_path=tmp_path / "state.sqlite",
        review_cwd=tmp_path,
        warned_slots=warned,
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    assert warned == set()


def test_live_review_thread_does_not_return_parent_launcher_when_child_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stderr_path = tmp_path / "stderr.txt"
    stderr_path.write_text("session id: launcher-thread\n", encoding="utf-8")
    monkeypatch.setattr(
        "review_suite_local.find_thread_by_id",
        lambda **_: {
            "id": "launcher-thread",
            "source": "{}",
            "cwd": str(tmp_path),
            "title": "review-title",
        },
    )
    monkeypatch.setattr("review_suite_local.find_review_child_thread", lambda **_: None)

    thread = _live_review_thread(
        run={
            "slot": "bravo",
            "title": "review-title",
            "stderr_path": str(stderr_path),
            "started_at": "2026-04-13T12:00:00Z",
        },
        variant={"model": "gpt-test", "reasoning_effort": "xhigh"},
        sqlite_path=tmp_path / "state.sqlite",
        review_cwd=tmp_path,
    )

    assert thread is None


def test_live_review_thread_uses_launcher_parent_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stderr_path = tmp_path / "stderr.txt"
    stderr_path.write_text("Session ID: launcher-thread\n", encoding="utf-8")
    observed: dict[str, object] = {}

    monkeypatch.setattr("review_suite_local.find_thread_by_id", lambda **_: None)

    def fake_find_review_child_thread(**kwargs: object) -> None:
        observed.setdefault("parent_thread_id", kwargs["parent_thread_id"])
        return None

    monkeypatch.setattr(
        "review_suite_local.find_review_child_thread", fake_find_review_child_thread
    )

    thread = _live_review_thread(
        run={
            "slot": "bravo",
            "title": "review-title",
            "stderr_path": str(stderr_path),
            "started_at": "2026-04-13T12:00:00Z",
            "effective_reasoning_effort": "xhigh",
        },
        variant={"model": "gpt-5.5", "reasoning_effort": "max"},
        sqlite_path=tmp_path / "state.sqlite",
        review_cwd=tmp_path,
    )

    assert thread is None
    assert observed["parent_thread_id"] == "launcher-thread"


def test_live_review_thread_rejects_non_review_launcher_without_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stderr_path = tmp_path / "stderr.txt"
    stderr_path.write_text("session id: reviewer-thread\n", encoding="utf-8")
    candidate = {
        "id": "reviewer-thread",
        "source": "",
        "cwd": str(tmp_path),
        "title": "review-title",
        "model": "gpt-test",
        "reasoning_effort": "xhigh",
        "created_at": int(
            __import__("datetime")
            .datetime.fromisoformat("2026-04-13T12:00:01+00:00")
            .timestamp()
        ),
    }
    monkeypatch.setattr("review_suite_local.find_thread_by_id", lambda **_: candidate)
    monkeypatch.setattr("review_suite_local.find_review_child_thread", lambda **_: None)

    thread = _live_review_thread(
        run={
            "slot": "bravo",
            "title": "review-title",
            "stderr_path": str(stderr_path),
            "started_at": "2026-04-13T12:00:00Z",
        },
        variant={"model": "gpt-test", "reasoning_effort": "xhigh"},
        sqlite_path=tmp_path / "state.sqlite",
        review_cwd=tmp_path,
    )

    assert thread is None


def test_live_review_thread_rejects_stale_direct_matching_review_thread(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stderr_path = tmp_path / "stderr.txt"
    stderr_path.write_text("session id: reviewer-thread\n", encoding="utf-8")
    candidate = {
        "id": "reviewer-thread",
        "source": "",
        "cwd": str(tmp_path),
        "title": "review-title",
        "model": "gpt-test",
        "reasoning_effort": "xhigh",
        "created_at": int(
            __import__("datetime")
            .datetime.fromisoformat("2026-04-13T11:59:00+00:00")
            .timestamp()
        ),
    }
    monkeypatch.setattr("review_suite_local.find_thread_by_id", lambda **_: candidate)
    monkeypatch.setattr("review_suite_local.find_review_child_thread", lambda **_: None)

    thread = _live_review_thread(
        run={
            "slot": "bravo",
            "title": "review-title",
            "stderr_path": str(stderr_path),
            "started_at": "2026-04-13T12:00:00Z",
        },
        variant={"model": "gpt-test", "reasoning_effort": "xhigh"},
        sqlite_path=tmp_path / "state.sqlite",
        review_cwd=tmp_path,
    )

    assert thread is None


def test_output_isatty_requires_stdout_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Stream:
        def __init__(self, *, tty: bool) -> None:
            self._tty = tty

        def isatty(self) -> bool:
            return self._tty

    monkeypatch.setattr(sys, "stdout", _Stream(tty=False))
    monkeypatch.setattr(sys, "stderr", _Stream(tty=True))

    assert output_isatty() is False


def test_find_blocking_rounds_for_caller_ignores_dismissed_and_completed_blocked_rounds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "review_suite_local.iter_round_payloads",
        lambda state_dir: [
            {
                "round_id": "dismissed-round",
                "caller_id": "caller-1",
                "review_cwd_normalized": str(tmp_path),
                "status": "dismissed",
                "runs": [],
            },
            {
                "round_id": "completed-blocked-round",
                "caller_id": "caller-1",
                "review_cwd_normalized": str(tmp_path),
                "status": "completed",
                "runs": [
                    {
                        "slot": "alpha",
                        "review_status": "timeout",
                        "grade_blocked": True,
                    }
                ],
            },
            {
                "round_id": "running-round",
                "caller_id": "caller-1",
                "review_cwd_normalized": str(tmp_path),
                "status": "running",
                "runs": [],
            },
        ],
    )
    monkeypatch.setattr(
        "review_suite_local.round_has_live_reviewer_process", lambda payload: False
    )

    blocking = find_blocking_rounds_for_caller(
        state_dir=tmp_path,
        caller_id="caller-1",
        review_cwd=tmp_path,
    )

    assert [payload["round_id"] for payload in blocking] == ["running-round"]


def test_reviewer_completion_status_never_classifies_review_content() -> None:
    assert (
        reviewer_completion_status(
            {
                "reviewer_output": "Full review",
                "status_summary": "summary",
                "review_status": "completed",
            }
        )
        == "completed"
    )
    assert (
        reviewer_completion_status(
            {"reviewer_output": "P2 - bug", "review_status": "completed"}
        )
        == "completed"
    )
    assert (
        reviewer_completion_status(
            {
                "review_status": "transport_stalled",
                "grade_block_reason": "review_transport_stalled",
            }
        )
        == "review_transport_stalled"
    )


def test_maybe_retry_capacity_run_emits_retry_notice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    launches: list[dict[str, object]] = []

    monkeypatch.setattr(
        "review_suite_local._summarize_live_run",
        lambda run: {"grade_block_reason": "selected_model_at_capacity"},
    )
    monkeypatch.setattr("review_suite_local._cleanup_run_artifacts", lambda run: True)
    monkeypatch.setattr("review_suite_local.time.sleep", lambda seconds: None)
    monkeypatch.setattr(
        "review_suite_local._launch_reviewer_process",
        lambda **kwargs: launches.append(kwargs) or kwargs["run"],
    )
    monkeypatch.setattr("review_suite_local.write_round", lambda *args, **kwargs: None)

    run = {"slot": "alpha", "variant_id": "model-a"}

    assert _maybe_retry_capacity_run(
        round_payload={"requested_prompt": "", "review_scope": {}},
        run=run,
        indexed={"model-a": {"model": "gpt-5.4", "reasoning_effort": "medium"}},
        state_dir=tmp_path,
        review_cwd=tmp_path,
    )

    captured = capsys.readouterr()
    assert "alpha hit capacity; retrying in 10s (attempt 1/1)" in captured.err
    assert run["capacity_retry_attempts"] == 1 and launches
    launches.clear()
    monkeypatch.setattr("review_suite_local._cleanup_run_artifacts", lambda run: False)
    retry_args = {
        "round_payload": {"status": "running", "requested_prompt": ""},
        "run": {"slot": "alpha", "variant_id": "model-a"},
        "indexed": {"model-a": {}},
        "state_dir": tmp_path,
        "review_cwd": tmp_path,
    }
    with pytest.raises(OSError, match="artifact cleanup incomplete"):
        _maybe_retry_capacity_run(**retry_args)
    assert not launches and retry_args["round_payload"]["status"] == "running"
    monkeypatch.setattr("review_suite_local._cleanup_run_artifacts", lambda run: True)
    assert _maybe_retry_capacity_run(**retry_args)
    assert retry_args["run"]["capacity_retry_attempts"] == 1 and len(launches) == 1


def test_launch_reviewer_process_writes_prompt_for_prompted_base_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakeStdin:
        def __init__(self) -> None:
            self.writes: list[str] = []
            self.closed = False

        def write(self, value: str) -> None:
            self.writes.append(value)

        def close(self) -> None:
            self.closed = True

    class FakeProc:
        def __init__(self) -> None:
            self.stdin = FakeStdin()
            self.pid = 4242

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["stdin"] = kwargs.get("stdin")
        captured["proc"] = FakeProc()
        return captured["proc"]

    monkeypatch.setattr("review_suite_local.subprocess.Popen", fake_popen)
    monkeypatch.setattr("review_suite_local.time.monotonic", lambda: 12.5)
    monkeypatch.setattr(
        "review_suite_local.utc_now_iso", lambda: "2026-04-21T18:30:00Z"
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.isolated_runtime_user_config_overrides",
        lambda: [],
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.validated_linear_review_range",
        lambda cwd, start_ref, end_ref, label: None,
    )

    launched = _launch_reviewer_process(
        round_payload={"round_id": "round-1"},
        run={"slot": "alpha", "variant_id": "gpt-5.5-xhigh"},
        variant={"model": "gpt-5.5", "reasoning_effort": "xhigh"},
        review_cwd=tmp_path,
        prompt="manual prompt",
        review_scope={"base": "main"},
        allow_unsafe_windows_wsl_fallback=False,
    )

    proc = captured["proc"]
    command = captured["command"]
    assert command[1] == "exec"
    assert "--ignore-user-config" in command
    assert "review" in command
    assert "--base" not in command
    assert "--commit" not in command
    assert command[-1] == "-"
    assert 'approval_policy="never"' in command
    assert len(proc.stdin.writes) == 1
    assert (
        "Review only for concrete technical merge-readiness risks"
        in proc.stdin.writes[0]
    )
    assert "manual prompt" in proc.stdin.writes[0]
    assert "base ref `main`" in proc.stdin.writes[0]
    assert "BEGIN DIFF" not in proc.stdin.writes[0]
    assert captured["stdin"] == subprocess.PIPE
    assert proc.stdin.closed is True
    assert launched["pid"] == 4242
    assert "final_message_path" in launched
    Path(str(launched["final_message_path"])).unlink(missing_ok=True)


def test_launch_round_staggers_identical_parallel_reviewers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        "review_suite_local._launch_reviewer_process",
        lambda **kwargs: events.append(f"launch:{kwargs['run']['slot']}"),
    )
    monkeypatch.setattr(
        "review_suite_local.time.sleep",
        lambda seconds: events.append(f"sleep:{seconds}"),
    )
    monkeypatch.setattr("review_suite_local.write_round", lambda *args: None)

    review_suite_local.launch_round(
        round_payload={
            "round_id": "round-identical",
            "status": "sampled",
            "runs": [
                {"slot": "alpha", "variant_id": "same-model"},
                {"slot": "bravo", "variant_id": "same-model"},
            ],
        },
        roster={
            "variants": [
                {
                    "id": "same-model",
                    "model": "gpt-5.6-sol",
                    "reasoning_effort": "medium",
                }
            ]
        },
        state_dir=tmp_path,
        review_cwd=tmp_path,
        prompt="review",
        review_scope={},
    )

    assert events == [
        "launch:alpha",
        f"sleep:{review_suite_local.MULTI_REVIEW_DISPATCH_STAGGER_SECONDS}",
        "launch:bravo",
    ]


def test_print_live_completed_run_includes_terminal_status(capsys) -> None:
    _print_live_completed_run(
        {
            "slot": "Alpha",
            "review_status": "completed",
            "status_summary": "summary only",
            "reviewer_output": "Full review",
        }
    )

    captured = capsys.readouterr()
    assert "Alpha: completed" in captured.err
    assert "summary only" not in captured.err
    assert "Full review" not in captured.err


def test_public_round_result_includes_public_task_name() -> None:
    result = public_round_result(
        {
            "round_id": "round-1",
            "task_class": "phase_review",
            "status": "completed",
            "runs": [],
        }
    )

    assert result["task"] == "review_t1"


def test_aggregate_records_tracks_finding_rates_and_low_quality_losses() -> None:
    roster = {
        "settings": {"elo_k_factor": 24},
        "variants": [
            _variant("alpha-model", task_classes=["phase_review"]),
            _variant("bravo-model", task_classes=["phase_review"]),
        ],
    }
    records = [
        _record(
            recorded_at="2026-04-12T13:00:00Z",
            round_id="round-valid",
            task_class="phase_review",
            alpha="alpha-model",
            beta="bravo-model",
            basis="valid_findings_vs_none",
        ),
        _record(
            recorded_at="2026-04-12T13:01:00Z",
            round_id="round-garbage",
            task_class="phase_review",
            alpha="alpha-model",
            beta="bravo-model",
            groups=[["bravo-model"], ["alpha-model"]],
            basis="false_positive_loss",
        ),
        _record(
            recorded_at="2026-04-12T13:02:00Z",
            round_id="round-tie",
            task_class="phase_review",
            alpha="alpha-model",
            beta="bravo-model",
            groups=[["alpha-model", "bravo-model"]],
            basis="tie_both_useful",
        ),
        _record(
            recorded_at="2026-04-12T13:03:00Z",
            round_id="round-quality",
            task_class="phase_review",
            alpha="alpha-model",
            beta="bravo-model",
            groups=[["bravo-model"], ["alpha-model"]],
            basis="better_finding_validity",
        ),
        _record(
            recorded_at="2026-04-12T13:04:00Z",
            round_id="round-scope-bloat",
            task_class="phase_review",
            alpha="alpha-model",
            beta="bravo-model",
            groups=[["bravo-model"], ["alpha-model"]],
            basis="scope_bloat_loss",
        ),
    ]

    summary = aggregate_records(
        roster=roster,
        records=records,
        operational_state=_operational_state(
            champion_ids=[], probation_ids=[], cooling={}
        ),
    )

    rows = {
        row["variant_id"]: row
        for row in summary["task_classes"]["phase_review"]["leaderboard"]
    }
    zero_low_quality_reasons = dict.fromkeys(LOW_QUALITY_LOSS_REASON_BASES, 0)
    assert rows["alpha-model"]["finding_opportunity_count"] == 3
    assert rows["alpha-model"]["valid_finding_count"] == 2
    assert rows["alpha-model"]["valid_finding_rate"] == pytest.approx(66.667)
    assert rows["alpha-model"]["missed_bug_loss_count"] == 0
    assert rows["alpha-model"]["missed_bug_loss_rate"] == 0.0
    assert rows["alpha-model"]["low_quality_loss_count"] == 3
    assert rows["alpha-model"]["low_quality_loss_rate"] == 60.0
    assert rows["alpha-model"]["low_quality_loss_reasons"] == {
        **zero_low_quality_reasons,
        **{
            "false_positive_loss": 1,
            "scope_bloat_loss": 1,
            "better_finding_validity": 1,
        },
    }
    assert rows["bravo-model"]["finding_opportunity_count"] == 3
    assert rows["bravo-model"]["valid_finding_count"] == 2
    assert rows["bravo-model"]["valid_finding_rate"] == pytest.approx(66.667)
    assert rows["bravo-model"]["missed_bug_loss_count"] == 1
    assert rows["bravo-model"]["missed_bug_loss_rate"] == pytest.approx(33.333)
    assert rows["bravo-model"]["low_quality_loss_count"] == 0
    assert rows["bravo-model"]["low_quality_loss_rate"] == 0.0
    assert rows["bravo-model"]["low_quality_loss_reasons"] == zero_low_quality_reasons


def test_aggregate_records_isolates_latest_pool_and_projects_placements() -> None:
    roster = {
        "settings": {"elo_k_factor": 24},
        "variants": [
            _variant(variant_id, task_classes=["phase_review"])
            for variant_id in ("alpha-model", "bravo-model", "charlie-model")
        ],
    }
    records = [
        _record(
            recorded_at="2026-04-12T12:00:00Z",
            round_id="old-pool",
            task_class="phase_review",
            alpha="alpha-model",
            beta="bravo-model",
            rating_pool_id="pool-a",
        ),
        {
            "recorded_at": "2026-04-12T13:00:00Z",
            "round_id": "new-pool",
            "task_class": "phase_review",
            "task_id": "task-1",
            "selection_mode": "true_scramble",
            "rating_pool_id": "pool-b",
            "placement_v1": {
                "groups": [["alpha-model"], ["bravo-model", "charlie-model"]],
                "basis": "better_bug_coverage",
            },
            "runs": [
                {
                    "variant_id": variant_id,
                    "elapsed_seconds": seconds,
                    "usage": {"input_tokens": int(seconds)},
                    "cost_usd": seconds / 100,
                }
                for variant_id, seconds in (
                    ("alpha-model", 10.0),
                    ("bravo-model", 20.0),
                    ("charlie-model", 30.0),
                )
            ],
        },
        {
            "task_class": "phase_review",
            "pairwise_outcome": {"winner": "bravo-model"},
            "runs": [
                {"variant_id": "alpha-model"},
                {"variant_id": "bravo-model"},
            ],
        },
    ]

    task = aggregate_records(
        roster=roster,
        records=records,
        operational_state=_operational_state(
            champion_ids=[], probation_ids=[], cooling={}
        ),
    )["task_classes"]["phase_review"]

    assert task["rating_pool_id"] == "pool-b"
    rows = {row["variant_id"]: row for row in task["leaderboard"]}
    assert {variant_id: row["elo"] for variant_id, row in rows.items()} == {
        "alpha-model": 1512.0,
        "bravo-model": 1494.0,
        "charlie-model": 1494.0,
    }
    assert all(row["sample_count"] == 1 for row in rows.values())
    recent = task["recent_rounds"]
    assert len(recent) == 1
    assert recent[0]["groups"] == [
        ["alpha-model"],
        ["bravo-model", "charlie-model"],
    ]
    participants = {row["variant_id"]: row for row in recent[0]["participants"]}
    assert participants["alpha-model"] == {
        "variant_id": "alpha-model",
        "rank": 1,
        "tied": False,
        "elo_before": 1500.0,
        "elo_after": 1512.0,
        "elo_delta": 12.0,
        "elapsed_seconds": 10.0,
        "usage": {"input_tokens": 10},
        "cost_usd": 0.1,
    }

    records[0]["reporting_pool"] = True
    preferred = aggregate_records(
        roster=roster,
        records=records,
        operational_state=_operational_state(
            champion_ids=[], probation_ids=[], cooling={}
        ),
    )["task_classes"]["phase_review"]
    assert preferred["rating_pool_id"] == "pool-a"
    assert len(preferred["recent_rounds"]) == 1


def test_aggregate_records_does_not_select_ungraded_reporting_pool() -> None:
    roster = _roster(_variant("alpha-model"), _variant("bravo-model"))
    records = [
        _record(
            recorded_at="2026-04-12T12:00:00Z",
            round_id="discovery",
            task_class="phase_review",
            alpha="alpha-model",
            beta="bravo-model",
            rating_pool_id="discovery-pool",
        ),
        {
            "task_class": "phase_review",
            "rating_pool_id": "arena-pool",
            "reporting_pool": True,
            "runs": [{"variant_id": "alpha-model"}, {"variant_id": "bravo-model"}],
        },
    ]

    task = aggregate_records(
        roster=roster,
        records=records,
        operational_state=_operational_state(
            champion_ids=[], probation_ids=[], cooling={}
        ),
    )["task_classes"]["phase_review"]

    assert task["rating_pool_id"] == "discovery-pool"


def test_aggregate_records_ignores_ungraded_and_legacy_pair_records() -> None:
    roster = {
        "settings": {"elo_k_factor": 24},
        "variants": [
            _variant("alpha-model", task_classes=["phase_review"]),
            _variant("bravo-model", task_classes=["phase_review"]),
        ],
    }

    summary = aggregate_records(
        roster=roster,
        records=[
            {
                "task_class": "phase_review",
                "runs": [{"variant_id": "alpha-model"}, {"variant_id": "bravo-model"}],
            }
        ],
        operational_state=_operational_state(
            champion_ids=[], probation_ids=[], cooling={}
        ),
    )

    rows = {
        row["variant_id"]: row
        for row in summary["task_classes"]["phase_review"]["leaderboard"]
    }
    assert rows["alpha-model"]["sample_count"] == 0
    assert rows["alpha-model"]["elo"] == 1500.0
    assert rows["alpha-model"]["wtl"] == "0/0/0"
    assert rows["alpha-model"]["median_elapsed_seconds"] is None
    assert rows["bravo-model"]["sample_count"] == 0
    assert rows["bravo-model"]["elo"] == 1500.0
    assert rows["bravo-model"]["wtl"] == "0/0/0"
    assert rows["bravo-model"]["median_elapsed_seconds"] is None
    assert summary["task_classes"]["phase_review"]["recent_rounds"] == []


def test_ungraded_round_exposure_records_include_pending_but_skip_graded(
    tmp_path: Path,
) -> None:
    write_round(
        tmp_path,
        {
            "round_id": "pending-round",
            "task_class": "pr_review",
            "status": "running",
            "runs": [{"variant_id": "gpt-5.5-high"}, {"variant_id": "gpt-5.4-xhigh"}],
        },
    )
    write_round(
        tmp_path,
        {
            "round_id": "graded-round",
            "task_class": "pr_review",
            "status": "completed",
            "graded_at": "2026-04-23T12:00:00Z",
            "runs": [{"variant_id": "gpt-5.5-low"}, {"variant_id": "gpt-5.4-medium"}],
        },
    )
    write_round(
        tmp_path,
        {
            "round_id": "dismissed-round",
            "task_class": "pr_review",
            "status": "dismissed",
            "runs": [{"variant_id": "gpt-5.5-medium"}, {"variant_id": "gpt-5.4-high"}],
        },
    )
    write_round(
        tmp_path,
        {
            "round_id": "rerolled-round",
            "rerolled_from_round_id": "pending-round",
            "task_class": "pr_review",
            "rating_pool_id": "fresh-pool",
            "status": "running",
            "runs": [{"variant_id": "gpt-5.5-high"}, {"variant_id": "gpt-5.4-xhigh"}],
        },
    )

    records = ungraded_round_exposure_records(tmp_path)

    assert records == [
        {
            "task_class": "pr_review",
            "rating_pool_id": "fresh-pool",
            "runs": [{"variant_id": "gpt-5.5-high"}, {"variant_id": "gpt-5.4-xhigh"}],
        }
    ]


def test_cleanup_stale_ungraded_rounds_dismisses_old_non_live_round(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "review_suite_local.utc_now",
        lambda: datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr("review_suite_local._process_is_running", lambda pid: False)
    artifact = tmp_path / "stdout"
    artifact.write_text("captured", encoding="utf-8")
    write_round(
        tmp_path,
        {
            "round_id": "old-running-round",
            "task_class": "pr_review",
            "status": "running",
            "sampled_at": "2026-05-02T11:59:00Z",
            "runs": [
                {
                    "variant_id": "gpt-5.5-high",
                    "pid": 12345,
                    "stdout_path": str(artifact),
                }
            ],
        },
    )

    cleaned = cleanup_stale_ungraded_rounds(tmp_path)

    assert cleaned == [
        {
            "round_id": "old-running-round",
            "previous_status": "running",
            "reason": "auto_stale_ungraded_round_24h",
        }
    ]
    payload = json.loads(
        (tmp_path / "rounds" / "old-running-round.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "dismissed"
    assert payload["dismissed_previous_status"] == "running"
    assert "_round_file_path" not in payload
    assert not artifact.exists()


def test_write_round_keeps_artifacts_when_dismissal_persistence_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = tmp_path / "stdout"
    artifact.write_text("captured", encoding="utf-8")

    def fail_write(path: Path, payload: dict[str, object]) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("review_suite_local.write_json", fail_write)
    with pytest.raises(OSError, match="disk full"):
        write_round(
            tmp_path,
            {
                "round_id": "interrupted",
                "status": "dismissed",
                "runs": [{"stdout_path": str(artifact)}],
            },
        )

    assert artifact.exists()


def test_write_round_retains_artifact_path_when_cleanup_fails(tmp_path: Path) -> None:
    cleaned = tmp_path / "cleaned"
    cleaned.write_text("captured", encoding="utf-8")
    artifact = tmp_path / "locked"
    artifact.mkdir()
    with pytest.raises(OSError, match="artifact cleanup incomplete"):
        write_round(
            tmp_path,
            {
                "round_id": "interrupted",
                "status": "dismissed",
                "dismissed_previous_status": "running",
                "runs": [{"stdout_path": str(cleaned), "stderr_path": str(artifact)}],
            },
        )
    assert cleanup_stale_ungraded_rounds(tmp_path, stale_seconds=0) == []
    saved = review_suite_local.load_round(tmp_path, "interrupted")
    assert saved["status"] == "cleanup_pending"
    assert saved["dismissed_previous_status"] == "running"
    assert saved["runs"][0] == {"stderr_path": str(artifact)}
    assert not cleaned.exists()


def test_ungraded_round_exposure_records_auto_skips_stale_rounds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "review_suite_local.utc_now",
        lambda: datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr("review_suite_local._process_is_running", lambda pid: False)
    write_round(
        tmp_path,
        {
            "round_id": "old-pending-round",
            "task_class": "pr_review",
            "status": "sampled",
            "sampled_at": "2026-05-02T11:59:00Z",
            "runs": [{"variant_id": "gpt-5.5-high"}, {"variant_id": "gpt-5.4-xhigh"}],
        },
    )

    assert ungraded_round_exposure_records(tmp_path) == []

    payload = json.loads(
        (tmp_path / "rounds" / "old-pending-round.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "dismissed"


def test_ungraded_round_exposure_records_readonly_skips_stale_without_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "review_suite_local.utc_now",
        lambda: datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr("review_suite_local._process_is_running", lambda pid: False)
    write_round(
        tmp_path,
        {
            "round_id": "old-pending-round",
            "task_class": "pr_review",
            "status": "sampled",
            "sampled_at": "2026-05-02T11:59:00Z",
            "runs": [{"variant_id": "gpt-5.5-high"}, {"variant_id": "gpt-5.4-xhigh"}],
        },
    )
    round_file = tmp_path / "rounds" / "old-pending-round.json"
    before = round_file.read_text(encoding="utf-8")

    assert ungraded_round_exposure_records(tmp_path, cleanup=False) == []

    after = round_file.read_text(encoding="utf-8")
    assert after == before
    assert json.loads(after)["status"] == "sampled"


def test_ungraded_round_exposure_records_excludes_stale_when_cleanup_write_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "review_suite_local.utc_now",
        lambda: datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr("review_suite_local._process_is_running", lambda pid: False)
    write_round(
        tmp_path,
        {
            "round_id": "old-pending-round",
            "task_class": "pr_review",
            "status": "sampled",
            "sampled_at": "2026-05-02T11:59:00Z",
            "runs": [{"variant_id": "gpt-5.5-high"}, {"variant_id": "gpt-5.4-xhigh"}],
        },
    )

    def fail_write(state_dir: Path, payload: dict[str, object]) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr("review_suite_local.write_round", fail_write)

    assert ungraded_round_exposure_records(tmp_path) == []


def test_cleanup_stale_ungraded_rounds_keeps_live_round(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "review_suite_local.utc_now",
        lambda: datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr("review_suite_local._process_is_running", lambda pid: True)
    write_round(
        tmp_path,
        {
            "round_id": "old-live-round",
            "task_class": "pr_review",
            "status": "running",
            "sampled_at": "2026-05-02T11:59:00Z",
            "runs": [{"variant_id": "gpt-5.5-high", "pid": 12345}],
        },
    )

    assert cleanup_stale_ungraded_rounds(tmp_path) == []
    payload = json.loads(
        (tmp_path / "rounds" / "old-live-round.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "running"


def test_select_pair_explores_least_exposed_under_sampled_variant_first() -> None:
    roster = _roster(
        _variant("gpt-5.5-low", task_classes=["pr_review"]),
        _variant("gpt-5.5-medium", task_classes=["pr_review"]),
        _variant("gpt-5.5-high", task_classes=["pr_review"]),
        _variant("established", task_classes=["pr_review"]),
    )
    records = [
        _record(
            recorded_at=f"2026-04-23T12:0{idx}:00Z",
            round_id=f"round-{idx}",
            task_class="pr_review",
            alpha="gpt-5.5-high",
            beta="established",
        )
        for idx in range(4)
    ]

    payload = select_pair(
        roster=roster,
        operational_state=_operational_state(
            champion_ids=[], probation_ids=[], cooling={}
        ),
        records=records,
        task_class="pr_review",
        review_cwd=None,
        seed=4,
    )

    assert payload["runs"][0]["variant_id"] in {"gpt-5.5-low", "gpt-5.5-medium"}


def test_configured_selection_uses_only_its_fresh_pool_schedule() -> None:
    roster = _roster(*(_variant(name) for name in "abcde"))
    variant_groups = [["a", "b", "c", "d"], ["b", "c", "d", "e"]]
    records = [
        {"task_class": "phase_review", "rating_pool_id": "legacy", "runs": []},
        {
            "task_class": "phase_review",
            "rating_pool_id": "fresh-pool",
            "runs": [],
        },
    ]

    payload = select_pair(
        roster=roster,
        operational_state=_operational_state(
            champion_ids=[], probation_ids=[], cooling={}
        ),
        records=records,
        task_class="phase_review",
        review_cwd=None,
        seed=None,
        rating_pool_id="fresh-pool",
        variant_groups=variant_groups,
    )

    assert payload["selection_mode"] == "configured"
    assert payload["rating_pool_id"] == "fresh-pool"
    assert payload["rating_pool_variant_ids"] == ["a", "b", "c", "d", "e"]
    assert payload["schedule_index"] == 1
    assert [run["variant_id"] for run in payload["runs"]] == ["b", "c", "d", "e"]


def test_configured_selection_evenly_bootstraps_drop_in_candidates() -> None:
    roster = _roster(*(_variant(name) for name in "abcdefghi"))
    records = [
        {
            "task_class": "phase_review",
            "rating_pool_id": "arena-pool",
            "runs": [{"variant_id": name} for name in "abcd"],
        }
        for _ in range(8)
    ]
    sampled: set[str] = set()

    for _ in range(3):
        payload = select_pair(
            roster=roster,
            operational_state=_operational_state(
                champion_ids=[], probation_ids=[], cooling={}
            ),
            records=records,
            task_class="phase_review",
            review_cwd=None,
            seed=None,
            rating_pool_id="arena-pool",
            variant_groups=[["a", "b", "c", "d"]],
            variant_ids=list("abcdefghi"),
        )
        selected = [run["variant_id"] for run in payload["runs"]]
        assert payload["selection_pairing"] == "configured_balanced"
        assert len(set(selected) & set("abcd")) == 2
        assert len(set(selected) & set("efghi")) == 2
        assert "schedule_index" not in payload
        sampled.update(selected)
        records.append(payload)

    assert set("efghi") <= sampled


@pytest.mark.parametrize(
    "pool_name,task_class",
    [("arena_phase", "phase_review"), ("arena_deep", "pr_review")],
)
def test_shipped_pools_can_select_after_the_initial_schedule(
    pool_name: str, task_class: str
) -> None:
    references = SCRIPT_DIR.parent / "references"
    pool = tomllib.loads((references / "arena_settings.toml").read_text())["arena"][
        "pools"
    ][pool_name]
    roster = review_suite_local.load_roster(references / "roster.json")
    records = [
        {"task_class": task_class, "rating_pool_id": pool["rating_pool_id"], "runs": []}
        for _ in pool["variant_groups"]
    ]
    payload = select_pair(
        roster=roster,
        operational_state=review_suite_local.default_operational_state(),
        records=records,
        task_class=task_class,
        review_cwd=None,
        seed=None,
        rating_pool_id=pool["rating_pool_id"],
        variant_groups=pool["variant_groups"],
        variant_ids=pool["variant_ids"],
    )
    assert len(payload["runs"]) == 4
    assert payload["selection_pairing"] == "configured_balanced"


def test_balanced_configured_selection_preserves_reviewer_slot_limit() -> None:
    roster = _roster(*(_variant(name) for name in "abcdefgh"))

    with pytest.raises(ValueError, match="configured arena group exceeds 6"):
        select_pair(
            roster=roster,
            operational_state=_operational_state(
                champion_ids=[], probation_ids=[], cooling={}
            ),
            records=[
                {
                    "task_class": "phase_review",
                    "rating_pool_id": "arena-pool",
                    "runs": [{"variant_id": name} for name in "abcdefg"],
                }
            ],
            task_class="phase_review",
            review_cwd=None,
            seed=None,
            rating_pool_id="arena-pool",
            variant_groups=[["a", "b", "c", "d", "e", "f", "g"]],
            variant_ids=["a", "b", "c", "d", "e", "f", "g", "h"],
        )


def test_select_pair_uses_scramble_even_when_champion_metadata_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roster = _roster(
        _variant("champion"),
        _variant("acting-high"),
        _variant("acting-low"),
        _variant("challenger"),
        _variant("probation"),
    )
    operational_state = _operational_state(
        champion_ids=["champion"],
        probation_ids=["probation"],
        cooling={"champion": {"until": "2999-01-01T00:00:00Z", "failure_count": 1}},
    )

    monkeypatch.setattr(
        "review_suite_local.aggregate_records",
        lambda **_: _summary_with_leaderboard(
            {"variant_id": "acting-high", "sample_count": 40, "elo": 1540.0},
            {"variant_id": "acting-low", "sample_count": 30, "elo": 1510.0},
            {"variant_id": "challenger", "sample_count": 12, "elo": 1530.0},
            {"variant_id": "probation", "sample_count": 50, "elo": 1560.0},
            {"variant_id": "champion", "sample_count": 60, "elo": 1600.0},
        ),
    )

    payload = select_pair(
        roster=roster,
        operational_state=operational_state,
        records=[],
        task_class="phase_review",
        review_cwd=None,
        seed=7,
    )

    assert payload["selection_mode"] == "legacy"
    assert payload["selection_anchor_kind"] is None
    assert payload["selection_fallback_reason"] is None
    assert payload["selection_champion_variant_ids"] == []
    assert len(payload["runs"]) == 2


def test_select_pair_does_not_require_acting_champion_after_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roster = _roster(
        _variant("champion"),
        _variant("low-quality"),
        _variant("probation"),
    )
    operational_state = _operational_state(
        champion_ids=["champion"],
        probation_ids=["probation"],
        cooling={"champion": {"until": "2999-01-01T00:00:00Z", "failure_count": 1}},
    )

    monkeypatch.setattr(
        "review_suite_local.aggregate_records",
        lambda **_: _summary_with_leaderboard(
            {"variant_id": "low-quality", "sample_count": 35, "elo": 1545.0},
            {"variant_id": "probation", "sample_count": 45, "elo": 1570.0},
            {"variant_id": "champion", "sample_count": 60, "elo": 1600.0},
        ),
    )

    payload = select_pair(
        roster=roster,
        operational_state=operational_state,
        records=[],
        task_class="phase_review",
        review_cwd=None,
        seed=7,
    )

    assert payload["selection_mode"] == "legacy"
    assert payload["selection_champion_variant_ids"] == []


def test_select_pair_true_scramble_rolls_both_slots_uniformly() -> None:
    roster = _roster(
        _variant("under-exposed-a"),
        _variant("under-exposed-b"),
        _variant("established-a"),
        _variant("established-b"),
    )
    roster["settings"]["selection_mode"] = "true_scramble"
    records = [
        _record(
            recorded_at=f"2026-04-23T12:{idx:02d}:00Z",
            round_id=f"round-{idx}",
            task_class="phase_review",
            alpha="established-a",
            beta="established-b",
            groups=[["established-a", "established-b"]],
            basis="tie_clean",
        )
        for idx in range(20)
    ]

    payload = select_pair(
        roster=roster,
        operational_state=_operational_state(
            champion_ids=[], probation_ids=[], cooling={}
        ),
        records=records,
        task_class="phase_review",
        review_cwd=None,
        seed=2,
    )

    assert payload["selection_mode"] == "true_scramble"
    assert payload["selection_pairing"] == "true_scramble_random"
    assert len({run["variant_id"] for run in payload["runs"]}) == 2


def test_select_pair_slight_bias_rolls_both_slots_from_same_elo_weighted_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roster = _roster(
        _variant("strong"),
        _variant("middle"),
        _variant("weak"),
    )
    roster["settings"]["selection_mode"] = "slight_bias"

    monkeypatch.setattr(
        "review_suite_local.aggregate_records",
        lambda **_: _summary_with_leaderboard(
            {"variant_id": "strong", "sample_count": 12, "elo": 1600.0},
            {"variant_id": "middle", "sample_count": 12, "elo": 1500.0},
            {"variant_id": "weak", "sample_count": 12, "elo": 1400.0},
        ),
    )

    payload = select_pair(
        roster=roster,
        operational_state=_operational_state(
            champion_ids=[], probation_ids=[], cooling={}
        ),
        records=[],
        task_class="phase_review",
        review_cwd=None,
        seed=11,
    )

    assert payload["selection_mode"] == "slight_bias"
    assert payload["selection_pairing"] == "slight_bias_elo_weighted"
    assert payload["selection_champion_variant_ids"] == []
    assert len({run["variant_id"] for run in payload["runs"]}) == 2


def test_select_pair_rejects_unknown_selection_mode() -> None:
    roster = _roster(_variant("alpha"), _variant("bravo"))
    roster["settings"]["selection_mode"] = "weighted_champion"

    with pytest.raises(ValueError, match="unknown selection_mode"):
        select_pair(
            roster=roster,
            operational_state=_operational_state(
                champion_ids=[], probation_ids=[], cooling={}
            ),
            records=[],
            task_class="phase_review",
            review_cwd=None,
            seed=1,
        )


def test_reroll_candidates_fall_back_to_acting_champion_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roster = _roster(
        _variant("champion"),
        _variant("acting-high"),
        _variant("acting-low"),
        _variant("probation"),
    )
    operational_state = _operational_state(
        champion_ids=["champion"],
        probation_ids=["probation"],
        cooling={"champion": {"until": "2999-01-01T00:00:00Z", "failure_count": 1}},
    )

    monkeypatch.setattr(
        "review_suite_local.aggregate_records",
        lambda **_: _summary_with_leaderboard(
            {"variant_id": "acting-high", "sample_count": 40, "elo": 1540.0},
            {"variant_id": "acting-low", "sample_count": 30, "elo": 1510.0},
            {"variant_id": "probation", "sample_count": 50, "elo": 1560.0},
            {"variant_id": "champion", "sample_count": 60, "elo": 1600.0},
        ),
    )

    candidates = _reroll_candidate_variants(
        roster=roster,
        operational_state=operational_state,
        records=[],
        round_payload={
            "task_class": "phase_review",
            "selection_mode": "champion",
            "selection_pairing": "champion_vs_probation",
            "selection_champion_variant_ids": ["champion"],
        },
        slot="alpha",
        excluded_variant_ids=set(),
    )

    assert [variant["id"] for variant in candidates] == ["acting-high", "acting-low"]


def test_build_reroll_slot_payload_pins_top_acting_champion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roster = _roster(
        _variant("champion"),
        _variant("acting-high"),
        _variant("acting-low"),
        _variant("probation"),
    )
    operational_state = _operational_state(
        champion_ids=["champion"],
        probation_ids=["probation"],
        cooling={"champion": {"until": "2999-01-01T00:00:00Z", "failure_count": 1}},
    )

    monkeypatch.setattr(
        "review_suite_local.aggregate_records",
        lambda **_: _summary_with_leaderboard(
            {"variant_id": "acting-high", "sample_count": 40, "elo": 1540.0},
            {"variant_id": "acting-low", "sample_count": 30, "elo": 1510.0},
            {"variant_id": "probation", "sample_count": 50, "elo": 1560.0},
            {"variant_id": "champion", "sample_count": 60, "elo": 1600.0},
        ),
    )
    monkeypatch.setattr(
        "review_suite_local.pick_weighted_without_replacement",
        lambda variants, *_args, **_kwargs: [variants[-1]],
    )

    payload = build_reroll_slot_payload(
        round_payload={
            "round_id": "round-1",
            "status": "completed",
            "task_class": "phase_review",
            "selection_mode": "champion",
            "selection_pairing": "champion_vs_probation",
            "selection_champion_variant_ids": ["champion"],
            "runs": [
                {
                    "slot": "alpha",
                    "variant_id": "champion",
                    "review_status": "interrupted_capacity",
                    "grade_blocked": True,
                    "grade_block_reason": "selected_model_at_capacity",
                    "status_summary": "capacity",
                    "reviewer_output": "",
                },
                {
                    "slot": "bravo",
                    "variant_id": "probation",
                    "review_status": "completed",
                    "grade_blocked": False,
                    "grade_block_reason": None,
                    "status_summary": "No findings.",
                    "reviewer_output": "No findings.",
                },
            ],
        },
        roster=roster,
        operational_state=operational_state,
        records=[],
        slot="alpha",
        seed=0,
    )

    assert payload["runs"][0]["variant_id"] == "acting-high"
    assert payload["selection_anchor_kind"] == "acting_champion"
    assert payload["selection_fallback_reason"] == "champion_pool_unavailable"


def test_configured_reroll_preserves_four_model_cohort() -> None:
    variants = [_variant(name) for name in ("a", "b", "c", "d")]
    runs = [
        {
            "slot": slot,
            "variant_id": variant["id"],
            "model": variant["model"],
            "reasoning_effort": variant["reasoning_effort"],
            "review_status": "interrupted_capacity"
            if slot == "charlie"
            else "completed",
            "grade_blocked": slot == "charlie",
            "grade_block_reason": "selected_model_at_capacity"
            if slot == "charlie"
            else None,
            "reviewer_output": "" if slot == "charlie" else "No findings.",
        }
        for slot, variant in zip(("alpha", "bravo", "charlie", "delta"), variants)
    ]

    payload = build_reroll_slot_payload(
        round_payload={
            "round_id": "round-1",
            "status": "completed",
            "task_class": "phase_review",
            "selection_mode": "configured",
            "selection_pairing": "configured_schedule",
            "rating_pool_id": "fresh-pool",
            "schedule_index": 3,
            "schedule_length": 13,
            "runs": runs,
        },
        roster=_roster(*variants),
        operational_state=_operational_state(
            champion_ids=[], probation_ids=[], cooling={}
        ),
        records=[],
        slot="charlie",
        seed=None,
    )

    assert [run["variant_id"] for run in payload["runs"]] == ["a", "b", "c", "d"]
    assert payload["runs"][2]["rerolled_from_variant_id"] == "c"
    assert payload["rating_pool_id"] == "fresh-pool"


def test_configured_reroll_replaces_timed_out_variant() -> None:
    variants = [_variant(name) for name in ("a", "b", "c", "d", "e")]
    outsider = _variant("f")
    previously_excluded = _variant("g")
    runs = [
        {
            "slot": slot,
            "variant_id": variant["id"],
            "model": variant["model"],
            "reasoning_effort": variant["reasoning_effort"],
            "review_status": "timeout" if slot == "charlie" else "completed",
            "grade_blocked": slot == "charlie",
            "grade_block_reason": "review_timed_out" if slot == "charlie" else None,
            "reviewer_output": "" if slot == "charlie" else "No findings.",
        }
        for slot, variant in zip(("alpha", "bravo", "charlie", "delta"), variants)
    ]

    payload = build_reroll_slot_payload(
        round_payload={
            "round_id": "round-1",
            "status": "completed",
            "task_class": "phase_review",
            "selection_mode": "configured",
            "selection_pairing": "configured_schedule",
            "rating_pool_id": "fresh-pool",
            "rating_pool_variant_ids": ["a", "b", "c", "d", "e", "g"],
            "excluded_variant_ids": ["g"],
            "runs": runs,
        },
        roster=_roster(*variants, outsider, previously_excluded),
        operational_state=_operational_state(
            champion_ids=[],
            probation_ids=[],
            cooling={"c": {"until": "2999-01-01T00:00:00Z", "failure_count": 1}},
        ),
        records=[],
        slot="charlie",
        seed=None,
    )

    assert [run["variant_id"] for run in payload["runs"]] == ["a", "b", "e", "d"]
    assert payload["runs"][2]["rerolled_from_variant_id"] == "c"
    assert payload["runs"][0]["reviewer_output"] == "No findings."
    assert payload["rating_pool_id"] == "fresh-pool"
    assert payload["rating_pool_variant_ids"] == ["a", "b", "c", "d", "e", "g"]
    assert {"f", "g"} <= set(payload["excluded_variant_ids"])


def test_configured_reroll_replaces_unsupported_model() -> None:
    variants = [_variant(name) for name in ("a", "b", "c", "d", "e", "g")]
    unsupported_peer = _variant("c-high")
    unsupported_peer["model"] = "c"
    outsider = _variant("f")
    runs = [
        {
            "slot": slot,
            "variant_id": variant["id"],
            "model": variant["model"],
            "reasoning_effort": variant["reasoning_effort"],
            "review_status": "unsupported_model" if slot == "charlie" else "completed",
            "grade_blocked": slot == "charlie",
            "grade_block_reason": "selected_model_not_supported"
            if slot == "charlie"
            else None,
            "reviewer_output": "" if slot == "charlie" else "No findings.",
        }
        for slot, variant in zip(("alpha", "bravo", "charlie", "delta"), variants)
    ]

    payload = build_reroll_slot_payload(
        round_payload={
            "round_id": "round-1",
            "status": "completed",
            "task_class": "phase_review",
            "selection_mode": "configured",
            "selection_pairing": "configured_schedule",
            "rating_pool_id": "fresh-pool",
            "rating_pool_variant_ids": ["a", "b", "c", "d", "e", "g"],
            "runs": runs,
        },
        roster=_roster(*variants, unsupported_peer, outsider),
        operational_state=_operational_state(
            champion_ids=[], probation_ids=[], cooling={}
        ),
        records=[],
        slot="charlie",
        seed=None,
    )

    first_replacement = payload["runs"][2]["variant_id"]
    assert first_replacement in {"e", "g"}
    assert payload["runs"][2]["rerolled_from_variant_id"] == "c"
    assert "c-high" in payload["excluded_variant_ids"]
    assert "f" in payload["excluded_variant_ids"]
    assert payload["rating_pool_variant_ids"] == ["a", "b", "c", "d", "e", "g"]

    payload["status"] = "completed"
    payload["runs"][2].update(
        review_status="unsupported_model",
        grade_blocked=True,
        grade_block_reason="selected_model_not_supported",
    )
    second = build_reroll_slot_payload(
        round_payload=payload,
        roster=_roster(*variants, unsupported_peer, outsider),
        operational_state=_operational_state(
            champion_ids=[], probation_ids=[], cooling={}
        ),
        records=[],
        slot="charlie",
        seed=None,
    )

    assert {first_replacement, second["runs"][2]["variant_id"]} == {"e", "g"}
    assert second["rating_pool_variant_ids"] == ["a", "b", "c", "d", "e", "g"]


def test_unconfigured_reroll_replaces_unsupported_model_family() -> None:
    failed = _variant("failed-medium")
    failed["model"] = "unsupported"
    unsupported_peer = _variant("failed-high")
    unsupported_peer["model"] = "unsupported"
    survivor = _variant("survivor")
    replacement = _variant("replacement")

    payload = build_reroll_slot_payload(
        round_payload={
            "round_id": "round-1",
            "status": "completed",
            "task_class": "phase_review",
            "selection_mode": "true_scramble",
            "selection_pairing": "true_scramble",
            "runs": [
                {
                    "slot": "alpha",
                    "variant_id": failed["id"],
                    "model": failed["model"],
                    "reasoning_effort": failed["reasoning_effort"],
                    "review_status": "unsupported_model",
                    "grade_blocked": True,
                    "grade_block_reason": "selected_model_not_supported",
                    "reviewer_output": "",
                },
                {
                    "slot": "bravo",
                    "variant_id": survivor["id"],
                    "model": survivor["model"],
                    "reasoning_effort": survivor["reasoning_effort"],
                    "review_status": "completed",
                    "grade_blocked": False,
                    "grade_block_reason": None,
                    "reviewer_output": "No findings.",
                },
            ],
        },
        roster=_roster(failed, unsupported_peer, survivor, replacement),
        operational_state=_operational_state(
            champion_ids=[], probation_ids=[], cooling={}
        ),
        records=[],
        slot="alpha",
        seed=None,
    )

    assert [run["variant_id"] for run in payload["runs"]] == [
        "replacement",
        "survivor",
    ]


def test_write_reports_includes_recent_match_history_and_model_header(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    rounds_dir = state_dir / "rounds"
    rounds_dir.mkdir(parents=True)
    records = []
    for idx in range(11):
        task_class = "phase_review" if idx % 2 == 0 else "pr_review"
        alpha = "gpt-5.5-medium" if task_class == "phase_review" else "gpt-5.4-low"
        beta = (
            "gpt-5.3-codex-medium" if task_class == "phase_review" else "gpt-5.5-high"
        )
        tied = idx % 3 == 0
        round_id = f"{task_class}-round-{idx}"
        records.append(
            _record(
                recorded_at=f"2026-04-12T13:{idx:02d}:00Z",
                round_id=round_id,
                task_class=task_class,
                alpha=alpha,
                beta=beta,
                groups=[[alpha, beta]] if tied else [[alpha], [beta]],
                basis="tie_clean" if tied else "valid_findings_vs_none",
            )
        )
        records[-1]["repo_name"] = f"repo-{idx}"
        (rounds_dir / f"{round_id}.json").write_text(
            json.dumps({"review_cwd": f"C:/Users/alice/.codex/worktrees/repo-{idx}"}),
            encoding="utf-8",
        )
    (state_dir / "runs.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    roster = {
        "settings": {"elo_k_factor": 24},
        "variants": [
            _variant("gpt-5.5-medium", task_classes=["phase_review"]),
            _variant("gpt-5.3-codex-medium", task_classes=["phase_review"]),
            _variant("gpt-5.4-low", task_classes=["pr_review"]),
            _variant("gpt-5.5-high", task_classes=["pr_review"]),
        ],
    }
    operational_state = {
        "generated_at": "2026-04-12T00:00:00Z",
        "task_classes": {
            "phase_review": {
                "mode": "champion",
                "champion_variant_id": "gpt-5.5-medium",
                "champion_variant_ids": ["gpt-5.5-medium"],
                "probation_variant_ids": [],
                "stable_variant_ids": [],
                "cooldowns": {},
            },
            "pr_review": {
                "mode": "scramble",
                "champion_variant_id": None,
                "champion_variant_ids": [],
                "probation_variant_ids": [],
                "stable_variant_ids": [],
                "cooldowns": {},
            },
        },
    }

    summary = aggregate_records(
        roster=roster, records=records, operational_state=operational_state
    )
    write_reports(state_dir, summary)

    leaderboard = (state_dir / "leaderboard.md").read_text(encoding="utf-8")

    assert (
        "| model | elo | samples | W/T/L | found/opp | found % | missed % | low-quality % | sec | tok/job | cost/job |"
        in leaderboard
    )
    assert "## round history" in leaderboard
    assert "| review | repo | rating pool | placements and Elo |" in leaderboard
    assert "## review_t1" in leaderboard
    assert "## review_t3" in leaderboard
    assert "| review_t1 | repo-10 | phase_review-v1 | 1. gpt-5.5-medium" in leaderboard
    repo_9_line = next(
        line
        for line in leaderboard.splitlines()
        if line.startswith("| review_t3 | repo-9 |")
    )
    assert "1. gpt-5.4-low" in repo_9_line
    assert " = gpt-5.5-high" in repo_9_line
    assert "repo-0" not in leaderboard


def test_write_reports_includes_independent_leaderboards_for_each_rating_pool(
    tmp_path: Path,
) -> None:
    roster = _roster(
        _variant("alpha-model", task_classes=["phase_review"]),
        _variant("bravo-model", task_classes=["phase_review"]),
        _variant("unused-model", task_classes=["phase_review"]),
    )
    discovery = _record(
        recorded_at="2026-04-12T12:00:00Z",
        round_id="discovery-round",
        task_class="phase_review",
        alpha="alpha-model",
        beta="bravo-model",
        rating_pool_id="discovery-pool",
    )
    arena = _record(
        recorded_at="2026-04-12T13:00:00Z",
        round_id="arena-round",
        task_class="phase_review",
        alpha="alpha-model",
        beta="bravo-model",
        groups=[["bravo-model"], ["alpha-model"]],
        rating_pool_id="arena-pool",
    )
    arena["reporting_pool"] = True

    summary = aggregate_records(
        roster=roster,
        records=[discovery, arena],
        operational_state=_operational_state(
            champion_ids=[], probation_ids=[], cooling={}
        ),
    )
    task = summary["task_classes"]["phase_review"]
    assert task["rating_pool_id"] == "arena-pool"
    assert [pool["rating_pool_id"] for pool in task["rating_pools"]] == [
        "arena-pool",
        "discovery-pool",
    ]

    write_reports(tmp_path, summary)

    leaderboard = (tmp_path / "leaderboard.md").read_text(encoding="utf-8")
    arena_section = leaderboard.split("### arena-pool", maxsplit=1)[1].split(
        "### discovery-pool", maxsplit=1
    )[0]
    discovery_section = leaderboard.split("### discovery-pool", maxsplit=1)[1].split(
        "## review_t3", maxsplit=1
    )[0]
    assert "| bravo-model | 1512.0 | 1 | 1/0/0 |" in arena_section
    assert "| alpha-model | 1512.0 | 1 | 1/0/0 |" in discovery_section
    assert "| unused-model |" not in leaderboard


def test_write_reports_match_history_uses_configured_k_factor(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    rounds_dir = state_dir / "rounds"
    rounds_dir.mkdir(parents=True)
    round_id = "phase_review-round-0"
    record = _record(
        recorded_at="2026-04-12T13:00:00Z",
        round_id=round_id,
        task_class="phase_review",
        alpha="gpt-5.5-medium",
        beta="gpt-5.3-codex-medium",
    )
    record["repo_name"] = "repo-k"
    record["rating_pool_id"] = "pool|v1\nnext <!-- & -->"
    (rounds_dir / f"{round_id}.json").write_text(
        json.dumps({"review_cwd": "C:/Users/alice/.codex/worktrees/repo-k"}),
        encoding="utf-8",
    )
    (state_dir / "runs.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    roster = {
        "settings": {"elo_k_factor": 10},
        "variants": [
            _variant("gpt-5.5-medium", task_classes=["phase_review"]),
            _variant("gpt-5.3-codex-medium", task_classes=["phase_review"]),
        ],
    }
    operational_state = {
        "generated_at": "2026-04-12T00:00:00Z",
        "task_classes": {
            "phase_review": {
                "mode": "champion",
                "champion_variant_id": "gpt-5.5-medium",
                "champion_variant_ids": ["gpt-5.5-medium"],
                "probation_variant_ids": [],
                "stable_variant_ids": [],
                "cooldowns": {},
            },
            "pr_review": {
                "mode": "scramble",
                "champion_variant_id": None,
                "champion_variant_ids": [],
                "probation_variant_ids": [],
                "stable_variant_ids": [],
                "cooldowns": {},
            },
        },
    }

    summary = aggregate_records(
        roster=roster, records=[record], operational_state=operational_state
    )
    write_reports(state_dir, summary)

    leaderboard = (state_dir / "leaderboard.md").read_text(encoding="utf-8")
    escaped_pool = "pool\\|v1 next &lt;!-- &amp; --&gt;"
    assert f"- Rating pool: {escaped_pool}" in leaderboard
    assert f"| review_t1 | repo-k | {escaped_pool} |" in leaderboard
    assert "gpt-5.5-medium 1500.0 -> 1505.0 (+5.0)" in leaderboard
    assert "gpt-5.3-codex-medium 1500.0 -> 1495.0 (-5.0)" in leaderboard
