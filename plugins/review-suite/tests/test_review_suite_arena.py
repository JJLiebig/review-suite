from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from argparse import Namespace

from review_suite_arena import (
    _blocking_round_error,
    _completed_round_payload,
    _has_direct_grade_inputs,
    _normalize_arena_task_class,
    _print_findings,
    _record_grade_result,
    build_parser,
    cmd_costs,
    cmd_grade,
    cmd_dismiss_round,
    cmd_prune_state,
    cmd_resume_round,
    cmd_show_last,
    cmd_show_round,
    resume_orchestrator_review_step,
    run_benchmarked_round,
)
from review_costs import ReviewCostRow, update_review_cost_row_cache
from review_state_prune import prune_review_state
from review_suite_local import public_round_result, write_round


BASIS = "valid_findings_vs_none"


@pytest.fixture(autouse=True)
def _stub_clean_git_worktree(monkeypatch) -> None:
    monkeypatch.setattr(
        "review_suite_arena.ensure_clean_git_worktree", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "review_suite_local.ensure_clean_git_worktree", lambda *args, **kwargs: None
    )


def test_grade_records_garbage_flags_and_penalizes_placements(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    roster_path = tmp_path / "roster.json"
    roster_path.write_text(
        json.dumps(
            {
                "settings": {"promotion_min_samples": 20},
                "variants": [
                    {"id": variant, "state": "active", "task_classes": ["phase_review"]}
                    for variant in ("model-a", "model-b", "model-c")
                ],
            }
        ),
        encoding="utf-8",
    )
    write_round(
        state_dir,
        {
            "round_id": "garbage-round",
            "repo_name": "test-repo",
            "task_class": "phase_review",
            "selection_mode": "scramble",
            "reporting_pool": True,
            "status": "completed",
            "runs": [
                {
                    "slot": slot,
                    "variant_id": variant,
                    "review_status": "completed",
                    "grade_blocked": False,
                    "usage": {},
                    "cost_usd": 0,
                }
                for slot, variant in zip(
                    ("alpha", "bravo", "charlie"), ("model-a", "model-b", "model-c")
                )
            ],
        },
    )
    args = build_parser().parse_args(
        [
            "grade",
            "--round-id",
            "garbage-round",
            "--task-id",
            "task",
            "--rating-pool-id",
            "test-pool",
            "--rank",
            "alpha,bravo,charlie",
            "--basis",
            "tie_both_useful",
            "--low-quality",
            "alpha",
            "--low-quality",
            "model-c",
            "--roster",
            str(roster_path),
            "--rubric",
            str(SCRIPT_DIR.parent / "references" / "rubric_v2.json"),
            "--state-dir",
            str(state_dir),
            "--caller-id",
            "test-caller",
        ]
    )
    assert cmd_grade(args) == 0
    record = json.loads((state_dir / "runs.jsonl").read_text(encoding="utf-8"))
    assert record["placement_v1"]["groups"] == [["model-b"], ["model-a", "model-c"]]
    assert [run["low_quality"] for run in record["runs"]] == [True, False, True]
    summary = json.loads((state_dir / "summary.json").read_text(encoding="utf-8"))
    rows = {
        row["variant_id"]: row
        for row in summary["task_classes"]["phase_review"]["leaderboard"]
    }
    assert rows["model-b"]["elo"] > rows["model-a"]["elo"]
    assert rows["model-a"]["low_quality_loss_rate"] == 100
    assert rows["model-b"]["low_quality_loss_rate"] == 0


def test_print_findings_prints_final_reviewer_output(capsys) -> None:
    _print_findings(
        {
            "round_id": "round-123",
            "live_completion_statuses": {"Alpha": "completed"},
            "runs": [
                {
                    "slot": "Alpha",
                    "review_status": "completed",
                    "status_summary": "Medium - thing",
                    "grade_blocked": False,
                    "reviewer_output": "full body",
                },
                {
                    "slot": "Bravo",
                    "review_status": "completed",
                    "status_summary": "No findings.",
                    "grade_blocked": False,
                    "reviewer_output": "Bravo body",
                },
            ],
        }
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Output:" in captured.out
    assert "Alpha:" in captured.out
    assert "full body" in captured.out
    assert "Bravo:" in captured.out
    assert "Bravo body" in captured.out


def test_cmd_dismiss_round_removes_reviewer_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifacts: list[Path] = []
    run = {"variant_id": "reviewer"}
    for name in ("stdout", "stderr", "final_message"):
        artifact = tmp_path / name
        artifact.write_text("captured", encoding="utf-8")
        artifacts.append(artifact)
        run[f"{name}_path"] = str(artifact)
    write_round(
        tmp_path,
        {"round_id": "interrupted", "status": "running", "runs": [run]},
    )
    monkeypatch.setattr("review_suite_arena.emit_toon", lambda payload: None)
    args = Namespace(
        state_dir=str(tmp_path),
        round_id="interrupted",
        caller_id=None,
        reason="manual_dismiss",
    )
    assert cmd_dismiss_round(args) == 0
    assert not any(path.exists() for path in artifacts)


def test_public_round_result_does_not_repeat_streamed_output() -> None:
    result = public_round_result(
        {
            "round_id": "round-123",
            "task_class": "phase_review",
            "status": "completed",
            "runs": [
                {
                    "slot": "alpha",
                    "review_status": "completed",
                    "status_summary": "Medium - thing",
                    "grade_blocked": False,
                    "reviewer_output": "already streamed body",
                    "reviewer_output_ref": "ref://alpha",
                }
            ],
        }
    )

    assert result["runs"][0]["slot"] == "alpha"
    assert "output" not in result["runs"][0]


def test_blocking_round_error_uses_compact_action_for_completed_round(
    tmp_path: Path,
) -> None:
    error = _blocking_round_error(
        payload={
            "round_id": "round-123",
            "status": "completed",
            "runs": [{"review_status": "completed", "grade_blocked": False}],
        },
        action="review_t1",
        state_dir=tmp_path / "state",
    )

    assert str(error) == "pending round blocks review_t1: round-123"
    assert "grade_command" not in str(error)
    assert "dismiss_command" not in str(error)
    assert set(error.action_payload) == {"cmd", "dismiss_cmd"}
    assert "--rating-pool-id RATING_POOL_ID" in str(error.action_payload["cmd"])
    assert "--rank" in str(error.action_payload["cmd"])
    assert "--round-id round-123" in str(error.action_payload["cmd"])
    assert f"--state-dir {tmp_path / 'state'}" in str(error.action_payload["cmd"])
    assert "dismiss-round" in str(error.action_payload["dismiss_cmd"])


def test_blocking_round_error_uses_compact_action_for_running_round() -> None:
    error = _blocking_round_error(
        payload={
            "round_id": "round-456",
            "status": "running",
            "runs": [{"review_status": "running"}],
        },
        action="review_t3",
    )

    assert str(error) == "pending round blocks review_t3: round-456 (running)"
    assert set(error.action_payload) == {"dismiss_cmd"}
    assert "cmd" not in error.action_payload
    assert "dismiss-round" in str(error.action_payload["dismiss_cmd"])


def test_blocking_round_error_uses_configured_rating_pool(tmp_path: Path) -> None:
    error = _blocking_round_error(
        payload={
            "round_id": "round-123",
            "status": "completed",
            "rating_pool_id": "discovery-phase-gpt-5.6-v1",
            "runs": [{"review_status": "completed", "grade_blocked": False}],
        },
        action="review_t1",
        state_dir=tmp_path,
    )

    assert "--rating-pool-id discovery-phase-gpt-5.6-v1" in str(
        error.action_payload["cmd"]
    )


def test_print_findings_uses_recovered_full_body(capsys) -> None:
    _print_findings(
        {
            "round_id": "round-456",
            "live_completion_statuses": {"Alpha": "other_status"},
            "runs": [
                {
                    "slot": "Alpha",
                    "review_status": "completed",
                    "status_summary": "placeholder",
                    "grade_blocked": False,
                    "reviewer_output": "Recovered full body",
                }
            ],
        }
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Alpha:" in captured.out
    assert "placeholder" not in captured.out
    assert "Recovered full body" in captured.out


def test_print_findings_prints_blocked_runs(capsys) -> None:
    _print_findings(
        {
            "runs": [
                {
                    "slot": "Alpha",
                    "review_status": "interrupted_capacity",
                    "status_summary": "capacity",
                    "grade_blocked": True,
                    "grade_block_reason": "selected_model_at_capacity",
                }
            ],
        }
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Alpha [interrupted_capacity]:" in captured.out
    assert "capacity" in captured.out


def test_cmd_show_round_prints_stored_reviewer_outputs(tmp_path: Path, capsys) -> None:
    write_round(
        tmp_path,
        {
            "round_id": "round-1",
            "task_class": "phase_review",
            "status": "completed",
            "runs": [
                {
                    "slot": "alpha",
                    "variant_id": "alpha-model",
                    "review_status": "completed",
                    "reviewer_output": "Alpha finding",
                },
                {
                    "slot": "bravo",
                    "variant_id": "bravo-model",
                    "review_status": "completed",
                    "status_summary": "No findings.",
                },
            ],
        },
    )

    result = cmd_show_round(
        Namespace(round_id="round-1", state_dir=str(tmp_path), json=False)
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "round_id: round-1" in captured.out
    assert "alpha:" in captured.out
    assert "Alpha finding" in captured.out
    assert "bravo:" in captured.out
    assert "No findings." in captured.out


def test_cmd_costs_uses_cache_and_ignores_stale_legacy_lock(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = tmp_path / "state"
    output = tmp_path / "costs.md"
    row = ReviewCostRow(
        repo="repo",
        folder="repo",
        branch="feat/costs",
        pr_number="42",
        worker_model="gpt-5.5 medium",
        implementation_tokens=1000,
        implementation_cost_usd=0.004,
        caller_threads=(),
        latest_review="2026-04-27T10:00:00Z",
        lane_sessions={
            "review_t1": 2,
            "review_t3": 0,
            "review_followup": 0,
        },
        review_seconds=123.0,
        tokens=456,
        cost_usd=0.012345,
        review_cwd_normalized=str(repo),
    )
    update_review_cost_row_cache(state_dir=state_dir, rows=[row])
    (state_dir / ".locks" / "cost-reports.lock").mkdir(parents=True)

    monkeypatch.setattr("review_suite_arena.resolve_repo_root", lambda cd: repo)
    monkeypatch.setattr(
        "review_costs._cached_metadata_for_cwd",
        lambda value: {
            "repo": "repo",
            "folder": "repo",
            "branch": "feat/costs",
            "pr_number": "42",
        },
    )

    result = cmd_costs(
        Namespace(
            cd=str(repo),
            all=False,
            state_dir=str(state_dir),
            output=str(output),
            json=False,
        )
    )

    captured = capsys.readouterr()
    assert result == 0
    assert output.exists()
    assert "# repo" in output.read_text(encoding="utf-8")
    assert "rows: 1" in captured.out
    assert "total_tokens: 456" in captured.out
    assert "total_cost_usd: 0.012345" in captured.out
    assert "total_implementation_tokens: 1000" in captured.out
    assert "total_implementation_cost_usd: 0.004" in captured.out
    assert (state_dir / ".locks" / "review-cost-accounting.lock").is_file()


def test_cmd_costs_renders_all_cached_rows_for_scoped_report(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = tmp_path / "state"
    output = tmp_path / "costs.md"
    cached_row = ReviewCostRow(
        repo="cached-repo",
        folder="cached-worktree",
        branch="feat/cached",
        pr_number="7",
        worker_model="gpt-5.5 medium",
        implementation_tokens=2000,
        implementation_cost_usd=0.02,
        caller_threads=(),
        latest_review="2026-04-26T10:00:00Z",
        lane_sessions={
            "review_t1": 1,
            "review_t3": 0,
            "review_followup": 0,
        },
        review_seconds=60,
        tokens=300,
        cost_usd=0.003,
    )
    current_row = ReviewCostRow(
        repo="repo",
        folder="repo",
        branch="feat/costs",
        pr_number="42",
        worker_model="gpt-5.5 medium",
        implementation_tokens=1000,
        implementation_cost_usd=0.004,
        caller_threads=(),
        latest_review="2026-04-27T10:00:00Z",
        lane_sessions={
            "review_t1": 2,
            "review_t3": 0,
            "review_followup": 0,
        },
        review_seconds=123.0,
        tokens=456,
        cost_usd=0.012345,
        review_cwd_normalized=str(repo),
    )
    update_review_cost_row_cache(state_dir=state_dir, rows=[cached_row, current_row])
    monkeypatch.setattr("review_suite_arena.resolve_repo_root", lambda cd: repo)
    monkeypatch.setattr(
        "review_costs._cached_metadata_for_cwd",
        lambda value: {
            "repo": "repo",
            "folder": "repo",
            "branch": "feat/costs",
            "pr_number": "42",
        },
    )

    result = cmd_costs(
        Namespace(
            cd=str(repo),
            all=False,
            state_dir=str(state_dir),
            output=str(output),
            json=False,
        )
    )

    captured = capsys.readouterr()
    markdown = output.read_text(encoding="utf-8")
    assert result == 0
    assert "# cached-repo" in markdown
    assert "# repo" in markdown
    assert "rows: 1" in captured.out
    assert "report_rows: 2" in captured.out
    assert "total_tokens: 456" in captured.out
    assert "report_total_tokens: 756" in captured.out


def test_cmd_costs_scoped_missing_cache_points_to_full_rebuild(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr("review_suite_arena.resolve_repo_root", lambda cd: repo)
    monkeypatch.setattr(
        "review_costs._cached_metadata_for_cwd",
        lambda value: {
            "repo": "repo",
            "folder": "repo",
            "branch": "feat/costs",
            "pr_number": "-",
        },
    )

    with pytest.raises(ValueError, match=r"run costs --all to rebuild it"):
        cmd_costs(
            Namespace(
                cd=str(repo),
                all=False,
                state_dir=str(tmp_path / "state"),
                output=None,
                json=False,
            )
        )


def test_cmd_costs_scoped_stale_implementation_points_to_full_rebuild(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = tmp_path / "state"
    row = ReviewCostRow(
        repo="repo",
        folder="repo",
        branch="feat/costs",
        pr_number="-",
        worker_model="-",
        implementation_tokens=0,
        implementation_cost_usd=0.0,
        caller_threads=(),
        latest_review="2026-04-27T10:00:00Z",
        lane_sessions={
            "review_t1": 1,
            "review_t3": 0,
            "review_followup": 0,
        },
        review_seconds=1,
        tokens=100,
        cost_usd=0.01,
        review_cwd_normalized=str(repo),
        implementation_costs_refreshed=False,
    )
    update_review_cost_row_cache(state_dir=state_dir, rows=[row])
    monkeypatch.setattr("review_suite_arena.resolve_repo_root", lambda cd: repo)
    monkeypatch.setattr(
        "review_costs._cached_metadata_for_cwd",
        lambda value: {
            "repo": "repo",
            "folder": "repo",
            "branch": "feat/costs",
            "pr_number": "-",
        },
    )

    with pytest.raises(ValueError, match=r"implementation costs are stale.*--all"):
        cmd_costs(
            Namespace(
                cd=str(repo),
                all=False,
                state_dir=str(state_dir),
                output=None,
                json=False,
            )
        )


def test_cmd_costs_all_ignores_cd_filter(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = tmp_path / "state"
    seen = {}

    def fake_resolve_repo_root(cd):
        if cd == str(repo):
            raise AssertionError("--all should not resolve --cd")
        return repo

    monkeypatch.setattr("review_suite_arena.resolve_repo_root", fake_resolve_repo_root)
    monkeypatch.setattr(
        "review_suite_arena.write_review_cost_report",
        lambda **kwargs: tmp_path / "costs.md",
    )
    monkeypatch.setattr("review_suite_arena.emit_toon", lambda payload: None)

    def fake_collect_review_cost_rows(**kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(
        "review_suite_arena.collect_review_cost_rows", fake_collect_review_cost_rows
    )

    result = cmd_costs(
        Namespace(
            cd=str(repo),
            all=True,
            state_dir=str(state_dir),
            output=None,
            json=False,
            codex_home=None,
        )
    )

    assert result == 0
    assert seen["review_cwd"] is None
    assert seen["include_all"] is True


def test_cmd_show_last_prints_latest_outputs_per_local_lane(
    tmp_path: Path, capsys
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    write_round(
        tmp_path,
        {
            "round_id": "older-t1",
            "task_class": "phase_review",
            "status": "completed",
            "sampled_at": "2026-04-25T09:00:00Z",
            "review_cwd": str(repo),
            "review_cwd_normalized": str(repo),
            "runs": [
                {
                    "slot": "alpha",
                    "review_status": "completed",
                    "reviewer_output": "Old T1",
                }
            ],
        },
    )
    write_round(
        tmp_path,
        {
            "round_id": "latest-t1",
            "task_class": "phase_review",
            "status": "completed",
            "sampled_at": "2026-04-25T10:00:00Z",
            "review_cwd": str(repo),
            "review_cwd_normalized": str(repo),
            "runs": [
                {
                    "slot": "alpha",
                    "review_status": "completed",
                    "reviewer_output": "Latest T1",
                }
            ],
        },
    )
    result = cmd_show_last(
        Namespace(cd=None, task=None, state_dir=str(tmp_path), json=False)
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "round_id: latest-t1" in captured.out
    assert "Latest T1" in captured.out
    assert "Old T1" not in captured.out


def test_cmd_show_round_finds_orchestrator_review_rounds(
    tmp_path: Path, capsys
) -> None:
    write_round(
        tmp_path / "orchestrator" / "review-rounds",
        {
            "round_id": "orchestrator-round",
            "task_class": "phase_review",
            "status": "completed",
            "runs": [
                {
                    "slot": "alpha",
                    "review_status": "completed",
                    "reviewer_output": "Orchestrator finding",
                }
            ],
        },
    )

    result = cmd_show_round(
        Namespace(round_id="orchestrator-round", state_dir=str(tmp_path), json=False)
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "round_id: orchestrator-round" in captured.out
    assert "Orchestrator finding" in captured.out


def test_cmd_show_last_filters_orchestrator_review_rounds_by_repo(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr("review_suite_arena.resolve_repo_root", lambda cd: repo)
    write_round(
        tmp_path / "orchestrator" / "review-rounds",
        {
            "round_id": "orchestrator-latest",
            "task_class": "phase_review",
            "status": "completed",
            "sampled_at": "2026-05-27T10:00:00Z",
            "review_cwd": str(repo),
            "review_cwd_normalized": str(repo),
            "runs": [
                {
                    "slot": "alpha",
                    "review_status": "completed",
                    "reviewer_output": "Nested latest",
                }
            ],
        },
    )

    result = cmd_show_last(
        Namespace(cd=str(repo), task="review_t1", state_dir=str(tmp_path), json=False)
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "round_id: orchestrator-latest" in captured.out
    assert "Nested latest" in captured.out


def _touch_old(path: Path) -> None:
    timestamp = datetime(2026, 6, 1, tzinfo=timezone.utc).timestamp()
    os.utime(path, (timestamp, timestamp))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_prune_state_removes_only_final_unreferenced_detail(tmp_path: Path) -> None:
    cycle_dir = tmp_path / "orchestrator" / "cycles"
    old_cycle = cycle_dir / "old-cycle.json"
    active_cycle = cycle_dir / "active-cycle.json"
    _write_json(
        old_cycle,
        {
            "stage": "review-green",
            "pending_action": None,
            "rounds": [{"round_id": "dismissed-old"}],
        },
    )
    _write_json(
        active_cycle,
        {
            "stage": "running",
            "pending_action": {"kind": "collect-review-step"},
            "rounds": [{"round_id": "active-dismissed"}],
        },
    )
    _touch_old(old_cycle)
    _touch_old(active_cycle)
    _write_json(
        tmp_path / "orchestrator" / "index.json",
        {
            "schema_version": 1,
            "ids": {"rvw_old": "old-cycle", "rvw_active": "active-cycle"},
            "cycle_keys": {"old-cycle": "rvw_old", "active-cycle": "rvw_active"},
        },
    )

    orchestrator_rounds = tmp_path / "orchestrator" / "review-rounds" / "rounds"
    for round_id, status in (
        ("dismissed-old", "dismissed"),
        ("active-dismissed", "dismissed"),
        ("ungraded-old", "completed"),
        ("running-old", "running"),
    ):
        _write_json(
            orchestrator_rounds / f"{round_id}.json",
            {
                "round_id": round_id,
                "status": status,
                "dismissed_at": "2026-06-01T00:00:00Z",
                "review_completed_at": "2026-06-01T00:00:00Z",
                "runs": [{"slot": "alpha", "stdout": "detail"}],
            },
        )
    _write_json(
        tmp_path / "rounds" / "graded-old.json",
        {
            "round_id": "graded-old",
            "status": "completed",
            "graded_at": "2026-06-01T00:00:00Z",
            "runs": [{"slot": "alpha", "reviewer_output": "done"}],
        },
    )

    canonical = {
        "runs.jsonl": '{"round_id":"graded-old"}\n',
        "wrapper_sessions.jsonl": '{"session_id":"review-wrapper"}\n',
        "summary.json": '{"stable":true}\n',
        "review_cost_rows/row.json": '{"repo":"kept"}\n',
        "workflow/repo/old.json": '{"last_reviewed_head":"abc123"}\n',
    }
    for name, content in canonical.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        _touch_old(path)

    result = prune_review_state(
        tmp_path,
        apply=True,
        now=datetime(2026, 7, 9, tzinfo=timezone.utc),
    )
    second = prune_review_state(
        tmp_path,
        apply=True,
        now=datetime(2026, 7, 9, tzinfo=timezone.utc),
    )

    assert result["orchestrator_cycles"]["deleted"] == 1
    assert result["rounds"]["deleted"] == 2
    assert not old_cycle.exists()
    assert active_cycle.exists()
    assert not (orchestrator_rounds / "dismissed-old.json").exists()
    assert (orchestrator_rounds / "active-dismissed.json").exists()
    assert (orchestrator_rounds / "ungraded-old.json").exists()
    assert (orchestrator_rounds / "running-old.json").exists()
    assert not (tmp_path / "rounds" / "graded-old.json").exists()
    index = json.loads((tmp_path / "orchestrator" / "index.json").read_text())
    assert index["ids"] == {"rvw_active": "active-cycle"}
    assert all(
        (tmp_path / name).read_text(encoding="utf-8") == content
        for name, content in canonical.items()
    )
    assert second["saved_b"] == 0


def test_prune_state_dry_run_does_not_write(tmp_path: Path) -> None:
    cycle = tmp_path / "orchestrator" / "cycles" / "old-cycle.json"
    _write_json(cycle, {"stage": "review-green", "pending_action": None})
    _touch_old(cycle)

    result = prune_review_state(
        tmp_path,
        now=datetime(2026, 7, 9, tzinfo=timezone.utc),
    )

    assert result["applied"] is False
    assert result["orchestrator_cycles"]["deleted"] == 1
    assert cycle.exists()
    assert not (tmp_path / ".locks").exists()


def test_prune_state_skips_locked_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cycle = tmp_path / "orchestrator" / "cycles" / "locked-cycle.json"
    _write_json(
        cycle,
        {
            "stage": "review-green",
            "pending_action": None,
            "rounds": [{"round_id": "protected"}],
        },
    )
    _touch_old(cycle)
    round_dir = tmp_path / "rounds"
    for round_id in ("protected", "locked", "free"):
        _write_json(
            round_dir / f"{round_id}.json",
            {
                "round_id": round_id,
                "status": "dismissed",
                "dismissed_at": "2026-06-01T00:00:00Z",
            },
        )
    unlink = Path.unlink

    def fail_locked(path: Path, *args, **kwargs) -> None:
        if path.name in {"locked-cycle.json", "locked.json"}:
            raise PermissionError("in use")
        unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_locked)

    result = prune_review_state(
        tmp_path,
        apply=True,
        now=datetime(2026, 7, 9, tzinfo=timezone.utc),
    )

    assert result["orchestrator_cycles"]["deleted"] == 0
    assert result["rounds"]["deleted"] == 1
    assert cycle.exists()
    assert (round_dir / "protected.json").exists()
    assert (round_dir / "locked.json").exists()
    assert not (round_dir / "free.json").exists()


def test_cmd_prune_state_reports_dry_run(tmp_path: Path, capsys) -> None:
    state_dir = tmp_path / "missing-state"
    result = cmd_prune_state(
        Namespace(state_dir=str(state_dir), older_than_days=14, apply=False)
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "status: ok" in captured.out
    assert "applied: false" in captured.out
    assert not state_dir.exists()


def test_print_findings_prints_blocked_body_after_completion_status(capsys) -> None:
    _print_findings(
        {
            "live_completion_statuses": {"Alpha": "selected_model_at_capacity"},
            "runs": [
                {
                    "slot": "Alpha",
                    "review_status": "interrupted_capacity",
                    "status_summary": "capacity",
                    "grade_blocked": True,
                    "grade_block_reason": "selected_model_at_capacity",
                    "reviewer_output": "Recovered full body",
                }
            ],
        }
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Alpha [interrupted_capacity]:" in captured.out
    assert "Recovered full body" in captured.out


def test_completed_round_payload_uses_reroll_command_key() -> None:
    payload = _completed_round_payload(
        round_result={"blocked": True, "runs": []},
        reroll_rows=[{"slot": "alpha", "command": "reroll-cmd"}],
    )

    assert payload["actions"] == [
        {"kind": "reroll", "slot": "alpha", "cmd": "reroll-cmd"}
    ]
    assert payload["blocked"] is True


def test_completed_round_payload_success_only_emits_grade_command() -> None:
    payload = _completed_round_payload(
        round_result={
            "blocked": False,
            "runs": [
                {
                    "slot": "alpha",
                    "status": "completed",
                    "summary": "No findings.",
                    "ref": "rollout://secret/model",
                }
            ],
        },
        grade_command="grade-cmd",
    )

    assert payload["Action"]["cmd"] == "grade-cmd"
    assert set(payload["Action"]) == {"cmd"}
    assert "runs" not in payload


def test_print_next_steps_uses_every_configured_rank(tmp_path, capsys) -> None:
    __import__("review_suite_arena")._print_next_steps(
        round_id="round-1",
        task_id="task-1",
        round_result={"runs": [{"slot": slot} for slot in "abcd"]},
        review_cwd=tmp_path,
        base="main",
        roster_path=tmp_path / "roster.json",
        rubric_path=tmp_path / "rubric.json",
        state_dir=tmp_path / "state",
        sqlite_path=tmp_path / "state.sqlite",
        allow_unsafe_windows_wsl_fallback=False,
    )

    assert capsys.readouterr().err.count("--rank") == 4


def test_completed_round_payload_omits_inspect_when_round_id_is_known() -> None:
    payload = _completed_round_payload(
        round_result={
            "round_id": "round-123",
            "blocked": False,
            "runs": [],
        },
        grade_command="grade-cmd",
    )

    assert payload["Action"]["cmd"] == "grade-cmd"
    assert "inspect" not in payload["Action"]


def test_completed_round_payload_manual_omits_run_rows_after_output() -> None:
    payload = _completed_round_payload(
        round_result={
            "round_id": "round-123",
            "blocked": False,
            "runs": [{"slot": "alpha"}],
        },
        manual=True,
    )

    assert payload == {"Action": {"note": "manual review complete"}}


def test_has_direct_grade_inputs_requires_complete_tuple() -> None:
    assert _has_direct_grade_inputs(
        task_id="task-123",
        rating_pool_id="pool-v1",
        rank_groups=["alpha", "bravo"],
        basis=BASIS,
    )
    assert not _has_direct_grade_inputs(
        task_id="task-123",
        rating_pool_id="pool-v1",
        rank_groups=["alpha", "bravo"],
        basis=None,
    )


def test_record_grade_refreshes_arena_state_without_cost_work(
    monkeypatch, tmp_path: Path
) -> None:
    events: list[tuple[str, object]] = []

    class Lock:
        def __init__(self, name: str) -> None:
            self.name = name

        def __enter__(self):
            events.append((self.name, {}))

        def __exit__(self, *args):
            return None

    round_payload = {
        "round_id": "round-1",
        "task_class": "phase_review",
        "review_cwd": str(tmp_path),
    }
    monkeypatch.setattr(
        "review_suite_arena.load_round", lambda state_dir, round_id: round_payload
    )
    monkeypatch.setattr(
        "review_suite_arena.build_record_from_grade",
        lambda **kwargs: {"recorded_at": "2026-07-10T17:00:00Z"},
    )
    monkeypatch.setattr(
        "review_suite_arena.append_record_if_new", lambda state_dir, record: True
    )
    monkeypatch.setattr(
        "review_suite_arena.write_round",
        lambda state_dir, payload: events.append(("write", dict(payload))),
    )
    monkeypatch.setattr(
        "review_suite_arena.state_lock", lambda state_dir, name: Lock(name)
    )
    monkeypatch.setattr(
        "review_suite_arena._refresh_state_and_reports_locked",
        lambda **kwargs: events.append(("state", kwargs)),
    )
    result = _record_grade_result(
        roster={},
        rubric={},
        state_dir=tmp_path / "state",
        round_id="round-1",
        task_id="task-1",
        rating_pool_id="pool-1",
        rank_groups=["alpha", "bravo"],
        basis=BASIS,
        note=None,
        caller_id="caller-1",
        caller_id_source="explicit",
    )

    assert result == {
        "status": "ok",
        "recorded": True,
        "duplicate": False,
        "round_id": "round-1",
    }
    assert [event[0] for event in events] == [
        "write",
        "runs",
        "operational-state",
        "reports",
        "state",
        "write",
    ]
    assert events[0][1]["grade_refresh_pending"] is True
    assert events[5][1]["grade_recorded"] is True
    assert "grade_refresh_pending" not in events[5][1]


def test_run_benchmarked_round_rejects_partial_grade_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="direct grading requires"):
        run_benchmarked_round(
            task_class="pr_review",
            review_cwd=tmp_path,
            roster_path=tmp_path / "roster.json",
            rubric_path=tmp_path / "rubric.json",
            state_dir=tmp_path / "state",
            sqlite_path=tmp_path / "state.sqlite",
            seed=None,
            progress_interval_seconds=30,
            allow_unsafe_windows_wsl_fallback=False,
            review_scope={"base": "main"},
            prompt="",
            caller_id="caller-1",
            caller_id_source="explicit",
            ignore_pending_grades=False,
            task_id="task-123",
            rating_pool_id="pool-v1",
            rank_groups=None,
            basis=BASIS,
            note=None,
        )


def test_run_benchmarked_round_direct_grade_uses_latest_pending_round(
    monkeypatch, tmp_path
) -> None:
    captured: dict[str, object] = {}
    emitted: list[dict[str, object]] = []

    monkeypatch.setattr(
        "review_suite_arena._ensure_no_pending_grades",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("should not check pending before direct grade")
        ),
    )
    monkeypatch.setattr("review_suite_arena.load_roster", lambda path: {"variants": []})
    monkeypatch.setattr(
        "review_suite_arena.load_rubric", lambda path: {"categories": []}
    )
    monkeypatch.setattr(
        "review_suite_arena.find_pending_rounds_for_caller",
        lambda **kwargs: [{"round_id": "latest-round", "task_class": "pr_review"}],
    )

    def fake_record_grade_result(**kwargs):
        captured.update(kwargs)
        return {"status": "graded", "round_id": kwargs["round_id"]}

    monkeypatch.setattr(
        "review_suite_arena._record_grade_result", fake_record_grade_result
    )
    monkeypatch.setattr(
        "review_suite_arena.emit_toon", lambda payload: emitted.append(payload)
    )

    result = run_benchmarked_round(
        task_class="pr_review",
        review_cwd=tmp_path,
        roster_path=tmp_path / "roster.json",
        rubric_path=tmp_path / "rubric.json",
        state_dir=tmp_path / "state",
        sqlite_path=tmp_path / "state.sqlite",
        seed=None,
        progress_interval_seconds=30,
        allow_unsafe_windows_wsl_fallback=False,
        review_scope={"base": "main"},
        prompt="",
        caller_id="caller-1",
        caller_id_source="explicit",
        ignore_pending_grades=False,
        task_id="task-123",
        rating_pool_id="pool-v1",
        rank_groups=["alpha", "bravo"],
        basis=BASIS,
        note="shared",
    )

    assert result == 0
    assert captured["round_id"] == "latest-round"
    assert captured["task_id"] == "task-123"
    assert captured["rating_pool_id"] == "pool-v1"
    assert captured["rank_groups"] == ["alpha", "bravo"]
    assert captured["basis"] == BASIS
    assert emitted == [
        {"status": "graded", "round_id": "latest-round", "task": "review_t3"}
    ]


def test_run_benchmarked_round_emits_round_banner_and_compact_payload(
    monkeypatch, tmp_path, capsys
) -> None:
    emitted: list[dict[str, object]] = []
    anchor_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "review_suite_arena._ensure_no_pending_grades", lambda **kwargs: None
    )
    monkeypatch.setattr("review_suite_arena.load_roster", lambda path: {"variants": []})
    monkeypatch.setattr(
        "review_suite_arena.load_rubric", lambda path: {"categories": []}
    )
    monkeypatch.setattr("review_suite_arena.read_jsonl", lambda path: [])
    monkeypatch.setattr("review_suite_arena.load_operational_state", lambda path: {})
    monkeypatch.setattr(
        "review_suite_arena.select_pair",
        lambda **kwargs: {"round_id": "round-1", "task_class": "phase_review"},
    )
    monkeypatch.setattr("review_suite_arena.write_round", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "review_suite_arena.run_round",
        lambda **kwargs: {
            "round_id": "round-1",
            "task_class": "phase_review",
            "runs": [],
        },
    )
    monkeypatch.setattr(
        "review_suite_arena.public_round_result",
        lambda *args, **kwargs: {"blocked": False, "runs": []},
    )
    monkeypatch.setattr(
        "review_suite_arena._print_findings", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "review_suite_arena._current_branch_name", lambda review_cwd: "branch-1"
    )
    monkeypatch.setattr("review_suite_arena._print_next_steps", lambda **kwargs: None)
    monkeypatch.setattr(
        "review_suite_arena.record_review_anchor",
        lambda **kwargs: anchor_calls.append(kwargs) or {},
    )
    monkeypatch.setattr(
        "review_suite_arena.emit_toon", lambda payload: emitted.append(payload)
    )

    result = run_benchmarked_round(
        task_class="phase_review",
        review_cwd=tmp_path,
        roster_path=tmp_path / "roster.json",
        rubric_path=tmp_path / "rubric.json",
        state_dir=tmp_path / "state",
        sqlite_path=tmp_path / "state.sqlite",
        seed=None,
        progress_interval_seconds=30,
        allow_unsafe_windows_wsl_fallback=False,
        review_scope={"base": "main"},
        prompt="",
        caller_id="caller-1",
        caller_id_source="explicit",
        ignore_pending_grades=False,
        task_id=None,
        rating_pool_id=None,
        rank_groups=None,
        basis=None,
        note=None,
        public_task_name="review_t1",
    )

    assert result == 0
    assert emitted[-1]["Action"]["cmd"]
    assert "task" not in emitted[-1]
    assert "round_id" not in emitted[-1]
    assert "runs" not in emitted[-1]
    assert "status" not in emitted[-1]
    assert anchor_calls[0]["lane"] == "review_t1"
    assert anchor_calls[0]["task_id"] == "branch-1"
    assert "[review-suite] round review_t1 round-1" in capsys.readouterr().err


def test_run_benchmarked_round_noninteractive_uses_toon_actions_without_stderr_next_steps(
    monkeypatch, tmp_path
) -> None:
    emitted: list[dict[str, object]] = []

    monkeypatch.setattr(
        "review_suite_arena._ensure_no_pending_grades", lambda **kwargs: None
    )
    monkeypatch.setattr("review_suite_arena.load_roster", lambda path: {"variants": []})
    monkeypatch.setattr(
        "review_suite_arena.load_rubric", lambda path: {"categories": []}
    )
    monkeypatch.setattr("review_suite_arena.read_jsonl", lambda path: [])
    monkeypatch.setattr("review_suite_arena.load_operational_state", lambda path: {})
    monkeypatch.setattr(
        "review_suite_arena.select_pair",
        lambda **kwargs: {"round_id": "round-1", "task_class": "phase_review"},
    )
    monkeypatch.setattr("review_suite_arena.write_round", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "review_suite_arena.guard_no_stage_step_down", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "review_suite_arena._validate_benchmarked_review_runtime", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "review_suite_arena.run_round",
        lambda **kwargs: {
            "round_id": "round-1",
            "task_class": "phase_review",
            "runs": [],
        },
    )
    monkeypatch.setattr(
        "review_suite_arena.public_round_result",
        lambda *args, **kwargs: {"blocked": False, "runs": []},
    )
    monkeypatch.setattr(
        "review_suite_arena._print_findings", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "review_suite_arena._current_branch_name", lambda review_cwd: "branch-1"
    )
    monkeypatch.setattr(
        "review_suite_arena.apply_review_cost_delta_best_effort",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "review_suite_arena._record_standalone_review_anchor_for_round",
        lambda **kwargs: True,
    )
    monkeypatch.setattr("review_suite_arena._output_isatty", lambda: False)
    monkeypatch.setattr(
        "review_suite_arena._print_next_steps",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("next-step prose should stay out of non-interactive output")
        ),
    )
    monkeypatch.setattr(
        "review_suite_arena.emit_toon", lambda payload: emitted.append(payload)
    )

    result = run_benchmarked_round(
        task_class="phase_review",
        review_cwd=tmp_path,
        roster_path=tmp_path / "roster.json",
        rubric_path=tmp_path / "rubric.json",
        state_dir=tmp_path / "state",
        sqlite_path=tmp_path / "state.sqlite",
        seed=None,
        progress_interval_seconds=30,
        allow_unsafe_windows_wsl_fallback=False,
        review_scope={"base": "main"},
        prompt="",
        caller_id="caller-1",
        caller_id_source="explicit",
        ignore_pending_grades=False,
        task_id=None,
        rating_pool_id=None,
        rank_groups=None,
        basis=None,
        note=None,
        public_task_name="review_t1",
    )

    assert result == 0
    assert set(emitted[-1]) == {"Action"}
    assert "--rating-pool-id RATING_POOL_ID" in emitted[-1]["Action"]["cmd"]
    assert emitted[-1]["Action"]["cmd"].count("--rank") == 2
    assert "--round-id" in emitted[-1]["Action"]["cmd"]
    assert "--task-id" in emitted[-1]["Action"]["cmd"]
    assert "--state-dir" in emitted[-1]["Action"]["cmd"]
    assert "--basis BASIS" in emitted[-1]["Action"]["cmd"]
    assert set(emitted[-1]["Action"]) == {"cmd"}


def test_run_benchmarked_round_warns_for_deep_review_without_model_names(
    monkeypatch, tmp_path, capsys
) -> None:
    emitted: list[dict[str, object]] = []

    monkeypatch.setattr(
        "review_suite_arena._ensure_no_pending_grades", lambda **kwargs: None
    )
    monkeypatch.setattr("review_suite_arena.load_roster", lambda path: {"variants": []})
    monkeypatch.setattr(
        "review_suite_arena.load_rubric", lambda path: {"categories": []}
    )
    monkeypatch.setattr("review_suite_arena.read_jsonl", lambda path: [])
    monkeypatch.setattr("review_suite_arena.load_operational_state", lambda path: {})
    monkeypatch.setattr(
        "review_suite_arena.select_pair",
        lambda **kwargs: {
            "round_id": "round-1",
            "task_class": "pr_review",
            "runs": [
                {
                    "slot": "alpha",
                    "variant_id": "hidden-alpha",
                    "reasoning_effort": "xhigh",
                },
                {
                    "slot": "bravo",
                    "variant_id": "hidden-bravo",
                    "reasoning_effort": "medium",
                },
            ],
        },
    )
    monkeypatch.setattr("review_suite_arena.write_round", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "review_suite_arena.run_round",
        lambda **kwargs: {"round_id": "round-1", "task_class": "pr_review", "runs": []},
    )
    monkeypatch.setattr(
        "review_suite_arena.public_round_result",
        lambda *args, **kwargs: {"blocked": False, "runs": []},
    )
    monkeypatch.setattr(
        "review_suite_arena._print_findings", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "review_suite_arena._current_branch_name", lambda review_cwd: "branch-1"
    )
    monkeypatch.setattr("review_suite_arena._output_isatty", lambda: False)
    monkeypatch.setattr("review_suite_arena.record_review_anchor", lambda **kwargs: {})
    monkeypatch.setattr(
        "review_suite_arena.emit_toon", lambda payload: emitted.append(payload)
    )

    result = run_benchmarked_round(
        task_class="pr_review",
        review_cwd=tmp_path,
        roster_path=tmp_path / "roster.json",
        rubric_path=tmp_path / "rubric.json",
        state_dir=tmp_path / "state",
        sqlite_path=tmp_path / "state.sqlite",
        seed=None,
        progress_interval_seconds=30,
        allow_unsafe_windows_wsl_fallback=False,
        review_scope={"base": "main"},
        prompt="",
        caller_id="caller-1",
        caller_id_source="explicit",
        ignore_pending_grades=False,
        task_id=None,
        rating_pool_id=None,
        rank_groups=None,
        basis=None,
        note=None,
        public_task_name="review_t3",
    )

    assert result == 0
    err = capsys.readouterr().err
    assert "reviews can take a long time" in err
    assert "hidden-alpha" not in err
    assert "hidden-bravo" not in err
    assert "xhigh" not in err


def test_run_benchmarked_round_dry_run_rejects_direct_grade_inputs(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="cannot be combined with direct grading"):
        run_benchmarked_round(
            task_class="phase_review",
            review_cwd=tmp_path,
            roster_path=tmp_path / "roster.json",
            rubric_path=tmp_path / "rubric.json",
            state_dir=tmp_path / "state",
            sqlite_path=tmp_path / "state.sqlite",
            seed=None,
            progress_interval_seconds=30,
            allow_unsafe_windows_wsl_fallback=False,
            review_scope={"base": "main"},
            prompt="",
            caller_id="caller-1",
            caller_id_source="explicit",
            ignore_pending_grades=False,
            task_id="task-1",
            rating_pool_id="pool-1",
            rank_groups=["alpha"],
            basis="valid_findings_vs_none",
            note=None,
            public_task_name="review_t1",
            dry_run=True,
        )


def test_run_benchmarked_round_dry_run_does_not_persist_or_launch(
    monkeypatch, tmp_path
) -> None:
    emitted: list[dict[str, object]] = []
    writes: list[dict[str, object]] = []
    cleanup_calls: list[object] = []
    roster = {
        "settings": {"selection_mode": "true_scramble"},
        "variants": [
            {
                "id": "alpha-low",
                "model": "gpt-5.5",
                "reasoning_effort": "low",
                "task_classes": ["phase_review"],
                "state": "active",
            },
            {
                "id": "bravo-low",
                "model": "gpt-5.6",
                "reasoning_effort": "low",
                "task_classes": ["phase_review"],
                "state": "active",
            },
        ],
    }
    monkeypatch.setattr("review_suite_arena.load_roster", lambda path: roster)
    monkeypatch.setattr("review_suite_arena.emit_toon", emitted.append)
    monkeypatch.setattr(
        "review_suite_arena.write_round",
        lambda state_dir, payload: writes.append(dict(payload)),
    )
    monkeypatch.setattr(
        "review_suite_arena.run_round",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("dry run must not launch reviewers")
        ),
    )
    monkeypatch.setattr(
        "review_suite_local.cleanup_stale_ungraded_rounds",
        lambda state_dir: cleanup_calls.append(state_dir),
    )

    exit_code = run_benchmarked_round(
        task_class="phase_review",
        review_cwd=tmp_path,
        roster_path=tmp_path / "roster.json",
        rubric_path=tmp_path / "rubric.json",
        state_dir=tmp_path / "state",
        sqlite_path=tmp_path / "state.sqlite",
        seed=1,
        progress_interval_seconds=30,
        allow_unsafe_windows_wsl_fallback=False,
        review_scope={"base": "main"},
        prompt="",
        caller_id="caller-1",
        caller_id_source="explicit",
        ignore_pending_grades=False,
        task_id=None,
        rating_pool_id=None,
        rank_groups=None,
        basis=None,
        note=None,
        public_task_name="review_t1",
        dry_run=True,
    )

    assert exit_code == 0
    assert writes == []
    assert cleanup_calls == []
    assert len(emitted) == 1
    payload = emitted[0]
    assert payload["status"] == "dry_run"
    assert {run["variant_id"] for run in payload["reviewers"]} == {
        "alpha-low",
        "bravo-low",
    }


def test_run_benchmarked_round_dirty_base_guard_does_not_persist_sampled_round(
    monkeypatch, tmp_path
) -> None:
    writes: list[dict[str, object]] = []

    monkeypatch.setattr(
        "review_suite_arena._ensure_no_pending_grades", lambda **kwargs: None
    )

    def fail_clean_check(*args, **kwargs):
        raise ValueError("review-suite requires a clean worktree")

    monkeypatch.setattr(
        "review_suite_arena.ensure_clean_git_worktree", fail_clean_check
    )
    monkeypatch.setattr(
        "review_suite_arena.select_pair",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("select_pair should not run")
        ),
    )
    monkeypatch.setattr(
        "review_suite_arena.write_round",
        lambda state_dir, payload: writes.append(dict(payload)),
    )

    with pytest.raises(ValueError, match="clean worktree"):
        run_benchmarked_round(
            task_class="phase_review",
            review_cwd=tmp_path,
            roster_path=tmp_path / "roster.json",
            rubric_path=tmp_path / "rubric.json",
            state_dir=tmp_path / "state",
            sqlite_path=tmp_path / "state.sqlite",
            seed=None,
            progress_interval_seconds=30,
            allow_unsafe_windows_wsl_fallback=False,
            review_scope={"base": "main"},
            prompt="",
            caller_id="caller-1",
            caller_id_source="explicit",
            ignore_pending_grades=False,
            task_id=None,
            rating_pool_id=None,
            rank_groups=None,
            basis=None,
            note=None,
            public_task_name="review_t1",
        )

    assert writes == []


def test_run_benchmarked_round_runtime_guard_does_not_persist_sampled_round(
    monkeypatch, tmp_path
) -> None:
    writes: list[dict[str, object]] = []

    monkeypatch.setattr(
        "review_suite_arena._ensure_no_pending_grades", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "review_suite_arena.ensure_clean_git_worktree", lambda *args, **kwargs: None
    )

    def fail_runtime_check(**kwargs):
        raise ValueError("rerun with --wsl")

    monkeypatch.setattr("review_suite_arena.validate_codex_runtime", fail_runtime_check)
    monkeypatch.setattr(
        "review_suite_arena.select_pair",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("select_pair should not run")
        ),
    )
    monkeypatch.setattr(
        "review_suite_arena.write_round",
        lambda state_dir, payload: writes.append(dict(payload)),
    )

    with pytest.raises(ValueError, match="--wsl"):
        run_benchmarked_round(
            task_class="phase_review",
            review_cwd=tmp_path,
            roster_path=tmp_path / "roster.json",
            rubric_path=tmp_path / "rubric.json",
            state_dir=tmp_path / "state",
            sqlite_path=tmp_path / "state.sqlite",
            seed=None,
            progress_interval_seconds=30,
            allow_unsafe_windows_wsl_fallback=False,
            review_scope={"base": "main"},
            prompt="",
            caller_id="caller-1",
            caller_id_source="explicit",
            ignore_pending_grades=False,
            task_id=None,
            rating_pool_id=None,
            rank_groups=None,
            basis=None,
            note=None,
            public_task_name="review_t1",
        )

    assert writes == []


def test_run_benchmarked_round_interactive_blocked_round_skips_final_toon(
    monkeypatch, tmp_path
) -> None:
    next_steps: list[dict[str, object]] = []

    monkeypatch.setattr(
        "review_suite_arena._ensure_no_pending_grades", lambda **kwargs: None
    )
    monkeypatch.setattr("review_suite_arena.load_roster", lambda path: {"variants": []})
    monkeypatch.setattr(
        "review_suite_arena.load_rubric", lambda path: {"categories": []}
    )
    monkeypatch.setattr("review_suite_arena.read_jsonl", lambda path: [])
    monkeypatch.setattr("review_suite_arena.load_operational_state", lambda path: {})
    monkeypatch.setattr(
        "review_suite_arena.select_pair",
        lambda **kwargs: {"round_id": "round-1", "task_class": "phase_review"},
    )
    monkeypatch.setattr("review_suite_arena.write_round", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "review_suite_arena.run_round",
        lambda **kwargs: {
            "round_id": "round-1",
            "task_class": "phase_review",
            "runs": [
                {"slot": "alpha", "review_status": "timeout", "grade_blocked": True}
            ],
        },
    )
    monkeypatch.setattr(
        "review_suite_arena.public_round_result",
        lambda *args, **kwargs: {
            "blocked": True,
            "runs": [{"slot": "alpha", "blocked": True}],
        },
    )
    monkeypatch.setattr(
        "review_suite_arena._print_findings", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "review_suite_arena._current_branch_name", lambda review_cwd: "branch-1"
    )
    monkeypatch.setattr("review_suite_arena._output_isatty", lambda: True)
    monkeypatch.setattr("review_suite_arena._print_round_banner", lambda **kwargs: None)
    monkeypatch.setattr(
        "review_suite_arena._print_next_steps",
        lambda **kwargs: next_steps.append(kwargs),
    )
    monkeypatch.setattr(
        "review_suite_arena.emit_toon",
        lambda payload: (_ for _ in ()).throw(
            AssertionError("interactive blocked runs should not emit final TOON")
        ),
    )

    result = run_benchmarked_round(
        task_class="phase_review",
        review_cwd=tmp_path,
        roster_path=tmp_path / "roster.json",
        rubric_path=tmp_path / "rubric.json",
        state_dir=tmp_path / "state",
        sqlite_path=tmp_path / "state.sqlite",
        seed=None,
        progress_interval_seconds=30,
        allow_unsafe_windows_wsl_fallback=False,
        review_scope={"base": "main"},
        prompt="",
        caller_id="caller-1",
        caller_id_source="explicit",
        ignore_pending_grades=False,
        task_id=None,
        rating_pool_id=None,
        rank_groups=None,
        basis=None,
        note=None,
        public_task_name="review_t1",
    )

    assert result == 0
    assert next_steps


def test_run_benchmarked_round_direct_grade_rejects_ambiguous_caller_pending_rounds(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "review_suite_arena.find_pending_rounds_for_caller",
        lambda **kwargs: [
            {"round_id": "round-a", "task_class": "pr_review"},
            {"round_id": "round-b", "task_class": "pr_review"},
        ],
    )

    with pytest.raises(
        ValueError, match="multiple pending pr_review rounds found for this caller"
    ):
        run_benchmarked_round(
            task_class="pr_review",
            review_cwd=tmp_path,
            roster_path=tmp_path / "roster.json",
            rubric_path=tmp_path / "rubric.json",
            state_dir=tmp_path / "state",
            sqlite_path=tmp_path / "state.sqlite",
            seed=None,
            progress_interval_seconds=30,
            allow_unsafe_windows_wsl_fallback=False,
            review_scope={"base": "main"},
            prompt="",
            caller_id="caller-1",
            caller_id_source="explicit",
            ignore_pending_grades=False,
            task_id="task-123",
            rating_pool_id="pool-v1",
            rank_groups=["alpha", "bravo"],
            basis=BASIS,
            note=None,
        )


def test_run_benchmarked_round_complete_grade_inputs_without_pending_falls_through(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "review_suite_arena._ensure_no_pending_grades", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "review_suite_arena.find_pending_rounds_for_caller", lambda **kwargs: []
    )
    monkeypatch.setattr("review_suite_arena.iter_round_payloads", lambda state_dir: [])
    monkeypatch.setattr("review_suite_arena.load_roster", lambda path: {"variants": []})
    monkeypatch.setattr(
        "review_suite_arena.load_rubric", lambda path: {"categories": []}
    )
    monkeypatch.setattr("review_suite_arena.read_jsonl", lambda path: [])
    monkeypatch.setattr("review_suite_arena.load_operational_state", lambda path: {})
    monkeypatch.setattr(
        "review_suite_arena.select_pair",
        lambda **kwargs: {"round_id": "new-round", "task_class": "pr_review"},
    )
    monkeypatch.setattr("review_suite_arena.write_round", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "review_suite_arena.run_round",
        lambda **kwargs: {
            "round_id": "new-round",
            "task_class": "pr_review",
            "runs": [],
        },
    )
    monkeypatch.setattr(
        "review_suite_arena.public_round_result",
        lambda *args, **kwargs: {"blocked": False, "runs": []},
    )
    monkeypatch.setattr(
        "review_suite_arena._print_findings", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "review_suite_arena._current_branch_name", lambda review_cwd: "branch-1"
    )
    monkeypatch.setattr("review_suite_arena._print_next_steps", lambda **kwargs: None)
    monkeypatch.setattr(
        "review_suite_arena._record_grade_result",
        lambda **kwargs: {"status": "graded", "round_id": kwargs["round_id"]},
    )
    monkeypatch.setattr("review_suite_arena.emit_toon", lambda payload: None)

    result = run_benchmarked_round(
        task_class="pr_review",
        review_cwd=tmp_path,
        roster_path=tmp_path / "roster.json",
        rubric_path=tmp_path / "rubric.json",
        state_dir=tmp_path / "state",
        sqlite_path=tmp_path / "state.sqlite",
        seed=None,
        progress_interval_seconds=30,
        allow_unsafe_windows_wsl_fallback=False,
        review_scope={"base": "main"},
        prompt="",
        caller_id="caller-1",
        caller_id_source="explicit",
        ignore_pending_grades=False,
        task_id="task-123",
        rating_pool_id="pool-v1",
        rank_groups=["alpha", "bravo"],
        basis=BASIS,
        note=None,
    )

    assert result == 0


def test_run_benchmarked_round_direct_grade_without_caller_uses_single_repo_pending_round(
    monkeypatch, tmp_path
) -> None:
    captured: dict[str, object] = {}
    emitted: list[dict[str, object]] = []

    monkeypatch.setattr(
        "review_suite_arena._ensure_no_pending_grades",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("should not check pending before direct grade")
        ),
    )
    monkeypatch.setattr("review_suite_arena.load_roster", lambda path: {"variants": []})
    monkeypatch.setattr(
        "review_suite_arena.load_rubric", lambda path: {"categories": []}
    )
    monkeypatch.setattr(
        "review_suite_arena.find_pending_rounds_for_caller", lambda **kwargs: []
    )
    monkeypatch.setattr(
        "review_suite_arena.iter_round_payloads",
        lambda state_dir: [
            {
                "round_id": "repo-round",
                "task_class": "pr_review",
                "status": "completed",
                "review_cwd_normalized": str(tmp_path),
                "runs": [
                    {
                        "slot": "alpha",
                        "review_status": "completed",
                        "grade_blocked": False,
                    },
                    {
                        "slot": "bravo",
                        "review_status": "completed",
                        "grade_blocked": False,
                    },
                ],
            }
        ],
    )

    def fake_record_grade_result(**kwargs):
        captured.update(kwargs)
        return {"status": "graded", "round_id": kwargs["round_id"]}

    monkeypatch.setattr(
        "review_suite_arena._record_grade_result", fake_record_grade_result
    )
    monkeypatch.setattr(
        "review_suite_arena.emit_toon", lambda payload: emitted.append(payload)
    )

    result = run_benchmarked_round(
        task_class="pr_review",
        review_cwd=tmp_path,
        roster_path=tmp_path / "roster.json",
        rubric_path=tmp_path / "rubric.json",
        state_dir=tmp_path / "state",
        sqlite_path=tmp_path / "state.sqlite",
        seed=None,
        progress_interval_seconds=30,
        allow_unsafe_windows_wsl_fallback=False,
        review_scope={"base": "main"},
        prompt="",
        caller_id=None,
        caller_id_source=None,
        ignore_pending_grades=False,
        task_id="task-123",
        rating_pool_id="pool-v1",
        rank_groups=["alpha", "bravo"],
        basis=BASIS,
        note=None,
    )

    assert result == 0
    assert captured["round_id"] == "repo-round"
    assert emitted == [
        {"status": "graded", "round_id": "repo-round", "task": "review_t3"}
    ]


def test_run_benchmarked_round_direct_grade_with_caller_falls_back_to_unique_repo_pending_round(
    monkeypatch, tmp_path
) -> None:
    captured: dict[str, object] = {}
    emitted: list[dict[str, object]] = []

    monkeypatch.setattr(
        "review_suite_arena._ensure_no_pending_grades",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("should not check pending before direct grade")
        ),
    )
    monkeypatch.setattr("review_suite_arena.load_roster", lambda path: {"variants": []})
    monkeypatch.setattr(
        "review_suite_arena.load_rubric", lambda path: {"categories": []}
    )
    monkeypatch.setattr(
        "review_suite_arena.find_pending_rounds_for_caller", lambda **kwargs: []
    )
    monkeypatch.setattr(
        "review_suite_arena.iter_round_payloads",
        lambda state_dir: [
            {
                "round_id": "repo-round",
                "task_class": "pr_review",
                "status": "completed",
                "review_cwd_normalized": str(tmp_path),
                "runs": [
                    {
                        "slot": "alpha",
                        "review_status": "completed",
                        "grade_blocked": False,
                    },
                    {
                        "slot": "bravo",
                        "review_status": "completed",
                        "grade_blocked": False,
                    },
                ],
            }
        ],
    )

    def fake_record_grade_result(**kwargs):
        captured.update(kwargs)
        return {"status": "graded", "round_id": kwargs["round_id"]}

    monkeypatch.setattr(
        "review_suite_arena._record_grade_result", fake_record_grade_result
    )
    monkeypatch.setattr(
        "review_suite_arena.emit_toon", lambda payload: emitted.append(payload)
    )

    result = run_benchmarked_round(
        task_class="pr_review",
        review_cwd=tmp_path,
        roster_path=tmp_path / "roster.json",
        rubric_path=tmp_path / "rubric.json",
        state_dir=tmp_path / "state",
        sqlite_path=tmp_path / "state.sqlite",
        seed=None,
        progress_interval_seconds=30,
        allow_unsafe_windows_wsl_fallback=False,
        review_scope={"base": "main"},
        prompt="",
        caller_id="caller-1",
        caller_id_source="explicit",
        ignore_pending_grades=False,
        task_id="task-123",
        rating_pool_id="pool-v1",
        rank_groups=["alpha", "bravo"],
        basis=BASIS,
        note=None,
    )

    assert result == 0
    assert captured["round_id"] == "repo-round"
    assert emitted == [
        {"status": "graded", "round_id": "repo-round", "task": "review_t3"}
    ]


def test_run_benchmarked_round_direct_grade_without_caller_rejects_ambiguous_repo_pending_rounds(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "review_suite_arena.find_pending_rounds_for_caller", lambda **kwargs: []
    )
    monkeypatch.setattr(
        "review_suite_arena.iter_round_payloads",
        lambda state_dir: [
            {
                "round_id": "round-a",
                "task_class": "pr_review",
                "status": "completed",
                "review_cwd_normalized": str(tmp_path),
                "runs": [],
            },
            {
                "round_id": "round-b",
                "task_class": "pr_review",
                "status": "completed",
                "review_cwd_normalized": str(tmp_path),
                "runs": [],
            },
        ],
    )

    with pytest.raises(ValueError, match="multiple pending pr_review rounds found"):
        run_benchmarked_round(
            task_class="pr_review",
            review_cwd=tmp_path,
            roster_path=tmp_path / "roster.json",
            rubric_path=tmp_path / "rubric.json",
            state_dir=tmp_path / "state",
            sqlite_path=tmp_path / "state.sqlite",
            seed=None,
            progress_interval_seconds=30,
            allow_unsafe_windows_wsl_fallback=False,
            review_scope={"base": "main"},
            prompt="",
            caller_id=None,
            caller_id_source=None,
            ignore_pending_grades=False,
            task_id="task-123",
            rating_pool_id="pool-v1",
            rank_groups=["alpha", "bravo"],
            basis=BASIS,
            note=None,
        )


def test_cmd_reroll_slot_records_workflow_anchor_when_completed(
    monkeypatch, tmp_path
) -> None:
    anchor_calls: list[dict[str, object]] = []
    emitted: list[dict[str, object]] = []

    monkeypatch.setattr("review_suite_arena.load_roster", lambda path: {"variants": []})
    monkeypatch.setattr("review_suite_arena.read_jsonl", lambda path: [])
    monkeypatch.setattr(
        "review_suite_arena.load_round",
        lambda state_dir, round_id: {
            "round_id": "round-1",
            "task_class": "pr_review",
            "review_cwd": str(tmp_path),
            "review_cwd_normalized": "old-normalized-cwd",
            "review_scope": {"base": "main", "reviewed_head": "head-sha"},
            "requested_prompt": "Stored reroll prompt",
            "arena_round": True,
            "grading_required": True,
            "reporting_pool": True,
        },
    )
    monkeypatch.setattr(
        "review_suite_arena.build_reroll_slot_payload",
        lambda **kwargs: {"round_id": "round-2", "task_class": "pr_review", "runs": []},
    )
    run_calls: list[dict[str, object]] = []
    writes: list[dict[str, object]] = []

    def fake_run_round(**kwargs):
        run_calls.append(dict(kwargs))
        assert kwargs["prompt"] == "Stored reroll prompt"
        assert kwargs["review_scope"] == {"base": "main", "reviewed_head": "head-sha"}
        return {"round_id": "round-2", "task_class": "pr_review", "runs": []}

    monkeypatch.setattr("review_suite_arena.run_round", fake_run_round)
    monkeypatch.setattr(
        "review_suite_arena.public_round_result",
        lambda *args, **kwargs: {"blocked": False, "runs": []},
    )
    monkeypatch.setattr(
        "review_suite_arena._print_findings", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "review_suite_arena._current_branch_name", lambda review_cwd: "branch-1"
    )
    monkeypatch.setattr("review_suite_arena._output_isatty", lambda: False)
    monkeypatch.setattr(
        "review_suite_arena.write_round",
        lambda state_dir, payload: writes.append(dict(payload)) or None,
    )
    monkeypatch.setattr(
        "review_suite_arena.record_review_anchor",
        lambda **kwargs: anchor_calls.append(kwargs) or {},
    )
    monkeypatch.setattr(
        "review_suite_arena.emit_toon", lambda payload: emitted.append(payload)
    )

    result = __import__("review_suite_arena").cmd_reroll_slot(
        Namespace(
            round_id="round-1",
            slot="alpha",
            cd=str(tmp_path),
            base="main",
            seed=None,
            roster=str(tmp_path / "roster.json"),
            rubric=str(tmp_path / "rubric.json"),
            state_dir=str(tmp_path / "state"),
            sqlite_path=str(tmp_path / "state.sqlite"),
            progress_interval_seconds=30,
            wsl=False,
        )
    )

    assert result == 0
    assert emitted[0]["Action"]["cmd"]
    assert "runs" not in emitted[0]
    assert len(run_calls) == 1
    assert writes[0]["requested_prompt"] == "Stored reroll prompt"
    assert writes[0]["review_scope"] == {"base": "main", "reviewed_head": "head-sha"}
    assert writes[0]["review_cwd_normalized"] == str(tmp_path.resolve())
    assert writes[0]["arena_round"] is True
    assert writes[0]["grading_required"] is True
    assert writes[0]["reporting_pool"] is True
    assert anchor_calls[0]["lane"] == "review_t3"
    assert anchor_calls[0]["task_id"] == "branch-1"


def test_cmd_reroll_slot_skips_workflow_anchor_for_orchestrated_round(
    monkeypatch, tmp_path
) -> None:
    anchor_calls: list[dict[str, object]] = []

    monkeypatch.setattr("review_suite_arena.load_roster", lambda path: {"variants": []})
    monkeypatch.setattr("review_suite_arena.read_jsonl", lambda path: [])
    monkeypatch.setattr(
        "review_suite_arena.load_round",
        lambda state_dir, round_id: {
            "round_id": "round-1",
            "task_class": "pr_review",
            "review_cwd": str(tmp_path),
            "review_scope": {"base": "main", "reviewed_head": "head-sha"},
            "requested_prompt": "Stored reroll prompt",
            "arena_round": True,
            "orchestrator_step": "arena-pr-review",
            "grading_required": True,
        },
    )
    monkeypatch.setattr(
        "review_suite_arena.build_reroll_slot_payload",
        lambda **kwargs: {
            "round_id": "round-2",
            "task_class": "pr_review",
            "arena_round": False,
            "orchestrator_step": "",
            "runs": [],
        },
    )
    writes: list[dict[str, object]] = []
    monkeypatch.setattr(
        "review_suite_arena.run_round",
        lambda **kwargs: {"round_id": "round-2", "task_class": "pr_review", "runs": []},
    )
    monkeypatch.setattr(
        "review_suite_arena.public_round_result",
        lambda *args, **kwargs: {"blocked": False, "runs": []},
    )
    monkeypatch.setattr(
        "review_suite_arena._print_findings", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "review_suite_arena._current_branch_name", lambda review_cwd: "branch-1"
    )
    monkeypatch.setattr("review_suite_arena._output_isatty", lambda: False)
    monkeypatch.setattr(
        "review_suite_arena.write_round",
        lambda state_dir, payload: writes.append(dict(payload)) or None,
    )
    monkeypatch.setattr(
        "review_suite_arena.record_review_anchor",
        lambda **kwargs: anchor_calls.append(kwargs) or {},
    )
    monkeypatch.setattr("review_suite_arena.emit_toon", lambda payload: None)

    result = __import__("review_suite_arena").cmd_reroll_slot(
        Namespace(
            round_id="round-1",
            slot="alpha",
            cd=str(tmp_path),
            base="main",
            seed=None,
            roster=str(tmp_path / "roster.json"),
            rubric=str(tmp_path / "rubric.json"),
            state_dir=str(tmp_path / "state"),
            sqlite_path=str(tmp_path / "state.sqlite"),
            progress_interval_seconds=30,
            wsl=False,
        )
    )

    assert result == 0
    assert writes[0]["arena_round"] is True
    assert writes[0]["orchestrator_step"] == "arena-pr-review"
    assert anchor_calls == []


def test_cmd_run_round_records_workflow_anchor_when_completed(
    monkeypatch, tmp_path, capsys
) -> None:
    anchor_calls: list[dict[str, object]] = []
    emitted: list[dict[str, object]] = []

    monkeypatch.setattr("review_suite_arena.load_roster", lambda path: {"variants": []})
    monkeypatch.setattr(
        "review_suite_arena.load_round",
        lambda state_dir, round_id: {
            "round_id": "round-1",
            "task_class": "pr_review",
            "review_cwd": str(tmp_path),
            "review_scope": {"base": "main", "reviewed_head": "head-sha"},
            "requested_prompt": "Stored arena prompt",
        },
    )
    run_calls: list[dict[str, object]] = []

    def fake_run_round(**kwargs):
        run_calls.append(dict(kwargs))
        assert kwargs["prompt"] == "Stored arena prompt"
        assert kwargs["review_scope"] == {"base": "main", "reviewed_head": "head-sha"}
        return {
            "round_id": "round-1",
            "task_class": "pr_review",
            "review_scope": {"base": "main", "reviewed_head": "head-sha"},
            "runs": [
                {
                    "slot": "alpha",
                    "review_status": "completed",
                    "reviewer_output": "Run-round body",
                }
            ],
        }

    monkeypatch.setattr(
        "review_suite_arena.run_round",
        fake_run_round,
    )
    monkeypatch.setattr("review_suite_arena.write_round", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "review_suite_arena.public_round_result",
        lambda *args, **kwargs: {"blocked": False, "runs": []},
    )
    monkeypatch.setattr("review_suite_arena._resolve_review_cwd", lambda cd: tmp_path)
    monkeypatch.setattr(
        "review_suite_arena._current_branch_name", lambda review_cwd: "branch-1"
    )
    monkeypatch.setattr("review_suite_arena._output_isatty", lambda: False)
    monkeypatch.setattr(
        "review_suite_arena.record_review_anchor",
        lambda **kwargs: anchor_calls.append(kwargs) or {},
    )
    monkeypatch.setattr(
        "review_suite_arena.emit_toon", lambda payload: emitted.append(payload)
    )

    result = __import__("review_suite_arena").cmd_run_round(
        Namespace(
            round_id="round-1",
            cd=str(tmp_path),
            base="main",
            roster=str(tmp_path / "roster.json"),
            state_dir=str(tmp_path / "state"),
            sqlite_path=str(tmp_path / "state.sqlite"),
            progress_interval_seconds=30,
            wsl=False,
        )
    )

    assert result == 0
    assert emitted[0]["Action"]["cmd"]
    assert "runs" not in emitted[0]
    assert len(run_calls) == 1
    assert anchor_calls[0]["lane"] == "review_t3"
    assert anchor_calls[0]["task_id"] == "branch-1"
    assert "Run-round body" in capsys.readouterr().out


def test_cmd_run_round_skips_workflow_anchor_for_orchestrated_round(
    monkeypatch, tmp_path
) -> None:
    anchor_calls: list[dict[str, object]] = []

    monkeypatch.setattr("review_suite_arena.load_roster", lambda path: {"variants": []})
    monkeypatch.setattr(
        "review_suite_arena.load_round",
        lambda state_dir, round_id: {
            "round_id": "round-1",
            "task_class": "pr_review",
            "review_cwd": str(tmp_path),
            "review_scope": {"base": "main", "reviewed_head": "head-sha"},
            "requested_prompt": "Stored arena prompt",
            "arena_round": True,
            "orchestrator_step": "arena-pr-review",
            "grading_required": True,
        },
    )
    monkeypatch.setattr(
        "review_suite_arena.run_round",
        lambda **kwargs: {
            "round_id": "round-1",
            "task_class": "pr_review",
            "review_scope": {"base": "main", "reviewed_head": "head-sha"},
            "runs": [
                {
                    "slot": "alpha",
                    "review_status": "completed",
                    "reviewer_output": "Run-round body",
                }
            ],
        },
    )
    monkeypatch.setattr("review_suite_arena.write_round", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "review_suite_arena.public_round_result",
        lambda *args, **kwargs: {"blocked": False, "runs": []},
    )
    monkeypatch.setattr("review_suite_arena._resolve_review_cwd", lambda cd: tmp_path)
    monkeypatch.setattr(
        "review_suite_arena._current_branch_name", lambda review_cwd: "branch-1"
    )
    monkeypatch.setattr("review_suite_arena._output_isatty", lambda: False)
    monkeypatch.setattr(
        "review_suite_arena.record_review_anchor",
        lambda **kwargs: anchor_calls.append(kwargs) or {},
    )
    monkeypatch.setattr("review_suite_arena.emit_toon", lambda payload: None)

    result = __import__("review_suite_arena").cmd_run_round(
        Namespace(
            round_id="round-1",
            cd=str(tmp_path),
            base="main",
            roster=str(tmp_path / "roster.json"),
            state_dir=str(tmp_path / "state"),
            sqlite_path=str(tmp_path / "state.sqlite"),
            progress_interval_seconds=30,
            wsl=False,
        )
    )

    assert result == 0
    assert anchor_calls == []


def test_cmd_resume_round_records_workflow_anchor_when_completed(
    monkeypatch, tmp_path, capsys
) -> None:
    anchor_calls: list[dict[str, object]] = []
    emitted: list[dict[str, object]] = []

    monkeypatch.setattr("review_suite_arena.load_roster", lambda path: {"variants": []})
    monkeypatch.setattr(
        "review_suite_arena.load_round",
        lambda state_dir, round_id: {
            "round_id": "round-1",
            "task_class": "pr_review",
            "status": "running",
            "review_cwd": str(tmp_path),
            "review_scope": {"base": "main"},
        },
    )
    monkeypatch.setattr(
        "review_suite_arena.collect_round_results",
        lambda **kwargs: {
            "round_id": "round-1",
            "task_class": "pr_review",
            "review_scope": {"base": "main"},
            "runs": [
                {
                    "slot": "alpha",
                    "review_status": "completed",
                    "reviewer_output": "Resume body",
                }
            ],
        },
    )
    monkeypatch.setattr(
        "review_suite_arena.public_round_result",
        lambda *args, **kwargs: {"blocked": False, "runs": []},
    )
    monkeypatch.setattr(
        "review_suite_arena._current_branch_name", lambda review_cwd: "branch-1"
    )
    monkeypatch.setattr("review_suite_arena._output_isatty", lambda: False)
    monkeypatch.setattr(
        "review_suite_arena.record_review_anchor",
        lambda **kwargs: anchor_calls.append(kwargs) or {},
    )
    monkeypatch.setattr(
        "review_suite_arena.emit_toon", lambda payload: emitted.append(payload)
    )

    result = cmd_resume_round(
        Namespace(
            round_id="round-1",
            cd=None,
            roster=str(tmp_path / "roster.json"),
            state_dir=str(tmp_path / "state"),
            sqlite_path=str(tmp_path / "state.sqlite"),
            progress_interval_seconds=30,
        )
    )

    assert result == 0
    assert emitted[0]["Action"]["cmd"]
    assert "runs" not in emitted[0]
    assert anchor_calls[0]["lane"] == "review_t3"
    assert anchor_calls[0]["task_id"] == "branch-1"
    assert "Resume body" in capsys.readouterr().out


def test_cmd_resume_round_skips_workflow_anchor_for_orchestrated_round(
    monkeypatch, tmp_path
) -> None:
    anchor_calls: list[dict[str, object]] = []

    monkeypatch.setattr("review_suite_arena.load_roster", lambda path: {"variants": []})
    monkeypatch.setattr(
        "review_suite_arena.load_round",
        lambda state_dir, round_id: {
            "round_id": "round-1",
            "task_class": "pr_review",
            "status": "running",
            "review_cwd": str(tmp_path),
            "review_scope": {"base": "main"},
            "arena_round": True,
            "orchestrator_step": "arena-pr-review",
            "grading_required": True,
        },
    )
    monkeypatch.setattr(
        "review_suite_arena.collect_round_results",
        lambda **kwargs: {
            "round_id": "round-1",
            "task_class": "pr_review",
            "review_scope": {"base": "main"},
            "runs": [
                {
                    "slot": "alpha",
                    "review_status": "completed",
                    "reviewer_output": "Resume body",
                }
            ],
        },
    )
    monkeypatch.setattr(
        "review_suite_arena.public_round_result",
        lambda *args, **kwargs: {"blocked": False, "runs": []},
    )
    monkeypatch.setattr(
        "review_suite_arena._current_branch_name", lambda review_cwd: "branch-1"
    )
    monkeypatch.setattr("review_suite_arena._output_isatty", lambda: False)
    monkeypatch.setattr(
        "review_suite_arena.record_review_anchor",
        lambda **kwargs: anchor_calls.append(kwargs) or {},
    )
    monkeypatch.setattr("review_suite_arena.emit_toon", lambda payload: None)

    result = cmd_resume_round(
        Namespace(
            round_id="round-1",
            cd=None,
            roster=str(tmp_path / "roster.json"),
            state_dir=str(tmp_path / "state"),
            sqlite_path=str(tmp_path / "state.sqlite"),
            progress_interval_seconds=30,
        )
    )

    assert result == 0
    assert anchor_calls == []


def test_resume_orchestrator_review_step_collects_existing_running_round(
    monkeypatch, tmp_path
) -> None:
    state_dir = tmp_path / "state"
    round_state_dir = state_dir / "orchestrator" / "review-rounds"
    review_cwd = tmp_path / "repo"
    review_cwd.mkdir()
    collect_calls: list[dict[str, object]] = []
    round_payload = {
        "round_id": "phase_review-round-1",
        "task_class": "phase_review",
        "status": "running",
        "review_cwd": str(review_cwd),
        "review_scope": {"base": "main", "reviewed_head": "head-1"},
        "runs": [
            {
                "slot": "alpha",
                "variant_id": "gpt-5.5-medium",
                "model": "gpt-5.5",
                "reasoning_effort": "medium",
                "service_tier": "flex",
            }
        ],
    }
    write_round(round_state_dir, round_payload)

    def fake_collect_round_results(**kwargs: object) -> dict[str, object]:
        collect_calls.append(dict(kwargs))
        assert kwargs["roster"] == {
            "settings": {},
            "variants": [
                {
                    "id": "gpt-5.5-medium",
                    "state": "active",
                    "model": "gpt-5.5",
                    "reasoning_effort": "medium",
                    "service_tier": "flex",
                    "task_classes": ["phase_review"],
                }
            ],
        }
        return {
            **round_payload,
            "status": "completed",
            "runs": [
                {
                    "slot": "alpha",
                    "review_status": "completed",
                    "status_summary": "No findings.",
                    "grade_blocked": False,
                    "grade_block_reason": None,
                    "reviewer_output": "No findings.",
                    "reviewer_output_ref": "rollout://phase_review-round-1/alpha",
                }
            ],
        }

    monkeypatch.setattr(
        "review_suite_arena.collect_round_results", fake_collect_round_results
    )
    monkeypatch.setattr(
        "review_suite_arena.apply_review_cost_delta_best_effort",
        lambda **kwargs: None,
    )
    monkeypatch.setattr("review_suite_arena._print_findings", lambda result: False)

    result = resume_orchestrator_review_step(
        round_id="phase_review-round-1",
        lane="review_t1",
        step_name="precision",
        review_cwd=review_cwd,
        state_dir=state_dir,
        round_state_dir=round_state_dir,
        sqlite_path=tmp_path / "state.sqlite",
        task_id="branch-1",
        progress_interval_seconds=1,
    )

    assert len(collect_calls) == 1
    assert result["round_id"] == "phase_review-round-1"
    assert result["status"] == "completed"
    assert result["output_refs"] == ["rollout://phase_review-round-1/alpha"]
    saved = json.loads(
        (round_state_dir / "rounds" / "phase_review-round-1.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved["task_id_hint"] == "branch-1"
    assert saved["public_task"] == "review_t1"
    assert saved["orchestrator_step"] == "precision"


def test_normalize_arena_task_class_accepts_public_aliases() -> None:
    assert _normalize_arena_task_class("review_t1") == "phase_review"
    assert _normalize_arena_task_class("review_t3") == "pr_review"


def test_cmd_sample_emits_public_task_alias(monkeypatch, tmp_path) -> None:
    emitted: list[dict[str, object]] = []
    captured: dict[str, object] = {}
    repo_root = tmp_path / "repo-root"

    monkeypatch.setattr(
        "review_suite_arena.resolve_caller_id", lambda caller_id: (None, None)
    )
    monkeypatch.setattr("review_suite_arena._resolve_review_cwd", lambda cd: repo_root)
    monkeypatch.setattr("review_suite_arena.load_roster", lambda path: {"variants": []})
    monkeypatch.setattr("review_suite_arena.read_jsonl", lambda path: [])
    monkeypatch.setattr(
        "review_suite_arena.load_operational_state", lambda path: {"task_classes": {}}
    )
    monkeypatch.setattr(
        "review_suite_arena.select_pair",
        lambda **kwargs: (
            captured.update(kwargs)
            or {
                "round_id": "round-1",
                "task_class": "phase_review",
                "status": "sampled",
                "runs": [],
            }
        ),
    )
    monkeypatch.setattr("review_suite_arena.write_round", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "review_suite_arena.emit_toon", lambda payload: emitted.append(payload)
    )

    result = __import__("review_suite_arena").cmd_sample(
        Namespace(
            task_class="review_t1",
            caller_id=None,
            roster=str(tmp_path / "roster.json"),
            state_dir=str(tmp_path / "state"),
            ignore_pending_grades=True,
            seed=None,
            exclude_variant_id=[],
        )
    )

    assert result == 0
    assert captured["task_class"] == "phase_review"
    assert captured["review_cwd"] == repo_root
    assert emitted == [
        {
            "round_id": "round-1",
            "task": "review_t1",
            "status": "sampled",
            "sampled_at": None,
            "reviewers": [],
        }
    ]


def test_cmd_sample_dry_run_emits_selection_without_writing(
    monkeypatch, tmp_path
) -> None:
    emitted: list[dict[str, object]] = []
    writes: list[object] = []
    cleanup_calls: list[object] = []
    repo_root = tmp_path / "repo-root"

    monkeypatch.setattr(
        "review_suite_arena.resolve_caller_id", lambda caller_id: (None, None)
    )
    monkeypatch.setattr("review_suite_arena._resolve_review_cwd", lambda cd: repo_root)
    monkeypatch.setattr("review_suite_arena.load_roster", lambda path: {"variants": []})
    monkeypatch.setattr("review_suite_arena.read_jsonl", lambda path: [])
    monkeypatch.setattr(
        "review_suite_arena.load_operational_state", lambda path: {"task_classes": {}}
    )
    monkeypatch.setattr(
        "review_suite_arena._raise_if_blocking_round_exists",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("dry run must skip the blocking check")
        ),
    )
    monkeypatch.setattr(
        "review_suite_arena.select_pair",
        lambda **kwargs: {
            "round_id": "round-1",
            "task_class": "phase_review",
            "status": "sampled",
            "runs": [
                {
                    "slot": "alpha",
                    "variant_id": "deepseek-v4.1-flash-low",
                    "model": "opencode::opencode-go/deepseek-v4.1-flash",
                    "reasoning_effort": "low",
                }
            ],
        },
    )
    monkeypatch.setattr(
        "review_suite_arena.write_round", lambda *args, **kwargs: writes.append(args)
    )
    monkeypatch.setattr(
        "review_suite_arena.emit_toon", lambda payload: emitted.append(payload)
    )
    monkeypatch.setattr(
        "review_suite_local.cleanup_stale_ungraded_rounds",
        lambda state_dir: cleanup_calls.append(state_dir),
    )

    result = __import__("review_suite_arena").cmd_sample(
        Namespace(
            task_class="review_t1",
            caller_id=None,
            roster=str(tmp_path / "roster.json"),
            state_dir=str(tmp_path / "state"),
            ignore_pending_grades=True,
            seed=None,
            exclude_variant_id=[],
            dry_run=True,
        )
    )

    assert result == 0
    assert writes == []
    assert cleanup_calls == []
    assert emitted == [
        {
            "status": "dry_run",
            "task": "review_t1",
            "round_id": "round-1",
            "selection_mode": None,
            "selection_pairing": None,
            "reviewers": [
                {
                    "slot": "alpha",
                    "variant_id": "deepseek-v4.1-flash-low",
                    "model": "opencode::opencode-go/deepseek-v4.1-flash",
                    "reasoning_effort": "low",
                }
            ],
        }
    ]
