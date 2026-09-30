from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import rollout_capture  # noqa: E402
import review_suite_core  # noqa: E402
import review_suite_local  # noqa: E402
from review_suite_core.opencode_driver import REVIEW_METADATA_PREFIX  # noqa: E402


THREAD_SCHEMA = """
create table threads (
    id text primary key,
    rollout_path text,
    cwd text,
    source text,
    created_at integer,
    updated_at integer,
    tokens_used integer,
    model text,
    reasoning_effort text,
    title text
)
"""


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _assistant_message(text: str) -> dict[str, object]:
    return {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "assistant",
            "phase": "final_answer",
            "content": [{"text": text}],
        },
    }


def _launcher_rollout(path: Path, text: str) -> None:
    _write_jsonl(
        path,
        [
            {"type": "session_meta", "payload": {}},
            {"type": "event_msg", "payload": {"type": "entered_review_mode"}},
            _assistant_message(text),
            {"type": "event_msg", "payload": {"type": "task_complete"}},
        ],
    )


def _child_rollout(
    path: Path,
    *,
    model: str,
    reasoning_effort: str,
    usage: dict[str, int],
    text: str,
    parent_thread_id: str,
) -> None:
    total_tokens = int(usage["input_tokens"]) + int(usage["output_tokens"])
    _write_jsonl(
        path,
        [
            {
                "type": "session_meta",
                "payload": {"parent_thread_id": parent_thread_id},
            },
            {
                "type": "turn_context",
                "payload": {
                    "model": model,
                    "effort": reasoning_effort,
                },
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {
                            **usage,
                            "reasoning_output_tokens": 0,
                            "total_tokens": total_tokens,
                        },
                        "last_token_usage": {
                            **usage,
                            "reasoning_output_tokens": 0,
                            "total_tokens": total_tokens,
                        },
                    },
                },
            },
            _assistant_message(text),
        ],
    )


def test_rollout_activity_summary_ignores_token_count_heartbeats(
    tmp_path: Path,
) -> None:
    rollout = tmp_path / "heartbeat-only.jsonl"
    _write_jsonl(
        rollout,
        [
            {
                "timestamp": "2026-04-23T20:00:00Z",
                "type": "session_meta",
                "payload": {},
            },
            {
                "timestamp": "2026-04-23T20:00:01Z",
                "type": "turn_context",
                "payload": {},
            },
            {
                "timestamp": "2026-04-23T20:05:01Z",
                "type": "event_msg",
                "payload": {"type": "token_count", "info": {}},
            },
        ],
    )

    summary = rollout_capture.rollout_activity_summary(rollout)

    assert summary["last_event_at"].isoformat() == "2026-04-23T20:05:01+00:00"
    assert summary["last_meaningful_at"] is None
    assert summary["last_meaningful_type"] is None


def test_rollout_activity_summary_tracks_assistant_activity(tmp_path: Path) -> None:
    rollout = tmp_path / "assistant-output.jsonl"
    _write_jsonl(
        rollout,
        [
            {
                "timestamp": "2026-04-23T20:00:00Z",
                "type": "session_meta",
                "payload": {},
            },
            {
                "timestamp": "2026-04-23T20:00:01Z",
                "type": "turn_context",
                "payload": {},
            },
            {
                "timestamp": "2026-04-23T20:03:01Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"text": "No findings."}],
                },
            },
        ],
    )

    summary = rollout_capture.rollout_activity_summary(rollout)

    assert summary["last_meaningful_at"].isoformat() == "2026-04-23T20:03:01+00:00"
    assert summary["last_meaningful_type"] == "message"


def test_rollout_activity_summary_tracks_custom_tool_activity(tmp_path: Path) -> None:
    rollout = tmp_path / "tool-output.jsonl"
    _write_jsonl(
        rollout,
        [
            {
                "timestamp": "2026-04-23T20:00:00Z",
                "type": "session_meta",
                "payload": {},
            },
            {
                "timestamp": "2026-04-23T20:00:01Z",
                "type": "turn_context",
                "payload": {},
            },
            {
                "timestamp": "2026-04-23T20:04:01Z",
                "type": "response_item",
                "payload": {"type": "custom_tool_call", "name": "exec_command"},
            },
        ],
    )

    summary = rollout_capture.rollout_activity_summary(rollout)

    assert summary["last_meaningful_at"].isoformat() == "2026-04-23T20:04:01+00:00"
    assert summary["last_meaningful_type"] == "custom_tool_call"


def test_rollout_activity_summary_preserves_activity_across_compaction(
    tmp_path: Path,
) -> None:
    rollout = tmp_path / "compacted.jsonl"
    _write_jsonl(
        rollout,
        [
            {
                "timestamp": "2026-04-23T20:00:00Z",
                "type": "response_item",
                "payload": {"type": "custom_tool_call_output"},
            },
            {
                "timestamp": "2026-04-23T20:00:01Z",
                "type": "turn_context",
                "payload": {},
            },
        ],
    )

    summary = rollout_capture.rollout_activity_summary(rollout)

    assert summary["last_meaningful_at"].isoformat() == "2026-04-23T20:00:00+00:00"
    assert summary["last_meaningful_type"] == "custom_tool_call_output"


def _insert_thread(
    con: sqlite3.Connection,
    *,
    thread_id: str,
    rollout_path: Path,
    cwd: str,
    source: str,
    created_at: int,
    updated_at: int,
    tokens_used: int,
    title: str,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> None:
    con.execute(
        """
        insert into threads (
            id, rollout_path, cwd, source, created_at, updated_at, tokens_used, model, reasoning_effort, title
        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            thread_id,
            str(rollout_path),
            cwd,
            source,
            created_at,
            updated_at,
            tokens_used,
            model,
            reasoning_effort,
            title,
        ],
    )


def _minimal_roster() -> dict[str, object]:
    return {
        "variants": [
            {
                "id": "gpt-5.4-xhigh",
                "model": "gpt-5.4",
                "reasoning_effort": "xhigh",
                "pricing": {
                    "input_per_million_usd": 2.5,
                    "cached_input_per_million_usd": 0.25,
                    "output_per_million_usd": 15.0,
                },
            },
            {
                "id": "gpt-5.3-codex-medium",
                "model": "gpt-5.3-codex",
                "reasoning_effort": "medium",
                "pricing": {
                    "input_per_million_usd": 1.75,
                    "cached_input_per_million_usd": 0.175,
                    "output_per_million_usd": 14.0,
                },
            },
        ]
    }


def _minimal_rubric() -> dict[str, object]:
    return {
        "basis": [
            "valid_findings_vs_none",
            "more_valid_findings",
            "better_finding_validity",
            "better_bug_coverage",
            "false_positive_loss",
            "hallucinated_finding_loss",
            "fringe_finding_loss",
            "tie_clean",
            "tie_both_useful",
        ],
    }


def test_collect_round_results_persists_each_exited_reviewer_immediately(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state_dir = tmp_path / "state"
    review_cwd = tmp_path / "repo"
    review_cwd.mkdir()
    sqlite_path = tmp_path / "state_5.sqlite"

    alpha_stdout = tmp_path / "alpha.stdout"
    alpha_stderr = tmp_path / "alpha.stderr"
    bravo_stdout = tmp_path / "bravo.stdout"
    bravo_stderr = tmp_path / "bravo.stderr"
    alpha_stdout.write_text("alpha completed body\n", encoding="utf-8")
    alpha_stderr.write_text("", encoding="utf-8")
    bravo_stdout.write_text("bravo completed body\n", encoding="utf-8")
    bravo_stderr.write_text("", encoding="utf-8")

    round_payload = {
        "round_id": "round-live-capture",
        "task_class": "pr_review",
        "selection_mode": "scramble",
        "status": "running",
        "runs": [
            {
                "slot": "alpha",
                "variant_id": "gpt-5.4-xhigh",
                "title": "review-suite::round-live-capture::alpha::gpt-5.4-xhigh",
                "command": ["codex", "review"],
                "pid": 101,
                "started_at": "2026-04-23T12:00:00Z",
                "stdout_path": str(alpha_stdout),
                "stderr_path": str(alpha_stderr),
            },
            {
                "slot": "bravo",
                "variant_id": "gpt-5.3-codex-medium",
                "title": "review-suite::round-live-capture::bravo::gpt-5.3-codex-medium",
                "command": ["codex", "review"],
                "pid": 202,
                "started_at": "2026-04-23T12:00:00Z",
                "stdout_path": str(bravo_stdout),
                "stderr_path": str(bravo_stderr),
            },
        ],
    }

    bravo_checks = {"count": 0}

    def fake_process_is_running(pid: int | None) -> bool:
        if pid == 101:
            return False
        if pid == 202:
            bravo_checks["count"] += 1
            return bravo_checks["count"] == 1
        return False

    writes: list[dict[str, object]] = []
    streamed: list[dict[str, object]] = []

    def fake_write_round(_state_dir: Path, payload: dict[str, object]) -> Path:
        writes.append(json.loads(json.dumps(payload)))
        return _state_dir / "rounds" / f"{payload['round_id']}.json"

    monkeypatch.setattr(
        review_suite_local, "_process_is_running", fake_process_is_running
    )
    monkeypatch.setattr(
        review_suite_local, "find_review_child_thread", lambda **_: None
    )
    monkeypatch.setattr(review_suite_local, "write_round", fake_write_round)
    monkeypatch.setattr(
        review_suite_local,
        "_print_live_completed_run",
        lambda run: streamed.append(dict(run)),
    )
    monkeypatch.setattr(review_suite_local.time, "sleep", lambda _: None)

    completed = review_suite_local.collect_round_results(
        round_payload=round_payload,
        roster=_minimal_roster(),
        state_dir=state_dir,
        review_cwd=review_cwd,
        sqlite_path=sqlite_path,
        progress_interval_seconds=3600,
        wait=True,
    )

    alpha_partial = next(
        snapshot
        for snapshot in writes
        if snapshot["status"] == "running"
        and snapshot["runs"][0].get("reviewer_output") == "alpha completed body"
        and not snapshot["runs"][1].get("reviewer_output")
    )
    assert alpha_partial["runs"][0]["review_status"] == "completed"
    assert [run["slot"] for run in streamed] == ["alpha", "bravo"]
    assert completed["live_completion_statuses"] == {
        "alpha": "completed",
        "bravo": "completed",
    }
    assert all(
        "stdout_path" not in run and "stderr_path" not in run and "pid" not in run
        for run in completed["runs"]
    )
    assert not alpha_stdout.exists()
    assert not alpha_stderr.exists()
    assert not bravo_stdout.exists()
    assert not bravo_stderr.exists()


def test_collect_round_results_resolves_identical_parallel_launches_by_parent(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    sqlite_path = tmp_path / "state_5.sqlite"
    review_cwd = r"C:\repo\demo"
    prompt_title = "Review the code changes against the base branch 'main'."

    launcher_alpha_rollout = tmp_path / "launcher-alpha.jsonl"
    child_alpha_rollout = tmp_path / "child-alpha.jsonl"
    launcher_bravo_rollout = tmp_path / "launcher-bravo.jsonl"
    child_bravo_rollout = tmp_path / "child-bravo.jsonl"
    _launcher_rollout(launcher_alpha_rollout, "launcher alpha text")
    _launcher_rollout(launcher_bravo_rollout, "launcher bravo text")
    _child_rollout(
        child_alpha_rollout,
        model="gpt-5.4",
        reasoning_effort="xhigh",
        usage={"input_tokens": 1200, "cached_input_tokens": 300, "output_tokens": 70},
        text="alpha child findings",
        parent_thread_id="launcher-alpha",
    )
    _child_rollout(
        child_bravo_rollout,
        model="gpt-5.4",
        reasoning_effort="xhigh",
        usage={"input_tokens": 900, "cached_input_tokens": 200, "output_tokens": 40},
        text="bravo child findings",
        parent_thread_id="launcher-bravo",
    )

    con = sqlite3.connect(sqlite_path)
    con.execute(THREAD_SCHEMA)
    _insert_thread(
        con,
        thread_id="launcher-alpha",
        rollout_path=launcher_alpha_rollout,
        cwd=review_cwd,
        source="exec",
        created_at=1_700_000_100,
        updated_at=1_700_000_160,
        tokens_used=0,
        title=prompt_title,
    )
    _insert_thread(
        con,
        thread_id="child-alpha",
        rollout_path=child_alpha_rollout,
        cwd=review_cwd,
        source='{"subagent":"review"}',
        created_at=1_700_000_101,
        updated_at=1_700_000_160,
        tokens_used=1_570,
        title=prompt_title,
        model="gpt-5.4",
        reasoning_effort="xhigh",
    )
    _insert_thread(
        con,
        thread_id="launcher-bravo",
        rollout_path=launcher_bravo_rollout,
        cwd=review_cwd,
        source="exec",
        created_at=1_700_000_100,
        updated_at=1_700_000_140,
        tokens_used=0,
        title=prompt_title,
    )
    _insert_thread(
        con,
        thread_id="child-bravo",
        rollout_path=child_bravo_rollout,
        cwd=review_cwd,
        source='{"subagent":"review"}',
        created_at=1_700_000_101,
        updated_at=1_700_000_140,
        tokens_used=1_140,
        title=prompt_title,
        model="gpt-5.4",
        reasoning_effort="xhigh",
    )
    con.commit()
    con.close()

    alpha_stdout = tmp_path / "alpha.stdout"
    alpha_stderr = tmp_path / "alpha.stderr"
    bravo_stdout = tmp_path / "bravo.stdout"
    bravo_stderr = tmp_path / "bravo.stderr"
    alpha_stdout.write_text("", encoding="utf-8")
    bravo_stdout.write_text("", encoding="utf-8")
    alpha_stderr.write_text("Session ID: launcher-alpha\n", encoding="utf-8")
    bravo_stderr.write_text("Session ID: launcher-bravo\n", encoding="utf-8")

    round_payload = {
        "round_id": "round-1",
        "task_class": "pr_review",
        "selection_mode": "scramble",
        "status": "running",
        "runs": [
            {
                "slot": "alpha",
                "variant_id": "gpt-5.4-xhigh",
                "title": "review-suite::round-1::alpha::gpt-5.4-xhigh",
                "command": ["codex", "review"],
                "started_at": "2023-11-14T22:15:00Z",
                "stdout_path": str(alpha_stdout),
                "stderr_path": str(alpha_stderr),
            },
            {
                "slot": "bravo",
                "variant_id": "gpt-5.3-codex-medium",
                "title": "review-suite::round-1::bravo::gpt-5.3-codex-medium",
                "command": ["codex", "review"],
                "started_at": "2023-11-14T22:15:00Z",
                "stdout_path": str(bravo_stdout),
                "stderr_path": str(bravo_stderr),
            },
        ],
    }

    completed = review_suite_local.collect_round_results(
        round_payload=round_payload,
        roster=_minimal_roster(),
        state_dir=state_dir,
        review_cwd=Path(review_cwd),
        sqlite_path=sqlite_path,
        wait=False,
    )

    runs_by_slot = {run["slot"]: run for run in completed["runs"]}
    assert runs_by_slot["alpha"]["session_id"] == "launcher-alpha"
    assert runs_by_slot["alpha"]["thread_id"] == "child-alpha"
    assert runs_by_slot["alpha"]["tokens_used"] == 1_570
    assert runs_by_slot["alpha"]["usage"] == {
        "input_tokens": 1200,
        "cached_input_tokens": 300,
        "output_tokens": 70,
    }
    assert runs_by_slot["alpha"]["cost_usd"] == pytest.approx(0.003375)
    assert runs_by_slot["alpha"]["reviewer_output"] == "alpha child findings"
    assert (
        runs_by_slot["alpha"]["reviewer_output_ref"]
        == "rollout://child-alpha/gpt-5.4-xhigh"
    )

    assert runs_by_slot["bravo"]["session_id"] == "launcher-bravo"
    assert runs_by_slot["bravo"]["thread_id"] == "child-bravo"
    assert runs_by_slot["bravo"]["tokens_used"] == 1_140
    assert runs_by_slot["bravo"]["usage"] == {
        "input_tokens": 900,
        "cached_input_tokens": 200,
        "output_tokens": 40,
    }
    assert runs_by_slot["bravo"]["cost_usd"] == pytest.approx(0.00182)
    assert runs_by_slot["bravo"]["reviewer_output"] == "bravo child findings"
    assert (
        runs_by_slot["bravo"]["reviewer_output_ref"]
        == "rollout://child-bravo/gpt-5.3-codex-medium"
    )
    assert review_suite_local.total_usage_tokens(runs_by_slot["alpha"]["usage"]) == 970
    assert review_suite_local.total_usage_tokens(runs_by_slot["bravo"]["usage"]) == 740

    record = review_suite_local.build_record_from_grade(
        round_payload=completed,
        roster=_minimal_roster(),
        rubric=_minimal_rubric(),
        task_id="demo",
        rating_pool_id="arena-phase-v1",
        rank_groups=["alpha", "bravo"],
        basis="valid_findings_vs_none",
        shared_note="ok",
    )
    record_runs = {run["variant_id"]: run for run in record["runs"]}
    assert record_runs["gpt-5.4-xhigh"]["cost_usd"] == pytest.approx(0.003375)
    assert record_runs["gpt-5.3-codex-medium"]["cost_usd"] == pytest.approx(0.00182)


def test_rollout_summary_requires_final_answer_phase(tmp_path: Path) -> None:
    rollout = tmp_path / "unfinished.jsonl"
    _write_jsonl(
        rollout,
        [
            {"type": "session_meta", "payload": {"parent_thread_id": "launcher"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "phase": "commentary",
                    "content": [{"text": "Still reviewing."}],
                },
            },
        ],
    )

    assert rollout_capture.rollout_final_assistant_text(rollout) == ""


def test_collect_round_results_rejects_child_lookup_when_launcher_id_is_missing(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    sqlite_path = tmp_path / "state_5.sqlite"
    review_cwd = r"C:\repo\demo"
    prompt_title = "You are reviewing a manually supplied diff artifact."

    child_rollout = tmp_path / "child-only.jsonl"
    _child_rollout(
        child_rollout,
        model="gpt-5.4",
        reasoning_effort="xhigh",
        usage={"input_tokens": 500, "cached_input_tokens": 100, "output_tokens": 25},
        text="child only findings",
        parent_thread_id="unknown-launcher",
    )

    con = sqlite3.connect(sqlite_path)
    con.execute(THREAD_SCHEMA)
    _insert_thread(
        con,
        thread_id="child-only",
        rollout_path=child_rollout,
        cwd=review_cwd,
        source='{"subagent":"review"}',
        created_at=1_700_000_200,
        updated_at=1_700_000_240,
        tokens_used=625,
        title=prompt_title,
        model="gpt-5.4",
        reasoning_effort="xhigh",
    )
    con.commit()
    con.close()

    stdout_path = tmp_path / "stdout.txt"
    stderr_path = tmp_path / "stderr.txt"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("no session marker here\n", encoding="utf-8")

    round_payload = {
        "round_id": "round-2",
        "task_class": "pr_review",
        "selection_mode": "scramble",
        "status": "running",
        "runs": [
            {
                "slot": "alpha",
                "variant_id": "gpt-5.4-xhigh",
                "title": "review-suite::round-2::alpha::gpt-5.4-xhigh",
                "command": ["codex", "review"],
                "started_at": "2023-11-14T22:16:40Z",
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            }
        ],
    }

    completed = review_suite_local.collect_round_results(
        round_payload=round_payload,
        roster={"variants": [_minimal_roster()["variants"][0]]},
        state_dir=state_dir,
        review_cwd=Path(review_cwd),
        sqlite_path=sqlite_path,
        wait=False,
    )

    run = completed["runs"][0]
    assert run["session_id"] is None
    assert run["thread_id"] is None
    assert run["reviewer_output"] == ""
    assert run["review_status"] == "process_died"


def test_collect_round_results_matches_review_child_created_after_default_window(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    sqlite_path = tmp_path / "state_5.sqlite"
    review_cwd = r"C:\repo\demo"
    prompt_title = "Review the code changes against the base branch 'main'."

    launcher_rollout = tmp_path / "launcher.jsonl"
    child_rollout = tmp_path / "child.jsonl"
    _launcher_rollout(launcher_rollout, "launcher text")
    _child_rollout(
        child_rollout,
        model="gpt-5.4",
        reasoning_effort="xhigh",
        usage={"input_tokens": 840, "cached_input_tokens": 240, "output_tokens": 36},
        text="slow child findings",
        parent_thread_id="launcher-alpha",
    )

    con = sqlite3.connect(sqlite_path)
    con.execute(THREAD_SCHEMA)
    _insert_thread(
        con,
        thread_id="launcher-alpha",
        rollout_path=launcher_rollout,
        cwd=review_cwd,
        source="exec",
        created_at=1_700_000_100,
        updated_at=1_700_000_200,
        tokens_used=0,
        title=prompt_title,
    )
    _insert_thread(
        con,
        thread_id="child-alpha",
        rollout_path=child_rollout,
        cwd=review_cwd,
        source='{"subagent":"review"}',
        created_at=1_700_000_112,
        updated_at=1_700_000_190,
        tokens_used=876,
        title=prompt_title,
        model="gpt-5.4",
        reasoning_effort="xhigh",
    )
    con.commit()
    con.close()

    stdout_path = tmp_path / "stdout.txt"
    stderr_path = tmp_path / "stderr.txt"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("Session ID: launcher-alpha\n", encoding="utf-8")

    round_payload = {
        "round_id": "round-slow-child",
        "task_class": "pr_review",
        "selection_mode": "scramble",
        "status": "running",
        "runs": [
            {
                "slot": "alpha",
                "variant_id": "gpt-5.4-xhigh",
                "title": "review-suite::round-slow-child::alpha::gpt-5.4-xhigh",
                "command": ["codex", "review"],
                "started_at": "2023-11-14T22:15:00Z",
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            }
        ],
    }

    completed = review_suite_local.collect_round_results(
        round_payload=round_payload,
        roster={"variants": [_minimal_roster()["variants"][0]]},
        state_dir=state_dir,
        review_cwd=Path(review_cwd),
        sqlite_path=sqlite_path,
        wait=False,
    )

    run = completed["runs"][0]
    assert run["session_id"] == "launcher-alpha"
    assert run["thread_id"] == "child-alpha"
    assert run["tokens_used"] == 876
    assert run["usage"] == {
        "input_tokens": 840,
        "cached_input_tokens": 240,
        "output_tokens": 36,
    }
    assert run["cost_usd"] == pytest.approx(0.0021)
    assert run["reviewer_output"] == "slow child findings"


def test_collect_round_results_retries_child_lookup_after_launcher_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    sqlite_path = tmp_path / "state_5.sqlite"
    review_cwd = r"C:\repo\demo"
    prompt_title = "Review the code changes against the base branch 'main'."

    launcher_rollout = tmp_path / "launcher.jsonl"
    child_rollout = tmp_path / "child.jsonl"
    _launcher_rollout(launcher_rollout, "launcher text")
    _child_rollout(
        child_rollout,
        model="gpt-5.4",
        reasoning_effort="xhigh",
        usage={"input_tokens": 750, "cached_input_tokens": 200, "output_tokens": 30},
        text="delayed child findings",
        parent_thread_id="launcher-alpha",
    )

    launcher_thread = {
        "id": "launcher-alpha",
        "rollout_path": str(launcher_rollout),
        "cwd": review_cwd,
        "source": "exec",
        "created_at": 1_700_000_100,
        "updated_at": 1_700_000_140,
        "tokens_used": 0,
        "model": None,
        "reasoning_effort": None,
        "title": prompt_title,
    }
    child_thread = {
        "id": "child-alpha",
        "rollout_path": str(child_rollout),
        "cwd": review_cwd,
        "source": '{"subagent":"review"}',
        "created_at": 1_700_000_101,
        "updated_at": 1_700_000_150,
        "tokens_used": 780,
        "model": "gpt-5.4",
        "reasoning_effort": "xhigh",
        "title": prompt_title,
    }

    stdout_path = tmp_path / "stdout.txt"
    stderr_path = tmp_path / "stderr.txt"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("Session ID: launcher-alpha\n", encoding="utf-8")

    round_payload = {
        "round_id": "round-3",
        "task_class": "pr_review",
        "selection_mode": "scramble",
        "status": "running",
        "runs": [
            {
                "slot": "alpha",
                "variant_id": "gpt-5.4-xhigh",
                "title": "review-suite::round-3::alpha::gpt-5.4-xhigh",
                "command": ["codex", "review"],
                "started_at": "2023-11-14T22:15:00Z",
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            }
        ],
    }

    child_lookup_calls = {"count": 0}

    monkeypatch.setattr(
        review_suite_local, "find_thread_by_id", lambda **_: launcher_thread
    )

    def delayed_child_lookup(**_: object) -> dict[str, object] | None:
        child_lookup_calls["count"] += 1
        if child_lookup_calls["count"] == 1:
            return None
        return child_thread

    monkeypatch.setattr(
        review_suite_local, "find_review_child_thread", delayed_child_lookup
    )
    monkeypatch.setattr(review_suite_local.time, "sleep", lambda _: None)

    completed = review_suite_local.collect_round_results(
        round_payload=round_payload,
        roster={"variants": [_minimal_roster()["variants"][0]]},
        state_dir=state_dir,
        review_cwd=Path(review_cwd),
        sqlite_path=sqlite_path,
        wait=False,
    )

    run = completed["runs"][0]
    assert child_lookup_calls["count"] >= 2
    assert run["session_id"] == "launcher-alpha"
    assert run["thread_id"] == "child-alpha"
    assert run["tokens_used"] == 780
    assert run["usage"] == {
        "input_tokens": 750,
        "cached_input_tokens": 200,
        "output_tokens": 30,
    }
    assert run["cost_usd"] == pytest.approx(0.001875)
    assert run["reviewer_output"] == "delayed child findings"


def test_core_usage_tokens_exclude_cache_reads_and_fold_reasoning() -> None:
    usage = {
        "input_tokens": 2_417_374,
        "cached_input_tokens": 2_340_864,
        "output_tokens": 23_880,
        "reasoning_output_tokens": 1_200,
    }

    assert review_suite_local.total_usage_tokens(usage) == 101_590
    assert review_suite_core.core_usage_tokens(usage) == 101_590


def test_core_usage_tokens_keep_cache_writes_as_input() -> None:
    usage = {
        "input_tokens": 1_000,
        "cached_input_tokens": 400,
        "cache_write_tokens": 100,
        "output_tokens": 50,
    }

    assert review_suite_core.core_usage_tokens(usage) == 650


def test_run_total_tokens_prefers_usage_then_falls_back_to_tokens_used() -> None:
    assert (
        review_suite_core.run_total_tokens(
            {
                "usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 80,
                    "output_tokens": 5,
                }
            }
        )
        == 25
    )
    assert (
        review_suite_core.run_total_tokens({"usage": {}, "tokens_used": 1_654_249})
        == 1_654_249
    )
    assert review_suite_core.run_total_tokens({}) == 0


def test_run_total_tokens_trusts_zero_core_usage_over_tokens_used() -> None:
    run = {
        "usage": {"input_tokens": 100, "cached_input_tokens": 100, "output_tokens": 0},
        "tokens_used": 5_000,
    }

    assert review_suite_core.run_total_tokens(run) == 0


def test_run_total_tokens_rejects_bool_tokens_used() -> None:
    assert review_suite_core.run_total_tokens({"usage": {}, "tokens_used": True}) == 0


def test_compact_benchmark_run_keeps_int_and_drops_bool_tokens_used() -> None:
    kept = review_suite_local.compact_benchmark_run(
        {"variant_id": "alpha", "tokens_used": 123}
    )
    dropped = review_suite_local.compact_benchmark_run(
        {"variant_id": "alpha", "tokens_used": True}
    )

    assert kept["tokens_used"] == 123
    assert "tokens_used" not in dropped


def test_compute_cost_and_total_tokens_treat_cached_input_as_subset() -> None:
    variant = _minimal_roster()["variants"][0]
    usage = {
        "input_tokens": 2_417_374,
        "cached_input_tokens": 2_340_864,
        "output_tokens": 23_880,
    }

    assert review_suite_local.total_usage_tokens(usage) == 100_390
    assert review_suite_local.compute_cost_usd(variant, usage) == pytest.approx(
        1.134691
    )


def test_compute_cost_splits_cache_write_tokens() -> None:
    variant = {
        "pricing": {
            "input_per_million_usd": 5.0,
            "cached_input_per_million_usd": 0.5,
            "cache_write_input_per_million_usd": 6.25,
            "output_per_million_usd": 30.0,
        }
    }
    usage = {
        "input_tokens": 100,
        "cached_input_tokens": 20,
        "cache_write_tokens": 30,
        "output_tokens": 10,
    }

    assert review_suite_local.compute_cost_usd(variant, usage) == pytest.approx(
        0.000748
    )


def test_default_roster_includes_current_model_pricing() -> None:
    roster_path = Path(__file__).resolve().parents[1] / "references" / "roster.json"
    roster = review_suite_local.load_roster(roster_path)
    index = review_suite_local.variant_index(roster)

    astra_layout = {
        variant["reasoning_effort"]: (variant["task_classes"], variant["state"])
        for variant in index.values()
        if variant["model"] == "gpt-6-astra"
    }
    assert astra_layout == {
        "low": (["phase_review", "pr_review"], "active"),
        "medium": (["phase_review", "pr_review"], "active"),
        "high": (["pr_review"], "active"),
        "xhigh": (["pr_review"], "active"),
        "max": (["pr_review"], "active"),
    }
    assert not any(
        variant["model"].startswith("gpt-5.4")
        for task_class in ("phase_review", "pr_review")
        for variant in review_suite_local.eligible_variants(roster, task_class)
    )

    expected_pricing = {
        "gpt-6-astra": {
            "input_per_million_usd": 10.0,
            "cached_input_per_million_usd": 1.0,
            "cache_write_input_per_million_usd": 12.5,
            "output_per_million_usd": 50.0,
        },
        "gpt-5.6-sol": {
            "input_per_million_usd": 5.0,
            "cached_input_per_million_usd": 0.5,
            "cache_write_input_per_million_usd": 6.25,
            "output_per_million_usd": 30.0,
        },
        "gpt-5.6-terra": {
            "input_per_million_usd": 2.0,
            "cached_input_per_million_usd": 0.2,
            "cache_write_input_per_million_usd": 2.5,
            "output_per_million_usd": 12.0,
        },
        "gpt-5.6-luna": {
            "input_per_million_usd": 0.2,
            "cached_input_per_million_usd": 0.02,
            "cache_write_input_per_million_usd": 0.25,
            "output_per_million_usd": 1.2,
        },
    }
    for variant in index.values():
        if variant["model"] in expected_pricing:
            assert variant["pricing"] == expected_pricing[variant["model"]]


def test_opencode_arena_candidates_are_registered(tmp_path: Path) -> None:
    roster_path = Path(__file__).resolve().parents[1] / "references" / "roster.json"
    roster = review_suite_local.load_roster(roster_path)
    index = review_suite_local.variant_index(roster)

    opencode_layout = {
        variant["id"]: (
            variant["model"],
            variant["reasoning_effort"],
            variant["task_classes"],
            variant["state"],
        )
        for variant in index.values()
        if variant.get("model", "").startswith("opencode::opencode-go/")
    }
    assert opencode_layout == {
        "deepseek-v4.1-flash-low": (
            "opencode::opencode-go/deepseek-v4.1-flash",
            "low",
            ["phase_review"],
            "active",
        ),
        "deepseek-v4.1-flash-high": (
            "opencode::opencode-go/deepseek-v4.1-flash",
            "high",
            ["phase_review", "pr_review"],
            "active",
        ),
        "deepseek-v4.1-flash-max": (
            "opencode::opencode-go/deepseek-v4.1-flash",
            "max",
            ["phase_review", "pr_review"],
            "active",
        ),
        "glm-5.3-flash-low": (
            "opencode::opencode-go/glm-5.3-flash",
            "low",
            ["phase_review"],
            "active",
        ),
        "glm-5.3-flash-high": (
            "opencode::opencode-go/glm-5.3-flash",
            "high",
            ["phase_review", "pr_review"],
            "active",
        ),
        "glm-5.3-flash-max": (
            "opencode::opencode-go/glm-5.3-flash",
            "max",
            ["phase_review", "pr_review"],
            "active",
        ),
    }

    config = review_suite_core.load_config(tmp_path / "state")
    phase_ids = config["arena"]["pools"]["arena_phase"]["variant_ids"]
    deep_ids = config["arena"]["pools"]["arena_deep"]["variant_ids"]
    for variant_id in phase_ids + deep_ids:
        assert variant_id in index
    assert set(opencode_layout) <= set(phase_ids)
    assert {
        "deepseek-v4.1-flash-high",
        "deepseek-v4.1-flash-max",
        "glm-5.3-flash-high",
        "glm-5.3-flash-max",
    } <= set(deep_ids)
    assert "deepseek-v4.1-flash-low" not in deep_ids
    assert "glm-5.3-flash-low" not in deep_ids


def test_collect_round_results_preserves_foreign_review_cwd_for_capture(
    monkeypatch, tmp_path: Path
) -> None:
    class ForeignReviewCwd:
        def __str__(self) -> str:
            return r"C:\repo\demo"

        def resolve(self) -> Path:
            raise AssertionError(
                "foreign review cwd should not be resolved during capture lookup"
            )

    state_dir = tmp_path / "state"
    stdout_path = tmp_path / "stdout.txt"
    stderr_path = tmp_path / "stderr.txt"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("Session ID: launcher-alpha\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_collect(**kwargs: object) -> dict[str, object]:
        captured["review_cwd"] = kwargs["review_cwd"]
        return {
            "slot": "alpha",
            "variant_id": "gpt-5.4-xhigh",
            "status": "completed",
            "reviewer_output": "No findings.",
        }

    monkeypatch.setattr(
        review_suite_local, "_collect_completed_run_from_artifacts", fake_collect
    )
    monkeypatch.setattr(review_suite_local, "_cleanup_run_artifacts", lambda item: None)

    completed = review_suite_local.collect_round_results(
        round_payload={
            "round_id": "round-foreign-cwd",
            "task_class": "pr_review",
            "selection_mode": "scramble",
            "status": "running",
            "runs": [
                {
                    "slot": "alpha",
                    "variant_id": "gpt-5.4-xhigh",
                    "title": "review-suite::round-foreign-cwd::alpha::gpt-5.4-xhigh",
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                }
            ],
        },
        roster={"variants": [_minimal_roster()["variants"][0]]},
        state_dir=state_dir,
        review_cwd=ForeignReviewCwd(),  # type: ignore[arg-type]
        wait=False,
    )

    assert completed["status"] == "completed"
    assert str(captured["review_cwd"]) == r"C:\repo\demo"


def test_collect_completed_review_capture_does_not_use_title_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stdout_path = tmp_path / "review.stdout.txt"
    stderr_path = tmp_path / "review.stderr.txt"
    stdout_path.write_text("No findings.\n", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(review_suite_local, "find_thread_by_id", lambda **_: None)
    monkeypatch.setattr(
        review_suite_local, "find_review_child_thread", lambda **_: None
    )
    monkeypatch.setattr(review_suite_local.time, "sleep", lambda _: None)

    monkeypatch.setattr(review_suite_local, "enrich_thread_record", lambda thread: {})

    variant = {
        "id": "gpt-5.5-medium",
        "model": "gpt-5.5",
        "reasoning_effort": "medium",
        "pricing": {
            "input_per_million_usd": 5.0,
            "cached_input_per_million_usd": 0.5,
            "output_per_million_usd": 30.0,
        },
    }
    started_at = "2026-04-13T10:00:00Z"
    review_suite_local.collect_completed_review_capture(
        slot="review-1",
        variant_id="gpt-5.5-medium",
        variant=variant,
        title="local-review::repo::review-1",
        command=["codex", "review"],
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        started_at=started_at,
        sqlite_path=tmp_path / "state_5.sqlite",
        review_cwd=tmp_path,
    )

    assert started_at


def test_collect_completed_review_capture_uses_launcher_parent_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stdout_path = tmp_path / "review.stdout.txt"
    stderr_path = tmp_path / "review.stderr.txt"
    stdout_path.write_text("No findings.\n", encoding="utf-8")
    stderr_path.write_text("Session ID: launcher-alpha\n", encoding="utf-8")
    observed: dict[str, object] = {}

    monkeypatch.setattr(review_suite_local, "find_thread_by_id", lambda **_: None)
    monkeypatch.setattr(review_suite_local.time, "sleep", lambda _: None)

    def fake_find_review_child_thread(**kwargs: object) -> None:
        observed.setdefault("parent_thread_id", kwargs["parent_thread_id"])
        return None

    monkeypatch.setattr(
        review_suite_local, "find_review_child_thread", fake_find_review_child_thread
    )
    monkeypatch.setattr(review_suite_local, "enrich_thread_record", lambda thread: {})

    review_suite_local.collect_completed_review_capture(
        slot="review-1",
        variant_id="gpt-5.5-max",
        variant={
            "model": "gpt-5.5",
            "reasoning_effort": "max",
            "effective_reasoning_effort": "xhigh",
        },
        title="local-review::repo::review-1",
        command=["codex", "review"],
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        started_at="2026-04-13T10:00:00Z",
        sqlite_path=tmp_path / "state_5.sqlite",
        review_cwd=tmp_path,
    )

    assert observed["parent_thread_id"] == "launcher-alpha"


def test_collect_completed_review_capture_prefers_final_message_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stdout_path = tmp_path / "review.stdout.txt"
    stderr_path = tmp_path / "review.stderr.txt"
    final_path = tmp_path / "review.final.txt"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    final_path.write_text("No findings from output file.\n", encoding="utf-8")

    monkeypatch.setattr(review_suite_local, "find_thread_by_id", lambda **_: None)
    monkeypatch.setattr(
        review_suite_local, "find_review_child_thread", lambda **_: None
    )
    monkeypatch.setattr(review_suite_local, "enrich_thread_record", lambda thread: {})
    monkeypatch.setattr(review_suite_local.time, "sleep", lambda _: None)

    variant = {
        "id": "gpt-5.5-medium",
        "model": "gpt-5.5",
        "reasoning_effort": "medium",
        "pricing": {
            "input_per_million_usd": 5.0,
            "cached_input_per_million_usd": 0.5,
            "output_per_million_usd": 30.0,
        },
    }

    capture = review_suite_local.collect_completed_review_capture(
        slot="review-1",
        variant_id="gpt-5.5-medium",
        variant=variant,
        title="local-review::repo::review-1",
        command=["codex", "exec"],
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        started_at="2026-04-13T10:00:00Z",
        sqlite_path=tmp_path / "state_5.sqlite",
        review_cwd=tmp_path,
        final_message_path=final_path,
    )

    assert capture["review_status"] == "completed"
    assert capture["reviewer_output"] == "No findings from output file."


def test_collect_completed_review_capture_uses_opencode_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stdout_path = tmp_path / "review.stdout.txt"
    stderr_path = tmp_path / "review.stderr.txt"
    stdout_path.write_text("Review result: clean\n", encoding="utf-8")
    metadata = {
        "session_id": "ses_opencode",
        "usage": {
            "input_tokens": 1500,
            "cached_input_tokens": 1200,
            "output_tokens": 60,
            "total_tokens": 1560,
        },
        "cost_usd": 0.0123,
    }
    stderr_path.write_text(
        REVIEW_METADATA_PREFIX + json.dumps(metadata) + "\n", encoding="utf-8"
    )

    def fail_codex_lookup(**_kwargs: object) -> None:
        raise AssertionError("OpenCode reviews must not query Codex threads")

    monkeypatch.setattr(review_suite_local, "find_thread_by_id", fail_codex_lookup)
    monkeypatch.setattr(
        review_suite_local, "find_review_child_thread", fail_codex_lookup
    )

    capture = review_suite_local.collect_completed_review_capture(
        slot="review-1",
        variant_id="deepseek-flash-medium",
        variant={
            "id": "deepseek-flash-medium",
            "model": "opencode::opencode-go/deepseek-v4.1-flash",
            "reasoning_effort": "medium",
        },
        title="review-suite::round::review-1",
        command=["python", "opencode_driver.py"],
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        started_at=None,
        sqlite_path=tmp_path / "state_5.sqlite",
        review_cwd=tmp_path,
    )

    assert capture["review_status"] == "completed"
    assert capture["session_id"] == "ses_opencode"
    assert capture["thread_id"] is None
    assert capture["usage"]["cached_input_tokens"] == 1200
    assert capture["tokens_used"] == 1560
    assert capture["cost_usd"] == pytest.approx(0.0123)
    assert capture["reviewer_output"] == "Review result: clean"
    assert capture["cooldown_eligible"] is False


def test_collect_completed_review_capture_uses_claude_subscription_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stdout_path = tmp_path / "review.stdout.txt"
    stderr_path = tmp_path / "review.stderr.txt"
    stdout_path.write_text("Review result: clean\n", encoding="utf-8")
    metadata = {
        "session_id": "claude-session",
        "cost_usd": 0.41,
        "actual_model": "claude-opus-5-5",
        "actual_models": ["claude-opus-5-5"],
        "usage": {
            "input_tokens": 40,
            "cached_input_tokens": 20,
            "output_tokens": 10,
            "total_tokens": 50,
        },
    }
    stderr_path.write_text(
        "[review-suite] claude-metadata: " + json.dumps(metadata) + "\n",
        encoding="utf-8",
    )

    def fail_codex_lookup(**_kwargs: object) -> None:
        raise AssertionError("Claude reviews must not query Codex threads")

    monkeypatch.setattr(review_suite_local, "find_thread_by_id", fail_codex_lookup)
    monkeypatch.setattr(
        review_suite_local, "find_review_child_thread", fail_codex_lookup
    )
    capture = review_suite_local.collect_completed_review_capture(
        slot="review-1",
        variant_id="claude-opus-5.5-medium",
        variant={
            "id": "claude-opus-5.5-medium",
            "model": "claude::claude-opus-5-5",
            "reasoning_effort": "medium",
            "state": "disabled",
        },
        title="review-suite::round::review-1",
        command=["python", "claude_driver.py"],
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        started_at=None,
        sqlite_path=tmp_path / "state.sqlite",
        review_cwd=tmp_path,
    )
    assert capture["review_status"] == "completed"
    assert capture["actual_model"] == "claude-opus-5-5"
    assert capture["usage"]["cached_input_tokens"] == 20
    assert capture["cost_usd"] == 0.41
    assert (
        review_suite_local.compact_benchmark_run(capture)["actual_model"]
        == "claude-opus-5-5"
    )


def test_collect_completed_review_capture_marks_opencode_failure_cooldown_eligible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stdout_path = tmp_path / "review.stdout.txt"
    stderr_path = tmp_path / "review.stderr.txt"
    stdout_path.write_text("", encoding="utf-8")
    metadata = {
        "session_id": "ses_opencode",
        "usage": {},
        "error_class": "capacity",
        "error_name": "ProviderError",
        "error_message": "rate limit exceeded",
    }
    stderr_path.write_text(
        REVIEW_METADATA_PREFIX + json.dumps(metadata) + "\n", encoding="utf-8"
    )

    def fail_codex_lookup(**_kwargs: object) -> None:
        raise AssertionError("OpenCode reviews must not query Codex threads")

    monkeypatch.setattr(review_suite_local, "find_thread_by_id", fail_codex_lookup)
    monkeypatch.setattr(
        review_suite_local, "find_review_child_thread", fail_codex_lookup
    )

    capture = review_suite_local.collect_completed_review_capture(
        slot="review-1",
        variant_id="deepseek-v4.1-flash-low",
        variant={
            "id": "deepseek-v4.1-flash-low",
            "model": "opencode::opencode-go/deepseek-v4.1-flash",
            "reasoning_effort": "low",
        },
        title="review-suite::round::review-1",
        command=["python", "opencode_driver.py"],
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        started_at=None,
        sqlite_path=tmp_path / "state_5.sqlite",
        review_cwd=tmp_path,
    )

    assert capture["review_status"] == "interrupted_capacity"
    assert capture["grade_blocked"] is True
    assert capture["grade_block_reason"] == "selected_model_at_capacity"
    assert capture["cooldown_eligible"] is True


def test_collect_completed_review_capture_maps_unsupported_model_error(
    tmp_path: Path,
) -> None:
    rollout_path = tmp_path / "unsupported-model.jsonl"
    _write_jsonl(
        rollout_path,
        [
            {"type": "session_meta", "payload": {}},
            {
                "type": "event_msg",
                "payload": {
                    "type": "task_complete",
                    "error": {
                        "message": json.dumps(
                            {
                                "detail": "The 'gpt-example' model is not supported when using Codex with a ChatGPT account."
                            }
                        )
                    },
                },
            },
        ],
    )
    sqlite_path = tmp_path / "state_5.sqlite"
    con = sqlite3.connect(sqlite_path)
    con.execute(THREAD_SCHEMA)
    _insert_thread(
        con,
        thread_id="unsupported-model",
        rollout_path=rollout_path,
        cwd=str(tmp_path),
        source=rollout_capture.REVIEW_SUBAGENT_SOURCE,
        created_at=1,
        updated_at=1,
        tokens_used=0,
        title="unsupported model",
    )
    con.commit()
    con.close()
    stdout_path = tmp_path / "review.stdout.txt"
    stderr_path = tmp_path / "review.stderr.txt"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("Session ID: unsupported-model\n", encoding="utf-8")

    capture = review_suite_local.collect_completed_review_capture(
        slot="alpha",
        variant_id="gpt-5.4-xhigh",
        variant=_minimal_roster()["variants"][0],
        title="unsupported model",
        command=["codex", "exec"],
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        started_at=None,
        sqlite_path=sqlite_path,
        review_cwd=tmp_path,
    )

    assert capture["review_status"] == "unsupported_model"
    assert capture["grade_block_reason"] == "selected_model_not_supported"


def test_rollout_capture_missing_threads_table_returns_none(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "state_5.sqlite"
    sqlite3.connect(sqlite_path).close()

    assert (
        rollout_capture.find_thread_by_id(sqlite_path=sqlite_path, thread_id="thread-1")
        is None
    )
    assert (
        rollout_capture.find_thread_by_title(sqlite_path=sqlite_path, title="review")
        is None
    )
    assert (
        rollout_capture.find_review_child_thread(
            sqlite_path=sqlite_path,
            parent_thread_id="launcher",
        )
        is None
    )


def test_review_child_lookup_skips_malformed_rollout(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "state_5.sqlite"
    malformed = tmp_path / "malformed.jsonl"
    non_object = tmp_path / "non-object.jsonl"
    matched = tmp_path / "matched.jsonl"
    malformed.write_text("{not-json\n", encoding="utf-8")
    non_object.write_text("null\n", encoding="utf-8")
    _write_jsonl(
        matched,
        [
            {
                "type": "session_meta",
                "payload": {"parent_thread_id": "launcher-alpha"},
            }
        ],
    )
    con = sqlite3.connect(sqlite_path)
    con.execute(THREAD_SCHEMA)
    _insert_thread(
        con,
        thread_id="malformed",
        rollout_path=malformed,
        cwd="repo",
        source='{"subagent":"review"}',
        created_at=2,
        updated_at=2,
        tokens_used=0,
        title="review",
    )
    _insert_thread(
        con,
        thread_id="non-object",
        rollout_path=non_object,
        cwd="repo",
        source='{"subagent":"review"}',
        created_at=3,
        updated_at=3,
        tokens_used=0,
        title="review",
    )
    _insert_thread(
        con,
        thread_id="matched",
        rollout_path=matched,
        cwd="repo",
        source='{"subagent":"review"}',
        created_at=1,
        updated_at=1,
        tokens_used=0,
        title="review",
    )
    con.commit()
    con.close()

    child = rollout_capture.find_review_child_thread(
        sqlite_path=sqlite_path, parent_thread_id="launcher-alpha"
    )

    assert child and child["id"] == "matched"
