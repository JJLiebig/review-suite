from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import review_costs
from review_costs import (
    ReviewCostRow,
    _cheap_cwd_key,
    _current_pr_number,
    _cwd_query_candidates,
    _cost_row_from_payload,
    _metadata_for_cwd,
    _price_from_usage,
    _read_rollout_model_metadata,
    _usage_from_rollout_line,
    _record_lane,
    _repo_from_worktree_folder,
    _thread_rows,
    apply_review_cost_delta_best_effort,
    collect_review_cost_rows,
    format_compact_number,
    read_review_cost_row_for_cwd,
    read_review_cost_row_cache,
    render_review_cost_markdown,
    update_review_cost_row_cache,
)


def test_cost_cwd_matching_understands_wsl_unc_and_native(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")

    canonical = "wsl:ubuntu:/home/alice/code/repo"

    assert _cheap_cwd_key(
        r"\\wsl.localhost\Ubuntu\home\alice\code\repo"
    ) == _cheap_cwd_key("/home/alice/code/repo")
    assert _cheap_cwd_key("/home/alice/code/repo") == canonical
    assert "/home/alice/code/repo" in _cwd_query_candidates({canonical})
    assert "//wsl.localhost/ubuntu/home/alice/code/repo" in _cwd_query_candidates(
        {canonical}
    )


def test_repo_from_worktree_folder_folds_wt_suffixes() -> None:
    assert _repo_from_worktree_folder("sample-web-wt-alpha") == "sample-web"
    assert _repo_from_worktree_folder("sample-web") == "sample-web"


def test_repo_from_worktree_folder_applies_explicit_overrides() -> None:
    assert (
        _repo_from_worktree_folder("sample-stack-allow-chat-skipped-default")
        == "sample-stack"
    )


def test_record_lane_accepts_followup_spellings() -> None:
    assert (
        _record_lane({"public_task": "review-followup", "task_class": "phase_review"})
        == "review_followup"
    )
    assert (
        _record_lane({"public_task": "review_followup", "task_class": "phase_review"})
        == "review_followup"
    )


def test_price_from_usage_splits_cache_write_tokens() -> None:
    usage = {
        "input_tokens": 100,
        "input_tokens_details": {"cached_tokens": 20, "cache_write_tokens": 30},
        "output_tokens": 10,
    }

    assert _price_from_usage("gpt-5.6-sol", usage) == pytest.approx(0.0007475)
    assert _price_from_usage("gpt-6-astra", usage) == pytest.approx(0.001395)
    assert _price_from_usage("gpt-5.4-mini", usage) == pytest.approx(0.0001065)


def test_prelaunch_usage_does_not_inherit_another_models_price() -> None:
    from review_suite_local import compute_cost_usd, load_roster

    model = "gpt-6-astra-minor"
    usage = {"input_tokens": 100, "output_tokens": 10}
    roster = load_roster(SCRIPT_DIR.parent / "references" / "roster.json")
    variant = next(v for v in roster["variants"] if v["model"] == model)

    assert compute_cost_usd(variant, usage) is None
    assert _price_from_usage(model, usage) is None
    assert review_costs._price_from_total_tokens(model, 110) is None


@pytest.mark.parametrize(
    "model,expected", [("gpt-6-sol", 0.000279), ("gpt-6-luna", 0.00001395)]
)
def test_gpt6_usage_prices_match_roster_and_ledger(model: str, expected: float) -> None:
    from review_suite_local import compute_cost_usd, load_roster

    usage = {
        "input_tokens": 100,
        "cached_input_tokens": 20,
        "cache_write_tokens": 30,
        "output_tokens": 10,
    }
    roster = load_roster(SCRIPT_DIR.parent / "references" / "roster.json")
    variant = next(v for v in roster["variants"] if v["model"] == model)
    assert compute_cost_usd(variant, usage) == pytest.approx(expected)
    assert _price_from_usage(model, usage) == pytest.approx(expected)


def test_rollout_usage_line_preserves_cache_write_tokens() -> None:
    line = json.dumps(
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": 100,
                        "input_tokens_details": {
                            "cached_tokens": 20,
                            "cache_write_tokens": 30,
                        },
                        "output_tokens": 10,
                    }
                },
            },
        }
    )

    assert _usage_from_rollout_line(line) == {
        "input_tokens": 100,
        "cached_input_tokens": 20,
        "cache_write_tokens": 30,
        "output_tokens": 10,
        "reasoning_output_tokens": 0,
    }


def test_metadata_for_missing_wsl_worktree_folds_into_parent_repo() -> None:
    metadata = _metadata_for_cwd(
        r"\\wsl.localhost\Ubuntu\home\alice\code\sample-web-wt-budget"
    )

    assert metadata == {
        "repo": "sample-web",
        "folder": "sample-web-wt-budget",
        "branch": "-",
        "pr_number": "-",
    }


def test_metadata_for_inaccessible_worktree_uses_fallback(monkeypatch) -> None:
    class InaccessiblePath:
        name = "sample-web-wt-budget"

        def exists(self) -> bool:
            raise OSError("unreachable")

    def inaccessible_path(normalized_cwd: str) -> InaccessiblePath:
        return InaccessiblePath()

    monkeypatch.setattr(review_costs, "cwd_path_from_normalized", inaccessible_path)

    metadata = _metadata_for_cwd("C:/work/sample-web-wt-budget")

    assert metadata == {
        "repo": "sample-web",
        "folder": "sample-web-wt-budget",
        "branch": "-",
        "pr_number": "-",
    }


def test_metadata_for_unusable_worktree_cwd_uses_fallback(monkeypatch) -> None:
    class UnusableCwdPath:
        name = "sample-web-wt-budget"

        def exists(self) -> bool:
            return True

        def __str__(self) -> str:
            return "C:/work/sample-web-wt-budget"

    def unusable_cwd_path(normalized_cwd: str) -> UnusableCwdPath:
        return UnusableCwdPath()

    def raise_os_error(*args: object, **kwargs: object) -> object:
        raise OSError("unusable cwd")

    monkeypatch.setattr(review_costs, "cwd_path_from_normalized", unusable_cwd_path)
    monkeypatch.setattr(review_costs.subprocess, "run", raise_os_error)

    metadata = _metadata_for_cwd("C:/work/sample-web-wt-budget")

    assert metadata == {
        "repo": "sample-web",
        "folder": "sample-web-wt-budget",
        "branch": "-",
        "pr_number": "-",
    }


def test_metadata_preserves_posix_folder_backslashes(monkeypatch) -> None:
    class BackslashFolderPath:
        name = r"sample\api-wt-budget"

        def exists(self) -> bool:
            return False

        def __str__(self) -> str:
            return r"/tmp/sample\api-wt-budget"

    def backslash_folder_path(normalized_cwd: str) -> BackslashFolderPath:
        return BackslashFolderPath()

    monkeypatch.setattr(review_costs, "cwd_path_from_normalized", backslash_folder_path)

    metadata = _metadata_for_cwd(r"/tmp/sample\api-wt-budget")

    assert metadata["folder"] == r"sample\api-wt-budget"


def test_cached_row_with_folder_repo_folds_wt_suffix() -> None:
    row = _cost_row_from_payload(
        {
            "repo": "sample-web-wt-budget",
            "folder": "sample-web-wt-budget",
            "branch": "-",
            "pr_number": "-",
            "worker_model": "-",
            "implementation_tokens": 0,
            "implementation_cost_usd": 0.0,
            "latest_review": "2026-04-27T00:00:00Z",
            "lane_sessions": {
                "review_t1": 0,
                "review_t3": 1,
                "review_followup": 0,
            },
            "review_seconds": 60,
            "tokens": 1000,
            "cost_usd": 0.01,
        }
    )

    assert row is not None
    assert row.repo == "sample-web"
    assert row.folder == "sample-web-wt-budget"


def _write_round(state_dir: Path, payload: dict[str, object]) -> None:
    rounds = state_dir / "rounds"
    rounds.mkdir(parents=True, exist_ok=True)
    if payload.get("status") == "completed":
        for run in payload.get("runs", []):
            if isinstance(run, dict):
                run.setdefault("review_status", "completed")
    (rounds / f"{payload['round_id']}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _write_orchestrator_round(state_dir: Path, payload: dict[str, object]) -> None:
    rounds = state_dir / "orchestrator" / "review-rounds" / "rounds"
    rounds.mkdir(parents=True, exist_ok=True)
    if payload.get("status") == "completed":
        for run in payload.get("runs", []):
            if isinstance(run, dict):
                run.setdefault("review_status", "completed")
    (rounds / f"{payload['round_id']}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _write_codex_thread(
    codex_home: Path,
    *,
    thread_id: str,
    cwd: Path,
    model: str,
    reasoning_effort: str,
    tokens_used: int = 0,
    title: str = "Implement the change",
    agent_role: str = "default",
    git_branch: str | None = None,
    rollout_usage: dict[str, int] | None = None,
    rollout_model: str | None = None,
    rollout_reasoning_effort: str | None = None,
) -> None:
    codex_home.mkdir(parents=True, exist_ok=True)
    rollout_path = None
    if rollout_usage is not None:
        rollout = codex_home / f"{thread_id}.jsonl"
        lines = [
            json.dumps(
                {
                    "type": "turn_context",
                    "payload": {
                        "model": rollout_model if rollout_model is not None else model,
                        "effort": rollout_reasoning_effort
                        if rollout_reasoning_effort is not None
                        else reasoning_effort,
                    },
                }
            ),
            json.dumps(
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "token_count",
                        "info": {"total_token_usage": rollout_usage},
                    },
                }
            ),
        ]
        rollout.write_text("\n".join(lines) + "\n", encoding="utf-8")
        rollout_path = str(rollout)
    with sqlite3.connect(codex_home / "state_5.sqlite") as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS threads (
                id TEXT,
                rollout_path TEXT,
                created_at TEXT,
                updated_at TEXT,
                source TEXT,
                model TEXT,
                reasoning_effort TEXT,
                cwd TEXT,
                title TEXT,
                tokens_used INTEGER,
                agent_role TEXT,
                git_branch TEXT
            )
            """
        )
        db.execute(
            """
            INSERT INTO threads (
                id, rollout_path, created_at, updated_at, source, model, reasoning_effort, cwd, title, tokens_used, agent_role, git_branch
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                thread_id,
                rollout_path,
                "2026-04-27T09:00:00Z",
                "2026-04-27T09:10:00Z",
                "cli",
                model,
                reasoning_effort,
                str(cwd),
                title,
                tokens_used,
                agent_role,
                git_branch,
            ),
        )


def test_collect_review_cost_rows_groups_local_reviews_by_worktree(
    monkeypatch, tmp_path: Path
) -> None:
    state_dir = tmp_path / "state"
    repo = tmp_path / "sample-api-wt-alpha"
    repo.mkdir()
    normalized = str(repo)
    monkeypatch.setattr(
        "review_costs._metadata_for_cwd",
        lambda value: {
            "repo": "sample-api",
            "folder": "sample-api-wt-alpha",
            "branch": "feat/cost-ledger",
            "pr_number": "123",
        },
    )
    _write_round(
        state_dir,
        {
            "round_id": "t1-round",
            "task_class": "phase_review",
            "graded_task_id": "feat/cost-ledger",
            "status": "completed",
            "sampled_at": "2026-04-27T10:00:00Z",
            "review_completed_at": "2026-04-27T10:02:00Z",
            "review_cwd_normalized": normalized,
            "caller_id": "019dd132-4116-71f3-b013-beaa9e5e95bd",
            "runs": [
                {
                    "slot": "alpha",
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                    "cost_usd": 0.001,
                },
                {
                    "slot": "bravo",
                    "usage": {"input_tokens": 200, "output_tokens": 30},
                    "cost_usd": 0.002,
                },
            ],
        },
    )
    _write_round(
        state_dir,
        {
            "round_id": "t3-round",
            "task_class": "pr_review",
            "graded_task_id": "feat/cost-ledger",
            "status": "completed",
            "sampled_at": "2026-04-27T11:00:00Z",
            "review_completed_at": "2026-04-27T11:04:00Z",
            "review_cwd_normalized": normalized,
            "runs": [
                {
                    "slot": "alpha",
                    "usage": {"input_tokens": 300, "output_tokens": 40},
                    "cost_usd": 0.003,
                },
            ],
        },
    )
    codex_home = tmp_path / "codex-home"
    _write_codex_thread(
        codex_home,
        thread_id="impl-1",
        cwd=repo,
        model="GPT 5.5",
        reasoning_effort="xhigh",
        git_branch="feat/cost-ledger",
        rollout_usage={
            "input_tokens": 1000,
            "cached_input_tokens": 800,
            "output_tokens": 100,
            "reasoning_output_tokens": 0,
            "total_tokens": 1100,
        },
    )
    _write_codex_thread(
        codex_home,
        thread_id="review-1",
        cwd=repo,
        model="gpt-5.5",
        reasoning_effort="xhigh",
        tokens_used=999,
        title="PR-scope review for branch",
        agent_role="review",
    )

    rows = collect_review_cost_rows(
        state_dir=state_dir, review_cwd=repo, codex_home=codex_home
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.repo == "sample-api"
    assert row.branch == "feat/cost-ledger"
    assert row.pr_number == "123"
    assert row.worker_model == "gpt-5.5 xhigh"
    assert row.caller_threads == ("019dd132-4116-71f3-b013-beaa9e5e95bd",)
    assert row.implementation_tokens == 1100
    assert row.implementation_cost_usd == 0.0044
    assert row.lane_sessions == {
        "review_t1": 2,
        "review_t3": 1,
        "review_followup": 0,
    }
    assert row.review_seconds == 360
    assert row.tokens == 690
    assert row.cost_usd == 0.006
    assert row.review_cost_cutover_at == ""


def test_collect_review_cost_rows_includes_completed_ungraded_local_rounds(
    monkeypatch, tmp_path: Path
) -> None:
    state_dir = tmp_path / "state"
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(
        "review_costs._metadata_for_cwd",
        lambda value: {
            "repo": "repo",
            "folder": "repo",
            "branch": "feat/current",
            "pr_number": "-",
        },
    )
    _write_round(
        state_dir,
        {
            "round_id": "ungraded-t1",
            "task_class": "phase_review",
            "status": "completed",
            "sampled_at": "2026-04-27T10:00:00Z",
            "review_completed_at": "2026-04-27T10:01:00Z",
            "review_cwd_normalized": str(repo),
            "runs": [
                {
                    "slot": "alpha",
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                    "cost_usd": 0.001,
                }
            ],
        },
    )

    rows = collect_review_cost_rows(
        state_dir=state_dir, review_cwd=repo, codex_home=tmp_path / "codex-home"
    )

    assert len(rows) == 1
    assert rows[0].lane_sessions["review_t1"] == 1
    assert rows[0].tokens == 120


def test_collect_review_cost_rows_includes_orchestrator_rounds(
    monkeypatch, tmp_path: Path
) -> None:
    state_dir = tmp_path / "state"
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(
        "review_costs._metadata_for_cwd",
        lambda value: {
            "repo": "repo",
            "folder": "repo",
            "branch": "feat/current",
            "pr_number": "-",
        },
    )
    _write_orchestrator_round(
        state_dir,
        {
            "round_id": "orc-review",
            "task_class": "phase_review",
            "public_task": "review_t1",
            "task_id_hint": "feat/current",
            "status": "completed",
            "sampled_at": "2026-04-27T10:00:00Z",
            "review_completed_at": "2026-04-27T10:02:00Z",
            "review_cwd_normalized": str(repo),
            "runs": [
                {
                    "slot": "alpha",
                    "model": "gpt-5.4",
                    "reasoning_effort": "medium",
                    "usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 80,
                        "output_tokens": 20,
                    },
                    "cost_usd": None,
                },
                {
                    "slot": "bravo",
                    "variant_id": "gpt-5.5-medium",
                    "usage": {"input_tokens": 40, "output_tokens": 10},
                    "cost_usd": None,
                },
            ],
        },
    )
    _write_orchestrator_round(
        state_dir,
        {
            "round_id": "orc-followup",
            "task_class": "phase_review",
            "public_task": "review-followup",
            "task_id_hint": "feat/current",
            "status": "completed",
            "sampled_at": "2026-04-27T10:03:00Z",
            "review_completed_at": "2026-04-27T10:04:00Z",
            "review_cwd_normalized": str(repo),
            "runs": [
                {
                    "slot": "alpha",
                    "variant_id": "gpt-5.5-medium-fast",
                    "usage": {"input_tokens": 20, "output_tokens": 5},
                    "cost_usd": None,
                }
            ],
        },
    )

    rows = collect_review_cost_rows(
        state_dir=state_dir, review_cwd=repo, codex_home=tmp_path / "codex-home"
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.lane_sessions["review_t1"] == 2
    assert row.lane_sessions["review_followup"] == 1
    assert row.tokens == 115
    assert row.cost_usd == 0.00112


def test_render_review_cost_markdown_groups_by_repo(
    monkeypatch, tmp_path: Path
) -> None:
    state_dir = tmp_path / "state"
    repo = tmp_path / "sample-api-wt-alpha"
    repo.mkdir()
    monkeypatch.setattr(
        "review_costs._metadata_for_cwd",
        lambda value: {
            "repo": "sample-api",
            "folder": "sample-api-wt-alpha",
            "branch": "feat/cost-ledger",
            "pr_number": "123",
        },
    )
    _write_round(
        state_dir,
        {
            "round_id": "t1-round",
            "task_class": "phase_review",
            "graded_task_id": "feat/cost-ledger",
            "status": "completed",
            "sampled_at": "2026-04-27T10:00:00Z",
            "review_completed_at": "2026-04-27T10:01:00Z",
            "review_cwd_normalized": str(repo),
            "runs": [
                {
                    "slot": "alpha",
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                    "cost_usd": 0.001,
                }
            ],
        },
    )

    codex_home = tmp_path / "codex-home"
    _write_codex_thread(
        codex_home,
        thread_id="impl-1",
        cwd=repo,
        model="GPT 5.4",
        reasoning_effort="medium",
        tokens_used=1000,
        git_branch="feat/cost-ledger",
    )

    markdown = render_review_cost_markdown(
        collect_review_cost_rows(
            state_dir=state_dir, review_cwd=repo, codex_home=codex_home
        )
    )

    assert "# sample-api" in markdown
    assert (
        "| Date | Folder | Branch | PR | Worker Model | Impl Tokens | Impl Cost | T1 | T3 | FU | Review Time | Review Tokens | Review Cost | Total Cost |"
        in markdown
    )
    assert (
        "| 2026-04-27 | sample-api-wt-alpha | feat/cost-ledger | 123 | gpt-5.4 medium | 1k | $0.00 | 1 | 0 | 0 | 1m 00s | 120 | $0.00 | $0.01 |"
        in markdown
    )


def test_format_compact_number_keeps_cells_short() -> None:
    assert format_compact_number(999) == "999"
    assert format_compact_number(1_250) == "1.2k"
    assert format_compact_number(12_500) == "12.5k"
    assert format_compact_number(125_000) == "125k"
    assert format_compact_number(1_500_000) == "1.5m"
    assert format_compact_number(100_000_000) == "100m"


def test_automatic_cost_delta_does_zero_cost_work_when_arena_is_disabled(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "review_costs.load_config", lambda state_dir: {"arena": {"enabled": False}}
    )
    monkeypatch.setattr(
        "review_costs.orchestrator_store_lock",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("lock acquired")),
    )
    monkeypatch.setattr(
        "review_costs._cached_metadata_for_cwd",
        lambda value: (_ for _ in ()).throw(AssertionError("metadata read")),
    )

    assert not apply_review_cost_delta_best_effort(
        state_dir=tmp_path / "state",
        review_cwd=tmp_path / "repo",
        record={"task_class": "phase_review", "runs": []},
    )
    assert not (tmp_path / "state").exists()


def test_automatic_cost_delta_updates_only_target_row_once(
    monkeypatch, tmp_path: Path
) -> None:
    state_dir = tmp_path / "state"
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(
        "review_costs.load_config", lambda state_dir: {"arena": {"enabled": True}}
    )
    monkeypatch.setattr(
        "review_costs._cached_metadata_for_cwd",
        lambda value: {
            "repo": "repo",
            "folder": "repo",
            "branch": "feat/costs",
            "pr_number": "42",
        },
    )
    lock_timeouts: list[int] = []
    real_lock = review_costs.orchestrator_store_lock

    def recording_lock(**kwargs):
        lock_timeouts.append(kwargs["timeout_seconds"])
        return real_lock(**kwargs)

    monkeypatch.setattr("review_costs.orchestrator_store_lock", recording_lock)
    unrelated = ReviewCostRow(
        repo="other",
        folder="other",
        branch="main",
        pr_number="-",
        worker_model="-",
        implementation_tokens=0,
        implementation_cost_usd=0.0,
        caller_threads=(),
        latest_review="2026-04-27T09:00:00Z",
        lane_sessions={lane: 0 for lane in review_costs.LANES},
        review_seconds=0,
        tokens=1,
        cost_usd=0.01,
        review_cwd_normalized=str(tmp_path / "other"),
    )
    update_review_cost_row_cache(state_dir=state_dir, rows=[unrelated])
    unrelated_path = next((state_dir / "review_cost_rows").glob("*.json"))
    unrelated_text = unrelated_path.read_text(encoding="utf-8")
    record = {
        "round_id": "round-1",
        "task_class": "phase_review",
        "status": "completed",
        "sampled_at": "2026-04-27T10:00:00Z",
        "review_completed_at": "2026-04-27T10:01:00Z",
        "runs": [
            {
                "slot": "alpha",
                "thread_id": "thread-a",
                "tokens_used": 100,
                "cost_usd": 0.01,
                "elapsed_seconds": 20,
                "review_status": "completed",
            },
            {
                "slot": "bravo",
                "thread_id": "thread-b",
                "tokens_used": 200,
                "cost_usd": 0.02,
                "elapsed_seconds": 30,
                "review_status": "completed",
            },
        ],
    }

    assert apply_review_cost_delta_best_effort(
        state_dir=state_dir, review_cwd=repo, record=record
    )
    assert not apply_review_cost_delta_best_effort(
        state_dir=state_dir, review_cwd=repo, record=record
    )
    same_second_record = {
        **record,
        "round_id": "round-2",
        "runs": [
            {
                "slot": "alpha",
                "thread_id": "thread-c",
                "tokens_used": 50,
                "cost_usd": 0.005,
                "elapsed_seconds": 10,
                "review_status": "completed",
            }
        ],
    }
    assert apply_review_cost_delta_best_effort(
        state_dir=state_dir, review_cwd=repo, record=same_second_record
    )

    row = read_review_cost_row_for_cwd(state_dir=state_dir, review_cwd=repo)
    assert row is not None
    assert row.lane_sessions["review_t1"] == 3
    assert row.tokens == 350
    assert row.cost_usd == pytest.approx(0.035)
    assert len(row.accounted_run_ids) == 3
    assert row.implementation_costs_refreshed is False
    assert lock_timeouts == [0, 0, 0]
    assert unrelated_path.read_text(encoding="utf-8") == unrelated_text


def test_automatic_cost_delta_skips_immediately_when_manual_lock_is_busy(
    monkeypatch, tmp_path: Path
) -> None:
    seen: list[int] = []
    monkeypatch.setattr(
        "review_costs.load_config", lambda state_dir: {"arena": {"enabled": True}}
    )

    def busy_lock(**kwargs):
        seen.append(kwargs["timeout_seconds"])
        raise TimeoutError("busy")

    monkeypatch.setattr("review_costs.orchestrator_store_lock", busy_lock)
    monkeypatch.setattr(
        "review_costs._cached_metadata_for_cwd",
        lambda value: (_ for _ in ()).throw(AssertionError("metadata read")),
    )

    assert not apply_review_cost_delta_best_effort(
        state_dir=tmp_path / "state",
        review_cwd=tmp_path / "repo",
        record={"task_class": "phase_review", "runs": []},
    )
    assert seen == [0]
    assert not (tmp_path / "state").exists()


def test_automatic_cost_delta_rejects_record_from_another_branch(
    monkeypatch, tmp_path: Path
) -> None:
    state_dir = tmp_path / "state"
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(
        "review_costs.load_config", lambda state_dir: {"arena": {"enabled": True}}
    )
    monkeypatch.setattr(
        "review_costs._cached_metadata_for_cwd",
        lambda value: {
            "repo": "repo",
            "folder": "repo",
            "branch": "branch-b",
            "pr_number": "-",
        },
    )

    assert not apply_review_cost_delta_best_effort(
        state_dir=state_dir,
        review_cwd=repo,
        record={
            "round_id": "round-a",
            "task_class": "phase_review",
            "task_id_hint": "branch-a",
            "review_completed_at": "2026-04-27T10:00:00Z",
            "runs": [
                {
                    "thread_id": "thread-a",
                    "tokens_used": 100,
                    "review_status": "completed",
                }
            ],
        },
    )
    assert read_review_cost_row_for_cwd(state_dir=state_dir, review_cwd=repo) is None


def test_fallback_run_identity_distinguishes_identical_independent_runs(
    monkeypatch, tmp_path: Path
) -> None:
    state_dir = tmp_path / "state"
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(
        "review_costs.load_config", lambda state_dir: {"arena": {"enabled": True}}
    )
    monkeypatch.setattr(
        "review_costs._cached_metadata_for_cwd",
        lambda value: {
            "repo": "repo",
            "folder": "repo",
            "branch": "feat/costs",
            "pr_number": "-",
        },
    )
    shared_run = {
        "slot": "alpha",
        "variant_id": "model-medium",
        "tokens_used": 100,
        "cost_usd": 0.01,
        "elapsed_seconds": 1,
        "review_status": "completed",
    }
    records = [
        {
            "round_id": f"round-{index}",
            "task_class": "phase_review",
            "review_completed_at": f"2026-04-27T10:0{index}:00Z",
            "runs": [dict(shared_run)],
        }
        for index in (1, 2)
    ]

    assert all(
        apply_review_cost_delta_best_effort(
            state_dir=state_dir, review_cwd=repo, record=record
        )
        for record in records
    )
    row = read_review_cost_row_for_cwd(state_dir=state_dir, review_cwd=repo)
    assert row is not None
    assert row.lane_sessions["review_t1"] == 2
    assert row.tokens == 200
    assert len(row.accounted_run_ids) == 2


def test_legacy_cutover_and_reroll_charge_only_the_new_run(
    monkeypatch, tmp_path: Path
) -> None:
    state_dir = tmp_path / "state"
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(
        "review_costs.load_config", lambda state_dir: {"arena": {"enabled": True}}
    )
    monkeypatch.setattr(
        "review_costs._cached_metadata_for_cwd",
        lambda value: {
            "repo": "repo",
            "folder": "repo",
            "branch": "feat/costs",
            "pr_number": "-",
        },
    )
    legacy = ReviewCostRow(
        repo="repo",
        folder="repo",
        branch="feat/costs",
        pr_number="-",
        worker_model="legacy-model",
        implementation_tokens=500,
        implementation_cost_usd=0.5,
        caller_threads=("legacy-caller",),
        latest_review="2026-04-27T10:00:00Z",
        lane_sessions={**{lane: 0 for lane in review_costs.LANES}, "review_t1": 1},
        review_seconds=60,
        tokens=100,
        cost_usd=0.01,
    )
    update_review_cost_row_cache(state_dir=state_dir, rows=[legacy])
    legacy_path = next((state_dir / "review_cost_rows").glob("*.json"))
    payload = json.loads(legacy_path.read_text(encoding="utf-8"))
    payload.pop("accounted_run_ids")
    payload.pop("review_cost_cutover_at")
    legacy_path.write_text(json.dumps(payload), encoding="utf-8")
    reroll = {
        "round_id": "round-2",
        "rerolled_from_round_id": "round-1",
        "task_class": "phase_review",
        "status": "completed",
        "sampled_at": "2026-04-27T10:04:00Z",
        "review_completed_at": "2026-04-27T10:05:00Z",
        "runs": [
            {
                "slot": "alpha",
                "thread_id": "legacy-survivor",
                "tokens_used": 100,
                "cost_usd": 0.01,
                "review_status": "completed",
            },
            {
                "slot": "bravo",
                "thread_id": "new-run",
                "rerolled_from_round_id": "round-1",
                "tokens_used": 50,
                "cost_usd": 0.005,
                "review_status": "completed",
            },
        ],
    }

    assert apply_review_cost_delta_best_effort(
        state_dir=state_dir, review_cwd=repo, record=reroll
    )
    row = read_review_cost_row_for_cwd(state_dir=state_dir, review_cwd=repo)
    assert row is not None
    assert row.tokens == 150
    assert row.cost_usd == pytest.approx(0.015)
    assert row.lane_sessions["review_t1"] == 2
    assert row.implementation_tokens == 500
    assert row.implementation_costs_refreshed is False
    assert row.review_cost_cutover_at == "2026-04-27T10:00:00Z"
    assert row.accounted_run_ids == ("thread_id:new-run",)
    assert not legacy_path.exists()


def test_update_review_cost_row_cache_replaces_pr_number_changes(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    old_row = ReviewCostRow(
        repo="repo",
        folder="worktree",
        branch="feat/demo",
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
        review_seconds=60,
        tokens=100,
        cost_usd=0.01,
    )
    new_row = ReviewCostRow(
        repo="repo",
        folder="worktree",
        branch="feat/demo",
        pr_number="263",
        worker_model="gpt-5.5 xhigh",
        implementation_tokens=109_000_000,
        implementation_cost_usd=67.0,
        caller_threads=("caller",),
        latest_review="2026-04-27T10:00:00Z",
        lane_sessions={
            "review_t1": 1,
            "review_t3": 0,
            "review_followup": 0,
        },
        review_seconds=60,
        tokens=100,
        cost_usd=0.01,
    )

    update_review_cost_row_cache(state_dir=state_dir, rows=[old_row])
    cache_dir = state_dir / "review_cost_rows"
    old_cache_path = next(cache_dir.glob("*.json"))
    legacy_cache_path = cache_dir / "legacy-prless-row.json"
    old_cache_path.rename(legacy_cache_path)
    update_review_cost_row_cache(state_dir=state_dir, rows=[new_row])

    cached_rows = read_review_cost_row_cache(state_dir)
    assert len(cached_rows) == 1
    assert cached_rows[0].pr_number == "263"
    assert cached_rows[0].worker_model == "gpt-5.5 xhigh"
    assert len(list((state_dir / "review_cost_rows").glob("*.json"))) == 1


def test_current_pr_number_can_skip_gh_for_background_refresh(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("REVIEW_SUITE_COST_SKIP_GH_PR_VIEW", "1")
    monkeypatch.setattr(
        "review_costs.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("gh should not run")
        ),
    )

    assert _current_pr_number(tmp_path) == ""


def test_cached_cost_metadata_never_looks_up_github_pr(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(
        "review_costs._git_text",
        lambda review_cwd, args: (
            "feat/costs"
            if args == ["branch", "--show-current"]
            else "https://github.com/example/repo.git"
        ),
    )
    monkeypatch.setattr(
        "review_costs._current_pr_number",
        lambda review_cwd: (_ for _ in ()).throw(AssertionError("gh pr view called")),
    )

    metadata = review_costs._cached_metadata_for_cwd(str(repo))

    assert metadata["branch"] == "feat/costs"
    assert metadata["pr_number"] == "-"


def test_collect_review_cost_rows_includes_implementation_only_worktree(
    monkeypatch, tmp_path: Path
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    codex_home = tmp_path / "codex-home"
    monkeypatch.setattr(
        "review_costs._metadata_for_cwd",
        lambda value: {
            "repo": "repo",
            "folder": "repo",
            "branch": "feat/costs",
            "pr_number": "-",
        },
    )
    _write_codex_thread(
        codex_home,
        thread_id="impl-only",
        cwd=repo,
        model="gpt-5.5",
        reasoning_effort="medium",
        tokens_used=2000,
        git_branch="feat/costs",
    )

    rows = collect_review_cost_rows(
        state_dir=state_dir, review_cwd=repo, codex_home=codex_home
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.worker_model == "gpt-5.5 medium"
    assert row.implementation_tokens == 2000
    assert row.tokens == 0
    assert row.lane_sessions == {
        "review_t1": 0,
        "review_t3": 0,
        "review_followup": 0,
    }


def test_collect_review_cost_rows_excludes_registered_wrapper_sessions(
    monkeypatch, tmp_path: Path
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    codex_home = tmp_path / "codex-home"
    monkeypatch.setattr(
        "review_costs._metadata_for_cwd",
        lambda value: {
            "repo": "repo",
            "folder": "repo",
            "branch": "feat/costs",
            "pr_number": "-",
        },
    )
    _write_codex_thread(
        codex_home,
        thread_id="review-wrapper-session",
        cwd=repo,
        model="gpt-5.5",
        reasoning_effort="medium",
        tokens_used=2000,
        git_branch="feat/costs",
        title="Review the commit range `abc..def` in the current repository.",
    )
    (state_dir / "wrapper_sessions.jsonl").write_text(
        json.dumps(
            {"session_id": "review-wrapper-session", "tool_name": "review-deslop"}
        )
        + "\n",
        encoding="utf-8",
    )

    rows = collect_review_cost_rows(
        state_dir=state_dir, review_cwd=repo, codex_home=codex_home
    )

    assert rows == []


def test_collect_review_cost_rows_includes_wrapper_caller_threads(
    monkeypatch, tmp_path: Path
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(
        "review_costs._metadata_for_cwd",
        lambda value: {
            "repo": "repo",
            "folder": "repo",
            "branch": "feat/costs",
            "pr_number": "-",
        },
    )
    (state_dir / "wrapper_sessions.jsonl").write_text(
        json.dumps(
            {
                "session_id": "review-wrapper-session",
                "tool_name": "review-deslop",
                "caller_thread_id": "019dd132-4116-71f3-b013-beaa9e5e95bd",
                "review_cwd": str(repo),
                "branch": "feat/costs",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = collect_review_cost_rows(
        state_dir=state_dir, review_cwd=repo, codex_home=tmp_path / "codex-home"
    )

    assert len(rows) == 1
    assert rows[0].caller_threads == ("019dd132-4116-71f3-b013-beaa9e5e95bd",)
    assert rows[0].lane_sessions == {
        "review_t1": 0,
        "review_t3": 0,
        "review_followup": 0,
    }


def test_collect_review_cost_rows_attributes_implementation_by_caller_thread(
    monkeypatch, tmp_path: Path
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    review_repo = tmp_path / "review-worktree"
    architect_repo = tmp_path / "architect-root"
    review_repo.mkdir()
    architect_repo.mkdir()
    codex_home = tmp_path / "codex-home"
    monkeypatch.setattr(
        "review_costs._metadata_for_cwd",
        lambda value: {
            "repo": "repo",
            "folder": "review-worktree",
            "branch": "feat/costs",
            "pr_number": "-",
        },
    )
    _write_codex_thread(
        codex_home,
        thread_id="caller-thread",
        cwd=architect_repo,
        model="gpt-5.5",
        reasoning_effort="xhigh",
        tokens_used=109_000_000,
        git_branch="main",
    )
    _write_codex_thread(
        codex_home,
        thread_id="cwd-guess",
        cwd=review_repo,
        model="gpt-5.5",
        reasoning_effort="medium",
        tokens_used=123,
        git_branch="feat/costs",
    )
    (state_dir / "wrapper_sessions.jsonl").write_text(
        json.dumps(
            {
                "session_id": "review-wrapper-session",
                "tool_name": "review-deslop",
                "caller_thread_id": "caller-thread",
                "review_cwd": str(review_repo),
                "branch": "feat/costs",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = collect_review_cost_rows(
        state_dir=state_dir, review_cwd=review_repo, codex_home=codex_home
    )

    assert len(rows) == 1
    assert rows[0].worker_model == "gpt-5.5 xhigh"
    assert rows[0].implementation_tokens == 109_000_000


def test_collect_review_cost_rows_skips_sampled_review_placeholders(
    monkeypatch, tmp_path: Path
) -> None:
    state_dir = tmp_path / "state"
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(
        "review_costs._metadata_for_cwd",
        lambda value: {
            "repo": "repo",
            "folder": "repo",
            "branch": "feat/costs",
            "pr_number": "-",
        },
    )
    _write_round(
        state_dir,
        {
            "round_id": "sampled-t1",
            "task_class": "phase_review",
            "task_id_hint": "feat/costs",
            "status": "sampled",
            "sampled_at": "2026-04-27T10:00:00Z",
            "review_cwd_normalized": str(repo),
            "runs": [
                {"slot": "alpha", "variant_id": "alpha-model"},
                {"slot": "bravo", "variant_id": "bravo-model"},
            ],
        },
    )

    assert (
        collect_review_cost_rows(
            state_dir=state_dir, review_cwd=repo, codex_home=tmp_path / "codex-home"
        )
        == []
    )


def test_collect_review_cost_rows_all_does_not_seed_unrelated_codex_cwds(
    monkeypatch, tmp_path: Path
) -> None:
    state_dir = tmp_path / "state"
    reviewed_repo = tmp_path / "reviewed"
    unrelated_repo = tmp_path / "unrelated"
    reviewed_repo.mkdir()
    unrelated_repo.mkdir()
    monkeypatch.setattr(
        "review_costs._metadata_for_cwd",
        lambda value: {
            "repo": Path(value).name,
            "folder": Path(value).name,
            "branch": "feat/costs",
            "pr_number": "-",
        },
    )
    _write_round(
        state_dir,
        {
            "round_id": "reviewed-t1",
            "task_class": "phase_review",
            "graded_task_id": "feat/costs",
            "status": "completed",
            "sampled_at": "2026-04-27T10:00:00Z",
            "review_completed_at": "2026-04-27T10:01:00Z",
            "review_cwd_normalized": str(reviewed_repo),
            "runs": [
                {
                    "slot": "alpha",
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                    "cost_usd": 0.001,
                }
            ],
        },
    )
    codex_home = tmp_path / "codex-home"
    _write_codex_thread(
        codex_home,
        thread_id="unrelated",
        cwd=unrelated_repo,
        model="gpt-5.5",
        reasoning_effort="medium",
        tokens_used=2000,
    )

    rows = collect_review_cost_rows(
        state_dir=state_dir, include_all=True, codex_home=codex_home
    )

    assert [row.folder for row in rows] == ["reviewed"]


def test_collect_review_cost_rows_excludes_support_review_sessions(
    monkeypatch, tmp_path: Path
) -> None:
    state_dir = tmp_path / "state"
    repo = tmp_path / "repo"
    repo.mkdir()
    codex_home = tmp_path / "codex-home"
    monkeypatch.setattr(
        "review_costs._metadata_for_cwd",
        lambda value: {
            "repo": "repo",
            "folder": "repo",
            "branch": "feat/costs",
            "pr_number": "-",
        },
    )
    _write_codex_thread(
        codex_home,
        thread_id="impl",
        cwd=repo,
        model="gpt-5.5",
        reasoning_effort="medium",
        tokens_used=2000,
        git_branch="feat/costs",
    )
    for thread_id, title in {
        "plan": "Review this implementation plan and scope.",
        "deslop": "Review the current repository changes against base branch `main`.",
        "followup": "Review this follow-up diff for correctness and regression risk.",
        "t1": "review-suite::phase_review-local",
    }.items():
        _write_codex_thread(
            codex_home,
            thread_id=thread_id,
            cwd=repo,
            model="gpt-5.5",
            reasoning_effort="medium",
            tokens_used=999_999,
            title=title,
        )

    rows = collect_review_cost_rows(
        state_dir=state_dir, review_cwd=repo, codex_home=codex_home
    )

    assert len(rows) == 1
    assert rows[0].implementation_tokens == 2000


def test_collect_review_cost_rows_filters_implementation_sessions_by_branch(
    monkeypatch, tmp_path: Path
) -> None:
    state_dir = tmp_path / "state"
    repo = tmp_path / "repo"
    repo.mkdir()
    codex_home = tmp_path / "codex-home"
    monkeypatch.setattr(
        "review_costs._metadata_for_cwd",
        lambda value: {
            "repo": "repo",
            "folder": "repo",
            "branch": "feat/current",
            "pr_number": "-",
        },
    )
    _write_codex_thread(
        codex_home,
        thread_id="current",
        cwd=repo,
        model="gpt-5.5",
        reasoning_effort="medium",
        tokens_used=2000,
        git_branch="feat/current",
    )
    _write_codex_thread(
        codex_home,
        thread_id="old",
        cwd=repo,
        model="gpt-5.5",
        reasoning_effort="medium",
        tokens_used=999_999,
        git_branch="feat/old",
    )
    _write_codex_thread(
        codex_home,
        thread_id="branchless",
        cwd=repo,
        model="gpt-5.5",
        reasoning_effort="medium",
        tokens_used=999_999,
    )

    rows = collect_review_cost_rows(
        state_dir=state_dir, review_cwd=repo, codex_home=codex_home
    )

    assert len(rows) == 1
    assert rows[0].implementation_tokens == 2000


def test_collect_review_cost_rows_keeps_older_threads_schema_without_git_branch(
    monkeypatch, tmp_path: Path
) -> None:
    state_dir = tmp_path / "state"
    repo = tmp_path / "repo"
    repo.mkdir()
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    monkeypatch.setattr(
        "review_costs._metadata_for_cwd",
        lambda value: {
            "repo": "repo",
            "folder": "repo",
            "branch": "feat/current",
            "pr_number": "-",
        },
    )
    with sqlite3.connect(codex_home / "state_5.sqlite") as db:
        db.execute(
            """
            CREATE TABLE threads (
                id TEXT,
                cwd TEXT,
                model TEXT,
                reasoning_effort TEXT,
                tokens_used INTEGER
            )
            """
        )
        db.execute(
            "INSERT INTO threads (id, cwd, model, reasoning_effort, tokens_used) VALUES (?, ?, ?, ?, ?)",
            ("legacy", str(repo), "gpt-5.5", "medium", 2000),
        )

    rows = collect_review_cost_rows(
        state_dir=state_dir, review_cwd=repo, codex_home=codex_home
    )

    assert len(rows) == 1
    assert rows[0].implementation_tokens == 2000


def test_collect_review_cost_rows_includes_subdirectory_sessions(
    monkeypatch, tmp_path: Path
) -> None:
    state_dir = tmp_path / "state"
    repo = tmp_path / "repo"
    subdir = repo / "plugins" / "review-suite"
    subdir.mkdir(parents=True)
    codex_home = tmp_path / "codex-home"
    monkeypatch.setattr(
        "review_costs._metadata_for_cwd",
        lambda value: {
            "repo": "repo",
            "folder": "repo",
            "branch": "feat/current",
            "pr_number": "-",
        },
    )
    _write_codex_thread(
        codex_home,
        thread_id="subdir",
        cwd=subdir,
        model="gpt-5.5",
        reasoning_effort="medium",
        tokens_used=2000,
        git_branch="feat/current",
    )

    rows = collect_review_cost_rows(
        state_dir=state_dir, review_cwd=repo, codex_home=codex_home
    )

    assert len(rows) == 1
    assert rows[0].implementation_tokens == 2000


def test_collect_review_cost_rows_filters_review_records_by_branch(
    monkeypatch, tmp_path: Path
) -> None:
    state_dir = tmp_path / "state"
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(
        "review_costs._metadata_for_cwd",
        lambda value: {
            "repo": "repo",
            "folder": "repo",
            "branch": "feat/current",
            "pr_number": "-",
        },
    )
    _write_round(
        state_dir,
        {
            "round_id": "old-t1",
            "task_class": "phase_review",
            "graded_task_id": "feat/old",
            "status": "completed",
            "sampled_at": "2026-04-27T10:00:00Z",
            "review_completed_at": "2026-04-27T10:01:00Z",
            "review_cwd_normalized": str(repo),
            "runs": [
                {
                    "slot": "alpha",
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                    "cost_usd": 0.001,
                }
            ],
        },
    )
    _write_round(
        state_dir,
        {
            "round_id": "current-t1",
            "task_class": "phase_review",
            "graded_task_id": "feat/current",
            "status": "completed",
            "sampled_at": "2026-04-27T11:00:00Z",
            "review_completed_at": "2026-04-27T11:01:00Z",
            "review_cwd_normalized": str(repo),
            "runs": [
                {
                    "slot": "alpha",
                    "usage": {"input_tokens": 200, "output_tokens": 30},
                    "cost_usd": 0.002,
                }
            ],
        },
    )

    rows = collect_review_cost_rows(
        state_dir=state_dir, review_cwd=repo, codex_home=tmp_path / "codex-home"
    )

    assert len(rows) == 1
    assert rows[0].lane_sessions["review_t1"] == 1
    assert rows[0].tokens == 230
    assert rows[0].cost_usd == 0.002


def test_collect_review_cost_rows_falls_back_when_rollout_usage_omits_total_tokens(
    monkeypatch, tmp_path: Path
) -> None:
    state_dir = tmp_path / "state"
    repo = tmp_path / "repo"
    repo.mkdir()
    codex_home = tmp_path / "codex-home"
    monkeypatch.setattr(
        "review_costs._metadata_for_cwd",
        lambda value: {
            "repo": "repo",
            "folder": "repo",
            "branch": "feat/current",
            "pr_number": "-",
        },
    )
    _write_codex_thread(
        codex_home,
        thread_id="usage-no-total",
        cwd=repo,
        model="gpt-5.5",
        reasoning_effort="medium",
        git_branch="feat/current",
        rollout_model="gpt-5.5",
        rollout_reasoning_effort="medium",
        rollout_usage={
            "input_tokens": 100,
            "cached_input_tokens": 50,
            "output_tokens": 25,
            "reasoning_output_tokens": 0,
        },
    )

    rows = collect_review_cost_rows(
        state_dir=state_dir, review_cwd=repo, codex_home=codex_home
    )

    assert len(rows) == 1
    assert rows[0].implementation_tokens == 125


def test_collect_review_cost_rows_reads_model_metadata_from_rollout(
    monkeypatch, tmp_path: Path
) -> None:
    state_dir = tmp_path / "state"
    repo = tmp_path / "repo"
    repo.mkdir()
    codex_home = tmp_path / "codex-home"
    monkeypatch.setattr(
        "review_costs._metadata_for_cwd",
        lambda value: {
            "repo": "repo",
            "folder": "repo",
            "branch": "feat/current",
            "pr_number": "-",
        },
    )
    _write_codex_thread(
        codex_home,
        thread_id="rollout-model-only",
        cwd=repo,
        model="",
        reasoning_effort="",
        git_branch="feat/current",
        rollout_model="gpt-5.5",
        rollout_reasoning_effort="medium",
        rollout_usage={
            "input_tokens": 100,
            "cached_input_tokens": 50,
            "output_tokens": 25,
            "reasoning_output_tokens": 0,
            "total_tokens": 125,
        },
    )

    rows = collect_review_cost_rows(
        state_dir=state_dir, review_cwd=repo, codex_home=codex_home
    )

    assert len(rows) == 1
    assert rows[0].worker_model == "gpt-5.5 medium"
    assert rows[0].implementation_cost_usd == 0.001025


def test_read_rollout_model_metadata_streams_jsonl(monkeypatch, tmp_path: Path) -> None:
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text(
        json.dumps(
            {
                "type": "turn_context",
                "payload": {"model": "gpt-5.5", "effort": "medium"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fail_read_text(self, *args, **kwargs):
        raise AssertionError("metadata reader should stream rollout files")

    monkeypatch.setattr(Path, "read_text", fail_read_text)

    assert _read_rollout_model_metadata(str(rollout)) == {
        "model": "gpt-5.5",
        "reasoning_effort": "medium",
    }


def test_thread_rows_filters_by_cwd_in_sqlite(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex-home"
    target = tmp_path / "target"
    unrelated = tmp_path / "unrelated"
    target.mkdir()
    unrelated.mkdir()
    _write_codex_thread(
        codex_home,
        thread_id="target",
        cwd=target,
        model="gpt-5.5",
        reasoning_effort="medium",
        tokens_used=1,
    )
    _write_codex_thread(
        codex_home,
        thread_id="unrelated",
        cwd=unrelated,
        model="gpt-5.5",
        reasoning_effort="medium",
        tokens_used=1,
    )

    rows = _thread_rows(codex_home / "state_5.sqlite", cwd_filters={str(target)})

    assert [row["id"] for row in rows] == ["target"]


def test_thread_rows_filters_by_thread_id_in_sqlite(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex-home"
    target = tmp_path / "target"
    unrelated = tmp_path / "unrelated"
    target.mkdir()
    unrelated.mkdir()
    _write_codex_thread(
        codex_home,
        thread_id="target-thread",
        cwd=target,
        model="gpt-5.5",
        reasoning_effort="medium",
        tokens_used=1,
    )
    _write_codex_thread(
        codex_home,
        thread_id="unrelated",
        cwd=unrelated,
        model="gpt-5.5",
        reasoning_effort="medium",
        tokens_used=1,
    )

    rows = _thread_rows(codex_home / "state_5.sqlite", id_filters={"target-thread"})

    assert [row["id"] for row in rows] == ["target-thread"]


def test_thread_rows_includes_descendant_cwds(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex-home"
    target = tmp_path / "target"
    child = target / "child"
    unrelated = tmp_path / "target-other"
    child.mkdir(parents=True)
    unrelated.mkdir()
    _write_codex_thread(
        codex_home,
        thread_id="child",
        cwd=child,
        model="gpt-5.5",
        reasoning_effort="medium",
        tokens_used=1,
    )
    _write_codex_thread(
        codex_home,
        thread_id="unrelated",
        cwd=unrelated,
        model="gpt-5.5",
        reasoning_effort="medium",
        tokens_used=1,
    )

    rows = _thread_rows(codex_home / "state_5.sqlite", cwd_filters={str(target)})

    assert [row["id"] for row in rows] == ["child"]


def test_thread_rows_escapes_descendant_like_wildcards(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex-home"
    target = tmp_path / "repo_a"
    child = target / "child"
    wildcard_sibling = tmp_path / "repoXa" / "child"
    child.mkdir(parents=True)
    wildcard_sibling.mkdir(parents=True)
    _write_codex_thread(
        codex_home,
        thread_id="child",
        cwd=child,
        model="gpt-5.5",
        reasoning_effort="medium",
        tokens_used=1,
    )
    _write_codex_thread(
        codex_home,
        thread_id="sibling",
        cwd=wildcard_sibling,
        model="gpt-5.5",
        reasoning_effort="medium",
        tokens_used=1,
    )

    rows = _thread_rows(codex_home / "state_5.sqlite", cwd_filters={str(target)})

    assert [row["id"] for row in rows] == ["child"]


def test_thread_rows_tolerates_missing_created_at_column(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    db_path = codex_home / "state_5.sqlite"
    with sqlite3.connect(db_path) as db:
        db.execute(
            """
            CREATE TABLE threads (
                id TEXT,
                cwd TEXT,
                model TEXT,
                tokens_used INTEGER
            )
            """
        )
        db.execute(
            "INSERT INTO threads (id, cwd, model, tokens_used) VALUES (?, ?, ?, ?)",
            ("thread-1", str(tmp_path / "repo"), "gpt-5.5", 1),
        )

    rows = _thread_rows(db_path, cwd_filters={str(tmp_path / "repo")})

    assert [row["id"] for row in rows] == ["thread-1"]


def test_current_pr_number_treats_missing_gh_as_optional(
    monkeypatch, tmp_path: Path
) -> None:
    def raise_missing(*args, **kwargs):
        raise FileNotFoundError("gh")

    monkeypatch.setattr("review_costs.subprocess.run", raise_missing)

    assert _current_pr_number(tmp_path) == ""
