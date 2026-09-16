from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import review
import review_suite_arena
from review_suite_core import orchestrator_runner, orchestrator_store
from review_suite_local import write_round


_GIT_ENV = os.environ | {
    "GIT_AUTHOR_EMAIL": "codex@example.invalid",
    "GIT_AUTHOR_NAME": "Codex",
    "GIT_COMMITTER_EMAIL": "codex@example.invalid",
    "GIT_COMMITTER_NAME": "Codex",
    "GIT_TERMINAL_PROMPT": "0",
}


@pytest.fixture(autouse=True)
def _isolate_default_state_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(review, "default_state_dir", lambda: tmp_path / "default-state")

    def unexpected_cleanup(**kwargs: object) -> None:
        raise AssertionError(
            "Test must stub cleanup or create the cycle with --skip-deslop"
        )

    monkeypatch.setattr(
        orchestrator_runner, "run_deslop_subprocess", unexpected_cleanup
    )


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=_GIT_ENV,
    )
    if proc.returncode != 0:
        raise AssertionError(
            proc.stderr or proc.stdout or f"git {' '.join(args)} failed"
        )
    return proc.stdout.strip()


def _init_repo(repo: Path) -> None:
    repo.mkdir()
    _git(repo, "init", "-b", "main")


def _commit_file(repo: Path, relative_path: str, content: str, message: str) -> str:
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _git(repo, "add", relative_path)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _amend_file(repo: Path, relative_path: str, content: str) -> str:
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _git(repo, "add", relative_path)
    _git(repo, "commit", "--amend", "--no-edit")
    return _git(repo, "rev-parse", "HEAD")


def _run_review(
    monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> tuple[int, dict[str, object]]:
    emitted: list[dict[str, object]] = []
    monkeypatch.setattr(review, "emit_toon", lambda payload: emitted.append(payload))
    monkeypatch.setattr(
        sys, "argv", ["review.py", *_without_state_dir_args(monkeypatch, args)]
    )

    exit_code = review.main()

    assert len(emitted) == 1
    return exit_code, emitted[0]


def _run_review_after_cleanup(
    monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> tuple[int, dict[str, object]]:
    result = _run_review(monkeypatch, args)
    if "--deslop-done" in str(result[1].get("Action", {}).get("cmd", "")):
        public_id = str(result[1]["review"])
        _run_review(monkeypatch, ["--id", public_id, "--deslop-done"])
        return _run_review(monkeypatch, ["--id", public_id])
    return result


def _assert_decision_action(action: object) -> dict[str, str]:
    payload = dict(action)  # type: ignore[arg-type]
    assert set(payload) == {"choices", "note"}
    assert payload["note"] == (
        "Classify the reviewer output, then record clean or findings."
    )
    choices = dict(payload["choices"])
    assert set(choices) == {"clean", "findings"}
    return choices


def test_main_emits_structured_runtime_error_for_artifact_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    def fail_artifact_cleanup(_args: object) -> int:
        raise OSError("review artifact cleanup incomplete")

    monkeypatch.setattr(review, "default_state_dir", lambda: tmp_path)
    monkeypatch.setattr(review, "cmd_branch_status", fail_artifact_cleanup)
    monkeypatch.setattr(sys, "argv", ["review.py", "--status"])

    assert review.main() == 1
    output = capsys.readouterr()
    assert "status: error" in output.out
    assert "review artifact cleanup incomplete" in output.out
    assert "Traceback" not in output.err


def _without_state_dir_args(
    monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> list[str]:
    cleaned: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--state-dir":
            if index + 1 >= len(args):
                raise AssertionError("--state-dir test fixture is missing a value")
            state_dir = Path(args[index + 1]).resolve(strict=False)
            monkeypatch.setattr(
                review, "default_state_dir", lambda state_dir=state_dir: state_dir
            )
            index += 2
            continue
        cleaned.append(arg)
        index += 1
    return cleaned


def _stub_deslop(monkeypatch: pytest.MonkeyPatch, *returncodes: int) -> list[list[str]]:
    calls: list[list[str]] = []
    codes = list(returncodes) or [0]

    def fake_run(*, command: list[str], cwd: Path) -> subprocess.CompletedProcess:
        calls.append(command)
        index = min(len(calls) - 1, len(codes) - 1)
        verdict = (
            "CONFORMS"
            if any(arg.startswith("--review-brief=") for arg in command)
            else "NOT_APPLICABLE"
        )
        return subprocess.CompletedProcess(
            command,
            codes[index],
            stdout=f"Conformance: {verdict}\nReview decision: clean\n",
            stderr="",
        )

    monkeypatch.setattr(orchestrator_runner, "run_deslop_subprocess", fake_run)
    return calls


def _review_discovery_config(config: dict[str, object]) -> dict[str, object]:
    return config


def _stub_review(
    monkeypatch: pytest.MonkeyPatch, *round_ids: str
) -> list[dict[str, object]]:
    load_config = review.load_config
    monkeypatch.setattr(
        review,
        "load_config",
        lambda state_dir: _review_discovery_config(deepcopy(load_config(state_dir))),
    )
    calls: list[dict[str, object]] = []
    ids = list(round_ids) or ["phase_review-round-1"]

    def fake_run(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        round_id = ids[min(len(calls) - 1, len(ids) - 1)]
        scope = kwargs.get("review_scope")
        reviewed_head = str(
            scope.get("reviewed_head") if isinstance(scope, dict) else "head-1"
        )
        on_round_started = kwargs.get("on_round_started")
        if callable(on_round_started):
            on_round_started(
                {
                    "round_id": round_id,
                    "round_state_dir": "state/orchestrator/review-rounds",
                    "reviewed_head": reviewed_head,
                }
            )
        return {
            "round_id": round_id,
            "lane": "review_t1",
            "kind": "review",
            "status": "completed",
            "blocked": False,
            "reviewed_head": reviewed_head,
            "output_refs": [f"rollout://{round_id}/alpha"],
            "runs": [
                {
                    "slot": "alpha",
                    "status": "completed",
                    "summary": "No findings.",
                    "ref": f"rollout://{round_id}/alpha",
                    "blocked": False,
                    "block": None,
                }
            ],
            "round_state_dir": "state/orchestrator/review-rounds",
        }

    monkeypatch.setattr(orchestrator_runner, "run_review_step", fake_run)
    return calls


def _stub_review_with_terminal(
    monkeypatch: pytest.MonkeyPatch,
    *commands: str,
    round_ids: tuple[str, ...] = ("phase_review-round-1", "phase_review-round-2"),
) -> list[dict[str, object]]:
    load_config = review.load_config
    monkeypatch.setattr(
        review,
        "load_config",
        lambda state_dir: _review_discovery_config(deepcopy(load_config(state_dir))),
    )
    calls: list[dict[str, object]] = []
    terminal_commands = list(commands) or ["clean"]

    def fake_run(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        index = len(calls) - 1
        round_id = round_ids[min(index, len(round_ids) - 1)]
        command = terminal_commands[min(index, len(terminal_commands) - 1)]
        scope = kwargs.get("review_scope")
        reviewed_head = str(
            scope.get("reviewed_head") if isinstance(scope, dict) else "head-1"
        )
        output = (
            "No findings.\n\nReview result: clean"
            if command == "clean"
            else "P1: concrete regression.\n\nReview result: findings"
        )
        on_round_started = kwargs.get("on_round_started")
        if callable(on_round_started):
            on_round_started(
                {
                    "round_id": round_id,
                    "round_state_dir": "state/orchestrator/review-rounds",
                    "reviewed_head": reviewed_head,
                }
            )
        return {
            "round_id": round_id,
            "lane": "review_t1",
            "kind": "review",
            "status": "completed",
            "blocked": False,
            "reviewed_head": reviewed_head,
            "output_refs": [f"rollout://{round_id}/alpha"],
            "runs": [
                {
                    "slot": "alpha",
                    "status": "completed",
                    "summary": output,
                    "reviewer_output": output,
                    "terminal_command": command,
                    "ref": f"rollout://{round_id}/alpha",
                    "blocked": False,
                    "block": None,
                }
            ],
            "round_state_dir": "state/orchestrator/review-rounds",
        }

    monkeypatch.setattr(orchestrator_runner, "run_review_step", fake_run)
    return calls


def _stub_followup(
    monkeypatch: pytest.MonkeyPatch, *round_ids: str
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    ids = list(round_ids) or ["followup-round-1"]

    def fake_run(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        round_id = ids[min(len(calls) - 1, len(ids) - 1)]
        scope = kwargs.get("review_scope")
        reviewed_head = str(
            scope.get("reviewed_head") if isinstance(scope, dict) else "head-2"
        )
        return {
            "round_id": round_id,
            "lane": "review-followup",
            "kind": "followup",
            "status": "completed",
            "blocked": False,
            "reviewed_head": reviewed_head,
            "output_refs": [f"rollout://{round_id}/alpha"],
            "runs": [
                {
                    "slot": "alpha",
                    "status": "completed",
                    "summary": "No findings.",
                    "ref": f"rollout://{round_id}/alpha",
                    "blocked": False,
                    "block": None,
                }
            ],
            "round_state_dir": "state/orchestrator/review-rounds",
        }

    monkeypatch.setattr(orchestrator_runner, "run_followup_review_step", fake_run)
    return calls


def _use_compact_normal_profile(
    monkeypatch: pytest.MonkeyPatch, state_dir: Path, *, include_deep: bool = False
) -> dict[str, object]:
    config = _review_discovery_config(deepcopy(review.load_config(state_dir)))
    profiles = config["orchestrator"]["profiles"]["stable"]
    profiles["normal"]["steps"] = [
        {
            "name": "broad-discovery",
            "count": 1,
            "model_ref": "signoff_normal_model",
        },
        *profiles["normal"]["steps"],
    ]
    if include_deep:
        profiles["deep"]["steps"] = [
            {
                "name": "broad-discovery",
                "count": 1,
                "model_ref": "signoff_normal_model",
            },
            profiles["deep"]["steps"][0],
            {
                "name": "deep-discovery",
                "count": 1,
                "model_ref": "signoff_normal_model",
            },
            profiles["deep"]["steps"][-1],
        ]
    monkeypatch.setattr(review, "load_config", lambda state_dir: config)
    return config


def _use_single_step_normal_profile(
    monkeypatch: pytest.MonkeyPatch, state_dir: Path
) -> None:
    config = deepcopy(review.load_config(state_dir))
    normal = config["orchestrator"]["profiles"]["stable"]["normal"]
    normal["steps"] = [
        {
            "name": "precision-signoff",
            "count": 2,
            "model_ref": "signoff_normal_model",
            "rerun_on_findings": True,
        }
    ]
    monkeypatch.setattr(review, "load_config", lambda state_dir: config)


def _cycle_payload(state_dir: Path, public_id: str) -> dict[str, object]:
    index = json.loads(
        (state_dir / "orchestrator" / "index.json").read_text(encoding="utf-8")
    )
    cycle_key = index["ids"][public_id]
    return json.loads(
        (state_dir / "orchestrator" / "cycles" / f"{cycle_key}.json").read_text(
            encoding="utf-8"
        )
    )


def _write_cycle_payload(
    state_dir: Path, public_id: str, payload: dict[str, object]
) -> None:
    index = json.loads(
        (state_dir / "orchestrator" / "index.json").read_text(encoding="utf-8")
    )
    cycle_key = index["ids"][public_id]
    (state_dir / "orchestrator" / "cycles" / f"{cycle_key}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_profile_resolution_serializes_configured_arena_pool(tmp_path: Path) -> None:
    config = review.load_config(tmp_path / "state")
    config["arena"]["enabled"] = True
    config["orchestrator"]["stable_defaults"]["normal_arena_loops"] = 1
    resolution = review.resolve_orchestrator_profile(
        config, mode="normal", selection="stable"
    )

    state = review._apply_profile_resolution({"selection": {}}, resolution)

    arena_step = state["review_plan"]["steps"][0]
    assert arena_step["name"] == "arena-phase-review"
    assert arena_step["rating_pool_id"] == "arena-phase-gpt-5.6-v1"
    assert arena_step["reporting_pool"] is True
    assert len(arena_step["variant_groups"]) == 13
    assert len(arena_step["variant_ids"]) == 19


def _enable_arena(state_dir: Path) -> None:
    state_dir.mkdir(parents=True)
    (state_dir / "settings.toml").write_text(
        "[arena]\nenabled = true\n"
        "[orchestrator.stable_defaults]\nnormal_arena_loops = 1\n",
        encoding="utf-8",
    )


def test_repo_no_arena_preference_handles_spaces_and_can_be_cleared(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo with spaces"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _enable_arena(state_dir)

    _, configured = _run_review(monkeypatch, ["--set-no-arena", "--cd", str(repo)])
    assert configured["arena"] == "disabled"
    assert (
        _git(
            repo,
            "config",
            "--local",
            "--type=bool",
            "--get",
            review.REPO_NO_ARENA_CONFIG,
        )
        == "true"
    )

    review_calls = _stub_review(monkeypatch)
    _, created = _run_review(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--skip-deslop",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    state = _cycle_payload(state_dir, str(created["review"]))
    assert all(step["kind"] != "arena" for step in state["review_plan"]["steps"])
    assert len(review_calls) == 1

    _, cleared = _run_review(monkeypatch, ["--clear-no-arena", "--cd", str(repo)])
    assert cleared["arena"] == "global-default"
    assert not review._repo_has_no_arena(repo)

    _, restarted = _run_review(
        monkeypatch,
        [
            "--id",
            str(created["review"]),
            "--restart-mode",
            "deep",
            "--reason",
            "exercise cleared Arena preference",
        ],
    )
    successor = _cycle_payload(state_dir, str(restarted["review"]))
    assert any(step["kind"] == "arena" for step in successor["review_plan"]["steps"])


def test_no_arena_skips_arena_for_one_cycle_without_setting_repo_preference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _enable_arena(state_dir)
    review_calls = _stub_review(monkeypatch)

    _, created = _run_review(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--skip-deslop",
            "--no-arena",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )

    state = _cycle_payload(state_dir, str(created["review"]))
    assert all(step["kind"] != "arena" for step in state["review_plan"]["steps"])
    assert state["arena_disabled"] == "cli"
    assert len(review_calls) == 1
    assert not review._repo_has_no_arena(repo)


def test_model_override_prefixes_provider_model_and_defaults_reasoning(
    tmp_path: Path,
) -> None:
    config = review.load_config(tmp_path / "state")
    override = review._requested_model_override(
        argparse.Namespace(model="opencode-go/deepseek-v4.1-flash", reasoning=None)
    )

    assert override == {"model": "opencode::opencode-go/deepseek-v4.1-flash"}

    resolved = review._config_with_model_override(config, override)
    defaults = resolved["orchestrator"]["stable_defaults"]
    assert (
        defaults["signoff_normal_model"]
        == "opencode::opencode-go/deepseek-v4.1-flash-high"
    )
    assert (
        defaults["signoff_deep_model"]
        == "opencode::opencode-go/deepseek-v4.1-flash-high"
    )
    assert resolved["normal"]["model"] == "opencode::opencode-go/deepseek-v4.1-flash"


def test_model_override_drops_opencode_service_tier(tmp_path: Path) -> None:
    config = review.load_config(tmp_path / "state")
    config["orchestrator"]["stable_defaults"]["signoff_normal_model"] = (
        "gpt-6-astra-medium-fast"
    )

    resolved = review._config_with_model_override(
        config,
        {"model": "opencode::opencode-go/glm-5.3-flash"},
    )

    defaults = resolved["orchestrator"]["stable_defaults"]
    assert (
        defaults["signoff_normal_model"] == "opencode::opencode-go/glm-5.3-flash-high"
    )
    fast_step = review.resolve_orchestrator_profile(
        resolved, mode="fast", selection="stable"
    ).steps[0]
    assert fast_step.service_tier is None

    codex_resolved = review._config_with_model_override(
        config, {"model": "gpt-6-astra"}
    )
    codex_step = review.resolve_orchestrator_profile(
        codex_resolved, mode="fast", selection="stable"
    ).steps[0]
    assert codex_step.service_tier == "fast"


def test_opencode_model_override_accepts_supported_reasoning(tmp_path: Path) -> None:
    config = review.load_config(tmp_path / "state")

    resolved = review._config_with_model_override(
        config,
        {"model": "opencode::opencode-go/glm-5.3-flash", "reasoning": "max"},
    )

    defaults = resolved["orchestrator"]["stable_defaults"]
    assert defaults["signoff_normal_model"] == "opencode::opencode-go/glm-5.3-flash-max"


def test_opencode_model_override_rejects_unsupported_reasoning(tmp_path: Path) -> None:
    config = review.load_config(tmp_path / "state")

    with pytest.raises(ValueError, match="unsupported for opencode-go/glm-5.3-flash"):
        review._config_with_model_override(
            config,
            {"model": "opencode::opencode-go/glm-5.3-flash", "reasoning": "medium"},
        )


def test_model_override_accepts_codex_model_and_reasoning_only(tmp_path: Path) -> None:
    assert review._requested_model_override(
        argparse.Namespace(model="gpt-6-astra", reasoning=None)
    ) == {"model": "gpt-6-astra"}
    assert review._requested_model_override(
        argparse.Namespace(model=None, reasoning="high")
    ) == {"reasoning": "high"}

    config = review.load_config(tmp_path / "state")
    resolved = review._config_with_model_override(config, {"reasoning": "high"})
    defaults = resolved["orchestrator"]["stable_defaults"]
    assert defaults["signoff_normal_model"] == "gpt-6-astra-high"
    assert defaults["signoff_deep_model"] == "gpt-6-astra-high"


def test_requested_model_override_rejects_malformed_opencode() -> None:
    with pytest.raises(ValueError, match="provider/model"):
        review._requested_model_override(
            argparse.Namespace(model="opencode::foo", reasoning=None)
        )
    with pytest.raises(ValueError, match="provider/model"):
        review._requested_model_override(
            argparse.Namespace(model="opencode-go/", reasoning=None)
        )


def test_model_override_persists_into_fast_cycle_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_deslop(monkeypatch)
    _stub_review(monkeypatch)
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/model-override")
    _commit_file(repo, "app.txt", "feature\n", "feature")

    _, created = _run_review(
        monkeypatch,
        [
            "--mode",
            "fast",
            "--model",
            "opencode-go/deepseek-v4.1-flash",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )

    state = _cycle_payload(state_dir, str(created["review"]))
    assert state["model_override"] == {
        "model": "opencode::opencode-go/deepseek-v4.1-flash"
    }
    step = state["review_plan"]["steps"][0]
    assert step["name"] == "fast-signoff"
    assert step["model"] == "opencode::opencode-go/deepseek-v4.1-flash"
    assert step["reasoning_effort"] == "high"
    assert step["service_tier"] is None


def test_model_override_survives_restart_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_deslop(monkeypatch)
    _stub_review(monkeypatch, "fast-round-1", "deep-round-1")
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _use_compact_normal_profile(monkeypatch, state_dir, include_deep=True)
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/model-restart")
    _commit_file(repo, "app.txt", "feature\n", "feature")

    _, created = _run_review(
        monkeypatch,
        [
            "--mode",
            "fast",
            "--model",
            "opencode-go/deepseek-v4.1-flash",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    old_id = str(created["review"])
    _run_review(monkeypatch, ["--id", old_id, "--state-dir", str(state_dir)])

    _, restarted = _run_review(
        monkeypatch,
        [
            "--id",
            old_id,
            "--restart-mode",
            "deep",
            "--reason",
            "test",
            "--state-dir",
            str(state_dir),
        ],
    )

    new_state = _cycle_payload(state_dir, str(restarted["review"]))
    assert new_state["model_override"] == {
        "model": "opencode::opencode-go/deepseek-v4.1-flash"
    }
    steps = {step["name"]: step for step in new_state["review_plan"]["steps"]}
    assert steps["deep-signoff"]["model"] == "opencode::opencode-go/deepseek-v4.1-flash"
    assert steps["deep-signoff"]["reasoning_effort"] == "high"


def _assert_github_handoff(
    action: object, *, public_id: str, state_dir: Path, blocked_by: list[str]
) -> None:
    assert isinstance(action, dict)
    payload = dict(action)
    assert "kind" not in payload
    assert "lane" not in payload
    assert payload["after"] == "PR create/update"
    assert "github_review" not in payload
    assert "merge_ready" not in payload
    cmd = str(payload["cmd"])
    assert f"--id {public_id}" in cmd
    assert "--github-review" in cmd
    assert "--state-dir" not in cmd
    assert str(state_dir) not in cmd
    assert "--github-force" not in cmd
    for result_cmd in dict(payload["result"]).values():
        result_command = str(result_cmd)
        assert f"--id {public_id}" in result_command
        assert "--state-dir" not in result_command
        assert str(state_dir) not in result_command
    if blocked_by:
        assert payload["blocked_by"] == blocked_by
    else:
        assert "validation_ready" not in payload
        assert "blocked_by" not in payload


def test_orchestrator_review_helper_uses_phase_prompt_without_predecision_anchor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/orchestrator-review")
    head = _commit_file(repo, "app.txt", "feature\n", "feature")
    captured: dict[str, object] = {}
    anchor_calls: list[dict[str, object]] = []
    banner_calls: list[dict[str, object]] = []
    refresh_calls: list[dict[str, object]] = []
    completion_events: list[str] = []

    def fake_run_round(
        *,
        round_payload: dict[str, object],
        roster: dict[str, object],
        state_dir: Path,
        review_cwd: Path,
        prompt: str,
        review_scope: dict[str, object],
        sqlite_path: Path,
        progress_interval_seconds: int,
        allow_unsafe_windows_wsl_fallback: bool,
    ) -> dict[str, object]:
        captured["prompt"] = prompt
        captured["review_scope"] = dict(review_scope)
        captured["round_payload"] = dict(round_payload)
        return {
            **round_payload,
            "status": "completed",
            "review_scope": dict(review_scope),
            "runs": [
                {
                    "slot": "alpha",
                    "review_status": "completed",
                    "status_summary": "No findings.",
                    "grade_blocked": False,
                    "grade_block_reason": None,
                    "reviewer_output": "No findings.",
                    "reviewer_output_ref": "rollout://alpha",
                }
            ],
        }

    monkeypatch.setattr(review_suite_arena, "run_round", fake_run_round)
    monkeypatch.setattr(
        review_suite_arena,
        "_print_round_banner",
        lambda **kwargs: banner_calls.append(kwargs),
    )
    monkeypatch.setattr(
        review_suite_arena,
        "_print_findings",
        lambda result: completion_events.append("findings") or False,
    )
    monkeypatch.setattr(
        review_suite_arena,
        "record_review_anchor",
        lambda **kwargs: anchor_calls.append(kwargs),
    )
    monkeypatch.setattr(
        review_suite_arena,
        "apply_review_cost_delta_best_effort",
        lambda **kwargs: (
            completion_events.append("cost") or refresh_calls.append(kwargs) or True
        ),
    )

    result = review_suite_arena.run_orchestrator_review_step(
        lane="review_t1",
        step_name="precision",
        reviewer_count=1,
        model="gpt-5.5",
        reasoning_effort="medium",
        service_tier="flex",
        review_cwd=repo,
        state_dir=state_dir,
        sqlite_path=tmp_path / "state.sqlite",
        review_scope={"base": "main"},
        task_id="feature/orchestrator-review",
        progress_interval_seconds=1,
        allow_unsafe_windows_wsl_fallback=False,
        step_position=1,
        step_total=2,
    )

    prompt = str(captured["prompt"])
    assert prompt.strip()
    assert "Review this implementation slice" in prompt
    assert "Reviewer output is advisory risk input" in prompt
    assert "=== BEGIN DIFF ===" not in prompt
    assert "manual_prompt_mode" not in dict(captured["review_scope"])
    assert dict(captured["round_payload"])["runs"][0]["service_tier"] == "flex"
    assert result["reviewed_head"] == head
    assert result["output_refs"] == ["rollout://alpha"]
    assert banner_calls == [
        {"task_name": "review 1/2 precision", "round_id": result["round_id"]}
    ]
    assert anchor_calls == []
    assert len(refresh_calls) == 1
    assert refresh_calls[0]["state_dir"] == state_dir
    assert refresh_calls[0]["review_cwd"] == repo
    assert refresh_calls[0]["record"]["status"] == "completed"
    assert completion_events == ["findings", "cost"]


def test_orchestrator_arena_helper_uses_pr_prompt_for_pr_review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/orchestrator-arena-pr")
    head = _commit_file(repo, "app.txt", "feature\n", "feature")
    captured: dict[str, object] = {}
    select_calls: list[dict[str, object]] = []
    anchor_calls: list[dict[str, object]] = []

    def fake_select_pair(**kwargs: object) -> dict[str, object]:
        select_calls.append(dict(kwargs))
        return {
            "round_id": "arena-pr-round-1",
            "task_class": kwargs["task_class"],
            "status": "sampled",
            "runs": [
                {
                    "slot": "alpha",
                    "variant_id": "alpha-model",
                    "model": "gpt-5.5",
                    "reasoning_effort": "xhigh",
                }
            ],
        }

    def fake_run_round(
        *,
        round_payload: dict[str, object],
        roster: dict[str, object],
        state_dir: Path,
        review_cwd: Path,
        prompt: str,
        review_scope: dict[str, object],
        sqlite_path: Path,
        progress_interval_seconds: int,
        allow_unsafe_windows_wsl_fallback: bool,
    ) -> dict[str, object]:
        captured["prompt"] = prompt
        captured["review_scope"] = dict(review_scope)
        captured["round_payload"] = dict(round_payload)
        return {
            **round_payload,
            "status": "completed",
            "review_scope": dict(review_scope),
            "runs": [
                {
                    "slot": "alpha",
                    "variant_id": "alpha-model",
                    "review_status": "completed",
                    "status_summary": "No findings.",
                    "grade_blocked": False,
                    "grade_block_reason": None,
                    "reviewer_output": "No findings.",
                    "reviewer_output_ref": "rollout://alpha",
                }
            ],
        }

    monkeypatch.setattr(
        review_suite_arena,
        "_validate_benchmarked_review_runtime",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        review_suite_arena, "load_roster", lambda path: {"settings": {}, "variants": []}
    )
    monkeypatch.setattr(review_suite_arena, "load_operational_state", lambda path: {})
    monkeypatch.setattr(review_suite_arena, "select_pair", fake_select_pair)
    monkeypatch.setattr(review_suite_arena, "run_round", fake_run_round)
    monkeypatch.setattr(
        review_suite_arena, "_print_round_banner", lambda **kwargs: None
    )
    monkeypatch.setattr(review_suite_arena, "_print_findings", lambda result: False)
    monkeypatch.setattr(
        review_suite_arena,
        "record_review_anchor",
        lambda **kwargs: anchor_calls.append(kwargs),
    )
    monkeypatch.setattr(
        review_suite_arena,
        "apply_review_cost_delta_best_effort",
        lambda **kwargs: True,
    )

    result = review_suite_arena.run_orchestrated_arena_round(
        lane="review_t3",
        task_class="pr_review",
        step_name="arena-pr-review",
        rating_pool_id="arena-deep-gpt-5.6-v1",
        variant_groups=[["a", "b", "c", "d"]],
        variant_ids=["a", "b", "c", "d", "e"],
        review_cwd=repo,
        state_dir=state_dir,
        sqlite_path=tmp_path / "state.sqlite",
        review_scope={"base": "main"},
        task_id="feature/orchestrator-arena-pr",
        progress_interval_seconds=1,
        allow_unsafe_windows_wsl_fallback=False,
        step_position=3,
        step_total=4,
    )

    prompt = str(captured["prompt"])
    assert "Review this PR-ready branch diff" in prompt
    assert "Review this implementation slice" not in prompt
    assert "manual_prompt_mode" not in dict(captured["review_scope"])
    assert dict(captured["round_payload"])["task_class"] == "pr_review"
    assert (
        dict(captured["round_payload"])["task_id_hint"]
        == "feature/orchestrator-arena-pr"
    )
    assert "Review this PR-ready branch diff" in str(
        dict(captured["round_payload"])["requested_prompt"]
    )
    assert dict(captured["round_payload"])["review_scope"] == dict(
        captured["review_scope"]
    )
    assert dict(captured["round_payload"])["review_cwd"] == str(repo)
    assert dict(captured["round_payload"])["progress_interval_seconds"] == 1
    assert select_calls[0]["task_class"] == "pr_review"
    assert select_calls[0]["rating_pool_id"] == "arena-deep-gpt-5.6-v1"
    assert select_calls[0]["variant_groups"] == [["a", "b", "c", "d"]]
    assert select_calls[0]["variant_ids"] == ["a", "b", "c", "d", "e"]
    assert result["lane"] == "review_t3"
    assert result["reviewed_head"] == head
    assert result["arena_round"] is True
    assert result["grading_required"] is True
    assert result["needs_grade"] is True
    assert result["graded"] is False
    assert anchor_calls == []


def test_decision_action_surfaces_arena_grade_command(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    round_state_dir = state_dir / "orchestrator" / "review-rounds"
    state = {
        "public_id": "rvw_example",
        "stage": "decision-pending",
        "pending_action": {
            "kind": "decision",
            "round_id": "arena-round-1",
            "lane": "review_t3",
        },
        "identity": {"branch": "feature/arena"},
        "deslop": {"tracked": False, "status": "closed"},
        "rounds": [
            {
                "round_id": "arena-round-1",
                "lane": "review_t3",
                "grading_required": True,
                "arena_round": True,
                "status": "completed",
                "task_id_hint": "feature/arena",
                "round_state_dir": str(round_state_dir),
            }
        ],
    }
    write_round(
        round_state_dir,
        {
            "round_id": "arena-round-1",
            "arena_round": True,
            "status": "completed",
            "task_id_hint": "feature/arena",
            "rating_pool_id": "arena-deep-gpt-5.6-v1",
            "runs": [
                {"slot": "alpha", "review_status": "completed", "grade_blocked": False}
            ],
        },
    )

    action = review._action_payload(state, state_dir=state_dir)

    assert action is not None
    assert "review_suite_arena.py grade" in str(action["cmd"])
    assert "--round-id arena-round-1" in str(action["cmd"])
    assert "--task-id feature/arena" in str(action["cmd"])
    assert "--rating-pool-id arena-deep-gpt-5.6-v1" in str(action["cmd"])
    assert str(action["cmd"]).count("--rank") == 2
    assert "--basis BASIS" in str(action["cmd"])
    assert "--state-dir" in str(action["cmd"])
    assert str(round_state_dir) in str(action["cmd"])
    assert "--id rvw_example" in str(action["next"])
    assert "alt" not in action


def test_arena_recovery_action_surfaces_single_reroll(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    state = {
        "public_id": "rvw_example",
        "stage": "retry-requested",
        "pending_action": {
            "kind": "arena-blocked",
            "round_id": "arena-round-1",
            "lane": "review_t3",
            "step_index": 0,
            "step": "arena-discovery",
        },
        "identity": {"cwd": str(repo), "base": "main", "branch": "feature/arena"},
        "deslop": {"tracked": False, "status": "closed"},
        "rounds": [
            {
                "round_id": "arena-round-1",
                "lane": "review_t3",
                "grading_required": True,
                "arena_round": True,
                "status": "completed",
                "runs": [{"slot": "alpha", "blocked": True}],
            }
        ],
    }

    action = review._action_payload(state, state_dir=tmp_path / "state")

    assert action is not None
    assert "review.py" in str(action["cmd"])
    assert "--id rvw_example" in str(action["cmd"])
    assert "review_suite_arena.py" not in str(action)
    backend = review._arena_recovery_backend_argv(state, state_dir=tmp_path / "state")
    assert backend is not None
    assert "reroll-slot" in backend
    assert "--round-id" in backend
    assert "arena-round-1" in backend
    assert "--slot" in backend
    assert "alpha" in backend


def test_arena_recovery_action_rerolls_one_blocked_slot_at_a_time(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    state = {
        "public_id": "rvw_example",
        "stage": "retry-requested",
        "pending_action": {
            "kind": "arena-blocked",
            "round_id": "arena-round-1",
            "lane": "review_t1",
        },
        "identity": {"cwd": str(repo), "base": "main", "branch": "feature/arena"},
        "deslop": {"tracked": False, "status": "closed"},
        "rounds": [
            {
                "round_id": "arena-round-1",
                "lane": "review_t1",
                "grading_required": False,
                "arena_round": True,
                "status": "completed",
                "runs": [
                    {"slot": "bravo", "blocked": True},
                    {"slot": "charlie", "blocked": True},
                ],
            }
        ],
    }

    action = review._action_payload(state, state_dir=tmp_path / "state")

    assert action is not None
    assert "review.py" in str(action["cmd"])
    assert "--id rvw_example" in str(action["cmd"])
    assert "review_suite_arena.py" not in str(action)
    backend = review._arena_recovery_backend_argv(state, state_dir=tmp_path / "state")
    assert backend is not None
    assert "reroll-slot" in backend
    assert "bravo" in backend
    assert "dismiss-round" not in backend


def test_arena_recovery_action_advances_after_dismissed_blocked_round(
    tmp_path: Path,
) -> None:
    state = {
        "public_id": "rvw_example",
        "stage": "retry-requested",
        "pending_action": {
            "kind": "arena-blocked",
            "round_id": "arena-round-1",
            "lane": "review_t1",
        },
        "identity": {"branch": "feature/arena"},
        "deslop": {"tracked": False, "status": "closed"},
        "rounds": [
            {
                "round_id": "arena-round-1",
                "lane": "review_t1",
                "grading_required": False,
                "arena_round": True,
                "status": "dismissed",
                "runs": [{"slot": "charlie", "blocked": True}],
            }
        ],
    }

    action = review._action_payload(state, state_dir=tmp_path / "state")

    assert action is not None
    assert "--id rvw_example" in str(action["cmd"])
    assert "reroll-slot" not in str(action)
    assert "dismiss-round" not in str(action)
    assert (
        review._arena_recovery_backend_argv(state, state_dir=tmp_path / "state") is None
    )


def test_arena_recovery_action_surfaces_run_for_sampled_replacement(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    state = {
        "public_id": "rvw_example",
        "stage": "retry-requested",
        "pending_action": {
            "kind": "arena-blocked",
            "round_id": "arena-round-2",
            "lane": "review_t3",
            "step_index": 0,
            "step": "arena-discovery",
        },
        "identity": {"cwd": str(repo), "base": "main", "branch": "feature/arena"},
        "deslop": {"tracked": False, "status": "closed"},
        "rounds": [
            {
                "round_id": "arena-round-2",
                "lane": "review_t3",
                "grading_required": True,
                "arena_round": True,
                "status": "sampled",
                "runs": [{"slot": "alpha"}],
            }
        ],
    }

    action = review._action_payload(state, state_dir=tmp_path / "state")

    assert action is not None
    assert "review.py" in str(action["cmd"])
    assert "--id rvw_example" in str(action["cmd"])
    assert "review_suite_arena.py" not in str(action)
    backend = review._arena_recovery_backend_argv(state, state_dir=tmp_path / "state")
    assert backend is not None
    assert "run-round" in backend
    assert "arena-round-2" in backend


def test_arena_recovery_backend_runs_internal_action(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    state = {
        "public_id": "rvw_example",
        "stage": "retry-requested",
        "pending_action": {
            "kind": "arena-blocked",
            "round_id": "arena-round-2",
            "lane": "review_t3",
        },
        "identity": {"cwd": str(repo), "base": "main", "branch": "feature/arena"},
        "rounds": [
            {
                "round_id": "arena-round-2",
                "lane": "review_t3",
                "grading_required": True,
                "arena_round": True,
                "status": "sampled",
                "runs": [{"slot": "alpha"}],
            }
        ],
    }
    calls: list[list[str]] = []

    def fake_run(
        command: list[str], check: bool, **kwargs: object
    ) -> subprocess.CompletedProcess:
        assert check is False
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, stdout="child toon\n", stderr="child progress\n"
        )

    monkeypatch.setattr(review.subprocess, "run", fake_run)

    assert (
        review._run_arena_recovery_backend_once(state, state_dir=tmp_path / "state")
        is True
    )
    captured = capsys.readouterr()
    assert "child toon" not in captured.out
    assert "child progress" in captured.err
    assert calls
    assert "review_suite_arena.py" in calls[0][1]
    assert "run-round" in calls[0]


def test_arena_recovery_action_surfaces_resume_for_running_replacement(
    tmp_path: Path,
) -> None:
    state = {
        "public_id": "rvw_example",
        "stage": "retry-requested",
        "pending_action": {
            "kind": "arena-blocked",
            "round_id": "arena-round-2",
            "lane": "review_t3",
            "step_index": 0,
            "step": "arena-discovery",
        },
        "identity": {"branch": "feature/arena"},
        "runtime": {"allow_unsafe_windows_wsl_fallback": True},
        "deslop": {"tracked": False, "status": "closed"},
        "rounds": [
            {
                "round_id": "arena-round-2",
                "lane": "review_t3",
                "grading_required": True,
                "arena_round": True,
                "status": "running",
                "round_state_dir": str(tmp_path / "round-state"),
                "runs": [{"slot": "alpha"}],
            }
        ],
    }

    action = review._action_payload(state, state_dir=tmp_path / "state")

    assert action is not None
    assert "review.py" in str(action["cmd"])
    assert "--id rvw_example" in str(action["cmd"])
    assert "review_suite_arena.py" not in str(action)
    backend = review._arena_recovery_backend_argv(state, state_dir=tmp_path / "state")
    assert backend is not None
    assert "resume-round" in backend
    assert "arena-round-2" in backend
    assert "--wsl" not in backend


def test_decision_rejects_ungraded_arena_round(tmp_path: Path) -> None:
    state = {
        "public_id": "rvw_example",
        "stage": "decision-pending",
        "pending_action": {
            "kind": "decision",
            "round_id": "arena-round-1",
            "lane": "review_t3",
        },
        "identity": {"branch": "feature/arena"},
        "review_progress": {"next_step_index": 1, "completed_steps": []},
        "rounds": [
            {
                "round_id": "arena-round-1",
                "lane": "review_t3",
                "grading_required": True,
                "arena_round": True,
                "status": "completed",
                "reviewed_head": "head-1",
            }
        ],
    }

    with pytest.raises(ValueError, match="grade the arena round"):
        review._apply_decision(state, "clean", state_dir=tmp_path / "state")


def test_decision_rejects_blocked_review_round(tmp_path: Path) -> None:
    state = {
        "public_id": "rvw_example",
        "stage": "decision-pending",
        "pending_action": {
            "kind": "decision",
            "round_id": "blocked-round-1",
            "lane": "review_t1",
        },
        "identity": {"branch": "feature/blocked"},
        "review_progress": {"next_step_index": 1, "completed_steps": []},
        "rounds": [
            {
                "round_id": "blocked-round-1",
                "lane": "review_t1",
                "review_blocked": True,
                "status": "completed",
                "reviewed_head": "head-1",
                "runs": [
                    {
                        "slot": "alpha",
                        "review_status": "process_died",
                        "grade_blocked": True,
                    }
                ],
            }
        ],
    }

    with pytest.raises(ValueError, match="blocked review round"):
        review._apply_decision(state, "clean", state_dir=tmp_path / "state")


def test_create_resume_and_id_reprint_use_one_pending_action(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    deslop_calls = _stub_deslop(monkeypatch)
    review_calls = _stub_review(
        monkeypatch, "phase_review-round-1", "phase_review-round-2"
    )
    repo = tmp_path / "repo"
    state_dir = tmp_path / "default-state"
    _use_compact_normal_profile(monkeypatch, state_dir)
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/review-shell")
    _commit_file(repo, "app.txt", "feature\n", "feature")

    args = [
        "--mode",
        "normal",
        "--cd",
        str(repo),
        "--base",
        "main",
        "--state-dir",
        str(state_dir),
    ]
    exit_code, payload = _run_review(monkeypatch, args)
    public_id = str(payload["review"])

    assert exit_code == 0
    assert public_id.startswith("rvw_")
    assert "stage" not in payload
    assert "mode" not in payload
    assert "selection" not in payload
    assert "grading" not in payload
    choices = _assert_decision_action(payload["Action"])
    assert f"--id {public_id}" in choices["clean"]
    assert "--state-dir" not in choices["clean"]
    assert str(state_dir) not in choices["clean"]
    assert "--decision clean" in choices["clean"]
    assert len(deslop_calls) == 0
    assert not (state_dir / "orchestrator" / "state_dirs.json").exists()
    state = _cycle_payload(state_dir, public_id)
    assert state["selection"] == {
        "requested": "auto",
        "effective": "stable",
        "reason": "auto_stable_profile",
    }
    assert "grading" not in state
    assert state["deslop"]["status"] == "tracked"
    assert len(state["rounds"]) == 1
    assert len(review_calls) == 1

    exit_code, resumed = _run_review(monkeypatch, args)
    assert exit_code == 0
    assert resumed["review"] == public_id
    assert "stage" not in resumed
    choices = _assert_decision_action(resumed["Action"])
    assert "--decision clean" in choices["clean"]
    assert "--decision findings" in choices["findings"]
    assert "--state-dir" not in choices["clean"]
    assert "--state-dir" not in choices["findings"]
    assert str(state_dir) not in choices["clean"]
    assert str(state_dir) not in choices["findings"]
    assert len(list((state_dir / "orchestrator" / "cycles").glob("*.json"))) == 1
    state = _cycle_payload(state_dir, public_id)
    assert len(state["rounds"]) == 1
    assert state["rounds"][0]["round_id"] == "phase_review-round-1"
    assert state["rounds"][0]["lane"] == "review_t1"
    assert state["rounds"][0]["review_status"] == "completed"
    assert state["rounds"][0]["output_refs"] == ["rollout://phase_review-round-1/alpha"]
    assert len(review_calls) == 1

    exit_code, by_id = _run_review(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )
    assert exit_code == 0
    assert by_id["review"] == public_id
    assert by_id["Action"] == resumed["Action"]
    assert len(_cycle_payload(state_dir, public_id)["rounds"]) == 1
    assert len(review_calls) == 1

    exit_code, first_clean = _run_review(
        monkeypatch,
        ["--id", public_id, "--decision", "clean", "--state-dir", str(state_dir)],
    )
    assert exit_code == 0
    assert first_clean["review"] == public_id
    assert "stage" not in first_clean
    assert f"--id {public_id}" in str(first_clean["Action"]["cmd"])
    assert "--state-dir" not in str(first_clean["Action"]["cmd"])
    assert str(state_dir) not in str(first_clean["Action"]["cmd"])
    assert "--decision" not in str(first_clean["Action"]["cmd"])
    state = _cycle_payload(state_dir, public_id)
    assert state["pending_action"] == {
        "kind": "run-review-step",
        "step_index": 1,
        "step": "precision-signoff",
    }
    assert state["review_progress"]["completed_steps"] == [
        {
            "index": 0,
            "name": "broad-discovery",
            "round_id": "phase_review-round-1",
            "lane": "review_t1",
            "reviewed_head": state["rounds"][0]["reviewed_head"],
        }
    ]
    assert len(review_calls) == 1

    exit_code, second_step = _run_review(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )
    assert "--deslop-done" in second_step["Action"]["cmd"]
    _run_review(monkeypatch, ["--id", public_id, "--deslop-done"])
    exit_code, second_step = _run_review(monkeypatch, ["--id", public_id])
    assert exit_code == 0
    assert second_step["review"] == public_id
    assert "stage" not in second_step
    assert "--decision clean" in _assert_decision_action(second_step["Action"])["clean"]
    assert len(review_calls) == 2

    assert review_calls[0]["step_name"] == "broad-discovery"
    assert review_calls[0]["step_position"] == 1
    assert review_calls[0]["step_total"] == 2
    assert review_calls[1]["step_name"] == "precision-signoff"
    assert review_calls[1]["step_position"] == 2
    assert review_calls[1]["step_total"] == 2
    state = _cycle_payload(state_dir, public_id)
    assert len(state["rounds"]) == 2
    assert state["rounds"][1]["round_id"] == "phase_review-round-2"

    exit_code, final_clean = _run_review(
        monkeypatch,
        ["--id", public_id, "--decision", "clean", "--state-dir", str(state_dir)],
    )
    assert exit_code == 0
    assert "stage" not in final_clean
    _, closure = _run_review(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )
    assert closure["conformance"] == "NOT_APPLICABLE"
    _, final_clean = _run_review(monkeypatch, ["--id", public_id, "--deslop-done"])
    _assert_github_handoff(
        final_clean["Action"],
        public_id=public_id,
        state_dir=state_dir,
        blocked_by=["full_suite:unknown", "ci:unknown"],
    )
    state = _cycle_payload(state_dir, public_id)
    assert state["pending_action"] is None
    assert state["validation"]["review_green"] == "passed"
    assert [
        item["round_id"] for item in state["review_progress"]["completed_steps"]
    ] == [
        "phase_review-round-1",
        "phase_review-round-2",
    ]

    exit_code, pending_validation = _run_review(
        monkeypatch,
        [
            "--id",
            public_id,
            "--focused-validation",
            "passed",
            "--full-suite",
            "pending",
            "--ci",
            "pending",
            "--state-dir",
            str(state_dir),
        ],
    )
    assert exit_code == 0
    assert "stage" not in pending_validation
    _assert_github_handoff(
        pending_validation["Action"],
        public_id=public_id,
        state_dir=state_dir,
        blocked_by=["full_suite:pending", "ci:pending"],
    )
    state = _cycle_payload(state_dir, public_id)
    assert state["validation"]["focused"] == "passed"
    assert state["validation"]["full_suite"] == "pending"
    assert state["validation"]["ci"] == "pending"
    assert len(deslop_calls) == 1
    assert len(review_calls) == 2

    exit_code, validation_ready = _run_review(
        monkeypatch,
        [
            "--id",
            public_id,
            "--full-suite",
            "passed",
            "--ci",
            "waived",
            "--validation-note",
            "CI unavailable for this docs-only change",
            "--state-dir",
            str(state_dir),
        ],
    )
    assert exit_code == 0
    _assert_github_handoff(
        validation_ready["Action"],
        public_id=public_id,
        state_dir=state_dir,
        blocked_by=[],
    )
    state = _cycle_payload(state_dir, public_id)
    assert state["validation"]["full_suite"] == "passed"
    assert state["validation"]["ci"] == "waived"
    assert state["validation"]["note"] == "CI unavailable for this docs-only change"
    assert len(deslop_calls) == 1
    assert len(review_calls) == 2


def test_new_cycle_defaults_to_normal_without_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_deslop(monkeypatch)
    _stub_review(monkeypatch)
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")

    _, created = _run_review(
        monkeypatch,
        [
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )

    state = _cycle_payload(state_dir, str(created["review"]))
    assert state["mode"] == {"requested": "normal", "effective": "normal"}
    mode_action = next(
        action for action in review.build_parser()._actions if action.dest == "mode"
    )
    assert tuple(mode_action.choices) == ("fast", "normal", "deep")


def test_head_change_waits_for_failed_deslop_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    deslop_calls = _stub_deslop(monkeypatch, 9, 0)
    review_calls = _stub_review(monkeypatch)
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _use_single_step_normal_profile(monkeypatch, state_dir)
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/deslop-head-change")
    _commit_file(repo, "app.txt", "feature\n", "feature")

    _, created = _run_review(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(created["review"])
    fixed_head = _commit_file(repo, "app.txt", "feature\nfix\n", "fix review")

    _run_review(monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)])

    state = _cycle_payload(state_dir, public_id)
    assert state["deslop"]["status"] == "done"
    assert state["deslop"]["reviewed_head"] == fixed_head
    assert state["review_progress"]["completed_steps"] == []
    assert state["identity"]["head"] == fixed_head
    assert len(deslop_calls) == 2
    assert review_calls == []
    _run_review(monkeypatch, ["--id", public_id, "--deslop-done"])
    _run_review(monkeypatch, ["--id", public_id])
    assert review_calls[0]["review_scope"]["reviewed_head"] == fixed_head


@pytest.mark.parametrize("flag", ["--skip-deslop", "--no-deslop"])
def test_create_with_skip_deslop_runs_review_without_sidecar(
    flag: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    deslop_calls = _stub_deslop(monkeypatch)
    review_calls = _stub_review(monkeypatch)
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _use_compact_normal_profile(monkeypatch, state_dir)
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/no-deslop")
    _commit_file(repo, "app.txt", "feature\n", "feature")

    exit_code, payload = _run_review(
        monkeypatch,
        [
            "--mode",
            "fast",
            flag,
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    _, resumed = _run_review(
        monkeypatch,
        [
            "--mode",
            "fast",
            flag,
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )

    assert exit_code == 0
    public_id = str(payload["review"])
    assert resumed["review"] == public_id
    assert len(deslop_calls) == 0
    assert len(review_calls) == 1
    assert len(list((state_dir / "orchestrator" / "cycles").glob("*.json"))) == 1
    state = _cycle_payload(state_dir, public_id)
    assert state["deslop"] == {"tracked": False, "status": "skipped", "source": "cli"}
    assert state["rounds"][0]["round_id"] == "phase_review-round-1"


def test_review_brief_is_frozen_and_public_output_reports_coverage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_review(monkeypatch)
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")

    _, created = _run_review(
        monkeypatch,
        [
            "--skip-deslop",
            "--mode",
            "fast",
            "--review-brief",
            "# Goal\n\nKeep it neutral.",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(created["review"])

    assert _cycle_payload(state_dir, public_id)["review_brief"] == (
        "# Goal\n\nKeep it neutral."
    )
    assert created["review_brief"] == "available"

    errors: list[str] = []
    monkeypatch.setattr(
        review,
        "emit_error",
        lambda message, **kwargs: errors.append(str(message)) or 2,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review.py",
            "--skip-deslop",
            "--mode",
            "fast",
            "--review-brief",
            "Changed goal",
            "--cd",
            str(repo),
            "--base",
            "main",
        ],
    )

    assert review.main() == 2
    assert errors[-1] == (
        "review brief is frozen for this cycle; use "
        f"--id {public_id} --restart-brief <MARKDOWN> --reason <WHY> to replace it"
    )

    terminal = _cycle_payload(state_dir, public_id)
    terminal["stage"] = "aborted"
    _write_cycle_payload(state_dir, public_id, terminal)
    errors.clear()
    assert review.main() == 2
    assert "review brief is frozen" in errors[-1]


def test_skip_deslop_does_not_resume_same_head_sidecar_cycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    deslop_calls = _stub_deslop(monkeypatch)
    review_calls = _stub_review(monkeypatch)
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _use_compact_normal_profile(monkeypatch, state_dir)
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/no-deslop")
    _commit_file(repo, "app.txt", "feature\n", "feature")

    _, sidecar = _run_review(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    _, skipped = _run_review(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--skip-deslop",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )

    assert sidecar["review"] != skipped["review"]
    assert len(deslop_calls) == 0
    assert len(review_calls) == 2
    assert len(list((state_dir / "orchestrator" / "cycles").glob("*.json"))) == 2
    assert (
        _cycle_payload(state_dir, str(sidecar["review"]))["deslop"]["status"]
        == "tracked"
    )
    assert _cycle_payload(state_dir, str(skipped["review"]))["deslop"] == {
        "tracked": False,
        "status": "skipped",
        "source": "cli",
    }


def test_id_auto_records_structured_clean_and_runs_next_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_deslop(monkeypatch)
    review_calls = _stub_review_with_terminal(monkeypatch, "clean", "clean")
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _use_compact_normal_profile(monkeypatch, state_dir)
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/auto-clean")
    _commit_file(repo, "app.txt", "feature\n", "feature")

    _, created = _run_review_after_cleanup(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(created["review"])
    state = _cycle_payload(state_dir, public_id)
    assert len(review_calls) == 1
    assert state["decisions"][0]["command"] == "clean"
    assert state["pending_action"] == {
        "kind": "run-review-step",
        "step_index": 1,
        "step": "precision-signoff",
    }

    exit_code, final = _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )

    assert exit_code == 0
    assert len(review_calls) == 2
    _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )
    _, final = _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--deslop-done"]
    )
    _assert_github_handoff(
        final["Action"],
        public_id=public_id,
        state_dir=state_dir,
        blocked_by=["full_suite:unknown", "ci:unknown"],
    )
    state = _cycle_payload(state_dir, public_id)
    assert [item["command"] for item in state["decisions"]] == ["clean", "clean"]
    assert state["validation"]["review_green"] == "passed"


def test_id_auto_records_structured_findings_before_fix_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_deslop(monkeypatch)
    review_calls = _stub_review_with_terminal(monkeypatch, "findings", "clean")
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _use_compact_normal_profile(monkeypatch, state_dir)
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/auto-findings")
    _commit_file(repo, "app.txt", "feature\n", "feature")

    _, created = _run_review_after_cleanup(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--review-brief",
            "# Goal\n\nKeep the implementation within the frozen contract.",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(created["review"])

    exit_code, findings = _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )

    assert exit_code == 0
    assert "--contract-conflict <dimension>" in findings["Action"]["note"]
    assert "--fixes-validated" in str(findings["Action"]["choices"])
    state = _cycle_payload(state_dir, public_id)
    assert state["decisions"][0]["command"] == "findings"
    assert state["active_findings"]["status"] == "fix-pending"
    assert len(review_calls) == 1

    _commit_file(repo, "app.txt", "fixed\n", "fix findings")
    exit_code, clean = _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )

    assert exit_code == 0
    assert "--decision" not in str(clean["Action"]["cmd"])
    assert len(review_calls) == 2
    state = _cycle_payload(state_dir, public_id)
    assert [item["command"] for item in state["decisions"]] == ["findings", "clean"]
    assert state["pending_action"] is None
    assert state["validation"]["review_green"] == "passed"


def test_auto_decision_requires_terminal_command(tmp_path: Path) -> None:
    state = {
        "stage": "decision-pending",
        "pending_action": {
            "kind": "decision",
            "round_id": "round-1",
            "lane": "review_t1",
        },
        "rounds": [
            {
                "round_id": "round-1",
                "lane": "review_t1",
                "status": "decision-pending",
                "review_status": "completed",
                "runs": [
                    {
                        "slot": "alpha",
                        "status": "completed",
                        "reviewer_output": "Review decision: clean",
                    }
                ],
            }
        ],
    }

    assert review._auto_decision_command(state, state_dir=tmp_path / "state") is None


def test_auto_decision_rejects_blocked_terminal_round(tmp_path: Path) -> None:
    state = {
        "stage": "decision-pending",
        "pending_action": {
            "kind": "decision",
            "round_id": "round-1",
            "lane": "review_t1",
        },
        "rounds": [
            {
                "round_id": "round-1",
                "lane": "review_t1",
                "status": "decision-pending",
                "review_status": "completed",
                "runs": [
                    {
                        "slot": "alpha",
                        "status": "completed",
                        "terminal_command": "clean",
                        "blocked": True,
                    }
                ],
            }
        ],
    }

    assert review._auto_decision_command(state, state_dir=tmp_path / "state") is None


def test_auto_decision_waits_for_arena_grade(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    round_state_dir = state_dir / "orchestrator" / "review-rounds"
    write_round(
        round_state_dir,
        {
            "round_id": "arena-round-1",
            "status": "completed",
            "arena_round": True,
            "runs": [
                {
                    "slot": "alpha",
                    "review_status": "completed",
                    "terminal_command": "clean",
                    "grade_blocked": False,
                }
            ],
        },
    )
    state = {
        "stage": "decision-pending",
        "pending_action": {
            "kind": "decision",
            "round_id": "arena-round-1",
            "lane": "review_t3",
        },
        "rounds": [
            {
                "round_id": "arena-round-1",
                "lane": "review_t3",
                "status": "decision-pending",
                "review_status": "completed",
                "grading_required": True,
                "arena_round": True,
                "round_state_dir": str(round_state_dir),
            }
        ],
    }

    assert review._auto_decision_command(state, state_dir=state_dir) is None


def test_pending_ungraded_arena_amend_still_records_findings(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    round_state_dir = state_dir / "orchestrator" / "review-rounds"
    write_round(
        round_state_dir,
        {
            "round_id": "arena-round-1",
            "status": "completed",
            "arena_round": True,
            "runs": [
                {
                    "slot": "alpha",
                    "review_status": "completed",
                    "reviewer_output": "P1 finding.",
                    "grade_blocked": False,
                }
            ],
        },
    )
    state = {
        "stage": "decision-pending",
        "pending_action": {
            "kind": "decision",
            "round_id": "arena-round-1",
            "lane": "review_t3",
        },
        "review_heads": {},
        "rounds": [
            {
                "round_id": "arena-round-1",
                "lane": "review_t3",
                "status": "decision-pending",
                "reviewed_head": "head-1",
                "review_status": "completed",
                "grading_required": True,
                "arena_round": True,
                "round_state_dir": str(round_state_dir),
            }
        ],
        "decisions": [],
        "active_findings": None,
    }

    fixed = review._auto_record_pending_decision_fix(
        state, current_head_value="head-2", state_dir=state_dir
    )

    assert fixed["decisions"][0]["command"] == "findings"
    assert fixed["decisions"][0]["reviewed_head"] == "head-1"
    assert fixed["review_heads"]["last_fix_head"] == "head-2"


def test_wsl_flag_persists_to_orchestrated_steps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    deslop_calls = _stub_deslop(monkeypatch)
    review_calls = _stub_review(monkeypatch, "phase_review-round-1")
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _use_single_step_normal_profile(monkeypatch, state_dir)
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/wsl-review")
    _commit_file(repo, "app.txt", "feature\n", "feature")

    args = [
        "--mode",
        "normal",
        "--cd",
        str(repo),
        "--base",
        "main",
        "--state-dir",
        str(state_dir),
        "--wsl",
    ]
    exit_code, payload = _run_review(monkeypatch, args)
    public_id = str(payload["review"])

    assert exit_code == 0
    assert len(deslop_calls) == 1
    state = _cycle_payload(state_dir, public_id)
    assert state["runtime"] == {"allow_unsafe_windows_wsl_fallback": True}

    _run_review(monkeypatch, ["--id", public_id, "--deslop-done"])
    exit_code, _resumed = _run_review(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )

    assert exit_code == 0
    assert review_calls[0]["allow_unsafe_windows_wsl_fallback"] is True

    _run_review(monkeypatch, ["--id", public_id, "--decision", "clean"])
    _run_review(monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)])
    assert "--wsl" in deslop_calls[0]


def test_state_dir_flag_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    errors: list[str] = []
    monkeypatch.setattr(
        review, "emit_error", lambda message, **kwargs: errors.append(str(message)) or 2
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["review.py", "--id", "rvw_example", "--state-dir", str(tmp_path / "state")],
    )

    assert review.main() == 2
    assert "unrecognized arguments: --state-dir" in errors[-1]


def test_public_id_allocation_preserves_index_updates_written_before_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state_dir = tmp_path / "state"
    existing_key = "orc-aaaaaaaaaaaaaaaa"
    new_key = "orc-bbbbbbbbbbbbbbbb"

    class Lock:
        def __enter__(self) -> None:
            orchestrator_store._write_index(
                state_dir,
                {
                    "schema_version": orchestrator_store.ORCHESTRATOR_INDEX_SCHEMA_VERSION,
                    "ids": {"rvw_aaaaaaaa": existing_key},
                    "cycle_keys": {existing_key: "rvw_aaaaaaaa"},
                },
            )

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(
        orchestrator_store, "orchestrator_store_lock", lambda **kwargs: Lock()
    )

    assert (
        orchestrator_store.public_id_for_cycle_key(state_dir, new_key) == "rvw_bbbbbbbb"
    )
    index = orchestrator_store.load_index(state_dir)
    assert index["ids"] == {
        "rvw_aaaaaaaa": existing_key,
        "rvw_bbbbbbbb": new_key,
    }
    assert index["cycle_keys"] == {
        existing_key: "rvw_aaaaaaaa",
        new_key: "rvw_bbbbbbbb",
    }


def test_id_show_findings_prefers_open_cleanup_then_latest_review_without_running(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _stub_deslop(monkeypatch)
    review_calls = _stub_review(monkeypatch, "phase_review-round-1")
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/show-findings")
    _commit_file(repo, "app.txt", "feature\n", "feature")

    args = [
        "--mode",
        "normal",
        "--cd",
        str(repo),
        "--base",
        "main",
        "--state-dir",
        str(state_dir),
    ]
    _, created = _run_review_after_cleanup(monkeypatch, args)
    public_id = str(created["review"])
    _run_review_after_cleanup(monkeypatch, args)
    state = _cycle_payload(state_dir, public_id)
    round_id = str(state["rounds"][0]["round_id"])
    round_state_dir = state_dir / "orchestrator" / "review-rounds"
    state["rounds"][0]["round_state_dir"] = str(round_state_dir)
    state["rounds"].append(
        {
            "round_id": "empty-later-round",
            "lane": "review_t1",
            "round_state_dir": str(round_state_dir),
            "runs": [],
        }
    )
    state["pending_action"] = {
        "kind": "decision",
        "round_id": "empty-later-round",
        "lane": "review_t1",
    }
    _write_cycle_payload(state_dir, public_id, state)
    write_round(
        round_state_dir,
        {
            "round_id": round_id,
            "task_class": "phase_review",
            "public_task": "review_t1",
            "review_cwd": str(repo),
            "runs": [
                {
                    "slot": "alpha",
                    "review_status": "completed",
                    "reviewer_output": "Alpha recovered finding",
                    "reviewer_output_ref": "rollout://phase_review-round-1/alpha",
                }
            ],
        },
    )
    write_round(
        round_state_dir,
        {
            "round_id": "empty-later-round",
            "task_class": "phase_review",
            "status": "completed",
            "runs": [],
        },
    )
    before_calls = len(review_calls)
    monkeypatch.setattr(
        review,
        "emit_toon",
        lambda payload: (_ for _ in ()).throw(AssertionError("should not emit status")),
    )
    monkeypatch.setattr(review, "default_state_dir", lambda: state_dir)
    monkeypatch.setattr(
        sys, "argv", ["review.py", "--id", public_id, "--show-findings"]
    )

    exit_code = review.main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"review: {public_id}" in captured.out
    assert "round_id:" not in captured.out
    assert "lane:" not in captured.out
    assert "task:" not in captured.out
    assert "status:" not in captured.out
    assert "Alpha recovered finding" in captured.out

    state["deslop"].update(
        tracked=True, status="done", findings="Closure recovered finding"
    )
    _write_cycle_payload(state_dir, public_id, state)
    assert review.main() == 0
    captured = capsys.readouterr()
    assert "Closure recovered finding" in captured.out
    assert "Alpha recovered finding" not in captured.out
    state["deslop"].update(tracked=False, status="closed")
    _write_cycle_payload(state_dir, public_id, state)
    assert review.main() == 0
    captured = capsys.readouterr()
    assert "Alpha recovered finding" in captured.out
    assert "Closure recovered finding" not in captured.out
    assert (
        _cycle_payload(state_dir, public_id)["deslop"]["findings"]
        == "Closure recovered finding"
    )
    assert len(review_calls) == before_calls


def test_id_show_status_reports_cycle_without_advancing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    review_calls = _stub_review(monkeypatch, "phase_review-round-1")

    def fail_deslop(*, command: list[str], cwd: Path) -> subprocess.CompletedProcess:
        raise AssertionError("show-status fixture must not run deslop")

    _stub_deslop(monkeypatch)
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/show-status")
    _commit_file(repo, "app.txt", "feature\n", "feature")

    _, created = _run_review_after_cleanup(
        monkeypatch,
        [
            "--mode",
            "fast",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(created["review"])
    monkeypatch.setattr(orchestrator_runner, "run_deslop_subprocess", fail_deslop)
    before_state = _cycle_payload(state_dir, public_id)
    before_calls = len(review_calls)
    (repo / "scratch.txt").write_text("dirty\n", encoding="utf-8")

    exit_code, payload = _run_review(
        monkeypatch, ["--id", public_id, "--show-status", "--state-dir", str(state_dir)]
    )

    assert exit_code == 0
    assert payload["review"] == public_id
    assert payload["status"] == "decision-pending"
    assert payload["mode"] == "fast"
    assert payload["cwd"] == str(repo)
    assert payload["base"] == "main"
    assert payload["branch"] == "feature/show-status"
    assert payload["head"] == str(dict(before_state["identity"])["head"])[:12]
    assert (
        payload["merge_base"] == str(dict(before_state["identity"])["merge_base"])[:12]
    )
    assert payload["rounds"] == 1
    assert payload["deslop"] == "skipped"
    assert payload["review_brief"] == "unavailable"
    assert payload["design_conformance_context"] == "unavailable"
    assert dict(payload["worktree"]) == {
        "branch": "feature/show-status",
        "head": str(dict(before_state["identity"])["head"])[:12],
        "dirty": True,
    }
    assert payload["Action"] == created["Action"]
    assert len(review_calls) == before_calls
    assert _cycle_payload(state_dir, public_id) == before_state


@pytest.mark.parametrize(
    ("reviewer_is_live", "expected_runtime"),
    [(True, "active"), (False, "collection_pending")],
)
def test_review_runtime_label(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reviewer_is_live: bool,
    expected_runtime: str,
) -> None:
    monkeypatch.setattr(
        review,
        "_load_output_round_payload",
        lambda *_args, **_kwargs: {"status": "running", "runs": [{"pid": 123}]},
    )
    monkeypatch.setattr(
        review,
        "round_has_live_reviewer_process",
        lambda payload: reviewer_is_live,
    )
    state = {
        "pending_action": {"kind": "collect-review-step", "round_id": "round-1"},
        "rounds": [{"round_id": "round-1"}],
    }

    assert review._review_runtime_label(state, state_dir=tmp_path) == expected_runtime


def test_show_status_requires_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    errors: list[str] = []
    monkeypatch.setattr(
        review, "emit_error", lambda message, **kwargs: errors.append(str(message)) or 2
    )
    monkeypatch.setattr(review, "default_state_dir", lambda: tmp_path / "state")
    monkeypatch.setattr(sys, "argv", ["review.py", "--show-status"])

    assert review.main() == 2
    assert errors[-1] == "--show-status requires --id"


def test_id_collects_running_round_without_spawning_duplicate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_deslop(monkeypatch)
    review_calls = _stub_review(
        monkeypatch, "phase_review-round-1", "phase_review-round-duplicate"
    )
    resume_calls: list[dict[str, object]] = []
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/running-review")
    _commit_file(repo, "app.txt", "feature\n", "feature")

    args = [
        "--mode",
        "normal",
        "--cd",
        str(repo),
        "--base",
        "main",
        "--state-dir",
        str(state_dir),
    ]
    _, created = _run_review_after_cleanup(monkeypatch, args)
    public_id = str(created["review"])
    _run_review_after_cleanup(monkeypatch, args)

    state = _cycle_payload(state_dir, public_id)
    round_id = str(state["rounds"][0]["round_id"])
    state["stage"] = "running"
    state["pending_action"] = {
        "kind": "collect-review-step",
        "round_id": round_id,
        "lane": "review_t1",
        "step_index": 0,
        "step": "broad-discovery",
        "round_state_dir": "state/orchestrator/review-rounds",
    }
    state["rounds"][0]["status"] = "running"
    _write_cycle_payload(state_dir, public_id, state)

    def fake_resume(**kwargs: object) -> dict[str, object]:
        resume_calls.append(dict(kwargs))
        return {
            "round_id": round_id,
            "lane": "review_t1",
            "kind": "review",
            "status": "completed",
            "blocked": False,
            "reviewed_head": state["rounds"][0]["reviewed_head"],
            "output_refs": ["rollout://phase_review-round-1/resumed"],
            "runs": [
                {
                    "slot": "alpha",
                    "status": "completed",
                    "summary": "No findings.",
                    "ref": "rollout://phase_review-round-1/resumed",
                    "blocked": False,
                    "block": None,
                }
            ],
            "round_state_dir": "state/orchestrator/review-rounds",
        }

    monkeypatch.setattr(orchestrator_runner, "resume_review_step", fake_resume)

    exit_code, payload = _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )

    assert exit_code == 0
    assert payload["review"] == public_id
    assert len(review_calls) == 1
    assert len(resume_calls) == 1
    assert resume_calls[0]["round_id"] == round_id
    saved = _cycle_payload(state_dir, public_id)
    assert saved["stage"] == "decision-pending"
    assert saved["pending_action"]["kind"] == "decision"
    assert [item["round_id"] for item in saved["rounds"]] == [round_id]
    assert saved["rounds"][0]["output_refs"] == [
        "rollout://phase_review-round-1/resumed"
    ]


def test_restart_mode_supersedes_cycle_and_starts_fresh_deep_ladder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    deslop_calls = _stub_deslop(monkeypatch)
    review_calls = _stub_review(monkeypatch, "normal-round-1", "deep-round-1")
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _use_compact_normal_profile(monkeypatch, state_dir, include_deep=True)
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/restart-deep")
    _commit_file(repo, "app.txt", "feature\n", "feature")

    _, created = _run_review(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
            "--wsl",
        ],
    )
    old_id = str(created["review"])
    _run_review(monkeypatch, ["--id", old_id, "--state-dir", str(state_dir)])

    exit_code, restarted = _run_review(
        monkeypatch,
        [
            "--id",
            old_id,
            "--restart-mode",
            "deep",
            "--reason",
            "review round budget exhausted",
            "--state-dir",
            str(state_dir),
        ],
    )

    new_id = str(restarted["review"])
    assert exit_code == 0
    assert new_id != old_id
    assert f"--id {new_id}" in _assert_decision_action(restarted["Action"])["clean"]
    assert len(review_calls) == 2
    assert len(list((state_dir / "orchestrator" / "cycles").glob("*.json"))) == 2

    old_state = _cycle_payload(state_dir, old_id)
    new_state = _cycle_payload(state_dir, new_id)
    assert old_state["stage"] == "aborted"
    assert old_state["recovery"]["status"] == "aborted"
    assert old_state["superseded_by"] == {
        "review": new_id,
        "cycle_key": new_state["cycle_key"],
        "mode": "deep",
        "reason": "review round budget exhausted",
        "kind": "mode-restart",
    }
    assert new_state["mode"] == {"requested": "deep", "effective": "deep"}
    assert new_state["identity"] == old_state["identity"]
    assert new_state["runtime"] == {"allow_unsafe_windows_wsl_fallback": True}
    assert new_state["cycle_key"] != old_state["cycle_key"]
    assert new_state["restart"]["token"] == f"{old_state['cycle_key']}:deep"
    assert new_state["restart"]["supersedes"] == old_id
    assert new_state["restart"]["supersedes_cycle_key"] == old_state["cycle_key"]
    assert new_state["restart"]["from_mode"] == "normal"
    assert review._is_successor_cycle(new_state) is True
    assert review._successor_needs_locked_resume(new_state) is False
    assert [step["name"] for step in new_state["review_plan"]["steps"]] == [
        "broad-discovery",
        "precision-signoff",
        "deep-discovery",
        "deep-signoff",
    ]

    messages: list[str] = []
    monkeypatch.setattr(
        review,
        "emit_error",
        lambda message, **kwargs: messages.append(str(message)) or 2,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["review.py", "--id", old_id, "--new-cycle"],
    )
    assert review.main() == 2
    assert "unrecognized arguments: --new-cycle" in messages[-1]

    old_state_without_redirect = dict(old_state)
    old_state_without_redirect.pop("superseded_by")
    _write_cycle_payload(state_dir, old_id, old_state_without_redirect)
    _, retry = _run_review(
        monkeypatch,
        [
            "--id",
            old_id,
            "--restart-mode",
            "deep",
            "--reason",
            "retry after partial write",
            "--state-dir",
            str(state_dir),
        ],
    )
    assert retry["review"] == new_id
    assert len(deslop_calls) == 0
    assert len(review_calls) == 2

    _, old_reprint = _run_review(
        monkeypatch, ["--id", old_id, "--state-dir", str(state_dir)]
    )
    assert f"--id {new_id}" in str(old_reprint["Action"]["cmd"])
    assert "superseded" in str(old_reprint["Action"]["note"])

    _, deep_review = _run_review(
        monkeypatch, ["--id", new_id, "--state-dir", str(state_dir)]
    )
    assert len(review_calls) == 2
    assert review_calls[1]["step_name"] == "broad-discovery"
    assert review_calls[1]["allow_unsafe_windows_wsl_fallback"] is True
    assert review_calls[1]["step_position"] == 1
    assert review_calls[1]["step_total"] == 4


def test_restart_brief_supersedes_invalid_cycle_with_fresh_same_mode_ladder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_deslop(monkeypatch)
    _use_compact_normal_profile(monkeypatch, tmp_path / "state", include_deep=True)
    review_calls = _stub_review(monkeypatch, "old-round", "replacement-round")
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/restart-brief")
    _commit_file(repo, "app.txt", "feature\n", "feature")

    _, created = _run_review(
        monkeypatch,
        [
            "--mode",
            "deep",
            "--review-brief",
            "C:/tmp/brief.md",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    old_id = str(created["review"])
    _run_review(
        monkeypatch,
        [
            "--id",
            old_id,
            "--focused-validation",
            "passed",
            "--full-suite",
            "passed",
            "--ci",
            "passed",
            "--state-dir",
            str(state_dir),
        ],
    )
    old_state = _cycle_payload(state_dir, old_id)
    assert old_state["stage"] == "decision-pending"
    assert old_state["validation"] == {
        "review_green": "unknown",
        "focused": "passed",
        "full_suite": "passed",
        "ci": "passed",
    }
    old_rounds = deepcopy(old_state["rounds"])
    _git(repo, "commit", "--allow-empty", "-m", "audit review input")
    current_head = _git(repo, "rev-parse", "HEAD")
    replacement_brief = "# Goal\n\nReview the complete contract."

    exit_code, restarted = _run_review(
        monkeypatch,
        [
            "--id",
            old_id,
            "--restart-brief",
            replacement_brief,
            "--reason",
            "the frozen brief was a file path",
            "--state-dir",
            str(state_dir),
        ],
    )

    new_id = str(restarted["review"])
    assert exit_code == 0
    assert new_id != old_id
    assert len(review_calls) == 2
    old_saved = _cycle_payload(state_dir, old_id)
    new_state = _cycle_payload(state_dir, new_id)
    assert old_saved["stage"] == "aborted"
    assert old_saved["review_brief"] == "C:/tmp/brief.md"
    assert old_saved["rounds"] == old_rounds
    assert old_saved["superseded_by"] == {
        "review": new_id,
        "cycle_key": new_state["cycle_key"],
        "mode": "deep",
        "reason": "the frozen brief was a file path",
        "kind": "review-brief-restart",
    }
    assert new_state["review_brief"] == replacement_brief
    assert new_state["mode"] == {"requested": "deep", "effective": "deep"}
    assert new_state["identity"]["head"] == current_head
    assert new_state["identity"]["merge_base"] == old_state["identity"]["merge_base"]
    assert new_state["validation"] == {
        "review_green": "unknown",
        "focused": "unknown",
        "full_suite": "unknown",
        "ci": "unknown",
    }
    assert new_state["deslop"] == {"tracked": True, "status": "tracked"}
    assert new_state["decisions"] == []
    assert new_state["convergence"]["accepted_findings_heads"] == []
    assert [item["round_id"] for item in new_state["rounds"]] == ["replacement-round"]
    assert new_state["restart"]["token"] == (
        f"{old_state['cycle_key']}:review-brief-restart"
    )
    assert new_state["restart"]["kind"] == "review-brief-restart"


def test_restart_brief_cannot_reset_accepted_findings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_review(monkeypatch)
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")

    _, created = _run_review(
        monkeypatch,
        [
            "--skip-deslop",
            "--mode",
            "fast",
            "--review-brief",
            "Mistaken brief",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(created["review"])
    _run_review(
        monkeypatch,
        [
            "--id",
            public_id,
            "--decision",
            "findings",
            "--state-dir",
            str(state_dir),
        ],
    )
    before = _cycle_payload(state_dir, public_id)
    errors: list[str] = []
    monkeypatch.setattr(
        review,
        "emit_error",
        lambda message, **kwargs: errors.append(str(message)) or 2,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review.py",
            "--id",
            public_id,
            "--restart-brief",
            "Replacement brief",
            "--reason",
            "replace input",
        ],
    )

    assert review.main() == 2
    assert errors[-1] == (
        "cannot replace review brief after findings or convergence activity; "
        "follow the existing review action"
    )
    assert _cycle_payload(state_dir, public_id) == before
    assert len(list((state_dir / "orchestrator" / "cycles").glob("*.json"))) == 1


def test_contract_conflict_and_one_use_continue_are_durable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    review_calls = _stub_review(
        monkeypatch, "fast-round-1", "fast-round-2", "next-plan-round"
    )
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/convergence")
    _commit_file(repo, "app.txt", "feature\n", "feature")

    _, opened = _run_review_after_cleanup(
        monkeypatch,
        [
            "--skip-deslop",
            "--mode",
            "fast",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(opened["review"])
    _run_review_after_cleanup(
        monkeypatch,
        ["--id", public_id, "--decision", "findings", "--state-dir", str(state_dir)],
    )

    _, conflict = _run_review_after_cleanup(
        monkeypatch,
        [
            "--id",
            public_id,
            "--contract-conflict",
            "scope",
            "--state-dir",
            str(state_dir),
        ],
    )
    assert conflict["convergence"]["reason"] == "contract_conflict"
    assert set(conflict["Action"]["choices"]) == {"CONTINUE", "REPLAN", "RESLICE"}

    _git(repo, "commit", "--allow-empty", "-m", "tree-identical head")
    _run_review_after_cleanup(
        monkeypatch,
        [
            "--id",
            public_id,
            "--convergence-decision",
            "continue",
            "--state-dir",
            str(state_dir),
        ],
    )
    continued_state = _cycle_payload(state_dir, public_id)
    assert continued_state["stage"] == "fix-pending"
    assert continued_state["review_heads"]["last_fix_head"] is None
    assert len(review_calls) == 1
    _run_review_after_cleanup(
        monkeypatch,
        [
            "--id",
            public_id,
            "--convergence-decision",
            "continue",
            "--state-dir",
            str(state_dir),
        ],
    )
    _commit_file(repo, "app.txt", "feature\nfix\n", "fix")
    _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )
    _, exhausted = _run_review_after_cleanup(
        monkeypatch,
        ["--id", public_id, "--decision", "findings", "--state-dir", str(state_dir)],
    )

    assert set(exhausted["Action"]["choices"]) == {"REPLAN", "RESLICE"}
    _commit_file(repo, "app.txt", "feature\nfix\nnext\n", "next fix")
    _, redirected = _run_review_after_cleanup(
        monkeypatch,
        [
            "--skip-deslop",
            "--mode",
            "deep",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    assert redirected["review"] == public_id
    _, replanned = _run_review_after_cleanup(
        monkeypatch,
        [
            "--id",
            public_id,
            "--convergence-decision",
            "replan",
            "--state-dir",
            str(state_dir),
        ],
    )
    assert replanned["convergence"]["decision"] == "REPLAN"
    state = _cycle_payload(state_dir, public_id)
    assert [item["decision"] for item in state["convergence"]["decisions"]] == [
        "CONTINUE",
        "REPLAN",
    ]
    _, next_plan = _run_review_after_cleanup(
        monkeypatch,
        [
            "--skip-deslop",
            "--mode",
            "fast",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    assert next_plan["review"] != public_id
    assert len(review_calls) == 3


@pytest.mark.parametrize("fix_before_classification", [False, True])
@pytest.mark.parametrize("fourth_decision", ["clean", "findings"])
def test_continue_consumes_material_fix_head_at_budget_stop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fix_before_classification: bool,
    fourth_decision: str,
) -> None:
    deslop_calls = _stub_deslop(monkeypatch)
    review_calls = _stub_review(
        monkeypatch, "round-1", "round-2", "round-3", "authorized-round"
    )
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/closure-order")
    _commit_file(repo, "app.txt", "feature\n", "feature")

    _, opened = _run_review_after_cleanup(
        monkeypatch,
        [
            "--mode",
            "fast",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(opened["review"])
    for index in (1, 2):
        _run_review_after_cleanup(
            monkeypatch,
            [
                "--id",
                public_id,
                "--decision",
                "findings",
                "--state-dir",
                str(state_dir),
            ],
        )
        _commit_file(
            repo,
            "app.txt",
            "feature\n" + "".join(f"fix-{n}\n" for n in range(1, index + 1)),
            f"fix {index}",
        )
        _run_review_after_cleanup(
            monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
        )

    if fix_before_classification:
        _commit_file(
            repo, "app.txt", "feature\nfix-1\nfix-2\nclosure-fix\n", "closure fix"
        )
    _, breaker = _run_review_after_cleanup(
        monkeypatch,
        ["--id", public_id, "--state-dir", str(state_dir)]
        + ([] if fix_before_classification else ["--decision", "findings"]),
    )
    assert breaker["convergence"]["reason"] == "budget_exhausted"
    assert len(review_calls) == 3

    authorized_head = (
        _git(repo, "rev-parse", "HEAD")
        if fix_before_classification
        else _commit_file(
            repo, "app.txt", "feature\nfix-1\nfix-2\nclosure-fix\n", "closure fix"
        )
    )
    _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )
    blocked_state = _cycle_payload(state_dir, public_id)
    assert blocked_state["convergence"]["status"] == "DECISION_REQUIRED"
    assert len(blocked_state["convergence"]["accepted_findings_heads"]) == 3
    assert blocked_state["review_heads"]["last_fix_head"] != authorized_head
    assert len(review_calls) == 3

    _run_review_after_cleanup(
        monkeypatch,
        [
            "--id",
            public_id,
            "--convergence-decision",
            "continue",
            "--state-dir",
            str(state_dir),
        ],
    )
    continued_state = _cycle_payload(state_dir, public_id)
    assert continued_state["review_heads"]["last_fix_head"] == authorized_head
    assert continued_state["convergence"]["continue_pending"] is True
    assert len(review_calls) == 3

    _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )
    assert review_calls[-1]["review_scope"]["reviewed_head"] == authorized_head
    _, classified = _run_review_after_cleanup(
        monkeypatch,
        [
            "--id",
            public_id,
            "--decision",
            fourth_decision,
            "--state-dir",
            str(state_dir),
        ],
    )
    if fourth_decision == "findings":
        assert set(classified["Action"]["choices"]) == {"REPLAN", "RESLICE"}
    else:
        clean = _cycle_payload(state_dir, public_id)
        assert len(deslop_calls) == 0
        assert clean["deslop"]["status"] == "skipped"
        assert classified["done"] is True
        assert clean["validation"]["full_suite"] == "unknown"
        assert len(clean["convergence"]["accepted_findings_heads"]) == 3
        _, closure = _run_review_after_cleanup(
            monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
        )
        assert closure["done"] is True
        assert len(deslop_calls) == 0


def test_repeat_continue_restores_completed_unclassified_round(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    review_calls = _stub_review(monkeypatch, "round-4")
    deslop_calls = _stub_deslop(monkeypatch)
    repo, state_dir = tmp_path / "repo", tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/continue")
    _commit_file(repo, "app.txt", "feature\n", "feature")
    _, opened = _run_review_after_cleanup(
        monkeypatch,
        [
            "--mode",
            "fast",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(opened["review"])
    args = ["--id", public_id, "--state-dir", str(state_dir)]
    pending = _cycle_payload(state_dir, public_id)
    legacy = deepcopy(pending)
    for index in range(3):
        legacy = review.record_findings_decision(
            legacy,
            round_id=f"round-{index}",
            lane="review_t1",
            reviewed_head=f"head-{index}",
        )
    legacy = review.record_convergence_decision(legacy, decision="CONTINUE")
    legacy["active_findings"] = None
    legacy["review_heads"] = pending["review_heads"]
    legacy["rounds"] = [*legacy["rounds"][1:], legacy["rounds"][0]]
    legacy["pending_action"] = {"kind": "fix-findings", "round_id": None}
    orchestrator_store.save_cycle(state_dir, legacy)

    for _ in range(2):
        code, output = _run_review_after_cleanup(
            monkeypatch, [*args, "--convergence-decision", "continue"]
        )
        assert code == 0
        _assert_decision_action(output["Action"])
        recovered = _cycle_payload(state_dir, public_id)
        assert recovered["stage"] == pending["stage"]
        assert recovered["pending_action"] == pending["pending_action"]
        for key in (
            "rounds",
            "decisions",
            "convergence",
            "identity",
            "deslop",
            "validation",
            "review_heads",
        ):
            assert recovered[key] == legacy[key]
    assert len(review_calls) == 1
    assert len(deslop_calls) == 0
    code, _ = _run_review_after_cleanup(monkeypatch, [*args, "--decision", "clean"])
    assert code == 0
    clean = _cycle_payload(state_dir, public_id)
    assert clean["decisions"][-1]["round_id"] == "round-4"
    assert clean["decisions"][-1]["command"] == "clean"
    assert len(clean["convergence"]["decisions"]) == 1
    assert clean["validation"]["full_suite"] == "unknown"
    assert clean["deslop"]["status"] == "skipped"


@pytest.mark.parametrize("skip_cleanup", [False, True])
def test_fast_review_can_restart_into_deep_without_becoming_a_restart_target(
    skip_cleanup: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    deslop_calls = _stub_deslop(monkeypatch)
    review_calls = _stub_review(monkeypatch, "fast-round-1", "deep-round-1")
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _use_compact_normal_profile(monkeypatch, state_dir, include_deep=True)
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/restart-fast")
    _commit_file(repo, "app.txt", "feature\n", "feature")

    _, created = _run_review(
        monkeypatch,
        [
            "--mode",
            "fast",
            *(["--skip-deslop"] if skip_cleanup else []),
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    old_id = str(created["review"])

    exit_code, restarted = _run_review(
        monkeypatch,
        [
            "--id",
            old_id,
            "--restart-mode",
            "deep",
            "--reason",
            "fast review needs deeper analysis",
            "--state-dir",
            str(state_dir),
        ],
    )

    new_id = str(restarted["review"])
    assert exit_code == 0
    assert new_id != old_id
    assert len(deslop_calls) == 0
    assert len(review_calls) == 2
    assert _cycle_payload(state_dir, old_id)["stage"] == "aborted"
    new_state = _cycle_payload(state_dir, new_id)
    assert new_state["mode"] == {"requested": "deep", "effective": "deep"}
    assert new_state["restart"]["from_mode"] == "fast"
    assert new_state["deslop"] == (
        {"tracked": False, "status": "skipped", "source": "cli"}
        if skip_cleanup
        else {"tracked": True, "status": "tracked"}
    )

    restart_action = next(
        action
        for action in review.build_parser()._actions
        if action.dest == "restart_mode"
    )
    assert tuple(restart_action.choices) == ("normal", "deep")


def test_restart_mode_requires_escalation_reason_and_stricter_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_deslop(monkeypatch)
    _stub_review(monkeypatch)
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")

    _, created = _run_review(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(created["review"])

    messages: list[str] = []
    monkeypatch.setattr(
        review,
        "emit_error",
        lambda message, **kwargs: messages.append(str(message)) or 2,
    )

    monkeypatch.setattr(
        sys, "argv", ["review.py", "--id", public_id, "--restart-mode", "deep"]
    )
    assert review.main() == 2
    assert messages[-1] == "--reason is required for --restart-mode"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review.py",
            "--id",
            public_id,
            "--restart-mode",
            "normal",
            "--reason",
            "try downgrade",
        ],
    )
    assert review.main() == 2
    assert "--restart-mode must increase strictness" in messages[-1]


def test_restart_mode_rejects_dirty_worktree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_deslop(monkeypatch)
    _stub_review(monkeypatch)
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")

    _, created = _run_review(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(created["review"])
    (repo / "app.txt").write_text("dirty\n", encoding="utf-8")

    messages: list[str] = []
    monkeypatch.setattr(
        review,
        "emit_error",
        lambda message, **kwargs: messages.append(str(message)) or 2,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review.py",
            "--id",
            public_id,
            "--restart-mode",
            "deep",
            "--reason",
            "rerun deeper",
        ],
    )

    assert review.main() == 2
    assert (
        messages[-1]
        == "cannot restart review cycle with a dirty worktree; commit or stash changes, then rerun"
    )
    assert "superseded_by" not in _cycle_payload(state_dir, public_id)
    assert len(list((state_dir / "orchestrator" / "cycles").glob("*.json"))) == 1


def test_restart_mode_rejects_changed_head(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_deslop(monkeypatch)
    _stub_review(monkeypatch)
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")

    _, created = _run_review(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(created["review"])
    _commit_file(repo, "app.txt", "changed head\n", "change head")

    messages: list[str] = []
    monkeypatch.setattr(
        review,
        "emit_error",
        lambda message, **kwargs: messages.append(str(message)) or 2,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review.py",
            "--id",
            public_id,
            "--restart-mode",
            "deep",
            "--reason",
            "rerun deeper",
        ],
    )

    assert review.main() == 2
    assert (
        messages[-1]
        == "cannot restart review cycle after HEAD changed; start a new review instead"
    )
    assert "superseded_by" not in _cycle_payload(state_dir, public_id)
    assert len(list((state_dir / "orchestrator" / "cycles").glob("*.json"))) == 1


def test_review_orchestrator_help_hides_internal_selection() -> None:
    help_text = review.build_parser().format_help()

    assert "--mode" in help_text
    assert "--restart-mode" in help_text
    assert "--selection" not in help_text
    assert "--state-dir" not in help_text


@pytest.mark.parametrize("changed_identity", ["head", "merge_base"])
def test_closure_dismissal_persists_without_bypassing_gates_or_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    changed_identity: str,
) -> None:
    deslop_calls = _stub_deslop(monkeypatch)
    review_calls = _stub_review(monkeypatch, "phase_review-round-1")
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/closure-drift")
    advanced_base = _commit_file(repo, "app.txt", "base\nstep\n", "stack step")
    head = _commit_file(repo, "app.txt", "base\nstep\nhead\n", "feature head")

    _, created = _run_review(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(created["review"])
    _, completed = _run_review(monkeypatch, ["--id", public_id, "--decision", "clean"])
    _, completed = _run_review(monkeypatch, ["--id", public_id])

    assert "--deslop-done" in str(completed["Action"]["cmd"])
    assert len(deslop_calls) == 1
    drifted = _cycle_payload(state_dir, public_id)
    drifted["deslop"].update(
        conformance="MATERIALLY_DRIFTED",
        decision="findings",
        findings="Reviewer claims the implementation violates scope.",
    )
    _write_cycle_payload(state_dir, public_id, drifted)
    _, pending = _run_review(monkeypatch, ["--id", public_id, "--show-status"])
    assert "--deslop-done --reason" in pending["Action"]["choices"]["dismiss"]

    reason = "The contract permits this implementation."
    dismiss_args = ["--id", public_id, "--deslop-done", "--reason", reason]
    with monkeypatch.context() as dirty:
        errors: list[str] = []
        dirty.setattr(
            review, "emit_error", lambda message, **kw: errors.append(message) or 2
        )
        dirty.setattr(sys, "argv", ["review.py", *dismiss_args])
        (repo / "app.txt").write_text("dirty\n", encoding="utf-8")
        assert review.main() == 2
        assert "exact clean branch, HEAD, and merge-base" in errors[-1]
        assert _cycle_payload(state_dir, public_id) == drifted
    (repo / "app.txt").write_text("base\nstep\nhead\n", encoding="utf-8")

    exit_code, dismissed = _run_review(monkeypatch, dismiss_args)
    assert exit_code == 0
    assert dismissed["review"] == public_id
    assert dismissed["done"] is False
    assert dismissed["conformance"] == "MATERIALLY_DRIFTED"
    assert dismissed["closure_dismissal_reason"] == reason
    assert review_calls == []
    assert (
        _cycle_payload(state_dir, public_id)["validation"]["review_green"] == "unknown"
    )
    closed = _cycle_payload(state_dir, public_id)
    assert closed["deslop"] == {
        **drifted["deslop"],
        "tracked": False,
        "status": "closed",
        "dismissal_reason": reason,
    }
    assert {**closed, "deslop": drifted["deslop"]} == drifted
    _run_review(monkeypatch, ["--id", public_id])
    _run_review(monkeypatch, ["--id", public_id, "--decision", "clean"])
    _, waived = _run_review(
        monkeypatch,
        ["--id", public_id, "--github-result", "waived", "--github-note", "local test"],
    )
    assert waived["done"] is False
    _run_review(
        monkeypatch, ["--id", public_id, "--full-suite", "passed", "--ci", "passed"]
    )
    _, status = _run_review(monkeypatch, ["--id", public_id, "--show-status"])
    assert status["done"] is True
    assert status["deslop"] == "closed"
    assert status["conformance"] == "MATERIALLY_DRIFTED"
    assert status["closure_dismissal_reason"] == reason

    if changed_identity == "merge_base":
        _git(repo, "branch", "-f", "main", advanced_base)
    else:
        head = _commit_file(repo, "app.txt", "changed\n", "changed head")
        advanced_base = _git(repo, "rev-parse", "main")
    exit_code, resumed = _run_review(monkeypatch, dismiss_args)

    assert exit_code == 0
    assert resumed["review"] == public_id
    state = _cycle_payload(state_dir, public_id)
    assert state["identity"]["head"] == head
    assert state["identity"]["merge_base"] == advanced_base
    assert state["stage"] == "created"
    assert state["pending_action"]["kind"] == "run-review-step"
    assert state["deslop"]["status"] == "closed"
    assert state["deslop"]["dismissal_reason"] == reason
    assert len(deslop_calls) == 1
    assert len(review_calls) == 1


@pytest.mark.parametrize(
    "change", ["empty_commit", "message_amend", "equivalent_rebase"]
)
def test_cleanup_drift_requires_dismissal_after_changes_without_material_fixes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, change: str
) -> None:
    cleanup_calls = _stub_deslop(monkeypatch)
    reviews = _stub_review(monkeypatch)
    repo, state_dir = tmp_path / "repo", tmp_path / "state"
    _use_single_step_normal_profile(monkeypatch, state_dir)
    _init_repo(repo)
    _commit_file(repo, "README.md", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/drift")
    head = _commit_file(repo, "app.txt", "feature\n", "feature")
    _, opened = _run_review(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(opened["review"])
    state = _cycle_payload(state_dir, public_id)
    state["deslop"].update(conformance="MATERIALLY_DRIFTED", decision="findings")
    _write_cycle_payload(state_dir, public_id, state)
    if change == "equivalent_rebase":
        _git(repo, "checkout", "main")
        _commit_file(repo, "README.md", "new base\n", "main advances")
        _git(repo, "checkout", "feature/drift")
        _git(repo, "rebase", "main")
    elif change == "empty_commit":
        _git(repo, "commit", "--allow-empty", "-m", "no code changes")
    else:
        _git(repo, "commit", "--amend", "-m", "new message")
    assert _git(repo, "rev-parse", "HEAD") != head
    _, pending = _run_review(monkeypatch, ["--id", public_id])
    assert "dismiss" in pending["Action"]["choices"]
    assert _cycle_payload(state_dir, public_id)["deslop"]["status"] == "done"
    if change == "equivalent_rebase":
        for extra_args in ([], ["--amend"]):
            _git(
                repo,
                "commit",
                "--allow-empty",
                *extra_args,
                "-m",
                "still no cleanup fixes",
            )
            _, pending = _run_review(monkeypatch, ["--id", public_id])
            assert "dismiss" in pending["Action"]["choices"]
            assert _cycle_payload(state_dir, public_id)["deslop"]["status"] == "done"
    errors = []
    monkeypatch.setattr(
        review, "emit_error", lambda message, **kwargs: errors.append(message) or 2
    )
    monkeypatch.setattr(sys, "argv", ["review.py", "--id", public_id, "--deslop-done"])
    assert review.main() == 2
    assert "explicit dismissal" in errors[-1]
    if change == "equivalent_rebase":
        _amend_file(repo, "app.txt", "simplified\n")
    else:
        _run_review(
            monkeypatch,
            [
                "--id",
                public_id,
                "--deslop-done",
                "--reason",
                "Allowed by the frozen brief",
            ],
        )
    _run_review(monkeypatch, ["--id", public_id])
    assert _cycle_payload(state_dir, public_id)["deslop"]["status"] == "closed"
    assert len(reviews) == 1
    assert len(cleanup_calls) == 1


def test_deslop_done_is_primary_action_when_no_other_action_remains() -> None:
    state = {"deslop": {"tracked": True, "status": "done"}}
    state_dir = Path("state")

    action = review._with_deslop_done_action(
        state, None, "rvw_example", state_dir=state_dir
    )

    assert action == {
        "cmd": review._review_command(
            "rvw_example", "--deslop-done", state_dir=state_dir
        ),
        "note": "Acknowledge the exact-head closure before continuing.",
    }


def test_deslop_done_requires_id_and_rejects_other_actions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    errors: list[str] = []
    monkeypatch.setattr(
        review, "emit_error", lambda message, **kwargs: errors.append(str(message)) or 2
    )
    monkeypatch.setattr(review, "default_state_dir", lambda: tmp_path / "state")
    monkeypatch.setattr(sys, "argv", ["review.py", "--deslop-done"])

    assert review.main() == 2
    assert errors[-1] == "--deslop-done requires --id"

    for read_only_flag in ("--show-findings", "--show-status"):
        monkeypatch.setattr(
            sys,
            "argv",
            ["review.py", "--id", "rvw_example", "--deslop-done", read_only_flag],
        )

        assert review.main() == 2
        assert "--deslop-done cannot be combined" in errors[-1]


def test_review_step_output_is_not_reprinted_by_review_py(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    review_calls: list[dict[str, object]] = []

    def fake_run(**kwargs: object) -> dict[str, object]:
        review_calls.append(dict(kwargs))
        print("Output:\nalpha:\nReviewer body.")
        return {
            "round_id": "phase_review-round-1",
            "lane": "review_t1",
            "kind": "review",
            "status": "completed",
            "blocked": False,
            "reviewed_head": "head-1",
            "output_refs": ["rollout://phase_review-round-1/alpha"],
            "runs": [
                {
                    "slot": "alpha",
                    "status": "completed",
                    "summary": "Reviewer body.",
                    "ref": "rollout://phase_review-round-1/alpha",
                    "blocked": False,
                    "block": None,
                }
            ],
        }

    monkeypatch.setattr(orchestrator_runner, "run_review_step", fake_run)
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")

    exit_code, payload = _run_review_after_cleanup(
        monkeypatch,
        [
            "--skip-deslop",
            "--mode",
            "fast",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "stage" not in payload
    assert captured.out.count("Output:") == 1
    assert "Reviewer body." in captured.out
    assert len(review_calls) == 1


def test_id_rejects_creation_context_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_deslop(monkeypatch)
    _stub_review(monkeypatch)
    repo = tmp_path / "repo"
    state_dir = tmp_path / "default-state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")

    _, created = _run_review(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    errors: list[tuple[str, dict[str, object]]] = []

    def fake_error(message: str, **kwargs: object) -> int:
        errors.append((message, dict(kwargs)))
        return 2

    monkeypatch.setattr(review, "emit_error", fake_error)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review.py",
            "--id",
            str(created["review"]),
            "--mode",
            "deep",
            "--cd",
            str(repo),
            "--base",
            "main",
        ],
    )

    exit_code = review.main()

    assert exit_code == 2
    assert len(errors) == 1
    message = errors[0][0]
    assert "--id already selects review context" in message
    assert "remove --mode, --cd, --base" in message
    assert "mode normal" in message
    assert str(repo.resolve()) in message


def test_id_rejects_model_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_deslop(monkeypatch)
    _stub_review(monkeypatch)
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")

    _, created = _run_review(
        monkeypatch,
        ["--mode", "fast", "--cd", str(repo), "--base", "main"],
    )
    errors: list[tuple[str, dict[str, object]]] = []

    def fake_error(message: str, **kwargs: object) -> int:
        errors.append((message, dict(kwargs)))
        return 2

    monkeypatch.setattr(review, "emit_error", fake_error)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review.py",
            "--id",
            str(created["review"]),
            "--model",
            "opencode-go/deepseek-v4.1-flash",
        ],
    )

    exit_code = review.main()

    assert exit_code == 2
    assert "remove --model" in errors[0][0]


def test_reject_model_override_on_existing_names_review() -> None:
    with pytest.raises(ValueError, match="frozen plan"):
        review._reject_model_override_on_existing(
            {"model": "gpt-6-astra"}, {"public_id": "rvw_example"}
        )

    review._reject_model_override_on_existing({}, {"public_id": "rvw_example"})


def test_status_rejects_model_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    errors: list[tuple[str, dict[str, object]]] = []

    def fake_error(message: str, **kwargs: object) -> int:
        errors.append((message, dict(kwargs)))
        return 2

    monkeypatch.setattr(review, "emit_error", fake_error)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review.py",
            "--status",
            "--model",
            "opencode-go/deepseek-v4.1-flash",
            "--cd",
            str(repo),
        ],
    )

    exit_code = review.main()

    assert exit_code == 2
    assert "--status cannot be combined" in errors[0][0]


def test_github_review_rejects_cycle_before_local_green(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_deslop(monkeypatch)
    _stub_review(monkeypatch)
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")

    _, created = _run_review(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(created["review"])
    _run_review(monkeypatch, ["--id", public_id, "--decision", "clean"])
    errors: list[tuple[str, dict[str, object]]] = []

    def fake_error(message: str, **kwargs: object) -> int:
        errors.append((message, dict(kwargs)))
        return 2

    monkeypatch.setattr(review, "emit_error", fake_error)
    monkeypatch.setattr(
        sys, "argv", ["review.py", "--id", public_id, "--github-review"]
    )

    exit_code = review.main()

    assert exit_code == 2
    assert errors == [
        (
            "--github-review requires local green review state",
            {
                "status": "usage_error",
                "help_items": [review._help_command()],
            },
        )
    ]


def test_github_review_runs_existing_lane_with_canonical_state_dir_and_force(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    review_calls = _stub_review(monkeypatch)
    _stub_deslop(monkeypatch)
    repo = tmp_path / "repo"
    state_dir = tmp_path / "default-state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")

    _, opened = _run_review_after_cleanup(
        monkeypatch,
        [
            "--mode",
            "fast",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(opened["review"])
    _, clean = _run_review_after_cleanup(
        monkeypatch,
        ["--id", public_id, "--decision", "clean", "--state-dir", str(state_dir)],
    )
    assert "stage" not in clean
    _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )
    _run_review_after_cleanup(monkeypatch, ["--id", public_id, "--deslop-done"])

    calls: list[list[str]] = []
    real_subprocess_run = review.subprocess.run

    def fake_subprocess_run(
        command: list[str], check: bool, **_kwargs: object
    ) -> subprocess.CompletedProcess:
        if command[0] == "git":
            return real_subprocess_run(command, check=check, **_kwargs)
        calls.append(command)
        return subprocess.CompletedProcess(command, 23)

    monkeypatch.setattr(review.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["review.py", "--id", public_id, "--github-review", "--github-force"],
    )

    exit_code = review.main()

    assert exit_code == 23
    assert calls == [
        [
            sys.executable,
            str((SCRIPT_DIR / "review_github.py").resolve()),
            "run",
            "--cd",
            str(repo.resolve()),
            "--state-dir",
            str(state_dir),
            "--force",
        ]
    ]
    assert len(review_calls) == 1


def test_github_result_findings_reenters_existing_cycle_for_final_signoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    deslop_calls = _stub_deslop(monkeypatch)
    review_calls = _stub_review(monkeypatch, "signoff-round-1", "signoff-round-2")
    followup_calls = _stub_followup(monkeypatch, "github-followup-1")
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _use_single_step_normal_profile(monkeypatch, state_dir)
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/github-findings")
    _commit_file(repo, "app.txt", "feature\n", "feature")

    _, opened = _run_review_after_cleanup(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(opened["review"])
    _run_review_after_cleanup(monkeypatch, ["--id", public_id, "--decision", "clean"])
    _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )
    _, green = _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--deslop-done"]
    )
    _assert_github_handoff(
        green["Action"],
        public_id=public_id,
        state_dir=state_dir,
        blocked_by=["full_suite:unknown", "ci:unknown"],
    )

    exit_code, github_findings = _run_review_after_cleanup(
        monkeypatch,
        [
            "--id",
            public_id,
            "--github-result",
            "findings",
            "--github-note",
            "GitHub found a stale replay bug.",
            "--state-dir",
            str(state_dir),
        ],
    )

    assert exit_code == 0
    assert github_findings["github_review"] == "findings"
    assert "--fixes-validated" in str(github_findings["Action"]["choices"])
    state = _cycle_payload(state_dir, public_id)
    assert state["active_findings"]["lane"] == "review-github"
    assert state["active_findings"]["profile_round_id"] == "signoff-round-1"
    assert state["validation"]["review_green"] == "unknown"

    fixed_head = _commit_file(repo, "app.txt", "feature\nfixed\n", "fix github finding")
    _, signoff = _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )
    assert len(followup_calls) == 0
    assert len(review_calls) == 2
    assert review_calls[1]["step_name"] == "precision-signoff"
    assert review_calls[1]["step_position"] == 1
    assert review_calls[1]["step_total"] == 1
    state = _cycle_payload(state_dir, public_id)
    assert state["pending_action"]["round_id"] == "signoff-round-2"
    assert state["review_progress"]["completed_steps"] == []

    _run_review_after_cleanup(monkeypatch, ["--id", public_id, "--decision", "clean"])
    _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )
    assert len(deslop_calls) == 1
    assert review_calls[-1]["review_scope"]["reviewed_head"] == fixed_head
    assert "--conformance-only" not in deslop_calls[-1]
    _, final_clean = _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--deslop-done"]
    )
    _assert_github_handoff(
        final_clean["Action"],
        public_id=public_id,
        state_dir=state_dir,
        blocked_by=["full_suite:unknown", "ci:unknown"],
    )
    assert (
        final_clean["Action"]["note"]
        == "GitHub findings were fixed and locally signed off; request GitHub review again."
    )


def test_github_result_clean_and_waived_are_terminal_for_existing_cycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_deslop(monkeypatch)
    _stub_review(monkeypatch)
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _use_single_step_normal_profile(monkeypatch, state_dir)
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")

    _, opened = _run_review_after_cleanup(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(opened["review"])
    _run_review_after_cleanup(monkeypatch, ["--id", public_id, "--decision", "clean"])
    _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )
    _run_review_after_cleanup(monkeypatch, ["--id", public_id, "--deslop-done"])

    exit_code, clean = _run_review_after_cleanup(
        monkeypatch,
        ["--id", public_id, "--github-result", "clean", "--state-dir", str(state_dir)],
    )

    assert exit_code == 0
    assert clean["github_review"] == "clean"
    assert clean["done"] is False
    assert clean["review_ladder"] == "pending"
    assert clean["next_action"] == "validation"
    assert clean["Action"]["blocked_by"] == ["full_suite:unknown", "ci:unknown"]
    assert "--full-suite FULL_SUITE_STATUS --ci CI_STATUS" in str(
        clean["Action"]["cmd"]
    )
    assert '--validation-note "reason"' in clean["Action"]["note"]
    assert "alt" not in clean["Action"]
    repair_command = review._validation_status_command(
        public_id, ["ci:waived_without_note"], state_dir=state_dir
    )
    assert "--ci waived --validation-note WAIVER_REASON" in repair_command

    fake_home = tmp_path / "home"
    default_state_dir = fake_home / ".codex" / "state" / "review-suite"
    shutil.copytree(state_dir, default_state_dir)
    before = _cycle_payload(default_state_dir, public_id)
    command = str(clean["Action"]["cmd"])
    invalid = subprocess.run(
        command if os.name == "nt" else shlex.split(command),
        capture_output=True,
        text=True,
        check=False,
        env=os.environ | {"HOME": str(fake_home), "USERPROFILE": str(fake_home)},
    )
    assert invalid.returncode == 2
    assert "invalid choice" in invalid.stdout
    assert _cycle_payload(default_state_dir, public_id) == before

    before_validation = _cycle_payload(state_dir, public_id)
    with pytest.raises(ValueError, match="invalid choice.*classified"):
        review.build_parser().parse_args(
            ["--id", public_id, "--full-suite", "classified"]
        )
    assert _cycle_payload(state_dir, public_id) == before_validation

    errors: list[str] = []
    monkeypatch.setattr(
        review,
        "emit_error",
        lambda message, **kwargs: errors.append(str(message)) or 2,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review.py",
            "--id",
            public_id,
            "--full-suite",
            "passed",
            "--ci",
            "waived",
        ],
    )
    monkeypatch.setattr(review, "default_state_dir", lambda: state_dir)
    exit_code = review.main()

    assert exit_code == 2
    assert errors == [
        "--validation-note is required when --full-suite or --ci is waived"
    ]
    assert _cycle_payload(state_dir, public_id) == before_validation

    exit_code, clean = _run_review_after_cleanup(
        monkeypatch,
        [
            "--id",
            public_id,
            "--full-suite",
            "passed",
            "--ci",
            "waived",
            "--validation-note",
            "CI unavailable for this docs-only change",
            "--state-dir",
            str(state_dir),
        ],
    )

    assert exit_code == 0
    assert clean["status"] == "done"
    assert clean["done"] is True
    assert clean["review_ladder"] == "complete"
    assert clean["next_action"] == "none"
    assert clean["github_review"] == "clean"
    assert clean["validation"]["note"] == "CI unavailable for this docs-only change"
    assert "Action" not in clean

    state = _cycle_payload(state_dir, public_id)
    state["github_review"] = {"status": "unknown"}
    state["validation"]["full_suite"] = "unknown"
    state["validation"]["ci"] = "unknown"
    _write_cycle_payload(state_dir, public_id, state)
    errors: list[tuple[str, dict[str, object]]] = []

    def fake_error(message: str, **kwargs: object) -> int:
        errors.append((message, dict(kwargs)))
        return 2

    monkeypatch.setattr(review, "emit_error", fake_error)
    monkeypatch.setattr(
        sys, "argv", ["review.py", "--id", public_id, "--github-result", "waived"]
    )

    assert review.main() == 2
    assert errors[0][0] == "--github-note is required when --github-result waived"

    monkeypatch.setattr(review, "emit_toon", lambda payload: None)
    exit_code, waived = _run_review_after_cleanup(
        monkeypatch,
        [
            "--id",
            public_id,
            "--github-result",
            "waived",
            "--github-note",
            "parent agent approved no-github after gh timeout",
            "--state-dir",
            str(state_dir),
        ],
    )

    assert exit_code == 0
    assert waived["github_review"] == "waived"
    assert waived["Action"]["blocked_by"] == ["full_suite:unknown", "ci:unknown"]

    reviewed_head = _git(repo, "rev-parse", "HEAD")
    stale_head = _commit_file(
        repo, "app.txt", "base\nnew work\n", "new work after github waiver"
    )
    exit_code, stale = _run_review_after_cleanup(
        monkeypatch,
        [
            "--id",
            public_id,
            "--full-suite",
            "passed",
            "--ci",
            "passed",
            "--state-dir",
            str(state_dir),
        ],
    )
    assert exit_code == 0
    assert stale["done"] is False
    assert stale["review_ladder"] == "pending"
    assert stale["next_action"] == "continue"
    assert "--full-suite" not in str(stale["Action"]["cmd"])
    state = _cycle_payload(state_dir, public_id)
    assert stale_head != reviewed_head and state["identity"]["head"] == stale_head
    assert state["github_review"] == {"status": "unknown"}
    assert state["validation"]["full_suite"] == "unknown"
    assert state["validation"]["ci"] == "unknown"
    assert state["deslop"]["status"] == "closed"


def test_github_result_findings_does_not_auto_start_followup_when_fix_already_committed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_deslop(monkeypatch)
    _stub_review(monkeypatch)
    followup_calls = _stub_followup(monkeypatch)
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/github-result-boundary")
    _commit_file(repo, "app.txt", "feature\n", "feature")

    _, opened = _run_review(
        monkeypatch,
        [
            "--mode",
            "fast",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(opened["review"])
    _run_review(
        monkeypatch,
        ["--id", public_id, "--decision", "clean", "--state-dir", str(state_dir)],
    )
    _commit_file(
        repo,
        "app.txt",
        "feature\nfix already committed\n",
        "fix before recording github result",
    )

    exit_code, findings = _run_review(
        monkeypatch,
        [
            "--id",
            public_id,
            "--github-result",
            "findings",
            "--state-dir",
            str(state_dir),
        ],
    )

    assert exit_code == 0
    assert len(followup_calls) == 0


def test_pending_github_review_after_amend_reuses_same_id_for_signoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_deslop(monkeypatch)
    review_calls = _stub_review(monkeypatch, "signoff-round-1", "signoff-round-2")
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _use_single_step_normal_profile(monkeypatch, state_dir)
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/github-amend")
    _commit_file(repo, "app.txt", "feature\n", "feature")

    _, opened = _run_review_after_cleanup(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(opened["review"])
    _, final_clean = _run_review_after_cleanup(
        monkeypatch,
        ["--id", public_id, "--decision", "clean", "--state-dir", str(state_dir)],
    )
    _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )
    _run_review_after_cleanup(monkeypatch, ["--id", public_id, "--deslop-done"])
    _run_review_after_cleanup(
        monkeypatch,
        [
            "--id",
            public_id,
            "--full-suite",
            "passed",
            "--ci",
            "passed",
            "--state-dir",
            str(state_dir),
        ],
    )
    amended_head = _amend_file(repo, "app.txt", "feature\nfix from github review\n")

    exit_code, status = _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--show-status", "--state-dir", str(state_dir)]
    )

    assert exit_code == 0
    assert status["review_ladder"] == "pending"
    assert status["next_action"] == "continue"
    assert status["current_head"] == amended_head
    assert status["Action"]["cmd"].endswith(f"review.py --id {public_id}")
    assert "--github-review" not in str(status["Action"]["cmd"])

    def fail_github_review(*args: object, **kwargs: object) -> int:
        raise AssertionError("--github-review must not run before amended head signoff")

    monkeypatch.setattr(review, "_run_github_review", fail_github_review)
    _, blocked_github = _run_review_after_cleanup(
        monkeypatch,
        ["--id", public_id, "--github-review", "--state-dir", str(state_dir)],
    )
    assert blocked_github["next_action"] == "continue"
    assert blocked_github["Action"]["cmd"].endswith(f"review.py --id {public_id}")
    assert "--github-review" not in str(blocked_github["Action"]["cmd"])

    _, rerun = _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )

    assert len(review_calls) == 2
    assert review_calls[1]["step_name"] == "precision-signoff"
    assert review_calls[1]["review_scope"]["reviewed_head"] == amended_head
    state = _cycle_payload(state_dir, public_id)
    assert state["validation"]["full_suite"] == "unknown"
    assert state["validation"]["ci"] == "unknown"

    _run_review_after_cleanup(
        monkeypatch,
        ["--id", public_id, "--decision", "clean", "--state-dir", str(state_dir)],
    )
    _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )
    _, final_clean = _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--deslop-done"]
    )
    _assert_github_handoff(
        final_clean["Action"],
        public_id=public_id,
        state_dir=state_dir,
        blocked_by=["full_suite:unknown", "ci:unknown"],
    )


def test_mode_rerun_after_pending_github_head_change_reuses_same_id_for_signoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_deslop(monkeypatch)
    review_calls = _stub_review(monkeypatch, "signoff-round-1", "signoff-round-2")
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _use_single_step_normal_profile(monkeypatch, state_dir)
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/github-head-change-mode-rerun")
    _commit_file(repo, "app.txt", "feature\n", "feature")

    _, opened = _run_review_after_cleanup(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(opened["review"])
    _run_review_after_cleanup(
        monkeypatch,
        ["--id", public_id, "--decision", "clean", "--state-dir", str(state_dir)],
    )
    changed_head = _commit_file(
        repo,
        "app.txt",
        "feature\nfix from github review\n",
        "fix github review finding",
    )

    exit_code, resumed = _run_review_after_cleanup(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )

    assert exit_code == 0
    assert resumed["review"] == public_id
    assert len(review_calls) == 2
    assert len(list((state_dir / "orchestrator" / "cycles").glob("*.json"))) == 1
    state = _cycle_payload(state_dir, public_id)
    assert state["identity"]["head"] == changed_head
    assert state["review_heads"]["last_fix_head"] == changed_head
    assert state["github_review"]["status"] == "unknown"


def test_patch_equivalent_rebase_after_cleanup_keeps_local_signoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_deslop(monkeypatch)
    review_calls = _stub_review(monkeypatch, "signoff-round-1")
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _use_single_step_normal_profile(monkeypatch, state_dir)
    _init_repo(repo)
    base_at_review = _commit_file(repo, "README.md", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/green-base-drift")
    original_head = _commit_file(repo, "src/app.txt", "feature\n", "feature")

    _, opened = _run_review_after_cleanup(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(opened["review"])
    _run_review_after_cleanup(
        monkeypatch,
        ["--id", public_id, "--decision", "clean", "--state-dir", str(state_dir)],
    )
    _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )
    _run_review_after_cleanup(monkeypatch, ["--id", public_id, "--deslop-done"])

    _git(repo, "checkout", "main")
    current_base = _commit_file(repo, "docs/notes.md", "main notes\n", "main moves")
    _git(repo, "checkout", "feature/green-base-drift")
    _git(repo, "rebase", "main")
    rebased_head = _git(repo, "rev-parse", "HEAD")

    exit_code, resumed = _run_review_after_cleanup(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )

    assert exit_code == 0
    assert resumed["review"] == public_id
    assert len(review_calls) == 1
    assert len(list((state_dir / "orchestrator" / "cycles").glob("*.json"))) == 1
    state = _cycle_payload(state_dir, public_id)
    assert state["stage"] == "review-green"
    assert state["identity"]["head"] == rebased_head
    assert state["identity"]["merge_base"] == current_base
    assert state["deslop"]["status"] == "closed"
    assert state["deslop"]["reviewed_head"] == rebased_head
    assert state["base_drift"] == {
        "status": "ignored_no_path_overlap",
        "recorded_merge_base": base_at_review,
        "current_merge_base": current_base,
        "reviewed_head": original_head,
        "current_head": rebased_head,
        "base_changed_path_count": 1,
        "base_changed_paths": ["docs/notes.md"],
        "overlap_paths": [],
        "patch_equivalent": True,
        "equivalent_reviewed_head": rebased_head,
    }


def test_github_result_after_amend_requires_same_id_signoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_deslop(monkeypatch)
    _stub_review(monkeypatch)
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _use_single_step_normal_profile(monkeypatch, state_dir)
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/github-result-amend")
    _commit_file(repo, "app.txt", "feature\n", "feature")

    _, opened = _run_review_after_cleanup(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(opened["review"])
    _run_review_after_cleanup(
        monkeypatch,
        ["--id", public_id, "--decision", "clean", "--state-dir", str(state_dir)],
    )
    amended_head = _amend_file(repo, "app.txt", "feature\nfix before github result\n")

    _, result = _run_review_after_cleanup(
        monkeypatch,
        ["--id", public_id, "--github-result", "clean", "--state-dir", str(state_dir)],
    )

    assert result["review_ladder"] == "pending"
    assert result["next_action"] == "continue"
    assert result["Action"]["cmd"].endswith(f"review.py --id {public_id}")
    assert "--github-review" not in str(result["Action"]["cmd"])
    state = _cycle_payload(state_dir, public_id)
    assert state["review_heads"]["last_fix_head"] == amended_head
    assert state["github_review"]["status"] == "unknown"


def test_mode_rerun_after_pending_review_amend_reuses_existing_cycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    deslop_calls = _stub_deslop(monkeypatch)
    review_calls = _stub_review(monkeypatch, "phase_review-round-1")
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _use_compact_normal_profile(monkeypatch, state_dir, include_deep=True)
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/deslop-amend")
    original_head = _commit_file(repo, "app.txt", "feature\n", "feature")

    _, created = _run_review_after_cleanup(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(created["review"])
    state = _cycle_payload(state_dir, public_id)
    assert state["stage"] == "decision-pending"
    assert state["deslop"]["status"] == "tracked"
    assert state["identity"]["head"] == original_head
    assert len(deslop_calls) == 0

    amended_head = _amend_file(repo, "app.txt", "feature\nfix from deslop\n")
    assert amended_head != original_head

    exit_code, resumed = _run_review_after_cleanup(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )

    assert exit_code == 0
    assert resumed["review"] == public_id
    assert len(deslop_calls) == 1
    assert len(review_calls) == 2
    assert len(list((state_dir / "orchestrator" / "cycles").glob("*.json"))) == 1
    state = _cycle_payload(state_dir, public_id)
    assert state["identity"]["head"] == amended_head
    assert state["review_heads"]["head"] == amended_head
    assert state["rounds"][0]["reviewed_head"] == amended_head

    exit_code, restarted = _run_review_after_cleanup(
        monkeypatch,
        [
            "--id",
            public_id,
            "--restart-mode",
            "deep",
            "--reason",
            "rerun deeper after amended continuation",
            "--state-dir",
            str(state_dir),
        ],
    )
    assert exit_code == 0
    assert restarted["review"] != public_id
    assert len(deslop_calls) == 1


def test_mode_rerun_allows_non_overlapping_merge_base_drift_without_rerunning_deslop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    deslop_calls = _stub_deslop(monkeypatch)
    review_calls = _stub_review(monkeypatch, "phase_review-round-1")
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _use_compact_normal_profile(monkeypatch, state_dir)
    _init_repo(repo)
    base_at_review = _commit_file(repo, "README.md", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/base-drift")
    original_head = _commit_file(repo, "src/app.txt", "feature\n", "feature")

    _, created = _run_review_after_cleanup(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(created["review"])
    assert len(deslop_calls) == 0

    _git(repo, "checkout", "main")
    current_base = _commit_file(repo, "docs/notes.md", "main notes\n", "main moves")
    _git(repo, "checkout", "feature/base-drift")
    _git(repo, "rebase", "main")
    rebased_head = _git(repo, "rev-parse", "HEAD")
    assert rebased_head != original_head

    exit_code, resumed = _run_review_after_cleanup(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )

    assert exit_code == 0
    assert resumed["review"] == public_id
    assert len(deslop_calls) == 1
    assert len(review_calls) == 2
    assert len(list((state_dir / "orchestrator" / "cycles").glob("*.json"))) == 1
    state = _cycle_payload(state_dir, public_id)
    assert state["identity"]["head"] == rebased_head
    assert state["identity"]["merge_base"] == current_base
    assert state["rounds"][0]["reviewed_head"] == rebased_head
    assert state["base_drift"] == {
        "status": "ignored_no_path_overlap",
        "recorded_merge_base": base_at_review,
        "current_merge_base": current_base,
        "reviewed_head": original_head,
        "current_head": rebased_head,
        "base_changed_path_count": 1,
        "base_changed_paths": ["docs/notes.md"],
        "overlap_paths": [],
        "patch_equivalent": True,
        "equivalent_reviewed_head": rebased_head,
    }


def test_mode_rerun_after_non_equivalent_base_drift_starts_fresh_cycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    deslop_calls = _stub_deslop(monkeypatch)
    _stub_review(monkeypatch, "phase_review-round-1")
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _use_compact_normal_profile(monkeypatch, state_dir)
    _init_repo(repo)
    _commit_file(repo, "README.md", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/base-drift-edit")
    _commit_file(repo, "src/app.txt", "feature\n", "feature")

    _, created = _run_review(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    old_id = str(created["review"])

    _git(repo, "checkout", "main")
    _commit_file(repo, "docs/notes.md", "main notes\n", "main moves")
    _git(repo, "checkout", "feature/base-drift-edit")
    _git(repo, "rebase", "main")
    edited_head = _amend_file(repo, "src/app.txt", "feature\nedited before review\n")

    exit_code, fresh = _run_review(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )

    assert exit_code == 0
    assert fresh["review"] != old_id
    assert len(deslop_calls) == 0
    state = _cycle_payload(state_dir, str(fresh["review"]))
    assert state["identity"]["head"] == edited_head
    assert len(list((state_dir / "orchestrator" / "cycles").glob("*.json"))) == 2


def test_mode_rerun_after_initial_review_commit_reuses_cycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    deslop_calls = _stub_deslop(monkeypatch)
    review_calls = _stub_review(
        monkeypatch, "phase_review-round-1", "phase_review-round-2"
    )
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _use_compact_normal_profile(monkeypatch, state_dir)
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/deslop-new-commit")
    _commit_file(repo, "app.txt", "feature\n", "feature")

    _, created = _run_review_after_cleanup(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    old_id = str(created["review"])
    new_head = _commit_file(repo, "app.txt", "feature\nnew work\n", "continue feature")

    exit_code, fresh = _run_review_after_cleanup(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )

    assert exit_code == 0
    assert fresh["review"] == old_id
    assert len(deslop_calls) == 1
    state = _cycle_payload(state_dir, old_id)
    assert state["identity"]["head"] == new_head
    assert len(review_calls) == 2
    assert state["rounds"][-1]["reviewed_head"] == new_head
    assert len(list((state_dir / "orchestrator" / "cycles").glob("*.json"))) == 1


def test_mode_rerun_after_overlapping_merge_base_drift_starts_fresh_cycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    deslop_calls = _stub_deslop(monkeypatch)
    _stub_review(monkeypatch, "phase_review-round-1")
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _use_compact_normal_profile(monkeypatch, state_dir)
    _init_repo(repo)
    _commit_file(repo, "app.txt", "one\ntwo\nthree\nfour\n", "base")
    _git(repo, "checkout", "-b", "feature/base-overlap")
    _commit_file(repo, "app.txt", "one\ntwo\nthree\nfeature\n", "feature")

    _, created = _run_review(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    old_id = str(created["review"])

    _git(repo, "checkout", "main")
    _commit_file(repo, "app.txt", "main\ntwo\nthree\nfour\n", "main moves")
    _git(repo, "checkout", "feature/base-overlap")
    _git(repo, "rebase", "main")
    rebased_head = _git(repo, "rev-parse", "HEAD")

    exit_code, fresh = _run_review(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )

    assert exit_code == 0
    assert fresh["review"] != old_id
    assert len(deslop_calls) == 0
    state = _cycle_payload(state_dir, str(fresh["review"]))
    assert state["identity"]["head"] == rebased_head
    assert len(list((state_dir / "orchestrator" / "cycles").glob("*.json"))) == 2


def test_mode_rerun_after_initial_review_reset_reuses_cycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    deslop_calls = _stub_deslop(monkeypatch)
    _stub_review(monkeypatch, "phase_review-round-1")
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _use_compact_normal_profile(monkeypatch, state_dir)
    _init_repo(repo)
    base_head = _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/deslop-reset")
    _commit_file(repo, "app.txt", "feature\n", "feature")

    _, created = _run_review_after_cleanup(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    old_id = str(created["review"])
    _git(repo, "checkout", "-B", "feature/deslop-reset", "main")

    exit_code, fresh = _run_review_after_cleanup(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )

    assert exit_code == 0
    assert fresh["review"] == old_id
    assert len(deslop_calls) == 1
    state = _cycle_payload(state_dir, old_id)
    assert state["identity"]["head"] == base_head
    assert len(list((state_dir / "orchestrator" / "cycles").glob("*.json"))) == 1


def test_id_rerun_after_pending_decision_amend_auto_verifies_same_cycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_deslop(monkeypatch)
    review_calls = _stub_review(
        monkeypatch, "phase_review-round-1", "phase_review-round-2"
    )
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _use_compact_normal_profile(monkeypatch, state_dir)
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/pending-amend")
    original_head = _commit_file(repo, "app.txt", "feature\n", "feature")

    _, created = _run_review_after_cleanup(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(created["review"])
    _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )
    state = _cycle_payload(state_dir, public_id)
    assert state["stage"] == "decision-pending"
    assert state["rounds"][0]["reviewed_head"] == original_head

    amended_head = _amend_file(repo, "app.txt", "feature\nfix pending finding\n")
    assert amended_head != original_head

    exit_code, verification = _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )

    assert exit_code == 0
    assert verification["review"] == public_id
    assert len(review_calls) == 2
    assert review_calls[1]["step_name"] == "precision-signoff"
    state = _cycle_payload(state_dir, public_id)
    assert state["decisions"][0]["command"] == "findings"
    assert state["decisions"][0]["reviewed_head"] == original_head
    assert state["review_heads"]["last_fix_head"] == amended_head
    assert state["rounds"][1]["reviewed_head"] == amended_head
    assert len(list((state_dir / "orchestrator" / "cycles").glob("*.json"))) == 1


def test_id_rerun_after_pending_decision_equivalent_base_drift_updates_reviewed_head(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_deslop(monkeypatch)
    review_calls = _stub_review(
        monkeypatch, "phase_review-round-1", "phase_review-round-2"
    )
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _use_compact_normal_profile(monkeypatch, state_dir)
    _init_repo(repo)
    _commit_file(repo, "README.md", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/pending-base-drift")
    original_head = _commit_file(repo, "app.txt", "feature\n", "feature")

    _, created = _run_review(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(created["review"])
    _run_review(monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)])

    _git(repo, "checkout", "main")
    current_base = _commit_file(repo, "docs/notes.md", "main notes\n", "main moves")
    _git(repo, "checkout", "feature/pending-base-drift")
    _git(repo, "rebase", "main")
    rebased_head = _git(repo, "rev-parse", "HEAD")
    assert rebased_head != original_head

    exit_code, reprint = _run_review(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )

    assert exit_code == 0
    assert reprint["review"] == public_id
    assert len(review_calls) == 1
    state = _cycle_payload(state_dir, public_id)
    assert state["stage"] == "decision-pending"
    assert state["identity"]["head"] == rebased_head
    assert state["identity"]["merge_base"] == current_base
    assert state["rounds"][0]["reviewed_head"] == rebased_head
    assert state["review_heads"]["last_reviewed_head"] == rebased_head
    assert state["base_drift"]["equivalent_reviewed_head"] == rebased_head

    _, findings = _run_review(
        monkeypatch,
        ["--id", public_id, "--decision", "findings", "--state-dir", str(state_dir)],
    )
    assert "--fixes-validated" in str(findings["Action"]["choices"])
    state = _cycle_payload(state_dir, public_id)
    assert state["active_findings"]["reviewed_head"] == rebased_head
    assert state["decisions"][0]["reviewed_head"] == rebased_head


def test_id_rerun_after_findings_fix_allows_non_overlapping_merge_base_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_deslop(monkeypatch)
    review_calls = _stub_review(
        monkeypatch, "phase_review-round-1", "phase_review-round-2"
    )
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _use_compact_normal_profile(monkeypatch, state_dir)
    _init_repo(repo)
    base_at_review = _commit_file(repo, "README.md", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/fix-after-base-drift")
    reviewed_head = _commit_file(repo, "app.txt", "feature\n", "feature")

    _, created = _run_review_after_cleanup(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(created["review"])
    _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )
    _run_review_after_cleanup(
        monkeypatch,
        ["--id", public_id, "--decision", "findings", "--state-dir", str(state_dir)],
    )

    _git(repo, "checkout", "main")
    current_base = _commit_file(repo, "docs/notes.md", "main notes\n", "main moves")
    _git(repo, "checkout", "feature/fix-after-base-drift")
    _git(repo, "rebase", "main")
    rebased_head = _git(repo, "rev-parse", "HEAD")
    fixed_head = _commit_file(
        repo, "app.txt", "feature\nfix\n", "fix findings after rebase"
    )

    exit_code, verification = _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )

    assert exit_code == 0
    assert verification["review"] == public_id
    assert len(review_calls) == 2
    assert review_calls[1]["step_name"] == "precision-signoff"
    state = _cycle_payload(state_dir, public_id)
    assert state["identity"]["head"] == fixed_head
    assert state["identity"]["merge_base"] == current_base
    assert state["review_heads"]["last_fix_head"] == fixed_head
    assert (
        state["pending_action"]["fix_verification"]["findings_reviewed_head"]
        == rebased_head
    )
    assert state["decisions"][0]["reviewed_head"] == rebased_head
    assert state["rounds"][0]["reviewed_head"] == rebased_head
    assert state["rounds"][1]["reviewed_head"] == fixed_head
    assert state["base_drift"] == {
        "status": "ignored_no_path_overlap",
        "recorded_merge_base": base_at_review,
        "current_merge_base": current_base,
        "reviewed_head": reviewed_head,
        "current_head": fixed_head,
        "base_changed_path_count": 1,
        "base_changed_paths": ["docs/notes.md"],
        "overlap_paths": [],
        "patch_equivalent": False,
        "equivalent_reviewed_head": rebased_head,
    }


@pytest.mark.parametrize("repeat_review", [True, False])
def test_id_findings_fix_allows_overlapping_base_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    repeat_review: bool,
) -> None:
    _stub_deslop(monkeypatch)
    review_calls = _stub_review(
        monkeypatch, "phase_review-round-1", "phase_review-round-2"
    )
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _use_compact_normal_profile(monkeypatch, state_dir)
    _init_repo(repo)
    _commit_file(repo, "spec.md", "base\nshared\n", "base")
    _git(repo, "checkout", "-b", "feature/fix-after-overlap")
    _commit_file(repo, "app.txt", "feature\n", "feature")
    reviewed_head = _commit_file(
        repo, "spec.md", "feature\nshared\n", "touch shared spec"
    )

    _, created = _run_review_after_cleanup(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(created["review"])
    _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )

    _git(repo, "checkout", "main")
    _commit_file(repo, "spec.md", "base\nmain moved\n", "main moves shared spec")
    _git(repo, "checkout", "feature/fix-after-overlap")
    _git(repo, "rebase", "main", "-X", "theirs")
    fixed_head = _amend_file(repo, "app.txt", "feature\nfix\n")

    _, findings = _run_review_after_cleanup(
        monkeypatch,
        ["--id", public_id, "--decision", "findings", "--state-dir", str(state_dir)],
    )
    assert "--fixes-validated" in str(findings["Action"]["choices"])

    if not repeat_review:
        code, _ = _run_review(
            monkeypatch,
            [
                "--id",
                public_id,
                "--fixes-validated",
                "Local P2 outside critical behavior; focused validation passed",
            ],
        )
        assert code == 0
        state = _cycle_payload(state_dir, public_id)
        assert len(review_calls) == 1
        assert state["identity"]["head"] == fixed_head
        assert state["validated_fixes"][-1]["findings_head"] == reviewed_head
        assert state["active_findings"] is None
        return

    exit_code, verification = _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )

    assert exit_code == 0
    assert verification["review"] == public_id
    assert len(review_calls) == 2
    assert review_calls[1]["step_name"] == "precision-signoff"
    assert review_calls[1]["review_scope"]["reviewed_head"] == fixed_head
    state = _cycle_payload(state_dir, public_id)
    assert state["identity"]["head"] == fixed_head
    assert state["review_heads"]["last_fix_head"] == fixed_head
    assert (
        state["pending_action"]["fix_verification"]["findings_reviewed_head"]
        == reviewed_head
    )
    assert state["decisions"][0]["reviewed_head"] == reviewed_head
    assert state["rounds"][0]["reviewed_head"] == reviewed_head
    assert state["rounds"][1]["reviewed_head"] == fixed_head
    assert len(list((state_dir / "orchestrator" / "cycles").glob("*.json"))) == 1


@pytest.mark.parametrize(
    "mode,github,remaining",
    [
        ("fast", False, False),
        ("normal", True, False),
        ("deep", False, True),
    ],
)
def test_validated_fixes_preserve_findings_and_remaining_gates_without_review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    github: bool,
    remaining: bool,
) -> None:
    def emit_error(message: str, **kwargs: object) -> int:
        review.emit_toon({"error": message})
        return 2

    monkeypatch.setattr(review, "emit_error", emit_error)
    calls = _stub_review_with_terminal(monkeypatch, "clean" if github else "findings")
    state_dir = tmp_path / "state"
    config = deepcopy(review.load_config(state_dir))
    steps = [
        {
            "name": "precision-signoff",
            "count": 2,
            "model_ref": "signoff_normal_model",
            "rerun_on_findings": True,
        }
    ]
    if remaining:
        steps.append(
            {"name": "final-signoff", "count": 2, "model_ref": "signoff_deep_model"}
        )
    config["orchestrator"]["profiles"]["stable"][mode]["steps"] = steps
    monkeypatch.setattr(review, "load_config", lambda _: config)
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature")
    findings_head = _commit_file(repo, "app.txt", "feature\n", "feature")
    _, opened = _run_review(
        monkeypatch,
        [
            "--mode",
            mode,
            "--cd",
            str(repo),
            "--base",
            "main",
            "--skip-deslop",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(opened["review"])
    if github:
        _run_review(monkeypatch, ["--id", public_id, "--github-result", "findings"])
    before = _cycle_payload(state_dir, public_id)
    assert before["stage"] == "fix-pending"

    # No committed fix, empty commits, dirty fixes and a different branch cannot close findings.
    def close() -> tuple[int, dict[str, object]]:
        return _run_review(
            monkeypatch,
            [
                "--id",
                public_id,
                "--fixes-validated",
                "Docs/test-only; relevant checks passed",
            ],
        )

    assert close()[0] == 2
    _git(repo, "commit", "--allow-empty", "-m", "empty")
    assert close()[0] == 2
    (repo / "app.txt").write_text("fixed\n")
    assert close()[0] == 2
    fix_head = _commit_file(repo, "app.txt", "fixed\n", "fix")
    _git(repo, "checkout", "-b", "other")
    assert close()[0] == 2
    _git(repo, "checkout", "feature")
    code, closed = close()
    assert code == 0
    assert len(calls) == 1
    state = _cycle_payload(state_dir, public_id)
    assert state["rounds"] == before["rounds"]
    assert state["decisions"] == before["decisions"]
    assert state["validated_fixes"][-1]["findings_head"] == findings_head
    assert state["validated_fixes"][-1]["head"] == fix_head
    assert state["active_findings"] is None
    assert state["validation"]["focused"] == "passed"
    assert state["validation"]["full_suite"] == "unknown"
    assert state["validation"]["ci"] == "unknown"
    assert state["stage"] == ("created" if remaining else "review-green")
    if remaining:
        assert state["pending_action"]["step_index"] == 1
    elif github:
        assert state["github_review"]["status"] == "waived"
        assert closed["next_action"] == "validation"
    else:
        assert closed["next_action"] == "none"
    _, status = _run_review(monkeypatch, ["--id", public_id, "--show-status"])
    assert status["review_ladder"] != "invalidated"
    assert len(calls) == 1


def test_clean_followup_note_does_not_leak_to_later_review_steps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_deslop(monkeypatch)
    _stub_review(
        monkeypatch,
        "phase_review-round-1",
        "phase_review-round-2",
        "phase_review-round-3",
    )
    _stub_followup(monkeypatch, "followup-round-1")
    config = deepcopy(review.load_config(tmp_path / "state"))
    config["orchestrator"]["profiles"]["stable"]["deep"]["steps"] = [
        {
            "name": "precision-signoff",
            "count": 1,
            "model_ref": "signoff_deep_model",
            "rerun_on_findings": True,
        },
        {"name": "final-sweep", "count": 1, "model_ref": "signoff_deep_model"},
    ]
    monkeypatch.setattr(review, "load_config", lambda state_dir: config)
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/signoff-note-stale")
    _commit_file(repo, "app.txt", "feature\n", "feature")

    _, created = _run_review(
        monkeypatch,
        [
            "--mode",
            "deep",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(created["review"])
    _run_review(monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)])
    _run_review(
        monkeypatch,
        ["--id", public_id, "--decision", "findings", "--state-dir", str(state_dir)],
    )
    _commit_file(repo, "app.txt", "fixed\n", "fix signoff finding")
    _run_review(monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)])

    _, rerun_ready = _run_review(
        monkeypatch,
        ["--id", public_id, "--decision", "clean", "--state-dir", str(state_dir)],
    )

    assert rerun_ready["Action"]["note"] == (
        "Clean follow-up is not final signoff; run review step 1/2 precision-signoff "
        "before treating the review as green."
    )

    _run_review(monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)])
    _, final_step_ready = _run_review(
        monkeypatch,
        ["--id", public_id, "--decision", "clean", "--state-dir", str(state_dir)],
    )

    assert set(dict(final_step_ready["Action"])) == {"cmd"}
    state = _cycle_payload(state_dir, public_id)
    assert state["pending_action"] == {
        "kind": "run-review-step",
        "step_index": 1,
        "step": "final-sweep",
    }


def test_followup_findings_loops_back_to_fix_pending(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_deslop(monkeypatch)
    _stub_review(monkeypatch, "phase_review-round-1")
    followup_calls = _stub_followup(monkeypatch, "followup-round-1")
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/followup-findings")
    _commit_file(repo, "app.txt", "feature\n", "feature")

    _, created = _run_review(
        monkeypatch,
        [
            "--mode",
            "deep",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(created["review"])
    _run_review(monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)])
    _run_review(
        monkeypatch,
        ["--id", public_id, "--decision", "findings", "--state-dir", str(state_dir)],
    )
    _commit_file(repo, "app.txt", "fixed once\n", "fix findings")

    _, followup = _run_review(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )
    assert "stage" not in followup
    assert len(followup_calls) == 1

    exit_code, findings = _run_review(
        monkeypatch,
        ["--id", public_id, "--decision", "findings", "--state-dir", str(state_dir)],
    )

    assert exit_code == 0
    assert "stage" not in findings
    assert "--fixes-validated" in str(findings["Action"]["choices"])
    state = _cycle_payload(state_dir, public_id)
    assert state["active_findings"]["round_id"] == "followup-round-1"
    assert state["active_findings"]["lane"] == "review-followup"
    assert state["active_findings"]["previous_round_id"] == "phase_review-round-1"
    assert state["active_findings"]["status"] == "fix-pending"
    assert state["validation"]["review_green"] == "unknown"

    exit_code, reprint = _run_review(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )
    assert exit_code == 0
    assert "stage" not in reprint
    assert len(followup_calls) == 1


def test_validation_flags_do_not_run_expensive_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    deslop_calls = _stub_deslop(monkeypatch)
    review_calls = _stub_review(monkeypatch, "phase_review-round-1")
    followup_calls = _stub_followup(monkeypatch, "followup-round-1")
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/validation-only")
    _commit_file(repo, "app.txt", "feature\n", "feature")

    _, created = _run_review_after_cleanup(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(created["review"])
    _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )
    _run_review_after_cleanup(
        monkeypatch,
        ["--id", public_id, "--decision", "findings", "--state-dir", str(state_dir)],
    )
    _commit_file(repo, "app.txt", "fixed\n", "fix findings")

    exit_code, payload = _run_review_after_cleanup(
        monkeypatch,
        ["--id", public_id, "--full-suite", "pending", "--state-dir", str(state_dir)],
    )

    assert exit_code == 0
    assert "stage" not in payload
    assert len(deslop_calls) == 1
    assert len(review_calls) == 1
    assert followup_calls == []
    state = _cycle_payload(state_dir, public_id)
    assert state["validation"]["full_suite"] == "pending"
    assert state["active_findings"]["round_id"] == "phase_review-round-1"


@pytest.mark.parametrize("resume", ["id", "acknowledge", "mode"])
@pytest.mark.parametrize("edit_kind", ["commit", "amend"])
def test_cleanup_edits_precede_first_normal_signoff_and_final_fixes_do_not_repeat_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, resume: str, edit_kind: str
) -> None:
    review_calls = _stub_review(monkeypatch, "signoff-1", "signoff-2")
    cleanup_calls = _stub_deslop(monkeypatch)
    repo, state_dir = tmp_path / "repo", tmp_path / "state"
    _use_single_step_normal_profile(monkeypatch, state_dir)
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/cleanup")
    _commit_file(repo, "app.txt", "feature\n", "feature")
    _, opened = _run_review(
        monkeypatch,
        [
            "--mode",
            "normal",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(opened["review"])
    assert len(cleanup_calls) == 1
    assert review_calls == []
    assert opened["done"] is False
    assert "--deslop-done" in opened["Action"]["cmd"]
    _run_review(monkeypatch, ["--id", public_id])
    assert review_calls == []
    state = _cycle_payload(state_dir, public_id)
    state["deslop"].update(conformance="MATERIALLY_DRIFTED", decision="findings")
    state["validation"].update(focused="passed", full_suite="passed", ci="passed")
    _write_cycle_payload(state_dir, public_id, state)

    cleanup_head = (
        _amend_file(repo, "app.txt", "simplified\n")
        if edit_kind == "amend"
        else _commit_file(repo, "app.txt", "simplified\n", "cleanup")
    )
    resume_args = (
        ["--mode", "normal", "--cd", str(repo), "--base", "main"]
        if resume == "mode"
        else ["--id", public_id]
        + (["--deslop-done"] if resume == "acknowledge" else [])
    )
    _, resumed = _run_review(monkeypatch, resume_args)
    assert resumed["review"] == public_id
    state = _cycle_payload(state_dir, public_id)
    assert state["deslop"]["status"] == "closed"
    assert state["identity"]["head"] == cleanup_head
    assert state["validation"]["review_green"] == "unknown"
    assert state["validation"]["full_suite"] == "unknown"
    assert resumed["done"] is False
    if resume == "acknowledge":
        assert review_calls == []
        _run_review(monkeypatch, ["--id", public_id])
    assert review_calls[0]["review_scope"]["reviewed_head"] == cleanup_head

    _run_review(monkeypatch, ["--id", public_id, "--decision", "findings"])
    fixed_head = _commit_file(repo, "app.txt", "simplified and fixed\n", "signoff fix")
    _run_review(monkeypatch, ["--id", public_id])
    assert len(review_calls) == 2
    assert review_calls[-1]["review_scope"]["reviewed_head"] == fixed_head
    _, green = _run_review(monkeypatch, ["--id", public_id, "--decision", "clean"])
    assert (
        _cycle_payload(state_dir, public_id)["validation"]["review_green"] == "passed"
    )
    _run_review(monkeypatch, ["--id", public_id])
    assert len(cleanup_calls) == 1
    assert _cycle_payload(state_dir, public_id)["deslop"]["status"] == "closed"


def test_fast_manual_github_findings_keeps_re_review_action(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    review_calls = _stub_review(monkeypatch, "signoff-round-1", "signoff-round-2")

    _stub_deslop(monkeypatch)
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/fast-github-findings")
    _commit_file(repo, "app.txt", "feature\n", "feature")

    _, opened = _run_review_after_cleanup(
        monkeypatch,
        [
            "--mode",
            "fast",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(opened["review"])
    _, clean = _run_review_after_cleanup(
        monkeypatch,
        ["--id", public_id, "--decision", "clean", "--state-dir", str(state_dir)],
    )
    _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )
    _, clean = _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--deslop-done"]
    )
    assert "Action" not in clean

    exit_code, github_findings = _run_review_after_cleanup(
        monkeypatch,
        [
            "--id",
            public_id,
            "--github-result",
            "findings",
            "--github-note",
            "GitHub found a stale replay bug.",
            "--state-dir",
            str(state_dir),
        ],
    )
    assert exit_code == 0
    assert github_findings["github_review"] == "findings"
    assert "--fixes-validated" in str(github_findings["Action"]["choices"])

    _commit_file(repo, "app.txt", "feature\nfixed\n", "fix github finding")
    _, signoff = _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )
    assert "--decision clean" in _assert_decision_action(signoff["Action"])["clean"]
    assert len(review_calls) == 2
    assert review_calls[1]["step_name"] == "fast-signoff"

    _, final_clean = _run_review_after_cleanup(
        monkeypatch,
        ["--id", public_id, "--decision", "clean", "--state-dir", str(state_dir)],
    )
    _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )
    _, final_clean = _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--deslop-done"]
    )
    _assert_github_handoff(
        final_clean["Action"],
        public_id=public_id,
        state_dir=state_dir,
        blocked_by=["full_suite:unknown", "ci:unknown"],
    )
    assert (
        final_clean["Action"]["note"]
        == "GitHub findings were fixed and locally signed off; request GitHub review again."
    )


def test_stale_decision_renders_current_action_without_mutating_cycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    review_calls = _stub_review(monkeypatch)

    _stub_deslop(monkeypatch)
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")

    _, payload = _run_review_after_cleanup(
        monkeypatch,
        [
            "--mode",
            "fast",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(payload["review"])
    _run_review_after_cleanup(
        monkeypatch,
        ["--id", public_id, "--decision", "clean", "--state-dir", str(state_dir)],
    )
    _run_review_after_cleanup(
        monkeypatch, ["--id", public_id, "--state-dir", str(state_dir)]
    )
    _run_review_after_cleanup(monkeypatch, ["--id", public_id, "--deslop-done"])
    before = _cycle_payload(state_dir, public_id)

    exit_code, stale = _run_review_after_cleanup(
        monkeypatch,
        ["--id", public_id, "--decision", "findings", "--state-dir", str(state_dir)],
    )

    assert exit_code == 0
    assert stale["status"] == "decision_not_pending"
    assert stale["decision"] == "findings"
    assert "already advanced past that decision" in str(stale["note"])
    assert "No further action is pending" in str(stale["note"])
    assert "Action" not in stale
    assert _cycle_payload(state_dir, public_id) == before
    assert len(review_calls) == 1

    exit_code, validation = _run_review_after_cleanup(
        monkeypatch,
        [
            "--id",
            public_id,
            "--decision",
            "clean",
            "--full-suite",
            "passed",
            "--ci",
            "passed",
            "--state-dir",
            str(state_dir),
        ],
    )

    assert exit_code == 0
    assert validation["status"] == "decision_not_pending"
    state = _cycle_payload(state_dir, public_id)
    assert state["validation"]["full_suite"] == "passed"
    assert state["validation"]["ci"] == "passed"


def test_stale_decision_persists_auto_resume_transition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    review_calls = _stub_review(monkeypatch)

    _stub_deslop(monkeypatch)
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature/stale-decision")
    _commit_file(repo, "app.txt", "feature\n", "feature")

    _, payload = _run_review_after_cleanup(
        monkeypatch,
        [
            "--mode",
            "fast",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(payload["review"])
    _commit_file(repo, "app.txt", "feature\nfix\n", "fix finding")

    exit_code, stale = _run_review_after_cleanup(
        monkeypatch,
        ["--id", public_id, "--decision", "clean", "--state-dir", str(state_dir)],
    )

    assert exit_code == 0
    assert stale["status"] == "decision_not_pending"
    assert stale["decision"] == "clean"
    assert "Continue with Action.cmd" in str(stale["note"])
    assert "Action.override" in str(stale["note"])
    assert "--decision" not in str(stale["Action"]["cmd"])
    state = _cycle_payload(state_dir, public_id)
    assert state["stage"] == "created"
    assert state["pending_action"]["kind"] == "run-review-step"
    assert state["decisions"][0]["command"] == "findings"
    assert state["review_heads"]["last_fix_head"] == _git(repo, "rev-parse", "HEAD")
    assert len(review_calls) == 1


def test_decision_pending_with_missing_metadata_still_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_review(monkeypatch)

    _stub_deslop(monkeypatch)
    repo = tmp_path / "repo"
    state_dir = tmp_path / "state"
    _init_repo(repo)
    _commit_file(repo, "app.txt", "base\n", "base")

    _, payload = _run_review_after_cleanup(
        monkeypatch,
        [
            "--mode",
            "fast",
            "--cd",
            str(repo),
            "--base",
            "main",
            "--state-dir",
            str(state_dir),
        ],
    )
    public_id = str(payload["review"])
    state = _cycle_payload(state_dir, public_id)
    state["pending_action"] = {"kind": "decision"}
    state["rounds"][0].pop("lane", None)
    _write_cycle_payload(state_dir, public_id, state)

    errors: list[str] = []
    monkeypatch.setattr(
        review, "emit_error", lambda message, **kwargs: errors.append(str(message)) or 2
    )
    monkeypatch.setattr(review, "default_state_dir", lambda: state_dir)
    monkeypatch.setattr(
        sys, "argv", ["review.py", "--id", public_id, "--decision", "clean"]
    )

    assert review.main() == 2
    assert errors[-1] == "no decision is pending for this review cycle"
    assert _cycle_payload(state_dir, public_id) == state
