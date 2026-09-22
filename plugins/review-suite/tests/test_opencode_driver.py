import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from review_suite_core.opencode_driver import (
    OPENCODE_ERROR_CLASS_CAPACITY,
    OPENCODE_ERROR_CLASS_FAILED,
    OPENCODE_ERROR_CLASS_UNAVAILABLE,
    _assistant_text_candidates,
    _exported_review_text,
    _format_metadata_line,
    _git_diff_command,
    _opencode_run_command,
    _parse_event_stream,
    _usage_from_export,
    build_parser,
    classify_opencode_error,
    parse_opencode_review_metadata,
)
from review_suite_core.opencode_runtime import opencode_review_env
from review_suite_core import opencode_driver as opencode_driver_module


def _args(**overrides: str | None) -> argparse.Namespace:
    values = {"base": None, "commit": None, "commit_end": None}
    values.update(overrides)
    return argparse.Namespace(**values)


def test_git_diff_command_for_branch_review() -> None:
    assert _git_diff_command(_args(base="main")) == [
        "git",
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "main...HEAD",
    ]


def test_git_diff_command_for_commit_range() -> None:
    assert _git_diff_command(_args(base="abc", commit_end="def")) == [
        "git",
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "abc..def",
    ]


def test_git_show_uses_first_parent_for_single_commit_review() -> None:
    assert _git_diff_command(_args(commit="abc")) == [
        "git",
        "show",
        "--first-parent",
        "--format=fuller",
        "--no-ext-diff",
        "--no-textconv",
        "abc",
    ]


def _run_args(tmp_path: Path, *extra: str) -> argparse.Namespace:
    return build_parser().parse_args(
        [
            "--model",
            "opencode-go/deepseek-v4.1-flash",
            "--dir",
            str(tmp_path),
            "--title",
            "review-suite::test",
            *extra,
        ]
    )


def test_run_command_uses_opencode_2_flags_and_model_variant(tmp_path: Path) -> None:
    patch = tmp_path / "target.patch"
    command = _opencode_run_command(
        "opencode", _run_args(tmp_path, "--variant", "high"), patch
    )

    assert command == [
        "opencode",
        "run",
        "--standalone",
        "--format",
        "json",
        "--model",
        "opencode-go/deepseek-v4.1-flash#high",
        "--agent",
        "review-suite",
        "--title",
        "review-suite::test",
        "--file",
        str(patch),
    ]


def test_run_command_omits_variant_when_unset(tmp_path: Path) -> None:
    patch = tmp_path / "target.patch"
    command = _opencode_run_command("opencode", _run_args(tmp_path), patch)

    assert command[command.index("--model") + 1] == "opencode-go/deepseek-v4.1-flash"


def test_event_stream_prefers_final_terminal_review_message() -> None:
    stdout = "\n".join(
        [
            '{"type":"step_start","sessionID":"ses_1"}',
            '{"type":"text","sessionID":"ses_1","part":{"text":"compaction summary"}}',
            '{"type":"step_finish","sessionID":"ses_1"}',
            '{"type":"step_start","sessionID":"ses_1"}',
            '{"type":"text","sessionID":"ses_1","part":{"text":"[P1] Bug\\nReview result: findings"}}',
            '{"type":"step_finish","sessionID":"ses_1"}',
        ]
    )

    result = _parse_event_stream(stdout)

    assert result["session_id"] == "ses_1"
    assert result["reviewer_output"] == "[P1] Bug\nReview result: findings"


def test_event_stream_rejects_incomplete_assistant_text() -> None:
    stdout = "\n".join(
        [
            '{"type":"step_start","sessionID":"ses_1"}',
            '{"type":"text","sessionID":"ses_1","part":{"text":"compaction summary"}}',
            '{"type":"step_finish","sessionID":"ses_1"}',
        ]
    )

    result = _parse_event_stream(stdout)

    assert result["session_id"] == "ses_1"
    assert result["reviewer_output"] is None


def test_event_stream_aggregates_step_finish_usage_and_cost() -> None:
    stdout = "\n".join(
        [
            json.dumps({"type": "step_start", "sessionID": "ses_1"}),
            json.dumps(
                {
                    "type": "step_finish",
                    "sessionID": "ses_1",
                    "part": {
                        "type": "step-finish",
                        "cost": 0.001,
                        "tokens": {
                            "total": 120309,
                            "input": 1899,
                            "output": 351,
                            "reasoning": 299,
                            "cache": {"read": 117760, "write": 0},
                        },
                    },
                }
            ),
            json.dumps(
                {
                    "type": "step_finish",
                    "sessionID": "ses_1",
                    "part": {
                        "type": "step-finish",
                        "cost": 0.002,
                        "tokens": {
                            "total": 200,
                            "input": 100,
                            "output": 50,
                            "reasoning": 0,
                            "cache": {"read": 0, "write": 50},
                        },
                    },
                }
            ),
        ]
    )

    result = _parse_event_stream(stdout)

    assert result["usage"] == {
        "input_tokens": 119809,
        "cached_input_tokens": 117760,
        "output_tokens": 401,
        "cache_write_tokens": 50,
        "reasoning_output_tokens": 299,
        "total_tokens": 120509,
    }
    assert result["cost_usd"] == pytest.approx(0.003)


def test_usage_from_export_sums_billed_messages() -> None:
    payload = {
        "info": {"id": "ses_1"},
        "messages": [
            {"type": "user"},
            {
                "type": "assistant",
                "cost": 0.5,
                "tokens": {
                    "total": 10,
                    "input": 4,
                    "output": 1,
                    "reasoning": 1,
                    "cache": {"read": 4, "write": 0},
                },
            },
            {
                "type": "assistant",
                "cost": 0.25,
                "tokens": {
                    "total": 8,
                    "input": 2,
                    "output": 1,
                    "reasoning": 0,
                    "cache": {"read": 5, "write": 0},
                },
            },
            {
                "type": "compaction",
                "cost": 0.125,
                "tokens": {
                    "total": 11,
                    "input": 3,
                    "output": 2,
                    "reasoning": 1,
                    "cache": {"read": 4, "write": 1},
                },
            },
        ],
    }

    usage, cost_usd = _usage_from_export(payload)

    assert usage == {
        "input_tokens": 23,
        "cached_input_tokens": 13,
        "output_tokens": 4,
        "cache_write_tokens": 1,
        "reasoning_output_tokens": 2,
        "total_tokens": 29,
    }
    assert cost_usd == pytest.approx(0.875)


def test_usage_from_export_treats_zero_cost_as_authoritative() -> None:
    payload = {
        "messages": [
            {
                "type": "assistant",
                "cost": 0,
                "tokens": {
                    "total": 0,
                    "input": 0,
                    "output": 0,
                    "reasoning": 0,
                    "cache": {"read": 0, "write": 0},
                },
            }
        ]
    }

    usage, cost_usd = _usage_from_export(payload)

    assert usage == {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
    }
    assert cost_usd == 0.0


def test_event_stream_omits_cost_when_provider_cost_absent() -> None:
    stdout = json.dumps(
        {
            "type": "step_finish",
            "sessionID": "ses_1",
            "part": {
                "type": "step-finish",
                "tokens": {"input": 10, "output": 2},
            },
        }
    )

    result = _parse_event_stream(stdout)

    assert result["usage"] == {
        "input_tokens": 10,
        "cached_input_tokens": 0,
        "output_tokens": 2,
    }
    assert result["cost_usd"] is None


def test_usage_from_export_omits_cost_when_provider_cost_absent() -> None:
    payload = {
        "messages": [
            {
                "type": "assistant",
                "tokens": {"input": 5, "output": 1},
                "cost": None,
            }
        ]
    }

    usage, cost_usd = _usage_from_export(payload)

    assert usage == {
        "input_tokens": 5,
        "cached_input_tokens": 0,
        "output_tokens": 1,
    }
    assert cost_usd is None


def test_usage_from_export_reports_cost_without_tokens() -> None:
    payload = {"messages": [{"type": "assistant", "tokens": None, "cost": 0.25}]}

    usage, cost_usd = _usage_from_export(payload)

    assert usage == {}
    assert cost_usd == pytest.approx(0.25)


def test_metadata_line_roundtrips() -> None:
    line = _format_metadata_line("ses_1", {"input_tokens": 5}, 0.25)

    parsed = parse_opencode_review_metadata(f"noise\n{line}\nmore noise")

    assert parsed == {
        "session_id": "ses_1",
        "usage": {"input_tokens": 5},
        "cost_usd": 0.25,
    }
    assert parse_opencode_review_metadata("no metadata here") == {}


def test_metadata_parser_prefers_final_driver_record() -> None:
    child_line = _format_metadata_line("ses_child", {"input_tokens": 1}, 0.1)
    driver_line = _format_metadata_line("ses_driver", {"input_tokens": 2}, 0.2)

    assert parse_opencode_review_metadata(f"{child_line}{driver_line}") == {
        "session_id": "ses_driver",
        "usage": {"input_tokens": 2},
        "cost_usd": 0.2,
    }


def test_metadata_parser_requires_prefix_at_line_start() -> None:
    indented = "  " + _format_metadata_line("ses_x", {}, None).rstrip("\n")

    assert parse_opencode_review_metadata(f"noise\n{indented}\n") == {}


def test_event_stream_captures_error_events() -> None:
    stdout = "\n".join(
        [
            json.dumps({"type": "step_start", "sessionID": "ses_1"}),
            json.dumps(
                {
                    "type": "error",
                    "sessionID": "ses_1",
                    "error": {
                        "name": "ProviderError",
                        "data": {"message": "Rate limit exceeded, retry later"},
                    },
                }
            ),
        ]
    )

    result = _parse_event_stream(stdout)

    assert result["errors"] == [
        {"name": "ProviderError", "message": "Rate limit exceeded, retry later"}
    ]


def test_event_stream_reports_no_errors_when_clean() -> None:
    result = _parse_event_stream(
        '{"type":"text","part":{"text":"Review result: clean"}}'
    )

    assert result["errors"] == []


def test_classify_opencode_error_maps_capacity_signatures() -> None:
    for message in (
        "429 Too Many Requests",
        "rate limit reached",
        "usage limit exceeded for this account",
        "insufficient quota",
        "model is at capacity",
        "resource_exhausted",
    ):
        assert (
            classify_opencode_error(
                returncode=1,
                errors=[{"name": "APIError", "message": message}],
            )
            == OPENCODE_ERROR_CLASS_CAPACITY
        )


def test_classify_opencode_error_maps_unavailable_signatures() -> None:
    assert (
        classify_opencode_error(
            returncode=1,
            errors=[{"name": "ProviderError", "message": "Model not found"}],
        )
        == OPENCODE_ERROR_CLASS_UNAVAILABLE
    )
    assert (
        classify_opencode_error(
            returncode=1,
            errors=[{"name": "AuthError", "message": "Unauthorized"}],
        )
        == OPENCODE_ERROR_CLASS_UNAVAILABLE
    )
    assert (
        classify_opencode_error(
            returncode=1,
            errors=[{"name": "ProviderError", "message": "Model unavailable"}],
        )
        == OPENCODE_ERROR_CLASS_UNAVAILABLE
    )
    assert (
        classify_opencode_error(
            returncode=1,
            errors=[{"name": "ProviderError", "message": "Unsupported model"}],
        )
        == OPENCODE_ERROR_CLASS_UNAVAILABLE
    )


def test_classify_opencode_error_treats_service_unavailable_as_failure() -> None:
    assert (
        classify_opencode_error(
            returncode=1,
            errors=[{"name": "ProviderError", "message": "503 Service Unavailable"}],
        )
        == OPENCODE_ERROR_CLASS_FAILED
    )


def test_classify_opencode_error_defaults_to_failed_on_nonzero_exit() -> None:
    assert (
        classify_opencode_error(
            returncode=1,
            errors=[{"name": "UnknownError", "message": "Unexpected server error."}],
        )
        == OPENCODE_ERROR_CLASS_FAILED
    )


def test_classify_opencode_error_returns_none_for_success_or_output() -> None:
    assert (
        classify_opencode_error(
            returncode=0,
            errors=[],
            reviewer_output="Review result: clean",
        )
        is None
    )
    assert (
        classify_opencode_error(
            returncode=1,
            errors=[{"name": "UnknownError", "message": "boom"}],
            reviewer_output="Review result: clean",
        )
        is None
    )


def test_classify_opencode_error_matches_standalone_status_codes() -> None:
    assert (
        classify_opencode_error(
            returncode=1, errors=[{"name": "APIError", "message": "HTTP 429"}]
        )
        == OPENCODE_ERROR_CLASS_CAPACITY
    )
    assert (
        classify_opencode_error(
            returncode=1,
            errors=[{"name": "APIError", "message": "server returned 403"}],
        )
        == OPENCODE_ERROR_CLASS_UNAVAILABLE
    )


def test_classify_opencode_error_ignores_numeric_substrings() -> None:
    assert (
        classify_opencode_error(
            returncode=1,
            errors=[
                {
                    "name": "UnknownError",
                    "message": "request 4290 finished after 1403ms",
                }
            ],
        )
        == OPENCODE_ERROR_CLASS_FAILED
    )


def test_metadata_line_roundtrips_error_class() -> None:
    line = _format_metadata_line(
        "ses_1",
        {},
        None,
        error_class=OPENCODE_ERROR_CLASS_CAPACITY,
        error_name="ProviderError",
        error_message="rate limit",
    )

    assert parse_opencode_review_metadata(line) == {
        "session_id": "ses_1",
        "usage": {},
        "error_class": "capacity",
        "error_name": "ProviderError",
        "error_message": "rate limit",
    }


def test_export_parser_collects_assistant_message_parts() -> None:
    payload = {
        "messages": [
            {
                "type": "assistant",
                "content": [
                    {"type": "text", "text": "Finding one"},
                    {"type": "text", "text": "Review result: findings"},
                ],
            }
        ]
    }

    assert _assistant_text_candidates(payload) == [
        "Finding one\nReview result: findings"
    ]


def test_completed_unmarked_response_is_recovered_from_export(monkeypatch) -> None:
    text = "[P1] Rejected reservation changes stock."
    events = [
        {"type": "step_start", "sessionID": "ses_1"},
        {"type": "text", "part": {"text": text}},
        {"type": "step_finish", "part": {"reason": "stop"}},
    ]
    stream = _parse_event_stream("\n".join(map(json.dumps, events)))
    assert stream["session_id"] == "ses_1"
    assert stream["reviewer_output"] is None
    # OpenCode exports a final assistant response with finish="stop".
    payload = {
        "messages": [
            {
                "type": "assistant",
                "finish": "tool-calls",
                "content": [{"type": "text", "text": "compaction summary"}],
            },
            {
                "type": "assistant",
                "finish": "stop",
                "content": [{"type": "text", "text": text}],
            },
        ]
    }

    def fake_export(command, *args, **kwargs):
        assert command == [
            "opencode",
            "session",
            "export",
            "--standalone",
            "ses_1",
        ]
        return subprocess.CompletedProcess(command, 0, json.dumps(payload))

    monkeypatch.setattr(
        subprocess,
        "run",
        fake_export,
    )
    assert _exported_review_text("opencode", "ses_1", Path.cwd()) == text


def test_driver_stdio_roundtrips_unicode(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONIOENCODING", "cp1252")
    text = "Review: café → 修复"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; expected={ascii(text)}; assert sys.stdin.read() == expected; sys.stdout.write(expected)",
        ],
        input=text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=opencode_review_env(),
        check=True,
    )
    assert proc.stdout == text


@pytest.mark.parametrize("recovered_output", [False, True])
def test_driver_main_preserves_recovered_output_or_reports_provider_failure(
    monkeypatch, tmp_path: Path, capsys, recovered_output: bool
) -> None:
    error_event = json.dumps(
        {
            "type": "error",
            "sessionID": "ses_err",
            "error": {
                "name": "ProviderError",
                "data": {"message": "429 Too Many Requests"},
            },
        }
    )

    def fake_run(command, *args, **kwargs):
        if command[:3] == ["opencode", "session", "export"]:
            if recovered_output:
                return subprocess.CompletedProcess(
                    command,
                    0,
                    json.dumps(
                        {
                            "messages": [
                                {
                                    "type": "assistant",
                                    "finish": "stop",
                                    "content": [
                                        {"type": "text", "text": "Review result: clean"}
                                    ],
                                }
                            ]
                        }
                    ),
                )
            return subprocess.CompletedProcess(command, 1, "")
        return subprocess.CompletedProcess(command, 1, error_event)

    monkeypatch.setattr(opencode_driver_module.shutil, "which", lambda name: "opencode")
    monkeypatch.setattr(opencode_driver_module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        opencode_driver_module,
        "_write_target_patch",
        lambda args, root: tmp_path / "target.patch",
    )
    monkeypatch.setattr(opencode_driver_module.sys, "stdin", io.StringIO("review this"))
    monkeypatch.setattr(
        opencode_driver_module.sys,
        "argv",
        [
            "opencode_driver.py",
            "--model",
            "opencode-go/deepseek-v4.1-flash",
            "--dir",
            str(tmp_path),
            "--title",
            "review-suite::test",
        ],
    )

    assert opencode_driver_module.main() == (0 if recovered_output else 1)

    captured = capsys.readouterr()
    assert captured.out == ("Review result: clean\n" if recovered_output else "")
    metadata = parse_opencode_review_metadata(captured.err)
    assert metadata.get("error_class") == (None if recovered_output else "capacity")
    assert metadata["error_name"] == "ProviderError"
    assert metadata["error_message"] == "429 Too Many Requests"


def test_driver_main_emits_adapter_error_when_cli_missing(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setattr(opencode_driver_module.shutil, "which", lambda name: None)
    monkeypatch.setattr(opencode_driver_module.sys, "stdin", io.StringIO("review this"))
    monkeypatch.setattr(
        opencode_driver_module.sys,
        "argv",
        [
            "opencode_driver.py",
            "--model",
            "opencode-go/deepseek-v4.1-flash",
            "--dir",
            str(tmp_path),
            "--title",
            "review-suite::test",
        ],
    )

    assert opencode_driver_module.main() == 127

    metadata = parse_opencode_review_metadata(capsys.readouterr().err)
    assert metadata["error_class"] == "adapter"
    assert metadata["error_message"] == "OpenCode CLI was not found on PATH"
