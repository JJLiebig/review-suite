import json
import os
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from review_suite_core.codex_runtime import (
    use_unsafe_windows_wsl_fallback,
    validate_codex_runtime,
    windows_wsl_codex_child_env,
)
from review_suite_core.lens_runtime import (
    TECHNICAL_REVIEW_DEVELOPER_INSTRUCTIONS,
    _codex_user_config_path,
    codex_exec_command,
    codex_exec_review_command,
    codex_review_prompt_instructions,
    emit_result,
    isolated_runtime_user_config_overrides,
    prepare_codex_review_launch,
    progress_heartbeat_line,
    record_wrapper_session,
)


def test_codex_user_config_path_honors_codex_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    codex_home = tmp_path / "codex-home"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    assert _codex_user_config_path() == codex_home.resolve(strict=False) / "config.toml"


def test_codex_user_config_path_expands_codex_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", "~/.codex-alt")

    assert (
        _codex_user_config_path()
        == (home / ".codex-alt").resolve(strict=False) / "config.toml"
    )


def test_isolated_runtime_user_config_overrides_preserves_only_provider_config(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
model = "ignored-user-default"
model_provider = "openrouter"
openai_base_url = "https://openai-proxy.example/v1"
oss_provider = "lmstudio"

[model_providers.openrouter]
name = "OpenRouter"
base_url = "https://openrouter.example/v1"
wire_api = "responses"

[model_providers.openrouter.env]
OPENROUTER_API_KEY = "env-var-name"

[plugins."github@openai-curated"]
enabled = true

[mcp_servers.node_repl]
command = "node_repl"
""".strip(),
        encoding="utf-8",
    )

    overrides = isolated_runtime_user_config_overrides(config_path)

    assert 'model_provider="openrouter"' in overrides
    assert 'openai_base_url="https://openai-proxy.example/v1"' in overrides
    assert 'oss_provider="lmstudio"' in overrides
    assert (
        'model_providers={openrouter = {name = "OpenRouter", base_url = "https://openrouter.example/v1", '
        'wire_api = "responses", env = {OPENROUTER_API_KEY = "env-var-name"}}}'
    ) in overrides
    assert all(not item.startswith("plugins.") for item in overrides)
    assert all(not item.startswith("mcp_servers.") for item in overrides)
    assert all(not item.startswith("model=") for item in overrides)


def test_isolated_runtime_user_config_overrides_preserves_dotted_provider_keys(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
model_provider = "azure.prod"

[model_providers."azure.prod"]
name = "Azure Prod"
base_url = "https://azure.example/openai/v1"
wire_api = "responses"
""".strip(),
        encoding="utf-8",
    )

    overrides = isolated_runtime_user_config_overrides(config_path)

    assert 'model_provider="azure.prod"' in overrides
    assert (
        'model_providers={"azure.prod" = {name = "Azure Prod", base_url = "https://azure.example/openai/v1", '
        'wire_api = "responses"}}'
    ) in overrides


def test_codex_exec_command_includes_runtime_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.shutil.which", lambda name: "codex"
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.validate_codex_runtime", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.isolated_runtime_user_config_overrides",
        lambda: [],
    )

    command = codex_exec_command(
        tool_name="review-followup",
        model="gpt-5.5",
        reasoning_effort="medium",
        service_tier="fast",
        prompt="Review this.",
        output_path=tmp_path / "out.txt",
        review_root=tmp_path,
        allow_unsafe_windows_wsl_fallback=False,
    )

    assert 'service_tier="fast"' in command
    assert command.index('service_tier="fast"') < command.index("--color")
    assert "--ignore-user-config" in command
    assert 'approval_policy="never"' in command
    assert command.index("--ignore-user-config") < command.index("-C")
    assert command.index('approval_policy="never"') < command.index("--color")
    assert 'windows.sandbox="unelevated"' in command
    assert "windows.sandbox_private_desktop=false" not in command
    assert command[command.index("-s") + 1] == "read-only"


def test_codex_exec_command_can_skip_git_repo_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.shutil.which", lambda name: "codex"
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.validate_codex_runtime", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.isolated_runtime_user_config_overrides",
        lambda: [],
    )

    command = codex_exec_command(
        tool_name="review-plan",
        model="gpt-5.5",
        reasoning_effort="medium",
        prompt="Review this.",
        output_path=tmp_path / "out.txt",
        review_root=tmp_path,
        allow_unsafe_windows_wsl_fallback=False,
        skip_git_repo_check=True,
    )

    assert "--skip-git-repo-check" in command
    assert command.index("--skip-git-repo-check") < command.index("-o")


def test_codex_exec_command_preserves_max_for_gpt_5_6(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.shutil.which", lambda name: "codex"
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.validate_codex_runtime", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.isolated_runtime_user_config_overrides",
        lambda: [],
    )

    command = codex_exec_command(
        tool_name="review-followup",
        model="gpt-5.6-sol",
        reasoning_effort="max",
        prompt="Review this.",
        output_path=tmp_path / "out.txt",
        review_root=tmp_path,
        allow_unsafe_windows_wsl_fallback=False,
    )

    assert 'model_reasoning_effort="max"' in command


def test_codex_exec_command_clamps_max_for_gpt_5_5(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.shutil.which", lambda name: "codex"
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.validate_codex_runtime", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.isolated_runtime_user_config_overrides",
        lambda: [],
    )

    command = codex_exec_command(
        tool_name="review-followup",
        model="gpt-5.5",
        reasoning_effort="max",
        prompt="Review this.",
        output_path=tmp_path / "out.txt",
        review_root=tmp_path,
        allow_unsafe_windows_wsl_fallback=False,
    )

    assert 'model_reasoning_effort="xhigh"' in command
    assert 'model_reasoning_effort="max"' not in command


def test_codex_exec_command_repasses_provider_overrides_before_review_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.shutil.which", lambda name: "codex"
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.validate_codex_runtime", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.isolated_runtime_user_config_overrides",
        lambda: [
            'model_provider="openrouter"',
            'model_providers={openrouter = {base_url = "https://openrouter.example/v1"}}',
            'openai_base_url="https://openai-proxy.example/v1"',
        ],
    )

    command = codex_exec_command(
        tool_name="review-followup",
        model="gpt-5.5",
        reasoning_effort="medium",
        prompt="Review this.",
        output_path=tmp_path / "out.txt",
        review_root=tmp_path,
        allow_unsafe_windows_wsl_fallback=False,
    )

    assert 'model_provider="openrouter"' in command
    assert (
        'model_providers={openrouter = {base_url = "https://openrouter.example/v1"}}'
        in command
    )
    assert 'openai_base_url="https://openai-proxy.example/v1"' in command
    assert command.index('model_provider="openrouter"') < command.index(
        'model="gpt-5.5"'
    )
    assert "plugins.github.enabled=true" not in command
    assert 'mcp_servers.node_repl.command="node_repl"' not in command


def test_codex_exec_command_keeps_wsl_fallback_read_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.shutil.which", lambda name: "codex"
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.validate_codex_runtime", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.use_unsafe_windows_wsl_fallback",
        lambda *args: True,
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.isolated_runtime_user_config_overrides",
        lambda: [],
    )

    command = codex_exec_command(
        tool_name="review-followup",
        model="gpt-5.5",
        reasoning_effort="medium",
        prompt="Review this.",
        output_path=tmp_path / "out.txt",
        review_root=tmp_path,
        allow_unsafe_windows_wsl_fallback=True,
    )

    assert command[0:3] == ["codex", "exec", "--ignore-user-config"]
    assert "--dangerously-bypass-approvals-and-sandbox" not in command
    assert 'approval_policy="never"' in command
    assert command[command.index("-s") + 1] == "read-only"


def test_codex_exec_review_command_keeps_wsl_fallback_read_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.shutil.which", lambda name: "codex"
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.validate_codex_runtime", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.use_unsafe_windows_wsl_fallback",
        lambda *args: True,
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.isolated_runtime_user_config_overrides",
        lambda: [],
    )

    command = codex_exec_review_command(
        tool_name="review-suite",
        model="gpt-5.5",
        reasoning_effort="medium",
        title="review-suite::round::alpha::gpt-5.5-medium",
        review_root=tmp_path,
        base="origin/main",
        allow_unsafe_windows_wsl_fallback=True,
    )

    assert command[0:3] == ["codex", "exec", "--ignore-user-config"]
    assert "--dangerously-bypass-approvals-and-sandbox" not in command
    assert 'approval_policy="never"' in command
    assert command[command.index("-s") + 1] == "read-only"
    assert command[-2:] == ["--base", "origin/main"]


def test_prepare_codex_review_launch_creates_prompted_exec_without_native_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.shutil.which", lambda name: "codex"
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.validate_codex_runtime", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.default_review_suite_state_dir",
        lambda: state_dir,
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.isolated_runtime_user_config_overrides",
        lambda: [],
    )

    launch = prepare_codex_review_launch(
        tool_name="review-suite",
        model="gpt-5.5",
        reasoning_effort="medium",
        service_tier="fast",
        title="review-suite::round::alpha::gpt-5.5-medium",
        review_root=tmp_path,
        base="origin/main",
        prompt="Review for correctness.",
        output_prefix="review-test-",
        allow_unsafe_windows_wsl_fallback=False,
    )

    try:
        assert launch.command[0:2] == ["codex", "exec"]
        assert "--ignore-user-config" in launch.command
        assert "review" in launch.command
        assert (
            launch.command[launch.command.index("--title") + 1]
            == "review-suite::round::alpha::gpt-5.5-medium"
        )
        assert "--base" not in launch.command
        assert "--commit" not in launch.command
        assert launch.command[-1] == "-"
        assert 'service_tier="fast"' in launch.command
        assert 'approval_policy="never"' in launch.command
        assert launch.command.index("--ignore-user-config") < launch.command.index("-C")
        assert launch.command.index('approval_policy="never"') < launch.command.index(
            "--title"
        )
        assert launch.command.index('service_tier="fast"') < launch.command.index(
            "--title"
        )
        assert launch.stdin_text is not None
        assert "base ref `origin/main`" in launch.stdin_text
        assert (
            "Review only for concrete technical merge-readiness risks"
            in launch.stdin_text
        )
        assert "because AI is involved" in launch.stdin_text
        assert "Review Suite instructions:" in launch.stdin_text
        assert "Review for correctness." in launch.stdin_text
        assert "BEGIN DIFF" not in launch.stdin_text
        assert launch.final_message_path is not None
        assert launch.final_message_path.parent == state_dir / "tmp"
        assert str(launch.final_message_path) in launch.command
        assert launch.cwd == tmp_path
        assert launch.env is None
    finally:
        if launch.final_message_path is not None:
            launch.final_message_path.unlink(missing_ok=True)


def test_prepare_codex_review_launch_uses_native_exec_review_without_prompt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.shutil.which", lambda name: "codex"
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.validate_codex_runtime", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.default_review_suite_state_dir",
        lambda: state_dir,
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.isolated_runtime_user_config_overrides",
        lambda: [],
    )

    launch = prepare_codex_review_launch(
        tool_name="review-suite",
        model="gpt-5.5",
        reasoning_effort="medium",
        title="review-suite::round::alpha::gpt-5.5-medium",
        review_root=tmp_path,
        base="origin/main",
        prompt="",
        output_prefix="review-test-",
        allow_unsafe_windows_wsl_fallback=False,
    )

    try:
        assert launch.command[0:2] == ["codex", "exec"]
        assert "--ignore-user-config" in launch.command
        assert "review" in launch.command
        assert (
            launch.command[launch.command.index("--title") + 1]
            == "review-suite::round::alpha::gpt-5.5-medium"
        )
        assert 'approval_policy="never"' in launch.command
        assert launch.command[-2:] == ["--base", "origin/main"]
        assert launch.command[-1] != "-"
        assert launch.stdin_text is None
        assert launch.final_message_path is None
        assert "-o" not in launch.command
    finally:
        if launch.final_message_path is not None:
            launch.final_message_path.unlink(missing_ok=True)


def test_prepare_codex_review_launch_keeps_prompt_only_stdin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.shutil.which", lambda name: "codex"
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.validate_codex_runtime", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.default_review_suite_state_dir",
        lambda: state_dir,
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.isolated_runtime_user_config_overrides",
        lambda: [],
    )

    launch = prepare_codex_review_launch(
        tool_name="review-suite",
        model="gpt-5.5",
        reasoning_effort="medium",
        title="review-suite::round::alpha::gpt-5.5-medium",
        review_root=tmp_path,
        prompt="Review this checkout.",
        output_prefix="review-test-",
        allow_unsafe_windows_wsl_fallback=False,
    )

    try:
        assert launch.command[-1] == "-"
        assert launch.stdin_text is not None
        assert "Do not modify files." in launch.stdin_text
        assert (
            "Review only for concrete technical merge-readiness risks"
            in launch.stdin_text
        )
        assert "Review Suite instructions:" in launch.stdin_text
        assert "Review this checkout." in launch.stdin_text
        assert launch.final_message_path is not None
    finally:
        if launch.final_message_path is not None:
            launch.final_message_path.unlink(missing_ok=True)


def test_codex_exec_review_command_rejects_prompted_native_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.shutil.which", lambda name: "codex"
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.validate_codex_runtime", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.isolated_runtime_user_config_overrides",
        lambda: [],
    )

    with pytest.raises(ValueError, match="cannot combine"):
        codex_exec_review_command(
            tool_name="review-suite",
            model="gpt-5.5",
            reasoning_effort="medium",
            title="review-suite::round::alpha::gpt-5.5-medium",
            review_root=tmp_path,
            base="origin/main",
            prompt="Review for correctness.",
            allow_unsafe_windows_wsl_fallback=False,
        )


def test_codex_review_prompt_instructions_prefixes_exact_charter() -> None:
    instructions = codex_review_prompt_instructions("Review result: clean")

    assert instructions.startswith(TECHNICAL_REVIEW_DEVELOPER_INSTRUCTIONS)
    assert instructions.endswith("Review result: clean")
    assert "Review Suite instructions:" in instructions


def test_prepare_codex_review_launch_does_not_prefix_deslop_prompt_with_technical_charter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.shutil.which", lambda name: "codex"
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.validate_codex_runtime", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.default_review_suite_state_dir",
        lambda: state_dir,
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.isolated_runtime_user_config_overrides",
        lambda: [],
    )

    launch = prepare_codex_review_launch(
        tool_name="review-deslop",
        model="gpt-5.5",
        reasoning_effort="medium",
        title="review-deslop::gpt-5.5-medium",
        review_root=tmp_path,
        base="origin/main",
        prompt="Find redundant code.",
        output_prefix="review-test-",
        allow_unsafe_windows_wsl_fallback=False,
    )

    try:
        assert launch.command[-1] == "-"
        assert launch.stdin_text is not None
        assert "base ref `origin/main`" in launch.stdin_text
        assert "Find redundant code." in launch.stdin_text
        assert (
            "Review only for concrete technical merge-readiness risks"
            not in launch.stdin_text
        )
        assert "Review Suite instructions:" not in launch.stdin_text
    finally:
        if launch.final_message_path is not None:
            launch.final_message_path.unlink(missing_ok=True)


def test_prepare_codex_review_launch_does_not_prefix_followup_prompt_with_technical_charter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.shutil.which", lambda name: "codex"
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.validate_codex_runtime", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.default_review_suite_state_dir",
        lambda: state_dir,
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.isolated_runtime_user_config_overrides",
        lambda: [],
    )

    launch = prepare_codex_review_launch(
        tool_name="review-followup",
        model="gpt-5.5",
        reasoning_effort="medium",
        title="review-followup::gpt-5.5-medium",
        review_root=tmp_path,
        base="origin/main",
        prompt="Verify the previous finding was fixed.",
        output_prefix="review-test-",
        allow_unsafe_windows_wsl_fallback=False,
    )

    try:
        assert launch.command[-1] == "-"
        assert launch.stdin_text is not None
        assert "Verify the previous finding was fixed." in launch.stdin_text
        assert (
            "Review only for concrete technical merge-readiness risks"
            not in launch.stdin_text
        )
        assert "Review Suite instructions:" not in launch.stdin_text
    finally:
        if launch.final_message_path is not None:
            launch.final_message_path.unlink(missing_ok=True)


def test_prepare_codex_review_launch_validates_base_commit_end_range(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    calls: list[tuple[Path, str, str, str]] = []
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.shutil.which", lambda name: "codex"
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.validate_codex_runtime", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.default_review_suite_state_dir",
        lambda: state_dir,
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.isolated_runtime_user_config_overrides",
        lambda: [],
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.validated_linear_review_range",
        lambda cwd, start_ref, end_ref, label: calls.append(
            (cwd, start_ref, end_ref, label)
        ),
    )

    launch = prepare_codex_review_launch(
        tool_name="review-suite",
        model="gpt-5.5",
        reasoning_effort="medium",
        title="review-suite::round::alpha::gpt-5.5-medium",
        review_root=tmp_path,
        base="old-head",
        commit_end="new-head",
        prompt="Review interdiff.",
        allow_unsafe_windows_wsl_fallback=False,
    )

    try:
        assert calls == [
            (tmp_path, "old-head", "new-head", "native commit-range review launch")
        ]
        assert launch.stdin_text is not None
        assert "commit range `old-head..new-head`" in launch.stdin_text
        assert "current checkout against base ref" not in launch.stdin_text
        assert (
            "Review only for concrete technical merge-readiness risks"
            in launch.stdin_text
        )
        assert "because AI is involved" in launch.stdin_text
        assert "Review interdiff." in launch.stdin_text
        assert "--base" not in launch.command
        assert "--commit" not in launch.command
        assert launch.command[-1] == "-"
    finally:
        if launch.final_message_path is not None:
            launch.final_message_path.unlink(missing_ok=True)


def test_prepare_codex_review_launch_uses_prompt_for_promptless_commit_end(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    calls: list[tuple[Path, str, str, str]] = []
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.shutil.which", lambda name: "codex"
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.validate_codex_runtime", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.default_review_suite_state_dir",
        lambda: state_dir,
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.isolated_runtime_user_config_overrides",
        lambda: [],
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.validated_linear_review_range",
        lambda cwd, start_ref, end_ref, label: calls.append(
            (cwd, start_ref, end_ref, label)
        ),
    )

    launch = prepare_codex_review_launch(
        tool_name="review-suite",
        model="gpt-5.5",
        reasoning_effort="medium",
        title="review-suite::round::alpha::gpt-5.5-medium",
        review_root=tmp_path,
        base="old-head",
        commit_end="new-head",
        prompt="",
        allow_unsafe_windows_wsl_fallback=False,
    )

    try:
        assert calls == [
            (tmp_path, "old-head", "new-head", "native commit-range review launch")
        ]
        assert launch.stdin_text is not None
        assert "commit range `old-head..new-head`" in launch.stdin_text
        assert "Review instructions:" in launch.stdin_text
        assert (
            "Review only for concrete technical merge-readiness risks"
            in launch.stdin_text
        )
        assert "because AI is involved" in launch.stdin_text
        assert "Review Suite instructions:" not in launch.stdin_text
        assert "--base" not in launch.command
        assert "--commit" not in launch.command
        assert launch.command[-1] == "-"
    finally:
        if launch.final_message_path is not None:
            launch.final_message_path.unlink(missing_ok=True)


def test_progress_heartbeat_line_is_sparse_for_agent_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Stderr:
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr("review_suite_core.lens_runtime.sys.stderr", Stderr())

    assert progress_heartbeat_line("review-deslop", 120) == "OK 2m: deslop"


def test_progress_heartbeat_line_keeps_elapsed_for_interactive_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Stderr:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr("review_suite_core.lens_runtime.sys.stderr", Stderr())

    assert progress_heartbeat_line("review-deslop", 120) == "deslop running (120s)"


def test_record_wrapper_session_writes_timestamped_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state_dir = tmp_path / "state"
    monkeypatch.setattr(
        "review_suite_core.lens_runtime.default_review_suite_state_dir",
        lambda: state_dir,
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime._caller_thread_id", lambda: "thread-123"
    )
    monkeypatch.setattr(
        "review_suite_core.lens_runtime._review_branch",
        lambda review_root: "feature/test",
    )

    record_wrapper_session(
        session_id="sess-123",
        tool_name="review-deslop",
        review_root=tmp_path / "repo",
        elapsed_seconds=12.3456,
    )

    payload = json.loads(
        (state_dir / "wrapper_sessions.jsonl").read_text(encoding="utf-8")
    )
    assert payload["session_id"] == "sess-123"
    assert payload["tool_name"] == "review-deslop"
    assert payload["caller_thread_id"] == "thread-123"
    assert payload["branch"] == "feature/test"
    assert payload["elapsed_seconds"] == 12.346
    assert payload["recorded_at"].endswith("Z")


@pytest.mark.parametrize(
    ("stderr", "timed_out", "final_message", "retryable"),
    [
        (
            "The process cannot access the file because it is being used by another process.",
            False,
            "",
            True,
        ),
        ("[WinError 32] file is in use", False, "", True),
        ("ERROR_SHARING_VIOLATION", False, "", True),
        ("process cannot access the file", False, "", False),
        ("model capacity exhausted", False, "", False),
        ("startup failed (os error 5)", False, "", False),
        ("startup failed (WinError 32)", True, "", False),
        ("startup failed (WinError 32)", False, "completed", False),
    ],
)
def test_emit_result_classifies_only_non_timeout_windows_sharing_violations(
    capsys: pytest.CaptureFixture[str],
    stderr: str,
    timed_out: bool,
    final_message: str,
    retryable: bool,
) -> None:
    assert (
        emit_result(
            tool_name="review-plan",
            result={
                "returncode": 1,
                "stdout": "",
                "stderr": stderr,
                "final_message": final_message,
                "session_id": None,
                "elapsed_seconds": 1,
                "timed_out": timed_out,
            },
        )
        == 1
    )

    output = capsys.readouterr().out
    assert stderr in output
    assert ("retryable: true" in output) is retryable
    assert ("error_code: codex_startup_file_locked" in output) is retryable
    assert ("Retry the review" in output) is retryable


def test_windows_wsl_fallback_requires_explicit_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("REVIEW_SUITE_AUTO_WSL_FALLBACK", "true")
    review_root = Path("//wsl.localhost/Ubuntu/home/alice/code/repo")

    assert not use_unsafe_windows_wsl_fallback(review_root, False)
    assert use_unsafe_windows_wsl_fallback(review_root, True)


@pytest.mark.skipif(os.name != "nt", reason="Windows UNC normalization")
def test_windows_wsl_codex_child_env_appends_exact_safe_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.autocrlf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "false")
    monkeypatch.delenv("GIT_CONFIG_KEY_1", raising=False)
    review_root = Path("//wsl.localhost/Ubuntu/home/alice/code/repo/../repo")

    env = windows_wsl_codex_child_env(review_root, True)

    assert env is not None
    assert env["GIT_CONFIG_COUNT"] == "2"
    assert env["GIT_CONFIG_KEY_0"] == "core.autocrlf"
    assert env["GIT_CONFIG_VALUE_0"] == "false"
    assert env["GIT_CONFIG_KEY_1"] == "safe.directory"
    assert env["GIT_CONFIG_VALUE_1"] == "//wsl.localhost/Ubuntu/home/alice/code/repo"
    assert os.environ["GIT_CONFIG_COUNT"] == "1"
    assert "GIT_CONFIG_KEY_1" not in os.environ


def test_validate_codex_runtime_requires_wsl_flag_for_unc_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")

    with pytest.raises(ValueError) as excinfo:
        validate_codex_runtime(
            tool_name="review-suite",
            codex_executable="codex",
            review_root=Path("//wsl.localhost/Ubuntu/home/alice/code/repo"),
            allow_unsafe_windows_wsl_fallback=False,
        )

    message = str(excinfo.value)
    assert "--wsl" in message
    assert "REVIEW_SUITE_AUTO_WSL_FALLBACK" not in message
    assert "bypass" not in message


def test_validate_codex_runtime_mentions_windows_unc_workaround_for_wsl_windows_shim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("review_suite_core.codex_runtime.running_in_wsl", lambda: True)
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")

    with pytest.raises(ValueError) as excinfo:
        validate_codex_runtime(
            tool_name="review-suite",
            codex_executable="/mnt/c/Users/alice/AppData/Roaming/npm/codex",
            review_root=Path("/home/alice/code/repo"),
            allow_unsafe_windows_wsl_fallback=False,
        )

    message = str(excinfo.value)
    assert "//wsl.localhost/Ubuntu/home/alice/code/repo" in message
    assert "--wsl" in message
    assert "install and authenticate" not in message
