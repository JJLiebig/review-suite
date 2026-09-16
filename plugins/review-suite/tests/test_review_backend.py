import json
import os
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import review_suite_core
from review_suite_core.lens_runtime import CodexReviewLaunch
from review_suite_core.opencode_runtime import (
    OPENCODE_REVIEW_AGENT,
    _opencode_review_prompt,
    opencode_review_env,
    prepare_opencode_review_launch,
)
from review_suite_core.review_backend import (
    prepare_review_launch,
    split_review_backend_model,
)


def test_plain_model_uses_codex_backend() -> None:
    assert split_review_backend_model("gpt-6-astra") == ("codex", "gpt-6-astra")


def test_prefixed_model_uses_opencode_backend() -> None:
    assert split_review_backend_model("opencode::opencode-go/deepseek-v4.1-flash") == (
        "opencode",
        "opencode-go/deepseek-v4.1-flash",
    )


@pytest.mark.parametrize(
    "model",
    [
        "opencode::deepseek-flash",
        "opencode::/deepseek-flash",
        "opencode::opencode-go/",
    ],
)
def test_opencode_model_requires_provider_qualified_id(model: str) -> None:
    with pytest.raises(ValueError, match="provider/model"):
        split_review_backend_model(model)


def test_package_launch_seam_is_provider_aware() -> None:
    assert review_suite_core.prepare_codex_review_launch is prepare_review_launch


def test_opencode_review_env_is_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", "user-config")
    env = opencode_review_env()
    config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
    agent = config["agent"][OPENCODE_REVIEW_AGENT]
    permission = agent["permission"]

    assert config["share"] == "disabled"
    assert agent["mode"] == "primary"
    assert permission["edit"] == "deny"
    assert permission["bash"] == "deny"
    assert permission["task"] == "deny"
    assert permission["webfetch"] == "deny"
    assert permission["websearch"] == "deny"
    assert permission["read"] == {
        "*": "allow",
        "*.env": "deny",
        "*.env.*": "deny",
        "*.env.example": "allow",
    }
    assert permission["glob"] == "allow"
    assert permission["grep"] == "allow"
    assert env["OPENCODE_DISABLE_AUTOUPDATE"] == "true"


@pytest.mark.parametrize(
    ("npm", "expected"),
    [
        ("@ai-sdk/openai-compatible", {"max_tokens": 131072}),
        ("@ai-sdk/anthropic", {"max_tokens": 131072}),
        ("@ai-sdk/openai", {"max_output_tokens": 131072}),
        ("@ai-sdk/google", {"generationConfig": {"maxOutputTokens": 131072}}),
    ],
)
def test_launch_scopes_advertised_output_maximum_to_review_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    npm: str,
    expected: dict,
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", "unchanged-parent-config")
    catalog = tmp_path / "opencode" / "models.json"
    catalog.parent.mkdir()
    original = json.dumps(
        {
            "opencode-go": {
                "npm": "unused-default",
                "models": {
                    "glm-5.3-flash": {
                        "provider": {"npm": npm},
                        "limit": {"output": 131072},
                    },
                    "other-model": {"limit": {"output": 100}},
                },
            }
        }
    )
    catalog.write_text(original, encoding="utf-8")
    launch = prepare_opencode_review_launch(
        tool_name="review-suite",
        model="opencode-go/glm-5.3-flash",
        reasoning_effort="low",
        title="test",
        review_root=tmp_path,
        base="main",
        allow_unsafe_windows_wsl_fallback=False,
    )
    config = json.loads(launch.env["OPENCODE_CONFIG_CONTENT"])
    assert config["providers"] == {
        "opencode-go": {
            "models": {
                "glm-5.3-flash": {"body": expected},
            }
        }
    }
    assert launch.command[launch.command.index("--variant") + 1] == "low"
    assert config["agent"][OPENCODE_REVIEW_AGENT]["permission"]["edit"] == "deny"
    assert (
        opencode_review_env()["OPENCODE_CONFIG_CONTENT"]
        != launch.env["OPENCODE_CONFIG_CONTENT"]
    )
    assert catalog.read_text(encoding="utf-8") == original
    assert os.environ["OPENCODE_CONFIG_CONTENT"] == "unchanged-parent-config"


def test_unavailable_output_catalogue_warns_without_inventing_a_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    with pytest.warns(UserWarning, match="using the provider default"):
        config = json.loads(
            opencode_review_env("opencode-go/glm-5.3-flash")["OPENCODE_CONFIG_CONTENT"]
        )
    assert config["providers"] == {}


def test_opencode_prompt_preserves_review_suite_contract() -> None:
    prompt = _opencode_review_prompt(
        tool_name="review-suite",
        prompt="CUSTOM CONTRACT\nReview result: clean",
        base="main",
        commit=None,
        commit_end=None,
    )

    assert "compare the current checkout against base ref `main`" in prompt
    assert "attaches the exact target diff" in prompt
    assert "CUSTOM CONTRACT" in prompt
    assert "Review Suite instructions:" in prompt
    assert "Do not modify files" in prompt


def test_prepare_opencode_review_launch_uses_driver_and_no_final_message_file(
    tmp_path: Path,
) -> None:
    launch = prepare_opencode_review_launch(
        tool_name="review-suite",
        model="opencode-go/deepseek-v4.1-flash",
        reasoning_effort="high",
        title="review-suite::test",
        review_root=tmp_path,
        base="main",
        prompt="Review result: clean",
        allow_unsafe_windows_wsl_fallback=False,
    )

    assert isinstance(launch, CodexReviewLaunch)
    assert launch.command[0] == sys.executable
    assert launch.command[1].endswith("opencode_driver.py")
    assert (
        launch.command[launch.command.index("--model") + 1]
        == "opencode-go/deepseek-v4.1-flash"
    )
    assert launch.command[launch.command.index("--variant") + 1] == "high"
    assert launch.command[launch.command.index("--base") + 1] == "main"
    assert launch.final_message_path is None
    assert launch.cwd == tmp_path.resolve()
    assert launch.effective_reasoning_effort == "high"
    assert "Review Suite instructions:" in str(launch.stdin_text)


def test_opencode_normalizes_inherited_reasoning_to_default(tmp_path: Path) -> None:
    launch = prepare_opencode_review_launch(
        tool_name="review-suite",
        model="opencode-go/deepseek-v4.1-flash",
        reasoning_effort="medium",
        title="review-suite::test",
        review_root=tmp_path,
        base="main",
        prompt="Review result: clean",
        allow_unsafe_windows_wsl_fallback=False,
    )

    assert launch.command[launch.command.index("--variant") + 1] == "high"
    assert launch.effective_reasoning_effort == "high"


def test_opencode_omits_variant_for_unmapped_model(tmp_path: Path) -> None:
    launch = prepare_opencode_review_launch(
        tool_name="review-suite",
        model="opencode-go/kimi-k2",
        reasoning_effort="medium",
        title="review-suite::test",
        review_root=tmp_path,
        base="main",
        prompt="Review result: clean",
        allow_unsafe_windows_wsl_fallback=False,
    )

    assert "--variant" not in launch.command
    assert launch.effective_reasoning_effort == "provider-default"


def test_opencode_rejects_codex_service_tier(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="service_tier"):
        prepare_opencode_review_launch(
            tool_name="review-suite",
            model="opencode-go/deepseek-v4.1-flash",
            reasoning_effort="medium",
            service_tier="fast",
            title="review-suite::test",
            review_root=tmp_path,
            base="main",
            prompt="Review result: clean",
            allow_unsafe_windows_wsl_fallback=False,
        )
