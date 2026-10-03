from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]


def test_commit_refreshes_workflow_without_including_unstaged_edits(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    for directory in (".githooks", "plugins/review-suite/scripts"):
        shutil.copytree(
            ROOT / directory,
            repo / directory,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for filename in (
        ".gitignore",
        ".gitattributes",
        ".python-version",
        "pyproject.toml",
        "uv.lock",
        "scripts/generate-workflow.py",
        "docs/review-workflow.md",
        "plugins/review-suite/default_settings.toml",
        "plugins/review-suite/references/workflow_settings.toml",
        "plugins/review-suite/references/arena_settings.toml",
    ):
        target = repo / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / filename, target)
    (repo / ".githooks/pre-commit").chmod(0o755)
    env = {**os.environ, "UV_PROJECT_ENVIRONMENT": sys.prefix, "UV_NO_SYNC": "1"}

    def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            env=env,
            text=True,
            capture_output=True,
            check=check,
        )

    git("init")
    git("config", "user.name", "Hook test")
    git("config", "user.email", "hook@example.invalid")
    git("config", "commit.gpgsign", "false")
    git("config", "core.autocrlf", "false")
    git("config", "core.hooksPath", ".githooks")
    git("add", ".")
    git("commit", "--no-verify", "-m", "Baseline")

    generator = repo / "scripts/generate-workflow.py"
    original_generator = generator.read_bytes()
    generator.write_text("raise SystemExit(42)\n", encoding="utf-8")
    (repo / "README.md").write_text("Unrelated change\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-m", "Unrelated change skips generation")
    generator.write_bytes(original_generator)

    settings_path = "plugins/review-suite/default_settings.toml"
    settings = repo / settings_path
    staged_settings = (
        '[normal]\nmodel = "hook-test-model"\nreasoning = "medium"\n'
        '[deep]\nmodel = "hook-test-model"\nreasoning = "xhigh"\n'
    )
    settings.write_text(staged_settings, encoding="utf-8")
    git("add", settings_path)
    settings.write_text(staged_settings + "# Unstaged edit\n", encoding="utf-8")
    diagram = repo / "docs/review-workflow.md"
    original_diagram = diagram.read_bytes()
    head = git("rev-parse", "HEAD").stdout
    result = git("commit", "-m", "Partially staged settings", check=False)
    assert result.returncode != 0
    assert "unstaged edits" in result.stderr
    assert git("rev-parse", "HEAD").stdout == head
    assert diagram.read_bytes() == original_diagram
    assert git("show", f":{settings_path}").stdout == staged_settings

    settings.write_text(staged_settings, encoding="utf-8")
    diagram.write_bytes(original_diagram + b"Unstaged diagram edit\n")
    result = git("commit", "-m", "Preserve diagram edits", check=False)
    assert result.returncode != 0
    assert "unstaged edits" in result.stderr
    assert diagram.read_bytes() == original_diagram + b"Unstaged diagram edit\n"
    diagram.write_bytes(original_diagram)

    settings.write_text("invalid = [\n", encoding="utf-8")
    git("add", settings_path)
    result = git("commit", "-m", "Invalid settings", check=False)
    assert result.returncode != 0
    assert git("rev-parse", "HEAD").stdout == head
    assert diagram.read_bytes() == original_diagram
    assert git("diff", "--cached", "--name-only").stdout.strip() == settings_path

    settings.write_text(staged_settings, encoding="utf-8")
    git("add", settings_path)
    git("commit", "-m", "Models refresh the committed workflow")
    committed_diagram = git("show", "HEAD:docs/review-workflow.md").stdout
    assert "2x hook-test-model medium" in committed_diagram
    assert "hook-test-model xhigh" in committed_diagram
    assert diagram.read_text(encoding="utf-8") == committed_diagram
    assert git("status", "--porcelain").stdout == ""
