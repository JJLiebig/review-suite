set default-list

[windows]
set shell := ["cmd.exe", "/d", "/c"]

# Install development dependencies and enable the workflow hook.
setup:
    uv sync --locked
    git config --local core.hooksPath .githooks

# Run the full test suite.
test:
    uv run --locked pytest -q

# Check Python lint and syntax.
lint:
    uv run --locked ruff check .

# Regenerate the shipped review workflow diagram.
workflow:
    uv run --locked python scripts/generate-workflow.py

# Check that the workflow diagram matches the shipped configuration.
workflow-check:
    uv run --locked python scripts/generate-workflow.py --check

# Run lint, workflow verification, and the full test suite.
check: lint workflow-check test
