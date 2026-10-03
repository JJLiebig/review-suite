[windows]
set shell := ["cmd.exe", "/d", "/c"]

# Install development dependencies and enable the workflow hook.
setup:
    uv sync --locked
    git config --local core.hooksPath .githooks
