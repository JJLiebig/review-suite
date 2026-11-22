from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from review_suite_core.orchestrator_store import orchestrator_store_lock


@pytest.fixture
def git_commands(monkeypatch):
    calls = []
    popen = subprocess.Popen

    def record(command, *args, **kwargs):
        if isinstance(command, (list, tuple)) and command[0] == "git":
            calls.append(command)
        return popen(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", record)
    return calls


@pytest.fixture
def review_process_lock():
    """Serializes tests that run Review Suite entry points with tests that migrate state.

    State migration waits for every other running review entry point, so these tests
    must not overlap across parallel workers.
    """
    with orchestrator_store_lock(
        state_dir=Path(tempfile.gettempdir()) / "review-suite-tests",
        name="review-processes",
        timeout_seconds=600,
    ):
        yield
