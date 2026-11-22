import os
from pathlib import Path
import shutil
import subprocess
import time

import pytest


git_seconds = []
run = subprocess.run


def measure_run(command, *args, **kwargs):
    git = isinstance(command, (list, tuple)) and Path(command[0]).name in {"git", "git.exe"}
    started = time.perf_counter()
    try:
        return run(command, *args, **kwargs)
    finally:
        if git:
            git_seconds.append(time.perf_counter() - started)


subprocess.run = measure_run
test = "test_equivalent_patch_with_older_followup_exposes_remaining_gates[original]"
test_file = "plugins/review-suite/tests/test_review_orchestrator_cli.py"
status = pytest.main(["-o", "addopts=", "-q", "--durations=1", f"{test_file}::{test}"])
print(f"logical_cpus: {os.process_cpu_count()}")
print(f"git_path: {shutil.which('git')}")
print(f"git_calls: {len(git_seconds)}")
print(f"git_seconds: {sum(git_seconds):.2f}")
raise SystemExit(status)
