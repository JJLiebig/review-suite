#!/usr/bin/env python3
from __future__ import annotations

import sys


def main() -> int:
    # Keep status reads in the launcher. Review commands still run from the
    # leased runtime, which can survive an install update during a long review.
    status_requested = any(
        arg.startswith("--show-s") and "--show-status".startswith(arg)
        for arg in sys.argv[1:]
    )
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    if status_requested:
        from review_suite_core.cycle_status import try_json_status

        result = try_json_status(sys.argv[1:])
        if result is not None:
            return result
    else:
        from review_suite_runtime_bootstrap import bootstrap_runtime

        bootstrap_runtime(__file__)
        sys.dont_write_bytecode = previous_dont_write_bytecode

    from review_orchestrator import main as orchestrate

    return orchestrate()


if __name__ == "__main__":
    raise SystemExit(main())
