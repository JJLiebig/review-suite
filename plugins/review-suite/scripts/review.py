#!/usr/bin/env python3
from __future__ import annotations

import sys


def _orchestrate() -> int:
    from review_orchestrator import main

    return main()


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

        import os

        from review_suite_runtime_bootstrap import DISABLE_ENV

        # The full CLI imports helpers that are also executable launchers.
        # Keep those imports in this interpreter for TOON and extra flags.
        previous_override = os.environ.get(DISABLE_ENV)
        os.environ[DISABLE_ENV] = "1"
        try:
            return _orchestrate()
        finally:
            if previous_override is None:
                os.environ.pop(DISABLE_ENV, None)
            else:
                os.environ[DISABLE_ENV] = previous_override
            sys.dont_write_bytecode = previous_dont_write_bytecode

    from review_suite_runtime_bootstrap import bootstrap_runtime

    bootstrap_runtime(__file__)
    sys.dont_write_bytecode = previous_dont_write_bytecode
    return _orchestrate()


if __name__ == "__main__":
    raise SystemExit(main())
