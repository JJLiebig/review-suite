# Review recovery

Use the installed launcher and existing review id. These commands preserve the existing review and validation gates.

- To dismiss a materially drifted closure, use `review.py --id <id> --deslop-done --reason "<why the findings are dismissed>"`. This records the caller's reason without changing the reviewer verdict or findings. It requires a completed closure on the exact clean branch, HEAD, and merge-base; all other review and validation gates remain in force.
- To replace an existing ladder with stricter review, use `review.py --id <id> --restart-mode deep --reason "<why>"` while the original repo/base/branch/head/merge-base still match and the worktree is clean; plain `--mode deep --cd <repo-root>` is not a restart.
- To replace mistaken literal text in a frozen brief before accepting any findings, use `review.py --id <id> --restart-brief (Get-Content -Raw "<brief.md>") --reason "<why the old brief is invalid>"`. This creates a fresh same-mode successor on the same clean branch, unchanged merge-base, and current descendant head. The original cycle and findings history remain stored; cleanup, review, and validation do not carry over.
- On `head_changed_after_review`, inspect `reviewed_head..current_head`. If the changes only fix stale tests to match already-reviewed behavior, do not rerun review; run the affected tests and required validation, then proceed. Rerun only if production code or intended behavior changed.
- For a read-only id check, run `review.py --id <id> --show-status`.
- If the caller session was restarted after reviewer output was produced, run `review.py --id <id> --show-findings` to recover stored reviewer text without launching another review.
