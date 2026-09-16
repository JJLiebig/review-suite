---
name: review
description: "Run local code review/status; choose `fast`, `normal`, or `deep` by risk."
---

# Review

Use for local code review.

```powershell
<python> <review-suite-plugin-root>/scripts/review.py --cd <repo-root>
```

Path rules:
- `<review-suite-plugin-root>` is the installed Codex plugin cache root, such as `%USERPROFILE%\.codex\plugins\cache\review-suite\review-suite\0.1.0` for the git marketplace install.
- When reviewing Review Suite itself, use the current source checkout only if the user explicitly requests dogfooding unsynced source changes.
- Do not run scripts from `%USERPROFILE%\.codex\.tmp\marketplaces\review-suite`; that is Codex's marketplace source clone.
- Follow the emitted `Action`; its commands should use installed-launcher paths for runtime-backed runs.

Mode:
- Omit `--mode` for `normal`, the default for ordinary changes.
- Use `--mode fast` for UI-only, local presentation, and other small, well-tested changes. It runs dual signoff using the configured normal model with no Arena or GitHub review; convergence and bounded closure match the other modes.
- Use `--mode deep` for billing, login/authentication, authorization/security, business-critical systems, database integrity or migrations, concurrency, and similarly critical or high-blast-radius logic.
- Treat those mappings as risk heuristics. A nominally UI-only change that crosses a trust or data-integrity boundary is not `fast`.

Rules:
- Run validation relevant to the changed surface before dispatch. Start any required slow checks after dispatch and track their final status. Run a full suite only when repository requirements or reachable effects justify it; record an explicit reason when waiving an unnecessary full-suite or CI gate.
- Immediately before final correctness signoff, normal and deep modes run one bounded pass for frozen-brief conformance and local cleanup. Fast mode currently excludes this pass. Handle or dismiss its output, then close it with `review.py --id <id> --deslop-done`. Accepted edits proceed to final signoff on the new exact head. Cleanup runs once per cycle and does not restart after final-review or GitHub fixes.
- Three distinct caller-accepted findings heads require a durable `CONTINUE`, `REPLAN`, or `RESLICE` decision; `CONTINUE` is available once for one additional fix head and its correctness decision. Report conflicts with the frozen goal, acceptance, scope, stop condition, owner, authorized behavior, or unit boundary immediately with `review.py --id <id> --contract-conflict <dimension>`.
- Review orchestration expects committed review changes. If `git diff` is non-empty but `base..HEAD` is empty, commit intended changes or stash unrelated worktree changes before rerunning.
- There is no `--allow-dirty` override.
- Read `Output:`, then follow the emitted `Action`: run `cmd` when present, or classify the output and run exactly one matching `choices` command.
- For arena grading actions, grade only after checking findings against the diff/repo. Plausible but unverified findings do not count as valid. Use `scope_bloat_loss` when a review asks for product behavior, AI guardrails, validation, fallback behavior, UX policy, or safety checks that are not required by the diff, a real bug, a trust boundary, or the user request.
- Supply the requested rating pool and repeat `--rank` from best to worst; comma-separated variants within one rank tie. The caller grades; Review Suite never promotes a winner automatically.
- Without an id, use `review.py --status --cd <repo-root>` for branch routing.
- The default base is the repository's remote default branch. Use `--base <ref>` only as an explicit override.
- Resume with `review.py --id <id>` without creation flags. Use `--decision clean|findings` only when auto-advance cannot classify the verdict or the caller intentionally disagrees.
- Verify findings against the diff/repo before fixing them.
- After fixing accepted findings, run relevant validation. **Do not rerun review** when every fix is either:
  1. Docs/test-only, preserving intended behavior and test coverage.
  2. A low-blast-radius P2 or lower, outside high-stakes or business-critical behavior such as security, billing, data integrity, or concurrency.
- Otherwise, rerun on the same review id. Do not rerun solely for a clean verdict, a changed commit hash, or extra confidence. Report: **“Fixed and validated; no repeat review required.”**
- For eligible fixes, commit/amend and run `review.py --id <id> --fixes-validated "<eligibility and validation passed>"`. This records caller validation, preserves the findings, and skips repeat review. Remaining scheduled steps and required validation still apply.
- After GitHub review returns, record the result on the owning review id: `--github-result clean`, `--github-result findings`, or `--github-result waived --github-note "why"`. Do not start a new ladder for GitHub findings.
- Do not call PR-final/merge-ready until required validation passes. Record full-suite/CI gates as passed or explicitly waived with a reason; follow the emitted validation commands.

For status-only inspection, session recovery, changed-head recovery, closure dismissal, or escalation to deep mode, read [references/recovery.md](references/recovery.md) before acting.
