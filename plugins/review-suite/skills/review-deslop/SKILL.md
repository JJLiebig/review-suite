---
name: review-deslop
description: "Run an explicit one-off simplification review for a completed implementation slice."
---

# Review Deslop

Use only for an explicit cleanup pass. Ordinary `review` runs manage their own one-shot cleanup pass immediately before final correctness signoff.

```powershell
<python> <review-suite-plugin-root>/scripts/review_deslop.py --cd <repo-root>
<python> <review-suite-plugin-root>/scripts/review_deslop.py --commit <sha> --cd <repo-root>
<python> <review-suite-plugin-root>/scripts/review_deslop.py --commit <from-sha> <to-sha> --cd <repo-root>
```

Use the installed Codex plugin cache root by default. When reviewing Review Suite itself, use the current source checkout only if the user explicitly requests dogfooding unsynced source changes. Never use `%USERPROFILE%\.codex\.tmp\marketplaces\review-suite`.

Rules:
- Simplification lane, not correctness signoff.
- Verify findings locally before editing.
- Use `--focus` for a coherence/reset cleanup pass.
- Do not loop this lane until green.
