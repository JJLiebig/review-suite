---
name: review-github
description: "Run the anchored GitHub PR review cycle and relay the returned review."
---

# Review GitHub

Use for PR-scoped GitHub review.

```powershell
<python> <review-suite-plugin-root>/scripts/review.py --id <id> --github-review
```

Use the installed Codex plugin cache root by default. When reviewing Review Suite itself, use the current source checkout only if the user explicitly requests dogfooding unsynced source changes. Never use `%USERPROFILE%\.codex\.tmp\marketplaces\review-suite`.

Rules:
- Run after local Review Suite review reaches the GitHub handoff action for the same id.
- Wait for wrapper output; do not post another request.
- Code only valid findings.
- Record the GitHub result on the owning local review id: `review.py --id <id> --github-result clean|findings`, or `--github-result waived --github-note "why"` when GitHub cannot run or no-GitHub is approved.
- After `--github-result findings`, fix the issue and follow the emitted actions. Apply the [no-repeat criteria](../review/SKILL.md): eligible fixes close with `--fixes-validated "<eligibility and validation passed>"` without another local or GitHub review; otherwise the same id reruns local signoff before GitHub is requested again.
- Do not call the PR final/merge-ready until full-suite/CI is passed or explicitly waived with a reason.
