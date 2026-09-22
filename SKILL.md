---
name: review-suite
description: Run Review Suite local review, plan review, simplification review, and GitHub review from an installed skill.
---

# Review Suite

This skill includes the Review Suite Python engine at `plugins/review-suite/`.
Use a Python 3.14.6+ interpreter and resolve paths relative to this `SKILL.md`:

| Task | Read the existing workflow | Entry point |
| --- | --- | --- |
| Local review or status | `plugins/review-suite/skills/review/SKILL.md` | `plugins/review-suite/scripts/review.py` |
| Plan review | `plugins/review-suite/skills/review-plan/SKILL.md` | `plugins/review-suite/scripts/review_plan.py` |
| One-off simplification | `plugins/review-suite/skills/review-deslop/SKILL.md` | `plugins/review-suite/scripts/review_deslop.py` |
| GitHub review | `plugins/review-suite/skills/review-github/SKILL.md` | `plugins/review-suite/scripts/review.py` |

Read the matching workflow before acting. In its commands, `<review-suite-plugin-root>`
means the `plugins/review-suite/` directory inside this installed skill. Use the
installed skill's files for every command, including status and follow-up actions.
The marketplace cache path rules in those workflows apply only to marketplace
installs. The calling agent can be any agent supported by the skills CLI. The
default review backend and the plan, deslop, and follow-up paths require Codex
CLI; selected local review models can use OpenCode when configured.
