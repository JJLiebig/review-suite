# Review strategy

Public modes describe risk, not model experiments:

| Mode | Use | Ladder |
| --- | --- | --- |
| `fast` | UI-only, local presentation, and other small, well-tested changes | Dual configured normal-model signoff; no cleanup; at most two local rounds |
| `normal` | Everything else | Optional phase Arena rounds, one cleanup pass, dual configured normal-model signoff with risk-based fix verification, GitHub review |
| `deep` | Billing, authentication/login, authorization/security, database integrity or migrations, concurrency, and similarly critical logic | Dual configured normal-model signoff with risk-based fix verification, optional deep Arena rounds, one cleanup pass, dual configured deep-model signoff with risk-based fix verification, GitHub review |

Omitting `--mode` creates a `normal` review. Risk wins over labels: a UI change
that crosses a trust or data-integrity boundary is not `fast`.

Arena is opt-in. Stable profiles omit Arena steps when it is disabled or their
configured loop count is zero. Discovery pools, historical ratings, and manual
calibration remain available, but stable profiles do not run fixed discovery
brawls. The caller grades Arena output; Review Suite does not automatically
select or promote models.

The [generated workflow diagram](review-workflow.md) shows the sequences,
reviewer models, and optional Arena rounds from the shipped defaults.

Arena uses its mixed-model roster when enabled. GitHub review uses the GitHub
Codex service; these model settings do not select its model. Accepted findings require
fixes and relevant validation. Do not repeat review for docs/test-only fixes preserving
intended behavior and coverage, or low-blast-radius P2-or-lower fixes outside high-stakes
or business-critical behavior (security, billing, data integrity, concurrency).
Record these with `--fixes-validated "<eligibility and validation passed>"`; otherwise
rerun on the same id. Do not rerun solely for a clean verdict, a changed commit hash,
or extra confidence. Remaining scheduled steps and required validation still apply.
Cleanup runs once per normal/deep cycle and checks
simplification and conformance to the review brief. Apply accepted cleanup changes
before final signoff so that it reviews the resulting code. Later signoff or
GitHub fixes do not restart cleanup.

Earlier review passes, plan review, follow-up, and cleanup use `normal`. The
final correctness signoff in deep mode uses `deep`, unless overridden per job.
See [model settings](../README.md#settings) and the [Arena maintainer guide](arena.md).
