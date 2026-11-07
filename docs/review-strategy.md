# Review strategy

Public modes describe risk, not model experiments:

| Mode | Use | Ladder |
| --- | --- | --- |
| `fast` | UI-only, local presentation, and other small, well-tested changes | Dual configured base-model signoff; no cleanup; at most two local rounds |
| `base` | Everything else | Optional base Arena rounds, one cleanup pass, dual configured base-model signoff with risk-based fix verification, GitHub review |
| `deep` | Billing, authentication/login, authorization/security, database integrity or migrations, concurrency, and similarly critical logic | Dual configured base-model signoff with risk-based fix verification, optional deep Arena rounds, one cleanup pass, dual configured deep-model signoff with risk-based fix verification, GitHub review |

Omitting `--mode` creates a `base` review. Risk wins over labels: a UI change
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
Cleanup runs once per base/deep cycle and checks
simplification and conformance to the review brief. Apply accepted cleanup changes
before final signoff so that it reviews the resulting code. Later signoff or
GitHub fixes do not restart cleanup.

Earlier review passes, plan review, follow-up, and cleanup use `base`. The
final correctness signoff in deep mode uses `deep`, unless overridden per job.
See [model settings](../README.md#settings) and the [Arena maintainer guide](arena.md).

## Mode diagram

The README's mode diagram is drawn by hand from
[its source](review-workflow.workflow.json), unlike the generated text workflow.
Update the source and redraw it whenever shipped models or steps change.
Use a temporary checkout of
[Archify v3.2.0](https://github.com/tt-a1i/archify/releases/tag/v3.2.0)
(commit `208acba8d412335185b2056fedf54b3405df43fb`), set
`meta.repository.revision` in the source to the commit it describes, and run
from the repository root with `<tmp>` outside the repository:

```powershell
$env:ARCHIFY_UPDATE_CHECK_DISABLED = '1'
node <archify>/archify/bin/archify.mjs finalize workflow docs/review-workflow.workflow.json <tmp>/review-workflow.html --repo-root . --quality showcase --json
node scripts/export-workflow-svg.mjs <tmp>/review-workflow.html docs/review-workflow-dark.svg docs/review-workflow-light.svg --loop f_sign,b_sign,d_base,d_sign
```

All four Archify gates must pass. The export needs Google Chrome; set `CHROME`
to its path if it is not in the default Windows location. Check both SVGs in a
browser before committing. `--loop` draws the repeat arrow under each signoff,
which Archify cannot route itself.
