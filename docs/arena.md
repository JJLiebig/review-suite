# Arena

Stable profiles do not run discovery brawls. Discovery pools and ratings remain
available for deliberate calibration.

Arena is an opt-in evaluation overlay. Normal and deep run their configured
Arena counts only when Arena is enabled and the count is positive. The calling
agent grades the outputs; Review Suite never selects or promotes a winner.
Each reporting pool uses its balanced groups once, then favors under-sampled
candidates and opponents they have met least often. When both cohorts can fill
half a group, bootstrap rounds mix under-sampled and established candidates
evenly. New candidates join the existing pool at 1500 Elo without resetting
established ratings.

The roster includes `gpt-6-sol` and `gpt-6-luna` with the supported
low/medium/high/xhigh/max efforts. Review task assignments follow Sol and Luna,
including disabled Luna low. `gpt-6-astra-minor` remains disabled and excluded
from the pools until the model resolves; its efforts provisionally follow Astra.
Sol and Luna rates follow the [launch announcement](https://openai.com/index/introducing-gpt-6-sol-and-luna/):
$2/$10 and $0.10/$0.50 per million input/output tokens respectively.
Cached reads use 0.1x input and cache writes 1.25x input under the
[GPT-5.6-and-later caching policy](https://developers.openai.com/api/docs/guides/prompt-caching).
Astra Minor remains unpriced.
Existing model defaults and rating pool IDs are unchanged.
Pinned user `variant_ids` overrides must include the new candidates to sample them.

## Configuration

Arena is disabled by default and is not needed for ordinary reviews.
The shipped maintainer settings are in
[`arena_settings.toml`](../plugins/review-suite/references/arena_settings.toml):
comparison pools, scheduling groups, rating pool IDs, enablement, and Arena loop counts.

To opt in, add only the overrides you need to
`~/.codex/state/review-suite/settings.toml`:

```toml
[arena]
enabled = true

[orchestrator.stable_defaults]
normal_arena_loops = 1
deep_arena_loops = 1
```

Existing overrides continue to work. Keep rating pool IDs stable when moving
settings so historical ratings retain their identity.

Use `review.py --no-arena --cd <repo-root>` to skip Arena for one new review
cycle. To disable Arena for future reviews in one repository while retaining
the global setting elsewhere, use `review.py --set-no-arena --cd <repo-root>`;
`--clear-no-arena` returns that repository to the global setting. These
repository preferences are local to the clone and do not modify tracked files.

Review sequences and reviewer counts live separately in
[`workflow_settings.toml`](../plugins/review-suite/references/workflow_settings.toml).
The loader merges workflow settings, Arena settings, model defaults, then user
overrides. Installed runtime copies include all three shipped files.
