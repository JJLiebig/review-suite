# Arena

Stable profiles do not run discovery brawls. Discovery pools and ratings remain
available for deliberate calibration.

Arena is an opt-in evaluation overlay. Base and deep run their configured
Arena counts only when Arena is enabled and the count is positive. The calling
agent grades the outputs; Review Suite never selects or promotes a winner.
Each reporting pool uses its balanced groups once, then favors under-sampled
candidates and opponents they have met least often. When both cohorts can fill
half a group, bootstrap rounds mix under-sampled and established candidates
evenly. New candidates join the existing pool at 1500 Elo without resetting
established ratings.

The roster includes `gpt-6-sol` and `gpt-6-luna` with the supported
low/medium/high/xhigh/max efforts. Review task assignments follow Sol and Luna; all supported Luna efforts are active, with low limited to base reviews. `gpt-6-astra-minor` remains disabled and excluded
from the pools until the model resolves; its efforts provisionally follow Astra.
Sol and Luna rates follow the [launch announcement](https://openai.com/index/introducing-gpt-6-sol-and-luna/):
$2/$10 and $0.10/$0.50 per million input/output tokens respectively.
Cached reads use 0.1x input and cache writes 1.25x input under the
[GPT-5.6-and-later caching policy](https://developers.openai.com/api/docs/guides/prompt-caching).
GPT-6.1 Sol uses the same effort allocation as GPT-6 Sol: base low/medium;
PR medium/high/xhigh/max. Its [launch pricing](https://developers.openai.com/api/docs/models/gpt-6.1-sol)
is $2 input, $0.10 cached input, and $10 output per million tokens; cache writes
use the policy above ($2.50 per million). It is enabled for rollout, but the
local ChatGPT-account availability check rejected the model on 2026-09-29.
Astra Minor remains unpriced.
Existing model defaults and rating pool IDs are unchanged.
Pinned user `variant_ids` overrides must include the new candidates to sample them.

GPT-6.1 Terra is available at low/medium/high/xhigh for base reviews and
medium/high/xhigh/max for deep reviews. Its [standard API rates](https://developers.openai.com/api/docs/models/gpt-6.1-terra)
are $1 input, $0.05 cached input, $1.25 cache write, and $5 output per million
tokens. Estimates use these standard short-context rates; long-context and
speed-tier pricing can differ.

DeepSeek V4.1 Flash 1017 uses separate `deepseek-v4.1-flash-1017-*` variants
at low/high/max effort (low is base-only). The earlier variants are disabled;
their ratings remain in history. The [1017 update](https://api-docs.deepseek.com/updates/)
keeps the API name and pricing unchanged, so the OpenCode route remains
`opencode-go/deepseek-v4.1-flash`.

## Automatic retirement

Arena automatically keeps the newest **two released model generations per
lineage** eligible for new rounds. Register a release by adding its variants with
an explicit `model_id` and appending that ID to the lineage's chronological list
in `references/roster.json`. Add its eligible variants to the comparison pools as
usual. There is no separate retirement command or refresh step to run for an
older generation: normal Arena loading and dispatch apply the policy.

The lists are release order, not a version-name sort. Sol, Terra, Luna, Astra,
Opus, and Sonnet are separate lineages. Each generation contains all its declared
reasoning efforts and task assignments. Unreleased/disabled previews must not be
added to a released lineage list. A snapshot that reuses an API route still gets
its own `model_id`, as the two DeepSeek Flash revisions do. Models without a
declared lineage can be retired explicitly.

The shipped policy retires GPT-5.6 Sol because GPT-6 Sol and GPT-6.1 Sol follow it.
GPT-5.6 Terra and legacy GPT-5.5 and GPT-5.4 models are explicitly retired.
GPT-5.6 Luna remains an eligible predecessor.
Other disabled states, task limits, reasoning settings, and ordinary signoff
jobs are unchanged.

Retirement is permanent for a model generation. Review Suite stores model and
variant tombstones in `~/.codex/state/review-suite/arena_retirements.json` (or the
selected state directory). Normal Arena commands record new automatic
retirements before dispatch; `--dry-run` applies the policy without writing
state. Retiring the predecessor early does not bring back an older generation.
Later roster edits, an old roster snapshot, a new reasoning effort, or removal and
re-addition of an ID cannot undo an existing tombstone. Keep this file with the
rest of the review state when backing up or migrating machines. Its retirement
date is when this installation first recorded the retirement, not a guessed
historical release or retirement date. Older scheduler binaries predate this
feature; retain a retirement-aware scheduler when rolling back an installation.

For an optional early retirement of every effort of one generation:

```powershell
<python> <plugin-root>/scripts/review_suite_arena.py retire-model --model-id MODEL_ID --reason "No longer useful for my reviews"
```

This updates the retirement ledger and reports; it does not call any models.
There is no unretire command. Queued Arena rounds are checked again before each
reviewer starts, including retries and resumed rounds. A retired queued reviewer
is blocked; already running reviewers can finish and their results remain valid.
Stale configured groups automatically select a full group from the remaining
eligible configured models, or fail clearly if there are too few. Existing
rating pool IDs are preserved.

Models with recorded results remain in `summary.json` and the local leaderboard,
with their metrics intact, including retired and disabled variants. Every
historical match continues to contribute to ratings. Summary rows include
their scheduling status and, for declared lineages, their generation position:

```json
{"lifecycle": {"status": "retired", "retired_at": "2026-11-02T12:00:00Z", "reason": "Retired in the roster", "lineage": "openai-terra", "generation": "previous"}}
```

Active rows use `{"status":"active"}`. Read-only previews may omit the date for
retirements not yet persisted. Arena Web consumes this metadata to separate the
active board from its retired-model archive; it still fits all retained matches.

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
base_arena_loops = 1
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

## Local rollout handoff

Deploy the paired Arena Web lifecycle consumer before updating the local Review
Suite installation, so the first expanded summary can distinguish archived rows.
Normal Arena work then records automatic retirements without an operator action.
Verify the effective local roster/pools and preserve the retirement ledger with
review history. Ordinary signoff configuration, Arena loop counts, and rating
math are unchanged by this feature.
