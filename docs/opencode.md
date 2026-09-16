# OpenCode review backend

Review Suite can run local reviewers through OpenCode without changing the review workflow or result contract.

Use an explicit `opencode::` model prefix. The remainder is passed to OpenCode as its normal `provider/model` identifier:

```toml
[normal]
model = "opencode::opencode-go/deepseek-v4.1-flash"
reasoning = "high"
```

For GLM-5.3-Flash, only the model value changes:

```toml
[normal]
model = "opencode::opencode-go/glm-5.3-flash"
reasoning = "high"
```

OpenCode maps `reasoning` to the `provider/model#variant` model suffix. The OpenCode Go models expose `low`, `high`, and `max` thinking levels (`opencode-go/deepseek-v4.1-flash` is DeepSeek V4.1 Flash, `opencode-go/glm-5.3-flash` is GLM-5.3-Flash). An explicit `reasoning` outside that set is rejected with the supported values; an unsupported level inherited from settings is normalized to the model's default (`high`). Models without a known mapping run at the provider's default reasoning behavior without a variant suffix.

## Runtime contract

The backend deliberately shares the existing Review Suite review prompt, target selection, terminal `Review result: clean|findings` protocol, classification, retries, orchestration, and Arena grading.

OpenCode is launched through a small adapter with a Review Suite-owned primary review agent. The adapter:

- runs `opencode run --standalone` non-interactively in the review checkout;
- requests the selected model's advertised maximum output from OpenCode's local `models.json` catalogue (`$XDG_CACHE_HOME/opencode`, default `~/.cache/opencode`), through subprocess-only configuration; ordinary OpenCode sessions and reasoning settings are unchanged. If the catalogue or a supported model adapter is unavailable, it warns and leaves the provider default in place;
- injects a review system prompt that mirrors Codex review's actionable-defect criteria while leaving Review Suite's output contract authoritative;
- denies shell, edits, subagents/tasks, web access, skills, questions, and external-directory access;
- allows only repository read/search/LSP tools;
- generates the bounded target patch itself and attaches it to the OpenCode run, so the model never needs shell access to inspect the diff;
- prefers the final JSON text step containing Review Suite's terminal result marker and falls back to `opencode session export --standalone <session>` when that step is missing. The export fallback also preserves completed responses without the marker for caller classification, as on the Codex path;
- captures per-step tokens and cost from the JSON event stream, reconciles them against `opencode session export --standalone <session>` (which also covers runs where OpenCode drops the final step event), and reports the provider's USD cost as authoritative.

Usage is normalized to Review Suite's token shape and the provider cost is stored as-is, so Arena leaderboards and the review cost ledger account for OpenCode reviews exactly like Codex reviews. OpenCode model pricing is never looked up locally; the provider-reported cost wins when present.

This intentionally keeps provider quirks inside the backend. Adding another OpenCode Go model should normally be a model/configuration change, not a new execution path.

## Availability, rate limits, and cooldowns

OpenCode reports provider failures as structured `{"type":"error", ...}` events with an `error.name` and `error.data.message` (and usually an empty stderr). The adapter classifies each failed run into one of three buckets and records it as `error_class` in its metadata line:

- `capacity` — rate limits, `429`, `Too Many Requests`, quota/usage limits, `resource_exhausted`, overloaded, or an "at capacity" message;
- `unavailable` — model not found / invalid / unsupported, authentication or authorization failures;
- `failed` — any other non-zero exit, including provider server errors that carry no recognizable message.

Review Suite maps these to `selected_model_at_capacity`, `selected_model_unavailable`, and `opencode_review_failed`. All three feed the same Arena cooldown/backoff already used for Codex capacity (`30m → 2h → 6h → 12h`). The failure count is retained across cooldown expiry so consecutive failures keep escalating, and it is cleared by the next successful review. The `capacity` bucket also gets the existing single 10-second retry before the round is finalized. OpenCode runs that fail without a usable review are marked cooldown-eligible, so a repeated provider failure rests the model instead of being re-selected every round.

Environment-level adapter failures (OpenCode CLI missing, an empty prompt, or a failed target-diff generation) are not provider failures and do not cool the variant down; they surface as tooling failures for the operator to fix.

## Prerequisites

Install and authenticate the OpenCode CLI normally, including the OpenCode Go subscription/provider. Review Suite does not copy or manage OpenCode credentials.

`service_tier` is Codex-specific and is rejected for OpenCode-backed reviewers rather than silently ignored.
