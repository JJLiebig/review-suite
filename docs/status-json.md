# Status JSON

`review.py --status --json --cd <worktree> [--base <base>]` reports the current
review without advancing it. Existing routing fields keep their meanings. When
a current review cycle exists, the response also includes:

- `steps`: the planned ladder in order, with zero-based `index`, `name`, and
  `state` (`done`, `current`, or `pending`), from `review_plan` and
  `review_progress`.
- `round_history`: the latest 20 round records, oldest first. `number` is the
  one-based position in the complete cycle, so a truncated history keeps its
  original numbering. Each entry has `round_id`, `lane`, `step` (the profile step
  name, or null), `status`, `outcome`, `findings`, `started_at`, and `finished_at`.
- `rounds`: the total number of round records in the full cycle. Subtract
  `round_history.length` to count earlier rounds omitted from the response.
- `started_at`: the first round's recorded `review_started_at`, or null. This
  marks the first reviewer launch, not cycle creation. If that time is missing,
  a later round's time is not substituted.
- `round_started_at`: the latest round's recorded `review_started_at`, or null,
  including when that round has completed.

Round times come from stored round output (`review_started_at` and
`review_completed_at`). They are ISO 8601 strings when recorded; missing times
are null. File modification times, sampling times, and elapsed durations are
never used to invent timestamps.

`outcome` is `findings`, `clean`, `blocked`, or null. Recorded caller corrections
and finding dispositions take precedence over the original audit decisions,
which take precedence over reviewer verdicts. Without a caller decision, blocked
flags take precedence over structured reviewer verdicts. An unfinished or unclassified
round has a null outcome. Review Suite does not persist numeric finding counts,
so `findings` is null even for clean rounds; it never counts reviewer prose.

The added fields contain no prompts, finding bodies, file contents, or raw round
payloads. They also appear in `review.py --id <id> --show-status`; the compact
default branch status output remains unchanged.
