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

## Status by id

`review.py --id <id> --show-status --json` reads the recorded status for one
review cycle from any directory. `head` is the full commit recorded by the
cycle. `current_head` is the head used to report the recorded status, and
`accepted_head` identifies the commit covered by acceptance when available.
Consumers compare `accepted_head` to the current PR head before treating
`done` as approval for that commit. Review commands record any acceptance
carried forward to a new head and any base interaction checks. Use
`review.py --status` to inspect the live checkout.

`status`, `done`, and `next_action` say whether the review is still running,
finished, or waiting on the caller. An unknown id exits with code 2 and prints
`{"error": "unknown review cycle id: <id>", "status": "usage_error"}`.

An unsuperseded decided `reslice` cycle (or a budget-exhausted `replan` with
active findings) reports `done: false` and `next_action: "restart_brief"` in
both status modes. Use the linked-successor command shown in text status;
for `--restart-brief`, only RESLICE accepts a merge-base that has advanced by
ancestry.

When a supported restart links cycles, the predecessor includes
`superseded_by: {"review": "<successor id>"}` and the successor includes
`supersedes: {"review": "<predecessor id>"}`. These optional fields expose the
recorded links without changing either cycle's verdict, history or readiness.
Consumers can read each linked id to follow a chain; status reads never advance it.
