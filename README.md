# Review Suite

Review Suite runs stateful local code review, tracks fixes and validation, and
can request a GitHub pull-request review. The calling agent can be Codex,
Claude Code, or another agent supported by the skills CLI. Review jobs use a
separate backend: Codex CLI by default.

## Install

Installation options:

| Route | Install | Update |
| --- | --- | --- |
| Codex marketplace | `codex plugin marketplace add JJLiebig/review-suite` then `codex plugin add review-suite@review-suite` | `codex plugin marketplace upgrade review-suite` |
| Skills CLI | `npx skills add JJLiebig/review-suite` | `npx skills update review-suite` |

The skills CLI installs into the current project by default. Add `--global`
to install for your user; update a global install with
`npx skills update -g review-suite`. It exposes one `review-suite` skill
containing the same four workflows and Python engine as the marketplace plugin.
The marketplace exposes those workflows as separate skills: `review`,
`review-plan`, `review-deslop`, and `review-github`.

`npx` runs the public skills CLI package; Review Suite itself is installed from
GitHub and needs no npm login or publish step.

Both routes need Python 3.14.6+ and Git. The skills CLI route also needs
Node.js/npm to run `npx`. Review Suite's default reviewer, plan review,
simplification review, and follow-up review require an authenticated Codex CLI.
Selected local review models can instead use an authenticated OpenCode CLI; see
[OpenCode configuration](docs/opencode.md). GitHub review also needs an
authenticated `gh` CLI and access to the GitHub `@codex review` bot.

## Review modes

| Mode | Use it for | Workflow |
| --- | --- | --- |
| `fast` | Small, well-tested changes with limited risk | Dual signoff; no cleanup or GitHub review; at most two local rounds |
| `normal` | Ordinary changes (default) | Optional Arena, one cleanup pass, dual signoff, then GitHub review |
| `deep` | Security, billing, migrations, concurrency, and other critical changes | Normal signoff followed by deeper signoff, with cleanup and GitHub review |

Choose by the change's actual risk, including trust and data-integrity
boundaries. [Review strategy](docs/review-strategy.md) describes the full
workflow.

## Run a review

Ask the installed skill for a local review, or run the launcher directly:

```powershell
<python> <plugin-root>/scripts/review.py --cd <repo-root>
```

`<plugin-root>` is the installed marketplace plugin root, or
`plugins/review-suite/` inside the installed skills CLI skill. Omit `--mode`
for `normal`; pass `--mode fast` or `--mode deep` when appropriate. Review Suite
detects the remote default branch; `--base <ref>` overrides it.

The launcher expects committed changes and a clean worktree. It prints an
`Action`: run its `cmd`, or inspect the review output and run exactly one
matching `choices` command. Continue an existing review with `--id <id>`.

Read-only status and recovery commands:

```powershell
<python> <plugin-root>/scripts/review.py --status --cd <repo-root>
<python> <plugin-root>/scripts/review.py --id <id> --show-status
<python> <plugin-root>/scripts/review.py --id <id> --show-findings
```

Run relevant validation before dispatch and record required full-suite and CI
results on the review id. A green review alone does not make a change
merge-ready. For interrupted reviews, changed heads, or escalation, use the
[recovery guide](plugins/review-suite/skills/review/references/recovery.md) and
the launcher's emitted action.

To skip Arena for one new cycle, pass `--no-arena`. To set or clear the
preference for future cycles in one repository, use `--set-no-arena` or
`--clear-no-arena` with `--cd <repo-root>`; existing cycles keep their frozen
plan.

## Settings

[Shipped defaults](plugins/review-suite/default_settings.toml) apply unless
overridden in `~/.codex/state/review-suite/settings.toml`. This state path is
used for both installation routes. The first run creates a comment-only
settings file, so future updates can change defaults you have not pinned.

For example, to override just the plan review's reasoning:

```toml
[jobs.plan]
reasoning = "high"
```

The jobs are `plan`, `deslop`, `followup`, `normal_signoff`, and `deep_signoff`.
Each can override `model`, `reasoning`, and `service_tier`; omitted values
inherit the normal or deep defaults. Review history and ratings live beside the
settings file. If only legacy `config.json` exists, the first run migrates its
non-model settings and keeps the original as a backup.

## Development

Development also requires `uv`. Run checks relevant to the changed files:

```powershell
uv sync
uv run ruff check .
```

Run `uv run pytest` with the tests relevant to your change. After changing
plugin files, `scripts/sync-installed-cache.ps1` refreshes the local marketplace
cache and source mirror. Marketplace launchers copy the installed plugin to
`~/.codex/plugin-runtimes/review-suite/` before running, so Windows file locks
do not block marketplace updates. Skills CLI installs run from their own skill
directory.
