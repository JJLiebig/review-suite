from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import review_github

from review_github import (
    cmd_run,
    inspect_existing_cycle,
    public_cycle_payload,
    run_review_cycle,
)

BOT = "chatgpt-codex-connector[bot]"


@pytest.fixture(autouse=True)
def github_session(monkeypatch) -> review_github.GitHubSession:
    session = review_github.GitHubSession(token="test-token")
    monkeypatch.setattr(review_github, "GITHUB", session)

    def no_network(*args: object) -> None:
        raise AssertionError("test tried to reach GitHub")

    monkeypatch.setattr(review_github, "http_send", no_network)
    return session


def _request_comment(
    *,
    comment_id: str = "request-1",
    created_at: str = "2026-04-20T10:00:00Z",
    head_sha: str = "deadbeef",
) -> dict[str, object]:
    return {
        "id": comment_id,
        "node_id": f"IC_{comment_id}",
        "body": review_github.build_request_body("@codex review", head_sha),
        "user": {"login": "alice"},
        "created_at": created_at,
        "html_url": "https://github.com/example-owner/example-repo/pull/87#issuecomment-request-1",
    }


def _snapshot(
    *,
    head_sha: str = "deadbeef",
    issue_comments: list[dict[str, object]] | None = None,
    reactions: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "head_sha": head_sha,
        "issue_comments": issue_comments or [],
        "reviews": [],
        "review_comments": [],
        "anchor_reactions": [
            {"content": content, "user": {"login": BOT}} for content in reactions
        ],
    }


def _patch_existing_cycle(
    monkeypatch,
    *,
    items: list[dict[str, object]],
    reactions: tuple[str, ...] = (),
    head_commit_at: datetime | None = None,
) -> None:
    snapshot = _snapshot(issue_comments=[_request_comment()], reactions=reactions)
    monkeypatch.setattr(
        "review_github.fetch_pr_snapshot", lambda *args, **kwargs: snapshot
    )
    monkeypatch.setattr("review_github.collect_cycle_items", lambda **kwargs: items)
    monkeypatch.setattr(
        "review_github.get_commit_timestamp", lambda *args, **kwargs: head_commit_at
    )


def _existing_cycle(reason: str, items: list[dict[str, object]]) -> dict[str, object]:
    return {
        "reason": reason,
        "request_comment_id": "request-1",
        "request_comment_node_id": "IC_request-1",
        "request_comment_url": "https://github.com/example-owner/example-repo/pull/87#issuecomment-request-1",
        "requested_at": "2026-04-20T10:00:00Z",
        "items": items,
    }


def _inspect() -> dict[str, object] | None:
    return inspect_existing_cycle(
        owner="example-owner",
        repo="sample-web",
        pr_number=87,
        body="@codex review",
        bot_login=BOT,
        head_sha="deadbeef",
    )


def _response_item(
    *,
    item_id: str,
    kind: str,
    body: str,
    created_at: str | None,
    url: str,
    updated_at: str | None = None,
    commit_id: str | None = None,
    path: str | None = None,
    line: int | None = None,
) -> dict[str, object]:
    return {
        "kind": kind,
        "id": item_id,
        "body": body,
        "created_at": created_at,
        "updated_at": updated_at,
        "commit_id": commit_id,
        "url": url,
        "html_url": url,
        "path": path,
        "line": line,
        "state": "COMMENTED" if kind == "review" else None,
    }


def test_build_request_body_keeps_public_comment_plain() -> None:
    assert (
        review_github.build_request_body("@codex review", "deadbeef") == "@codex review"
    )
    assert "Reviewed commit" not in review_github.build_request_body(
        "@codex review", "deadbeef"
    )


def test_collect_cycle_items_ignores_codex_progress_summary(monkeypatch) -> None:
    summary = {
        "id": 1,
        "user": {"login": "chatgpt-codex-connector[bot]"},
        "body": "<!-- codex-pull-request-review-summary -->\n## Codex Review Summary\n| 📝 Code Review | ⏳ In progress |",
        "created_at": "2026-04-20T10:01:00Z",
    }
    final_review = {
        "id": 2,
        "user": {"login": "chatgpt-codex-connector[bot]"},
        "body": "### Codex Review\n\nReviewed commit: `deadbeef`",
        "submitted_at": "2026-04-20T10:02:00Z",
    }
    quoted_marker = {
        "id": 3,
        "user": {"login": "chatgpt-codex-connector[bot]"},
        "body": "Codex Review: Check `<!-- codex-pull-request-review-summary -->`.",
        "created_at": "2026-04-20T10:03:00Z",
    }
    comments = [summary]
    reviews = []
    kwargs = {
        "snapshot": {
            "issue_comments": comments,
            "review_comments": [],
            "reviews": reviews,
        },
        "bot_login": "chatgpt-codex-connector[bot]",
        "head_sha": "deadbeef",
        "anchor_since": datetime(2026, 4, 20, 10, 0, tzinfo=timezone.utc),
    }

    assert review_github.collect_cycle_items(**kwargs) == []
    summary["body"] = summary["body"].replace("In progress", "Completed")
    comments.append(quoted_marker)
    reviews.append(final_review)
    assert [item["body"] for item in review_github.collect_cycle_items(**kwargs)] == [
        quoted_marker["body"],
        final_review["body"]
    ]


def test_request_body_matching_is_exact_after_whitespace_normalization() -> None:
    assert review_github.request_body_matches(
        request_body=" @codex   review ",
        expected_prefix="@codex review",
        head_sha="deadbeef",
    )
    assert not review_github.request_body_matches(
        request_body="@codex review\n\nReviewed commit: `deadbeef`",
        expected_prefix="@codex review",
        head_sha="deadbeef",
    )


def test_public_cycle_payload_surfaces_verbatim_responses_without_duplicate_aliases() -> (
    None
):
    payload = {
        "status": "response_found",
        "pr_number": 87,
        "url": "https://github.com/example-owner/example-repo/pull/87",
        "head_sha": "deadbeef",
        "request_attempts": 2,
        "items": [
            _response_item(
                item_id="issue-1",
                kind="issue_comment",
                body="Codex Review: Didn't find any major issues. Already looking forward to the next diff.",
                created_at="2026-04-20T10:01:00Z",
                url="https://github.com/example-owner/example-repo/pull/87#issuecomment-1",
            ),
            _response_item(
                item_id="comment-1",
                kind="review_comment",
                body="Please rename this helper for clarity.",
                created_at="2026-04-20T10:01:05Z",
                url="https://github.com/example-owner/example-repo/pull/87#discussion_r1",
                path="src/app.ts",
                line=42,
            ),
        ],
    }

    result = public_cycle_payload(payload)

    assert "summary" not in result
    assert "findings" not in result
    assert result["attempts"] == 2
    assert "main_response" not in result
    assert "child_comments" not in result
    assert (
        result["responses"][0]["body"]
        == "Codex Review: Didn't find any major issues. Already looking forward to the next diff."
    )
    assert result["responses"][1]["body"] == "Please rename this helper for clarity."
    assert result["responses"][1]["loc"] == "src/app.ts:42"


def test_inspect_existing_cycle_reuses_any_bot_response_without_cleanliness_judgment(
    monkeypatch,
) -> None:
    response = _response_item(
        item_id="issue-1",
        kind="issue_comment",
        body="Codex Review: Didn't find any major issues. Already looking forward to the next diff.",
        created_at="2026-04-20T10:01:00Z",
        url="https://github.com/example-owner/example-repo/pull/87#issuecomment-1",
    )

    _patch_existing_cycle(
        monkeypatch,
        items=[response],
    )

    cycle = _inspect()

    assert cycle == _existing_cycle("existing_request_has_response", [response])


def test_inspect_existing_cycle_prefers_eyes_reaction_over_recently_updated_partial_response(
    monkeypatch,
) -> None:
    response = _response_item(
        item_id="issue-1",
        kind="issue_comment",
        body="Codex Review: still streaming.",
        created_at="2026-04-20T10:01:00Z",
        url="https://github.com/example-owner/example-repo/pull/87#issuecomment-1",
        updated_at="2026-04-20T10:01:25Z",
    )

    _patch_existing_cycle(
        monkeypatch,
        items=[response],
        reactions=("eyes",),
    )
    monkeypatch.setattr(
        "review_github.utc_now",
        lambda: datetime(2026, 4, 20, 10, 1, 30, tzinfo=timezone.utc),
    )

    cycle = _inspect()

    assert cycle == _existing_cycle("existing_request_has_eyes", [])


def test_inspect_existing_cycle_reuses_settled_response_even_if_eyes_reaction_remains(
    monkeypatch,
) -> None:
    response = _response_item(
        item_id="issue-1",
        kind="issue_comment",
        body="Codex Review: No findings.",
        created_at="2026-04-20T10:01:00Z",
        url="https://github.com/example-owner/example-repo/pull/87#issuecomment-1",
        updated_at="2026-04-20T10:01:05Z",
    )

    _patch_existing_cycle(
        monkeypatch,
        items=[response],
        reactions=("eyes",),
    )
    monkeypatch.setattr(
        "review_github.utc_now",
        lambda: datetime(2026, 4, 20, 10, 1, 30, tzinfo=timezone.utc),
    )

    cycle = _inspect()

    assert cycle == _existing_cycle("existing_request_has_response", [response])


def test_inspect_existing_cycle_does_not_reuse_response_when_head_commit_is_newer_than_response_creation(
    monkeypatch,
) -> None:
    response = _response_item(
        item_id="issue-1",
        kind="issue_comment",
        body="Codex Review: still streaming.",
        created_at="2026-04-20T10:01:00Z",
        url="https://github.com/example-owner/example-repo/pull/87#issuecomment-1",
        updated_at="2026-04-20T10:01:25Z",
    )

    _patch_existing_cycle(
        monkeypatch,
        items=[response],
        reactions=("eyes",),
        head_commit_at=datetime(2026, 4, 20, 10, 1, 20, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        "review_github.utc_now",
        lambda: datetime(2026, 4, 20, 10, 1, 30, tzinfo=timezone.utc),
    )

    cycle = _inspect()

    assert cycle is None


def test_inspect_existing_cycle_ignores_later_child_comment_creation_for_head_freshness(
    monkeypatch,
) -> None:
    top_level_response = _response_item(
        item_id="issue-1",
        kind="issue_comment",
        body="Codex Review: still streaming.",
        created_at="2026-04-20T10:01:00Z",
        url="https://github.com/example-owner/example-repo/pull/87#issuecomment-1",
        updated_at="2026-04-20T10:01:25Z",
    )
    child_comment = _response_item(
        item_id="comment-1",
        kind="review_comment",
        body="Please rename this helper.",
        created_at="2026-04-20T10:01:25Z",
        url="https://github.com/example-owner/example-repo/pull/87#discussion_r1",
        path="src/app.ts",
        line=42,
    )

    _patch_existing_cycle(
        monkeypatch,
        items=[top_level_response, child_comment],
        reactions=("eyes",),
        head_commit_at=datetime(2026, 4, 20, 10, 1, 10, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        "review_github.utc_now",
        lambda: datetime(2026, 4, 20, 10, 1, 30, tzinfo=timezone.utc),
    )

    cycle = _inspect()

    assert cycle is None


def test_inspect_existing_cycle_handles_response_items_without_created_at(
    monkeypatch,
) -> None:
    response = _response_item(
        item_id="issue-1",
        kind="issue_comment",
        body="Codex Review: No findings.",
        created_at=None,
        url="https://github.com/example-owner/example-repo/pull/87#issuecomment-1",
        updated_at="2026-04-20T10:01:05Z",
    )

    _patch_existing_cycle(
        monkeypatch,
        items=[response],
    )

    cycle = _inspect()

    assert cycle == _existing_cycle("existing_request_has_response", [response])


def test_emit_new_items_ignores_stale_top_level_response_with_reviewed_commit_body() -> (
    None
):
    anchor_since = datetime(2026, 4, 20, 10, 0, 0, tzinfo=timezone.utc)
    old_head = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    current_head = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

    emitted = review_github.emit_new_items(
        kind="issue_comment",
        items=[
            _response_item(
                item_id="issue-1",
                kind="issue_comment",
                body=f"Codex Review: No findings.\n\nReviewed commit: `{old_head}`",
                created_at="2026-04-20T10:01:00Z",
                url="https://github.com/example-owner/example-repo/pull/87#issuecomment-1",
            )
        ],
        head_sha=current_head,
        anchor_since=anchor_since,
        seen=set(),
    )

    assert emitted == []


def test_emit_new_items_accepts_matching_reviewed_commit_prefix_body() -> None:
    anchor_since = datetime(2026, 4, 20, 10, 0, 0, tzinfo=timezone.utc)
    current_head = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

    emitted = review_github.emit_new_items(
        kind="issue_comment",
        items=[
            _response_item(
                item_id="issue-1",
                kind="issue_comment",
                body="Codex Review: No findings.\n\nReviewed commit: `bbbbbbb`",
                created_at="2026-04-20T10:01:00Z",
                url="https://github.com/example-owner/example-repo/pull/87#issuecomment-1",
            )
        ],
        head_sha=current_head,
        anchor_since=anchor_since,
        seen=set(),
    )

    assert len(emitted) == 1
    assert emitted[0]["commit_id"] == "bbbbbbb"


def test_inspect_existing_cycle_uses_latest_explicit_head_response_for_head_freshness(
    monkeypatch,
) -> None:
    early_response = _response_item(
        item_id="issue-1",
        kind="issue_comment",
        body="Codex Review: still streaming.",
        created_at="2026-04-20T10:01:00Z",
        url="https://github.com/example-owner/example-repo/pull/87#issuecomment-1",
    )
    final_response = _response_item(
        item_id="issue-2",
        kind="issue_comment",
        body="Codex Review: No findings.",
        created_at="2026-04-20T10:01:20Z",
        url="https://github.com/example-owner/example-repo/pull/87#issuecomment-2",
        commit_id="deadbeef",
    )

    _patch_existing_cycle(
        monkeypatch,
        items=[early_response, final_response],
        head_commit_at=datetime(2026, 4, 20, 10, 1, 10, tzinfo=timezone.utc),
    )

    cycle = _inspect()

    assert cycle == _existing_cycle("existing_request_has_response", [early_response, final_response])


def test_inspect_existing_cycle_uses_latest_top_level_response_timestamp_when_no_commit_binding_exists(
    monkeypatch,
) -> None:
    early_response = _response_item(
        item_id="issue-1",
        kind="issue_comment",
        body="Codex Review: still streaming.",
        created_at="2026-04-20T10:01:00Z",
        url="https://github.com/example-owner/example-repo/pull/87#issuecomment-1",
    )
    final_response = _response_item(
        item_id="issue-2",
        kind="issue_comment",
        body="Codex Review: No findings.",
        created_at="2026-04-20T10:01:20Z",
        url="https://github.com/example-owner/example-repo/pull/87#issuecomment-2",
    )

    _patch_existing_cycle(
        monkeypatch,
        items=[early_response, final_response],
        head_commit_at=datetime(2026, 4, 20, 10, 1, 10, tzinfo=timezone.utc),
    )

    cycle = _inspect()

    assert cycle == _existing_cycle("existing_request_has_response", [early_response, final_response])


def test_inspect_existing_cycle_uses_explicit_inline_comment_binding_for_head_freshness(
    monkeypatch,
) -> None:
    early_response = _response_item(
        item_id="issue-1",
        kind="issue_comment",
        body="Codex Review: still streaming.",
        created_at="2026-04-20T10:01:00Z",
        url="https://github.com/example-owner/example-repo/pull/87#issuecomment-1",
    )
    inline_comment = _response_item(
        item_id="comment-1",
        kind="review_comment",
        body="Please rename this helper.",
        created_at="2026-04-20T10:01:20Z",
        url="https://github.com/example-owner/example-repo/pull/87#discussion_r1",
        commit_id="deadbeef",
        path="src/app.ts",
        line=42,
    )

    _patch_existing_cycle(
        monkeypatch,
        items=[early_response, inline_comment],
        head_commit_at=datetime(2026, 4, 20, 10, 1, 10, tzinfo=timezone.utc),
    )

    cycle = _inspect()

    assert cycle == _existing_cycle("existing_request_has_response", [early_response, inline_comment])


def test_inspect_existing_cycle_uses_later_top_level_reply_after_explicit_head_binding(
    monkeypatch,
) -> None:
    early_response = _response_item(
        item_id="issue-1",
        kind="issue_comment",
        body="Codex Review: still streaming.",
        created_at="2026-04-20T10:01:00Z",
        url="https://github.com/example-owner/example-repo/pull/87#issuecomment-1",
    )
    inline_comment = _response_item(
        item_id="comment-1",
        kind="review_comment",
        body="Please rename this helper.",
        created_at="2026-04-20T10:01:15Z",
        url="https://github.com/example-owner/example-repo/pull/87#discussion_r1",
        commit_id="deadbeef",
        path="src/app.ts",
        line=42,
    )
    final_response = _response_item(
        item_id="issue-2",
        kind="issue_comment",
        body="Codex Review: No findings.",
        created_at="2026-04-20T10:01:20Z",
        url="https://github.com/example-owner/example-repo/pull/87#issuecomment-2",
    )

    _patch_existing_cycle(
        monkeypatch,
        items=[early_response, inline_comment, final_response],
        head_commit_at=datetime(2026, 4, 20, 10, 1, 18, tzinfo=timezone.utc),
    )

    cycle = _inspect()

    assert cycle == _existing_cycle("existing_request_has_response", [early_response, inline_comment, final_response])


def test_inspect_existing_cycle_does_not_reuse_acknowledged_without_body_when_head_moved(
    monkeypatch,
) -> None:
    _patch_existing_cycle(
        monkeypatch,
        items=[],
        reactions=("+1",),
        head_commit_at=datetime(2026, 4, 20, 10, 1, 0, tzinfo=timezone.utc),
    )

    cycle = inspect_existing_cycle(
        owner="example-owner",
        repo="sample-web",
        pr_number=87,
        body="@codex review",
        bot_login="chatgpt-codex-connector[bot]",
        head_sha="cafebabe",
    )

    assert cycle is None


def test_inspect_existing_cycle_does_not_reuse_eyes_only_request_when_head_moved(
    monkeypatch,
) -> None:
    _patch_existing_cycle(
        monkeypatch,
        items=[],
        reactions=("eyes",),
        head_commit_at=datetime(2026, 4, 20, 10, 1, 0, tzinfo=timezone.utc),
    )

    cycle = inspect_existing_cycle(
        owner="example-owner",
        repo="sample-web",
        pr_number=87,
        body="@codex review",
        bot_login="chatgpt-codex-connector[bot]",
        head_sha="cafebabe",
    )

    assert cycle is None


def test_inspect_existing_cycle_does_not_reuse_active_request_when_head_moved(
    monkeypatch,
) -> None:
    _patch_existing_cycle(
        monkeypatch,
        items=[],
        head_commit_at=datetime(2026, 4, 20, 10, 1, 0, tzinfo=timezone.utc),
    )

    cycle = inspect_existing_cycle(
        owner="example-owner",
        repo="sample-web",
        pr_number=87,
        body="@codex review",
        bot_login="chatgpt-codex-connector[bot]",
        head_sha="cafebabe",
    )

    assert cycle is None


def test_cmd_run_records_workflow_anchor_for_completed_cycle(
    monkeypatch, tmp_path: Path
) -> None:
    emitted: list[dict[str, object]] = []
    recorded: list[dict[str, object]] = []

    monkeypatch.setattr(review_github, "resolve_repo_root", lambda cd: tmp_path)
    monkeypatch.setattr(
        review_github,
        "local_checkout_contains_reviewed_head",
        lambda review_root, head_sha: True,
    )
    monkeypatch.setattr(
        review_github, "merge_base", lambda review_root, base_ref: "base-sha-123"
    )
    monkeypatch.setattr(
        review_github,
        "get_pr_context",
        lambda owner, repo, pr_number: {
            "owner": "example-owner",
            "repo": "sample-web",
            "pr_number": 87,
            "url": "https://github.com/example-owner/example-repo/pull/87",
            "head_sha": "deadbeef",
            "base_ref": "main",
        },
    )
    monkeypatch.setattr(
        review_github,
        "run_review_cycle",
        lambda **kwargs: {
            "status": "response_found",
            "owner": "example-owner",
            "repo": "sample-web",
            "pr_number": 87,
            "url": "https://github.com/example-owner/example-repo/pull/87",
            "head_sha": "deadbeef",
            "items": [
                {
                    "kind": "issue_comment",
                    "url": "https://github.com/example-owner/example-repo/pull/87#issuecomment-1",
                    "body": "No findings.",
                    "created_at": "2026-04-20T10:01:00Z",
                }
            ],
        },
    )
    monkeypatch.setattr(
        review_github,
        "record_review_anchor",
        lambda **kwargs: recorded.append(kwargs) or {},
    )
    monkeypatch.setattr(review_github, "default_state_dir", lambda: tmp_path / "state")
    monkeypatch.setattr(
        review_github, "emit_toon", lambda payload: emitted.append(payload)
    )

    exit_code = cmd_run(
        review_github.build_parser().parse_args(["run", "--cd", str(tmp_path)]),
    )

    assert exit_code == 0
    assert emitted
    assert recorded[0]["lane"] == "review-github"
    assert recorded[0]["base"] == "main"
    assert recorded[0]["reviewed_head"] == "deadbeef"
    assert recorded[0]["review_scope"]["merge_base"] == "base-sha-123"


def test_cmd_run_does_not_record_workflow_anchor_for_acknowledged_without_body(
    monkeypatch, tmp_path: Path
) -> None:
    emitted: list[dict[str, object]] = []
    recorded: list[dict[str, object]] = []
    custom_state_dir = tmp_path / "custom-state"

    monkeypatch.setattr(review_github, "resolve_repo_root", lambda cd: tmp_path)
    monkeypatch.setattr(
        review_github, "merge_base", lambda review_root, base_ref: "base-sha-456"
    )
    monkeypatch.setattr(
        review_github,
        "get_pr_context",
        lambda owner, repo, pr_number: {
            "owner": "example-owner",
            "repo": "sample-web",
            "pr_number": 87,
            "url": "https://github.com/example-owner/example-repo/pull/87",
            "head_sha": "deadbeef",
            "base_ref": "main",
        },
    )
    monkeypatch.setattr(
        review_github,
        "run_review_cycle",
        lambda **kwargs: {
            "status": "acknowledged_without_body",
            "owner": "example-owner",
            "repo": "sample-web",
            "pr_number": 87,
            "url": "https://github.com/example-owner/example-repo/pull/87",
            "head_sha": "deadbeef",
            "request_comment_url": "https://github.com/example-owner/example-repo/pull/87#issuecomment-request-1",
            "items": [],
        },
    )
    monkeypatch.setattr(
        review_github,
        "record_review_anchor",
        lambda **kwargs: recorded.append(kwargs) or {},
    )
    monkeypatch.setattr(
        review_github, "emit_toon", lambda payload: emitted.append(payload)
    )

    exit_code = cmd_run(
        review_github.build_parser().parse_args(
            ["run", "--cd", str(tmp_path), "--state-dir", str(custom_state_dir)]
        ),
    )

    assert exit_code == 0
    assert emitted
    assert recorded == []


def test_cmd_run_does_not_record_workflow_anchor_for_existing_completed_cycle(
    monkeypatch, tmp_path: Path
) -> None:
    emitted: list[dict[str, object]] = []
    recorded: list[dict[str, object]] = []

    monkeypatch.setattr(review_github, "resolve_repo_root", lambda cd: tmp_path)
    monkeypatch.setattr(
        review_github,
        "get_pr_context",
        lambda owner, repo, pr_number: {
            "owner": "example-owner",
            "repo": "sample-web",
            "pr_number": 87,
            "url": "https://github.com/example-owner/example-repo/pull/87",
            "head_sha": "deadbeef",
            "base_ref": "main",
        },
    )
    monkeypatch.setattr(
        review_github,
        "run_review_cycle",
        lambda **kwargs: {
            "status": "existing_completed_cycle",
            "owner": "example-owner",
            "repo": "sample-web",
            "pr_number": 87,
            "url": "https://github.com/example-owner/example-repo/pull/87",
            "head_sha": "deadbeef",
            "items": [
                {
                    "kind": "issue_comment",
                    "url": "https://github.com/example-owner/example-repo/pull/87#issuecomment-1",
                    "body": "No findings.",
                    "created_at": "2026-04-20T10:01:00Z",
                }
            ],
        },
    )
    monkeypatch.setattr(
        review_github,
        "record_review_anchor",
        lambda **kwargs: recorded.append(kwargs) or {},
    )
    monkeypatch.setattr(
        review_github, "emit_toon", lambda payload: emitted.append(payload)
    )

    exit_code = cmd_run(
        review_github.build_parser().parse_args(["run", "--cd", str(tmp_path)]),
    )

    assert exit_code == 0
    assert emitted
    assert recorded == []


def test_cmd_run_skips_workflow_anchor_when_local_checkout_lacks_reviewed_head(
    monkeypatch, tmp_path: Path
) -> None:
    emitted: list[dict[str, object]] = []
    recorded: list[dict[str, object]] = []

    monkeypatch.setattr(review_github, "resolve_repo_root", lambda cd: tmp_path)
    monkeypatch.setattr(
        review_github,
        "local_checkout_contains_reviewed_head",
        lambda review_root, head_sha: False,
    )
    monkeypatch.setattr(
        review_github,
        "get_pr_context",
        lambda owner, repo, pr_number: {
            "owner": "example-owner",
            "repo": "sample-web",
            "pr_number": 87,
            "url": "https://github.com/example-owner/example-repo/pull/87",
            "head_sha": "deadbeef",
            "base_ref": "main",
        },
    )
    monkeypatch.setattr(
        review_github,
        "run_review_cycle",
        lambda **kwargs: {
            "status": "response_found",
            "owner": "example-owner",
            "repo": "sample-web",
            "pr_number": 87,
            "url": "https://github.com/example-owner/example-repo/pull/87",
            "head_sha": "deadbeef",
            "items": [
                {
                    "kind": "issue_comment",
                    "url": "https://github.com/example-owner/example-repo/pull/87#issuecomment-1",
                    "body": "No findings.",
                    "created_at": "2026-04-20T10:01:00Z",
                }
            ],
        },
    )
    monkeypatch.setattr(
        review_github,
        "record_review_anchor",
        lambda **kwargs: recorded.append(kwargs) or {},
    )
    monkeypatch.setattr(
        review_github, "emit_toon", lambda payload: emitted.append(payload)
    )

    exit_code = cmd_run(
        review_github.build_parser().parse_args(["run", "--cd", str(tmp_path)]),
    )

    assert exit_code == 0
    assert emitted
    assert recorded == []


def test_cmd_run_allows_explicit_repo_without_local_checkout(
    monkeypatch, tmp_path: Path
) -> None:
    emitted: list[dict[str, object]] = []
    recorded: list[dict[str, object]] = []

    monkeypatch.setattr(
        review_github,
        "resolve_repo_root",
        lambda cd: (_ for _ in ()).throw(
            AssertionError("should not resolve repo root")
        ),
    )
    monkeypatch.setattr(
        review_github,
        "get_pr_context",
        lambda owner, repo, pr_number: {
            "owner": "example-owner",
            "repo": "sample-web",
            "pr_number": 87,
            "url": "https://github.com/example-owner/example-repo/pull/87",
            "head_sha": "deadbeef",
            "base_ref": "main",
        },
    )
    monkeypatch.setattr(
        review_github,
        "run_review_cycle",
        lambda **kwargs: {
            "status": "response_found",
            "owner": "example-owner",
            "repo": "sample-web",
            "pr_number": 87,
            "url": "https://github.com/example-owner/example-repo/pull/87",
            "head_sha": "deadbeef",
            "items": [],
        },
    )
    monkeypatch.setattr(
        review_github,
        "record_review_anchor",
        lambda **kwargs: recorded.append(kwargs) or {},
    )
    monkeypatch.setattr(
        review_github, "emit_toon", lambda payload: emitted.append(payload)
    )

    exit_code = cmd_run(
        review_github.build_parser().parse_args(
            [
                "run",
                "--owner",
                "example-owner",
                "--repo",
                "sample-web",
                "--pr-number",
                "87",
            ]
        ),
    )

    assert exit_code == 0
    assert emitted
    assert recorded == []


def test_cmd_run_polls_github_every_15_seconds_by_default(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        review_github,
        "get_pr_context",
        lambda owner, repo, pr_number: {
            "owner": "example-owner",
            "repo": "sample-web",
            "pr_number": 87,
            "url": "https://github.com/example-owner/example-repo/pull/87",
            "head_sha": "deadbeef",
            "base_ref": "main",
        },
    )

    def fake_run_review_cycle(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "status": "existing_completed_cycle",
            "owner": "example-owner",
            "repo": "sample-web",
            "pr_number": 87,
            "url": "https://github.com/example-owner/example-repo/pull/87",
            "head_sha": "deadbeef",
            "items": [],
        }

    monkeypatch.setattr(review_github, "run_review_cycle", fake_run_review_cycle)
    monkeypatch.setattr(review_github, "emit_toon", lambda payload: None)

    exit_code = cmd_run(
        review_github.build_parser().parse_args(
            [
                "run",
                "--owner",
                "example-owner",
                "--repo",
                "sample-web",
                "--pr-number",
                "87",
            ]
        ),
    )

    assert exit_code == 0
    assert captured["poll_seconds"] == 15


def test_run_review_cycle_updates_request_url_when_reusing_active_cycle_after_pickup_timeout(
    monkeypatch,
) -> None:
    response = _response_item(
        item_id="issue-1",
        kind="issue_comment",
        body="Codex Review: No findings.",
        created_at="2026-04-20T10:03:00Z",
        url="https://github.com/example-owner/example-repo/pull/87#issuecomment-1",
    )
    context = {
        "owner": "example-owner",
        "repo": "sample-web",
        "pr_number": 87,
        "url": "https://github.com/example-owner/example-repo/pull/87",
        "head_sha": "deadbeef",
    }
    now_calls = {"count": 0}
    inspect_calls = {"count": 0}
    collect_calls = {"count": 0}

    def fake_now() -> datetime:
        now_calls["count"] += 1
        return datetime(2026, 4, 20, 10, 0, 0, tzinfo=timezone.utc) + timedelta(
            seconds=40 * now_calls["count"]
        )

    def fake_inspect_existing_cycle(**kwargs: object) -> dict[str, object] | None:
        inspect_calls["count"] += 1
        if inspect_calls["count"] == 1:
            return None
        return {
            "reason": "existing_request_has_eyes",
            "request_comment_id": "request-2",
            "request_comment_node_id": "IC_request-2",
            "request_comment_url": "https://github.com/example-owner/example-repo/pull/87#issuecomment-request-2",
            "requested_at": "2026-04-20T10:02:00Z",
            "items": [],
        }

    def fake_collect_cycle_items(**kwargs: object) -> list[dict[str, object]]:
        collect_calls["count"] += 1
        return [response] if collect_calls["count"] >= 4 else []

    monkeypatch.setattr("review_github.utc_now", fake_now)
    monkeypatch.setattr("review_github.get_pr_context", lambda *args, **kwargs: context)
    monkeypatch.setattr(
        "review_github.inspect_existing_cycle", fake_inspect_existing_cycle
    )
    monkeypatch.setattr(
        "review_github.post_request",
        lambda **kwargs: {
            "request_comment_id": "request-1",
            "request_comment_node_id": "IC_request-1",
            "request_comment_url": "https://github.com/example-owner/example-repo/pull/87#issuecomment-request-1",
            "requested_at": "2026-04-20T10:00:00Z",
        },
    )
    monkeypatch.setattr(
        "review_github.fetch_pr_snapshot", lambda *args, **kwargs: _snapshot()
    )
    monkeypatch.setattr("review_github.collect_cycle_items", fake_collect_cycle_items)
    monkeypatch.setattr("review_github.time.sleep", lambda seconds: None)

    payload = run_review_cycle(
        owner="example-owner",
        repo="sample-web",
        pr_number=87,
        body="@codex review",
        bot_login="chatgpt-codex-connector[bot]",
        poll_seconds=1,
        timeout_minutes=10,
        status_interval_seconds=60,
        re_request_after_seconds=120,
        max_request_attempts=2,
        settle_seconds=0,
        force=False,
    )

    assert payload["status"] == "response_found"
    assert payload["request_comment_id"] == "request-2"
    assert (
        payload["request_comment_url"]
        == "https://github.com/example-owner/example-repo/pull/87#issuecomment-request-2"
    )


def test_run_review_cycle_uses_status_interval_seconds_for_wait_updates(
    monkeypatch,
) -> None:
    events: list[str] = []
    timeline = iter(
        [
            datetime(2026, 4, 20, 10, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 4, 20, 10, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 4, 20, 10, 0, 3, tzinfo=timezone.utc),
            datetime(2026, 4, 20, 10, 0, 6, tzinfo=timezone.utc),
            datetime(2026, 4, 20, 10, 0, 9, tzinfo=timezone.utc),
            datetime(2026, 4, 20, 10, 1, 1, tzinfo=timezone.utc),
        ]
    )
    context = {
        "owner": "example-owner",
        "repo": "sample-web",
        "pr_number": 87,
        "url": "https://github.com/example-owner/example-repo/pull/87",
        "head_sha": "deadbeef",
    }

    monkeypatch.setattr("review_github.utc_now", lambda: next(timeline))
    monkeypatch.setattr(
        "review_github.emit_state_change", lambda message: events.append(message)
    )
    monkeypatch.setattr("review_github.get_pr_context", lambda *args, **kwargs: context)
    monkeypatch.setattr("review_github.inspect_existing_cycle", lambda **kwargs: None)
    monkeypatch.setattr(
        "review_github.post_request",
        lambda **kwargs: {
            "request_comment_id": "request-1",
            "request_comment_node_id": "IC_request-1",
            "request_comment_url": "https://github.com/example-owner/example-repo/pull/87#issuecomment-request-1",
            "requested_at": "2026-04-20T10:00:00Z",
        },
    )
    monkeypatch.setattr(
        "review_github.fetch_pr_snapshot", lambda *args, **kwargs: _snapshot()
    )
    monkeypatch.setattr("review_github.collect_cycle_items", lambda **kwargs: [])
    monkeypatch.setattr("review_github.time.sleep", lambda seconds: None)

    payload = run_review_cycle(
        owner="example-owner",
        repo="sample-web",
        pr_number=87,
        body="@codex review",
        bot_login="chatgpt-codex-connector[bot]",
        poll_seconds=1,
        timeout_minutes=1,
        status_interval_seconds=5,
        re_request_after_seconds=120,
        max_request_attempts=2,
        settle_seconds=0,
        force=False,
    )

    assert payload["status"] == "timeout"
    assert any(message == "OK 1m: github eyes=no responses=0" for message in events)


# GraphQL nodes recorded from JJLiebig/review-suite#129 (clean review) and #93
# (review with an inline finding), with long bodies trimmed.
RECORDED_HEAD = "b4e7bc37b9b4fb1d0342c7927f92dfa3423b635b"
RECORDED_REQUEST = {
    "id": "IC_kwDOSXR2o88AAAABkLNB-A",
    "databaseId": 6722634232,
    "body": "@codex review",
    "createdAt": "2026-11-08T14:21:04Z",
    "updatedAt": "2026-11-08T14:21:04Z",
    "url": "https://github.com/JJLiebig/review-suite/pull/129#issuecomment-6722634232",
    "author": {"__typename": "User", "login": "JJLiebig"},
}
RECORDED_SUMMARY = {
    "id": "IC_kwDOSXR2o88AAAABkLNPKg",
    "databaseId": 6722637610,
    "body": '<!-- codex-pull-request-review-summary -->\n<!-- codex-review-state:v1 {"code_review":{"status":"completed"}} -->\n\n<details>\n<summary><strong>Codex Review Summary</strong>',
    "createdAt": "2026-11-08T14:21:24Z",
    "updatedAt": "2026-11-08T14:23:20Z",
    "url": "https://github.com/JJLiebig/review-suite/pull/129#issuecomment-6722637610",
    "author": {"__typename": "Bot", "login": "chatgpt-codex-connector"},
}
RECORDED_CLEAN_RECEIPT = {
    "id": "IC_kwDOSXR2o88AAAABkLOgpA",
    "databaseId": 6722658468,
    "body": "<!-- codex-review-receipt:v1 -->\n\nCodex Review: Didn't find any major issues. Nice work!\n\n**Reviewed commit:** `b4e7bc37b9`",
    "createdAt": "2026-11-08T14:23:19Z",
    "updatedAt": "2026-11-08T14:23:19Z",
    "url": "https://github.com/JJLiebig/review-suite/pull/129#issuecomment-6722658468",
    "author": {"__typename": "Bot", "login": "chatgpt-codex-connector"},
}
RECORDED_FINDINGS_REVIEW = {
    "databaseId": 5311840870,
    "body": "\n### 💡 Codex Review\n\nHere are some automated review suggestions for this pull request.\n\n**Reviewed commit:** `b4e7bc37b9`\n",
    "state": "COMMENTED",
    "submittedAt": "2026-11-08T14:27:02Z",
    "url": "https://github.com/JJLiebig/review-suite/pull/93#pullrequestreview-5311840870",
    "commit": {"oid": RECORDED_HEAD},
    "author": {"__typename": "Bot", "login": "chatgpt-codex-connector"},
    "comments": {
        "nodes": [
            {
                "databaseId": 4099832605,
                "body": "**Match the summary marker only in its metadata position**",
                "createdAt": "2026-11-08T14:27:02Z",
                "updatedAt": "2026-11-08T14:27:02Z",
                "path": "plugins/review-suite/scripts/review_github.py",
                "line": None,
                "url": "https://github.com/JJLiebig/review-suite/pull/93#discussion_r4099832605",
                "commit": {"oid": RECORDED_HEAD},
                "author": {"__typename": "Bot", "login": "chatgpt-codex-connector"},
            }
        ]
    },
}
BOT_REACTION = {"nodes": [{"user": {"login": BOT}}]}
NO_REACTIONS: dict[str, list[object]] = {"nodes": []}


def _graphql_snapshot_response(
    *, comments: list[dict[str, object]], reviews=(), eyes=False, plus_one=False
) -> dict[str, object]:
    return {
        "data": {
            "repository": {
                "pullRequest": {
                    "headRefOid": RECORDED_HEAD,
                    "comments": {"nodes": comments},
                    "reviews": {"nodes": list(reviews)},
                }
            },
            "anchor": {
                "eyes": BOT_REACTION if eyes else NO_REACTIONS,
                "plusOne": BOT_REACTION if plus_one else NO_REACTIONS,
            },
        }
    }


@pytest.mark.parametrize(
    ("response", "status", "kinds", "eyes_confirmed"),
    [
        pytest.param(
            _graphql_snapshot_response(
                comments=[RECORDED_REQUEST, RECORDED_SUMMARY, RECORDED_CLEAN_RECEIPT]
            ),
            "response_found",
            ["issue_comment"],
            False,
            id="clean",
        ),
        pytest.param(
            _graphql_snapshot_response(
                comments=[RECORDED_REQUEST, RECORDED_SUMMARY],
                reviews=[RECORDED_FINDINGS_REVIEW],
                eyes=True,
            ),
            "response_found",
            ["review_comment", "review"],
            True,
            id="findings",
        ),
        pytest.param(
            _graphql_snapshot_response(comments=[RECORDED_REQUEST], plus_one=True),
            "acknowledged_without_body",
            [],
            False,
            id="done-without-body",
        ),
        pytest.param(
            _graphql_snapshot_response(
                comments=[RECORDED_REQUEST, RECORDED_SUMMARY], eyes=True
            ),
            "timeout",
            [],
            True,
            id="still-waiting",
        ),
    ],
)
def test_run_review_cycle_decides_from_one_graphql_snapshot_per_poll(
    monkeypatch, response, status, kinds, eyes_confirmed
) -> None:
    clock = {"now": datetime(2026, 11, 8, 14, 21, 5, tzinfo=timezone.utc)}
    requests: list[dict[str, object]] = []
    sleeps: list[float] = []

    def fake_now() -> datetime:
        clock["now"] += timedelta(seconds=20)
        return clock["now"]

    def fake_send(method, url, headers, data):
        requests.append({"method": method, "url": url, **json.loads(data)})
        return 200, {"x-ratelimit-remaining": "4999"}, json.dumps(response).encode()

    monkeypatch.setattr(review_github, "utc_now", fake_now)
    monkeypatch.setattr(review_github, "http_send", fake_send)
    monkeypatch.setattr(review_github.time, "sleep", sleeps.append)
    monkeypatch.setattr(
        review_github,
        "get_pr_context",
        lambda *args: {
            "url": "https://github.com/JJLiebig/review-suite/pull/129",
            "head_sha": RECORDED_HEAD,
        },
    )
    monkeypatch.setattr(review_github, "inspect_existing_cycle", lambda **kwargs: None)
    monkeypatch.setattr(
        review_github,
        "post_request",
        lambda **kwargs: {
            "request_comment_id": str(RECORDED_REQUEST["databaseId"]),
            "request_comment_node_id": RECORDED_REQUEST["id"],
            "request_comment_url": RECORDED_REQUEST["url"],
            "requested_at": RECORDED_REQUEST["createdAt"],
        },
    )

    payload = run_review_cycle(
        owner="JJLiebig",
        repo="review-suite",
        pr_number=129,
        body="@codex review",
        bot_login=BOT,
        poll_seconds=review_github.DEFAULT_POLL_SECONDS,
        timeout_minutes=2,
        status_interval_seconds=60,
        re_request_after_seconds=120,
        max_request_attempts=1,
        settle_seconds=review_github.DEFAULT_SETTLE_SECONDS,
        force=False,
    )

    assert payload["status"] == status
    assert payload["head_sha"] == RECORDED_HEAD
    assert payload["eyes_confirmed"] is eyes_confirmed
    assert [item["kind"] for item in payload["items"]] == kinds
    assert requests and all(
        request["url"] == "https://api.github.com/graphql"
        and request["variables"]["bot"] == BOT
        and request["variables"]["anchor"] == RECORDED_REQUEST["id"]
        for request in requests
    )
    # One request per poll: every poll after the first follows one sleep.
    assert len(requests) in {len(sleeps), len(sleeps) + 1}
    assert set(sleeps) <= {review_github.DEFAULT_POLL_SECONDS}


@pytest.fixture
def clock(monkeypatch) -> dict[str, object]:
    state: dict[str, object] = {"now": 1_800_000_000.0, "sleeps": [], "progress": []}

    def sleep(seconds: float) -> None:
        state["sleeps"].append(seconds)
        state["now"] += seconds

    monkeypatch.setattr(review_github.time, "time", lambda: state["now"])
    monkeypatch.setattr(review_github.time, "sleep", sleep)
    monkeypatch.setattr(review_github, "emit_progress", state["progress"].append)
    return state


def _scripted_send(monkeypatch, responses: list[tuple[int, dict[str, str], object]]):
    sent: list[dict[str, str]] = []

    def fake_send(method, url, headers, data):
        sent.append(headers)
        status, response_headers, body = responses.pop(0)
        return status, response_headers, json.dumps(body).encode()

    monkeypatch.setattr(review_github, "http_send", fake_send)
    return sent


SECONDARY_LIMIT = (
    403,
    {"x-ratelimit-remaining": "4000"},
    {"message": "You have exceeded a secondary rate limit. Please wait a few minutes."},
)


def test_secondary_rate_limit_backs_off_from_30s_doubling_to_a_cap(
    monkeypatch, clock
) -> None:
    _scripted_send(
        monkeypatch,
        [SECONDARY_LIMIT] * 7
        + [(200, {}, {"ok": 1}), SECONDARY_LIMIT, (200, {}, {"ok": 2})],
    )

    assert review_github.github_request("GET", "repos/o/r/pulls/1") == {"ok": 1}
    assert clock["sleeps"] == [30, 60, 120, 240, 480, 900, 900]
    assert clock["progress"][0] == (
        "GitHub rate limit reached; waiting 30s before the next request"
    )
    assert clock["progress"][-1] == (
        "GitHub rate limit reached; waiting 15m before the next request"
    )

    assert review_github.github_request("GET", "repos/o/r/pulls/1") == {"ok": 2}
    assert clock["sleeps"][-1] == 30


@pytest.mark.parametrize(
    ("status", "headers", "body", "expected_wait"),
    [
        pytest.param(
            429, {"retry-after": "7"}, {"message": "slow down"}, 7, id="retry-after"
        ),
        pytest.param(
            403,
            {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1800000120"},
            {"message": "API rate limit exceeded"},
            121,
            id="primary-reset",
        ),
        pytest.param(
            200,
            {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1800000060"},
            {"errors": [{"type": "RATE_LIMITED", "message": "API rate limit exceeded"}]},
            61,
            id="graphql-rate-limited",
        ),
    ],
)
def test_rate_limit_waits_for_what_github_asks(
    monkeypatch, clock, status, headers, body, expected_wait
) -> None:
    _scripted_send(monkeypatch, [(status, headers, body), (200, {}, {"ok": True})])

    assert review_github.github_request("POST", "graphql", body={}) == {"ok": True}
    assert clock["sleeps"] == [expected_wait]


def test_exhausted_quota_waits_for_reset_before_the_next_request(
    monkeypatch, clock
) -> None:
    _scripted_send(
        monkeypatch,
        [
            (
                200,
                {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1800000090"},
                {"n": 1},
            ),
            (200, {"x-ratelimit-remaining": "4999"}, {"n": 2}),
        ],
    )

    assert review_github.github_request("GET", "rate_limit") == {"n": 1}
    assert clock["sleeps"] == []
    assert review_github.github_request("GET", "rate_limit") == {"n": 2}
    assert clock["sleeps"] == [91]


def test_rate_limit_past_the_review_deadline_ends_the_cycle_as_timeout(
    monkeypatch, clock
) -> None:
    _scripted_send(
        monkeypatch, [(429, {"retry-after": "3600"}, {"message": "slow down"})]
    )
    events: list[str] = []
    monkeypatch.setattr(review_github, "emit_state_change", events.append)
    monkeypatch.setattr(
        review_github,
        "utc_now",
        lambda: datetime.fromtimestamp(clock["now"], timezone.utc),
    )
    monkeypatch.setattr(
        review_github,
        "get_pr_context",
        lambda *args: {"url": "https://github.com/o/r/pull/1", "head_sha": "deadbeef"},
    )
    monkeypatch.setattr(review_github, "inspect_existing_cycle", lambda **kwargs: None)
    monkeypatch.setattr(
        review_github,
        "post_request",
        lambda **kwargs: {
            "request_comment_id": "1",
            "request_comment_node_id": "IC_1",
            "request_comment_url": "https://github.com/o/r/pull/1#issuecomment-1",
            "requested_at": "2027-01-15T08:00:00Z",
        },
    )

    payload = run_review_cycle(
        owner="o",
        repo="r",
        pr_number=1,
        body="@codex review",
        bot_login=BOT,
        poll_seconds=15,
        timeout_minutes=30,
        status_interval_seconds=60,
        re_request_after_seconds=120,
        max_request_attempts=1,
        settle_seconds=20,
        force=False,
    )

    assert payload["status"] == "timeout"
    assert payload["head_sha"] == "deadbeef"
    assert clock["sleeps"] == []
    assert any("lasts past the review timeout" in event for event in events)


def test_github_request_rereads_the_token_once_on_401_and_never_reports_it(
    monkeypatch, clock
) -> None:
    tokens = iter(["fresh-token"])
    monkeypatch.setattr(review_github, "read_gh_token", lambda: next(tokens))
    sent = _scripted_send(
        monkeypatch,
        [
            (401, {}, {"message": "Bad credentials"}),
            (200, {}, {"ok": True}),
            (403, {}, {"message": "Resource not accessible by integration"}),
        ],
    )

    assert review_github.github_request("GET", "repos/o/r/pulls/1") == {"ok": True}
    assert [headers["Authorization"] for headers in sent] == [
        "Bearer test-token",
        "Bearer fresh-token",
    ]
    with pytest.raises(RuntimeError) as error:
        review_github.github_request("GET", "repos/o/r/pulls/1")
    assert "HTTP 403" in str(error.value)
    assert "fresh-token" not in str(error.value)
    assert clock["sleeps"] == []
