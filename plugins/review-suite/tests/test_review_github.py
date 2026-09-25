from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


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


def _request_comment(
    *,
    comment_id: str = "request-1",
    created_at: str = "2026-04-20T10:00:00Z",
    head_sha: str = "deadbeef",
) -> dict[str, object]:
    return {
        "id": comment_id,
        "body": review_github.build_request_body("@codex review", head_sha),
        "user": {"login": "alice"},
        "created_at": created_at,
        "html_url": "https://github.com/example-owner/example-repo/pull/87#issuecomment-request-1",
    }


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
    monkeypatch.setattr("review_github.get_issue_comments", lambda *args: comments)
    monkeypatch.setattr("review_github.get_review_comments", lambda *args: [])
    monkeypatch.setattr("review_github.get_reviews", lambda *args: reviews)
    kwargs = {
        "owner": "example-owner",
        "repo": "sample-web",
        "pr_number": 87,
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

    monkeypatch.setattr(
        "review_github.get_issue_comments", lambda *args, **kwargs: [_request_comment()]
    )
    monkeypatch.setattr(
        "review_github.collect_cycle_items", lambda **kwargs: [response]
    )
    monkeypatch.setattr(
        "review_github.get_commit_timestamp", lambda *args, **kwargs: None
    )
    monkeypatch.setattr("review_github.has_eyes_reaction", lambda **kwargs: False)
    monkeypatch.setattr("review_github.has_plus_one_reaction", lambda **kwargs: False)

    cycle = inspect_existing_cycle(
        owner="example-owner",
        repo="sample-web",
        pr_number=87,
        body="@codex review",
        bot_login="chatgpt-codex-connector[bot]",
        head_sha="deadbeef",
    )

    assert cycle == {
        "reason": "existing_request_has_response",
        "request_comment_id": "request-1",
        "request_comment_url": "https://github.com/example-owner/example-repo/pull/87#issuecomment-request-1",
        "requested_at": "2026-04-20T10:00:00Z",
        "items": [response],
    }


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

    monkeypatch.setattr(
        "review_github.get_issue_comments", lambda *args, **kwargs: [_request_comment()]
    )
    monkeypatch.setattr(
        "review_github.collect_cycle_items", lambda **kwargs: [response]
    )
    monkeypatch.setattr(
        "review_github.get_commit_timestamp", lambda *args, **kwargs: None
    )
    monkeypatch.setattr("review_github.has_eyes_reaction", lambda **kwargs: True)
    monkeypatch.setattr("review_github.has_plus_one_reaction", lambda **kwargs: False)
    monkeypatch.setattr(
        "review_github.utc_now",
        lambda: datetime(2026, 4, 20, 10, 1, 30, tzinfo=timezone.utc),
    )

    cycle = inspect_existing_cycle(
        owner="example-owner",
        repo="sample-web",
        pr_number=87,
        body="@codex review",
        bot_login="chatgpt-codex-connector[bot]",
        head_sha="deadbeef",
    )

    assert cycle == {
        "reason": "existing_request_has_eyes",
        "request_comment_id": "request-1",
        "request_comment_url": "https://github.com/example-owner/example-repo/pull/87#issuecomment-request-1",
        "requested_at": "2026-04-20T10:00:00Z",
        "items": [],
    }


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

    monkeypatch.setattr(
        "review_github.get_issue_comments", lambda *args, **kwargs: [_request_comment()]
    )
    monkeypatch.setattr(
        "review_github.collect_cycle_items", lambda **kwargs: [response]
    )
    monkeypatch.setattr(
        "review_github.get_commit_timestamp", lambda *args, **kwargs: None
    )
    monkeypatch.setattr("review_github.has_eyes_reaction", lambda **kwargs: True)
    monkeypatch.setattr("review_github.has_plus_one_reaction", lambda **kwargs: False)
    monkeypatch.setattr(
        "review_github.utc_now",
        lambda: datetime(2026, 4, 20, 10, 1, 30, tzinfo=timezone.utc),
    )

    cycle = inspect_existing_cycle(
        owner="example-owner",
        repo="sample-web",
        pr_number=87,
        body="@codex review",
        bot_login="chatgpt-codex-connector[bot]",
        head_sha="deadbeef",
    )

    assert cycle == {
        "reason": "existing_request_has_response",
        "request_comment_id": "request-1",
        "request_comment_url": "https://github.com/example-owner/example-repo/pull/87#issuecomment-request-1",
        "requested_at": "2026-04-20T10:00:00Z",
        "items": [response],
    }


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

    monkeypatch.setattr(
        "review_github.get_issue_comments", lambda *args, **kwargs: [_request_comment()]
    )
    monkeypatch.setattr(
        "review_github.collect_cycle_items", lambda **kwargs: [response]
    )
    monkeypatch.setattr(
        "review_github.get_commit_timestamp",
        lambda *args, **kwargs: datetime(2026, 4, 20, 10, 1, 20, tzinfo=timezone.utc),
    )
    monkeypatch.setattr("review_github.has_eyes_reaction", lambda **kwargs: True)
    monkeypatch.setattr("review_github.has_plus_one_reaction", lambda **kwargs: False)
    monkeypatch.setattr(
        "review_github.utc_now",
        lambda: datetime(2026, 4, 20, 10, 1, 30, tzinfo=timezone.utc),
    )

    cycle = inspect_existing_cycle(
        owner="example-owner",
        repo="sample-web",
        pr_number=87,
        body="@codex review",
        bot_login="chatgpt-codex-connector[bot]",
        head_sha="deadbeef",
    )

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

    monkeypatch.setattr(
        "review_github.get_issue_comments", lambda *args, **kwargs: [_request_comment()]
    )
    monkeypatch.setattr(
        "review_github.collect_cycle_items",
        lambda **kwargs: [top_level_response, child_comment],
    )
    monkeypatch.setattr(
        "review_github.get_commit_timestamp",
        lambda *args, **kwargs: datetime(2026, 4, 20, 10, 1, 10, tzinfo=timezone.utc),
    )
    monkeypatch.setattr("review_github.has_eyes_reaction", lambda **kwargs: True)
    monkeypatch.setattr("review_github.has_plus_one_reaction", lambda **kwargs: False)
    monkeypatch.setattr(
        "review_github.utc_now",
        lambda: datetime(2026, 4, 20, 10, 1, 30, tzinfo=timezone.utc),
    )

    cycle = inspect_existing_cycle(
        owner="example-owner",
        repo="sample-web",
        pr_number=87,
        body="@codex review",
        bot_login="chatgpt-codex-connector[bot]",
        head_sha="deadbeef",
    )

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

    monkeypatch.setattr(
        "review_github.get_issue_comments", lambda *args, **kwargs: [_request_comment()]
    )
    monkeypatch.setattr(
        "review_github.collect_cycle_items", lambda **kwargs: [response]
    )
    monkeypatch.setattr(
        "review_github.get_commit_timestamp", lambda *args, **kwargs: None
    )
    monkeypatch.setattr("review_github.has_eyes_reaction", lambda **kwargs: False)
    monkeypatch.setattr("review_github.has_plus_one_reaction", lambda **kwargs: False)

    cycle = inspect_existing_cycle(
        owner="example-owner",
        repo="sample-web",
        pr_number=87,
        body="@codex review",
        bot_login="chatgpt-codex-connector[bot]",
        head_sha="deadbeef",
    )

    assert cycle == {
        "reason": "existing_request_has_response",
        "request_comment_id": "request-1",
        "request_comment_url": "https://github.com/example-owner/example-repo/pull/87#issuecomment-request-1",
        "requested_at": "2026-04-20T10:00:00Z",
        "items": [response],
    }


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

    monkeypatch.setattr(
        "review_github.get_issue_comments", lambda *args, **kwargs: [_request_comment()]
    )
    monkeypatch.setattr(
        "review_github.collect_cycle_items",
        lambda **kwargs: [early_response, final_response],
    )
    monkeypatch.setattr(
        "review_github.get_commit_timestamp",
        lambda *args, **kwargs: datetime(2026, 4, 20, 10, 1, 10, tzinfo=timezone.utc),
    )
    monkeypatch.setattr("review_github.has_eyes_reaction", lambda **kwargs: False)
    monkeypatch.setattr("review_github.has_plus_one_reaction", lambda **kwargs: False)

    cycle = inspect_existing_cycle(
        owner="example-owner",
        repo="sample-web",
        pr_number=87,
        body="@codex review",
        bot_login="chatgpt-codex-connector[bot]",
        head_sha="deadbeef",
    )

    assert cycle == {
        "reason": "existing_request_has_response",
        "request_comment_id": "request-1",
        "request_comment_url": "https://github.com/example-owner/example-repo/pull/87#issuecomment-request-1",
        "requested_at": "2026-04-20T10:00:00Z",
        "items": [early_response, final_response],
    }


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

    monkeypatch.setattr(
        "review_github.get_issue_comments", lambda *args, **kwargs: [_request_comment()]
    )
    monkeypatch.setattr(
        "review_github.collect_cycle_items",
        lambda **kwargs: [early_response, final_response],
    )
    monkeypatch.setattr(
        "review_github.get_commit_timestamp",
        lambda *args, **kwargs: datetime(2026, 4, 20, 10, 1, 10, tzinfo=timezone.utc),
    )
    monkeypatch.setattr("review_github.has_eyes_reaction", lambda **kwargs: False)
    monkeypatch.setattr("review_github.has_plus_one_reaction", lambda **kwargs: False)

    cycle = inspect_existing_cycle(
        owner="example-owner",
        repo="sample-web",
        pr_number=87,
        body="@codex review",
        bot_login="chatgpt-codex-connector[bot]",
        head_sha="deadbeef",
    )

    assert cycle == {
        "reason": "existing_request_has_response",
        "request_comment_id": "request-1",
        "request_comment_url": "https://github.com/example-owner/example-repo/pull/87#issuecomment-request-1",
        "requested_at": "2026-04-20T10:00:00Z",
        "items": [early_response, final_response],
    }


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

    monkeypatch.setattr(
        "review_github.get_issue_comments", lambda *args, **kwargs: [_request_comment()]
    )
    monkeypatch.setattr(
        "review_github.collect_cycle_items",
        lambda **kwargs: [early_response, inline_comment],
    )
    monkeypatch.setattr(
        "review_github.get_commit_timestamp",
        lambda *args, **kwargs: datetime(2026, 4, 20, 10, 1, 10, tzinfo=timezone.utc),
    )
    monkeypatch.setattr("review_github.has_eyes_reaction", lambda **kwargs: False)
    monkeypatch.setattr("review_github.has_plus_one_reaction", lambda **kwargs: False)

    cycle = inspect_existing_cycle(
        owner="example-owner",
        repo="sample-web",
        pr_number=87,
        body="@codex review",
        bot_login="chatgpt-codex-connector[bot]",
        head_sha="deadbeef",
    )

    assert cycle == {
        "reason": "existing_request_has_response",
        "request_comment_id": "request-1",
        "request_comment_url": "https://github.com/example-owner/example-repo/pull/87#issuecomment-request-1",
        "requested_at": "2026-04-20T10:00:00Z",
        "items": [early_response, inline_comment],
    }


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

    monkeypatch.setattr(
        "review_github.get_issue_comments", lambda *args, **kwargs: [_request_comment()]
    )
    monkeypatch.setattr(
        "review_github.collect_cycle_items",
        lambda **kwargs: [early_response, inline_comment, final_response],
    )
    monkeypatch.setattr(
        "review_github.get_commit_timestamp",
        lambda *args, **kwargs: datetime(2026, 4, 20, 10, 1, 18, tzinfo=timezone.utc),
    )
    monkeypatch.setattr("review_github.has_eyes_reaction", lambda **kwargs: False)
    monkeypatch.setattr("review_github.has_plus_one_reaction", lambda **kwargs: False)

    cycle = inspect_existing_cycle(
        owner="example-owner",
        repo="sample-web",
        pr_number=87,
        body="@codex review",
        bot_login="chatgpt-codex-connector[bot]",
        head_sha="deadbeef",
    )

    assert cycle == {
        "reason": "existing_request_has_response",
        "request_comment_id": "request-1",
        "request_comment_url": "https://github.com/example-owner/example-repo/pull/87#issuecomment-request-1",
        "requested_at": "2026-04-20T10:00:00Z",
        "items": [early_response, inline_comment, final_response],
    }


def test_inspect_existing_cycle_does_not_reuse_acknowledged_without_body_when_head_moved(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "review_github.get_issue_comments", lambda *args, **kwargs: [_request_comment()]
    )
    monkeypatch.setattr("review_github.collect_cycle_items", lambda **kwargs: [])
    monkeypatch.setattr(
        "review_github.get_commit_timestamp",
        lambda *args, **kwargs: datetime(2026, 4, 20, 10, 1, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr("review_github.has_eyes_reaction", lambda **kwargs: False)
    monkeypatch.setattr("review_github.has_plus_one_reaction", lambda **kwargs: True)

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
    monkeypatch.setattr(
        "review_github.get_issue_comments", lambda *args, **kwargs: [_request_comment()]
    )
    monkeypatch.setattr("review_github.collect_cycle_items", lambda **kwargs: [])
    monkeypatch.setattr(
        "review_github.get_commit_timestamp",
        lambda *args, **kwargs: datetime(2026, 4, 20, 10, 1, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr("review_github.has_eyes_reaction", lambda **kwargs: True)
    monkeypatch.setattr("review_github.has_plus_one_reaction", lambda **kwargs: False)

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
    monkeypatch.setattr(
        "review_github.get_issue_comments", lambda *args, **kwargs: [_request_comment()]
    )
    monkeypatch.setattr("review_github.collect_cycle_items", lambda **kwargs: [])
    monkeypatch.setattr(
        "review_github.get_commit_timestamp",
        lambda *args, **kwargs: datetime(2026, 4, 20, 10, 1, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr("review_github.has_eyes_reaction", lambda **kwargs: False)
    monkeypatch.setattr("review_github.has_plus_one_reaction", lambda **kwargs: False)

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


def test_cmd_run_uses_fast_github_review_polling_default(monkeypatch) -> None:
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
    assert captured["poll_seconds"] == 3


def test_post_request_includes_repo_selector(monkeypatch) -> None:
    gh_calls: list[list[str]] = []
    request_comment = _request_comment(head_sha="deadbeef")

    monkeypatch.setattr(
        review_github, "run_gh", lambda args: gh_calls.append(list(args)) or ""
    )
    monkeypatch.setattr(
        review_github, "get_issue_comments", lambda *args, **kwargs: [request_comment]
    )
    monkeypatch.setattr(
        review_github,
        "utc_now",
        lambda: datetime(2026, 4, 20, 10, 0, 0, tzinfo=timezone.utc),
    )

    payload = review_github.post_request(
        owner="example-owner",
        repo="sample-web",
        pr_number=87,
        body="@codex review",
        head_sha="deadbeef",
        bot_login="chatgpt-codex-connector[bot]",
    )

    assert gh_calls == [
        [
            "pr",
            "comment",
            "87",
            "--repo",
            "example-owner/sample-web",
            "--body",
            "@codex review",
        ]
    ]
    assert payload["request_comment_id"] == "request-1"


def test_run_review_cycle_returns_response_found_for_top_level_message(
    monkeypatch,
) -> None:
    response = _response_item(
        item_id="issue-1",
        kind="issue_comment",
        body="Codex Review: Didn't find any major issues. Already looking forward to the next diff.",
        created_at="2026-04-20T10:01:00Z",
        url="https://github.com/example-owner/example-repo/pull/87#issuecomment-1",
    )
    context = {
        "owner": "example-owner",
        "repo": "sample-web",
        "pr_number": 87,
        "url": "https://github.com/example-owner/example-repo/pull/87",
        "head_sha": "deadbeef",
    }

    monkeypatch.setattr("review_github.get_pr_context", lambda *args, **kwargs: context)
    monkeypatch.setattr("review_github.inspect_existing_cycle", lambda **kwargs: None)
    monkeypatch.setattr(
        "review_github.post_request",
        lambda **kwargs: {
            "request_comment_id": "request-1",
            "request_comment_url": "https://github.com/example-owner/example-repo/pull/87#issuecomment-request-1",
            "requested_at": "2026-04-20T10:00:00Z",
        },
    )
    monkeypatch.setattr(
        "review_github.get_pr_head_sha", lambda *args, **kwargs: "deadbeef"
    )
    monkeypatch.setattr("review_github.has_eyes_reaction", lambda **kwargs: False)
    monkeypatch.setattr(
        "review_github.collect_cycle_items", lambda **kwargs: [response]
    )
    monkeypatch.setattr("review_github.has_plus_one_reaction", lambda **kwargs: False)

    payload = run_review_cycle(
        owner="example-owner",
        repo="sample-web",
        pr_number=87,
        body="@codex review",
        bot_login="chatgpt-codex-connector[bot]",
        poll_seconds=1,
        timeout_minutes=1,
        status_interval_seconds=60,
        re_request_after_seconds=120,
        max_request_attempts=2,
        settle_seconds=0,
        force=False,
    )

    assert payload["status"] == "response_found"
    assert payload["items"] == [response]


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
            "request_comment_url": "https://github.com/example-owner/example-repo/pull/87#issuecomment-request-1",
            "requested_at": "2026-04-20T10:00:00Z",
        },
    )
    monkeypatch.setattr(
        "review_github.get_pr_head_sha", lambda *args, **kwargs: "deadbeef"
    )
    monkeypatch.setattr("review_github.has_eyes_reaction", lambda **kwargs: False)
    monkeypatch.setattr("review_github.collect_cycle_items", fake_collect_cycle_items)
    monkeypatch.setattr("review_github.has_plus_one_reaction", lambda **kwargs: False)
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
            "request_comment_url": "https://github.com/example-owner/example-repo/pull/87#issuecomment-request-1",
            "requested_at": "2026-04-20T10:00:00Z",
        },
    )
    monkeypatch.setattr(
        "review_github.get_pr_head_sha", lambda *args, **kwargs: "deadbeef"
    )
    monkeypatch.setattr("review_github.has_eyes_reaction", lambda **kwargs: False)
    monkeypatch.setattr("review_github.collect_cycle_items", lambda **kwargs: [])
    monkeypatch.setattr("review_github.has_plus_one_reaction", lambda **kwargs: False)
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
