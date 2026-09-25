#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_previous_dont_write_bytecode = sys.dont_write_bytecode
sys.dont_write_bytecode = True

from review_suite_runtime_bootstrap import (
    bootstrap_from_installed_cache,
    launcher_script_path,
)

bootstrap_from_installed_cache(__file__)
sys.dont_write_bytecode = _previous_dont_write_bytecode

from review_suite_core import (
    AxiArgumentParser,
    current_head,
    emit_error,
    emit_toon,
    format_command,
    is_ancestor,
    merge_base,
    record_review_anchor,
    resolve_ref,
    resolve_repo_root,
)
from review_suite_local import default_state_dir

DEFAULT_BOT_LOGIN = "chatgpt-codex-connector[bot]"
DEFAULT_REQUEST_BODY = "@codex review"
REVIEW_SUMMARY_MARKER = "<!-- codex-pull-request-review-summary -->"
EXISTING_RESPONSE_SETTLE_SECONDS = 20
DEFAULT_POLL_SECONDS = 3
DEFAULT_TIMEOUT_MINUTES = 30
DEFAULT_STATUS_INTERVAL_SECONDS = 60
DEFAULT_RE_REQUEST_AFTER_SECONDS = 120
DEFAULT_MAX_REQUEST_ATTEMPTS = 2
DEFAULT_SETTLE_SECONDS = 20
GH_CWD: str | None = None
REVIEWED_COMMIT_RE = re.compile(
    r"\bReviewed commit:\s*`?([0-9a-f]{7,40})`?", re.IGNORECASE
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat().replace("+00:00", "Z")


def parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def gh_executable() -> str:
    return shutil.which("gh") or shutil.which("gh.exe") or "gh"


def run_gh(args: list[str]) -> str:
    proc = subprocess.run(
        [gh_executable(), *args],
        capture_output=True,
        cwd=GH_CWD,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        message = (
            proc.stderr or proc.stdout or ""
        ).strip() or f"gh exited with code {proc.returncode}"
        raise RuntimeError(f"gh {' '.join(args)} failed: {message}")
    return proc.stdout


def run_gh_json(args: list[str]) -> Any:
    output = run_gh(args).strip()
    if not output:
        return None
    return json.loads(output)


def emit_progress(message: str) -> None:
    print(f"[review-github] {message}", file=sys.stderr, flush=True)


def emit_state_change(message: str) -> None:
    emit_progress(message)


def get_pr_context(
    owner: str | None, repo: str | None, pr_number: int | None
) -> dict[str, Any]:
    if owner and repo and pr_number:
        pr = run_gh_json(["api", f"repos/{owner}/{repo}/pulls/{pr_number}"])
        return {
            "owner": owner,
            "repo": repo,
            "pr_number": pr_number,
            "url": pr["html_url"],
            "head_sha": pr["head"]["sha"],
            "base_ref": pr["base"]["ref"],
        }
    pr = run_gh_json(
        [
            "pr",
            "view",
            "--json",
            "number,url,headRefOid,headRepositoryOwner,headRepository,baseRefName",
        ]
    )
    return {
        "owner": pr["headRepositoryOwner"]["login"],
        "repo": pr["headRepository"]["name"],
        "pr_number": int(pr["number"]),
        "url": pr["url"],
        "head_sha": pr["headRefOid"],
        "base_ref": pr["baseRefName"],
    }


def get_pr_head_sha(owner: str, repo: str, pr_number: int) -> str:
    pr = run_gh_json(["api", f"repos/{owner}/{repo}/pulls/{pr_number}"])
    return str(pr["head"]["sha"])


def get_commit_timestamp(owner: str, repo: str, commit_sha: str) -> datetime | None:
    if not commit_sha:
        return None
    payload = run_gh_json(["api", f"repos/{owner}/{repo}/commits/{commit_sha}"])
    raw = (((payload or {}).get("commit") or {}).get("committer") or {}).get(
        "date"
    ) or (((payload or {}).get("commit") or {}).get("author") or {}).get("date")
    return parse_iso(str(raw)) if raw else None


def get_issue_comments(owner: str, repo: str, pr_number: int) -> list[dict[str, Any]]:
    return list(
        run_gh_json(
            ["api", f"repos/{owner}/{repo}/issues/{pr_number}/comments?per_page=100"]
        )
        or []
    )


def get_review_comments(owner: str, repo: str, pr_number: int) -> list[dict[str, Any]]:
    return list(
        run_gh_json(
            ["api", f"repos/{owner}/{repo}/pulls/{pr_number}/comments?per_page=100"]
        )
        or []
    )


def get_reviews(owner: str, repo: str, pr_number: int) -> list[dict[str, Any]]:
    return list(
        run_gh_json(
            ["api", f"repos/{owner}/{repo}/pulls/{pr_number}/reviews?per_page=100"]
        )
        or []
    )


def get_comment_reactions(
    owner: str, repo: str, comment_id: str
) -> list[dict[str, Any]]:
    return list(
        run_gh_json(
            [
                "api",
                "-H",
                "Accept: application/vnd.github+json",
                f"repos/{owner}/{repo}/issues/comments/{comment_id}/reactions?per_page=100",
            ]
        )
        or []
    )


def item_timestamp(item: dict[str, Any]) -> datetime | None:
    raw = item.get("created_at") or item.get("submitted_at") or item.get("updated_at")
    return parse_iso(raw) if raw else None


def response_activity_timestamp(item: dict[str, Any]) -> datetime | None:
    raw = item.get("updated_at") or item.get("created_at")
    return parse_iso(str(raw)) if raw else None


def response_creation_timestamp(item: dict[str, Any]) -> datetime | None:
    raw = item.get("created_at") or item.get("updated_at")
    return parse_iso(str(raw)) if raw else None


def item_url(item: dict[str, Any]) -> str:
    return str(item.get("html_url") or item.get("url") or "")


def normalize_request_body(value: str) -> str:
    return " ".join(value.strip().split()).lower()


def build_request_body(body_prefix: str, head_sha: str) -> str:
    return str(body_prefix or "").strip()


def request_body_matches(
    *, request_body: str, expected_prefix: str, head_sha: str
) -> bool:
    normalized_expected = normalize_request_body(expected_prefix)
    normalized_request = normalize_request_body(request_body)
    return normalized_request == normalized_expected


def latest_review_request(
    issue_comments: list[dict[str, Any]],
    *,
    bot_login: str,
    since: datetime,
    body_prefix: str,
    head_sha: str,
) -> dict[str, Any] | None:
    matching = []
    for item in issue_comments:
        body = str(item.get("body") or "")
        author = str((item.get("user") or {}).get("login") or "")
        created_at = item_timestamp(item)
        if not created_at:
            continue
        if author == bot_login:
            continue
        if not request_body_matches(
            request_body=body, expected_prefix=body_prefix, head_sha=head_sha
        ):
            continue
        if created_at < since:
            continue
        matching.append(item)
    if not matching:
        return None
    matching.sort(key=lambda item: item_timestamp(item) or since)
    return matching[-1]


def commit_ids_match(left: str, right: str) -> bool:
    left = left.strip().lower()
    right = right.strip().lower()
    if not left or not right:
        return True
    return left.startswith(right) or right.startswith(left)


def response_item_commit_id(item: dict[str, Any], body: str) -> str:
    commit_id = str(item.get("commit_id") or "").strip()
    if commit_id:
        return commit_id
    match = REVIEWED_COMMIT_RE.search(body)
    return match.group(1) if match else ""


def emit_new_items(
    *,
    kind: str,
    items: list[dict[str, Any]],
    head_sha: str,
    anchor_since: datetime,
    seen: set[str],
) -> list[dict[str, Any]]:
    emitted: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda value: item_timestamp(value) or anchor_since):
        ts = item_timestamp(item)
        if not ts or ts < anchor_since:
            continue
        body = str(item.get("body") or "")
        item_head = response_item_commit_id(item, body)
        if item_head and head_sha and not commit_ids_match(item_head, head_sha):
            continue
        item_id = str(item.get("id") or "")
        if not item_id:
            continue
        key = f"{kind}|{item_id}"
        if key in seen:
            continue
        seen.add(key)
        emitted.append(
            {
                "kind": kind,
                "id": item_id,
                "created_at": ts.isoformat().replace("+00:00", "Z"),
                "updated_at": item.get("updated_at"),
                "commit_id": item_head or None,
                "path": item.get("path"),
                "line": item.get("line"),
                "url": item_url(item),
                "body": body,
                "state": item.get("state"),
            }
        )
    return emitted


def post_request(
    *,
    owner: str,
    repo: str,
    pr_number: int,
    body: str,
    head_sha: str,
    bot_login: str,
    anchor_window_seconds: int = 5,
) -> dict[str, Any]:
    started_at = utc_now()
    request_body = build_request_body(body, head_sha)
    run_gh(
        [
            "pr",
            "comment",
            str(pr_number),
            "--repo",
            f"{owner}/{repo}",
            "--body",
            request_body,
        ]
    )
    issue_comments = get_issue_comments(owner, repo, pr_number)
    anchor = latest_review_request(
        issue_comments,
        bot_login=bot_login,
        since=started_at - timedelta(seconds=anchor_window_seconds),
        body_prefix=body,
        head_sha=head_sha,
    )
    if not anchor:
        raise RuntimeError(
            "posted @codex review but could not resolve the matching request comment"
        )
    requested_at = item_timestamp(anchor) or utc_now()
    return {
        "request_comment_id": str(anchor["id"]),
        "request_comment_url": item_url(anchor),
        "requested_at": requested_at.isoformat().replace("+00:00", "Z"),
    }


def has_eyes_reaction(
    *, owner: str, repo: str, comment_id: str, bot_login: str
) -> bool:
    for reaction in get_comment_reactions(owner, repo, comment_id):
        if str(reaction.get("content") or "") != "eyes":
            continue
        if str((reaction.get("user") or {}).get("login") or "") != bot_login:
            continue
        return True
    return False


def has_plus_one_reaction(
    *, owner: str, repo: str, comment_id: str, bot_login: str
) -> bool:
    for reaction in get_comment_reactions(owner, repo, comment_id):
        if str(reaction.get("content") or "") != "+1":
            continue
        if str((reaction.get("user") or {}).get("login") or "") != bot_login:
            continue
        return True
    return False


def collect_cycle_items(
    *,
    owner: str,
    repo: str,
    pr_number: int,
    bot_login: str,
    head_sha: str,
    anchor_since: datetime,
    seen: set[str] | None = None,
) -> list[dict[str, Any]]:
    if seen is None:
        seen = set()
    issue_comments = get_issue_comments(owner, repo, pr_number)
    bot_issue_comments = [
        item
        for item in issue_comments
        if str((item.get("user") or {}).get("login") or "") == bot_login
        and not str(item.get("body") or "").startswith(REVIEW_SUMMARY_MARKER)
    ]
    review_comments = [
        item
        for item in get_review_comments(owner, repo, pr_number)
        if str((item.get("user") or {}).get("login") or "") == bot_login
    ]
    reviews = [
        item
        for item in get_reviews(owner, repo, pr_number)
        if str((item.get("user") or {}).get("login") or "") == bot_login
    ]
    items: list[dict[str, Any]] = []
    items.extend(
        emit_new_items(
            kind="issue_comment",
            items=bot_issue_comments,
            head_sha=head_sha,
            anchor_since=anchor_since,
            seen=seen,
        )
    )
    items.extend(
        emit_new_items(
            kind="review_comment",
            items=review_comments,
            head_sha=head_sha,
            anchor_since=anchor_since,
            seen=seen,
        )
    )
    items.extend(
        emit_new_items(
            kind="review",
            items=reviews,
            head_sha=head_sha,
            anchor_since=anchor_since,
            seen=seen,
        )
    )
    return items


def compact_location(item: dict[str, Any]) -> str | None:
    path = str(item.get("path") or "").strip()
    line = item.get("line")
    if path and line:
        return f"{path}:{line}"
    if path:
        return path
    return None


def _record_anchor_warning(exc: Exception) -> None:
    print(
        f"[review-suite] WARNING: failed to record workflow anchor: {exc}",
        file=sys.stderr,
        flush=True,
    )


def sorted_cycle_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        items,
        key=lambda item: (item_timestamp(item) or utc_now(), str(item.get("id") or "")),
    )


def cycle_coverage_timestamp(
    items: list[dict[str, Any]], *, head_sha: str
) -> datetime | None:
    explicit_head_matches = [
        response_creation_timestamp(item)
        for item in sorted_cycle_items(items)
        if head_sha and str(item.get("commit_id") or "").strip() == head_sha
    ]
    explicit_head_matches = [
        timestamp for timestamp in explicit_head_matches if timestamp is not None
    ]
    top_level_items = [
        item
        for item in sorted_cycle_items(items)
        if str(item.get("kind") or "") in {"issue_comment", "review"}
        and str(item.get("body") or "").strip()
    ]
    top_level = [response_creation_timestamp(item) for item in top_level_items]
    top_level = [timestamp for timestamp in top_level if timestamp is not None]
    if explicit_head_matches:
        latest_explicit_head_match = explicit_head_matches[-1]
        latest_top_level = top_level[-1] if top_level else None
        if (
            latest_top_level is not None
            and latest_top_level > latest_explicit_head_match
        ):
            return latest_top_level
        return latest_explicit_head_match
    if top_level:
        return top_level[-1]
    fallback = [response_creation_timestamp(item) for item in sorted_cycle_items(items)]
    fallback = [timestamp for timestamp in fallback if timestamp is not None]
    return fallback[-1] if fallback else None


def public_response_item(item: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "kind": str(item.get("kind") or ""),
        "url": item_url(item),
        "body": str(item.get("body") or ""),
    }
    created_at = str(item.get("created_at") or "").strip()
    if created_at:
        result["created_at"] = created_at
    location = compact_location(item)
    if location:
        result["loc"] = location
    state = str(item.get("state") or "").strip()
    if state:
        result["state"] = state
    commit_id = str(item.get("commit_id") or "").strip()
    if commit_id:
        result["commit"] = commit_id
    return result


def public_cycle_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": payload["status"],
        "pr": payload["pr_number"],
        "url": payload["url"],
        "head": payload["head_sha"],
    }
    items = list(payload.get("items") or [])
    if payload["status"] == "timeout":
        result["error"] = (
            f"review cycle timed out after {payload.get('timeout_minutes')} minutes"
        )
    note = str(payload.get("note") or "").strip()
    if note:
        result["note"] = note
    completion_signal = str(payload.get("completion_signal") or "").strip()
    if completion_signal:
        result["completion_signal"] = completion_signal
    if items:
        ordered_items = sorted_cycle_items(items)
        result["responses"] = [public_response_item(item) for item in ordered_items]
    attempts = payload.get("request_attempts")
    if attempts not in (None, 0, 1):
        result["attempts"] = attempts
    return result


def cycle_has_review_body(payload: dict[str, Any]) -> bool:
    return any(
        str(item.get("body") or "").strip() for item in list(payload.get("items") or [])
    )


def local_checkout_contains_reviewed_head(review_root: Path, head_sha: str) -> bool:
    normalized_head = str(head_sha or "").strip()
    if not normalized_head:
        return False
    try:
        resolved_head = resolve_ref(review_root, normalized_head)
        local_head = current_head(review_root)
        return resolved_head == local_head or is_ancestor(
            review_root, resolved_head, local_head
        )
    except ValueError:
        return False


def github_review_scope(
    *, review_root: Path, base_ref: str | None, head_sha: str
) -> dict[str, Any]:
    scope: dict[str, Any] = {"reviewed_head": head_sha}
    normalized_base = str(base_ref or "").strip()
    if not normalized_base:
        return scope
    scope["base"] = normalized_base
    try:
        scope["merge_base"] = merge_base(review_root, normalized_base)
    except Exception:
        pass
    return scope


def _help_command() -> str:
    return format_command(
        [sys.executable, str(launcher_script_path(__file__)), "run", "--help"]
    )


def inspect_existing_cycle(
    *,
    owner: str,
    repo: str,
    pr_number: int,
    body: str,
    bot_login: str,
    head_sha: str,
) -> dict[str, Any] | None:
    issue_comments = get_issue_comments(owner, repo, pr_number)
    requests = []
    for item in issue_comments:
        author = str((item.get("user") or {}).get("login") or "")
        created_at = item_timestamp(item)
        if author == bot_login or not created_at:
            continue
        if not request_body_matches(
            request_body=str(item.get("body") or ""),
            expected_prefix=body,
            head_sha=head_sha,
        ):
            continue
        requests.append(item)
    if not requests:
        return None
    requests.sort(key=lambda item: item_timestamp(item) or utc_now(), reverse=True)
    request = requests[0]
    anchor_since = item_timestamp(request) or utc_now()
    anchor_id = str(request["id"])
    head_commit_at = get_commit_timestamp(owner, repo, head_sha)
    items = collect_cycle_items(
        owner=owner,
        repo=repo,
        pr_number=pr_number,
        bot_login=bot_login,
        head_sha=head_sha,
        anchor_since=anchor_since,
    )
    if items:
        response_coverage_at = (
            cycle_coverage_timestamp(items, head_sha=head_sha) or anchor_since
        )
        if head_commit_at and head_commit_at > response_coverage_at:
            return None
        if has_eyes_reaction(
            owner=owner, repo=repo, comment_id=anchor_id, bot_login=bot_login
        ):
            has_child_comments = any(
                str(item.get("kind") or "") == "review_comment" for item in items
            )
            latest_response_activity_at = max(
                (response_activity_timestamp(item) or anchor_since for item in items),
                default=anchor_since,
            )
            response_age_seconds = max(
                int((utc_now() - latest_response_activity_at).total_seconds()), 0
            )
            if (
                not has_child_comments
                and response_age_seconds < EXISTING_RESPONSE_SETTLE_SECONDS
            ):
                return {
                    "reason": "existing_request_has_eyes",
                    "request_comment_id": anchor_id,
                    "request_comment_url": item_url(request),
                    "requested_at": anchor_since.isoformat().replace("+00:00", "Z"),
                    "items": [],
                }
        return {
            "reason": "existing_request_has_response",
            "request_comment_id": anchor_id,
            "request_comment_url": item_url(request),
            "requested_at": anchor_since.isoformat().replace("+00:00", "Z"),
            "items": items,
        }
    if has_eyes_reaction(
        owner=owner, repo=repo, comment_id=anchor_id, bot_login=bot_login
    ):
        if head_commit_at and head_commit_at > anchor_since:
            return None
        return {
            "reason": "existing_request_has_eyes",
            "request_comment_id": anchor_id,
            "request_comment_url": item_url(request),
            "requested_at": anchor_since.isoformat().replace("+00:00", "Z"),
            "items": [],
        }
    if has_plus_one_reaction(
        owner=owner, repo=repo, comment_id=anchor_id, bot_login=bot_login
    ):
        if head_commit_at and head_commit_at > anchor_since:
            return None
        return {
            "reason": "existing_request_acknowledged_without_body",
            "request_comment_id": anchor_id,
            "request_comment_url": item_url(request),
            "requested_at": anchor_since.isoformat().replace("+00:00", "Z"),
            "items": [],
        }
    if head_commit_at and head_commit_at > anchor_since:
        return None
    return {
        "reason": "existing_request_still_active",
        "request_comment_id": anchor_id,
        "request_comment_url": item_url(request),
        "requested_at": anchor_since.isoformat().replace("+00:00", "Z"),
        "items": [],
    }


def run_review_cycle(
    *,
    owner: str,
    repo: str,
    pr_number: int,
    body: str,
    bot_login: str,
    poll_seconds: int,
    timeout_minutes: int,
    status_interval_seconds: int,
    re_request_after_seconds: int,
    max_request_attempts: int,
    settle_seconds: int,
    force: bool,
) -> dict[str, Any]:
    context = get_pr_context(owner, repo, pr_number)
    emit_state_change(
        "GitHub review can take up to 30m. Wait for wrapper output before acting."
    )
    deadline = utc_now() + timedelta(minutes=timeout_minutes)
    seen: set[str] = set()
    request_attempts = 0
    eyes_confirmed = False
    collected_items: list[dict[str, Any]] = []
    last_new_item_at: datetime | None = None
    last_status_emit_at: datetime | None = None
    request_comment_url = ""

    existing_cycle = inspect_existing_cycle(
        owner=owner,
        repo=repo,
        pr_number=pr_number,
        body=body,
        bot_login=bot_login,
        head_sha=context["head_sha"],
    )
    if existing_cycle:
        anchor_id = existing_cycle["request_comment_id"]
        request_comment_url = str(existing_cycle.get("request_comment_url") or "")
        anchor_since = parse_iso(existing_cycle["requested_at"])
        if existing_cycle["reason"] == "existing_request_has_eyes":
            eyes_confirmed = True
            emit_state_change("review in progress")
        elif existing_cycle["reason"] == "existing_request_acknowledged_without_body":
            emit_state_change("reusing existing completed cycle")
            return {
                "status": "existing_completed_cycle",
                "owner": owner,
                "repo": repo,
                "pr_number": pr_number,
                "url": context["url"],
                "head_sha": context["head_sha"],
                "request_attempts": 0,
                "eyes_confirmed": False,
                "request_comment_id": anchor_id,
                "request_comment_url": existing_cycle["request_comment_url"],
                "anchor_since": anchor_since.isoformat().replace("+00:00", "Z"),
                "suppressed_post_reason": existing_cycle["reason"],
                "completion_signal": "+1",
                "note": "A completed Codex review cycle already exists for the latest @codex review request, but GitHub only shows the bot acknowledgement reaction and no response body.",
                "items": [],
            }
        elif existing_cycle["reason"] == "existing_request_has_response":
            if force:
                emit_state_change(
                    "existing completed cycle found; forcing fresh request"
                )
                existing_cycle = None
            else:
                emit_state_change("reusing existing completed cycle")
                return {
                    "status": "existing_completed_cycle",
                    "owner": owner,
                    "repo": repo,
                    "pr_number": pr_number,
                    "url": context["url"],
                    "head_sha": context["head_sha"],
                    "request_attempts": 0,
                    "eyes_confirmed": False,
                    "request_comment_id": anchor_id,
                    "request_comment_url": existing_cycle["request_comment_url"],
                    "anchor_since": anchor_since.isoformat().replace("+00:00", "Z"),
                    "suppressed_post_reason": existing_cycle["reason"],
                    "note": "A completed Codex review cycle already exists for the latest @codex review request. Decide whether that feedback was already addressed before rerunning. Use --force to post a fresh request anyway.",
                    "items": existing_cycle["items"],
                }
        else:
            emit_state_change("existing review request still active")

    if existing_cycle is None:
        request = post_request(
            owner=owner,
            repo=repo,
            pr_number=pr_number,
            body=body,
            head_sha=context["head_sha"],
            bot_login=bot_login,
        )
        request_attempts += 1
        anchor_id = request["request_comment_id"]
        request_comment_url = request["request_comment_url"]
        anchor_since = parse_iso(request["requested_at"])
        emit_state_change(f"review request posted (attempt {request_attempts})")

    while utc_now() < deadline:
        now = utc_now()
        head_sha = get_pr_head_sha(owner, repo, pr_number)
        if not eyes_confirmed and has_eyes_reaction(
            owner=owner, repo=repo, comment_id=anchor_id, bot_login=bot_login
        ):
            eyes_confirmed = True
            emit_state_change("review in progress")

        items = collect_cycle_items(
            owner=owner,
            repo=repo,
            pr_number=pr_number,
            bot_login=bot_login,
            head_sha=head_sha,
            anchor_since=anchor_since,
            seen=seen,
        )
        if items:
            collected_items.extend(items)
            last_new_item_at = utc_now()
            emit_state_change("review response received")

        if collected_items and last_new_item_at is not None:
            settle_elapsed = int((utc_now() - last_new_item_at).total_seconds())
            if settle_elapsed >= settle_seconds:
                return {
                    "status": "response_found",
                    "owner": owner,
                    "repo": repo,
                    "pr_number": pr_number,
                    "url": context["url"],
                    "head_sha": head_sha,
                    "request_attempts": request_attempts,
                    "eyes_confirmed": eyes_confirmed,
                    "request_comment_id": anchor_id,
                    "request_comment_url": request_comment_url,
                    "anchor_since": anchor_since.isoformat().replace("+00:00", "Z"),
                    "items": collected_items,
                }

        if not collected_items and has_plus_one_reaction(
            owner=owner, repo=repo, comment_id=anchor_id, bot_login=bot_login
        ):
            emit_state_change("review completed without response body")
            return {
                "status": "acknowledged_without_body",
                "owner": owner,
                "repo": repo,
                "pr_number": pr_number,
                "url": context["url"],
                "head_sha": head_sha,
                "request_attempts": request_attempts,
                "eyes_confirmed": eyes_confirmed,
                "request_comment_id": anchor_id,
                "request_comment_url": request_comment_url,
                "anchor_since": anchor_since.isoformat().replace("+00:00", "Z"),
                "completion_signal": "+1",
                "note": "GitHub shows the bot acknowledgement reaction, but no response body was posted for this review cycle.",
                "items": [],
            }

        if status_interval_seconds > 0:
            if (
                last_status_emit_at is None
                or int((now - last_status_emit_at).total_seconds())
                >= status_interval_seconds
            ):
                seconds_since_request = max(
                    int((now - anchor_since).total_seconds()), 0
                )
                minutes_since_request = max(1, seconds_since_request // 60)
                emit_state_change(
                    f"OK {minutes_since_request}m: github eyes={'yes' if eyes_confirmed else 'no'} responses={len(collected_items)}"
                )
                last_status_emit_at = now
        seconds_since_request = int((now - anchor_since).total_seconds())
        if (
            not eyes_confirmed
            and request_attempts < max_request_attempts
            and seconds_since_request >= re_request_after_seconds
        ):
            existing_cycle = inspect_existing_cycle(
                owner=owner,
                repo=repo,
                pr_number=pr_number,
                body=body,
                bot_login=bot_login,
                head_sha=head_sha,
            )
            if existing_cycle:
                anchor_id = existing_cycle["request_comment_id"]
                request_comment_url = str(
                    existing_cycle.get("request_comment_url") or ""
                )
                anchor_since = parse_iso(existing_cycle["requested_at"])
                if existing_cycle["reason"] == "existing_request_has_eyes":
                    eyes_confirmed = True
                    emit_state_change("review in progress")
                    continue
                if (
                    existing_cycle["reason"]
                    == "existing_request_acknowledged_without_body"
                ):
                    emit_state_change("reusing existing completed cycle")
                    return {
                        "status": "existing_completed_cycle",
                        "owner": owner,
                        "repo": repo,
                        "pr_number": pr_number,
                        "url": context["url"],
                        "head_sha": head_sha,
                        "request_attempts": request_attempts,
                        "eyes_confirmed": eyes_confirmed,
                        "request_comment_id": anchor_id,
                        "request_comment_url": existing_cycle["request_comment_url"],
                        "anchor_since": anchor_since.isoformat().replace("+00:00", "Z"),
                        "suppressed_post_reason": existing_cycle["reason"],
                        "completion_signal": "+1",
                        "note": "A completed Codex review cycle already exists for the latest @codex review request, but GitHub only shows the bot acknowledgement reaction and no response body.",
                        "items": [],
                    }
                if existing_cycle["reason"] == "existing_request_has_response":
                    if force:
                        emit_state_change(
                            "existing completed cycle found; forcing fresh request"
                        )
                    else:
                        emit_state_change("reusing existing completed cycle")
                        return {
                            "status": "existing_completed_cycle",
                            "owner": owner,
                            "repo": repo,
                            "pr_number": pr_number,
                            "url": context["url"],
                            "head_sha": head_sha,
                            "request_attempts": request_attempts,
                            "eyes_confirmed": eyes_confirmed,
                            "request_comment_id": anchor_id,
                            "request_comment_url": existing_cycle[
                                "request_comment_url"
                            ],
                            "anchor_since": anchor_since.isoformat().replace(
                                "+00:00", "Z"
                            ),
                            "suppressed_post_reason": existing_cycle["reason"],
                            "note": "A completed Codex review cycle already exists for the latest @codex review request. Decide whether that feedback was already addressed before rerunning. Use --force to post a fresh request anyway.",
                            "items": existing_cycle["items"],
                        }
                emit_state_change("reposting review request after pickup timeout")
            request = post_request(
                owner=owner,
                repo=repo,
                pr_number=pr_number,
                body=body,
                head_sha=head_sha,
                bot_login=bot_login,
            )
            request_attempts += 1
            anchor_id = request["request_comment_id"]
            request_comment_url = request["request_comment_url"]
            anchor_since = parse_iso(request["requested_at"])
            emit_state_change(f"review request reposted (attempt {request_attempts})")
            continue

        time.sleep(poll_seconds)

    emit_state_change("review cycle timed out")
    return {
        "status": "timeout",
        "owner": owner,
        "repo": repo,
        "pr_number": pr_number,
        "url": context["url"],
        "head_sha": get_pr_head_sha(owner, repo, pr_number),
        "request_attempts": request_attempts,
        "eyes_confirmed": eyes_confirmed,
        "request_comment_id": anchor_id,
        "request_comment_url": request_comment_url,
        "anchor_since": anchor_since.isoformat().replace("+00:00", "Z"),
        "timeout_minutes": timeout_minutes,
        "items": [],
    }


def cmd_run(args: argparse.Namespace) -> int:
    global GH_CWD
    explicit_target = bool(args.owner and args.repo and args.pr_number)
    review_root: Path | None = None
    if explicit_target and not args.cd:
        GH_CWD = None
    else:
        review_root = resolve_repo_root(args.cd)
        GH_CWD = str(review_root)
    context = get_pr_context(args.owner, args.repo, args.pr_number)
    payload = run_review_cycle(
        owner=context["owner"],
        repo=context["repo"],
        pr_number=context["pr_number"],
        body=DEFAULT_REQUEST_BODY,
        bot_login=args.bot_login,
        poll_seconds=DEFAULT_POLL_SECONDS,
        timeout_minutes=DEFAULT_TIMEOUT_MINUTES,
        status_interval_seconds=DEFAULT_STATUS_INTERVAL_SECONDS,
        re_request_after_seconds=DEFAULT_RE_REQUEST_AFTER_SECONDS,
        max_request_attempts=DEFAULT_MAX_REQUEST_ATTEMPTS,
        settle_seconds=DEFAULT_SETTLE_SECONDS,
        force=args.force,
    )
    if payload["status"] in {
        "response_found",
        "acknowledged_without_body",
    } and cycle_has_review_body(payload):
        try:
            if review_root is None:
                raise RuntimeError(
                    "no local review root available for workflow anchor recording"
                )
            if not local_checkout_contains_reviewed_head(
                review_root, str(payload["head_sha"])
            ):
                emit_state_change(
                    "skipping workflow anchor; local checkout does not contain the reviewed head"
                )
                emit_toon(public_cycle_payload(payload))
                return 0
            output_refs = [
                str(item.get("url") or "").strip()
                for item in list(payload.get("items") or [])
                if str(item.get("url") or "").strip()
            ]
            request_comment_url = str(payload.get("request_comment_url") or "").strip()
            if request_comment_url and request_comment_url not in output_refs:
                output_refs.append(request_comment_url)
            base_ref = str(context.get("base_ref") or "").strip() or None
            record_review_anchor(
                state_dir=Path(args.state_dir),
                review_cwd=review_root,
                lane="review-github",
                base=base_ref,
                review_scope=github_review_scope(
                    review_root=review_root,
                    base_ref=base_ref,
                    head_sha=str(payload["head_sha"]),
                ),
                reviewed_head=str(payload["head_sha"]),
                task_id=f"pr-{context['pr_number']}",
                output_refs=output_refs,
            )
        except Exception as exc:  # pragma: no cover - warning path only
            if review_root is not None:
                _record_anchor_warning(exc)
    emit_toon(public_cycle_payload(payload))
    return (
        0
        if payload["status"]
        in {"response_found", "acknowledged_without_body", "existing_completed_cycle"}
        else 2
    )


def build_parser() -> argparse.ArgumentParser:
    parser = AxiArgumentParser(
        description="Deterministic GitHub PR review request and anchored polling."
    )
    sub = parser.add_subparsers(
        dest="command", required=True, parser_class=AxiArgumentParser
    )

    run = sub.add_parser("run")
    run.add_argument("--cd")
    run.add_argument("--owner")
    run.add_argument("--repo")
    run.add_argument("--pr-number", type=int)
    run.add_argument("--bot-login", default=DEFAULT_BOT_LOGIN, help=argparse.SUPPRESS)
    run.add_argument(
        "--state-dir", default=str(default_state_dir()), help=argparse.SUPPRESS
    )
    run.add_argument("--force", action="store_true")

    return parser


def main() -> int:
    parser = build_parser()
    try:
        args = parser.parse_args()
        if args.command == "run":
            return cmd_run(args)
    except (RuntimeError, ValueError) as exc:
        return emit_error(
            str(exc),
            status="error",
            help_items=[_help_command()],
        )
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
