from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from review_suite_core import normalize_usage_tokens

DEFAULT_CODEX_HOME = Path.home() / ".codex"
DEFAULT_SQLITE_STATE_PATH = DEFAULT_CODEX_HOME / "state_5.sqlite"
REVIEW_SUBAGENT_SOURCE = json.dumps({"subagent": "review"}, separators=(",", ":"))

type _FileVersion = tuple[int, int]
type _RolloutRecords = tuple[list[dict[str, Any]], bool]

_JSONL_CACHE: dict[Path, tuple[_FileVersion, _RolloutRecords]] = {}
_ROLLOUT_SUMMARY_CACHE: dict[Path, tuple[_FileVersion, dict[str, Any]]] = {}
_ROLLOUT_META_CACHE: dict[Path, tuple[_FileVersion, dict[str, Any]]] = {}
_THREAD_LOOKUP_CACHE: dict[
    tuple[str, int, int, str, str | None, int | None], dict[str, Any] | None
] = {}


def _path_version(path: Path) -> _FileVersion:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def _read_rollout_records(path: Path) -> _RolloutRecords:
    version = _path_version(path)
    cached = _JSONL_CACHE.get(path)
    if cached is not None and cached[0] == version:
        return cached[1]
    rows: list[dict[str, Any]] = []
    incomplete = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                incomplete = True
    result = rows, incomplete
    _JSONL_CACHE[path] = version, result
    return result


def _token_usage_dict(value: dict[str, Any] | None) -> dict[str, int] | None:
    return normalize_usage_tokens(value)


def read_rollout_summary(path: Path) -> dict[str, Any]:
    version = _path_version(path)
    cached = _ROLLOUT_SUMMARY_CACHE.get(path)
    if cached is not None and cached[0] == version:
        return cached[1]
    rows, incomplete = _read_rollout_records(path)
    if incomplete:
        # A missing record can hide a turn boundary. Only use readable records
        # to track activity, not to recover a verdict.
        return {"rollout_incomplete": True}
    last_turn_context_index = 0
    for index, row in enumerate(rows):
        if row.get("type") == "turn_context":
            last_turn_context_index = index
    first_total: dict[str, int] | None = None
    first_last: dict[str, int] | None = None
    final_total: dict[str, int] | None = None
    final_text = ""
    task_error = ""
    for row in rows[last_turn_context_index:]:
        row_type = row.get("type")
        if row_type == "event_msg":
            payload = row.get("payload", {})
            if not isinstance(payload, dict):
                continue
            if payload.get("type") == "task_complete":
                error = payload.get("error")
                if isinstance(error, dict):
                    task_error = str(error.get("message") or "")
                continue
            if payload.get("type") != "token_count":
                continue
            info = payload.get("info", {})
            if not isinstance(info, dict):
                continue
            total_usage = _token_usage_dict(info.get("total_token_usage"))
            last_usage = _token_usage_dict(info.get("last_token_usage"))
            if total_usage is None:
                continue
            if first_total is None:
                first_total = total_usage
                first_last = last_usage or {
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "cache_write_tokens": 0,
                    "output_tokens": 0,
                }
            final_total = total_usage
            continue
        if row_type != "response_item":
            continue
        payload = row.get("payload", {})
        if not isinstance(payload, dict):
            continue
        if (
            payload.get("type") != "message"
            or payload.get("role") != "assistant"
            or payload.get("phase") != "final_answer"
        ):
            continue
        content = payload.get("content", [])
        if not isinstance(content, list):
            continue
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(item["text"])
        if parts:
            final_text = "\n".join(parts).strip()
    usage: dict[str, int] | None = None
    if final_total is not None:
        if first_total is None:
            usage = final_total
        else:
            if first_last is None:
                first_last = {
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "cache_write_tokens": 0,
                    "output_tokens": 0,
                }
            usage = {
                metric: max(
                    0,
                    final_total.get(metric, 0)
                    - first_total.get(metric, 0)
                    + first_last.get(metric, 0),
                )
                for metric in (
                    "input_tokens",
                    "cached_input_tokens",
                    "cache_write_tokens",
                    "output_tokens",
                )
                if final_total.get(metric, 0)
                or first_total.get(metric, 0)
                or first_last.get(metric, 0)
            }
    summary = {
        "usage": usage,
        "reviewer_output": final_text,
        "task_error": task_error,
    }
    _ROLLOUT_SUMMARY_CACHE[path] = version, summary
    return summary


def rollout_final_token_usage(path: Path) -> dict[str, int] | None:
    return read_rollout_summary(path).get("usage")


def rollout_final_assistant_text(path: Path) -> str:
    return str(read_rollout_summary(path).get("reviewer_output") or "")


def _rollout_session_meta(path: Path) -> dict[str, Any]:
    try:
        version = _path_version(path)
    except OSError:
        return {}
    cached = _ROLLOUT_META_CACHE.get(path)
    if cached is not None and cached[0] == version:
        return cached[1]
    meta = {}
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    break
                if row.get("type") == "session_meta":
                    payload = row.get("payload", {})
                    if isinstance(payload, dict):
                        meta = {
                            field: payload.get(field)
                            for field in ("id", "parent_thread_id", "source")
                        }
                    break
    except OSError, json.JSONDecodeError:
        pass
    _ROLLOUT_META_CACHE[path] = version, meta
    return meta


def rollout_parent_thread_id(path: Path) -> str | None:
    return str(_rollout_session_meta(path).get("parent_thread_id") or "") or None


def _parse_event_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _is_meaningful_activity(row: dict[str, Any]) -> bool:
    row_type = row.get("type")
    payload = row.get("payload", {})
    if not isinstance(payload, dict):
        return False
    if row_type == "response_item":
        payload_type = str(payload.get("type") or "")
        if payload_type in {
            "custom_tool_call",
            "custom_tool_call_output",
            "function_call",
            "function_call_output",
            "reasoning",
        }:
            return True
        if payload_type != "message" or payload.get("role") != "assistant":
            return False
        content = payload.get("content", [])
        if not isinstance(content, list):
            return False
        return any(
            isinstance(item, dict)
            and str(item.get("text") or item.get("output_text") or "").strip()
            for item in content
        )
    if row_type == "event_msg":
        return str(payload.get("type") or "") in {"agent_message", "task_complete"}
    return False


def rollout_activity_summary(path: Path) -> dict[str, Any]:
    rows, _ = _read_rollout_records(path)
    last_event_at: datetime | None = None
    last_meaningful_at: datetime | None = None
    last_meaningful_type: str | None = None
    for row in rows:
        timestamp = _parse_event_timestamp(row.get("timestamp"))
        if timestamp is not None:
            last_event_at = timestamp
        if _is_meaningful_activity(row):
            last_meaningful_at = timestamp
            payload = row.get("payload", {})
            last_meaningful_type = str(payload.get("type") or row.get("type") or "")
    return {
        "last_event_at": last_event_at,
        "last_meaningful_at": last_meaningful_at,
        "last_meaningful_type": last_meaningful_type,
    }


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except TypeError, ValueError:
        return None


def _missing_threads_table(exc: sqlite3.Error) -> bool:
    return "no such table: threads" in str(exc).lower()


def find_thread_by_title(
    *,
    sqlite_path: Path,
    title: str,
    cwd: str | None = None,
    created_after: int | None = None,
) -> dict[str, Any] | None:
    if not sqlite_path.is_file():
        return None
    stat = sqlite_path.stat()
    cache_key = (
        str(sqlite_path),
        stat.st_mtime_ns,
        stat.st_size,
        title,
        cwd,
        created_after,
    )
    cached = _THREAD_LOOKUP_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached) if cached else None
    con = sqlite3.connect(str(sqlite_path))
    con.row_factory = sqlite3.Row
    try:
        try:
            sql = """
                select id, rollout_path, cwd, source, created_at, updated_at, tokens_used, model, reasoning_effort, title
                from threads
                where title = ?
            """
            params: list[Any] = [title]
            if cwd is not None:
                sql += " and cwd = ?"
                params.append(cwd)
            if created_after is not None:
                sql += " and created_at >= ?"
                params.append(created_after)
            sql += " order by updated_at desc limit 1"
            row = con.execute(sql, params).fetchone()
        except sqlite3.OperationalError as exc:
            if not _missing_threads_table(exc):
                raise
            row = None
    finally:
        con.close()
    result = dict(row) if row else None
    _THREAD_LOOKUP_CACHE[cache_key] = result
    return dict(result) if result else None


def find_review_child_thread(
    *,
    sqlite_path: Path,
    parent_thread_id: str,
    started_at: str | None = None,
    review_cwd: Path | None = None,
) -> dict[str, Any] | None:
    parent_id = str(parent_thread_id or "").strip()
    if not parent_id:
        return None
    indexed = _find_indexed_review_child_thread(sqlite_path, parent_id)
    if indexed is not None:
        return indexed

    started = _parse_event_timestamp(started_at)
    if started is None:
        return None
    # Codex uses local dates for session paths. Include adjacent days for midnight.
    start_day = started.astimezone().date()
    codex_home = Path(os.environ.get("CODEX_HOME") or DEFAULT_CODEX_HOME)
    sessions = (review_cwd or Path.cwd()) / codex_home / "sessions"
    for offset in (0, 1, -1):
        day = start_day + timedelta(days=offset)
        directory = sessions / f"{day:%Y/%m/%d}"
        for path in sorted(directory.glob("rollout-*.jsonl"), reverse=True):
            if rollout_parent_thread_id(path) != parent_id:
                continue
            meta = _rollout_session_meta(path)
            thread_id = meta.get("id")
            if (
                meta.get("source") == {"subagent": "review"}
                and isinstance(thread_id, str)
                and thread_id
            ):
                return {
                    "id": thread_id,
                    "rollout_path": str(path),
                    "source": REVIEW_SUBAGENT_SOURCE,
                }
    return None


def _find_indexed_review_child_thread(
    sqlite_path: Path, parent_id: str
) -> dict[str, Any] | None:
    if not sqlite_path.is_file():
        return None
    stat = sqlite_path.stat()
    cache_key = (
        str(sqlite_path),
        stat.st_mtime_ns,
        stat.st_size,
        f"review_child_parent:{parent_id}",
        None,
        None,
    )
    cached = _THREAD_LOOKUP_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached) if cached else None
    con = sqlite3.connect(str(sqlite_path))
    con.row_factory = sqlite3.Row
    try:
        try:
            sql = """
                select id, rollout_path, cwd, source, created_at, updated_at, tokens_used, model, reasoning_effort, title
                from threads
                where source = ?
                order by created_at desc, id asc
            """
            row = None
            for candidate in con.execute(sql, [REVIEW_SUBAGENT_SOURCE]):
                rollout_path = Path(str(candidate["rollout_path"] or ""))
                if (
                    rollout_path.is_file()
                    and rollout_parent_thread_id(rollout_path) == parent_id
                ):
                    row = candidate
                    break
        except sqlite3.OperationalError as exc:
            if not _missing_threads_table(exc):
                raise
            row = None
    finally:
        con.close()
    result = dict(row) if row else None
    _THREAD_LOOKUP_CACHE[cache_key] = result
    return dict(result) if result else None


def find_thread_by_id(*, sqlite_path: Path, thread_id: str) -> dict[str, Any] | None:
    if not sqlite_path.is_file():
        return None
    con = sqlite3.connect(str(sqlite_path))
    con.row_factory = sqlite3.Row
    try:
        try:
            row = con.execute(
                """
                select id, rollout_path, cwd, source, created_at, updated_at, tokens_used, model, reasoning_effort, title
                from threads
                where id = ?
                limit 1
                """,
                [thread_id],
            ).fetchone()
        except sqlite3.OperationalError as exc:
            if not _missing_threads_table(exc):
                raise
            row = None
    finally:
        con.close()
    return dict(row) if row else None


def enrich_thread_record(thread_row: dict[str, Any]) -> dict[str, Any]:
    rollout_path = Path(thread_row["rollout_path"])
    summary = read_rollout_summary(rollout_path) if rollout_path.is_file() else {}
    enriched = dict(thread_row)
    usage = summary.get("usage") or {}
    enriched["usage"] = usage
    if "tokens_used" not in enriched and usage:
        enriched["tokens_used"] = usage.get("input_tokens", 0) + usage.get(
            "output_tokens", 0
        )
    enriched["reviewer_output"] = summary.get("reviewer_output") or ""
    enriched["task_error"] = summary.get("task_error") or ""
    enriched["rollout_incomplete"] = bool(summary.get("rollout_incomplete"))
    return enriched
