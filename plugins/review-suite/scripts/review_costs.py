from __future__ import annotations

import re
import json
import os
import sqlite3
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import blake2s
from pathlib import Path
from typing import Any

from review_suite_core import (
    SUPPORTED_REASONING_EFFORTS,
    SUPPORTED_SERVICE_TIERS,
    cwd_path_from_normalized,
    load_config,
    normalize_usage_tokens,
    price_usage_tokens,
    run_total_tokens,
)
from review_suite_core.orchestrator_store import orchestrator_store_lock
from review_suite_local import (
    _run_is_finalized,
    read_jsonl,
    iter_round_payloads,
    normalize_record_review_cwd_value,
    normalize_review_cwd_value,
    write_json,
)

LANES = ("review_t1", "review_t3", "review_followup")
TASK_TO_LANE = {
    "phase_review": "review_t1",
    "pr_review": "review_t3",
}
PUBLIC_TASK_TO_LANE = {
    "review_t1": "review_t1",
    "review_t3": "review_t3",
    "review-followup": "review_followup",
    "review_followup": "review_followup",
}
DEFAULT_COST_REPORT_FILENAME = "review_cost_ledger.md"
DEFAULT_COST_CACHE_DIRNAME = "review_cost_rows"
REVIEW_COSTS_LOCK_NAME = "review-cost-accounting"
REVIEW_COSTS_LOCK_TIMEOUT_SECONDS = 10 * 60
AUTOMATIC_COST_DELTA_LOCK_TIMEOUT_SECONDS = 0
WRAPPER_SESSION_LOG_FILENAME = "wrapper_sessions.jsonl"
ORCHESTRATOR_REVIEW_STATE_DIR = Path("orchestrator") / "review-rounds"
FOLDER_REPO_OVERRIDES = {
    "sample-stack-allow-chat-skipped-default": "sample-stack",
}
DEFAULT_CODEX_SQLITE_FILENAME = "state_5.sqlite"
MODEL_PRICING_PER_MILLION = {
    "gpt-6-sol": {
        "input": 2.00,
        "output": 10.00,
        "cached_input": 0.20,
        "cache_write": 2.50,
    },
    "gpt-6-luna": {
        "input": 0.10,
        "output": 0.50,
        "cached_input": 0.01,
        "cache_write": 0.125,
    },
    "gpt-6-astra": {
        "input": 10.00,
        "output": 50.00,
        "cached_input": 1.00,
        "cache_write": 12.50,
    },
    "gpt-5.6-sol": {
        "input": 5.00,
        "output": 30.00,
        "cached_input": 0.50,
        "cache_write": 6.25,
    },
    "gpt-5.6-terra": {
        "input": 2.00,
        "output": 12.00,
        "cached_input": 0.20,
        "cache_write": 2.50,
    },
    "gpt-5.6-luna": {
        "input": 0.20,
        "output": 1.20,
        "cached_input": 0.02,
        "cache_write": 0.25,
    },
    "gpt-5.5": {"input": 5.00, "output": 30.00, "cached_input": 0.50},
    "gpt-5.4": {"input": 2.50, "output": 15.00, "cached_input": 0.25},
    "gpt-5-mini": {"input": 0.75, "output": 4.50, "cached_input": 0.075},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50, "cached_input": 0.075},
    "gpt-5-nano": {"input": 0.15, "output": 0.60, "cached_input": 0.015},
    "gpt-5.4-nano": {"input": 0.15, "output": 0.60, "cached_input": 0.015},
    "gpt-5.3-codex": {"input": 2.00, "output": 10.00, "cached_input": 0.20},
    "gpt-5.3-codex-spark": {"input": 2.00, "output": 10.00, "cached_input": 0.20},
    "gpt-5.2-codex": {"input": 2.00, "output": 10.00, "cached_input": 0.20},
    "gpt-5.2": {"input": 2.00, "output": 10.00, "cached_input": 0.20},
    "gpt-4.1": {"input": 2.00, "output": 8.00, "cached_input": 0.50},
    "o3": {"input": 2.00, "output": 8.00, "cached_input": 0.50},
    "o4-mini": {"input": 1.10, "output": 4.40, "cached_input": 0.275},
}
MODEL_ALIASES = {
    "codex-mini-latest": "o4-mini",
    "gpt 5.6 sol": "gpt-5.6-sol",
    "gpt-5.6 sol": "gpt-5.6-sol",
    "gpt 5.6 terra": "gpt-5.6-terra",
    "gpt-5.6 terra": "gpt-5.6-terra",
    "gpt 5.6 luna": "gpt-5.6-luna",
    "gpt-5.6 luna": "gpt-5.6-luna",
    "gpt 5.5": "gpt-5.5",
    "gpt 5.4": "gpt-5.4",
    "gpt 5 mini": "gpt-5-mini",
    "gpt-5 mini": "gpt-5-mini",
    "gpt 5.4 mini": "gpt-5.4-mini",
    "gpt-5.4 mini": "gpt-5.4-mini",
    "gpt 5 nano": "gpt-5-nano",
    "gpt-5 nano": "gpt-5-nano",
    "gpt 5.4 nano": "gpt-5.4-nano",
    "gpt-5.4 nano": "gpt-5.4-nano",
}
MODEL_HYPHEN_CHARS = re.compile(r"[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]")
HEURISTIC_INPUT_FRACTION = 0.75
HEURISTIC_OUTPUT_FRACTION = 0.25
HEURISTIC_CACHE_HIT_RATE = 0.95


@dataclass(frozen=True)
class ReviewCostRow:
    repo: str
    folder: str
    branch: str
    pr_number: str
    worker_model: str
    implementation_tokens: int
    implementation_cost_usd: float
    caller_threads: tuple[str, ...]
    latest_review: str
    lane_sessions: dict[str, int]
    review_seconds: float
    tokens: int
    cost_usd: float
    review_cwd_normalized: str = ""
    accounted_run_ids: tuple[str, ...] = ()
    review_cost_cutover_at: str = ""
    implementation_costs_refreshed: bool = True


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
    except ValueError:
        return None


def _seconds_between(start: str | None, end: str | None) -> float | None:
    start_dt = _parse_iso(start)
    end_dt = _parse_iso(end)
    if start_dt is None or end_dt is None:
        return None
    return max(0.0, (end_dt - start_dt).total_seconds())


def _run_seconds(record: dict[str, Any], runs: list[dict[str, Any]]) -> float:
    exact = _seconds_between(
        str(record.get("sampled_at") or record.get("round_started_at") or ""),
        str(record.get("review_completed_at") or record.get("recorded_at") or ""),
    )
    if exact is not None:
        return exact
    elapsed_values = [
        float(run["elapsed_seconds"])
        for run in runs
        if isinstance(run.get("elapsed_seconds"), (int, float))
    ]
    return max(elapsed_values) if elapsed_values else 0.0


def _run_tokens(run: dict[str, Any]) -> int:
    return run_total_tokens(run)


def _run_model_name(run: dict[str, Any]) -> str:
    model_name = _normalize_model_name(run.get("model"))
    if model_name:
        return model_name
    parts = str(run.get("variant_id") or "").strip().split("-")
    suffixes = SUPPORTED_REASONING_EFFORTS | SUPPORTED_SERVICE_TIERS
    while parts and parts[-1] in suffixes:
        parts.pop()
    return _normalize_model_name("-".join(parts))


def _run_cost(run: dict[str, Any]) -> float:
    cost = run.get("cost_usd")
    if isinstance(cost, (int, float)):
        return float(cost)
    model_name = _run_model_name(run)
    usage = dict(run.get("usage") or {})
    return (
        _price_from_usage(model_name, usage)
        or _price_from_total_tokens(model_name, _run_tokens(run))
        or 0.0
    )


def _default_codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex"))


def _normalize_model_name(raw_model: str | None) -> str:
    if not raw_model:
        return ""
    lookup = MODEL_HYPHEN_CHARS.sub("-", str(raw_model)).lower().strip()
    lookup = re.sub(r"\s+", " ", lookup)
    return MODEL_ALIASES.get(lookup, lookup)


def _usage_tokens(usage: dict[str, Any] | None) -> int:
    if not usage:
        return 0
    total = usage.get("total_tokens")
    if isinstance(total, (int, float)) and total > 0:
        return int(total)
    return int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)


def _price_from_usage(model_name: str, usage: dict[str, Any] | None) -> float | None:
    pricing = MODEL_PRICING_PER_MILLION.get(model_name)
    return price_usage_tokens(pricing or {}, usage)


def _price_from_total_tokens(model_name: str, total_tokens: int) -> float | None:
    pricing = MODEL_PRICING_PER_MILLION.get(model_name)
    if not pricing:
        return None
    effective_input_rate = ((1 - HEURISTIC_CACHE_HIT_RATE) * pricing["input"]) + (
        HEURISTIC_CACHE_HIT_RATE * pricing["cached_input"]
    )
    rate = (HEURISTIC_INPUT_FRACTION * effective_input_rate) + (
        HEURISTIC_OUTPUT_FRACTION * pricing["output"]
    )
    return (total_tokens / 1_000_000) * rate


def _read_rollout_usage(rollout_path: str | None) -> dict[str, int] | None:
    if not rollout_path:
        return None
    path = Path(rollout_path)
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            position = handle.tell()
            tail = b""
            while position > 0:
                size = min(1024 * 1024, position)
                position -= size
                handle.seek(position)
                tail = handle.read(size) + tail
                lines = tail.splitlines()
                if position > 0 and lines:
                    lines = lines[1:]
                for raw_line in reversed(lines):
                    usage = _usage_from_rollout_line(
                        raw_line.decode("utf-8", errors="replace")
                    )
                    if usage is not None:
                        return usage
    except OSError:
        return None
    return None


def _read_rollout_model_metadata(rollout_path: str | None) -> dict[str, str]:
    if not rollout_path:
        return {}
    path = Path(rollout_path)
    if not path.exists():
        return {}
    metadata: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip() or (
                    "model" not in line
                    and "effort" not in line
                    and "reasoning_effort" not in line
                ):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") == "turn_context" and isinstance(
                    obj.get("payload"), dict
                ):
                    payload = obj["payload"]
                    if payload.get("model") and not metadata.get("model"):
                        metadata["model"] = str(payload["model"])
                    if payload.get("effort") and not metadata.get("reasoning_effort"):
                        metadata["reasoning_effort"] = str(payload["effort"])
                    settings = (
                        payload.get("collaboration_mode", {}).get("settings")
                        if isinstance(payload.get("collaboration_mode"), dict)
                        else None
                    )
                    if isinstance(settings, dict):
                        if settings.get("model") and not metadata.get("model"):
                            metadata["model"] = str(settings["model"])
                        if settings.get("reasoning_effort") and not metadata.get(
                            "reasoning_effort"
                        ):
                            metadata["reasoning_effort"] = str(
                                settings["reasoning_effort"]
                            )
                if obj.get("type") == "session_meta" and isinstance(
                    obj.get("payload"), dict
                ):
                    payload = obj["payload"]
                    if payload.get("model") and not metadata.get("model"):
                        metadata["model"] = str(payload["model"])
                if metadata.get("model") and metadata.get("reasoning_effort"):
                    break
    except OSError:
        return metadata
    return metadata


def _usage_from_rollout_line(line: str) -> dict[str, int] | None:
    if not line.strip() or '"token_count"' not in line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    payload = obj.get("payload")
    if (
        obj.get("type") != "event_msg"
        or not isinstance(payload, dict)
        or payload.get("type") != "token_count"
    ):
        return None
    info = payload.get("info")
    usage = info.get("total_token_usage") if isinstance(info, dict) else None
    if not isinstance(usage, dict):
        return None
    normalized = normalize_usage_tokens(usage)
    if normalized is None:
        return None
    normalized["reasoning_output_tokens"] = int(
        usage.get("reasoning_output_tokens") or 0
    )
    if "total_tokens" in usage:
        normalized["total_tokens"] = int(usage.get("total_tokens") or 0)
    return normalized


def _parse_source(raw_source: str | None) -> dict[str, Any]:
    if not raw_source:
        return {}
    text = str(raw_source).strip()
    if not text:
        return {}
    if text.startswith(("{", "[")):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"kind": text}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {"kind": text}


def _is_review_title(title: str | None) -> bool:
    value = str(title or "").strip().lower()
    return (
        value.startswith("review-suite::")
        or value.startswith("review the code changes against the base branch ")
        or value.startswith("you are reviewing a manually supplied diff artifact.")
        or value.startswith("brief review for commit range ")
        or value.startswith("independent brief review ")
        or value.startswith("tight-scope integration review ")
        or value.startswith("pr-scope review ")
        or value.startswith("independent second-round pr review ")
        or value.startswith("second independent pr-scope review ")
        or value.startswith("implementation-review preflight ")
        or value.startswith("implementation-review postflight ")
        or value.startswith("review this implementation plan ")
        or value.startswith("review the current repository changes ")
        or value.startswith("review this follow-up diff ")
    )


def _is_review_session(row: dict[str, Any]) -> bool:
    agent_role = str(row.get("agent_role") or "").lower()
    if agent_role.startswith("review"):
        return True
    source = _parse_source(row.get("source"))
    if str(source.get("subagent") or "").lower().startswith("review"):
        return True
    return _is_review_title(row.get("title"))


def _wrapper_session_ids(state_dir: Path) -> set[str]:
    session_ids: set[str] = set()
    for row in read_jsonl(state_dir / WRAPPER_SESSION_LOG_FILENAME):
        session_id = str(row.get("session_id") or "").strip()
        tool_name = str(row.get("tool_name") or "").strip()
        if session_id and tool_name.startswith("review-"):
            session_ids.add(session_id)
    return session_ids


def _cheap_cwd_key(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = normalize_review_cwd_value(text) or text
    text = re.sub(r"^\\\\\?\\", "", text)
    text = text.replace("\\", "/")
    if re.match(r"^[A-Z]:", text):
        text = text[0].lower() + text[1:]
    return text.lower().rstrip("/")


def _cwd_query_candidates(normalized_cwds: set[str]) -> list[str]:
    candidates: set[str] = set()
    for normalized_cwd in normalized_cwds:
        value = str(normalized_cwd or "").strip()
        if not value:
            continue
        slash = value.replace("\\", "/")
        backslash = slash.replace("/", "\\")
        candidates.add(value.lower())
        candidates.add(slash.lower())
        candidates.add(backslash.lower())
        if value.lower().startswith("wsl:"):
            _scheme, rest = value.split(":", 1)
            distro, sep, posix_path = rest.partition(":")
            if sep and posix_path.startswith("/"):
                unc = f"//wsl.localhost/{distro}{posix_path}"
                candidates.add(unc.lower())
                candidates.add(unc.replace("/", "\\").lower())
                candidates.add(posix_path.lower())
        if re.match(r"^[a-z]:", backslash, re.IGNORECASE):
            candidates.add(f"\\\\?\\{backslash}".lower())
    return sorted(candidates)


def _sqlite_like_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _cwd_query_descendant_patterns(normalized_cwds: set[str]) -> list[str]:
    patterns: set[str] = set()
    for candidate in _cwd_query_candidates(normalized_cwds):
        patterns.add(_sqlite_like_literal(candidate.rstrip("/") + "/") + "%")
        patterns.add(_sqlite_like_literal(candidate.rstrip("\\") + "\\") + "%")
    return sorted(patterns)


def _matching_target_cwd(row_cwd: object, target_by_key: dict[str, str]) -> str:
    key = _cheap_cwd_key(row_cwd)
    if key in target_by_key:
        return target_by_key[key]
    matches = [
        (target_key, normalized_cwd)
        for target_key, normalized_cwd in target_by_key.items()
        if key.startswith(f"{target_key}/")
    ]
    if not matches:
        return ""
    return max(matches, key=lambda item: len(item[0]))[1]


def _thread_rows(
    sqlite_path: Path,
    *,
    cwd_filters: set[str] | None = None,
    id_filters: set[str] | None = None,
) -> list[dict[str, Any]]:
    if not sqlite_path.exists():
        return []
    try:
        with sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            columns = {row["name"] for row in db.execute("PRAGMA table_info(threads)")}
            if not columns:
                return []
            wanted = [
                "id",
                "rollout_path",
                "created_at",
                "updated_at",
                "source",
                "model",
                "reasoning_effort",
                "cwd",
                "title",
                "tokens_used",
                "agent_role",
                "git_branch",
            ]
            selected = [name for name in wanted if name in columns]
            if "cwd" not in selected:
                return []
            query = f"SELECT {', '.join(selected)} FROM threads"
            params: list[str] = []
            clauses: list[str] = []
            if cwd_filters:
                cwd_params = _cwd_query_candidates(cwd_filters)
                descendant_patterns = _cwd_query_descendant_patterns(cwd_filters)
                cwd_clauses: list[str] = []
                if cwd_params:
                    placeholders = ", ".join("?" for _ in cwd_params)
                    cwd_clauses.append(f"lower(cwd) IN ({placeholders})")
                    params.extend(cwd_params)
                if descendant_patterns:
                    cwd_clauses.extend(
                        "lower(cwd) LIKE ? ESCAPE '\\'" for _ in descendant_patterns
                    )
                    params.extend(descendant_patterns)
                if cwd_clauses:
                    clauses.append("(" + " OR ".join(cwd_clauses) + ")")
            if id_filters and "id" in selected:
                thread_ids = sorted(
                    str(value).strip() for value in id_filters if str(value).strip()
                )
                if thread_ids:
                    placeholders = ", ".join("?" for _ in thread_ids)
                    clauses.append(f"id IN ({placeholders})")
                    params.extend(thread_ids)
            if clauses:
                query += " WHERE " + " OR ".join(clauses)
            elif cwd_filters or id_filters:
                return []
            if "created_at" in selected:
                query += " ORDER BY created_at ASC"
            return [dict(row) for row in db.execute(query, params)]
    except sqlite3.Error:
        return []


def _model_label(model_name: str, reasoning_effort: str | None) -> str:
    if not model_name:
        return ""
    effort = str(reasoning_effort or "").strip()
    return f"{model_name} {effort}" if effort else model_name


def _summarize_worker_model(model_tokens: dict[str, int]) -> str:
    labels = [label for label in model_tokens if label]
    if not labels:
        return "-"
    if len(labels) == 1:
        return labels[0]
    top = max(labels, key=lambda label: model_tokens[label])
    return f"mixed ({top})"


def _empty_implementation_cost() -> dict[str, Any]:
    return {
        "worker_model": "-",
        "implementation_tokens": 0,
        "implementation_cost_usd": 0.0,
    }


def _caller_thread_ids(caller_threads: set[str] | None) -> set[str]:
    thread_ids: set[str] = set()
    for value in caller_threads or set():
        for part in str(value or "").split(":"):
            part = part.strip()
            if part:
                thread_ids.add(part)
    return thread_ids


def _implementation_costs_for_cwds(
    normalized_cwds: set[str],
    *,
    codex_home: Path | None = None,
    branches_by_cwd: dict[str, str] | None = None,
    excluded_session_ids: set[str] | None = None,
    caller_threads_by_cwd: dict[str, set[str]] | None = None,
) -> dict[str, dict[str, Any]]:
    if not normalized_cwds:
        return {}
    sqlite_path = (codex_home or _default_codex_home()) / DEFAULT_CODEX_SQLITE_FILENAME
    buckets: dict[str, dict[str, Any]] = {
        normalized_cwd: {
            "model_tokens": {},
            "implementation_tokens": 0,
            "implementation_cost_usd": 0.0,
        }
        for normalized_cwd in normalized_cwds
    }
    target_by_key = {
        _cheap_cwd_key(normalized_cwd): normalized_cwd
        for normalized_cwd in normalized_cwds
    }
    explicit_target_by_thread: dict[str, str] = {}
    explicit_threads_by_cwd: dict[str, set[str]] = {}
    for normalized_cwd, caller_threads in (caller_threads_by_cwd or {}).items():
        if normalized_cwd not in normalized_cwds:
            continue
        thread_ids = _caller_thread_ids(caller_threads)
        if thread_ids:
            explicit_threads_by_cwd[normalized_cwd] = thread_ids
        for thread_id in thread_ids:
            explicit_target_by_thread.setdefault(thread_id, normalized_cwd)
    excluded_ids = excluded_session_ids or set()
    seen_ids: set[str] = set()
    thread_rows = _thread_rows(
        sqlite_path,
        cwd_filters=normalized_cwds,
        id_filters=set(explicit_target_by_thread),
    )
    cwd_with_matched_explicit_threads = {
        explicit_target_by_thread[str(row.get("id") or "").strip()]
        for row in thread_rows
        if str(row.get("id") or "").strip() in explicit_target_by_thread
    }
    for row in thread_rows:
        row_id = str(row.get("id") or "").strip()
        if row_id and row_id in seen_ids:
            continue
        if row_id:
            seen_ids.add(row_id)
        if row_id in excluded_ids:
            continue
        explicit_match = explicit_target_by_thread.get(row_id)
        normalized_cwd = explicit_match or _matching_target_cwd(
            row.get("cwd"), target_by_key
        )
        if not normalized_cwd:
            continue
        if normalized_cwd in cwd_with_matched_explicit_threads and not explicit_match:
            continue
        expected_branch = str((branches_by_cwd or {}).get(normalized_cwd) or "")
        row_branch = str(row.get("git_branch") or "")
        if not explicit_match and expected_branch and expected_branch != "-":
            if "git_branch" in row and (
                not row_branch or row_branch != expected_branch
            ):
                continue
        if _is_review_session(row):
            continue
        bucket = buckets[normalized_cwd]
        usage = _read_rollout_usage(row.get("rollout_path"))
        rollout_metadata = _read_rollout_model_metadata(row.get("rollout_path"))
        model_name = _normalize_model_name(
            row.get("model") or rollout_metadata.get("model")
        )
        reasoning_effort = str(
            row.get("reasoning_effort")
            or rollout_metadata.get("reasoning_effort")
            or ""
        )
        tokens = _usage_tokens(usage)
        if tokens <= 0 and isinstance(row.get("tokens_used"), int):
            tokens = int(row["tokens_used"])
        cost = _price_from_usage(model_name, usage)
        if cost is None:
            cost = _price_from_total_tokens(model_name, tokens) or 0.0
        label = _model_label(model_name, reasoning_effort)
        bucket["implementation_tokens"] += tokens
        bucket["implementation_cost_usd"] += cost
        bucket["model_tokens"][label] = bucket["model_tokens"].get(label, 0) + tokens
    return {
        normalized_cwd: {
            "worker_model": _summarize_worker_model(bucket["model_tokens"]),
            "implementation_tokens": int(bucket["implementation_tokens"]),
            "implementation_cost_usd": round(
                float(bucket["implementation_cost_usd"]), 6
            ),
        }
        for normalized_cwd, bucket in buckets.items()
    }


def _implementation_cost_for_cwd(
    normalized_cwd: str, *, codex_home: Path | None = None
) -> dict[str, Any]:
    return _implementation_costs_for_cwds({normalized_cwd}, codex_home=codex_home).get(
        normalized_cwd,
        _empty_implementation_cost(),
    )


def _git_text(review_cwd: Path, args: list[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(review_cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _repo_from_remote(url: str) -> str:
    value = str(url or "").strip()
    if not value:
        return ""
    value = value.rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]
    if ":" in value and "/" in value and not value.startswith(("http://", "https://")):
        value = value.rsplit(":", 1)[-1]
    return value.rsplit("/", 1)[-1]


def _repo_from_worktree_folder(folder: str) -> str:
    value = str(folder or "").strip()
    if not value:
        return ""
    if value in FOLDER_REPO_OVERRIDES:
        return FOLDER_REPO_OVERRIDES[value]
    if "-wt-" in value:
        return value.split("-wt-", 1)[0]
    return value


def _current_pr_number(review_cwd: Path) -> str:
    if os.environ.get("REVIEW_SUITE_COST_SKIP_GH_PR_VIEW") == "1":
        return ""
    try:
        proc = subprocess.run(
            ["gh", "pr", "view", "--json", "number", "--jq", ".number"],
            cwd=str(review_cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=10,
        )
    except OSError, subprocess.TimeoutExpired:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _metadata_for_cwd_impl(
    normalized_cwd: str, *, include_pr_number: bool
) -> dict[str, str]:
    review_cwd = cwd_path_from_normalized(normalized_cwd)
    review_cwd_text = str(review_cwd)
    folder = review_cwd.name
    if "/" in folder or (
        folder == review_cwd_text and review_cwd_text.startswith("\\\\")
    ):
        folder = ""
    if not folder:
        folder = (review_cwd_text or normalized_cwd).rstrip("\\/").replace(
            "\\", "/"
        ).rsplit("/", 1)[-1] or "-"
    fallback_repo = _repo_from_worktree_folder(folder) or folder
    try:
        exists = review_cwd.exists()
    except OSError:
        exists = False
    if not exists:
        return {
            "repo": fallback_repo,
            "folder": folder,
            "branch": "-",
            "pr_number": "-",
        }
    branch = _git_text(review_cwd, ["branch", "--show-current"]) or "-"
    repo = (
        _repo_from_remote(_git_text(review_cwd, ["remote", "get-url", "origin"]))
        or fallback_repo
    )
    pr_number = (_current_pr_number(review_cwd) or "-") if include_pr_number else "-"
    return {"repo": repo, "folder": folder, "branch": branch, "pr_number": pr_number}


def _metadata_for_cwd(normalized_cwd: str) -> dict[str, str]:
    return _metadata_for_cwd_impl(normalized_cwd, include_pr_number=True)


def _cached_metadata_for_cwd(normalized_cwd: str) -> dict[str, str]:
    return _metadata_for_cwd_impl(normalized_cwd, include_pr_number=False)


def _record_task_id(record: dict[str, Any]) -> str:
    return str(
        record.get("graded_task_id")
        or record.get("task_id")
        or record.get("task_id_hint")
        or ""
    )


def _record_lane(record: dict[str, Any]) -> str | None:
    public_task = str(record.get("public_task") or "").strip()
    if public_task in PUBLIC_TASK_TO_LANE:
        return PUBLIC_TASK_TO_LANE[public_task]
    return TASK_TO_LANE.get(str(record.get("task_class") or ""))


def _record_timestamp(record: dict[str, Any]) -> str:
    return str(
        record.get("review_completed_at")
        or record.get("recorded_at")
        or record.get("sampled_at")
        or ""
    )


def _run_contribution_id(
    run: dict[str, Any], *, record: dict[str, Any], run_index: int
) -> str:
    for key in ("thread_id", "reviewer_output_ref", "rollout_path", "session_id"):
        value = str(run.get(key) or "").strip()
        if value:
            return f"{key}:{value}"
    stable = {
        key: run.get(key)
        for key in (
            "slot",
            "variant_id",
            "model",
            "reasoning_effort",
            "service_tier",
            "usage",
            "tokens_used",
            "cost_usd",
            "elapsed_seconds",
            "reviewer_output",
        )
        if run.get(key) is not None
    }
    stable["record_id"] = str(
        record.get("round_id") or _record_timestamp(record) or "unknown-record"
    )
    stable["run_index"] = run_index
    raw = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str)
    return f"fallback:{blake2s(raw.encode('utf-8'), digest_size=16).hexdigest()}"


def _record_cost_runs(record: dict[str, Any]) -> list[dict[str, Any]]:
    runs = [
        run
        for run in list(record.get("runs") or [])
        if isinstance(run, dict) and _run_is_finalized(run)
    ]
    if record.get("rerolled_from_round_id"):
        parent_round_id = str(record["rerolled_from_round_id"])
        runs = [
            run
            for run in runs
            if str(run.get("rerolled_from_round_id") or "") == parent_round_id
        ]
    return runs


def _iter_review_round_payloads(state_dir: Path) -> list[dict[str, Any]]:
    return [
        *iter_round_payloads(state_dir),
        *iter_round_payloads(state_dir / ORCHESTRATOR_REVIEW_STATE_DIR),
    ]


def _new_bucket() -> dict[str, Any]:
    return {
        "lane_sessions": {lane: 0 for lane in LANES},
        "review_seconds": 0.0,
        "tokens": 0,
        "cost_usd": 0.0,
        "latest_review": "",
        "caller_threads": set(),
        "accounted_run_ids": set(),
    }


def _add_record(
    bucket: dict[str, Any],
    *,
    lane: str,
    record: dict[str, Any],
    runs: list[dict[str, Any]],
) -> None:
    seen = bucket["accounted_run_ids"]
    identified = [
        (_run_contribution_id(run, record=record, run_index=index), run)
        for index, run in enumerate(runs)
    ]
    new_items = [(run_id, run) for run_id, run in identified if run_id not in seen]
    if not new_items:
        return
    seen.update(run_id for run_id, _run in new_items)
    new_runs = [run for _run_id, run in new_items]
    bucket["lane_sessions"][lane] += len(new_runs)
    bucket["review_seconds"] += _run_seconds(record, new_runs)
    bucket["tokens"] += sum(_run_tokens(run) for run in new_runs)
    bucket["cost_usd"] += sum(_run_cost(run) for run in new_runs)
    timestamp = _record_timestamp(record)
    if timestamp and timestamp > str(bucket.get("latest_review") or ""):
        bucket["latest_review"] = timestamp
    _add_caller_thread(bucket, record.get("caller_id"))


def _add_caller_thread(bucket: dict[str, Any], value: object) -> None:
    caller = str(value or "").strip()
    if caller:
        bucket.setdefault("caller_threads", set()).add(caller)


def _wrapper_session_records_by_cwd(state_dir: Path) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(state_dir / WRAPPER_SESSION_LOG_FILENAME):
        tool_name = str(row.get("tool_name") or "").strip()
        if not tool_name.startswith("review-"):
            continue
        normalized_cwd = str(normalize_record_review_cwd_value(row) or "")
        if not normalized_cwd:
            continue
        records.setdefault(normalized_cwd, []).append(row)
    return records


def collect_review_cost_rows(
    *,
    state_dir: Path,
    review_cwd: Path | None = None,
    include_all: bool = False,
    codex_home: Path | None = None,
) -> list[ReviewCostRow]:
    requested_cwd = (
        normalize_review_cwd_value(review_cwd) if review_cwd is not None else ""
    )
    buckets: dict[str, dict[str, Any]] = {}
    metadata_by_cwd: dict[str, dict[str, str]] = {}
    wrapper_records_by_cwd = _wrapper_session_records_by_cwd(state_dir)

    def metadata_for(normalized_cwd: str) -> dict[str, str]:
        metadata = metadata_by_cwd.get(normalized_cwd)
        if metadata is None:
            metadata = _metadata_for_cwd(normalized_cwd)
            metadata_by_cwd[normalized_cwd] = metadata
        return metadata

    def record_matches_current_branch(
        normalized_cwd: str, record: dict[str, Any]
    ) -> bool:
        branch = metadata_for(normalized_cwd)["branch"]
        if not branch or branch == "-":
            return True
        task_id = _record_task_id(record)
        return not task_id or task_id == branch

    for payload in _iter_review_round_payloads(state_dir):
        lane = _record_lane(payload)
        if lane not in {"review_t1", "review_t3", "review_followup"}:
            continue
        normalized_cwd = str(normalize_record_review_cwd_value(payload) or "")
        if not normalized_cwd:
            continue
        if requested_cwd and normalized_cwd != requested_cwd:
            continue
        if not include_all and not requested_cwd:
            continue
        if not record_matches_current_branch(normalized_cwd, dict(payload)):
            continue
        runs = _record_cost_runs(payload)
        if not runs:
            continue
        bucket = buckets.setdefault(normalized_cwd, _new_bucket())
        _add_record(bucket, lane=lane, record=dict(payload), runs=runs)
    for normalized_cwd, records in wrapper_records_by_cwd.items():
        if requested_cwd and normalized_cwd != requested_cwd:
            continue
        if not include_all and not requested_cwd:
            continue
        caller_threads = [
            str(record.get("caller_thread_id") or "").strip()
            for record in records
            if record_matches_current_branch(
                normalized_cwd, {"task_id": str(record.get("branch") or "")}
            )
        ]
        if not any(caller_threads):
            continue
        bucket = buckets.setdefault(normalized_cwd, _new_bucket())
        for caller_thread in caller_threads:
            _add_caller_thread(bucket, caller_thread)
    target_cwds = set(buckets)
    if requested_cwd:
        target_cwds.add(requested_cwd)
    for normalized_cwd in target_cwds:
        metadata_for(normalized_cwd)
    implementation_by_cwd = _implementation_costs_for_cwds(
        target_cwds,
        codex_home=codex_home,
        branches_by_cwd={
            normalized_cwd: metadata["branch"]
            for normalized_cwd, metadata in metadata_by_cwd.items()
        },
        excluded_session_ids=_wrapper_session_ids(state_dir),
        caller_threads_by_cwd={
            normalized_cwd: set(
                str(item) for item in bucket.get("caller_threads", set()) if item
            )
            for normalized_cwd, bucket in buckets.items()
        },
    )
    if requested_cwd and requested_cwd not in buckets:
        implementation = implementation_by_cwd.get(
            requested_cwd, _empty_implementation_cost()
        )
        if (
            implementation["implementation_tokens"]
            or implementation["worker_model"] != "-"
        ):
            buckets[requested_cwd] = _new_bucket()
    rows: list[ReviewCostRow] = []
    for normalized_cwd, bucket in buckets.items():
        metadata = metadata_by_cwd.get(normalized_cwd) or _metadata_for_cwd(
            normalized_cwd
        )
        implementation = implementation_by_cwd.get(
            normalized_cwd, _empty_implementation_cost()
        )
        rows.append(
            ReviewCostRow(
                repo=metadata["repo"],
                folder=metadata["folder"],
                branch=metadata["branch"],
                pr_number=metadata["pr_number"],
                worker_model=str(implementation["worker_model"]),
                implementation_tokens=int(implementation["implementation_tokens"]),
                implementation_cost_usd=float(
                    implementation["implementation_cost_usd"]
                ),
                caller_threads=tuple(
                    sorted(
                        str(item)
                        for item in bucket.get("caller_threads", set())
                        if item
                    )
                ),
                latest_review=str(bucket.get("latest_review") or "-"),
                lane_sessions=dict(bucket["lane_sessions"]),
                review_seconds=float(bucket["review_seconds"]),
                tokens=int(bucket["tokens"]),
                cost_usd=round(float(bucket["cost_usd"]), 6),
                review_cwd_normalized=normalized_cwd,
                accounted_run_ids=tuple(sorted(bucket["accounted_run_ids"])),
                review_cost_cutover_at="",
                implementation_costs_refreshed=True,
            )
        )
    return sorted(
        rows, key=lambda row: (row.repo.lower(), row.latest_review), reverse=True
    )


def format_duration(seconds: float) -> str:
    total = int(round(max(0.0, seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def format_compact_number(value: int | float) -> str:
    number = float(value or 0)
    sign = "-" if number < 0 else ""
    number = abs(number)
    if number < 1_000:
        return f"{sign}{int(round(number))}"
    for suffix, factor in (("b", 1_000_000_000), ("m", 1_000_000), ("k", 1_000)):
        if number >= factor:
            scaled = number / factor
            if scaled >= 100:
                text = f"{scaled:.0f}"
            elif scaled >= 10:
                text = f"{scaled:.0f}" if scaled.is_integer() else f"{scaled:.1f}"
            else:
                text = f"{scaled:.1f}"
            if "." in text:
                text = text.rstrip("0").rstrip(".")
            return f"{sign}{text}{suffix}"
    return f"{sign}{int(round(number))}"


def _md_cell(value: object) -> str:
    text = str(value)
    return re.sub(r"\s+", " ", text.replace("|", "\\|")).strip() or "-"


def render_review_cost_markdown(rows: list[ReviewCostRow]) -> str:
    lines = [
        "# Review Cost Ledger",
        "",
        "Generated from local review-suite state. T1/T3/FU columns are reviewer session counts.",
        "",
    ]
    if not rows:
        lines.extend(["No review cost rows found.", ""])
        return "\n".join(lines)
    repos = sorted({row.repo for row in rows}, key=str.lower)
    for repo in repos:
        lines.extend(
            [
                f"# {repo}",
                "",
                "| Date | Folder | Branch | PR | Worker Model | Impl Tokens | Impl Cost | T1 | T3 | FU | Review Time | Review Tokens | Review Cost | Total Cost |",
                "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        repo_rows = sorted(
            [row for row in rows if row.repo == repo],
            key=lambda row: row.latest_review,
            reverse=True,
        )
        for row in repo_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _md_cell(
                            row.latest_review[:10]
                            if row.latest_review and row.latest_review != "-"
                            else "-"
                        ),
                        _md_cell(row.folder),
                        _md_cell(row.branch),
                        _md_cell(row.pr_number),
                        _md_cell(row.worker_model),
                        format_compact_number(row.implementation_tokens),
                        f"${row.implementation_cost_usd:.2f}",
                        str(row.lane_sessions.get("review_t1", 0)),
                        str(row.lane_sessions.get("review_t3", 0)),
                        str(row.lane_sessions.get("review_followup", 0)),
                        _md_cell(format_duration(row.review_seconds)),
                        format_compact_number(row.tokens),
                        f"${row.cost_usd:.2f}",
                        f"${(row.implementation_cost_usd + row.cost_usd):.2f}",
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def write_review_cost_report(*, rows: list[ReviewCostRow], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_review_cost_markdown(rows), encoding="utf-8")
    return output_path


def _cost_row_cache_dir(state_dir: Path) -> Path:
    return state_dir / DEFAULT_COST_CACHE_DIRNAME


def _cost_row_identity(row: ReviewCostRow) -> tuple[str, ...]:
    if row.review_cwd_normalized:
        return (row.review_cwd_normalized, row.branch)
    return (row.repo, row.folder, row.branch)


def _cost_row_cache_key(row: ReviewCostRow) -> str:
    raw = "\n".join(_cost_row_identity(row))
    return blake2s(raw.encode("utf-8"), digest_size=12).hexdigest()


def _cost_row_payload(row: ReviewCostRow) -> dict[str, Any]:
    return {
        "repo": row.repo,
        "folder": row.folder,
        "branch": row.branch,
        "pr_number": row.pr_number,
        "worker_model": row.worker_model,
        "implementation_tokens": row.implementation_tokens,
        "implementation_cost_usd": row.implementation_cost_usd,
        "caller_threads": list(row.caller_threads),
        "latest_review": row.latest_review,
        "lane_sessions": row.lane_sessions,
        "review_seconds": row.review_seconds,
        "tokens": row.tokens,
        "cost_usd": row.cost_usd,
        "review_cwd_normalized": row.review_cwd_normalized,
        "accounted_run_ids": list(row.accounted_run_ids),
        "review_cost_cutover_at": row.review_cost_cutover_at,
        "implementation_costs_refreshed": row.implementation_costs_refreshed,
    }


def _cost_row_from_payload(payload: dict[str, Any]) -> ReviewCostRow | None:
    try:
        folder = str(payload["folder"])
        repo = str(payload["repo"])
        if repo == folder:
            repo = _repo_from_worktree_folder(folder) or repo
        return ReviewCostRow(
            repo=repo,
            folder=folder,
            branch=str(payload["branch"]),
            pr_number=str(payload["pr_number"]),
            worker_model=str(payload["worker_model"]),
            implementation_tokens=int(payload["implementation_tokens"]),
            implementation_cost_usd=float(payload["implementation_cost_usd"]),
            caller_threads=tuple(
                str(item)
                for item in list(payload.get("caller_threads") or [])
                if str(item).strip()
            ),
            latest_review=str(payload["latest_review"]),
            lane_sessions=dict(payload["lane_sessions"]),
            review_seconds=float(payload["review_seconds"]),
            tokens=int(payload["tokens"]),
            cost_usd=float(payload["cost_usd"]),
            review_cwd_normalized=str(payload.get("review_cwd_normalized") or ""),
            accounted_run_ids=tuple(
                str(item)
                for item in list(payload.get("accounted_run_ids") or [])
                if str(item).strip()
            ),
            review_cost_cutover_at=str(payload.get("review_cost_cutover_at") or ""),
            implementation_costs_refreshed=bool(
                payload.get("implementation_costs_refreshed", True)
            ),
        )
    except KeyError, TypeError, ValueError:
        return None


def read_review_cost_row_cache(state_dir: Path) -> list[ReviewCostRow]:
    rows: list[ReviewCostRow] = []
    for path in sorted(_cost_row_cache_dir(state_dir).glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except OSError, json.JSONDecodeError:
            continue
        row = _cost_row_from_payload(payload) if isinstance(payload, dict) else None
        if row is not None:
            rows.append(row)
    return rows


def update_review_cost_row_cache(*, state_dir: Path, rows: list[ReviewCostRow]) -> None:
    cache_dir = _cost_row_cache_dir(state_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    next_keys_by_identity = {
        _cost_row_identity(row): f"{_cost_row_cache_key(row)}.json" for row in rows
    }
    next_keys_by_legacy_identity = {
        (row.repo, row.folder, row.branch): f"{_cost_row_cache_key(row)}.json"
        for row in rows
    }
    for path in sorted(cache_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except OSError, json.JSONDecodeError:
            continue
        row = _cost_row_from_payload(payload) if isinstance(payload, dict) else None
        if row is None:
            continue
        expected_name = next_keys_by_identity.get(
            _cost_row_identity(row)
        ) or next_keys_by_legacy_identity.get((row.repo, row.folder, row.branch))
        if expected_name and path.name != expected_name:
            path.unlink(missing_ok=True)
    for row in rows:
        write_json(
            cache_dir / f"{_cost_row_cache_key(row)}.json", _cost_row_payload(row)
        )


def _cached_review_cost_row(
    *, state_dir: Path, review_cwd: Path
) -> tuple[ReviewCostRow | None, Path, Path, dict[str, str], str, bool]:
    normalized_cwd = str(normalize_review_cwd_value(review_cwd) or "")
    metadata = _cached_metadata_for_cwd(normalized_cwd)
    empty = ReviewCostRow(
        repo=metadata["repo"],
        folder=metadata["folder"],
        branch=metadata["branch"],
        pr_number=metadata["pr_number"],
        worker_model="-",
        implementation_tokens=0,
        implementation_cost_usd=0.0,
        caller_threads=(),
        latest_review="-",
        lane_sessions={lane: 0 for lane in LANES},
        review_seconds=0.0,
        tokens=0,
        cost_usd=0.0,
        review_cwd_normalized=normalized_cwd,
        implementation_costs_refreshed=False,
    )
    cache_dir = _cost_row_cache_dir(state_dir)
    target_path = cache_dir / f"{_cost_row_cache_key(empty)}.json"
    legacy = replace(empty, review_cwd_normalized="")
    legacy_path = cache_dir / f"{_cost_row_cache_key(legacy)}.json"
    for path in (target_path, legacy_path):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except OSError, json.JSONDecodeError:
            return None, target_path, legacy_path, metadata, normalized_cwd, False
        row = _cost_row_from_payload(payload) if isinstance(payload, dict) else None
        return (
            row,
            target_path,
            legacy_path,
            metadata,
            normalized_cwd,
            isinstance(payload, dict) and "accounted_run_ids" not in payload,
        )
    return None, target_path, legacy_path, metadata, normalized_cwd, False


def read_review_cost_row_for_cwd(
    *, state_dir: Path, review_cwd: Path
) -> ReviewCostRow | None:
    row, _target, _legacy, _metadata, _cwd, _is_legacy = _cached_review_cost_row(
        state_dir=state_dir, review_cwd=review_cwd
    )
    return row


def apply_review_cost_delta_best_effort(
    *, state_dir: Path, review_cwd: Path, record: dict[str, Any]
) -> bool:
    try:
        if not bool(dict(load_config(state_dir).get("arena") or {}).get("enabled")):
            return False
        lane = _record_lane(record)
        if lane not in LANES:
            return False
        with orchestrator_store_lock(
            state_dir=state_dir,
            name=REVIEW_COSTS_LOCK_NAME,
            timeout_seconds=AUTOMATIC_COST_DELTA_LOCK_TIMEOUT_SECONDS,
        ):
            (
                row,
                target_path,
                legacy_path,
                metadata,
                normalized_cwd,
                is_legacy,
            ) = _cached_review_cost_row(state_dir=state_dir, review_cwd=review_cwd)
            task_id = _record_task_id(record)
            if task_id and metadata["branch"] != "-" and task_id != metadata["branch"]:
                return False
            if row is None:
                row = ReviewCostRow(
                    repo=metadata["repo"],
                    folder=metadata["folder"],
                    branch=metadata["branch"],
                    pr_number=metadata["pr_number"],
                    worker_model="-",
                    implementation_tokens=0,
                    implementation_cost_usd=0.0,
                    caller_threads=(),
                    latest_review="-",
                    lane_sessions={item: 0 for item in LANES},
                    review_seconds=0.0,
                    tokens=0,
                    cost_usd=0.0,
                    review_cwd_normalized=normalized_cwd,
                    implementation_costs_refreshed=False,
                )
            cutover = row.review_cost_cutover_at
            if is_legacy:
                cutover = row.latest_review if row.latest_review != "-" else ""
            timestamp = _record_timestamp(record)
            if cutover and timestamp and timestamp <= cutover:
                return False
            runs = _record_cost_runs(record)
            seen = set(row.accounted_run_ids)
            identified = [
                (_run_contribution_id(run, record=record, run_index=index), run)
                for index, run in enumerate(runs)
            ]
            new_items = [
                (run_id, run) for run_id, run in identified if run_id not in seen
            ]
            if not new_items:
                return False
            seen.update(run_id for run_id, _run in new_items)
            new_runs = [run for _run_id, run in new_items]
            lane_sessions = dict(row.lane_sessions)
            lane_sessions[lane] = lane_sessions.get(lane, 0) + len(new_runs)
            updated = replace(
                row,
                repo=metadata["repo"],
                folder=metadata["folder"],
                branch=metadata["branch"],
                pr_number=row.pr_number,
                latest_review=max(
                    row.latest_review if row.latest_review != "-" else "", timestamp
                )
                or "-",
                lane_sessions=lane_sessions,
                review_seconds=row.review_seconds + _run_seconds(record, new_runs),
                tokens=row.tokens + sum(_run_tokens(run) for run in new_runs),
                cost_usd=round(
                    row.cost_usd + sum(_run_cost(run) for run in new_runs), 6
                ),
                review_cwd_normalized=normalized_cwd,
                accounted_run_ids=tuple(sorted(seen)),
                review_cost_cutover_at=cutover,
                implementation_costs_refreshed=False,
            )
            write_json(target_path, _cost_row_payload(updated))
            if legacy_path != target_path:
                legacy_path.unlink(missing_ok=True)
            return True
    except Exception:
        return False
