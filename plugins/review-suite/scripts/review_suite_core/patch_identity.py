"""Exact, versioned review patch receipts and current-head acceptance."""

from __future__ import annotations

import codecs
import hashlib
import json
import re
import subprocess
from bisect import bisect_right
from collections.abc import Iterable
from contextlib import ExitStack, closing, contextmanager
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Any, BinaryIO, Iterator

from .cycle_status import PATCH_RECEIPT_VERSION as VERSION, _no_base_drift, bound_interaction as bound_interaction
from .git_reads import cached_git_read
from .workflow_state import merge_base

PATHSPEC_BUDGET = 24_000  # Leave room for Git options under Windows' command-line limit.
HUNK_BATCH_BUDGET = 32_000  # Limit retained interval data when many shared lines change.


def git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, check=False)
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8", "replace"))
    return result.stdout


def alignment_boundary(lines: list[bytes], start: int, count: int) -> bytes | None:
    """Find context that identifies the edit even without its removed statement."""
    radius = 1
    while radius <= len(lines) * 2:
        before = lines[max(0, start - radius) : start]
        after = lines[start + count : start + count + radius]
        locations = set()
        for at in range(len(before), len(lines) + 1):
            if lines[at - len(before) : at] != before:
                continue
            for gap in range(count + 1):
                if (
                    at + gap + len(after) <= len(lines)
                    and lines[at + gap : at + gap + len(after)] == after
                ):
                    locations.add(at)
                    break
        if locations == {start}:
            left, right = b"".join(before), b"".join(after)
            return (
                len(left).to_bytes(8, "big")
                + left
                + len(right).to_bytes(8, "big")
                + right
            )
        radius *= 2
    return None


@contextmanager
def _git_stream(
    repo: Path, *args: str, requests: bool = False
) -> Iterator[subprocess.Popen[bytes]]:
    with subprocess.Popen(
        ["git", *args],
        cwd=repo,
        stdin=subprocess.PIPE if requests else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ) as process:
        yield process
        if process.stdin is not None:
            process.stdin.close()
        if returncode := process.wait():
            raise ValueError(f"Git stream exited with status {returncode}")


def _read_old_file(
    repo: Path, revision: str, request: BinaryIO, output: BinaryIO
) -> bytes | None:
    # Git flushes each reply in --batch mode. Alternate requests and replies so
    # neither pipe needs to hold more than one file's data.
    request.write(f"{revision}\n".encode("utf-8"))
    request.flush()
    header = output.readline()
    if header.endswith((b" missing\n", b" ambiguous\n")):
        return None
    _, kind, size = header.split()
    content = output.read(int(size))
    output.read(1)  # Batch output adds one newline after each object's bytes.
    if kind == b"blob":
        return content
    # A gitlink can resolve to a commit already present in this repository.
    try:
        return git(repo, "show", revision)
    except ValueError:
        return None


def _patch_sections(patch: bytes) -> Iterator[bytes]:
    start = None
    for header in re.finditer(rb"^diff --git ", patch, re.MULTILINE):
        if start is not None:
            yield patch[start:header.start()]
        start = header.end()
    if start is not None:
        yield patch[start:]


@cached_git_read
def patch_identity(repo: Path, base: str, head: str) -> dict[str, Any]:
    # Byte output avoids platform newline conversion and preserves whitespace.
    patch = git(
        repo,
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
        "--binary",
        "--full-index",
        "--unified=0",
        base,
        head,
    )
    raw = git(repo, "diff", "--raw", "--no-abbrev", "--no-renames", "-z", base, head)
    digest = hashlib.sha256()

    def add_part(part: bytes) -> None:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)

    for item in raw.split(b"\0"):
        if item.startswith(b":"):
            add_part(b" ".join(item.split(b" ")[:2]))

    ambiguous = False
    with ExitStack() as resources:
        batch = None
        for section in _patch_sections(patch):
            lines = section.splitlines(keepends=True)
            binary = b"GIT binary patch\n" in lines
            # A repeated deletion or insertion boundary cannot identify where an edit belongs.
            # Retain blob identities in that case, so base movement cannot silently relocate it.
            old_path = next(
                (line[6:].rstrip(b"\n") for line in lines if line.startswith(b"--- a/")),
                None,
            )
            if not binary and any(line.startswith(b'--- "') for line in lines):
                ambiguous = True
            old_lines = []
            if old_path and not old_path.startswith(b'"'):
                try:
                    revision = f"{base}:{old_path.decode('utf-8')}"
                except UnicodeError:
                    ambiguous = True
                else:
                    if batch is None:
                        batch = resources.enter_context(_git_stream(repo, "cat-file", "--batch", requests=True))
                    assert batch.stdin is not None and batch.stdout is not None
                    old_file = _read_old_file(repo, revision, batch.stdin, batch.stdout)
                    if old_file is None:
                        ambiguous = True
                    else:
                        old_lines = old_file.splitlines(keepends=True)
            normalized = []
            for line in lines:
                if line.startswith(b"index ") and not binary:
                    continue
                if line.startswith(b"@@ "):
                    # Preserve edit boundaries/counts, excluding offsets and function context.
                    match = re.match(rb"@@ -\d+(,\d+)? \+\d+(,\d+)? @@", line)
                    if not match:
                        raise ValueError("unrecognized patch hunk")
                    location = re.match(rb"@@ -(\d+)(?:,(\d+))?", line)
                    start = int(location[1])
                    count = int(location[2]) if location[2] is not None else 1
                    if old_lines:
                        boundary = alignment_boundary(
                            old_lines, start - 1 if count else start, count
                        )
                        if boundary is None:
                            ambiguous = True
                        else:
                            normalized.append(
                                b"boundary:" + hashlib.sha256(boundary).digest()
                            )
                    line = (
                        b"@@ -"
                        + (match[1] or b",1")
                        + b" +"
                        + (match[2] or b",1")
                        + b" @@\n"
                    )
                normalized.append(line)
            add_part(b"".join(normalized))
    if ambiguous:
        add_part(
            git(repo, "diff", "--raw", "--no-abbrev", "--no-renames", base, head)
        )
    return {
        "version": VERSION,
        "digest": digest.hexdigest(),
        "alignment": "unknown" if ambiguous else "unique",
    }


def reviewed_receipt(state: dict[str, Any]) -> dict[str, Any] | None:
    identity = dict(state.get("identity") or {})
    heads = dict(state.get("review_heads") or {})
    head = (
        heads.get("last_reviewed_head")
        or heads.get("last_followup_head")
        or identity.get("head")
    )
    base = identity.get("merge_base")
    if not identity.get("cwd") or not head or not base:
        return None
    repo = Path(identity["cwd"])
    existing = dict(state.get("review_patch") or {})
    if existing.get("reviewed_head") == head:
        if existing.get("repo") != str(repo.resolve()) or existing.get(
            "base"
        ) != identity.get("base"):
            return None
        if dict(existing.get("patch") or {}).get("version") != VERSION:
            return None
        return existing
    return {
        "repo": str(repo.resolve()),
        "base": identity.get("base"),
        "reviewed_head": head,
        "reviewed_base": base,
        "patch": patch_identity(repo, base, head),
    }


@cached_git_read
def _changed_paths(repo: Path, left: str, right: str) -> set[bytes]:
    paths = git(repo, "diff", "--name-only", "-z", left, right)
    return set(paths.split(b"\0")) - {b""}


def _path_batches(
    paths: Iterable[bytes], hunk_estimates: dict[bytes, int] | None = None
) -> Iterator[dict[bytes, str]]:
    batch: dict[bytes, str] = {}
    size = 0
    hunks = 0
    for raw_path in sorted(paths):
        path = raw_path.decode("utf-8", "surrogateescape")
        quoted = subprocess.list2cmdline([path]).encode("utf-8", "surrogateescape")
        cost = len(quoted) + 1
        hunk_cost = hunk_estimates.get(raw_path, 0) if hunk_estimates else 0
        if batch and (size + cost > PATHSPEC_BUDGET or hunks + hunk_cost > HUNK_BATCH_BUDGET):
            yield batch
            batch = {}
            size = 0
            hunks = 0
        batch[raw_path] = path
        size += cost
        hunks += hunk_cost
    if batch:
        yield batch


def _patch_lines(
    repo: Path, left: str, right: str, batch: dict[bytes, str], *, include_edits: bool = False
) -> Iterator[tuple[set[bytes], bytes]]:
    with _git_stream(
        repo,
        "--literal-pathspecs",
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
        "--no-color",
        "--no-prefix",
        "--unified=0",
        left,
        right,
        "--",
        *batch.values(),
    ) as process:
        assert process.stdout is not None
        paths: set[bytes] = set()
        in_hunk = False
        for line in process.stdout:
            if line.startswith((b"diff --git ", b"Submodule ")):
                paths = set()
                in_hunk = False
            elif line.startswith(b"@@ "):
                in_hunk = True
                if paths:
                    yield paths, line
            elif in_hunk:
                if paths and include_edits:
                    yield paths, line
            elif line.startswith((b"--- ", b"+++ ")):
                path = line[4:].removesuffix(b"\n").removesuffix(b"\t")
                if path.startswith(b'"'):
                    path = codecs.escape_decode(path[1:-1])[0]
                # An inline submodule diff reports paths under the selected gitlink.
                while path and path != b"/dev/null":
                    if path in batch:
                        paths.add(path)
                    path = path.rpartition(b"/")[0]


def _has_moved_lines(lines: Iterator[bytes]) -> bool:
    removed: set[bytes] = set()
    added: set[bytes] = set()
    for line in lines:
        for edit in line.splitlines():
            if edit.startswith(b"-") and not edit.startswith(b"---"):
                own, other = removed, added
            elif edit.startswith(b"+") and not edit.startswith(b"+++"):
                own, other = added, removed
            else:
                continue
            content = edit[1:]
            if content in other:
                return True
            own.add(content)
    return False


@cached_git_read
def _base_moved_lines(repo: Path, left: str, right: str, paths: frozenset[bytes]) -> bool:
    for batch in _path_batches(paths):
        with closing(_patch_lines(repo, left, right, batch, include_edits=True)) as lines:
            # Equivalent patches preserve file types and existence. Each shared
            # path's edits are contiguous, including hunks from an inline submodule.
            for _, file_lines in groupby(lines, key=itemgetter(0)):
                if _has_moved_lines(line for _, line in file_lines):
                    return True
    return False


def acceptance(state: dict[str, Any], current_head: str) -> dict[str, Any] | None:
    try:
        receipt = reviewed_receipt(state)
        if not receipt or receipt["reviewed_base"] == receipt["reviewed_head"]:
            return None
        repo = Path(receipt["repo"])
        base = merge_base(repo, receipt["base"], current_head)
        reviewed_base = receipt["reviewed_base"]
        reviewed_head = receipt["reviewed_head"]
        # The stored receipt already describes this exact pair of commits.
        if (base, current_head) != (reviewed_base, reviewed_head):
            if patch_identity(repo, base, current_head) != receipt["patch"]:
                return None
        if base != reviewed_base:
            # Identical edit bytes cannot prove location across a base-side move.
            branch_paths = _changed_paths(repo, reviewed_base, reviewed_head)
            drift_paths, _ = _paths_and_structure(repo, reviewed_base, base, numstat=True)
            if _base_moved_lines(repo, reviewed_base, base, drift_paths & branch_paths):
                return None
        return {**receipt, "accepted_head": current_head, "accepted_base": base}
    except OSError, ValueError:
        return None


def _hunk_intervals(
    repo: Path, left: str, right: str, batch: dict[bytes, str]
) -> Iterator[tuple[set[bytes], int, int]]:
    with closing(_patch_lines(repo, left, right, batch)) as lines:
        for edited_paths, header in lines:
            match = re.match(rb"^@@ -(\d+)(?:,(\d+))? ", header)
            if match is None:
                continue
            start = int(match[1])
            count = int(match[2]) if match[2] is not None else 1
            # Insertion boundaries touch both neighboring edits.
            yield edited_paths, start, start + count


def _intervals_by_path(
    repo: Path, left: str, right: str, batch: dict[bytes, str]
) -> dict[bytes, list[tuple[int, int]]]:
    intervals: dict[bytes, list[tuple[int, int]]] = {}
    for edited_paths, start, end in _hunk_intervals(repo, left, right, batch):
        for path in edited_paths:
            intervals.setdefault(path, []).append((start, end))
    for ranges in intervals.values():
        ranges.sort()
        # Keep the furthest end so a binary search also works for nested ranges.
        max_end = 0
        for index, (start, end) in enumerate(ranges):
            max_end = max(max_end, end)
            if max_end != end:
                ranges[index] = (start, max_end)
    return intervals


def _line_overlap_paths(
    repo: Path, old_base: str, new_base: str, head: str, batch: dict[bytes, str]
) -> set[bytes]:
    overlaps = set()
    branch_intervals = _intervals_by_path(repo, old_base, head, batch)
    if not branch_intervals:
        return overlaps
    with closing(_hunk_intervals(repo, old_base, new_base, batch)) as base_intervals:
        for paths, start, end in base_intervals:
            for path in paths:
                if ranges := branch_intervals.get(path):
                    index = bisect_right(ranges, end, key=itemgetter(0))
                    if index and ranges[index - 1][1] >= start:
                        overlaps.add(path)
                        del branch_intervals[path]
            if not branch_intervals:
                break
    return overlaps


def _hunk_estimates(repo: Path, left: str, right: str, batch: dict[bytes, str]) -> dict[bytes, int]:
    numstat = git(
        repo, "--literal-pathspecs", "diff", "--no-renames", "--no-ext-diff", "--no-textconv",
        "--numstat", "-z", left, right, "--", *batch.values(),
    )
    estimates = {}
    for entry in filter(None, numstat.split(b"\0")):
        added, deleted, path = entry.split(b"\t", 2)
        if added != b"-":
            # Each text hunk adds or removes at least one line.
            cost = int(added) + int(deleted)
            while path:
                if path in batch:
                    estimates[path] = estimates.get(path, 0) + cost
                path = path.rpartition(b"/")[0]
    return estimates


@cached_git_read
def _paths_and_structure(
    repo: Path, left: str, right: str, *, numstat: bool,
) -> tuple[frozenset[bytes], bool]:
    # Git terminates path records with NUL and appends the summary as text.
    output = git(
        repo, "diff", "--no-renames", "--numstat" if numstat else "--raw",
        "--summary", "-z", left, right,
    )
    records, _, summary = output.rpartition(b"\0")
    fields = records.split(b"\0")
    if numstat:
        rows = [row.split(b"\t", 2) for row in fields if row]
        paths = frozenset(path for _, _, path in rows)
        binary = any(added == deleted == b"-" for added, deleted, _ in rows)
    else:
        paths = frozenset(fields[1::2])
        binary = False
    structural = binary or any(
        line.lstrip().startswith(
            (b"delete mode", b"create mode", b"old mode", b"new mode")
        )
        for line in summary.splitlines()
    )
    return paths, structural


@cached_git_read
def interaction(repo: Path, old_base: str, new_base: str, head: str) -> dict[str, Any]:
    drift, base_structural = _paths_and_structure(repo, old_base, new_base, numstat=True)
    branch, branch_structural = _paths_and_structure(repo, old_base, head, numstat=False)
    overlap = drift & branch
    structural = base_structural or branch_structural
    line_overlap = set()
    for batch in _path_batches(overlap):
        estimates = _hunk_estimates(repo, old_base, head, batch) if len(batch) > 1 else None
        for bounded_batch in _path_batches(set(batch), estimates):
            line_overlap.update(_line_overlap_paths(repo, old_base, new_base, head, bounded_batch))
    return {
        "check": bool(overlap or structural),
        "line_overlap_paths": [p.decode("utf-8", "replace") for p in sorted(line_overlap)],
        "reason": "structural_or_unknown"
        if structural
        else "shared_paths"
        if overlap
        else "inspect_dependencies",
        "paths": [p.decode("utf-8", "replace") for p in sorted(overlap)],
    }


def delta_prompt(old_base: str, old_head: str, new_base: str, new_head: str) -> str:
    return (
        "Review only changes to the PR patch since its previous review. "
        f"Compare `git diff --binary {old_base} {old_head}` with "
        f"`git diff --binary {new_base} {new_head}`. "
        "Inspect differing edits and their reachable effects; do not restart the whole PR review. "
        f"Check concrete interactions with base changes using `git diff {old_base} {new_base}`. "
        "Renames, moves, binaries, deletions and ambiguous alignment need a targeted check. "
        "Preserve prior findings. Report only material correctness defects with a reachable trigger. "
        "End with `Review result: clean` or `Review result: findings`."
    )


def _interaction_scope(accepted: dict[str, Any]) -> dict[str, Any]:
    if accepted["reviewed_base"] == accepted["accepted_base"]:
        return _no_base_drift()
    return interaction(
        Path(accepted["repo"]), accepted["reviewed_base"],
        accepted["accepted_base"], accepted["reviewed_head"],
    )


def record_acceptance(state: dict[str, Any], accepted: dict[str, Any]) -> dict[str, Any]:
    if state.get("accepted_review") == accepted and "accepted_review_scope" in state:
        return state
    github_head = (state.get("github_review") or {}).get("reviewed_head")
    github_acceptance = (
        acceptance(state, github_head)
        if github_head and github_head not in {accepted["reviewed_head"], accepted["accepted_head"]}
        else None
    )
    return {
        **state,
        "accepted_review": accepted,
        "accepted_review_scope": _interaction_scope(accepted),
        "accepted_review_github_head": github_head if github_acceptance else None,
    }


def interaction_prompt(accepted: dict[str, Any], scope: dict[str, Any]) -> str:
    repo = Path(accepted["repo"])
    old, base, head = (
        accepted["reviewed_base"],
        accepted["accepted_base"],
        accepted["accepted_head"],
    )
    paths = scope["paths"]
    base_paths = (
        paths
        if scope["reason"] == "shared_paths"
        else [
            p.decode("utf-8", "surrogateescape")
            for p in git(repo, "diff", "--name-only", "-z", old, base).split(b"\0")
            if p
        ]
    )
    base_command = json.dumps(["git", "diff", "--no-ext-diff", "--no-textconv", old, base, "--", *base_paths])
    pr_command = json.dumps(["git", "diff", "--no-ext-diff", "--no-textconv", base, head, "--", *paths]) if paths else "No shared-path hunks; trace affected PR call sites."
    return (
        "Review only interactions between the flagged base changes and the PR code that uses or touches them. "
        "Never review the full PR. Inspect affected call sites and dependencies as needed, and only their "
        "relevant changed hunks. For structural changes trace callers of removed, moved or changed contracts. "
        "Read these diffs in bounded sections, focusing on the flagged contracts and their callers. "
        "Execute the supplied Git argument vectors directly without a shell; never interpolate repository paths into shell commands. "
        "Report only material correctness defects with a reachable trigger. "
        f"Current head: {head}; base: {base}. Flag reason: {scope['reason']}.\n"
        f"Flagged base diff command: {base_command}\n"
        f"Shared-path PR hunks command: {pr_command}\n"
        "End with `Review result: clean` or `Review result: findings`."
    )
