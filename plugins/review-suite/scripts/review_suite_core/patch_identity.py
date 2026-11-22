"""Exact, versioned review patch receipts and current-head acceptance."""

from __future__ import annotations

import codecs
import hashlib
import json
import re
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Any

VERSION = 1


def git(repo: Path, *args: str, input: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", *args], cwd=repo, input=input, capture_output=True, check=False
    )
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


def _read_old_files(repo: Path, base: str, paths: set[str]) -> dict[str, bytes]:
    if not paths:
        return {}
    revisions = {path: f"{base}:{path}" for path in paths}
    requests = "".join(f"{revision}\n" for revision in revisions.values())
    output = BytesIO(git(repo, "cat-file", "--batch", input=requests.encode("utf-8")))
    files = {}
    for path, revision in revisions.items():
        header = output.readline()
        if header.endswith((b" missing\n", b" ambiguous\n")):
            continue
        _, kind, size = header.split()
        content = output.read(int(size))
        output.read(1)  # Batch output adds one newline after each object's bytes.
        if kind == b"blob":
            files[path] = content
        else:
            # A gitlink can resolve to a commit already present in this repository.
            try:
                files[path] = git(repo, "show", revision)
            except ValueError:
                pass
    return files


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
    canonical = [
        b" ".join(item.split(b" ")[:2])
        for item in raw.split(b"\0")
        if item.startswith(b":")
    ]

    ambiguous = False
    sections = []
    for section in re.split(rb"(?m)^diff --git ", patch)[1:]:
        lines = section.splitlines(keepends=True)
        binary = b"GIT binary patch\n" in lines
        # A repeated deletion or insertion boundary cannot identify where an edit belongs.
        # Retain blob identities in that case, so base movement cannot silently relocate it.
        raw_path = next(
            (line[6:].rstrip(b"\n") for line in lines if line.startswith(b"--- a/")),
            None,
        )
        if not binary and any(line.startswith(b'--- "') for line in lines):
            ambiguous = True
        old_path = None
        if raw_path and not raw_path.startswith(b'"'):
            try:
                old_path = raw_path.decode("utf-8")
            except UnicodeError:
                ambiguous = True
        sections.append((lines, binary, old_path))

    old_files = _read_old_files(repo, base, {path for _, _, path in sections if path})
    for lines, binary, old_path in sections:
        old_lines = []
        if old_path:
            if old_path in old_files:
                old_lines = old_files[old_path].splitlines(keepends=True)
            else:
                ambiguous = True
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
        canonical.append(b"".join(normalized))
    if ambiguous:
        canonical.append(
            git(repo, "diff", "--raw", "--no-abbrev", "--no-renames", base, head)
        )
    digest = hashlib.sha256(
        b"".join(len(part).to_bytes(8, "big") + part for part in canonical)
    ).hexdigest()
    return {
        "version": VERSION,
        "digest": digest,
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


def _changed_paths(repo: Path, left: str, right: str, *options: str) -> set[bytes]:
    paths = git(repo, "diff", *options, "--name-only", "-z", left, right)
    return set(paths.split(b"\0")) - {b""}


def _patches_by_path(
    repo: Path, left: str, right: str, paths: set[bytes]
) -> dict[bytes, bytes]:
    if not paths:
        return {}
    patch = git(
        repo,
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
        "--no-color",
        "--no-prefix",
        "--submodule=short",
        "--unified=0",
        left,
        right,
    )
    patches = {}
    for section in re.split(rb"(?m)^diff --git ", patch)[1:]:
        names, _, body = section.partition(b"\n")
        # With no prefixes or renames, Git prints the same name twice. The
        # midpoint works for names containing spaces and for C-quoted names.
        path = names[: len(names) // 2]
        if path.startswith(b'"'):
            path = codecs.escape_decode(path[1:-1])[0]
        if path in paths:
            # A file type change can emit separate deletion and addition sections.
            patches[path] = patches.get(path, b"") + body
    return patches


def _has_moved_lines(patch: bytes) -> bool:
    removed = set()
    added = set()
    for line in patch.splitlines():
        if line.startswith(b"-") and not line.startswith(b"---"):
            removed.add(line[1:])
        if line.startswith(b"+") and not line.startswith(b"+++"):
            added.add(line[1:])
    return bool(removed & added)


def acceptance(state: dict[str, Any], current_head: str) -> dict[str, Any] | None:
    try:
        receipt = reviewed_receipt(state)
        if not receipt or receipt["reviewed_base"] == receipt["reviewed_head"]:
            return None
        repo = Path(receipt["repo"])
        base = git(repo, "merge-base", receipt["base"], current_head).decode().strip()
        reviewed_base = receipt["reviewed_base"]
        reviewed_head = receipt["reviewed_head"]
        # The stored receipt already describes this exact pair of commits.
        if (base, current_head) != (reviewed_base, reviewed_head):
            if patch_identity(repo, base, current_head) != receipt["patch"]:
                return None
        if base != reviewed_base:
            # Identical edit bytes cannot prove location across a base-side move.
            branch_paths = _changed_paths(repo, reviewed_base, reviewed_head)
            drift_paths = _changed_paths(repo, reviewed_base, base, "--no-renames")
            patches = _patches_by_path(repo, reviewed_base, base, branch_paths & drift_paths)
            if any(_has_moved_lines(patch) for patch in patches.values()):
                return None
        return {**receipt, "accepted_head": current_head, "accepted_base": base}
    except OSError, ValueError:
        return None


def _intervals(patch: bytes) -> list[tuple[int, int]]:
    result = []
    for match in re.finditer(rb"^@@ -(\d+)(?:,(\d+))? ", patch, re.MULTILINE):
        start = int(match[1])
        count = int(match[2]) if match[2] is not None else 1
        # Insertion boundaries touch both neighboring edits.
        result.append((start, start + count))
    return sorted(result)


def _line_overlap(left: bytes, right: bytes) -> bool:
    left_intervals = iter(_intervals(left))
    right_intervals = iter(_intervals(right))
    left_interval = next(left_intervals, None)
    right_interval = next(right_intervals, None)
    while left_interval is not None and right_interval is not None:
        if left_interval[1] < right_interval[0]:
            left_interval = next(left_intervals, None)
        elif right_interval[1] < left_interval[0]:
            right_interval = next(right_intervals, None)
        else:
            return True
    return False


def interaction(repo: Path, old_base: str, new_base: str, head: str) -> dict[str, Any]:
    drift = _changed_paths(repo, old_base, new_base, "--no-renames")
    branch = _changed_paths(repo, old_base, head, "--no-renames")
    overlap = drift & branch
    raw = git(repo, "diff", "--no-renames", "--numstat", old_base, new_base)
    branch_summary = git(repo, "diff", "--no-renames", "--summary", old_base, head)
    structural = b"-\t-\t" in raw or any(
        line.lstrip().startswith(
            (b"delete mode", b"create mode", b"old mode", b"new mode")
        )
        for line in (
            git(repo, "diff", "--no-renames", "--summary", old_base, new_base)
            + branch_summary
        ).splitlines()
    )

    base_patches = _patches_by_path(repo, old_base, new_base, overlap)
    branch_patches = _patches_by_path(repo, old_base, head, overlap)
    ordered_paths = sorted(overlap)
    line_overlap = [
        path.decode("utf-8", "replace")
        for path in ordered_paths
        if _line_overlap(base_patches.get(path, b""), branch_patches.get(path, b""))
    ]
    return {
        "check": bool(overlap or structural),
        "line_overlap_paths": line_overlap,
        "reason": "structural_or_unknown"
        if structural
        else "shared_paths"
        if overlap
        else "inspect_dependencies",
        "paths": [p.decode("utf-8", "replace") for p in ordered_paths],
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


def bound_interaction(
    state: dict[str, Any], accepted: dict[str, Any]
) -> dict[str, Any]:
    result = ({"check": False, "reason": "no_base_drift", "paths": [], "line_overlap_paths": []}
              if accepted["reviewed_base"] == accepted["accepted_base"] else interaction(
                  Path(accepted["repo"]), accepted["reviewed_base"],
                  accepted["accepted_base"], accepted["reviewed_head"]))
    binding = {
        "head": accepted["accepted_head"],
        "base": accepted["accepted_base"],
        "patch": accepted["patch"],
    }
    receipt = dict(state.get("base_interaction_receipt") or {})
    if receipt.get("binding") == binding:
        result.update(check=False, status=receipt["status"])
        if receipt["status"] == "overridden":
            result["override"] = {
                "reason": receipt["reason"],
                "head": binding["head"],
                "base": binding["base"],
            }
        else:
            result["receipt"] = receipt
    else:
        result["status"] = "outstanding" if result["check"] else "not_required"
    return result


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
