"""Exact, versioned review patch receipts and current-head acceptance."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Any

VERSION = 1


def git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, check=False)
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8", "replace"))
    return result.stdout


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

    binary = False
    ambiguous = False
    for section in re.split(rb"(?m)^diff --git ", patch)[1:]:
        lines = section.splitlines(keepends=True)
        binary = b"GIT binary patch\n" in lines
        normalized = []
        # A repeated deletion or insertion boundary cannot identify where an edit belongs.
        # Retain blob identities in that case, so base movement cannot silently relocate it.
        old_path = next(
            (line[6:].rstrip(b"\n") for line in lines if line.startswith(b"--- a/")),
            None,
        )
        old_lines = []
        if not binary and any(line.startswith(b'--- "') for line in lines):
            ambiguous = True
        if old_path and not old_path.startswith(b'"'):
            try:
                old_lines = git(
                    repo, "show", f"{base}:{old_path.decode('utf-8')}"
                ).splitlines(keepends=True)
            except UnicodeError, ValueError:
                ambiguous = True
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
                    if count:
                        removed = old_lines[start - 1 : start - 1 + count]
                        matches = sum(
                            old_lines[i : i + count] == removed
                            for i in range(len(old_lines) - count + 1)
                        )
                        ambiguous |= matches != 1
                        boundary = (
                            old_lines[max(0, start - 2) : start - 1]
                            + old_lines[start - 1 + count : start + count]
                        )
                        normalized.append(
                            b"boundary:" + hashlib.sha256(b"".join(boundary)).digest()
                        )
                    else:
                        boundary = old_lines[max(0, start - 1) : start + 1]
                        matches = sum(
                            old_lines[i : i + len(boundary)] == boundary
                            for i in range(len(old_lines) - len(boundary) + 1)
                        )
                        ambiguous |= matches != 1
                        # Bind insertions to a unique boundary rather than just inserted bytes.
                        normalized.append(
                            b"boundary:" + hashlib.sha256(b"".join(boundary)).digest()
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


def acceptance(state: dict[str, Any], current_head: str) -> dict[str, Any] | None:
    try:
        receipt = reviewed_receipt(state)
        if not receipt or receipt["reviewed_base"] == receipt["reviewed_head"]:
            return None
        repo = Path(receipt["repo"])
        base = git(repo, "merge-base", receipt["base"], current_head).decode().strip()
        patch = patch_identity(repo, base, current_head)
        if patch != receipt["patch"]:
            return None
        return {**receipt, "accepted_head": current_head, "accepted_base": base}
    except OSError, ValueError:
        return None


def interaction(repo: Path, old_base: str, new_base: str, head: str) -> dict[str, Any]:
    drift = git(
        repo, "diff", "--no-renames", "--name-only", "-z", old_base, new_base
    ).split(b"\0")
    branch = git(
        repo, "diff", "--no-renames", "--name-only", "-z", old_base, head
    ).split(b"\0")
    overlap = sorted(set(drift) & set(branch) - {b""})
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

    def intervals(left: str, right: str, path: str) -> list[tuple[int, int]]:
        patch = git(
            repo,
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--unified=0",
            left,
            right,
            "--",
            path,
        )
        result = []
        for match in re.finditer(rb"^@@ -(\d+)(?:,(\d+))? ", patch, re.MULTILINE):
            start = int(match[1])
            count = int(match[2]) if match[2] is not None else 1
            # Insertion boundaries touch both neighboring edits.
            result.append((start, start + count))
        return result

    line_overlap = []
    for raw_path in overlap:
        path = raw_path.decode("utf-8", "replace")
        if any(
            a <= d and c <= b
            for a, b in intervals(old_base, new_base, path)
            for c, d in intervals(old_base, head, path)
        ):
            line_overlap.append(path)
    return {
        "check": bool(overlap or structural),
        "line_overlap_paths": line_overlap,
        "reason": "structural_or_unknown"
        if structural
        else "shared_paths"
        if overlap
        else "inspect_dependencies",
        "paths": [p.decode("utf-8", "replace") for p in overlap],
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
