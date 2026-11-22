from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def utc_now_iso() -> str:
    return utc_now().isoformat().replace("+00:00", "Z")


def _windows_single_slash_posix_path(cwd: str) -> str | None:
    if sys.platform != "win32":
        return None
    normalized = str(cwd or "").strip().replace("\\", "/")
    if not normalized.startswith("/") or normalized.startswith("//"):
        return None
    return normalized


def _list_wsl_distributions() -> list[str]:
    if sys.platform != "win32":
        return []
    proc = subprocess.run(
        ["wsl.exe", "-l", "-q"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        return []
    seen: set[str] = set()
    rows: list[str] = []
    for raw_line in (proc.stdout or "").splitlines():
        line = raw_line.replace("\x00", "").strip()
        if not line or line in seen:
            continue
        seen.add(line)
        rows.append(line)
    return rows


def _path_exists(path_text: str) -> bool:
    try:
        return Path(path_text).exists()
    except OSError:
        return False


def _candidate_windows_paths_for_posix_input(
    cwd: str, *, wsl_distributions: tuple[str, ...] | None = None,
) -> list[str]:
    normalized = _windows_single_slash_posix_path(cwd)
    if not normalized:
        return []
    candidates: list[str] = []
    if (
        normalized.startswith("/mnt/")
        and len(normalized) >= 7
        and normalized[5].isalpha()
        and normalized[6] == "/"
    ):
        drive = normalized[5].upper()
        remainder = normalized[7:]
        windows_path = f"{drive}:/{remainder}" if remainder else f"{drive}:/"
        candidates.append(windows_path)
    for distro in _list_wsl_distributions() if wsl_distributions is None else wsl_distributions:
        candidates.append(f"//wsl.localhost/{distro}{normalized}")
    return candidates


def _translate_windows_posix_path(
    cwd: str, *, wsl_distributions: tuple[str, ...] | None = None,
) -> str | None:
    candidates = _candidate_windows_paths_for_posix_input(cwd, wsl_distributions=wsl_distributions)
    if candidates:
        preferred = candidates[0]
        if preferred.lower().startswith(
            tuple(f"{drive}:/" for drive in "abcdefghijklmnopqrstuvwxyz")
        ) and _path_exists(preferred):
            return preferred
    existing = [candidate for candidate in candidates if _path_exists(candidate)]
    unique_existing = list(dict.fromkeys(existing))
    if len(unique_existing) == 1:
        return unique_existing[0]
    return None


def _windows_posix_path_message(cwd: str) -> str | None:
    normalized = _windows_single_slash_posix_path(cwd)
    if not normalized:
        return None
    lines = [
        f"received a POSIX-style repo path on Windows: {cwd}",
        "If this repo lives inside WSL, rerun from Windows with "
        f"`--cd //wsl.localhost/<Distro>{normalized}`.",
    ]
    if (
        normalized.startswith("/mnt/")
        and len(normalized) >= 7
        and normalized[5].isalpha()
        and normalized[6] == "/"
    ):
        drive = normalized[5].upper()
        remainder = normalized[7:]
        windows_path = f"{drive}:/{remainder}" if remainder else f"{drive}:/"
        lines.append(
            f"If this repo actually lives on Windows, use a native Windows path such as `{windows_path}` instead."
        )
    else:
        lines.append(
            "If this repo actually lives on Windows, use a native Windows path such as `C:/Code/your-repo` instead."
        )
    return " ".join(lines)


def _git_top_level(path: Path) -> Path:
    proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or proc.stdout or "").strip()
        raise ValueError(
            f"review-suite requires a git repo root or repo subdirectory: {stderr or 'git rev-parse --show-toplevel failed'}"
        )
    top_level = (proc.stdout or "").strip()
    if not top_level:
        raise ValueError("review-suite could not determine the git top-level directory")
    return Path(top_level)


def resolve_cd_path(
    cd: str | Path, *, wsl_distributions: tuple[str, ...] | None = None,
) -> Path:
    translated = _translate_windows_posix_path(str(cd), wsl_distributions=wsl_distributions)
    if translated:
        return Path(translated)
    path_message = _windows_posix_path_message(str(cd))
    if path_message:
        raise ValueError(path_message)
    return Path(cd).resolve()


def resolve_repo_root(cd: str | None) -> Path:
    if cd:
        return _git_top_level(resolve_cd_path(cd))
    return _git_top_level(Path.cwd().resolve())


def _canonical_wsl_key(distro: str, posix_path: str) -> str | None:
    distro_name = str(distro or "").strip()
    path = str(posix_path or "").strip().replace("\\", "/")
    if not distro_name or not path.startswith("/"):
        return None
    return f"wsl:{distro_name.lower()}:{path.rstrip('/') or '/'}"


def _wsl_unc_key(cwd: str) -> str | None:
    normalized = str(cwd or "").strip().replace("\\", "/")
    lowered = normalized.lower()
    for prefix in ("//wsl.localhost/", "//wsl$/"):
        if not lowered.startswith(prefix):
            continue
        rest = normalized[len(prefix) :]
        distro, sep, posix_path = rest.partition("/")
        if not sep:
            return None
        return _canonical_wsl_key(distro, f"/{posix_path}")
    return None


def _native_wsl_key(cwd: str) -> str | None:
    if sys.platform == "win32":
        return None
    distro = (os.environ.get("WSL_DISTRO_NAME") or "").strip()
    if not distro:
        return None
    normalized = str(cwd or "").strip().replace("\\", "/")
    if not normalized.startswith("/") or normalized.startswith("//"):
        return None
    if (
        len(normalized) >= 7
        and normalized.startswith("/mnt/")
        and normalized[5].isalpha()
        and normalized[6] == "/"
    ):
        return None
    return _canonical_wsl_key(distro, normalized)


def cwd_path_from_normalized(normalized_cwd: str, *, discover_wsl: bool = True) -> Path:
    value = str(normalized_cwd or "").strip()
    if value.lower().startswith("wsl:"):
        _scheme, rest = value.split(":", 1)
        distro, sep, posix_path = rest.partition(":")
        if sep and posix_path.startswith("/"):
            current_distro = (os.environ.get("WSL_DISTRO_NAME") or "").strip().lower()
            if sys.platform != "win32" and current_distro == distro.lower():
                return Path(posix_path)
            if sys.platform == "win32":
                matching_distro = next(
                    (
                        item
                        for item in (_list_wsl_distributions() if discover_wsl else ())
                        if item.lower() == distro.lower()
                    ),
                    distro,
                )
                return Path(f"//wsl.localhost/{matching_distro}{posix_path}")
    return Path(value)


def normalize_cwd(cwd: str) -> str:
    value = str(cwd or "").strip()
    if not value:
        return str(Path(value).resolve())
    if value.lower().startswith("wsl:"):
        _scheme, rest = value.split(":", 1)
        distro, sep, posix_path = rest.partition(":")
        return _canonical_wsl_key(distro, posix_path) if sep else value
    wsl_key = _wsl_unc_key(value) or _native_wsl_key(value)
    if wsl_key:
        return wsl_key
    return str(Path(value).resolve())
