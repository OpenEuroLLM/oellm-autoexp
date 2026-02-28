from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
from collections.abc import Iterable, Sequence


DEFAULT_EXCLUDE_PATTERNS = (r".*\\.hash$",)
DEFAULT_HASH_ALGORITHM = "md5"
ALGORITHM_HEADER_PREFIX = "# hash-algorithm: "


class IntegrityError(RuntimeError):
    """Raised when integrity verification fails."""


@dataclass(frozen=True)
class IntegrityReport:
    mode: str
    hash_file: Path
    checked_files: int
    hash_algorithm: str


def check_integrity(
    root: str | Path,
    targets: Sequence[str | Path] | None = None,
    hash_file: str | Path | None = None,
    exclude_patterns: Sequence[str] | None = None,
    force_hash: bool = False,
    hash_algorithm: str = DEFAULT_HASH_ALGORITHM,
) -> IntegrityReport:
    """Check integrity for a directory using mtime + optional full hash
    verification.

    Workflow:
    - If hash file does not exist, create it and return mode='created'.
    - If force_hash is set, run a full hash check.
    - Else if hash file is older than any tracked data file, run a full hash check.
      On success, rewrite hash file and return mode='hash'.
      On failure, raise IntegrityError.
    - Otherwise, return mode='mtime'.
    """
    root_path = Path(root).resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Root path does not exist: {root_path}")

    algorithm = _normalize_algorithm(hash_algorithm)

    patterns = list(DEFAULT_EXCLUDE_PATTERNS)
    if exclude_patterns:
        patterns.extend(exclude_patterns)
    regexes = [re.compile(p) for p in patterns]

    hash_path = _resolve_hash_path(root_path, hash_file)
    files = _collect_files(root_path, targets, regexes, hash_path)

    if not hash_path.exists():
        _write_hash_file(hash_path, root_path, files, algorithm)
        return IntegrityReport(
            mode="created",
            hash_file=hash_path,
            checked_files=len(files),
            hash_algorithm=algorithm,
        )

    hash_mtime = hash_path.stat().st_mtime
    newest_data_mtime = max((p.stat().st_mtime for p in files), default=0.0)

    if force_hash or hash_mtime < newest_data_mtime:
        expected_algorithm, expected = _read_hash_file(hash_path)
        if expected_algorithm != algorithm:
            raise IntegrityError(
                f"Hash algorithm mismatch: hash file uses '{expected_algorithm}', requested '{algorithm}'"
            )

        actual_keys = {_to_tracking_path(root_path, p) for p in files}
        if set(expected.keys()) != actual_keys:
            missing = sorted(set(expected.keys()) - actual_keys)
            added = sorted(actual_keys - set(expected.keys()))
            details = []
            if missing:
                details.append(f"missing files: {missing}")
            if added:
                details.append(f"new files: {added}")
            raise IntegrityError("Tracked file list changed; " + "; ".join(details))

        mismatches = []
        for file_path in files:
            rel = _to_tracking_path(root_path, file_path)
            digest = _file_hash(file_path, algorithm)
            if expected[rel] != digest:
                mismatches.append(rel)

        if mismatches:
            raise IntegrityError(f"Hash mismatch for files: {mismatches}")

        _write_hash_file(hash_path, root_path, files, algorithm)
        return IntegrityReport(
            mode="hash",
            hash_file=hash_path,
            checked_files=len(files),
            hash_algorithm=algorithm,
        )

    return IntegrityReport(
        mode="mtime",
        hash_file=hash_path,
        checked_files=len(files),
        hash_algorithm=algorithm,
    )


def _normalize_algorithm(name: str) -> str:
    normalized = name.lower().replace("-", "")
    try:
        hashlib.new(normalized)
    except ValueError as exc:
        raise IntegrityError(f"Unsupported hash algorithm: {name}") from exc
    return normalized


def _resolve_hash_path(root: Path, hash_file: str | Path | None) -> Path:
    if hash_file is None:
        return root / "checksums.hash"
    path = Path(hash_file)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _collect_files(
    root: Path,
    targets: Sequence[str | Path] | None,
    exclude_regexes: Sequence[re.Pattern[str]],
    hash_path: Path,
) -> list[Path]:
    selected: list[Path] = []

    if not targets:
        candidates = [root]
    else:
        candidates = []
        for item in targets:
            p = Path(item)
            if not p.is_absolute():
                p = root / p
            candidates.append(p)

    for candidate in candidates:
        if not candidate.exists():
            raise FileNotFoundError(f"Target does not exist: {candidate}")

        if candidate.is_file():
            selected.append(candidate.resolve())
            continue

        for p in candidate.rglob("*"):
            if p.is_file():
                selected.append(p.resolve())

    unique: dict[str, Path] = {}
    for path in selected:
        if path == hash_path:
            continue
        rel = _to_tracking_path(root, path)
        if any(r.search(rel) for r in exclude_regexes):
            continue
        unique[rel] = path

    return [unique[k] for k in sorted(unique.keys())]


def _file_hash(path: Path, algorithm: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _write_hash_file(
    hash_path: Path,
    root: Path,
    files: Iterable[Path],
    algorithm: str,
) -> None:
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{ALGORITHM_HEADER_PREFIX}{algorithm}\n"]
    for file_path in files:
        rel = _to_tracking_path(root, file_path)
        lines.append(f"{_file_hash(file_path, algorithm)} *{rel}\n")
    hash_path.write_text("".join(lines), encoding="utf-8")


def _to_tracking_path(root: Path, file_path: Path) -> str:
    return Path(os.path.relpath(file_path, root)).as_posix()


def _read_hash_file(hash_path: Path) -> tuple[str, dict[str, str]]:
    results: dict[str, str] = {}
    algorithm = DEFAULT_HASH_ALGORITHM

    for line in hash_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith(ALGORITHM_HEADER_PREFIX):
            algorithm = _normalize_algorithm(stripped[len(ALGORITHM_HEADER_PREFIX) :].strip())
            continue

        parts = stripped.split(maxsplit=1)
        if len(parts) != 2:
            raise IntegrityError(f"Invalid hash file line: {line!r}")

        digest, path_part = parts
        path_part = path_part.lstrip("*")
        results[path_part] = digest

    return algorithm, results
