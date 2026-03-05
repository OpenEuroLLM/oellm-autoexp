from __future__ import annotations

import argparse
import sys

from .checker import DEFAULT_HASH_ALGORITHM, IntegrityError, check_integrity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Directory integrity checker")
    parser.add_argument("root", help="Root directory to scan")
    parser.add_argument(
        "targets",
        nargs="*",
        help="Optional files/directories relative to root to include",
    )
    parser.add_argument(
        "--hash-file",
        default=None,
        help="Path to hash file (default: <root>/checksums.hash)",
    )
    parser.add_argument(
        "--hash-algorithm",
        default=DEFAULT_HASH_ALGORITHM,
        help="Hash algorithm (for example: md5, sha1, sha256, sha512)",
    )
    parser.add_argument(
        "--exclude-regex",
        action="append",
        default=None,
        help="Regex for excluding files (repeatable). Default excludes *.hash.",
    )
    parser.add_argument(
        "--force-hash",
        action="store_true",
        help="Force a full hash verification and refresh hash file on success.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = check_integrity(
            root=args.root,
            targets=args.targets or None,
            hash_file=args.hash_file,
            exclude_patterns=args.exclude_regex,
            force_hash=args.force_hash,
            hash_algorithm=args.hash_algorithm,
        )
    except (IntegrityError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        "OK "
        f"mode={report.mode} "
        f"algorithm={report.hash_algorithm} "
        f"files={report.checked_files} "
        f"hash_file={report.hash_file}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
