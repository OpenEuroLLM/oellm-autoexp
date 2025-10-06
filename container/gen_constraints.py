#!/usr/bin/env python3
"""
Write constraints from *installed* packages as clean 'name==version' pins,
ignoring how they were originally installed (wheel path, URL, VCS, editable).

Usage:
  python constraints_from_installed.py [-o constraints.txt] [--only requirements.txt] [--exclude pip setuptools wheel]
"""

from __future__ import annotations
import argparse
import sys
from packaging.utils import canonicalize_name

try:
    import importlib.metadata as imd  # Py3.8+
except Exception:  # pragma: no cover
    import importlib_metadata as imd  # type: ignore


def installed_map():
    out = {}
    for dist in imd.distributions():
        name = dist.metadata.get("Name") or dist.metadata.get("Summary") or dist.metadata.get("Home-page")
        if not name:
            continue
        out[canonicalize_name(name)] = dist.version
    return out


def read_requirements_names(path: str):
    from packaging.requirements import Requirement

    names = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("-"):
                continue
            if "#" in s:
                s = s.split("#", 1)[0].strip()
                if not s:
                    continue
            try:
                req = Requirement(s)
            except Exception:
                continue
            if req.marker and not req.marker.evaluate():
                continue
            names.add(canonicalize_name(req.name))
    return names


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="constraints.installed.txt")
    ap.add_argument("--only", help="If set, only pin packages that appear in this requirements file")
    ap.add_argument("--exclude", nargs="*", default=[], help="Package names to exclude from constraints")
    args = ap.parse_args(argv)

    inst = installed_map()
    exclude = {canonicalize_name(x) for x in args.exclude}
    if args.only:
        keep = read_requirements_names(args.only)
    else:
        keep = set(inst.keys())

    lines = []
    for name in sorted(keep, key=str.lower):
        if name in exclude:
            continue
        ver = inst.get(name)
        if ver:
            lines.append(f"{name}=={ver}")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

    # print(f"[ok] wrote {len(lines)} pins to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
