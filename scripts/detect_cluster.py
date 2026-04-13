#!/usr/bin/env python3
"""Infer cluster identity (name, partition, account) from hostname patterns.

Sources:
  https://github.com/OpenEuroLLM/oellm-cli/blob/main/oellm/resources/clusters.yaml
  TensorWave (MI325X): tus1-vm-amd-misc-NN

Usage (print cluster name):
  python scripts/detect_cluster.py

Usage (print all fields as KEY=VALUE):
  python scripts/detect_cluster.py --env

Usage (import):
  from scripts.detect_cluster import detect_cluster, ClusterInfo
  info = detect_cluster()           # raises if unknown
  info = detect_cluster("uan04")    # override hostname for testing
  print(info.name, info.partition, info.account)
"""


import argparse
import fnmatch
import socket
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class ClusterInfo:
    name: str
    partition: str
    account: str
    hostname_patterns: list[str]


# Source: https://github.com/OpenEuroLLM/oellm-cli/blob/main/oellm/resources/clusters.yaml
# First match wins.
_CLUSTERS: list[ClusterInfo] = [
    ClusterInfo("lumi",     "standard-g",       "project_462000963", ["uan*"]),
    ClusterInfo("jupiter",  "booster",        "reformo",           ["jp*"]),
    ClusterInfo("leonardo", "boost_usr_prod", "OELLM_prod2026",    ["*.leonardo.local"]),
    ClusterInfo("juwels",   "dc-gpu",         "synthlaion",        ["*.jureca", "*.juwels"]),
    ClusterInfo("snellius", "gpu_h100",       "thomaso",           ["*.snellius.surf.nl"]),
    ClusterInfo("mi325x",   "amd-tw-verification", "", ["tus*"]),
    ClusterInfo("ci",       "gpu",            "test",              ["ip-*", "[0-9]*-[0-9]*-[0-9]*-[0-9]*"]),
]


def detect_cluster(hostname: str | None = None) -> ClusterInfo:
    """Return ClusterInfo for the given (or current) hostname.

    Raises:
        ValueError: if no pattern matches the hostname.
    """
    if hostname is None:
        hostname = socket.gethostname()

    for cluster in _CLUSTERS:
        if any(fnmatch.fnmatch(hostname, p) for p in cluster.hostname_patterns):
            return cluster

    raise ValueError(
        f"Could not detect cluster for hostname {hostname!r}. "
        "Set CLUSTER= manually (e.g. lumi, jupiter, juwels, mi325x)."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", action="store_true", help="Print all fields as KEY=VALUE (shell-sourceable via eval)")
    parser.add_argument("--fields", action="store_true", help="Print name, partition, account on separate lines (for shell read)")
    parser.add_argument("--field", choices=["name", "partition", "account"], help="Print a single field")
    parser.add_argument("hostname", nargs="?", help="Override hostname (for testing)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        info = detect_cluster(args.hostname)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    if args.fields:
        print(info.name)
        print(info.partition)
        print(info.account)
    elif args.env:
        print(f"CLUSTER={info.name}")
        print(f"SLURM_PARTITION={info.partition}")
        print(f"SLURM_ACCOUNT={info.account}")
    elif args.field:
        print(getattr(info, args.field))
    else:
        print(info.name)
