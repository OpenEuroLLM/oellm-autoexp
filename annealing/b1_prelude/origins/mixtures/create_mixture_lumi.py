#!/usr/bin/env python3

import argparse
import sys

BASE = "/scratch/project_465002530/training/collection/"


def render(path, base, out):
    with open(path) as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ratio, rel = line.split(" ", 1)
            out.append("{} {}{}".format(ratio, base, rel))


def main():
    parser = argparse.ArgumentParser(
        description="Render <ratio> <relpath> datamix lines as a one-line LUMI datamix")
    parser.add_argument("--inputs", nargs="+",
                        help="source file(s) in '<ratio> <relpath>' format, in order")
    parser.add_argument("--dominant", type=str, default=None,
                        help="optional baseline dominant file prepended before the inputs")
    parser.add_argument("--base", type=str, default=BASE,
                        help="absolute collection prefix (default: %(default)s)")
    arguments = parser.parse_args()
    out = []
    if arguments.dominant:
        render(arguments.dominant, arguments.base, out)

    for file in arguments.inputs:
        render(file, arguments.base, out)
    sys.stdout.write(" ".join(out) + "\n")


if __name__ == "__main__":
    main()
