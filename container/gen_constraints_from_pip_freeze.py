# gen_constraints_from_freeze.py
import os
import re
from sys import stdin
import argparse


def extract_editables():
    constraints = []
    for line in stdin:
        if line.startswith("-e "):
            match = re.search(r"#egg=([^\s]+)", line)
            if match:
                path = line[3:].split("#egg=")[0]
                pkg = match.group(1) if match else None
                if pkg and os.path.exists(path):
                    abs_path = os.path.abspath(path)
                    constraints.append(f"{pkg} @ file://{abs_path}")
            match = re.search(r"/([^\s/]+).git@([^\s]+)", line)
            if match:
                pkg = match.group(1) if match else None
                if pkg:
                    constraints.append(f"{pkg} @ {line.strip()[3:]}")

        if " @ " in line:
            if "/home/conda" not in line:
                constraints.append(f"{line.strip()}")
        if line.strip().startswith("# Editable Git install") or line.strip().startswith("# Editable install"):
            # print(line)
            match = re.search(r"\((.*)\)", line)
            pkg = match.group(1) if match else None
            if pkg:
                constraints.append(f"{pkg}")
    return constraints


if __name__ == "__main__":
    for line in extract_editables():
        print(line)
