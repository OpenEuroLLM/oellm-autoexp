#!/usr/bin/env python3
"""Script to extract versions from frozen requirements and create a conda
environment.yaml file.

Takes a frozen requirements file (with exact versions) and a latest
requirements file (with package names, possibly with version
constraints), extracts the exact versions from the frozen file, and
generates a conda environment.yaml file.

Supports extending an existing base environment by excluding already-
installed packages. For packages installed via @
file://,
converts absolute paths to relative paths for portability.
"""

import argparse
import re
from pathlib import Path
from urllib.parse import urlparse


def make_relative_path(file_url: str, output_dir: Path) -> str:
    """
    Convert a file:// URL to a relative path from the output directory.

    Args:
        file_url: URL string like "file:///path/to/package.whl"
        output_dir: Directory where the environment.yaml will be written

    Returns:
        Relative path string suitable for environment.yaml
    """
    # Parse the file URL
    parsed = urlparse(file_url)
    if parsed.scheme != "file":
        return file_url  # Not a file URL, return as-is

    # Get the absolute path from the URL
    abs_path = Path(parsed.path)

    # Try to make it relative to the output directory
    try:
        rel_path = abs_path.relative_to(output_dir.resolve())
        return f"./{rel_path}"
    except ValueError:
        # If not under the same tree, try to compute relative path
        try:
            rel_path = Path("..") / abs_path.relative_to(output_dir.parent.resolve())
            return str(rel_path)
        except ValueError:
            # Fall back to absolute path if we can't make it relative
            return str(abs_path)


def parse_frozen_requirements(frozen_file: Path, output_dir: Path) -> dict[str, str]:
    """Parse frozen requirements file and extract package names with their
    versions.

    Args:
        frozen_file: Path to frozen requirements file
        output_dir: Directory where environment.yaml will be written (for relative path conversion)

    Returns:
        Dict mapping normalized package names to their full requirement specs
    """
    packages = {}

    with open(frozen_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Extract package name (before ==, @, etc.)
            match = re.match(r"^([a-zA-Z0-9_\-\.]+)", line)
            if not match:
                continue

            pkg_name = match.group(1)
            # Normalize package name (lowercase, replace - with _)
            normalized_name = pkg_name.lower().replace("-", "_")

            # Handle different installation formats
            if " @ " in line:
                # Format: package @ file:///path or package @ git+https://...
                parts = line.split(" @ ", 1)
                if len(parts) == 2:
                    spec = parts[1].strip()

                    # Convert file:// URLs to relative paths
                    if spec.startswith("file://"):
                        rel_path = make_relative_path(spec, output_dir)
                        packages[normalized_name] = rel_path
                    else:
                        # Keep git+https://, https://, etc. as-is
                        packages[normalized_name] = line.strip()
            else:
                # Standard format: package==version
                version_match = re.search(r"==([^\s;]+)", line)
                if version_match:
                    version = version_match.group(1)
                    packages[normalized_name] = f"{pkg_name}=={version}"

    return packages


def parse_latest_requirements(latest_file: Path) -> set[str]:
    """Parse latest requirements file and extract package names.

    Returns:
        Set of normalized package names
    """
    packages = set()

    with open(latest_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Extract package name (before >=, <=, ==, etc.)
            match = re.match(r"^([a-zA-Z0-9_\-\.]+)", line)
            if match:
                pkg_name = match.group(1)
                # Normalize package name
                normalized_name = pkg_name.lower().replace("-", "_")
                packages.add(normalized_name)

    return packages


def parse_base_environment(base_file: Path) -> dict[str, str]:
    """Parse base environment file to get list of already-installed packages
    with their versions.

    Args:
        base_file: Path to file listing base environment packages (one per line, with or without versions)

    Returns:
        Dict mapping normalized package names to their version specs (or empty string if no version)
    """
    packages = {}

    with open(base_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Extract package name and version if present
            match = re.match(r"^([a-zA-Z0-9_\-\.]+)", line)
            if match:
                pkg_name = match.group(1)
                # Normalize package name
                normalized_name = pkg_name.lower().replace("-", "_")

                # Check if version is specified
                version_match = re.search(r"==([^\s;]+)", line)
                if version_match:
                    # Store with exact version
                    packages[normalized_name] = f"{pkg_name}=={version_match.group(1)}"
                else:
                    # Store package name only (version will be looked up from frozen)
                    packages[normalized_name] = pkg_name

    return packages


def parse_conda_packages(conda_file: Path) -> list[str]:
    """Parse conda packages file for non-Python dependencies.

    Args:
        conda_file: Path to file listing conda packages (one per line, optionally with versions)

    Returns:
        List of conda package specifications
    """
    packages = []

    with open(conda_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            packages.append(line)

    return packages


def create_conda_env_yaml(
    pip_packages: dict[str, str],
    conda_packages: list[str],
    output_file: Path,
    env_name: str = "myenv",
    include_python: bool = True,
):
    """Create a conda environment.yaml file with the specified packages.

    Args:
        pip_packages: Dict of pip package specs (name==version or paths)
        conda_packages: List of conda package specifications
        output_file: Path to output environment.yaml file
        env_name: Name of the conda environment
        include_python: Whether to include python and pip in dependencies
    """
    with open(output_file, "w") as f:
        f.write(f"name: {env_name}\n")
        f.write("channels:\n")
        f.write("  - defaults\n")
        f.write("dependencies:\n")

        if include_python:
            f.write("  - python\n")
            f.write("  - pip\n")

        # Add conda packages
        if conda_packages:
            for pkg_spec in sorted(conda_packages):
                f.write(f"  - {pkg_spec}\n")

        # Add pip packages if any
        if pip_packages:
            f.write("  - pip:\n")
            for pkg_spec in sorted(pip_packages.values()):
                f.write(f"    - {pkg_spec}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract versions from frozen requirements and create conda environment.yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  %(prog)s frozen.txt latest.txt -o environment.yaml

  # With custom environment name
  %(prog)s frozen.txt latest.txt -o env.yaml -n myproject

Notes:
  - Packages with @ file:// URLs are converted to relative paths for portability
  - The relative paths are computed from the directory containing the output file
  - This allows distributing the environment.yaml alongside wheel files
        """,
    )
    parser.add_argument(
        "frozen_reqs", type=Path, help="Path to frozen requirements file (with exact versions)"
    )
    parser.add_argument(
        "latest_reqs", type=Path, help="Path to latest requirements file (subset without versions)"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("environment.yaml"),
        help="Output conda environment.yaml file (default: environment.yaml)",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="myenv",
        help="Name of the conda environment (default: myenv)",
    )
    parser.add_argument(
        "--skip-local",
        action="store_true",
        help="Skip packages installed from local files (@ file://). Only include PyPI packages.",
    )
    parser.add_argument(
        "--base-env",
        type=Path,
        help=(
            "Path to file listing base environment packages (one per line). "
            + "Packages in this file will be excluded from the output."
        ),
    )
    parser.add_argument(
        "--conda-packages",
        type=Path,
        help=(
            "Path to file listing conda packages to install (e.g., tmux, git). "
            + "One per line, optionally with versions (e.g., 'tmux>=3.0')."
        ),
    )
    parser.add_argument(
        "--no-python",
        action="store_true",
        help="Don't include python and pip in the dependencies (useful when extending an existing environment).",
    )

    args = parser.parse_args()

    # Determine output directory for relative path conversion
    output_dir = args.output.parent if args.output.parent != Path() else Path(".")

    # Parse frozen requirements
    print(f"Parsing frozen requirements from: {args.frozen_reqs}")
    frozen_packages = parse_frozen_requirements(args.frozen_reqs, output_dir.resolve())
    print(f"Found {len(frozen_packages)} packages in frozen requirements")

    # Parse latest requirements
    print(f"Parsing latest requirements from: {args.latest_reqs}")
    latest_packages = parse_latest_requirements(args.latest_reqs)
    print(f"Found {len(latest_packages)} packages in latest requirements")

    # Parse base environment if provided
    base_packages = set()
    if args.base_env:
        print(f"Parsing base environment from: {args.base_env}")
        base_packages = parse_base_environment(args.base_env)
        print(f"Found {len(base_packages)} packages in base environment (will be excluded)")

    # Parse conda packages if provided
    conda_pkgs = []
    if args.conda_packages:
        print(f"Parsing conda packages from: {args.conda_packages}")
        conda_pkgs = parse_conda_packages(args.conda_packages)
        print(f"Found {len(conda_pkgs)} conda packages to install")

    # First, add all base environment packages as constraints (to prevent upgrades)
    matched_packages = {}
    if base_packages:
        print(
            f"\nAdding {len(base_packages)} base environment packages as constraints (to prevent pip upgrades):"
        )
        for pkg_name, pkg_spec in base_packages.items():
            # If version not specified in base_env file, look it up from frozen
            if "==" not in pkg_spec and pkg_name in frozen_packages:
                pkg_spec = frozen_packages[pkg_name]
            matched_packages[pkg_name] = pkg_spec
            print(f"  - {pkg_spec}")

    # Extract versions for packages in latest requirements
    missing_packages = []
    skipped_local = []
    added_new = []

    for pkg_name in latest_packages:
        # Skip if already added from base environment
        if pkg_name in matched_packages:
            continue

        if pkg_name in frozen_packages:
            spec = frozen_packages[pkg_name]
            # Check if this is a local file installation
            if args.skip_local and (spec.startswith("/") or spec.startswith("./")):
                skipped_local.append(pkg_name)
            else:
                matched_packages[pkg_name] = spec
                added_new.append(pkg_name)
        else:
            missing_packages.append(pkg_name)

    print(f"\nTotal packages in output: {len(matched_packages)}")
    if added_new:
        print(f"Added {len(added_new)} new packages from latest requirements")

    if skipped_local:
        print(
            f"Skipped {len(skipped_local)} local file packages (use without --skip-local to include):"
        )
        for pkg in sorted(skipped_local):
            print(f"  - {pkg}")

    if missing_packages:
        print(f"WARNING: {len(missing_packages)} packages not found in frozen requirements:")
        for pkg in sorted(missing_packages):
            print(f"  - {pkg}")

    # Create conda environment.yaml
    print(f"\nCreating conda environment file: {args.output}")
    create_conda_env_yaml(
        matched_packages, conda_pkgs, args.output, args.name, include_python=not args.no_python
    )
    print("Done!")
    print("\nTo use this environment:")
    if args.base_env or args.no_python:
        print(f"  conda env update -f {args.output}")
        print("  (Use 'update' to add to existing environment)")
    else:
        print(f"  conda env create -f {args.output}")


if __name__ == "__main__":
    main()
