#!/usr/bin/env python3
"""User-level Singularity build helper.

This script mirrors the behavior of build_container.sh but stages the
build using a sandbox and replays the %post section inside a writable
container as a non-root user. It enables installations into user-managed
environments such as Conda without requiring elevated privileges.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ENV_VAR_PATTERN = re.compile(r"\$(?:\{([^}]+)\}|([A-Za-z_][A-Za-z0-9_]*))")
PROHIBITED_COMMANDS = re.compile(r"\b(apt|apt-get|yum|dnf|zypper|rpm|sudo|apk|pacman)\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an Apptainer/Singularity image by staging a sandbox and "
            "executing the %post section as an unprivileged user."
        )
    )
    parser.add_argument(
        "--backend", default="megatron", help="Backend folder containing definition template."
    )
    parser.add_argument(
        "--definition",
        default="MegatronTraining",
        help="Definition template name without extension.",
    )
    parser.add_argument(
        "--requirements",
        help=(
            "Requirements file copied into the container. Defaults to <backend>/requirements_latest.txt."
        ),
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("CONTAINER_CACHE_DIR", os.getcwd()),
        help="Directory for the resulting .sif image.",
    )
    parser.add_argument(
        "--append-date", action="store_true", help="Append UTC timestamp to the output image name."
    )
    parser.add_argument(
        "--base-image",
        default=os.environ.get("BASE_IMAGE"),
        help="Override the source image (SIF file) used as the sandbox seed.",
    )
    parser.add_argument(
        "--container-cmd",
        default=os.environ.get("CONTAINER_RUNTIME", "singularity"),
        help="Container CLI to invoke (singularity or apptainer).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output image if it already exists.",
    )
    parser.add_argument(
        "--keep-sandbox",
        action="store_true",
        help="Preserve the intermediate sandbox directory for inspection.",
    )
    parser.add_argument(
        "--tempdir-prefix",
        default="oellm_sandbox",
        type=str,
    )
    return parser.parse_args()


def substitute_env_vars(template: str, values: dict[str, str]) -> str:
    env = os.environ.copy()
    env.update(values)

    def replacer(match: re.Match[str]) -> str:
        key = match.group(1) or match.group(2) or ""
        return env.get(key, "")

    return ENV_VAR_PATTERN.sub(replacer, template)


def parse_definition_sections(definition: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current_name: str | None = None
    current_lines: list[str] = []

    for raw_line in definition.splitlines():
        stripped = raw_line.strip()
        section_match = re.match(r"%(\w+)", stripped)
        if section_match:
            if current_name is not None:
                sections.setdefault(current_name, []).append("\n".join(current_lines).strip("\n"))
            current_name = section_match.group(1).lower()
            current_lines = []
            continue

        if current_name is None:
            continue
        current_lines.append(raw_line)

    if current_name is not None:
        sections.setdefault(current_name, []).append("\n".join(current_lines).strip("\n"))

    return {key: "\n\n".join(value).strip() for key, value in sections.items()}


def parse_files_entries(section_body: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    if not section_body:
        return entries

    for line in section_body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = shlex.split(stripped)
        if len(parts) != 2:
            raise SystemExit(f"Malformed %files entry: {line!r}")
        entries.append((parts[0], parts[1]))
    return entries


def run_command(cmd, *, cwd=None, env=None):
    printable = " ".join(shlex.quote(str(c)) for c in cmd)
    print(f"[+] {printable}")
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def ensure_no_prohibited_commands(script: str) -> None:
    match = PROHIBITED_COMMANDS.search(script)
    if match:
        raise RuntimeError(
            f"Unsupported privileged command detected in %post section: {match.group(1)}"
        )


def copy_into_sandbox(entries: list[tuple[str, str]], sandbox_dir: Path) -> None:
    for src, dst in entries:
        src_path = Path(src).expanduser().resolve()
        if not src_path.exists():
            raise SystemExit(f"%files source path not found: {src_path}")

        if not dst.startswith("/"):
            raise SystemExit(f"%files destination must be absolute: {dst}")

        dst_relative = dst.lstrip("/")
        dst_path = sandbox_dir / dst_relative
        if src_path.is_dir():
            if dst_path.exists():
                shutil.rmtree(dst_path)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_path, dst_path, symlinks=True)
        else:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if dst_path.exists():
                dst_path.unlink()
            shutil.copy2(src_path, dst_path)


def collect_git_metadata(repo_root: Path) -> tuple[str, bool, str, str]:
    def run(args: list[str]) -> str:
        try:
            result = subprocess.run(
                args,
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return ""
        return result.stdout.strip()

    commit = run(["git", "rev-parse", "HEAD"]) or "unknown"
    status = run(["git", "status", "--porcelain"])
    dirty = bool(status)
    diff = run(["git", "diff"]) if dirty else ""
    return commit, dirty, status, diff


def write_provenance_file(
    repo_root: Path,
    *,
    base_image: str,
    args: argparse.Namespace,
    container_cmd: str,
    requirements_file: Path,
) -> tuple[Path, tempfile.TemporaryDirectory]:
    commit, dirty, status, diff = collect_git_metadata(repo_root)
    temp_dir = tempfile.TemporaryDirectory(prefix="oellm_provenance")
    output_path = Path(temp_dir.name) / "container_provenance.json"
    payload = {
        "built_at": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
        "git_commit": commit,
        "git_dirty": dirty,
        "git_status": status,
        "git_diff": diff,
        "build_command": shlex.join(sys.argv),
        "backend": args.backend,
        "definition": args.definition,
        "requirements_file": str(requirements_file),
        "base_image": base_image,
        "container_runtime": container_cmd,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path, temp_dir


def main() -> None:
    args = parse_args()

    spec_root = Path(__file__).resolve().parent
    backend_dir = spec_root / args.backend
    definition_template = backend_dir / f"{args.definition}.def.in"

    if not definition_template.is_file():
        raise SystemExit(f"Definition template not found: {definition_template}")

    requirements_file = (
        Path(args.requirements) if args.requirements else backend_dir / "requirements_latest.txt"
    )
    requirements_file = requirements_file.expanduser().resolve()
    if not requirements_file.is_file():
        raise SystemExit(f"Requirements file not found: {requirements_file}")

    container_cmd = shutil.which(args.container_cmd) if args.container_cmd else None
    if not container_cmd:
        raise SystemExit(f"Container command not found in PATH: {args.container_cmd}")

    repo_root = spec_root.parent.resolve()
    arch = platform.machine()
    stamp = ""
    if args.append_date:
        stamp = f"_{subprocess.check_output(['date', '-u', '+%Y%m%d%H%M']).decode().strip()}"

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_name = f"{args.definition}_{arch}{stamp}.sif"
    target_path = output_dir / image_name

    if target_path.exists():
        if args.force:
            target_path.unlink()
        else:
            raise SystemExit(
                f"Target image {target_path} already exists. Use --force to overwrite."
            )

    substitutions = {
        "REPO_ROOT": str(repo_root),
        "REQUIREMENTS_PATH": str(requirements_file),
        "REQUIREMENTS_BASENAME": requirements_file.name,
        "ARCH": arch,
    }
    if args.base_image:
        substitutions["BASE_IMAGE"] = args.base_image

    template_text = definition_template.read_text()
    rendered_definition = substitute_env_vars(template_text, substitutions)
    sections = parse_definition_sections(rendered_definition)
    post_script = sections.get("post", "")
    files_entries = parse_files_entries(sections.get("files", ""))

    if post_script:
        ensure_no_prohibited_commands(post_script)

    base_image = args.base_image
    if not base_image:
        from_match = re.search(r"^From:\s*(.+)$", rendered_definition, re.MULTILINE)
        if not from_match:
            raise SystemExit("Unable to determine base image. Provide --base-image.")
        base_image = from_match.group(1).strip()

    base_image_path = Path(base_image).expanduser()
    if not base_image_path.exists():
        raise SystemExit(
            "Base image must be a local .sif file when building without setuid. "
            f"Provided path not found: {base_image_path}"
        )

    provenance_path, provenance_tmp = write_provenance_file(
        repo_root,
        base_image=str(base_image_path),
        args=args,
        container_cmd=container_cmd,
        requirements_file=requirements_file,
    )
    substitutions["PROVENANCE_PATH"] = str(provenance_path)

    temp_dir_manager = None
    if args.keep_sandbox:
        temp_dir = Path(tempfile.mkdtemp(prefix=args.tempdir_prefix))
    else:
        temp_dir_manager = tempfile.TemporaryDirectory(prefix=args.tempdir_prefix)
        temp_dir = Path(temp_dir_manager.name)

    sandbox_dir = temp_dir / "sandbox"

    sandbox_build_cmd = [
        container_cmd,
        "build",
        "--sandbox",
        "--force",
        "--fix-perms",
        str(sandbox_dir),
        str(base_image_path),
    ]
    run_command(sandbox_build_cmd)

    if files_entries:
        copy_into_sandbox(files_entries, sandbox_dir)

    if post_script:
        post_payload = f"{post_script}\n"  # set -euo pipefail\n
        exec_cmd = [
            container_cmd,
            "exec",
            "--writable",
            "--no-home",
            "--no-umask",
            str(sandbox_dir),
            "/bin/bash",
            "-c",
            post_payload,
        ]
        run_command(exec_cmd)
    else:
        print("[+] No %post section detected; skipping user-level install step.")

    final_build_cmd = [container_cmd, "build"]
    if args.force:
        final_build_cmd.append("--force")
    final_build_cmd += [str(target_path), str(sandbox_dir)]
    run_command(final_build_cmd)

    if not args.keep_sandbox and temp_dir_manager is not None:
        temp_dir_manager.cleanup()
    if provenance_tmp is not None:
        provenance_tmp.cleanup()

    print(f"[+] Image written to {target_path}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
