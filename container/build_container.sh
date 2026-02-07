#!/bin/bash
# Simple wrapper to build the oellm-autoexp container image.

set -euo pipefail

ORIGINAL_ARGS=("$@")

show_usage() {
  cat <<USAGE
Usage: ./build_container.sh [OPTIONS]

Options:
  --backend NAME      Backend folder containing the definition template (default: megatron)
  --definition NAME   Singularity definition template name inside the backend folder (default: MegatronTraining)
  --requirements PATH Optional requirements file copied into the container and
                      passed to pip install (default: container/megatron/requirements_latest.txt).
  --output DIR        Directory to write the resulting .sif image (default: current dir).
  --append-date       Append a UTC timestamp to the generated image name.
  --base-image IMAGE  Override the base image (default: nvcr.io/nvidia/pytorch:25.08-py3).
  --container-cmd CONTAINER_RUNTIME   Override container command (default: singularity).
USAGE
}

SPEC_ROOT="$(dirname "$0")"
BACKEND="megatron"
DEFINITION="MegatronTraining"
export REQUIREMENTS_PATH="${SPEC_ROOT}/${BACKEND}/requirements_latest.txt"
OUTPUT_DIR="${CONTAINER_CACHE_DIR:-$(pwd)}"
APPEND_DATE=false
BASE_IMAGE=${BASE_IMAGE:-nvcr.io/nvidia/pytorch:25.08-py3}
CONTAINER_RUNTIME="singularity"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --definition)
      shift
      DEFINITION="$1"
      ;;
    --backend)
      shift
      BACKEND="$1"
      ;;
    --additional-tag)
      shift
      ADDITIONAL_TAG="$1"
      ;;
    --requirements)
      shift
      export REQUIREMENTS_PATH="$1"
      ;;
    --output)
      shift
      OUTPUT_DIR="$1"
      ;;
    --append-date)
      APPEND_DATE=true
      ;;
    --base-image)
      shift
      BASE_IMAGE="$1"
      ;;
    --container-cmd)
      shift
      CONTAINER_RUNTIME="$1"
      ;;
    -h|--help)
      show_usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      show_usage >&2
      exit 1
      ;;
  esac
  shift || true
done

SPEC_DIR="${SPEC_ROOT}/${BACKEND}"
DEFINITION_TEMPLATE="${SPEC_DIR}/${DEFINITION}.def.in"

if [[ ! -f "$DEFINITION_TEMPLATE" ]]; then
  echo "Definition template $DEFINITION_TEMPLATE not found" >&2
  exit 1
fi

if [[ ! -f "$REQUIREMENTS_PATH" ]]; then
  echo "Requirements file $REQUIREMENTS_PATH not found" >&2
  exit 1
fi

if [[ $REQUIREMENTS_PATH != /* ]]; then
  export REQUIREMENTS_PATH="$(cd "$(dirname "$REQUIREMENTS_PATH")" && pwd)/$(basename "$REQUIREMENTS_PATH")"
fi


export ARCH=$(uname -m)
STAMP=""
if [[ "$APPEND_DATE" == "true" ]]; then
  STAMP="_$(date -u +%Y%m%d%H%M)"
fi

TMP_DEF="${BACKEND}_${DEFINITION}_${ARCH}.def"
export BASE_IMAGE
export CONTAINER_RUNTIME
export REPO_ROOT="$(cd "${SPEC_ROOT}/.." && pwd)"

PROVENANCE_TMP_DIR=$(mktemp -d -t oellm_provenance.XXXXXX)
PROVENANCE_JSON="${PROVENANCE_TMP_DIR}/container_provenance.json"
BUILD_COMMAND="$0"
for arg in "${ORIGINAL_ARGS[@]}"; do
  BUILD_COMMAND+=" $(printf '%q' "$arg")"
done

export BUILD_COMMAND
export PROVENANCE_PATH="$PROVENANCE_JSON"

python3 <<'PY'
import json
import os
import pathlib
import subprocess
from datetime import datetime, timezone

repo_root = pathlib.Path(os.environ.get("REPO_ROOT", "."))
provenance_path = pathlib.Path(os.environ["PROVENANCE_PATH"])

def run_git(args):
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return ""
    return result.stdout.strip()

commit = run_git(["rev-parse", "HEAD"]) or "unknown"
status = run_git(["status", "--porcelain"])
dirty = bool(status)
diff = run_git(["diff"]) if dirty else ""

payload = {
    "built_at": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
    "git_commit": commit,
    "git_dirty": dirty,
    "git_status": status,
    "git_diff": diff,
    "build_command": os.environ.get("BUILD_COMMAND", ""),
    "backend": os.environ.get("BACKEND"),
    "definition": os.environ.get("DEFINITION"),
    "requirements_path": os.environ.get("REQUIREMENTS_PATH"),
    "base_image": os.environ.get("BASE_IMAGE"),
    "container_runtime": os.environ.get("CONTAINER_RUNTIME"),
}
provenance_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY



export REQUIREMENTS_BASENAME=$(echo $REQUIREMENTS_PATH | grep -o -e '[^/]*$')
envsubst < "$DEFINITION_TEMPLATE" > "$TMP_DEF"

cat $TMP_DEF

cleanup() {
  rm -f "$TMP_DEF"
  rm -rf "$PROVENANCE_TMP_DIR"
}
trap cleanup EXIT

IMAGE_NAME="${DEFINITION}${ADDITIONAL_TAG}_${ARCH}${STAMP}.sif"
TARGET_PATH="$OUTPUT_DIR/$IMAGE_NAME"

mkdir -p "$OUTPUT_DIR"

if ! command -v $CONTAINER_RUNTIME >/dev/null 2>&1; then
  echo "$CONTAINER_RUNTIME not found in PATH" >&2
  exit 1
fi

set -x
$CONTAINER_RUNTIME build --fakeroot "$TARGET_PATH" "$TMP_DEF"
set +x

echo "Image written to $TARGET_PATH"
