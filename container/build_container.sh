#!/bin/bash
# Simple wrapper to build the oellm-autoexp container image.

set -euo pipefail

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
USAGE
}

SPEC_ROOT="$(dirname "$0")"
BACKEND="megatron"
DEFINITION="MegatronTraining"
REQUIREMENTS_FILE="${SPEC_ROOT}/${BACKEND}/requirements_latest.txt"
OUTPUT_DIR="${CONTAINER_CACHE_DIR:-$(pwd)}"
APPEND_DATE=false
BASE_IMAGE=${BASE_IMAGE:-nvcr.io/nvidia/pytorch:25.08-py3}

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
    --requirements)
      shift
      REQUIREMENTS_FILE="$1"
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

if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
  echo "Requirements file $REQUIREMENTS_FILE not found" >&2
  exit 1
fi

if [[ $REQUIREMENTS_FILE != /* ]]; then
  REQUIREMENTS_FILE="$(cd "$(dirname "$REQUIREMENTS_FILE")" && pwd)/$(basename "$REQUIREMENTS_FILE")"
fi

export ARCH=$(uname -m)
STAMP=""
if [[ "$APPEND_DATE" == "true" ]]; then
  STAMP="_$(date -u +%Y%m%d%H%M)"
fi

TMP_DEF="${BACKEND}_${DEFINITION}_${ARCH}.def"
export BASE_IMAGE
export REPO_ROOT="$(cd "${SPEC_ROOT}/.." && pwd)"
export REQUIREMENTS_PATH="$REQUIREMENTS_FILE"
export REQUIREMENTS_BASENAME=$(basename "$REQUIREMENTS_FILE")

envsubst < "$DEFINITION_TEMPLATE" > "$TMP_DEF"

IMAGE_NAME="${DEFINITION}_${ARCH}${STAMP}.sif"
TARGET_PATH="$OUTPUT_DIR/$IMAGE_NAME"

mkdir -p "$OUTPUT_DIR"

if ! command -v apptainer >/dev/null 2>&1; then
  echo "apptainer not found in PATH" >&2
  exit 1
fi

set -x
apptainer build "$TARGET_PATH" "$TMP_DEF"
set +x

echo "Image written to $TARGET_PATH"
