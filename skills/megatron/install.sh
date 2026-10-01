#!/usr/bin/env bash
set -euo pipefail

usage() { echo "Usage: $0 [--copy] [--target DIR]"; }
mode=link
target="${CODEX_HOME:-$HOME/.codex}/skills"
while (($#)); do
  case "$1" in
    --copy) mode=copy ;;
    --target) shift; target=${1:?missing target} ;;
    --help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
  shift
done
root=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$target"
for skill in "$root"/megatron-*; do
  name=$(basename "$skill")
  dest="$target/$name"
  if [[ -e "$dest" || -L "$dest" ]]; then rm -rf "$dest"; fi
  if [[ "$mode" == link ]]; then ln -s "$skill" "$dest"; else cp -R "$skill" "$dest"; fi
  echo "Installed $name -> $dest"
done
