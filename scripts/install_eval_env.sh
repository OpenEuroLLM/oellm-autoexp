#!/usr/bin/env bash
# Set up the Python environment for the OELLM-CLI evaluation backend on
# the current cluster. The install path depends on the cluster:
#
#   juwels    -> uv venv at $OELLM_EVAL_VENV (default ~/work/eval_venv)
#                pip install -e oellm_evals[eval]   (pulls torch from PyPI)
#   jupiter   -> uv venv at $OELLM_EVAL_VENV (default ~/work/venv)
#                pip install -e oellm_evals[eval]
#   leonardo  -> pip install --user -e oellm_evals[eval-base] inside
#                eval_env-leonardo.sif (container ships torch + transformers)
#   lumi      -> pip install --target $HOME/eval_local -e oellm_evals[eval-base]
#                inside laif-rocm-….sif (container ships rocm torch)
#
# The `[eval]` / `[eval-base]` extras are declared in
# submodules/oellm_evals/pyproject.toml — they're the single source of
# truth for what lm-eval-harness needs.
#
# Run from the repo root on a login node with internet access. Idempotent:
# re-running upgrades existing installs.
#
# Usage:
#   bash scripts/install_eval_env.sh [--cluster <name>] [--prefetch]
#
# Optional:
#   --cluster NAME     Override autodetection (juwels|leonardo|lumi|jupiter)
#   --prefetch         After installing, also run scripts/prefetch_datasets.py
#                      (downloads HF datasets into the cluster's HF_HOME)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OELLM_EVALS_DIR="$REPO_ROOT/submodules/oellm_evals"
AUTOEXP_DIR="$REPO_ROOT"

CLUSTER=""
PREFETCH=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cluster) CLUSTER="$2"; shift 2 ;;
        --prefetch) PREFETCH=1; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$CLUSTER" ]]; then
    CLUSTER="$(uv run --python 3.12 python "$REPO_ROOT/scripts/detect_cluster.py" 2>/dev/null \
               || python3 "$REPO_ROOT/scripts/detect_cluster.py" \
               || echo unknown)"
fi
echo "Cluster: $CLUSTER"

case "$CLUSTER" in
    juwels|jupiter)
        VENV="${OELLM_EVAL_VENV:-$HOME/work/eval_venv}"
        if [[ ! -x "$VENV/bin/python" ]]; then
            echo "Creating venv at $VENV"
            uv venv --python 3.12 "$VENV"
            "$VENV/bin/python" -m ensurepip --default-pip
        fi
        PIP=("$VENV/bin/python" -m pip)
        # The Stages-2025 modules expose Python 3.13 site-packages via
        # PYTHONPATH; that breaks pip installs into a 3.12 venv.
        PYTHONPATH= "${PIP[@]}" install --no-cache-dir --upgrade pip
        PYTHONPATH= "${PIP[@]}" install --no-cache-dir "setuptools<80" wheel
        # `rouge-score` (transitive via lm-eval) still references
        # pkg_resources at build time; bypass build isolation so it
        # picks up our pinned setuptools.
        PYTHONPATH= "${PIP[@]}" install --no-cache-dir --no-build-isolation \
            rouge-score
        PYTHONPATH= "${PIP[@]}" install --no-cache-dir \
            -e "$OELLM_EVALS_DIR[eval]" -e "$AUTOEXP_DIR" "compoconf==0.1.14"
        echo "✓ Venv ready at $VENV"
        ;;

    leonardo)
        SIF="${OELLM_EVAL_SIF:-/leonardo_work/OELLM_prod2026/container_images/eval_env-leonardo.sif}"
        BINDS=(--bind /leonardo_scratch --bind /leonardo --bind /leonardo_work)
        echo "Installing oellm[eval-base] into container user-site (\$HOME/.local)…"
        singularity exec "${BINDS[@]}" --env HF_HUB_OFFLINE=0 "$SIF" bash -c "
            set -e
            pip install --user --no-cache-dir --upgrade pip setuptools wheel
            pip install --user --no-cache-dir --upgrade \
                -e '$OELLM_EVALS_DIR[eval-base]' -e '$AUTOEXP_DIR' \
                'compoconf==0.1.14'
        "
        # oellm-evals's local mode needs *some* venv path to `source activate`;
        # the container is already activated, so point at an empty stub.
        if [[ ! -f "$HOME/eval_venv_stub/bin/activate" ]]; then
            mkdir -p "$HOME/eval_venv_stub/bin"
            : > "$HOME/eval_venv_stub/bin/activate"
            echo "✓ Stub venv at \$HOME/eval_venv_stub"
        fi
        echo "✓ Leonardo eval env ready"
        ;;

    lumi)
        SIF="${OELLM_EVAL_SIF:-/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif}"
        BINDS=(--bind "$HOME" --bind /pfs --bind /scratch)
        TARGET="$HOME/eval_local/lib/python3.12/site-packages"
        mkdir -p "$TARGET"
        echo "Installing oellm[eval-base] into $TARGET via --target…"
        # `pip install --user` is rejected inside Lumi's container (a venv
        # is already active); `--target` is the equivalent escape hatch.
        singularity exec "${BINDS[@]}" --env HF_HUB_OFFLINE=0 "$SIF" bash -c "
            set -e
            pip install --no-cache-dir --upgrade --target '$TARGET' \
                -e '$OELLM_EVALS_DIR[eval-base]' -e '$AUTOEXP_DIR' \
                'compoconf==0.1.14'
        "
        if [[ ! -f "$HOME/eval_venv_stub/bin/activate" ]]; then
            mkdir -p "$HOME/eval_venv_stub/bin"
            : > "$HOME/eval_venv_stub/bin/activate"
        fi
        echo "✓ Lumi eval env ready at $TARGET (PYTHONPATH set by slurm/lumi_eval.yaml)"
        ;;

    *)
        echo "ERROR: unknown cluster '$CLUSTER' — pass --cluster explicitly." >&2
        exit 1
        ;;
esac

# One-time runtime patches — idempotent.
echo "Patching Megatron-Bridge model imports for tolerant loading…"
python3 "$REPO_ROOT/container/megatron/patch_bridge_lazy_imports.py" \
    "$REPO_ROOT/submodules/Megatron-Bridge/src/megatron/bridge/models"

echo "Pulling heavy tokenizer files (Qwen3-0.6B)…"
python3 "$REPO_ROOT/scripts/download_tokenizers.py" --tokenizers Qwen/Qwen3-0.6B || \
    echo "(download_tokenizers.py failed — re-run manually if needed)"

if [[ "$PREFETCH" == "1" ]]; then
    echo "Pre-fetching eval datasets into HF_HOME=${HF_HOME:-<unset>}…"
    case "$CLUSTER" in
        leonardo)
            singularity exec "${BINDS[@]}" \
                --env HF_HUB_OFFLINE=0 --env HF_DATASETS_OFFLINE=0 \
                --env HF_HOME="${HF_HOME:?HF_HOME must be set for --prefetch}" \
                "$SIF" python3 "$REPO_ROOT/scripts/prefetch_datasets.py" \
                open-sci-0.01 "$OELLM_EVALS_DIR/oellm/resources/task-groups.yaml"
            ;;
        lumi)
            singularity exec "${BINDS[@]}" \
                --env HF_HUB_OFFLINE=0 --env HF_DATASETS_OFFLINE=0 \
                --env HF_HOME="${HF_HOME:?HF_HOME must be set for --prefetch}" \
                --env PYTHONPATH="$TARGET" \
                "$SIF" python3 "$REPO_ROOT/scripts/prefetch_datasets.py" \
                open-sci-0.01 "$OELLM_EVALS_DIR/oellm/resources/task-groups.yaml"
            ;;
        juwels|jupiter)
            HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 \
                "$VENV/bin/python" "$REPO_ROOT/scripts/prefetch_datasets.py" \
                open-sci-0.01 "$OELLM_EVALS_DIR/oellm/resources/task-groups.yaml"
            ;;
    esac
fi

echo
echo "Done. Next step: submit a chain experiment, e.g."
echo "  PYTHONPATH=. python scripts/run_autoexp.py \\"
echo "      --config-name experiments/korbi/chain_qwen3_bridge_train_eval_$CLUSTER"
