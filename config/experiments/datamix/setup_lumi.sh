#!/usr/bin/env bash
# Interactive setup of the DataMix chain on LUMI, from a fresh clone to a test run.
#
#   git clone -b datamix --recurse-submodules https://github.com/OpenEuroLLM/oellm-autoexp.git
#   cd oellm-autoexp
#   bash config/experiments/datamix/setup_lumi.sh
#
# Every step first checks whether it is already done and skips it if so; otherwise it
# says what it will do, asks (with a default you can accept with Enter or change) and shows
# each command before running it. Nothing is deleted or overwritten without asking. Safe to
# run again at any time, e.g. after fixing a failed step.
#
# Steps: 1 checks, 2 project, 3 paths + ~/datamix_env.sh, 4 uv + Python env,
#        5 containers, 6 eval env, 7 tokenizer, 8 test run
#
# Options:
#   --yes          accept all defaults without asking (the test run then needs --data)
#   --data PATH    training data prefix for the test run (path without .bin/.idx)
#   --scales LIST  scales of the test run, default 0.1B (e.g. 0.1B,0.4B,1.4B)
#   --no-test      stop before the test run
#
# See lumi_autoexp_installation_guide.md for what the steps do and for troubleshooting.
set -euo pipefail

YES=false; DATA=""; SCALES=0.1B; TEST=true
while [ $# -gt 0 ]; do
    case "$1" in
        --yes) YES=true; shift ;;
        --data) DATA=$2; shift 2 ;;
        --scales) SCALES=$2; shift 2 ;;
        --no-test) TEST=false; shift ;;
        -h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 1 ;;
    esac
done

cd "$(dirname "$(realpath "$0")")/../../.."   # repo root
REPO=$PWD
ENV_FILE=$HOME/datamix_env.sh
UV_VERSION=0.4.15
DEFAULT_PROJECT=project_465002530
SIF_TRAIN=AutoExp_2026-07-02.sif
SIF_CONVERT=MegatronTrainingLumi_bridge_x86_64.sif
SIF_EVAL=laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
TOKENIZER=openeurollm/tokenizer-256k
MISSING=()   # what is still missing at the end

# --- helpers ---------------------------------------------------------------------------
B=$'\e[1m'; G=$'\e[32m'; Y=$'\e[33m'; R=$'\e[31m'; N=$'\e[0m'
[ -t 1 ] || { B=; G=; Y=; R=; N=; }
step() { echo; echo "${B}[$1/8] $2${N}"; }
ok()   { echo "  ${G}✓${N} $*"; }
warn() { echo "  ${Y}!${N} $*"; }
bad()  { echo "  ${R}✗${N} $*"; }
info() { echo "    $*"; }
die()  { bad "$*"; exit 1; }
# ask "question" default -> $REPLY
ask() {
    local shown="Enter = $2"; [ -n "$2" ] || shown="Enter = none"
    if $YES; then REPLY=$2; echo "  $1 [$shown]: $2"; return; fi
    local r; read -r -p "  $1 [$shown]: " r </dev/tty || true; REPLY=${r:-$2}
}
# confirm "question" y|n -> exit status
confirm() {
    local d=${2:-y} r hint; [ "$d" = y ] && hint="Y/n, Enter = yes" || hint="y/N, Enter = no"
    if $YES; then echo "  $1 [$hint]: $d"; [ "$d" = y ]; return; fi
    read -r -p "  $1 [$hint]: " r </dev/tty || true; r=${r:-$d}
    [[ $r =~ ^[Yy] ]]
}
run() { echo "    \$ $*"; "$@"; }

echo "${B}DataMix setup on LUMI${N} (repo: $REPO)"
echo "At each question, press Enter to accept the value in [brackets], or type another one."

# --- 1 checks ---------------------------------------------------------------------------
step 1 "Checks"
[[ $(hostname) == uan* ]] && ok "LUMI login node ($(hostname))" \
    || warn "not on a LUMI login node ($(hostname)); downloads and builds need a login node"
for c in git singularity tmux wget squeue sbatch; do
    command -v $c >/dev/null || die "command '$c' not found"
done
ok "git, singularity, tmux, wget, slurm commands"
branch=$(git rev-parse --abbrev-ref HEAD)
[ "$branch" = datamix ] && ok "branch datamix" \
    || warn "branch is '$branch', not 'datamix'; the DataMix configs are on datamix"
if git submodule status --recursive | grep -q '^-'; then
    warn "some submodules are not checked out"
    if ! git config --global --get url."https://github.com/".insteadOf >/dev/null \
        && confirm "Let git use HTTPS for GitHub's SSH URLs (needed without an SSH key; sets git config --global)?" y; then
        run git config --global url."https://github.com/".insteadOf git@github.com:
    fi
    confirm "Check out the submodules now?" y && run git submodule update --init --recursive \
        || die "the submodules are needed"
fi
ok "submodules"

# --- 2 project --------------------------------------------------------------------------
step 2 "LUMI project"
mapfile -t groups < <(id -Gn | tr ' ' '\n' | grep -E '^project_[0-9]+$' || true)
[ ${#groups[@]} -gt 0 ] || die "you are in no LUMI project (id -Gn)"
info "your projects: ${groups[*]}"
default=${DATAMIX_PROJECT:-}
if [ -z "$default" ]; then
    default=${groups[0]}
    for g in "${groups[@]}"; do [ "$g" = $DEFAULT_PROJECT ] && default=$g; done
fi
info "The project is billed for all jobs (training, conversion, evaluation)."
ask "Project" "$default"; PROJECT=$REPLY
[[ " ${groups[*]} " == *" $PROJECT "* ]] || warn "you are not a member of $PROJECT"
[ -d /scratch/$PROJECT ] || die "/scratch/$PROJECT does not exist"
ok "project $PROJECT"

# --- 3 paths + env file -----------------------------------------------------------------
step 3 "Paths and ~/datamix_env.sh"
info "Scratch dir: your personal directory on the project's scratch disk. The setup installs here:"
info "  .local/bin/                     uv (~30 MB)"
info "  .cache/uv/, .local/share/uv/    uv's package cache and its Python (~200 MB, ~10k files)"
info "  eval_local/                     the eval environment (~1.3 GB, ~45k files)"
info "Not in \$HOME, which has a quota of 20 GB and 100k files. Your clone and the run outputs"
info "don't have to be in this dir."
ask "Scratch dir" "${WORK:-/scratch/$PROJECT/users/$USER}"; WORK_DIR=$REPLY

if [ -n "${HF_HOME:-}" ]; then d=$HF_HOME
elif [ -d /scratch/$PROJECT/cache/hub ]; then d=/scratch/$PROJECT/cache
else d=$WORK_DIR/hf_cache; fi
echo
info "HF cache (HF_HOME): Hugging Face cache with the training tokenizer and the eval datasets."
info "The jobs run offline and read them from here. A shared project cache can be used by everyone"
info "and already has the tokenizer; a new dir starts empty (the tokenizer is downloaded in step 7,"
info "the eval datasets before each run)."
ask "HF cache" "$d"; HF_DIR=$REPLY

if [ -n "${DATAMIX_CONTAINERS:-}" ]; then d=$DATAMIX_CONTAINERS
elif [ -r /scratch/$PROJECT/containers/$SIF_TRAIN ]; then d=/scratch/$PROJECT/containers
elif [ -r /scratch/$DEFAULT_PROJECT/containers/$SIF_TRAIN ]; then d=/scratch/$DEFAULT_PROJECT/containers
else d=/scratch/$PROJECT/containers; fi
echo
info "Containers dir: where the train, convert and eval containers are (~14 GB each). The shared"
info "dir /scratch/$DEFAULT_PROJECT/containers has all three. Step 5 checks the dir you give and"
info "can build the convert container there; the other two must be copied in."
ask "Containers dir" "$d"; CONTAINERS=$REPLY

new_env=$(cat <<EOF
# DataMix on LUMI, written by config/experiments/datamix/setup_lumi.sh.
# Source it in every shell and tmux window: source ~/datamix_env.sh
# your scratch dir; uv in .local/bin, its caches in .cache, the eval env in eval_local
export WORK=$WORK_DIR
export PATH="\$WORK/.local/bin:\$PATH"
export UV_CACHE_DIR=\$WORK/.cache/uv
export UV_PYTHON_INSTALL_DIR=\$WORK/.local/share/uv/python
# HF cache: tokenizer + eval datasets, read offline by the jobs
export HF_HOME=$HF_DIR
# billed project (aux.account) and containers dir (aux.containers_dir), see datamix.yaml
export DATAMIX_PROJECT=$PROJECT
export DATAMIX_CONTAINERS=$CONTAINERS
EOF
)
if [ -f "$ENV_FILE" ] && [ "$(cat "$ENV_FILE")" = "$new_env" ]; then
    ok "$ENV_FILE is up to date"
else
    if [ -f "$ENV_FILE" ]; then
        warn "$ENV_FILE exists and differs:"; diff <(cat "$ENV_FILE") <(echo "$new_env") | sed 's/^/      /' || true
        confirm "Replace it (the old one is kept as $ENV_FILE.bak)?" y || die "keep $ENV_FILE: edit it by hand, then run this again"
        cp "$ENV_FILE" "$ENV_FILE.bak"
    else
        info "Will write $ENV_FILE:"; echo "$new_env" | sed 's/^/      /'
        confirm "Write it?" y || die "the env file is needed by the later steps"
    fi
    echo "$new_env" > "$ENV_FILE"; ok "wrote $ENV_FILE"
fi
mkdir -p "$WORK_DIR"
# shellcheck disable=SC1090
source "$ENV_FILE"

# --- 4 uv + Python env ------------------------------------------------------------------
step 4 "uv and the Python environment"
if command -v uv >/dev/null; then
    ok "uv $(uv --version | awk '{print $2}') ($(command -v uv))"
else
    info "autoexp runs in a uv environment; uv $UV_VERSION goes into $WORK/.local/bin."
    confirm "Install uv $UV_VERSION?" y || die "uv is needed"
    mkdir -p "$WORK/.local/bin"
    tmp=$(mktemp -d)
    run wget -qO "$tmp/uv.tar.gz" https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz
    run tar xzf "$tmp/uv.tar.gz" -C "$tmp"
    run mv "$tmp/uv-x86_64-unknown-linux-gnu/uv" "$tmp/uv-x86_64-unknown-linux-gnu/uvx" "$WORK/.local/bin/"
    rm -rf "$tmp"
    ok "uv $(uv --version | awk '{print $2}')"
fi
run uv sync -q && ok "Python environment .venv"

# --- 5 containers -----------------------------------------------------------------------
step 5 "Containers in $CONTAINERS"
have_train=false; have_eval=false
for s in train:$SIF_TRAIN convert:$SIF_CONVERT eval:$SIF_EVAL; do
    stage=${s%%:*}; f=$CONTAINERS/${s#*:}
    if [ -r "$f" ]; then
        ok "$stage: $f"
        case $stage in train) have_train=true ;; eval) have_eval=true ;; esac
    else
        bad "$stage: $f not found or not readable"
    fi
done
if ! $have_train || ! $have_eval; then
    info "The train and eval containers can't be built here. Copy them from"
    info "/scratch/$DEFAULT_PROJECT/containers/ (members of $DEFAULT_PROJECT can read it) into $CONTAINERS,"
    info "or set another containers dir (step 3)."
    $have_train || MISSING+=("train container $CONTAINERS/$SIF_TRAIN")
    $have_eval || MISSING+=("eval container $CONTAINERS/$SIF_EVAL")
fi
if [ ! -r "$CONTAINERS/$SIF_CONVERT" ]; then
    if $have_train; then
        info "The convert container is the train container + nvidia-modelopt and pulp (Megatron-Bridge"
        info "needs them). It can be built here (~15 min, in a sandbox on /tmp; LUMI has no fakeroot)."
        if confirm "Build it into $CONTAINERS/$SIF_CONVERT?" y; then
            mkdir -p "$CONTAINERS"
            run bash container/megatron/build_lumi_bridge_container.sh "$CONTAINERS/$SIF_CONVERT" "$CONTAINERS/$SIF_TRAIN" \
                && ok "convert container built" || MISSING+=("convert container (build failed)")
        else MISSING+=("convert container $CONTAINERS/$SIF_CONVERT"); fi
    else MISSING+=("convert container (needs the train container first)"); fi
fi

# --- 6 eval env -------------------------------------------------------------------------
step 6 "Eval environment ~/eval_local"
if [ -x "$HOME/eval_local/bin/oellm-eval" ] && [ -x "$HOME/eval_venv_stub/bin/python" ]; then
    ok "~/eval_local -> $(readlink -f "$HOME/eval_local")"
elif ! $have_eval; then
    bad "needs the eval container (step 5)"; MISSING+=("eval env ~/eval_local")
else
    info "oellm-eval + lm-eval are installed on top of the eval container into $WORK/eval_local"
    info "(~44k files, ~10 min), with ~/eval_local linking there. The installer also patches"
    info "submodules/Megatron-Bridge in place so that model bridges which don't import with the"
    info "OpenEuroLLM Megatron-LM fork are skipped; git then shows ~35 modified files there (intended)."
    if confirm "Install it?" y; then
        run env EVAL_LOCAL_DIR="$WORK/eval_local" OELLM_EVAL_SIF="$CONTAINERS/$SIF_EVAL" \
            bash scripts/install_eval_env.sh --cluster lumi && ok "eval env installed" \
            || MISSING+=("eval env (installer failed)")
    else MISSING+=("eval env ~/eval_local"); fi
fi

# --- 7 tokenizer ------------------------------------------------------------------------
step 7 "Tokenizer $TOKENIZER in $HF_HOME"
tok=$HF_HOME/hub/models--${TOKENIZER/\//--}
if [ -f "$tok/refs/main" ]; then
    ok "$tok"
else
    info "Training loads the tokenizer offline from the HF cache."
    if confirm "Download $TOKENIZER into $HF_HOME?" y; then
        mkdir -p "$HF_HOME"
        run env HF_HUB_OFFLINE=0 uv run python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$TOKENIZER'))" \
            && ok "tokenizer downloaded" || MISSING+=("tokenizer $TOKENIZER in $HF_HOME")
    else MISSING+=("tokenizer $TOKENIZER in $HF_HOME"); fi
fi

# --- 8 test run -------------------------------------------------------------------------
step 8 "Test run"
if [ ${#MISSING[@]} -gt 0 ]; then
    bad "skipped, still missing:"; printf '      - %s\n' "${MISSING[@]}"
    info "Fix them and run this script again; finished steps are skipped."
    exit 1
fi
ok "everything for the test run is in place"
# printed at every exit from here on: how to start your own run
next_steps() {
    echo
    info "Your own run (see lumi_autoexp_installation_guide.md, \"Your own run\"):"
    info "  1. set aux.name, aux.data_path and aux.eval_task_groups in chain-datamix.yaml"
    info "  2. download the eval datasets of your config, once and after every change of the task"
    info "     groups (the eval jobs run offline and autoexp doesn't download them):"
    info "       bash config/experiments/datamix/prefetch_eval.sh --config-name experiments/datamix/chain-datamix"
    info "  3. in tmux, after source ~/datamix_env.sh:"
    info "       uv run python scripts/run_autoexp.py --config-name experiments/datamix/chain-datamix"
}
$TEST || { info "--no-test: not starting it."; next_steps; exit 0; }
info "The test runs the whole chain small on dev-g: per scale 100 training iterations,"
info "conversion and evaluation of 2 checkpoints, one job at a time (dev-g allows one job per"
info "project). One scale takes ~1 h. Before it starts, it downloads the eval datasets of the"
info "test (prefetch_eval.sh, ~1 min). It needs training data: the prefix of a .bin/.idx pair"
info "tokenized with $TOKENIZER (path without .bin)."
if [ -z "$DATA" ] && ! $YES; then ask "Training data prefix (empty = skip the test)" ""; DATA=$REPLY; fi
[ -n "$DATA" ] || { info "No data given, not starting the test."; next_steps; exit 0; }
[ -r "$DATA.bin" ] && [ -r "$DATA.idx" ] || die "$DATA.bin / $DATA.idx not found or not readable"
ask "Scales" "$SCALES"; SCALES=$REPLY
busy=$(squeue -h -A "$PROJECT" -p dev-g -o '%u %i %j' || true)
[ -z "$busy" ] || warn "dev-g is in use in $PROJECT, the test waits for it: $busy"
args=("aux.data_path=[$DATA]" "aux.scales=[$SCALES]")
if confirm "Start the test (in tmux session datamix-test)?" y; then
    run bash config/experiments/datamix/run_test.sh "${args[@]}"
else
    info "Start it later with:"; info "  source ~/datamix_env.sh"
    info "  bash config/experiments/datamix/run_test.sh $(printf "'%s' " "${args[@]}")"
fi
next_steps
