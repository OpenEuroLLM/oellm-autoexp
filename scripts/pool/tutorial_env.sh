# Environment of the pool tutorial (docs/sub_slurm_jobs_tutorial.md).
# `source scripts/pool/tutorial_env.sh` from the repo root, in EVERY terminal you use:
# it puts the local fake SLURM (tests/fakeslurm/bin) first on PATH, so `sbatch`, `squeue`, ...
# act on local processes. A terminal without it has no sbatch, and every host fails to start.
cd "$(git rev-parse --show-toplevel)" || return
export R=$PWD/dump/pool_tutorial PYTHONPATH=. FAKESLURM_DIR=$PWD/dump/pool_tutorial/fs
export POOL_TEST_ROOT=$R
case ":$PATH:" in
  *":$PWD/tests/fakeslurm/bin:"*) ;;
  *) export PATH=$PWD/tests/fakeslurm/bin:$PATH ;;
esac
mkdir -p "$R/fs"
MS=$R/ms
POOL=$R/pool
SESSION=$MS/pool_localpool
# enqueue a dummy job (scripts/sigterm_dummy_trainer.py: Megatron's SIGTERM/checkpoint semantics)
enq() { python scripts/run_autoexp.py --no-monitor --monitor-state-dir "$MS" --pool "$POOL" \
  --config-name experiments/tests/pool_local_dummy "$@" 2>&1 | grep -E "^Enqueued|CANNOT|Error"; }
# (re)create the pool from the local host config
mkpool() { python scripts/run_autoexp.py --no-monitor --monitor-state-dir "$MS" \
  --config-name experiments/tests/pool_local_host "$@" 2>&1 | grep -E "Pool ready|Error"; }
mon() { python scripts/monitor_autoexp.py --session-dir "$SESSION" --poll-interval 2; }
ctl() { python scripts/pool_ctl.py "$POOL" "$@"; }
crash() { python tests/fakeslurm/fakeslurm.py _crash "$@"; }
echo "pool tutorial: sbatch -> $(command -v sbatch)  (must be .../tests/fakeslurm/bin/sbatch)"
