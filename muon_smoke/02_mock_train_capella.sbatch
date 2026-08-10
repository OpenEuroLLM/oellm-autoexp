#!/bin/bash
#SBATCH --job-name=muon_mock_train
#SBATCH --partition=capella-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100_1g.12gb:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/h1/sala597i/oellm-autoexp/muon_smoke/logs/03_smoke_step_%j.log

# RUNG C of NEW_CLUSTER_SETUP.md's smoke ladder: real Megatron training steps
# (still mock_data=true) through the full muon_50M_50BT.yaml config. Proves
# Megatron actually builds the Muon+Adam optimizer and takes training steps,
# not just the isolated Newton-Schulz check from 01_smoke_capella.sbatch.

set -eo pipefail
cd /home/h1/sala597i/oellm-autoexp

SANDBOX=/data/horse/ws/sala597i-slaing/containers/.tmp_sandbox/oellm_sandboxv514iw4i/sandbox

singularity exec --underlay --nv -B /data/horse/ws/sala597i-slaing \
  --env PYTHONPATH=$PWD:$PWD/muon-pylibs:submodules/Megatron-LM \
  --env TORCHDYNAMO_DISABLE=1 \
  "$SANDBOX" python3 -u scripts/run_autoexp.py \
    --config-name experiments/laingsam/muon_50M_50BT \
    backend.megatron.train_iters=10 \
    backend.megatron.eval_iters=0 \
    backend.megatron.save_interval=1000000 \
    backend.megatron.micro_batch_size=4 \
    --array-subset 0 \
    --local
# micro_batch_size=4 (not touching the base 16 meant for real multi-GPU runs):
# OOM'd on the output-layer vocab projection ([seq_len=2048, mbs=16, vocab=50304]
# logits, ~3GB+) on this 1g.12gb MIG slice (~10.75GB total, only 2.78GB free
# when it needed 3.07GB more). 4x smaller batch -> proportionally smaller
# activation tensors; only affects this smoke run via CLI override.
# backend.megatron.disable_jit_fuser=true does NOT work here: bias_swiglu is
# decorated with @jit_fuser at *import time* (megatron/core/jit.py binds
# jit_fuser=torch.compile unconditionally when the module loads), long before
# args.disable_jit_fuser is ever parsed at runtime — reassigning the name
# later doesn't un-decorate the already-wrapped function. TORCHDYNAMO_DISABLE
# is read by torch.compile's call-time machinery instead of at decoration
# time, so it correctly no-ops every torch.compile call process-wide,
# regardless of import order. Pure speed opt either way — no architecture change.
# Note: dropped --no-monitor — submit_jobs() only registers the plan; the
# actual client.submit() (launching the process) happens inside run_loop(),
# which --no-monitor skips entirely. Confirmed via job.json showing
# "submitted": False after the first (silent, instant "success") attempt.
# --array-subset 0 restricts to just the "stable" stage (sweep index 0) so
# we don't sit waiting on the "decay50B" stage's start_condition, which
# depends on a checkpoint iteration the 10-iter override will never reach.
