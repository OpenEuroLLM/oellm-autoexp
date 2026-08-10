#!/bin/bash
#SBATCH --job-name=muon_qwen3_real_data_test
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:20:00
#SBATCH --output=/home/h1/sala597i/oellm-autoexp/muon_smoke/logs/04_qwen3_real_data_step_%j.log

# Real Qwen3 0.1B_ne architecture (multilingual_scaling's model file) + Muon,
# against the real converted Nemotron data — same goal/pattern as
# 03_real_data_test_capella.sbatch (RUNG C real-data, already PASSED on the
# generic 50M proxy model): confirm GPTDataset + real Qwen3 arch + Muon
# optimizer all build/load/train together without crashing. Cancel manually
# once a couple of real iterations with sane lm loss show up in the log.
#
# ~backend.megatron.norm_epsilon override: qwen3_dense_0.1B_ne.yaml (SHARED
# model file, also used by slaing00/multilingual_scaling — not ours to edit)
# still uses the pre-v0.17 field name `norm_epsilon`. The yaml's own
# layernorm_epsilon override doesn't remove that stale key from the merged
# config, and compoconf rejects unknown keys regardless of value — so it must
# be deleted via this Hydra `~` override at invocation time instead of a file
# edit. See muon_qwen3_0.1B_longctx.yaml's comment for detail.

set -eo pipefail
cd /home/h1/sala597i/oellm-autoexp

SANDBOX=/data/horse/ws/sala597i-slaing/containers/.tmp_sandbox/oellm_sandboxv514iw4i/sandbox

singularity exec --underlay --nv -B /data/horse/ws/sala597i-slaing \
  --env PYTHONPATH=$PWD:$PWD/muon-pylibs:submodules/Megatron-LM \
  --env TORCHDYNAMO_DISABLE=1 \
  "$SANDBOX" python3 -u scripts/run_autoexp.py \
    --config-name experiments/laingsam/muon_qwen3_0.1B_longctx \
    '~backend.megatron.norm_epsilon' \
    backend.megatron.eval_iters=0 \
    backend.megatron.save_interval=20 \
    --array-subset 0 \
    --local
# save_interval=20 (vs the yaml's default 1000000): deliberately low so a real
# checkpoint gets written within this short smoke run, for a visible sanity
# check that saving works with this model+optimizer combo. Lands in the same
# flat ./output/checkpoints as every other --local run in this checkout (see
# ++job.base_output_dir=./output override) — not experiment-namespaced, but
# still entirely within your own repo checkout.
# Note: train_iters CLI override is still known to not take effect (the
# sweep's own token-budget formula wins) — same cosmetic issue flagged in
# RESULT.md. Harmless for a "watch a few iterations then cancel" test.
