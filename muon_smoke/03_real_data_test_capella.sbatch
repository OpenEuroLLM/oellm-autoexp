#!/bin/bash
#SBATCH --job-name=muon_real_data_test
#SBATCH --partition=capella-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100_1g.12gb:1
#SBATCH --time=00:20:00
#SBATCH --output=/home/h1/sala597i/oellm-autoexp/muon_smoke/logs/04_real_data_step_%j.log

# Small real-data test: same as 02_mock_train_capella.sbatch (RUNG C, already
# PASSED with mock_data), but now against the converted Nemotron train.bin/idx
# (mock_data: false, data_path set — see muon_50M_50BT.yaml + DATA_CONVERSION.md).
# Goal: just confirm GPTDataset actually builds/loads/iterates on real data
# without crashing — cancel the job manually once a couple of real iterations
# with sane lm loss show up in the log, same as we did for RUNG C mock-data.
# All the proven fixes from RUNG C carried over unchanged (see that script's
# comments for why each one is needed): TORCHDYNAMO_DISABLE, PYTHONPATH with
# muon-pylibs, micro_batch_size=4, --array-subset 0.

set -eo pipefail
cd /home/h1/sala597i/oellm-autoexp

SANDBOX=/data/horse/ws/sala597i-slaing/containers/.tmp_sandbox/oellm_sandboxv514iw4i/sandbox

singularity exec --underlay --nv -B /data/horse/ws/sala597i-slaing \
  --env PYTHONPATH=$PWD:$PWD/muon-pylibs:submodules/Megatron-LM \
  --env TORCHDYNAMO_DISABLE=1 \
  "$SANDBOX" python3 -u scripts/run_autoexp.py \
    --config-name experiments/laingsam/muon_50M_50BT \
    backend.megatron.eval_iters=0 \
    backend.megatron.save_interval=1000000 \
    backend.megatron.micro_batch_size=4 \
    --array-subset 0 \
    --local
# Note: train_iters CLI override is still known to not take effect (the
# sweep's own 50B-token formula wins) — same cosmetic issue flagged in
# RESULT.md. Not fixed here since it's harmless for a "watch a few iterations
# then cancel" test; only matters for a real unattended full run.
