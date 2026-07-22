# Muon 50M smoke test — self-run instructions

Do STEP 1 and STEP 2 in any order; both must be done before STEP 3.

## STEP 1: clear any stale AITER JIT lock (safe, your home dir)
A killed run can leave AMD's kernel-JIT lock behind, making the next run hang
on "[aiter] waiting for baton release". This removes ONLY the empty lock file
(0 bytes) and keeps your compiled kernels in ck/:

    rm -f /users/laingsam/.aiter/jit/build/lock_module_aiter_enum

## STEP 2: get an interactive allocation
Single line. Note the job id it prints (`Granted job allocation <JOBID>`) —
it is DIFFERENT every time you allocate:

    salloc --no-shell --account=project_465002530 --partition=dev-g --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=7 --mem=60G --time=00:30:00

## STEP 3: run the GPU smoke test (Muon Newton-Schulz on ROCm)
Replace <JOBID> with the number from your CURRENT salloc (STEP 2).
Single line (copy the whole thing):

    srun --jobid=19922159 --overlap --ntasks=1 --gpus-per-node=1 singularity exec -B /pfs -B /scratch -B /flash -B /opt/cray -B /var/spool/slurmd -B /appl --overlay /scratch/project_465002530/users/laingsam/oellm-muon/muon-overlay.img /scratch/project_465002530/containers/MegatronTrainingLumi_x86_64.sif bash -c 'cd /scratch/project_465002530/users/laingsam/oellm-muon && export TRITON_HOME=/tmp/tri HF_HOME=/tmp/hf HF_HUB_OFFLINE=1 CC=gcc-12 CXX=g++-12 PYTHONPATH=submodules/Megatron-LM && python -u muon_smoke/muon_gpu_smoke.py' 2>&1 | tee /scratch/project_465002530/users/laingsam/oellm-muon/muon_smoke/01_optimizer_construct.log

PASS looks like:
    device: AMD Instinct MI250X
    single-rank process group: OK
    >>> Newton-Schulz on GPU: OK
    HAVE_EMERGING_OPTIMIZERS: True
    MUON GPU SMOKE: PASS

First real run compiles AITER kernels (a few minutes, one-time; cached after).
`python -u` streams output live so you can see progress, not a blank hang.

## STEP 4: free the allocation when done
    scancel <JOBID>

## Command notes (LUMI gotchas baked into STEP 3)
1. NO `--rocm`: it injects host GPU libs needing GLIBC_2.33 the container lacks.
   The container ships its own ROCm; GPU comes from the binds + /dev passthrough.
2. `bash -c`, NOT `bash -lc`: a login shell sources the profile, which tries to
   init Cray Lmod (needs /usr/bin/lua5.3, absent in-container) -> harmless but
   noisy "bad interpreter" error. `-c` skips the profile; we set env ourselves.
3. `python -u`: unbuffered, so output isn't lost if the step is killed.
Matches templates/debug.slurm (the framework's own LUMI launcher).

## What this proves
Muon's core orthogonalization math runs on this ROCm stack, and Megatron v0.17
sees the optimizer. This is the "does Muon work here at all" gate.

## Next (full pipeline — after STEP 3 passes)
Turning `optimizer: muon` on inside the autoexp pipeline for a real training
step needs: regenerate the v0.17 schema, then run_autoexp with the config
`experiments/laingsam/muon_50M_50BT`. Diana's data lives under
/scratch/project_462000963/... (needs read access). Do this after STEP 3 passes.

## DIAGNOSTIC: which C++ compiler does the container have?
The full Megatron v0.17 import (Part 2 of the smoke) fails because AITER can't
find a C++ compiler (`which c++` -> not found). Run this to see what IS there
(reuse your live JOBID from squeue, or re-salloc first). Single line:

    srun --jobid=19921691 --overlap --ntasks=1 singularity exec -B /pfs -B /scratch -B /opt/cray -B /appl /scratch/project_465002530/containers/MegatronTrainingLumi_x86_64.sif bash -c 'echo "PATH=$PATH"; for c in c++ g++ gcc g++-12 gcc-12 hipcc clang++; do printf "%-10s " "$c:"; which $c 2>/dev/null || echo MISSING; done'

Paste the output; then CC/CXX get set to whatever exists and STEP 3 is rerun.


  sbatch /scratch/project_465002530/users/laingsam/oellm-muon/muon_smoke/short_run.sbatch


# ============================================================================
# NEXT TEST (2026-07-16): AutoExp container + real Qwen3 0.1B_ne arch
# ============================================================================
# Tests the two things that DON'T need the blocked multilingual data:
#   1. does the muon overlay port to AutoExp_2026-07-02.sif (the container the
#      real multilingual runs actually use)?
#   2. does Muon train the REAL Qwen3 0.1B_ne architecture, sharded?
# Mock data, 20 iters, 4 GPUs, ~0.2 GPU-hr. Fails fast: Part 1 exits before
# any GPU work if the overlay doesn't port.
#
# RUN THIS (single line):

    sbatch /scratch/project_465002530/users/laingsam/oellm-muon/muon_smoke/04_autoexp_qwen3_muon.sbatch

# Then check the log (single line; replace <JOBID> with what sbatch prints):

    tail -40 /scratch/project_465002530/users/laingsam/oellm-muon/muon_smoke/04_autoexp_qwen3_<JOBID>.log

# PASS looks like:
#   >>> OVERLAY PORTS OK
#   iteration 1/20 | lm loss: 1.2E+01 ...
#   use_layer_wise_distributed_optimizer ............ True
#
# If Part 1 prints "OVERLAY DOES NOT PORT", the overlay needs rebuilding
# against the AutoExp .sif -- and no GPU hours are burned finding that out.

# ----------------------------------------------------------------------------
# RESOLVED 2026-07-16: the "use_distributed_optimizer = False" scare
# ----------------------------------------------------------------------------
# NOT a bug. Megatron arguments.py:1464-1466 -- for emerging optimizers (muon),
# --use-distributed-optimizer is TRANSLATED, not ignored:
#     args.use_layer_wise_distributed_optimizer = True
#     args.use_distributed_optimizer = False
# Muon shards via the LAYER-WISE distributed optimizer. Job 19928413 confirmed
# use_layer_wise_distributed_optimizer = True. Sharding worked all along.