# Muon pipeline — in-between steps (MOCKUP / DRAFT)

These are DRAFTS to show the shape. Exact override keys get confirmed as we run.
Reuses the proven recipe: no --rocm, the bind list, CC=gcc-12 CXX=g++-12,
--overlay muon-overlay.img, bash -c, python -u. <JOBID> = current salloc.

Shorthands used below:
  SIF   = /scratch/project_465002530/containers/MegatronTrainingLumi_x86_64.sif
  OVL   = /scratch/project_465002530/users/laingsam/oellm-muon/muon-overlay.img
  BINDS = -B /pfs -B /scratch -B /flash -B /opt/cray -B /var/spool/slurmd -B /appl
  ENV   = export TRITON_HOME=/tmp/tri HF_HOME=/tmp/hf HF_HUB_OFFLINE=1 CC=gcc-12 CXX=g++-12 PYTHONPATH=.:submodules/Megatron-LM


## RUNG A — regenerate the v0.17 schema (one-time, in allocation)
Makes optimizer=muon + muon_* valid autoexp config keys. Writes config_schema.py,
cli_metadata.py, base_defaults.yaml in YOUR worktree. Fill in <JOBID>. Single line:

    srun --jobid=<JOBID> --overlap --ntasks=1 singularity exec -B /pfs -B /scratch -B /flash -B /opt/cray -B /var/spool/slurmd -B /appl --overlay /scratch/project_465002530/users/laingsam/oellm-muon/muon-overlay.img /scratch/project_465002530/containers/MegatronTrainingLumi_x86_64.sif bash -c 'cd /scratch/project_465002530/users/laingsam/oellm-muon && export TRITON_HOME=/tmp/tri HF_HOME=/tmp/hf HF_HUB_OFFLINE=1 CC=gcc-12 CXX=g++-12 PYTHONPATH=.:submodules/Megatron-LM && python -u scripts/generate_megatron_config.py && python -u scripts/generate_megatron_dataclass.py' 2>&1 | tee /scratch/project_465002530/users/laingsam/oellm-muon/muon_smoke/00_schema_regen.log

After it runs, check muon is now a valid key:
    grep -n "muon" config/backend/megatron/config_schema.py | head


## RUNG B — dry-run render (renders the command, submits nothing)
Confirms the config resolves, muon flags appear, and shows how the sweep expands.
Fill in <JOBID>. Single line:

    srun --jobid=<JOBID> --overlap --ntasks=1 singularity exec -B /pfs -B /scratch -B /flash -B /opt/cray -B /var/spool/slurmd -B /appl --overlay /scratch/project_465002530/users/laingsam/oellm-muon/muon-overlay.img /scratch/project_465002530/containers/MegatronTrainingLumi_x86_64.sif bash -c 'cd /scratch/project_465002530/users/laingsam/oellm-muon && export CC=gcc-12 CXX=g++-12 PYTHONPATH=.:submodules/Megatron-LM && python -u scripts/run_autoexp.py --config-name experiments/laingsam/muon_50M_50BT --dry-run' 2>&1 | tee /scratch/project_465002530/users/laingsam/oellm-muon/muon_smoke/02_dry_run.log

Look in the rendered command for:  --optimizer muon  --muon-momentum 0.9  ... etc.
Then PASTE the output back — it tells us exactly how to submit the short run (RUNG C).


## RUNG C — mock-data smoke (~10 iters, synthetic data, no real data)
THE key in-between: proves Megatron builds the Muon+Adam optimizer and takes real
train steps. Two ways:

### C1 (through autoexp): render an sbatch WITHOUT submitting, then inspect/run it
    cd /scratch/project_465002530/users/laingsam/oellm-muon && PYTHONPATH=. python scripts/run_autoexp.py --config-name experiments/laingsam/muon_50M_50BT backend.megatron.mock_data=true backend.megatron.train_iters=10 backend.megatron.eval_iters=0 backend.megatron.save_interval=1000000 --no-submit
    # -> writes an sbatch to disk; inspect it, then sbatch it (or run its srun line in our allocation)

### C2 (direct, most controllable): take the command RUNG B printed, run it here
Run the rendered pretrain_gpt.py directly in the allocation with mock data + few iters:

    srun --jobid=<JOBID> --overlap --ntasks=1 --gpus-per-node=1 singularity exec $BINDS --overlay $OVL $SIF bash -c 'cd /scratch/project_465002530/users/laingsam/oellm-muon && export TRITON_HOME=/tmp/tri HF_HOME=/tmp/hf HF_HUB_OFFLINE=1 CC=gcc-12 CXX=g++-12 PYTHONPATH=submodules/Megatron-LM && python -u submodules/Megatron-LM/pretrain_gpt.py <ALL THE FLAGS FROM RUNG B> --mock-data --train-iters 10 --eval-iters 0' 2>&1 | tee muon_smoke/03_mock_train.log

PASS = you see `iteration  1/10 ... lm loss: X.XXX` lines, loss finite/decreasing,
no crash in optimizer construction. That means Muon trains the 50M model.


## RUNG D — real data (after C passes)
Same as C but drop --mock-data; needs read access to /scratch/project_462000963/...
Then it's the actual experiment (via autoexp submit, multi-stage sweep, etc.).


## Notes / open items to confirm when we run
- The config has an inline sweep (stable+decay). For a single smoke we likely
  override to one stage or disable the sweep; confirm at RUNG B from what it renders.
- RUNG A's generators may pull in apex->aiter (hence CC/CXX); first run recompiles.
- torchrun vs plain python: single-GPU can use plain python; multi-GPU needs torchrun.
