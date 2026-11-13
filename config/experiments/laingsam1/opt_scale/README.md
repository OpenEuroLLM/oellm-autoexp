# opt_scale: optimizer scaling-law runs on JUPITER — setup guide

Optimizer scaling-law study (AdamW, Muon, AngularMuown, muown_classical) on the team's hybrid
Gated-DeltaNet MoE (Swagatam's architecture), trained with the team Megatron
`oellm/v0.19-hybrid` on JUPITER GH200, on the OpenEuroLLM v8ml data mix.

Intended as a guide/setup document, edit if something proves incorrect/incomplete. The lab notebook with all smoke-test
results and debugging history is `smoke_test/README.md`.

## What's here

| Path | What |
|---|---|
| `setup_jupiter_hybrid.sh` | one-time setup: Megatron checkout, emerging_optimizers, venv, dirs |
| `smoke_test/05_hybrid_env_check.sh` | checks the environment inside the container (login node, free) |
| `smoke_test/06_hybrid100m_team_arch.yaml` | **reference config**: 100M-active hybrid MoE, 4 optimizers, 100 steps |
| `smoke_test/08_throughput_gbs256.yaml` | throughput test at gbs 256 (micro-batch / kernel variants, AdamW) |
| `hyb100m/07_hyb100m_20N.yaml` | first real run: 100M, D = 20N (2.02B tokens), 4 optimizers |
| `data/datamix_v8ml_flagfull_jsc.txt` | v8ml data mix (Megatron `--data-args-path`), made by `data/make_datamix_v8ml_flagfull.py` |
| `smoke_test/01-04_*.yaml`, `setup_jupiter.sh`, `00_env_check.sh` | older smoke tests on a previous Megatron fork + container; history only, don't use |

## Stack (pinned)

| What | Value |
|---|---|
| Megatron | `OpenEuroLLM/NVIDIA-Megatron-LM` branch `oellm/v0.19-hybrid` @ `013763961` (detached checkout) |
| Container | `/e/project1/e-sta-openeurollm/container/MegatronTraining-JUPITER-te218-fa3_aarch64_202608280932.sif` (torch 2.11, TE 2.18, flash-attn 3, fla 0.4.2). Python **must** be `/opt/venv/bin/python` (configs set it) |
| emerging_optimizers | 0.2.0 in `$OPT_SCALE_BASE/pyext`, put first on `PYTHONPATH` (the container's 0.1.0 makes Megatron silently drop Muon) |
| Tokenizer | `/e/data1/datasets/playground/mmlaion/oellm_testing/tokenizers/tokenizer-256k-agent` |
| Launcher | `pretrain_hybrid.py`, pattern `LELELE*E` x 4 (L = Gated DeltaNet, * = gated attention, E = MoE) |

## 0. Accounts and access (once; this is the slow part, start early)

Everything is requested in **JuDoor** (https://judoor.fz-juelich.de). Approvals come from the
project PIs / JSC and can take a few days.

### 0a. JSC account and login to JUPITER
1. Create a JuDoor account (institutional e-mail) and set up 2-factor authentication (MFA).
2. You get a JSC user name like `smith1`. You can only log in to JUPITER once you are member of a
   project with JUPITER access (0b) and have accepted that system's usage agreement in JuDoor.
3. Upload an SSH public key in JuDoor (Manage SSH keys). JSC requires a `from=` clause (allowed
   source IPs/domains), see the JuDoor key page. Then: `ssh <user>@login.jupiter.fz-juelich.de`.

### 0b. Project memberships (JuDoor -> "Join a project", give the project ID)

| Unix group (= JuDoor project) | Why you need it | Who approves | Without it |
|---|---|---|---|
| `e-aif-2026sc04-211` | compute time on the booster (Slurm `--account`) + scratch space `/e/scratch/e-aif-2026sc04-211/users/<you>` | PI `klein11` | no jobs, no scratch dir |
| `e-sta-openeurollm` | the container (`/e/project1/e-sta-openeurollm/container/`) and 81 % of the v8ml mix (flag collection in `/e/fscratch/e-sta-openeurollm/training_v2/`) | PI `jitsev1` | setup script stops: "Cannot read ...sif" |
| `datasets` | tokenizer + remaining 19 % of the mix (`/e/data1/datasets/playground/mmlaion/oellm_testing/`) | JSC datasets project (request in JuDoor, project `datasets`) | data loading fails at job start |

Not needed for this setup: `mmlaion`, `e-ext-2025e02-108` (only used by the legacy smoke tests 01-04).

If you use a different compute project than `e-aif-2026sc04-211`, set `OPT_SCALE_ACCOUNT` and put
`OPT_SCALE_BASE` in that project's scratch (step 1). You still need the two data groups.

### 0c. Container access
JSC only lets members of the `container` group run Apptainer. Request it in JuDoor
("Software" / "Request access to restricted software" -> **Container Runtime Engine**).
Without it `apptainer exec` fails with a permission error.

### 0d. GitHub
Both repos are public (`OpenEuroLLM/oellm-autoexp`, `OpenEuroLLM/NVIDIA-Megatron-LM`), so cloning
needs no GitHub account. You only need one (with write access to the OpenEuroLLM org + an SSH key
on JUPITER registered at GitHub) if you want to push your own branch.

### 0e. Check you have everything
Group changes only apply to **new** logins: log out and back in first.
```bash
for g in e-aif-2026sc04-211 e-sta-openeurollm datasets container; do
  id -Gn | tr ' ' '\n' | grep -qx $g && echo "OK      $g" || echo "MISSING $g"; done
test -r /e/project1/e-sta-openeurollm/container/MegatronTraining-JUPITER-te218-fa3_aarch64_202608280932.sif && echo "OK      container image readable"
test -r /e/data1/datasets/playground/mmlaion/oellm_testing/tokenizers/tokenizer-256k-agent/tokenizer.json && echo "OK      tokenizer readable"
test -r /e/fscratch/e-sta-openeurollm/training_v2/training/collection/flag && echo "OK      flag data readable"
test -w /e/scratch/e-aif-2026sc04-211/users && echo "OK      can create your scratch dir"
```
All lines must say OK before you continue.

## 1. Environment variables (add to `~/.bashrc`)

```bash
export OPT_SCALE_BASE=/e/scratch/<project>/users/<you>   # your root for everything below
export HF_HOME=$OPT_SCALE_BASE/hf_home                   # required by the Slurm templates
# only if you are NOT on e-aif-2026sc04-211:
# export OPT_SCALE_ACCOUNT=<your-compute-project>
# only if your oellm-autoexp clone is NOT at $OPT_SCALE_BASE/oellm-autoexp:
# export OPT_SCALE_REPO=/path/to/oellm-autoexp
```

All configs read these (`backend.aux.base`, `backend.aux.repo_dir`, `slurm.sbatch.account`); unset,
they fall back to Sam's paths. Keep big things on scratch: `$HOME` quota is small.

## 2. Repo + setup (login node, once)

```bash
cd $OPT_SCALE_BASE
git clone -b optimizer_scaling_laws https://github.com/OpenEuroLLM/oellm-autoexp.git
cd oellm-autoexp
bash config/experiments/laingsam1/opt_scale/setup_jupiter_hybrid.sh
source $OPT_SCALE_BASE/venvs/autoexp/bin/activate        # every new shell
```

Needs `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`). The script never deletes anything
and is safe to re-run.

## 3. Environment check (login node, free)

```bash
bash config/experiments/laingsam1/opt_scale/smoke_test/05_hybrid_env_check.sh
```

Must end with `ENV CHECK PASSED` (9 checks: pinned + clean Megatron, Megatron loaded from your
checkout, TE >= 2.18, emerging_optimizers 0.2.0 from pyext, Muon + AngularMuown available, fla
available, one Gated DeltaNet kernel gives finite output). The first run takes a few minutes
(container start, imports, Triton compile); it can look frozen after the version lines.

## 4. Dry run, then submit (always one job first)

```bash
E=experiments/laingsam1/opt_scale/smoke_test
python scripts/run_autoexp.py --config-name $E/06_hybrid100m_team_arch --dry-run
```

The dry run only renders `job.sbatch` files into `$OPT_SCALE_BASE/experiments/...`; nothing is
submitted. Look at one: account, `--hybrid-layer-pattern`, your Megatron path, the te218 sif.

```bash
# job 0 alone (it builds your data-index cache, ~25-30 min the first time -> 1 h limit)
python scripts/run_autoexp.py --config-name $E/06_hybrid100m_team_arch \
  --array-subset 0 --submit-and-exit slurm.sbatch.time=01:00:00
# when job 0 is training (step lines in the log):
python scripts/run_autoexp.py --config-name $E/06_hybrid100m_team_arch \
  --array-subset 1-3 --submit-and-exit
```

Sweep indices in 06/07: 0 adam, 1 muon, 2 angular_muown, 3 muown_classical.

Logs: `$OPT_SCALE_BASE/logs/<job>-<jobid>.{out,err}` (training output mostly in `.out`, errors in
`.err`). Outputs (tensorboard, offline wandb, checkpoints): `$OPT_SCALE_BASE/experiments/opt_scale/...`.

Expected for 06 (1 node, 100 steps): ~5.6-6.4 s/step, ~36-40 TFLOP/s/GPU, peak ~48 GB/GPU,
val loss at step 100 roughly adam 6.4, angular_muown 4.9, muown_classical 5.3.

## Guardrails (don't burn compute)

- **Always** `--dry-run` first, then `--array-subset 0`, then the rest.
- **Always** `--submit-and-exit`: without it the script keeps monitoring and may auto-resubmit.
- Keep `slurm.sbatch.time` tight: cost cap = nodes x 4 GPUs x time limit.
- Cost of what you used (live): 
  ```bash
  sacct -u $USER -S 2026-11-01 -X -n -o ElapsedRaw,AllocTRES%60 \
    | awk '{match($2,/gres\/gpu=([0-9]+)/,m); s+=$1*m[1]} END{printf "%.1f GPU-h\n", s/3600}'
  jutil project cpuquota -p e-aif-2026sc04-211    # project total, core-h (288 core-h = 1 node-h), lags ~1 day
  ```

## Writing new configs: gotchas

- Copy 06 (or 07) and change only what you need. Paths go through `${backend.aux.base}`.
- Megatron flags the autoexp schema doesn't know (`--hybrid-layer-pattern`, `--linear-beta-adam`,
  `--optimizer angular_muown|muown_classical`, `--muon-qkv-headwise`, ...) go into
  `backend.aux.arch_args` / `backend.aux.opt_args`, which are appended to the launcher command.
- Unknown keys under `backend.megatron` are rejected (strict schema), e.g. it's `norm_epsilon`,
  not `layernorm_epsilon`.
- The top-level `sweep` must be `type: product` with `groups:` (a bare `type: list` is rejected).
- Command-line overrides of keys that aren't in the YAML need a `+`, e.g.
  `'+slurm.sbatch_extra_directives=["--dependency=afterany:<jobid>"]'`.
- Arguments equal to Megatron's default are not emitted (e.g. no `--optimizer adam` in the
  command line; that's fine).
- Keep `data_integrity_check: false` (otherwise it hashes TBs of data on the login node and tries
  to write into read-only data dirs).
- A new (global batch, seq length, split, #samples) combination rebuilds the data index on
  rank 0 (~25-30 min for the 522 v8ml datasets) while the other ranks wait, hence
  `distributed_timeout_minutes: 120`. Submit such a job alone first.
- `cross_entropy_fusion_impl: te` is disabled in this Megatron; use `native`.

## Data caveats

v8ml weights are the official ones; the flag part points at the original (not decontaminated)
flag collection, 9 missing instruct sets (1.95 % of the mix) are dropped and weights rescaled.
Validation uses the per-dataset `split` (`99,1,0`); don't use `oellm_testing/v8ml-valid` with this
mix. Details in `smoke_test/README.md` ("Data").
