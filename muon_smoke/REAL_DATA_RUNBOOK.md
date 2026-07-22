# Muon + Qwen3 0.1B_ne on REAL multilingual data — runbook

Written 2026-07-16. Everything here needs **no access to project_462000963**.

## What this is

The real multilingual baseline reads project_462000963, which you can't touch
(you're in no `project_462*` group; the project root is 2770 so it can't even be
traversed — no colleague can chmod around that, only a LUMI admin).

**Workaround:** `/scratch/project_465002530/preprocessed/oellm-v1-256k/long-ctx-sample/`
is readable, uses the SAME openeurollm 256k tokenizer, and was sampled FROM the
blocked data (proved by the `input:`/`output:` lines in its `.stats.txt` files).
153 files, ~121G, ~32B tokens, covering the same dataset families.

**THE CAVEAT — do not lose this:** long-ctx-sample oversamples long documents ~20x
(hplt3-fra: 3.5% of source tokens are long-doc vs 71% in the sample). So:
  - GOOD for: pipeline validation; Muon-vs-Adam where YOU run both arms
    (identical data => the contrast is internally valid).
  - NOT valid for: comparing to the existing 0.1B_ne Adam baselines in
    /scratch/project_465002530/multilingual_scaling/0.1B_ne/training/.
    Those ran the real mix. Different data => not comparable. 

## What was created (all in your own space, nothing pushed)

| File | What |
|---|---|
| `muon_smoke/build_longctx_datamix.py` | generates the datamix; dry-run by default |
| `config/backend/megatron/multilingual_scaling/data/oellm_256k_lumi_longctx.yaml` | data config -> long-ctx-sample |
| `config/experiments/laingsam/muon_qwen3_0.1B_longctx.yaml` | the run: Qwen3 0.1B_ne + muon |

Modelled on `exp_diana:config/experiments/multilingual_scaling/training/0.1B_ne_lumi.yaml`
(the config that produced the real runs). Only three things differ: data group,
`optimizer: muon`, `sweep: none`.

## Key finding: the overlay does NOT work with autoexp

`muon-overlay.img` works for hand-rolled `singularity exec --overlay`. But
autoexp's `ContainerConfig` (oellm_autoexp/config/schema.py:76) has fields
image/runtime/bind/env/pwd/python and **no overlay field** — grep for "overlay"
across the repo returns nothing. So the pipeline can never see the overlay.

Since `emerging_optimizers` is pure Python, the fix is to install it to a plain
directory and put it on PYTHONPATH via `container.env`. That's STEP 1 below.
No container rebuild, no autoexp patch. (This is also the cleaner thing to
upstream if muon gets adopted.)

---

# STEP 1 — install emerging_optimizers to a plain dir (ONE TIME)

Needs network. Run inside an allocation, not on the login node.

Get an allocation (single line; note the JOBID it prints):

    salloc --no-shell --account=project_465002530 --partition=dev-g --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=7 --mem=60G --time=01:00:00

Then install (single line; replace <JOBID>):

    srun --jobid=<JOBID> --overlap --ntasks=1 singularity exec -B /pfs -B /scratch -B /flash -B /opt/cray -B /var/spool/slurmd -B /appl /scratch/project_465002530/containers/MegatronTrainingLumi_x86_64.sif bash -c 'pip install --no-deps --target=/scratch/project_465002530/users/laingsam/muon-pylibs "git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@v0.2.0"'

`--no-deps` is important: without it pip drags an entire torch into that dir.

Verify it imports (single line; replace <JOBID>):

    srun --jobid=<JOBID> --overlap --ntasks=1 singularity exec -B /pfs -B /scratch -B /flash -B /opt/cray -B /var/spool/slurmd -B /appl /scratch/project_465002530/containers/MegatronTrainingLumi_x86_64.sif bash -c 'export PYTHONPATH=.:/scratch/project_465002530/users/laingsam/muon-pylibs; python -u -c "import emerging_optimizers as e; print(\"OK\", e.__file__)"'

# STEP 2 — build the datamix

Dry run first — reports what maps, writes NOTHING (single line; replace <JOBID>):

    srun --jobid=<JOBID> --overlap --ntasks=1 singularity exec -B /pfs -B /scratch /scratch/project_465002530/containers/MegatronTrainingLumi_x86_64.sif bash -c 'cd /scratch/project_465002530/users/laingsam/oellm-muon && python3 -u muon_smoke/build_longctx_datamix.py'

Check the report:
  - "weights sum to 1.000000" -> the official mix transcription is intact.
    If this FAILS the script refuses to write. That's the transcription guard.
  - the DROP lines -> entries with no long-ctx-sample file (expect kat_Geor;
    the sample has no `kat`). Their weight is renormalized across the rest.

Then actually write it (single line; replace <JOBID>):

    srun --jobid=<JOBID> --overlap --ntasks=1 singularity exec -B /pfs -B /scratch /scratch/project_465002530/containers/MegatronTrainingLumi_x86_64.sif bash -c 'cd /scratch/project_465002530/users/laingsam/oellm-muon && python3 -u muon_smoke/build_longctx_datamix.py --write'

Writes exactly one file: muon_smoke/datamix4-longctx-lumi.txt

# STEP 3 — dry-run the config (renders the command, submits nothing)

Single line; replace <JOBID>:

    srun --jobid=<JOBID> --overlap --ntasks=1 singularity exec -B /pfs -B /scratch -B /flash -B /opt/cray -B /appl /scratch/project_465002530/containers/MegatronTrainingLumi_x86_64.sif bash -c 'cd /scratch/project_465002530/users/laingsam/oellm-muon && export CONTAINER_CACHE_DIR=/scratch/project_465002530/containers PYTHONPATH=.:/scratch/project_465002530/users/laingsam/muon-pylibs && python3 -u scripts/run_autoexp.py --config-name experiments/laingsam/muon_qwen3_0.1B_longctx --dry-run' 2>&1 | tee /scratch/project_465002530/users/laingsam/oellm-muon/muon_smoke/05_dry_run.log

Look for in the rendered command:
    --optimizer muon --muon-momentum 0.9 --muon-scalar-optimizer adam ...
    --data-args-path /scratch/.../muon_smoke/datamix4-longctx-lumi.txt
    --tokenizer-model openeurollm/tokenizer-256k

**Likely failure here:** `optimizer: muon` may not be a valid config key yet —
the checked-in schema was generated from Megatron v0.2, and muon is v0.17. If it
errors with an unknown key, the schema needs regenerating (see PIPELINE_NEXT.md
RUNG A). That is expected and is the next thing to fix.

Paste the output back before submitting anything.

# STEP 4 — free the allocation

    scancel <JOBID>

---

## Then, and only then: the actual run

1 node / 8 GPUs / dev-g / 50 iters (~6.5M tokens). Do NOT submit until STEP 3
renders cleanly.

## Known open questions

- Schema regen (STEP 3's likely failure) — v0.17 schema so muon_* are valid keys.
- Muon LR is Adam's 5e-4, untuned. Fine for a smoke, wrong for a comparison.
- Diana's real runs used `AutoExp_2026-07-02.sif`; your branch's container/lumi.yaml
  points at `MegatronTrainingLumi_x86_64.sif`. Both are readable. The PYTHONPATH
  approach works either way, which is why it's better than the overlay.
- For a real Muon-vs-Adam result, run this config twice with `optimizer: muon`
  and `optimizer: adam` and compare those two — never against the 462000963
  baselines.
