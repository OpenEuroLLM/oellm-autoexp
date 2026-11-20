# Standalone B1 annealing on JUPITER

This package prepares separate B1 32B annealing experiments from a retained native tensorwise checkpoint, using either the Prelude multilingual1+dominant1 or multilingual1+dominant2 blend. TP4/PP4, sequence4096, GBS4096, FP8 tensorwise with BF16 edge layers, Adam/RNG restoration and sample-based WSD `minus_sqrt` follow the B1 production setup. The new data phase starts at the fork; its mixture is consumed from offset zero. Original9B linear-WSD configs and all original mixture inputs are under [origins](origins/README.md), with source paths and hashes. The executable preparer here is for32B; those9B historical examples require a separate JUPITER port and validation.

Use a dedicated checkout of `prod/oellm_32b_dense`, initialize its pinned Megatron submodule and activate the existing autoexp environment. The tested base is `5df9c35c16b3bce8b74f608bc62f7fdd2b7aaa70`, Megatron `c9cf7e84cf584dee5ca456124180c5acc98e80ba`. No package installation is needed on the prepared JSC environment.

```bash
git submodule update --init submodules/Megatron-LM
source /e/fscratch/e-sta-openeurollm/jj1/envs/oellm-autoexp/bin/activate
export SEED=/absolute/checkpoints/iter_0416000
export RUN=/new/absolute/scratch/anneal-dominant1
python annealing/b1_prelude/prepare.py --root "$RUN" --seed "$SEED" \
  --mix dominant1 --tokens 20000000000 --nodes 512 --attempts 4 --walltime 12:00:00
```

Preparation does not submit training. It checks all indexed seed shards, every294dataset-prefix pair, creates an arm-owned checkpoint tree with a seed symlink, copies the frozen mixture and renders the model/config/batch script. Review `recipe.json`, `config/annealing_arm.yaml` and `training_core/job_*.sbatch`. All logs, monitor state, persistent/rolling checkpoints and cache belong to the new arm. Optional `--cache /absolute/existing/cache` reuses compatible indices only after exact preparation/coverage checks; it never assumes that the presence of files makes them valid. The seed must remain available until a validated replacement exists.

| Token budget | Updates | Actual tokens |
|---|---:|---:|
|20B|1,193|20,015,218,688|
|50B|2,981|50,012,880,896|
|100B|5,961|100,008,984,576|
|300B|17,882|300,010,176,512|
|2T|119,210|2,000,011,919,360|

The endpoint is seed iteration plus `ceil(tokens/16,777,216)`. At512nodes each allocation uses2,048GPUs; the hard reserved upper bound is GPUs×walltime×maximum allocations. The prior2T plan used up to32allocations of12hours, with roughly13allocations forecast from historical speed. Forecast and hard maximum are different quantities. Run cache preparation and smoke validation before admitting any full experiment. Full runs retain rolling saves every125updates and the bounded production continuation policy; interrupted full-state resume and terminal rolling cleanup require their own validation, beyond the three-update smoke provided here.

## Prepare and verify data before training

```bash
python annealing/b1_prelude/cache.py prepare --root "$RUN"
# Review cache-preparation/settings.json, render.json and the batch script.
sbatch --test-only "$RUN"/cache-preparation/job_*.sbatch
python annealing/b1_prelude/cache.py submit --root "$RUN"
# Once Slurm reports COMPLETED:
python annealing/b1_prelude/cache.py check --root "$RUN"
```

The adapter inherits production `data_cache_build_fscratch_trainshape.yaml`: one rank, two-layer stand-in model, full tokenizer, real data builder, one actual tiny-model update and no checkpoint writes. Preparation has its own four-hour/four-GPU maximum16GPU-hours. It derives the exact train phase and validation sample counts from the resolved target, including evaluation timing at the fork; its sizing regression executes the corresponding function from the pinned Megatron source. The supported recipes have no test split; nonzero test splits are rejected rather than silently mis-sized. The original production header/array-size checker must pass, and the adapter requires the matching train/valid blend descriptions, ordered294prefixes, weights, per-child sample counts, tokenizer/seed/sequence/split identities and successful preparation completion. Cached files alone do not pass the gate.

Cache construction and training take the same cache-writer lease. Do not overlap other writers or manually delete partial arrays. Interrupted groups must be quarantined with quiescent writers before rebuilding; retain complete reusable groups. Changing tokenizer/data/sampling settings invalidates the matching preparation evidence. The initial `launch.sh` runs the cache checker before submission; the continuous autoexp monitor uses the existing restart rules thereafter. STOP prevents new admission. Preserve submission intents and reconcile uncertain scheduler submissions rather than retrying blindly.

The November12 jobs failed after successful dataset construction because `eval_iters=0` yielded an empty validation sampler (`no sample to consume: 0`). This was not a disk-quota failure. On November17 fscratch had about26TiB free and8.2million free inodes; the project limit was40/44TB and each reusable arm cache about70.773GiB. The corrected smoke uses one16-sample validation iteration. The cache preparer verifies a positive validation sample count before submission. The project quota utility's fscratch usage field was zero while `df` showed actual usage, so use both quota policy and actual filesystem/write probes for capacity checks.

## Short validation, then full execution

Use a new root with `--smoke --nodes 32` to request three updates in a single ten-minute allocation, with checkpoint writes and retries disabled. This preserves the production global batch and optimizer state. Use a sufficiently large node count to fit three updates plus restore inside ten minutes; the earlier16-node smoke did not establish throughput. Preparation remains a separate allocation and must pass first. Smoke validation reads complete logs, requires restoration of the selected seed, three consecutive finite loss/gradient updates with positive LR/timing/throughput and zero skipped/NaN updates, and Slurm COMPLETED/exit0. It records each step and consecutive loss changes; it does not demand monotonically falling loss over three updates.

For the tested two-arm campaign the standalone coordinator is:

```bash
python annealing/b1_prelude/validate.py init --root "$STATE" --arm "$ARM1" --arm "$ARM2"
python annealing/b1_prelude/validate.py watch --root "$STATE" --execute
python annealing/b1_prelude/validate.py verify --root "$STATE"
```

It pins exact scripts/configs/sources and calls the cache gate before submitting each short training job once. `status.json`, per-arm response reports/step CSVs and `all_passed.json` distinguish pending jobs, failed jobs and validated training. A finite failed smoke remains a review block; the coordinator never claims a passing test because its watcher is alive. The November 20 smoke attempts and reused preparation receipts are tracked under `/e/fscratch/e-ext-2025e02-108/jitsev1/b1-annealing-20261111/validation-20261120`. A publication guard re-verifies both actual training results and the reviewed package before publishing; `publication.json` is the publication receipt. Until it exists with `published_commit`, this package is not claimed published on the target branch. `validation_result.json`, included only after successful validation, records the tested outcome and limitations.

At initial restore, the package runs native Megatron setup and then resets the validation cursor to zero for the new blend. A resume from a later checkpoint in the arm keeps its saved validation cursor. The smoke report checks for the reset in the job log. Both November 17 smoke jobs restored iteration 416000, then failed because the checkpoint supplied a validation offset of 1,679,360 samples to a 294-sample validation blend.

A new arm with matching cache inputs can reuse an existing completed preparation. First render its own preparation with `cache.py prepare`. Then run `cache.py reuse --root "$RUN" --from-root "$PRIOR"`. This checks the same Megatron gitlink, normalized data settings, mixture, tokenizer and cache path, verifies the original preparation and reruns the checker and coverage test. The November 20 retries reuse preparations 2967355 and 2967358.

After smoke validation, execute a separately prepared full arm with `$RUN/launch.sh` in a named tmux workflow. Retain its monitor session and standalone restoration command. Keep checkpoint/optimizer/RNG/data-position/schedule state intact on continuation, preserve the explicit endpoint and cumulative budget, and never point multiple experiments at the same rolling checkpoint directory. The full training source remains the normal autoexp/production implementation; no Megatron training-source modification is required for this package.

## Reproduce source mixtures and inspect provenance

```bash
python annealing/b1_prelude/datamix/rebuild.py --mix dominant1 --output /new/dominant1.txt
cmp /new/dominant1.txt annealing/b1_prelude/datamix/dominant1.txt
python -m unittest discover -s annealing/b1_prelude -p 'test_*.py' -v
```

Both blends retain the original weights and prefix order, including the original AgentTrove prefix. Current B1 production pretraining uses a separately filtered AgentTrove index; this annealing package does not silently substitute it. Full LUMI/JSC byte comparison is still incomplete:431/588matches, zero mismatches,157pending/39.194TB at the November17 snapshot. The bounded CPU continuation and receipt locations are documented in `origins/README.md`. This is explicitly a remaining provenance limitation, separate from JSC cache/training sanity validation.
