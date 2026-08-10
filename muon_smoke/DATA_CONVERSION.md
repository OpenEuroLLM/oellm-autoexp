# Converting the Nemotron sample to Megatron format (Capella)

One-time data-prep step — not part of the training loop itself. Run once,
the output sits in your own workspace, and every future training run just
reads from it. **Per-user**: since the output lands in your own private
workspace (not shared/group-readable), anyone else following this same path
needs to redo this themselves — see note at the bottom.

## What this does

- Reads (read-only) the already-tokenized (GPT2Tokenizer) arrow shards from
  the coworker's granted directory:
  `/data/horse/ws/jasi149i-fastlm/data/nemotron-cc-sample-mtsynth/tokenized_gpt2/ctx_2048/{train,valid}/`
- Converts them into Megatron's native `.bin`/`.idx` indexed-dataset format
  (no re-tokenization — the token IDs already exist, this is a pure format
  conversion) via `muon_smoke/convert_nemotron_arrow_to_megatron.py`.
- Writes output only into your own workspace:
  `/data/horse/ws/sala597i-slaing/data/nemotron_gpt2_megatron/{train,valid}.{bin,idx}`

**Write-safety**: the script opens the source shards via `pa.memory_map(path, "r")`
(read-only) and only ever writes through `IndexedDatasetBuilder`, which is
hardcoded to `OUTPUT_DIR` (your workspace). There is no other file-write call
in the script. As a second layer of protection, the source directory's
permission bits (`other::r-x`, confirmed via `getfacl`) would deny a write
attempt there even if the code were wrong.

No SLURM/GPU allocation needed — this is pure CPU/I/O work, run directly on
the login node. Doesn't touch the scheduler or any shared/other-user state.

## Timing

Estimated from a single-shard benchmark (8.8s for ~504MB, 57.3 MB/s),
extrapolated linearly across all 436 `train` shards (203GB total):
**~60 minutes** for the full `train` split. `valid` is tiny (~39MB, seconds).
This is an estimate, not a guarantee — shard sizes aren't perfectly uniform,
and the full-scale run (`IndexedDatasetBuilder` keeps growing in-memory lists
until `finalize()`) has only been tested at 1/436th scale so far.

## Exact commands

```bash
cd /home/h1/sala597i/oellm-autoexp
SANDBOX=/data/horse/ws/sala597i-slaing/containers/.tmp_sandbox/oellm_sandboxv514iw4i/sandbox

# 1. valid split (tiny, seconds) — run this first
singularity exec --underlay -B /data/horse/ws/jasi149i-fastlm -B /data/horse/ws/sala597i-slaing \
  --env PYTHONPATH=$PWD:submodules/Megatron-LM "$SANDBOX" \
  python3 muon_smoke/convert_nemotron_arrow_to_megatron.py --split valid

# 2. train split, full (all 436 shards, ~60 min) — nohup so an SSH
#    disconnect doesn't kill it; runs in the background, returns your prompt
nohup singularity exec --underlay -B /data/horse/ws/jasi149i-fastlm -B /data/horse/ws/sala597i-slaing \
  --env PYTHONPATH=$PWD:submodules/Megatron-LM "$SANDBOX" \
  python3 muon_smoke/convert_nemotron_arrow_to_megatron.py --split train \
  > muon_smoke/logs/convert_train.log 2>&1 &

echo "Started, PID: $!"
disown
```

Check progress any time:

```bash
tail -20 /home/h1/sala597i/oellm-autoexp/muon_smoke/logs/convert_train.log
```

Done when the log ends with a `DONE: ... rows, ... tokens, 436 shards in ...s` line.

## After it finishes

1. Verify the output loads correctly:
   ```bash
   singularity exec --underlay -B /data/horse/ws/sala597i-slaing \
     --env PYTHONPATH=$PWD:submodules/Megatron-LM "$SANDBOX" python3 -c "
   from megatron.core.datasets.indexed_dataset import IndexedDataset
   ds = IndexedDataset('/data/horse/ws/sala597i-slaing/data/nemotron_gpt2_megatron/train')
   print('num sequences:', len(ds))
   "
   ```
2. In `config/experiments/laingsam/muon_50M_50BT.yaml`, set:
   - `mock_data: false`
   - `data_path: ["/data/horse/ws/sala597i-slaing/data/nemotron_gpt2_megatron/train"]`
3. Rerun the RUNG C smoke ladder against real data (see `RESULT.md`) — the
   real-data *loading* path (`GPTDataset` sampling/splitting) has never
   actually run yet in this setup, only `mock_data`. Some new small issue on
   first attempt would match the pattern of everything else in this session
   (tokenizer path, PYTHONPATH, OOM sizing) — not guaranteed to be flawless
   the first try, but everything it depends on (optimizer, data format) is
   already solid.

## Incident (2026-07-27/28): OOM kill at 99.5%, recovered without rerunning

The `--split train` conversion (`nohup`'d, ~4hrs) got killed by the login node's
memory cgroup right near the end:

```
Memory cgroup out of memory: Killed process (python3) anon-rss:7.4GB, cpuset=user-*.slice
```

Root cause: `IndexedDatasetBuilder` (`megatron/core/datasets/indexed_dataset.py:961-962`)
keeps `sequence_lengths`/`document_indices` as plain Python lists for the *entire* run,
only flushed to `.idx` in `finalize()` at the very end. Across 436 shards / ~26.4M rows,
those lists (plus the resulting GC pressure) grew RSS from ~4.8GB to 7.4GB until Capella's
per-user memory cgroup killed it — one shard short of `finalize()` ever running.

Also worth knowing: a `nohup`'d/`disown`'d background job on this login node is **not**
guaranteed to survive an SSH disconnect — `KillUserProcesses` is left at its systemd
default (`yes`, confirmed via `/etc/systemd/logind.conf` being commented-out, i.e.
inherited) and `loginctl show-user` shows `Linger=no` for this account. It survived one
disconnect during this run, but that's luck, not a guarantee — don't count on it.

**Recovery — no rerun needed.** `train.bin` was intact (216.7GB, only `train.idx` was
missing): its size divided *exactly* evenly by 8196 bytes (2049 int32 tokens × 4 bytes),
proving it died cleanly on a row boundary with zero corruption. Wrote
`muon_smoke/reconstruct_train_idx.py`, which rebuilds `train.idx` directly from that
known fixed row size — no re-reading of the arrow shards, no re-tokenizing, just
metadata math (seconds, not hours):

```bash
singularity exec --cleanenv --underlay -B /data/horse/ws/jasi149i-fastlm -B /data/horse/ws/sala597i-slaing \
  --env PYTHONPATH=$PWD:submodules/Megatron-LM "$SANDBOX" \
  python3 muon_smoke/reconstruct_train_idx.py
```

Caveat: this treats each 2049-token packed row as **one single document** (sequence
length 2049), rather than preserving the original intra-row `docs_lengths` sub-document
boundaries that a clean finalize would have recorded (compare: the `valid` split, which
finished normally, has 18,602 documents of *varying* length, e.g. `ds[0].shape == (448,)`
— real sub-document boundaries). This only matters if something reads those boundaries;
`reset_attention_mask: false` and `reset_position_ids: false` in
`config/backend/megatron/base_defaults.yaml:552-553` mean Megatron never uses them
during training, so it's functionally equivalent here. If that ever flips to `true`,
this reconstructed `train` split would need to be redone properly (full rerun) instead.

**Result**: both splits verified loadable and ready to use as of 2026-07-28, no
further action needed:
- `train`: 26,442,113 rows (~99.5% of the full ~217.7GB source — the last ~0.5%,
  whatever didn't get flushed before the OOM kill, is simply absent, not corrupted)
- `valid`: 18,602 rows

**Before repeating this at larger scale** (e.g. the full 972GB raw corpus, or if
someone reruns `train` from scratch): `convert_nemotron_arrow_to_megatron.py`'s
unbounded in-memory lists are a real bug worth fixing first — e.g. periodic
`builder.finalize()` + `add_index()` merging per N shards, so a crash only loses
the current chunk instead of risking the whole run.

## If someone else wants to follow this

The output workspace (`/data/horse/ws/sala597i-slaing/`) is `drwx------` —
private to this account, not shared. Anyone else repeating this setup needs:
their own read grant to `jasi149i-fastlm`'s sample data, and to run this same
conversion into their own workspace. This isn't a shared/reusable artifact
as it stands.
