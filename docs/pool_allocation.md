# Pool allocation: nodes, exclusions, and GRES

This note records the node/GPU accounting used by the MN5 pool and the
troubleshooting path for `Insufficient GRES` and
`Requested node configuration is not available`.

## Terms

- **Host allocation**: the Slurm job owned by a pool host, for example
  `host007`, job `48685344`. It owns a fixed node set and four H100 GPUs per
  node.
- **Host exclusion**: `<pool>/hosts/<host>/exclude_nodes`. The host supervisor
  avoids these nodes when it builds a run's `srun` nodelist. It does not remove
  them from the Slurm job.
- **Pool exclusion**: `<pool>/exclude_nodes`. Every host avoids these nodes.
  This is pool state, not a Slurm job modification.
- **Run nodelist**: the concrete node list written into a run's step
  (`AUTOEXP_STEP_NODELIST`). This is the list that `srun --nodelist=...` sees.
- **GRES**: the four GPUs reserved per node. A 256-node step requests 1024
  GPUs (`--gres=gpu:4` per node).

## The failure modes are different

`Requested node configuration is not available`
: the step names a node that is not in the current Slurm allocation (for
  example a node Slurm removed after a node failure).

`Insufficient GRES available in allocation`
: the step names nodes that are in the allocation, but at least one selected
  node cannot supply the requested GPUs. This is a **per-node / nodelist**
  problem, not a statement that the whole allocation has no GPUs.

A one-node GPU step can still work while a 256-node step fails. Test capacity at
several sizes before assuming the allocation is lost.

## Capacity probe

Run this without changing the Slurm job:

```bash
for n in 1 4 16 64 128 256; do
  echo "==== N=$n"
  timeout 90 srun \
    --jobid="$JOB_ID" --overlap \
    --nodes="$n" \
    --ntasks="$((4 * n))" --ntasks-per-node=4 \
    --cpus-per-task=1 --gres=gpu:4 \
    /bin/true
  echo "RC=$?"
done
```

This reserves four GPUs per node. `/bin/true` proves **Slurm GPU allocation**;
it does not prove CUDA, NCCL, or InfiniBand health. For that, run a short
all-reduce or the node-health benchmark on the exact proposed node set.

### 2026-11-17 incident on host007

- Original allocation: 264 nodes.
- `as04r3b13` failed during the 36000 validation. Slurm removed it; the job
  became 263 nodes.
- Pool exclusions contained four older bad-IB nodes:
  `as05r1b08`, `as01r1b09`, `as06r2b06`, `as07r2b26`.
- The pool's attempted 256-node nodelist included five nodes that were **not**
  in Slurm's usable 256-node selection:
  `as07r5b05`, `as07r5b06`, `as07r5b08`, `as07r5b09`, `as07r5b10`.
- The pool omitted five nodes that the successful selection used:
  `as01r1b09`, `as05r1b08`, `as06r2b06`, `as07r2b24`, `as07r2b26`.
- Capacity sweep result: **1, 4, 16, 64, 128, and 256 nodes all succeeded**
  with `--gres=gpu:4`.
- No stale `pretrain_hybrid.py` processes were found on any of the 263 nodes.
  `pkill` was therefore not a fix.

The point was to compare the **pool nodelist** with the **Slurm-selected
nodelist**:

```bash
# successful scheduler choice, expanded
srun --jobid="$JOB_ID" --overlap --nodes=256 --ntasks=1024 \
  --ntasks-per-node=4 --gres=gpu:4 \
  bash -c 'echo "$SLURM_JOB_NODELIST"' > good.out

# pool's pinned list from its last failed run
POOLNL=$(sed -n '1s/.*nodes //p' run.log)
scontrol show hostnames "$POOLNL" | sort -u > pool.nodes
scontrol show hostnames "$(grep -o 'as[^ ]*' good.out | head -1)" | sort -u > good.nodes

comm -23 pool.nodes good.nodes   # nodes in pool but not usable
comm -13 pool.nodes good.nodes   # usable nodes omitted by pool
```

## Remediation without touching Slurm

1. Add the nodes found only in `pool.nodes` to the pool exclusion list:

   ```bash
   PYTHONPATH=. python scripts/pool_ctl.py "$POOL" \
     exclude as07r5b05 as07r5b06 as07r5b08 as07r5b09 as07r5b10
   ```

2. Remove host-level exclusions only when the capacity probe proves those
   nodes are usable, and keep at least 256 usable nodes. Host exclusions live
   in `<pool>/hosts/<host>/exclude_nodes`; pool exclusions live in
   `<pool>/exclude_nodes`. Neither is a Slurm change.

3. Re-run the capacity probe on the exact 256-node set Slurm will use.

4. Resubmit the run with the normal pool submission. The checkpoint and tracker
   are unchanged.

Never use a broad `pkill pretrain_hybrid.py`; it can kill another run on a
shared host. If cleanup is needed, filter by `AUTOEXP_RUN_ID` or the exact
run's `--load` path and signal only those ranks.

## Invariants

- Never edit, requeue, resize, or cancel an existing Slurm job without the
  user's explicit permission.
- Pool exclusions and host exclusions are the supported repair surface.
- A failed node stays removed by Slurm; re-adding it requires Slurm-side
  action.
- A node with bad IB can pass a GPU-only allocation test. Run a communication
  check before putting it in a long training set.
