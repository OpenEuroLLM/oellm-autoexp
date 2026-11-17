#!/bin/bash
# Render, verify and enqueue the OPEN-ENDED continuation of the v4 line (128 nodes, mbs 1, gbs 1024):
# no --exit-interval, so it trains until --train-iters (236733) or a manual stop. It waits for the
# experiment segment's 36900 checkpoint and runs under the same pool singleton, so the two can never
# overlap in runs/main.
#
#   bash scripts/korbi/recall_surgery/launch_v4_cont_continue.sh [--no-submit]
#
# Gates before anything is enqueued: render; check_control_render.py with ALLOW_OPEN_ENDED=1 and the
# continuation's own load path (runs/main/ckpt); the dataset rows of this recipe in the main run's
# cache. After submitting it verifies the pool's own step.sh: mbs 1, gbs 1024, NO --exit-interval.
set -uo pipefail

REPO=/gpfs/projects/ehpc533/users/korbip/hybrid_v019/autoexp_v4cont1
PY=/gpfs/projects/ehpc533/users/korbip/hybrid_v019/autoexp_env/bin/python
POOL=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p264
STATE=/gpfs/projects/ehpc533/users/korbip/hybrid_v019/pool_state
CONFIG=experiments/korbi/recall_surgery/v4_cont_continue_mn5
CACHE=/gpfs/scratch/ehpc533/hybrid_nucleus/data_cache_shared
KEY_SOURCE=${KEY_SOURCE:-/gpfs/scratch/ehpc533/hybrid_nucleus/production/kvshare_k_fresh35900_20261116/runs/data_cache_shared}
KEY_REF=${KEY_REF:-$POOL/runs/mn5-pool-p264.1314/step.sh}   # the running segment: same recipe/key
DRY=/gpfs/scratch/ehpc533/hybrid_nucleus/production/v4_cont/dryrun_continue
SUBMIT=1
[ "${1:-}" = "--no-submit" ] && SUBMIT=0

# keys the config file cannot set at submission time (only the dry render sees them) + the stop point
OVERRIDES=(backend.megatron.micro_batch_size=1 backend.megatron.recompute_num_layers=2 backend.megatron.exit_interval=null)

cd "$REPO" || exit 97
rm -rf "$DRY"

echo "=== 1/6 render"
PYTHONPATH=. HYBRID_OUTPUT_DIR=/gpfs/scratch/ehpc533/hybrid_nucleus \
    "$PY" scripts/run_autoexp.py --dry-run --dry-run-root "$DRY" --config-name "$CONFIG" --yes \
    "${OVERRIDES[@]}" > /tmp/cont_continue_dry.log 2>&1
rc=$?
echo "render rc=$rc"; tail -3 /tmp/cont_continue_dry.log
[ "$rc" = 0 ] || { echo "REFUSED: render failed"; exit 3; }

RENDER=$(find "$DRY" -name "*.sbatch" | head -1)
echo "render: $RENDER"
[ -n "$RENDER" ] || { echo "REFUSED: no sbatch rendered"; exit 3; }

echo "=== 2/6 verify (open-ended, load = the continuation's own ckpt)"
ALLOW_OPEN_ENDED=1 EXPECT_LOAD_SUBSTRING=v4_cont/runs/main/ckpt \
    "$PY" scripts/korbi/recall_surgery/check_control_render.py "$RENDER" "$KEY_REF" \
    || { echo "REFUSED: verifier said no"; exit 3; }

echo "=== 3/6 dataset rows of this recipe"
"$PY" scripts/korbi/recall_surgery/cache_ready.py "$CACHE" "$KEY_SOURCE" | tail -4
[ "${PIPESTATUS[0]}" = 0 ] || { echo "REFUSED: cache not ready"; exit 3; }

if [ "$SUBMIT" = 0 ]; then echo "=== --no-submit: stopping after verification"; exit 0; fi

echo "=== 4/6 submit"
PYTHONPATH=. HYBRID_OUTPUT_DIR=/gpfs/scratch/ehpc533/hybrid_nucleus \
    "$PY" scripts/run_autoexp.py --submit-and-exit --yes --monitor-state-dir "$STATE" \
    --pool "$POOL" --config-name "$CONFIG" "${OVERRIDES[@]}" 2>&1 | tail -3

echo "=== 5/6 the enqueued job record"
# The pool creates runs/<rid>/step.sh only when the start condition fires (36900). Until then the
# authoritative record is the job json in the monitor's state dir: it holds the resolved config and the
# CLI overrides this submission carries, so it is what must be checked here.
REC=$("$PY" - "$STATE" <<'PYREC'
import glob, json, os, sys
state = sys.argv[1]
best = None
for f in glob.glob(os.path.join(state, "pool_*", "*.job.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if (d.get("definition") or {}).get("name") != "ambitious_v4_cont_main_continue":
        continue
    ts = float((d.get("metadata") or {}).get("pool_enqueued_ts") or 0)
    if best is None or ts > best[0]:
        best = (ts, f)
print(best[1] if best else "")
PYREC
)
[ -n "$REC" ] || { echo "REFUSED: no job record for the continuation in $STATE"; exit 3; }
echo "  record: $(basename "$REC")"

echo "=== 6/6 verify the record (geometry, stop point, saves, load, start condition)"
"$PY" - "$REC" <<'PYCHK'
import json, sys
d = json.load(open(sys.argv[1]))
name = (d.get("definition") or {}).get("name")
cfg = ((d.get("definition") or {}).get("base_config") or {}).get("config") or {}
m = (cfg.get("backend") or {}).get("megatron") or {}
sc = (d.get("definition") or {}).get("start_condition") or {}
md = (d.get("definition") or {}).get("metadata") or {}
rt = d.get("runtime") or {}
bad = []
if m.get("micro_batch_size") != 1: bad.append(f"micro_batch_size={m.get('micro_batch_size')}")
if m.get("global_batch_size") != 1024: bad.append(f"global_batch_size={m.get('global_batch_size')}")
if m.get("exit_interval") is not None: bad.append(f"exit_interval={m.get('exit_interval')} (must be none)")
if m.get("save_interval") != 1000: bad.append(f"save_interval={m.get('save_interval')}")
if m.get("non_persistent_save_interval") != 100: bad.append(f"non_persistent_save_interval={m.get('non_persistent_save_interval')}")
if m.get("save_retain_interval") != 10000: bad.append(f"save_retain_interval={m.get('save_retain_interval')}")
if m.get("recompute_num_layers") != 2: bad.append(f"recompute_num_layers={m.get('recompute_num_layers')}")
if not str(m.get("load", "")).endswith("v4_cont/runs/main/ckpt"): bad.append(f"load={m.get('load')!r}")
if sc.get("pattern") != r"^\s*36900\s*$": bad.append(f"start pattern={sc.get('pattern')!r}")
if not str(sc.get("path", "")).endswith("runs/main/ckpt/latest_checkpointed_iteration.txt"):
    bad.append(f"start path={sc.get('path')!r}")
if md.get("pool_singleton") != "v4_cont_main": bad.append(f"pool_singleton={md.get('pool_singleton')!r}")
print(f"  name={name} submitted={rt.get('submitted')} attempts={rt.get('attempts')}")
print(f"  mbs={m.get('micro_batch_size')} gbs={m.get('global_batch_size')} exit_interval={m.get('exit_interval')} "
      f"save={m.get('save_interval')}/retain {m.get('save_retain_interval')}/rolling {m.get('non_persistent_save_interval')}")
print(f"  load={m.get('load')}")
print(f"  waits for: {sc.get('path')} == {sc.get('pattern')}  singleton={md.get('pool_singleton')}")
if bad:
    print("  BAD: " + "; ".join(bad)); sys.exit(3)
print("  OK: open-ended (no exit-interval), mbs 1, resumes from runs/main/ckpt, waits for 36900")
PYCHK
if [ "$?" != 0 ]; then
    # never leave a mis-specified continuation waiting in the queue: cancel it in the pool (not in Slurm)
    "$PY" -m oellm_autoexp.monitor.pool_ctl cancel --pool "$POOL" --job ambitious_v4_cont_main_continue \
        >/dev/null 2>&1 || true
    echo "REFUSED: record check failed -> cancelled in the pool"
    exit 3
fi
echo "  (runs/<rid>/step.sh appears at start; check it with check_control_render.py then)"
