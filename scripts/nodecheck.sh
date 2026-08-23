#!/usr/bin/env bash
# =============================================================================
# PER-NODE HEALTH PROBE for the JUPITER node-catching campaign.
# =============================================================================
#
# Runs ONE task per node (the jupiter slurm config already defaults to
# ntasks_per_node=1, ntasks=nodes) and prints a single verdict line per node:
#
#     [nodecheck] FAIL <node>: <reason>        -> monitor appends <node> to the
#                                                 exclusion file
#     [nodecheck] WARN <node>: <reason>        -> logged only, NOT excluded
#     [nodecheck] OK   <node> <metrics...>     -> healthy
#
# WHY THESE CHECKS AND NOT OTHERS
# -------------------------------
# Every check below reproduces a fault that has ALREADY cost this project a
# large allocation. They are deliberately the same conditions NHC drains for,
# because the failure mode we are guarding against is a node that NHC has
# already flagged once, was resumed, and is still broken — see the header of
# config/exclude/jupiter_exclude_nodes.txt. Mapping from that file:
#
#   check_hw_physmem_free            -> MemAvailable            (jpbo-034-38, ...)
#   check_free_memory_per_numa_node  -> per-socket MemFree      (jpbo-121-03, ...)
#   check_residual_mounts            -> leftover squashfuse     (jpbo-044-15)
#   check_fs_mount                   -> /e/project1 unmounted   (jpbo-075-09)
#   check_link_downed_counters       -> IB Link Downed          (jpbo-074-31)
#   check_gpu_remapped_rows_pending  -> row remapper pending    (jpbo-096-37)
#   XID 48 / uncorrectable ECC       -> ECC aggregate counters  (jpbo-114-47)
#   check_rpcRmApiAlloc_failed       -> nvidia-smi hangs/fails  (jpbo-120-36)
#
# HOST TOOLS ONLY, NO CONTAINER, NO CUDA. That is the whole point: the probe
# must cost seconds, not the ~4 min a container start + NCCL init costs, so a
# 512-node draw is a ~3 min allocation rather than a training run. The price is
# that it cannot see fabric-level or collective faults — those only show up in
# a real at-scale run, which is why the campaign config can also be pointed at
# the real training job (see config/experiments/oellm_32b_dense/node_catch_n512.yaml).
#
# ALWAYS EXITS 0, EVEN WHEN A NODE FAILS. A nonzero task exit would make the
# whole step FAILED and (with JUPITER's ForceRequeueOnFail prolog flags in the
# picture) muddy the sacct record the analysis script reads. The verdict travels
# in the log line, not the exit code.
#
# Usage:
#   srun --ntasks-per-node=1 bash scripts/nodecheck.sh
#   NODECHECK_MEM_FAIL_GB=300 srun ... bash scripts/nodecheck.sh   # retune
#
# Thresholds (all overridable via environment):
#   NODECHECK_MEM_FAIL_GB      MemAvailable below this -> FAIL      (default 400)
#   NODECHECK_MEM_WARN_GB      MemAvailable below this -> WARN      (default 600)
#   NODECHECK_NUMA_FAIL_GB     any socket below this   -> FAIL      (default 75)
#   NODECHECK_NUMA_WARN_GB     any socket below this   -> WARN      (default 95)
#   NODECHECK_GPUS             expected GPU count                   (default 4)
#   NODECHECK_LINKDOWN_FAIL    IB link_downed at/above -> FAIL      (default 2)
#   NODECHECK_ECC_VOL_FAIL     uncorrectable ECC THIS BOOT above    (default 0)
#                              this -> FAIL. The real active-fault bar.
#   NODECHECK_ECC_AGG_FAIL     LIFETIME uncorrectable ECC at/above  (default 64)
#                              this -> FAIL. Must stay high: aggregate never
#                              resets, so a low bar fails healthy nodes.
#   NODECHECK_ECC_AGG_WARN     lifetime ECC at/above this -> WARN   (default 8)
#   NODECHECK_MOUNTS           colon-separated required mounts
#                              (default /e/project1:/e/scratch)
#   NODECHECK_SLOW_S          probe wall time above this -> SLOW    (default 60)
#   NODECHECK_TIMEOUT_S        per-command timeout                  (default 20)
# =============================================================================

NODE="${SLURMD_NODENAME:-$(hostname -s)}"

MEM_FAIL_GB="${NODECHECK_MEM_FAIL_GB:-400}"
MEM_WARN_GB="${NODECHECK_MEM_WARN_GB:-600}"
NUMA_FAIL_GB="${NODECHECK_NUMA_FAIL_GB:-75}"
NUMA_WARN_GB="${NODECHECK_NUMA_WARN_GB:-95}"
EXPECT_GPUS="${NODECHECK_GPUS:-4}"
LINKDOWN_FAIL="${NODECHECK_LINKDOWN_FAIL:-2}"
# ECC bars. See section 4 for why volatile and aggregate are treated so
# differently: volatile is "errors since this boot" (an active fault), aggregate
# is "errors ever" (mostly retired history). The aggregate FAIL bar of 64 is set
# above the entire observed 2026-08-23 distribution of healthy nodes (1-20, with
# the single genuine outlier at 2429) and well below that outlier.
ECC_VOL_FAIL="${NODECHECK_ECC_VOL_FAIL:-0}"
ECC_AGG_FAIL="${NODECHECK_ECC_AGG_FAIL:-64}"
ECC_AGG_WARN="${NODECHECK_ECC_AGG_WARN:-8}"
REQUIRED_MOUNTS="${NODECHECK_MOUNTS:-/e/project1:/e/scratch}"
SLOW_S="${NODECHECK_SLOW_S:-60}"
CMD_TIMEOUT="${NODECHECK_TIMEOUT_S:-20}"

start_s=$SECONDS
fails=()
warns=()

fail() { fails+=("$1"); }
warn() { warns+=("$1"); }

# ---------------------------------------------------------------------------
# 1. Host memory. Residue from a previous job (leaked processes holding
#    hundreds of GB) is the single most common reason this fleet drains nodes,
#    and it is invisible to SLURM: the node is IDLE and allocatable.
#
#    NB on GH200 /proc/meminfo counts GPU HBM as well: MemTotal is 858 GB
#    (4 sockets x 118 GB + 4 GPUs x 95 GB) and a healthy idle node reports
#    MemAvailable 774 GB (measured, jpbo-040-48). The defaults below are set
#    against that, and it is why the per-socket check in 2. carries the real
#    weight — this one only catches gross leaks.
# ---------------------------------------------------------------------------
mem_avail_gb=-1
if [[ -r /proc/meminfo ]]; then
    mem_kb=$(sed -n 's/^MemAvailable: *\([0-9]*\) kB/\1/p' /proc/meminfo)
    if [[ -n "$mem_kb" ]]; then
        mem_avail_gb=$(( mem_kb / 1048576 ))
        if (( mem_avail_gb < MEM_FAIL_GB )); then
            fail "mem_available=${mem_avail_gb}GB < ${MEM_FAIL_GB}GB (leftover processes?)"
        elif (( mem_avail_gb < MEM_WARN_GB )); then
            warn "mem_available=${mem_avail_gb}GB < ${MEM_WARN_GB}GB"
        fi
    else
        warn "could not read MemAvailable from /proc/meminfo"
    fi
else
    warn "/proc/meminfo unreadable"
fi

# ---------------------------------------------------------------------------
# 2. Per-NUMA free memory. A rank is pinned to its socket, so a single starved
#    NUMA node stalls one rank — and at 2048 ranks one stalled rank stalls the
#    collective. Aggregate MemAvailable can look fine while one socket is empty,
#    which is exactly why NHC has a separate check for it.
#
#    ONLY THE CPU SOCKETS COUNT, AND GETTING THAT WRONG FAILS EVERY NODE. A
#    GH200 node presents 36 NUMA nodes, measured on jpbo-040-48:
#        node0-3      118-119 GB, cpus 0-71 / 72-143 / ...   <- the real sockets
#        node4,12,20,28  95 GB, NO cpus                      <- GPU HBM
#        the other 28     0 GB, no cpus                      <- empty placeholders
#    Taking the minimum over all of them yields numa_min_free=0 GB on a
#    perfectly healthy node. The first one-node test did exactly that and
#    blacklisted jpbo-038-27; at 512 nodes it would have written the entire
#    allocation into the exclusion file. Require a non-empty cpulist AND
#    MemTotal > 0.
#
#    MemFree + FilePages, not bare MemFree: page cache is not dropped between
#    jobs, so a node that has just run a heavy-IO job can show a low MemFree
#    while being entirely healthy. Page cache is reclaimable; leaked anonymous
#    memory — the thing worth catching — is not. NHC's own
#    check_free_memory_per_numa_node uses bare MemFree with a 75000 MB
#    threshold, hence the default below: same number, strictly safer metric.
# ---------------------------------------------------------------------------
numa_min_gb=-1
numa_sockets=0
for nodedir in /sys/devices/system/node/node*; do
    [[ -r "$nodedir/meminfo" ]] || continue
    cpus=$(cat "$nodedir/cpulist" 2>/dev/null)
    [[ -n "$cpus" ]] || continue          # CPU-less: GPU HBM or a placeholder
    total_kb=$(sed -n 's/.*MemTotal: *\([0-9]*\) kB/\1/p' "$nodedir/meminfo")
    [[ -n "$total_kb" ]] && (( total_kb > 0 )) || continue
    free_kb=$(sed -n 's/.*MemFree: *\([0-9]*\) kB/\1/p' "$nodedir/meminfo")
    cache_kb=$(sed -n 's/.*FilePages: *\([0-9]*\) kB/\1/p' "$nodedir/meminfo")
    [[ -n "$free_kb" ]] || continue
    [[ -n "$cache_kb" ]] || cache_kb=0
    numa_sockets=$(( numa_sockets + 1 ))
    free_gb=$(( (free_kb + cache_kb) / 1048576 ))
    if (( numa_min_gb < 0 )) || (( free_gb < numa_min_gb )); then
        numa_min_gb=$free_gb
    fi
done
if (( numa_min_gb >= 0 )); then
    if (( numa_min_gb < NUMA_FAIL_GB )); then
        fail "numa_min_free=${numa_min_gb}GB < ${NUMA_FAIL_GB}GB"
    elif (( numa_min_gb < NUMA_WARN_GB )); then
        warn "numa_min_free=${numa_min_gb}GB < ${NUMA_WARN_GB}GB"
    fi
else
    # Not "everything is fine": it means the socket topology is not what this
    # check expects, so the per-NUMA guard is silently doing nothing.
    warn "no CPU-bearing NUMA node found — per-socket memory NOT checked"
fi

# ---------------------------------------------------------------------------
# 3. GPUs. `nvidia-smi -L` is run under `timeout` on purpose: when the GSP
#    firmware channel is wedged (jpbo-120-36, NHC check_rpcRmApiAlloc_failed)
#    nvidia-smi does not error, it HANGS — and so does every CUDA context that
#    lands there. A hang is therefore a FAIL, not a skip.
# ---------------------------------------------------------------------------
gpu_count=-1
if command -v nvidia-smi >/dev/null 2>&1; then
    if gpu_list=$(timeout "$CMD_TIMEOUT" nvidia-smi -L 2>&1); then
        gpu_count=$(printf '%s\n' "$gpu_list" | grep -c '^GPU ')
        if (( gpu_count != EXPECT_GPUS )); then
            fail "gpu_count=${gpu_count}, expected ${EXPECT_GPUS}"
        fi
    else
        rc=$?
        if (( rc == 124 )); then
            fail "nvidia-smi -L timed out after ${CMD_TIMEOUT}s (wedged GSP / driver)"
        else
            fail "nvidia-smi -L failed (rc=${rc})"
        fi
    fi
else
    warn "nvidia-smi not on PATH"
fi

# ---------------------------------------------------------------------------
# 4. GPU memory health: uncorrectable ECC and row remapping.
#    A pending row remap means failing HBM that has not been retired yet — the
#    jpbo-096-37 signature, which surfaced in the run as
#    "Invalid access of peer GPU memory over nvlink" (cudaErrorContained), and
#    an uncorrectable ECC count is the XID 48 signature that made job 1370589
#    show 4x throughput variance.
#    Fields are queried SEPARATELY: nvidia-smi fails the whole --query-gpu call
#    if any one field is unsupported, and support varies by driver.
#
#    VOLATILE, NOT AGGREGATE, IS THE FAULT SIGNAL — measured the hard way.
#    `ecc.errors.uncorrected.aggregate.total` is a LIFETIME counter that
#    persists across reboots and driver reloads, so it stays nonzero forever
#    after a single event that row remapping then retired successfully. Failing
#    on `> 0` therefore flags healthy hardware: in the 2026-08-23 campaign that
#    rule failed 71 nodes in four 512-node draws — 38 of them at aggregate=1,
#    13 at =2, 14 at =3 — and NOT ONE of them tripped the row-remap check below,
#    which is the check that actually says "this HBM is failing right now".
#    Extrapolated over eight draws it would have excluded a few hundred nodes
#    and shrunk the pool the production run schedules on for no reason.
#    `...volatile.total` resets when the driver reloads, so a nonzero value
#    means errors THIS boot — an active fault, which is what jpbo-114-47's
#    XID 48 was. Aggregate is kept only as a high-water mark: a GPU with a long
#    history of retired errors is a legitimate thing to avoid for a 40-day run,
#    but the bar has to be a lot higher than one.
# ---------------------------------------------------------------------------
ecc_unc="?"
ecc_vol="?"
remap_pending="?"
if (( gpu_count > 0 )); then
    if out=$(timeout "$CMD_TIMEOUT" nvidia-smi \
                --query-gpu=ecc.errors.uncorrected.volatile.total \
                --format=csv,noheader,nounits 2>/dev/null); then
        total=0
        seen=0
        while read -r value; do
            value="${value//[[:space:]]/}"
            [[ "$value" =~ ^[0-9]+$ ]] || continue
            total=$(( total + value ))
            seen=1
        done <<< "$out"
        if (( seen == 1 )); then
            ecc_vol=$total
            (( total > ECC_VOL_FAIL )) && \
                fail "uncorrectable ECC errors this boot (volatile)=${total}"
        fi
    fi

    if out=$(timeout "$CMD_TIMEOUT" nvidia-smi \
                --query-gpu=ecc.errors.uncorrected.aggregate.total \
                --format=csv,noheader,nounits 2>/dev/null); then
        total=0
        seen=0
        while read -r value; do
            value="${value//[[:space:]]/}"
            [[ "$value" =~ ^[0-9]+$ ]] || continue
            total=$(( total + value ))
            seen=1
        done <<< "$out"
        if (( seen == 1 )); then
            ecc_unc=$total
            if (( total >= ECC_AGG_FAIL )); then
                fail "uncorrectable ECC errors (lifetime aggregate)=${total} >= ${ECC_AGG_FAIL}"
            elif (( total >= ECC_AGG_WARN )); then
                warn "uncorrectable ECC errors (lifetime aggregate)=${total}"
            fi
        fi
    fi

    if out=$(timeout "$CMD_TIMEOUT" nvidia-smi \
                --query-gpu=remapped_rows.pending,remapped_rows.failure \
                --format=csv,noheader 2>/dev/null); then
        bad=0
        seen=0
        while IFS=, read -r pending failure; do
            pending="${pending//[[:space:]]/}"
            failure="${failure//[[:space:]]/}"
            [[ -n "$pending" ]] || continue
            seen=1
            # Drivers report either Yes/No or 1/0 depending on version.
            [[ "$pending" == "Yes" || "$pending" == "1" ]] && bad=$(( bad + 1 ))
            [[ "$failure" == "Yes" || "$failure" == "1" ]] && bad=$(( bad + 1 ))
        done <<< "$out"
        if (( seen == 1 )); then
            remap_pending=$bad
            (( bad > 0 )) && fail "row remapping pending/failed on ${bad} GPU field(s) — node needs a reset"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# 5. Filesystems. jpbo-075-09 had /e/project1 unmounted; its ranks blocked
#    forever on the first log write and the whole 512-node collective went
#    silent with NO error line anywhere. `timeout ... ls` rather than a
#    /proc/mounts grep on purpose: a hung GPFS mount is still LISTED.
# ---------------------------------------------------------------------------
IFS=':' read -r -a mount_list <<< "$REQUIRED_MOUNTS"
for mnt in "${mount_list[@]}"; do
    [[ -n "$mnt" ]] || continue
    if ! timeout "$CMD_TIMEOUT" ls "$mnt" >/dev/null 2>&1; then
        fail "required path ${mnt} not readable within ${CMD_TIMEOUT}s (unmounted or hung)"
    fi
done

# ---------------------------------------------------------------------------
# 6. Residual apptainer/squashfuse mounts from a previous job (jpbo-044-15).
#    Leftover mounts make the next container start fail or, worse, silently
#    read a stale image.
# ---------------------------------------------------------------------------
if [[ -r /proc/mounts ]]; then
    residual=$(grep -c -E 'squashfuse|squashfs.*apptainer|gocryptfs' /proc/mounts 2>/dev/null)
    residual="${residual:-0}"
    (( residual > 0 )) && fail "residual container mounts=${residual} (stale apptainer state)"
fi

# ---------------------------------------------------------------------------
# 7. InfiniBand. link_downed is cumulative since boot, so a single historic
#    blip is normal and only a repeat offender is worth excluding — hence the
#    default FAIL threshold of 2, which is what NHC drained jpbo-074-31 on
#    (mlx5_0: 2 / mlx5_1: 2). A port that is not ACTIVE is unconditionally bad.
# ---------------------------------------------------------------------------
linkdown_max=0
ib_seen=0
for port in /sys/class/infiniband/*/ports/*; do
    [[ -d "$port" ]] || continue
    ib_seen=1
    hca=$(basename "$(dirname "$(dirname "$port")")")
    if [[ -r "$port/state" ]]; then
        state=$(<"$port/state")
        if [[ "$state" != *ACTIVE* ]]; then
            fail "IB ${hca} port state=${state// /} (not ACTIVE)"
        fi
    fi
    if [[ -r "$port/counters/link_downed" ]]; then
        downed=$(<"$port/counters/link_downed")
        downed="${downed//[[:space:]]/}"
        [[ "$downed" =~ ^[0-9]+$ ]] || continue
        (( downed > linkdown_max )) && linkdown_max=$downed
        if (( downed >= LINKDOWN_FAIL )); then
            fail "IB ${hca} link_downed=${downed} >= ${LINKDOWN_FAIL}"
        elif (( downed > 0 )); then
            warn "IB ${hca} link_downed=${downed}"
        fi
    fi
done
(( ib_seen == 0 )) && warn "no InfiniBand ports found under /sys/class/infiniband"

# ---------------------------------------------------------------------------
# Verdict. FAIL lines are what config/job/node_catch.yaml regexes on, so the
# format is load-bearing: "[nodecheck] FAIL <node>: <reason>".
# ---------------------------------------------------------------------------
elapsed=$(( SECONDS - start_s ))
metrics="mem=${mem_avail_gb}GB numa_min=${numa_min_gb}GB/${numa_sockets}sock gpus=${gpu_count} ecc_vol=${ecc_vol} ecc_agg=${ecc_unc} remap=${remap_pending} linkdown=${linkdown_max} t=${elapsed}s"

for reason in "${warns[@]}"; do
    echo "[nodecheck] WARN ${NODE}: ${reason}"
done

if (( ${#fails[@]} > 0 )); then
    for reason in "${fails[@]}"; do
        echo "[nodecheck] FAIL ${NODE}: ${reason}"
    done
else
    # A node that passes every check but took far longer than its peers is
    # still suspect (slow storage, throttled CPU). Reported as SLOW rather than
    # FAIL: it is a ranking signal for the analysis script, not grounds for
    # removing a node from a 5500-node fleet on its own.
    if (( elapsed > SLOW_S )); then
        echo "[nodecheck] SLOW ${NODE}: probe took ${elapsed}s > ${SLOW_S}s"
    fi
    echo "[nodecheck] OK ${NODE} ${metrics}"
fi

# Rank 0 marks the end so the monitor can finish the job the moment the sweep
# is done instead of waiting for SLURM's own bookkeeping.
if [[ "${SLURM_PROCID:-0}" == "0" ]]; then
    # Give the other tasks a moment to flush into the shared stdout so the
    # marker really is the last interesting line in the log.
    sleep "${NODECHECK_DRAIN_S:-20}"
    echo "[nodecheck] SWEEP COMPLETE nodes=${SLURM_NNODES:-?} job=${SLURM_JOB_ID:-?}"
fi

exit 0
