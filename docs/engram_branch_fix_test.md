# Engram gate fix + per-stream gates: host003 node check and shrink policy

Status 2026-11-26. The code (Megatron `engram_gate_logit_scale` / `engram_gate_query_eps`, the
scale-invariant query in the branch path, the surgery and the 4/8/16-node arms) is in the
`engram-branch-fix` worktree. The full-model arms are blocked by node health and node memory, not by
the Engram code.

## 1. Full-model arms: memory wall on the 64 GB c136 nodes

| nodes | cap | result |
|---:|---|---|
| 4 | 0.80 | OOM at model construction: 52.97 GiB in use vs 50.63 GiB allowed (`MoETransformerLayer` / `HybridStack`) |
| 8 | 0.80 | OOM at the first step: 52.1 GiB vs 50.63 GiB |
| 8 | 0.95 | OOM at 62.69 GiB of the node's 63.29 GiB — the node is physically full |
| 16 | 0.95 | OOM at step 0: ~62.7 GiB in use vs the 60.12 GiB cap (the 63.29 GiB node is full) |

The 128-node stream arm peaked at ~56.7 GiB. At 8 and 16 nodes the arms OOMed at ~62.7 GiB (the
63.29 GiB node is full): the 16-node estimate of ~58 GiB was wrong, and the activation memory, not the
parameter shard, dominates. 16 nodes is **not** viable; the 64-node arm is the next step.

## 2. Node check on host003's free nodes (2026-11-26 03:18)

Host003 was using 96 of 132 nodes (`v4b_ctx64_gbf16x_gcf` 64 + eight 4-node `cds_window_*` evals 32);
the three cancelled arms were freeing 32 more. The bench ran as a separate pool run
`node_health_host003_free_d2bb82` (`mn5-pool-c136.2371`), `node_health_mn5`, 36 free nodes, light
profile.

Free nodes: `as02r3b31`, `as03r2b27`, `as03r2b29`, `as03r2b31`, `as03r2b32`, `as05r2b17`–`as05r2b20`,
`as06r2b24`, `as06r2b28`, `as06r4b17`, `as06r4b19`, `as06r5b22`, `as06r5b32`, `as07r1b27`,
`as07r1b29`, `as07r2b05`, `as07r2b11`, `as07r2b13`, `as07r2b14`, `as07r2b18`, `as07r2b26`–`as07r2b28`,
`as07r2b30`–`as07r2b31`, `as07r3b01`–`as07r3b03`, `as07r3b05`–`as07r3b10`.

| measure | value | threshold | verdict |
|---|---:|---:|---|
| GPU matmul median | 739.0 TFLOP/s | — | — |
| GPU matmul min | 646.6 (`as07r2b13`) | 0.75 × median = 554 | pass |
| ring median | 12.18 GB/s | — | — |
| ring min | 10.33 (`as02r3b31`↔`as03r2b27`, `as07r3b10`↔`as02r3b31`) | 0.8 × median = 9.74 | pass |
| GPU groups | 23.8–24.1 GB/s | 0.85 × median ≈ 20.4 | pass |

**Verdict: pass, no exclusions.** Watch items (slowest ring, still above threshold): `as02r3b31`,
`as07r3b10`. The 32 nodes held by the active runs and the 4 spare were not in this run.

## 3. host003 exclusions

| node | level | reason |
|---|---|---|
| `as07r3b28` | host003 | host preflight |
| `as05r2b27` | host003 | host preflight |
| `as02r3b02` | host003 | the arms' light node check (round 1 of 2) |
| `as06r4b23` | host003 | the arms' light node check (a later attempt) |
| `as05r2b29` | pool | TCP-store rendezvous failure (`TCPStore::doWait`, `RendezvousConnectionError`) — not a GPU/IB check item, but it fails the run's rendezvous |

The four host-level exclusions live in `hosts/host003/exclude_nodes`; `as05r2b29` is in the pool's
`exclude_nodes`, so no host launches on it.

The 16-node arms did start, but they OOMed at step 0 at ~62.7 GiB (vs the 60.12 GiB cap). The
rendezvous errors on the batch node (`as01r1b17` for fix/fix_stream, `as01r1b29` for fix_read;
`Broken pipe`, `RendezvousConnectionError`) follow the dead ranks. `fix_read` also excluded
`as02r3b02`. The light check passes the GPU and the ring; it does not test memory or the TCP store.

## 4. Idle policy (updated 2026-11-26)

- **135-node hosts (c136):** no freeze. When the host is idle for >= 5 min it is freed automatically
  (`idle_exit_s=300` on the c136 pool). The pool's chain successors keep the capacity.
- **264-node hosts (p264):** freeze/hold when idle; only 264-node jobs are frozen.
- The earlier "shrink on >= 16 idle for 5 min" procedure is superseded for the 135-node hosts: they
  exit instead of being resized. A resize remains available for a 264-node host if needed.

## 5. Current state and next steps

- host003: cancelled by the user (nothing was running); the c136 pool's `idle_exit_s=300`.
- Main run: `ckpt` 61000, `ckpt_rolling` 61250; p264 job `48871798` (264 nodes), unaffected.
- The three 16-node arms are cancelled. The 64-node rerun needs a 135-node c136 host
  (host004/006/007/008 when one starts); the host frees itself 5 min after it goes idle.
- Next: rerun the arms at 64 nodes (the 128-node stream arm peaked at 56.7 GiB), or run the 10-layer
  e2e on one node. The offline fixed-query NLL job still needs the `--docs ''` env fix. The in-place
  1k-step main-run switch is at the next boundary (62,000).
