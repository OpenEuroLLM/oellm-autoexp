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
| 16 | 0.95 | not reached: node-set failures (see §3) |

The per-GPU parameter/optimizer shard scales as 1/nodes. The 128-node stream arm peaked at ~56.7 GiB;
at 8 nodes the shard adds ~9.5 GiB (≈66 GiB, matching the 62.7 GiB observed with
`recompute_num_layers: 4`); at 16 nodes it adds ~4.4 GiB (≈58 GiB, inside the 60.12 GiB cap). So 16
nodes is the first viable size on these nodes.

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

The 16-node failures were these node-set issues, not the Engram code: `fix_read` attempt 2328
excluded `as02r3b02`; earlier attempts failed the rendezvous on `as05r2b29`. The light check passes
both the GPU and the ring; it does not cover the TCP store.

## 4. Shrink policy

Trigger: the host has **≥ 16 idle nodes for ≥ 5 min**.

Procedure (a resize changes the Slurm job, so it needs the user's explicit go):

1. `pool_ctl freeze host003` — no new run starts.
2. Wait until no run is active on the host, or preempt (only with the user's go).
3. `pool_ctl shrink host003 --keep <used nodes + batch node> --apply`.
4. `pool_ctl unfreeze host003`.

Caveat from `pool_ctl`'s own note: a step running during `scontrol update NodeList=` stays RESIZING
and keeps its GPUs after it ends, which gives `Insufficient GRES` to every later step on those
nodes. Keep the active runs' nodes and prefer the shrink when no run is active.

Current plan: host003 uses 96 nodes (`v4b_ctx64` 64 + eight cds evals 32) with 36 idle. Shrink to
**96** after the active runs finish (or with the user's go to preempt them). The 36 idle nodes pass
the light check, so the shrink removes only nodes that are healthy but unused.

## 5. Current state and next steps

- host003: 96 used / 36 free; the three 16-node arms are cancelled (`fix` attempt 19, `fix_read` 13,
  `fix_stream` 11).
- Main run: `ckpt` 59000, `ckpt_rolling` 59800; p264 job `48871798` (264 nodes), unaffected.
- Next: resubmit the 16-node arms on host003 after excluding `as05r2b29` (and keeping the light-check
  exclusions), or run the 10-layer e2e on one node. The offline fixed-query NLL job still needs the
  `--docs ''` env fix. The in-place 1k-step main-run switch remains at the next boundary (59,000).
