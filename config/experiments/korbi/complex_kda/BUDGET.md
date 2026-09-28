# Complex-KDA sweep: recipes and compute (generated)

Per chain: one stable trunk (to 0.8 x largest budget) and one cooldown (20% linear decay) per budget. Hours are wall-clock of the slowest arm at the chosen width; GPU-h summed over all run arms (attn, attn_qknorm, kda_sig_lowrank, ckda_shipped_lowrank, kda_sig_hybrid_lowrank, ckda_shipped_hybrid_lowrank).

## ladder: 222 points, 3,298 GPU-h

| size | gbs | lr | budgets (BT) | nodes | mbs | trunk iters | trunk h | max cooldown h | GPU-h |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| s47M | 32 | 0.002 | 6 | 1 | 8 | 36,621 | 3.0 | 0.9 | 62 |
| s47M | 64 | 0.002 | 12, 20, 30 | 1 | 16 | 91,552 | 7.2 | 1.9 | 225 |
| s47M | 128 | 0.002 | 50 | 1 | 16 | 76,294 | 12.1 | 3.1 | 303 |
| s124M | 64 | 0.001 | 6, 12, 20 | 1 | 16 | 61,035 | 8.7 | 2.3 | 289 |
| s124M | 128 | 0.001 | 30, 50 | 1 | 16 | 76,294 | 21.7 | 5.6 | 668 |
| s302M | 64 | 0.001 | 6, 12 | 1 | 8 | 36,621 | 10.3 | 2.9 | 323 |
| s302M | 128 | 0.001 | 20, 30, 50 | 2 | 8 | 76,294 | 21.4 | 5.5 | 1,429 |

## ladder_medium: 216 points, 14,133 GPU-h

| size | gbs | lr | budgets (BT) | nodes | mbs | trunk iters | trunk h | max cooldown h | GPU-h |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| s588M | 64 | 0.0005 | 6, 12 | 2 | 4 | 36,621 | 9.3 | 2.6 | 554 |
| s588M | 128 | 0.0005 | 20, 30, 50 | 4 | 4 | 76,294 | 19.3 | 5.0 | 2,449 |
| s983M | 64 | 0.0005 | 6, 12 | 2 | 4 | 36,621 | 13.0 | 3.6 | 800 |
| s983M | 128 | 0.0005 | 20, 30, 50 | 8 | 4 | 76,294 | 13.5 | 3.5 | 3,537 |
| s1p7B | 64 | 0.0005 | 6 | 4 | 2 | 18,311 | 5.6 | 1.7 | 611 |
| s1p7B | 128 | 0.0005 | 12, 20, 30, 50 | 8 | 2 | 76,294 | 23.2 | 6.0 | 6,182 |

## ladder_finegrained: 174 points, 356 GPU-h

| size | gbs | lr | budgets (BT) | nodes | mbs | trunk iters | trunk h | max cooldown h | GPU-h |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| s47M | 16 | 0.002 | 1, 1.5 | 1 | 4 | 18,311 | 1.4 | 0.7 | 42 |
| s47M | 32 | 0.002 | 2, 4 | 1 | 8 | 24,414 | 2.0 | 0.7 | 49 |
| s124M | 16 | 0.001 | 1 | 1 | 4 | 12,207 | 1.3 | 0.8 | 39 |
| s124M | 32 | 0.001 | 1.5, 2, 4 | 1 | 8 | 24,414 | 2.5 | 0.9 | 85 |
| s302M | 32 | 0.001 | 1, 1.5, 2, 4 | 1 | 8 | 24,414 | 3.4 | 1.2 | 141 |

GPU-h per arm, all ladders:

| arm | GPU-h |
|---|---:|
| attn | 2,572 |
| attn_qknorm | 2,572 |
| kda_sig_lowrank | 3,248 |
| ckda_shipped_lowrank | 3,346 |
| kda_sig_hybrid_lowrank | 2,959 |
| ckda_shipped_hybrid_lowrank | 3,092 |
