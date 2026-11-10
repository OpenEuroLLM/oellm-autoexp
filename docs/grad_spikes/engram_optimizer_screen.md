# Engram table optimizer screen (screen5engopt_s3, 2026-11-10)

Question: before changing the v4_2ph main run's Engram tables at 25k, does DeepSeek-V4.1's Sinkhorn-balanced momentum
update (report §2.5, Algorithm 1; Megatron f05a44c25 `--engram-table-optimizer sinkhorn`) cost loss, and how do the
table row norms behave compared with Adam at x5 (the run up to 25k) and x1 (the 25k setting)?

Setup: `config/experiments/korbi/campaign_3b/screen5engopt_s3_mn5.yaml` (autoexp 81302f73): screen5e_s3's h5_moe_eng45
base (s3, hidden 1280, CKDA 5:1 hybrid, 80 experts top-3 + shared, Engram [2, 3] x 8 heads, angular MuOwn, flag 8k,
gbs 64 x 8192, 4000 of 7629 steps) with the main run's table settings (row-wise v, bf16 momentum). Arms differ only
in the table update: adam5 (TableAdam, engram_lr_mult 5), adam1 (x1), sink5 (Sinkhorn, x5, lr ratio 0.18, momentum
0.95, K 11). Same train_iters / seed / gbs: paired data order. Pool runs p264.0382-0384 (host003), 3.1 s/step.

## Result (4000 steps)

| | adam5 | adam1 | sink5 |
|---|---|---|---|
| train loss 3500-3990 (mean) | 2.2184 | 2.2192 | 2.2211 |
| validation @ 4000 | 2.1878 | 2.1893 | 2.1904 |
| vs adam5, train / val (mnat) | - | +0.9 / +1.5 | +2.7 / +2.6 |
| largest table row @ 4000 | 8.87 | 1.78 | 12.91 |
| mean table row norm @ 4000 | 3.3 | 0.73 | 6.2 |
| largest / mean row norm | ~2.7 | ~2.4 | ~2.0 |
| grad-norm median / p99 / steps > 1.0 (2000-4000) | 0.243 / 0.400 / 2 | 0.245 / 0.397 / 2 | 0.241 / 0.379 / 0 |

- sink5 is slightly but consistently worse: +2.7 ... +3.6 mnat vs adam5 and +1.7 ... +2.0 vs adam1 in every
  500-step window from step 500 on, confirmed by every validation point from 1000 on (+2.6 ... +4.8 / +1.1 ... +2.8).
- adam1 costs ~1 mnat vs adam5 (within noise, systematic) with ~5x smaller rows.
- Sinkhorn makes the row-norm distribution the most uniform (largest ~2x mean, constant) but grows the rows fastest:
  mean 6.2 and still rising, largest 12.9 -- above Adam x5. With a row-norm cap at 7 it would bind on many rows.
- Gradient spikes: none of the arms spikes at this scale; sink5 has a slightly smaller p99 and no clipped step.

## Decision

The main run took the Adam x1 package at 25k (engram_lr_mult 1.0, engram_clip_grad 1.0, row-norm cap 13 -> 7 over
25000-26000; restart 2026-11-10 04:17, pool run p264.0403). sink5 is prepared but not deployed (snapshot
autoexp_v4hd256p2sink5); it would need an Adam -> Sinkhorn resume test, and on this screen it is not better.
Unscreened: Sinkhorn at engram_lr_mult 1 (~0.18 x lr per element, uniform), the candidate if the Engram tail or
Engram-led spikes come back after the cap ramp.
