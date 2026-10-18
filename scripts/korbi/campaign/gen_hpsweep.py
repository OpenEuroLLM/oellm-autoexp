"""Generate the LR x batch-size sweeps of the 3B campaign (docs/hybrid_moe_3b_campaign.md 3.5).

    python scripts/korbi/campaign/gen_hpsweep.py

Stable (constant LR) trunks to aux.tokens on the size's base model (screen geometry, datamix4);
the cooled loss is predicted with losspred's family fit (cooldown drop law), so no cooldowns are
trained here. Per optimizer a grid of LR (x2 steps around the ComplexKDA law's eta*) and global
batch size; MuOwn / angular MuOwn / nGPT start without warmup (AdamW: 1000 steps), with a
warmup A/B for MuOwn. Configs are listed flat (nested products break the sweep filter).
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "config/experiments/korbi/campaign_3b"
B = "backend.megatron."

LR = [2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3]
# MuOwn: the first grid had its optimum at the lowest point (2.5e-4); extended downwards
LR_MUOWN_LOW = [6.25e-5, 1.25e-4]
LR_NGPT = [1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2]
GBS = [32, 64, 128]


def points():
    pts = []

    def add(name, comment, **kv):
        pts.append((name, comment, kv))

    for lr in LR:
        for gbs in GBS:
            add(f"adamw_lr{lr:g}_b{gbs}", "AdamW, 1000-step warmup", **{"aux.lr": lr, "aux.gbs": gbs})
    for lr in LR:
        for gbs in GBS:
            add(f"muown_lr{lr:g}_b{gbs}", "MuOwn, no warmup", **{
                "aux.lr": lr, "aux.gbs": gbs, B + "optimizer": "muown", "aux.warmup_iters": 0})
    for lr in LR_MUOWN_LOW:
        for gbs in GBS:
            add(f"muown_lr{lr:g}_b{gbs}", "MuOwn, no warmup (low-LR extension)", **{
                "aux.lr": lr, "aux.gbs": gbs, B + "optimizer": "muown", "aux.warmup_iters": 0})
    for lr in (1e-3, 2e-3):
        add(f"muown_lr{lr:g}_b64_wu", "MuOwn with 1000-step warmup (A/B)", **{
            "aux.lr": lr, "aux.gbs": 64, B + "optimizer": "muown"})
    for lr in LR:
        add(f"amuown_lr{lr:g}_b64", "angular MuOwn, no warmup", **{
            "aux.lr": lr, "aux.gbs": 64, B + "optimizer": "angular_muown", "aux.warmup_iters": 0})
    for lr in LR_NGPT:
        add(f"ngpt_lr{lr:g}_b64", "nGPT (AdamW), no warmup", **{
            "aux.lr": lr, "aux.gbs": 64, B + "ngpt": True, "aux.warmup_iters": 0})
    # cooldown calibration: the stable trunks' cooled loss needs each optimizer's drop (the
    # losspred drop law lr^0.47 gbs^-0.32 is AdamW-fitted): WSD to aux.tokens with a 20 % linear
    # cooldown to lr / 10 at each optimizer's best LR and its neighbours (gbs 64). The cooldown
    # favours higher LRs (drop ~ lr^k, k ~ 0.43 AdamW / 0.64 angular MuOwn / 0.80 MuOwn), so the
    # third LR brackets the cooled optimum from above
    # (min_lr literal: an interpolation in a sweep entry resolves against the base aux.lr)
    wsd = {B + "lr_decay_style": "WSD", B + "lr_wsd_decay_style": "linear",
           B + "lr_wsd_decay_iters": "${oc.eval:'int(0.2 * ${backend.megatron.train_iters})'}"}
    for opt, lrs, extra in (
        ("adamw", (5e-4, 1e-3, 2e-3, 4e-3), {}),
        ("muown", (2.5e-4, 5e-4, 1e-3), {B + "optimizer": "muown", "aux.warmup_iters": 0}),
        ("amuown", (2e-3, 4e-3, 8e-3), {B + "optimizer": "angular_muown", "aux.warmup_iters": 0}),
        ("ngpt", (1e-3, 5e-4, 2e-3, 4e-3, 8e-3), {B + "ngpt": True, "aux.warmup_iters": 0}),
    ):
        for lr in lrs:
            add(f"{opt}_lr{lr:g}_b64_wsd", "cooldown calibration: WSD, 20 % linear cooldown to lr / 10",
                **{"aux.lr": lr, "aux.gbs": 64, **extra, **wsd, B + "min_lr": lr / 10})
    # batch-size axis for the chosen optimizer (angular MuOwn, 4e-3 cooled optimum at gbs 64):
    # stable trunks over lr x gbs, and one WSD branch per gbs (its drop law across batch sizes)
    am = {B + "optimizer": "angular_muown", "aux.warmup_iters": 0}
    for gbs in (32, 128, 256):
        for lr in (2e-3, 4e-3, 8e-3):
            add(f"amuown_lr{lr:g}_b{gbs}", "angular MuOwn, no warmup (batch-size axis)",
                **{"aux.lr": lr, "aux.gbs": gbs, **am})
        # the decay length as a literal: the shared ``wsd`` interpolation resolves against the
        # base gbs (64), i.e. a 10 / 40 / 80 % cooldown at gbs 32 / 128 / 256
        add(f"amuown_lr0.004_b{gbs}_wsd", "angular MuOwn, WSD branch (batch-size axis)",
            **{"aux.lr": 4e-3, "aux.gbs": gbs, **am, **wsd, B + "min_lr": 4e-4,
               B + "lr_wsd_decay_iters": int(0.2 * (int(1e9) // (gbs * 4096)))})
    # b*(D): the batch-size axis again at 3BT (WSD only, lr 4e-3), for the D exponent of b*
    for gbs in (64, 128, 256, 512):
        add(f"amuown_lr0.004_b{gbs}_3bt_wsd", "angular MuOwn, 3BT WSD (b* vs tokens)",
            **{"aux.lr": 4e-3, "aux.gbs": gbs, "aux.tokens": 3.0e9, **am, **wsd, B + "min_lr": 4e-4,
               B + "lr_wsd_decay_iters": int(0.2 * (int(3e9) // (gbs * 4096)))})
    # warmup study: the standard transformer (attention + qk-norm, AdamW) at gbs 128 for exactly
    # 2000 steps (2000 x 128 x 4096 tokens), constant LR after 50 / 200 / 1000 warmup steps, at
    # AdamW's cooled optimum (2e-3) and a high LR (4e-3); every step logged (short spikes)
    for lr in (2e-3, 4e-3):
        for wu in (50, 200, 1000):
            add(f"adamw_wu{wu}_lr{lr:g}_b128_2k", "warmup study: AdamW, 2000 steps, stable",
                **{"aux.lr": lr, "aux.gbs": 128, "aux.tokens": 2000 * 128 * 4096,
                   "aux.warmup_iters": wu, B + "log_interval": 1})
    # the same warmup study for angular MuOwn (its cooled optimum 4e-3; 0 = its no-warmup default)
    for wu in (0, 50, 200, 1000):
        add(f"amuown_wu{wu}_lr0.004_b128_2k", "warmup study: angular MuOwn, 2000 steps, stable",
            **{"aux.lr": 4e-3, "aux.gbs": 128, "aux.tokens": 2000 * 128 * 4096,
               "aux.warmup_iters": wu, B + "optimizer": "angular_muown", B + "log_interval": 1})
    # health grid: where the runs become unhealthy, for the linear health threshold over
    # hyperparameters (hp_model.py). 2000 steps, stable, every step logged.
    for g in (64, 256):
        for lr in (4e-3, 8e-3, 1.6e-2):
            for wu in (0, 50, 200):
                add(f"health_adamw_lr{lr:g}_wu{wu}_b{g}", "health grid: AdamW",
                    **{"aux.lr": lr, "aux.gbs": g, "aux.tokens": 2000 * g * 4096, "aux.warmup_iters": wu,
                       B + "log_interval": 1})
        for lr in (8e-3, 1.6e-2, 3.2e-2):
            add(f"health_amuown_lr{lr:g}_b{g}", "health grid: angular MuOwn, no warmup",
                **{"aux.lr": lr, "aux.gbs": g, "aux.tokens": 2000 * g * 4096, "aux.warmup_iters": 0,
                   B + "optimizer": "angular_muown", B + "log_interval": 1})
    return pts


HEADER = """# @package _global_
# GENERATED by scripts/korbi/campaign/gen_hpsweep.py -- edit there.
# LR x batch-size sweep, size {size} base model (attention, hidden {hidden} x {blocks} blocks),
# datamix4, STABLE trunks (constant LR after the optimizer's warmup) to aux.tokens; the cooled
# loss is predicted (losspred family fit). Per run: aux.lr, aux.gbs, aux.warmup_iters.
defaults:
  - screen_{size}_mn5
  - _self_

aux:
  suite: hp_{size}
  arm: hp
  tokens: {tokens}
  lr: 1.0e-3
  gbs: 64
  # 1000 warmup steps (the usual setting; the first sweeps' 1 % = 19-76 steps was far too short)
  warmup_iters: 1000

backend:
  megatron:
    lr: ${{aux.lr}}
    min_lr: ${{aux.lr}}
    lr_decay_style: constant
    lr_wsd_decay_iters: null
    lr_warmup_iters: ${{aux.warmup_iters}}
    eval_interval: 250

sweep:
  type: list
  store_sweep_json: true
  filter: "\\\\${{oc.eval:'len(\\\\${{aux.only}}) == 0 or \\"\\\\${{aux.arm}}\\" in \\\\${{aux.only}}'}}"
  groups:
    - type: list
      configs:
"""


def main():
    for size, hidden, blocks, tokens in (("s1", 768, 12, "1.0e9"),):
        lines = [HEADER.format(size=size, hidden=hidden, blocks=blocks, tokens=tokens)]
        for name, comment, kv in points():
            lines.append(f"        # {comment}\n        - aux.arm: {name}\n")
            for k, v in kv.items():
                val = yaml.safe_dump(v, default_flow_style=True).strip().removesuffix("...").strip()
                lines.append(f"          {k}: {val}\n")
        (OUT / f"hp_{size}_mn5.yaml").write_text("".join(lines))
        print(f"hp_{size}_mn5.yaml: {len(points())} runs")


if __name__ == "__main__":
    main()
