#!/usr/bin/env python3
"""Generate the Complex-KDA Megatron scaling sweeps from their spec.

    .venv/bin/python scripts/korbi/gen_ckda_sweep.py [--check-ladder-matched PATH]

Reads   config/experiments/korbi/complex_kda/sweep_spec.yaml
Writes  config/experiments/korbi/complex_kda/
          ckda_scaling_all.yaml          every ladder, one sweep
          ckda_<ladder>.yaml             one sweep per ladder
          MODELS.md                      parameter match, arm x size
          BUDGET.md                      chains, widths, hours, GPU-h

Run once after editing the spec and commit the outputs. Nothing here runs at
submit time: the generated YAML is plain data, every number in it a literal
(matched widths, branch iterations, node counts), so a sweep can be read and
priced without executing anything.

Each sweep is, per size, the product

    MODEL  (arm at this size, MLP width matched to the dense arm's params)
  x RECIPE (data|gbs|lr: one WSD chain = stable trunk + cooldown per budget)

and each cooldown is split into `firstcd` (load the trunk at the branch, one
step, save into its own dir) and `contcd` (resume that dir to the end, restart
safe across 12h windows) -- the multilingual main sweep's scheme.

PARAMETER COUNTS are analytic but mirror Megatron's module structure layer by
layer (fused TE norms, fla mixers, tied embedding counted once). They are
checked against ComplexKDA's `ladder_matched.json`, whose numbers come from
instantiating the real fla models, with --check-ladder-matched; every model
entry also carries `aux.params_by_size`, the number Megatron prints as
"number of parameters on (tensor, pipeline) model parallel rank (0, 0)".
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "config" / "experiments" / "korbi" / "complex_kda"
SPEC = OUT_DIR / "sweep_spec.yaml"
CONFIG_PREFIX = "experiments/korbi/complex_kda"
VALID_SEQS = 204_800  # the reference's held-out set (0.838BT at seq 4096)


# --------------------------------------------------------------------------
# Parameter counting, Megatron module by module.
# --------------------------------------------------------------------------
def _attention_te(g, qk_norm):
    """Megatron TE self-attention, pre-norm fused into linear_qkv."""
    d, h, hd, grp = g["d"], g["heads"], g["hd"], g["groups"]
    n = d  # fused input norm
    n += d * (h * hd + 2 * grp * hd)  # linear_qkv
    n += h * hd * d  # linear_proj
    if qk_norm:
        n += 2 * hd
    return n


def _attention_gated_nope(g, qk_norm):
    """fla Attention with Qwen3-Next output gate (ComplexKDAHybridAttention).
    Own projections, so the block builder places a real input norm."""
    d, grp = g["d"], g["groups"]
    kv = grp * (d // g["heads"])
    n = d  # input norm (fuse_input_layernorm False)
    n += d * d + 2 * d * kv + d * d  # q, k, v, o
    n += d * d  # g_proj
    if qk_norm:
        n += 2 * (d // g["heads"])
    return n


def _complex_kda(g, megatron):
    """fla ComplexKimiDeltaAttention as built by megatron/core/ssm/complex_kda.py:
    num_heads = d / head_dim, expand_v 1, short convs without bias."""
    d, hd, c = g["d"], g["hd"], g["conv"]
    h = d // hd
    key = value = h * hd
    hv = hd
    n = d  # input norm (fuse_input_layernorm False)
    n += 2 * d * key + d * value  # q, k, v
    n += (2 * key + value) * c  # q/k/v short convs
    n += d * hv + hv * key  # f_proj (low rank decay gate)
    n += d * h  # b_proj
    n += h + key  # A_log, dt_bias
    if megatron.get("linear_output_gate", "lowrank") == "lowrank":
        n += d * hv + hv * value + value  # g_proj (low rank, bias on 2nd)
    else:
        n += d * value + value
    n += hv  # o_norm (gated RMSNorm over head_v_dim)
    n += value * d  # o_proj
    return n


def _gated_delta_net(g, megatron):
    """megatron/core/ssm/gated_delta_net.py, pre-norm fused into in_proj."""
    d, c = g["d"], g["conv"]
    kh, vh = megatron["linear_num_key_heads"], megatron["linear_num_value_heads"]
    qk = megatron["linear_key_head_dim"] * kh
    v = megatron["linear_value_head_dim"] * vh
    n = d  # fused input norm
    n += d * (2 * qk + 2 * v + 2 * vh)  # in_proj: q, k, v, z, b, a
    n += (2 * qk + v) * c  # conv1d (conv_bias False)
    n += 2 * vh  # dt_bias, A_log
    n += megatron["linear_value_head_dim"]  # out_norm
    n += v * d  # out_proj
    return n


MIXER_PARAMS = {
    "complex_kda": _complex_kda,
    "gated_delta_net": _gated_delta_net,
}


def count_params(geom, ffn, layout, megatron, spec):
    d, layers = geom["d_model"], geom["layers"]
    g = {"d": d, "heads": geom["heads"], "groups": geom["heads"],
         "hd": spec["head_dim"], "conv": megatron.get("linear_conv_kernel_dim", 4)}
    qk_norm = bool(megatron.get("qk_layernorm", False))
    mlp = d + 3 * d * ffn  # fused pre-MLP norm + SwiGLU fc1 (2x) + fc2

    mixer = layout["mixer"]
    every = layout.get("attn_every")
    if mixer == "attention":
        per_attn = _attention_te(g, qk_norm)
        blocks = layers * (per_attn + mlp)
    else:
        per_lin = MIXER_PARAMS[mixer](g, megatron)
        if every:
            n_attn = sum(1 for i in range(layers) if (i + 1) % every == 0)
            attn_fn = (_attention_gated_nope if layout.get("hybrid_attn") == "gated_nope"
                       else _attention_te)
            per_attn = attn_fn(g, qk_norm)
        else:
            n_attn, per_attn = 0, 0
        blocks = (layers - n_attn) * (per_lin + mlp) + n_attn * (per_attn + mlp)
    embed = spec["vocab_size"] * d * (1 if spec["tie_embeddings"] else 2)
    return embed + blocks + d  # + final norm


def match_ffn(target, geom, layout, megatron, spec):
    """ffn on the `ffn_multiple` grid closest to the target count (exact
    search; N is linear in ffn)."""
    m = spec["ffn_multiple"]
    n0 = count_params(geom, 0, layout, megatron, spec)
    slope = count_params(geom, m, layout, megatron, spec) - n0
    k = (target - n0) / slope
    best = min((max(1, math.floor(k)), max(1, math.ceil(k))),
               key=lambda kk: abs(count_params(geom, kk * m, layout, megatron, spec) - target))
    return best * m, count_params(geom, best * m, layout, megatron, spec)


# --------------------------------------------------------------------------
# Models: arm x size.
# --------------------------------------------------------------------------
def arm_megatron(arm, geom):
    """The arm's Megatron flags at one size. Pure linear stacks get
    linear_attention_freq = layers + 1 (no layer satisfies (i+1) % freq == 0);
    `HEADS` placeholders take the size's head count."""
    out = {}
    for k, v in arm["megatron"].items():
        out[k] = geom["heads"] if v == "HEADS" else v
    layout = arm["layout"]
    if layout["mixer"] == "attention":
        out.setdefault("experimental_attention_variant", None)
        out.setdefault("linear_attention_freq", None)
    elif not layout.get("attn_every"):
        out["linear_attention_freq"] = geom["layers"] + 1
    elif out.get("linear_attention_freq") != layout["attn_every"]:
        raise SystemExit(f"arm layout attn_every {layout['attn_every']} disagrees "
                         f"with megatron linear_attention_freq {out.get('linear_attention_freq')}")
    return out


def rates_for(spec, arm, size, cap):
    """tok/s/GPU by micro-batch, Megatron-scaled, next-lower measured mbs."""
    table = spec["rates"][arm["rate_as"]][size]
    table = {int(k): float(v) for k, v in table.items()}
    out = {}
    m = 1
    while m <= cap:
        below = [k for k in table if k <= m]
        r = table[max(below)] if below else table[min(table)]
        out[m] = int(round(r * spec["megatron_rate_factor"]))
        m *= 2
    return out


def build_models(spec):
    models = {}
    for size, geom in spec["sizes"].items():
        target = count_params(geom, geom["ffn"], {"mixer": "attention"}, {}, spec)
        models[size] = {}
        for name, arm in spec["arms"].items():
            meg = arm_megatron(arm, geom)
            if arm["layout"]["mixer"] == "attention":
                ffn = geom["ffn"]  # the dense arms define the target
                n = count_params(geom, ffn, arm["layout"], meg, spec)
            else:
                ffn, n = match_ffn(target, geom, arm["layout"], meg, spec)
            models[size][name] = {
                "ffn": ffn, "params": n, "delta_pct": 100.0 * (n - target) / target,
                "target": target, "megatron": meg,
                "tok_s": rates_for(spec, arm, size, geom["mbs_cap"]),
            }
    return models


# --------------------------------------------------------------------------
# Recipes: chains, iterations, widths.
# --------------------------------------------------------------------------
def iters_for(tokens, gbs, seq):
    return (int(tokens) + seq * gbs - 1) // (seq * gbs)


def budget_tag(bt):
    return f"{bt:g}".replace(".", "p")


def plan_chain(chain, spec, models):
    seq, df = spec["seq_len"], spec["decay_fraction"]
    size, gbs = chain["size"], chain["gbs"]
    geom = spec["sizes"][size]
    cells = []
    for bt in sorted(chain["budgets"]):
        it = iters_for(bt * 1e9, gbs, seq)
        start = int(it * (1 - df))
        cells.append({"bt": bt, "tag": budget_tag(bt), "tokens": int(round(bt * 1e9)),
                      "iters": it, "start": start})
    stable_iters = max(c["start"] for c in cells)

    sz = spec["sizing"]
    options = []
    for n in range(geom["min_nodes"], sz["max_nodes"] + 1):
        dp = n * spec["gpus_per_node"]
        if gbs % dp:
            continue
        per_rank = gbs // dp
        mbs = max(m for m in range(1, geom["mbs_cap"] + 1) if per_rank % m == 0)
        hours = {}
        for arm in spec["run_arms"]:
            rate = models[size][arm]["tok_s"][mbs] * dp  # tok/s, whole job
            st = stable_iters * gbs * seq / rate / 3600
            # + the end-of-job eval over the full held-out set (forward ~1/3 step)
            ev = VALID_SEQS // gbs / 3.0
            cds = [(c["iters"] - c["start"] + ev) * gbs * seq / rate / 3600 for c in cells]
            hours[arm] = {"stable": st, "cooldowns": cds, "gpu_h": (st + sum(cds)) * dp}
        worst_st = max(h["stable"] for h in hours.values())
        worst_cd = max(max(h["cooldowns"]) for h in hours.values())
        cost = sum(h["gpu_h"] for h in hours.values())
        options.append({"nodes": n, "mbs": mbs, "hours": hours, "worst_stable": worst_st,
                        "worst_cooldown": worst_cd, "gpu_h": cost})
    fits = [o for o in options
            if o["worst_stable"] <= sz["stable_max_h"] and o["worst_cooldown"] <= sz["cooldown_max_h"]]
    pick = (min(fits, key=lambda o: (round(o["gpu_h"], 1), o["nodes"])) if fits
            else min(options, key=lambda o: (round(o["worst_stable"], 2), o["gpu_h"])))
    pick["fits"] = bool(fits)

    # Persistent trunk saves every ~stable_save_every_h of the SLOWEST arm.
    slowest_rate = min(models[size][a]["tok_s"][pick["mbs"]] for a in spec["run_arms"])
    iters_per_h = slowest_rate * pick["nodes"] * spec["gpus_per_node"] * 3600 / (gbs * seq)
    save_every = max(250, int(sz["stable_save_every_h"] * iters_per_h))
    return {**chain, "cells": cells, "stable_iters": stable_iters, "save_every": save_every,
            **{k: pick[k] for k in ("nodes", "mbs", "hours", "worst_stable",
                                    "worst_cooldown", "gpu_h", "fits")}}


# --------------------------------------------------------------------------
# YAML emission (hand-rolled so the output reads like a written config).
# --------------------------------------------------------------------------
def v(x):
    """Scalar/flow value as YAML. Strings go through JSON (valid YAML double
    quotes), so an escaped interpolation `\\${...}` round-trips exactly."""
    if x is None:
        return "null"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return f"{x:_}" if abs(x) >= 1_000_000 else str(x)
    if isinstance(x, float):
        return repr(x)
    if isinstance(x, str):
        return json.dumps(x)
    if isinstance(x, list):
        return "[" + ", ".join(v(e) for e in x) + "]"
    if isinstance(x, dict):
        return "{" + ", ".join(f"{k}: {v(e)}" for k, e in x.items()) + "}"
    raise TypeError(type(x))


def emit_mapping(lines, indent, mapping, first_prefix=None):
    pad = " " * indent
    for i, (k, val) in enumerate(mapping.items()):
        prefix = first_prefix if (i == 0 and first_prefix) else pad
        lines.append(f"{prefix}{k}: {v(val)}")


E = "\\$"  # an escaped interpolation: resolved per point, at final composition
OUT = E + "{oc.env:OUTPUT_DIR,'./output'}"


def firstcd_dir():
    """Mirrors job.name / base_output_dir of ckda_base.yaml for the firstcd stage."""
    return (f"{OUT}/{E}{{aux.experiment_group}}/{E}{{aux.arm}}_{E}{{aux.size}}_{E}{{aux.ladder}}"
            f"_gbs{E}{{aux.gbs}}_lr{E}{{backend.megatron.lr}}_firstcd_decay{E}{{aux.budget_tag}}BT")


def tracker_ge(dir_expr, it):
    """True once `dir` holds a COMPLETE checkpoint at iteration >= it. The
    tracker is written after the save finishes (an iter_ dir exists from the
    moment the save starts), and a trunk's saves are monotone."""
    return {"class_name": "ShellCommandCondition",
            "command": f"test \"$(cat {dir_expr}/latest_checkpointed_iteration.txt 2>/dev/null || echo 0)\" -ge {it}"}


def emit_arms(lines, indent, spec, models, sizes):
    """One entry per arm, carrying its per-size tables (matched ffn, expected
    params, tok/s by micro-batch). The recipe side sets aux.size, and the base
    looks the width up -- so MODEL = arm x size without a nested product."""
    for arm in spec["run_arms"]:
        meg = dict(models[sizes[0]][arm]["megatron"])
        if spec["arms"][arm]["layout"]["mixer"] != "attention" \
                and not spec["arms"][arm]["layout"].get("attn_every"):
            # pure linear stack: no layer satisfies (i+1) % freq == 0
            meg["linear_attention_freq"] = f"{E}{{oc.eval:'{E}{{aux.layers}}+1'}}"
        for size in sizes[1:]:
            other = dict(models[size][arm]["megatron"])
            other["linear_attention_freq"] = meg["linear_attention_freq"]
            if other != meg:
                raise SystemExit(f"{arm}: megatron flags differ across sizes ({size})")
        lines.append(f"{' ' * indent}# {arm}: " + ", ".join(
            f"{s} {models[s][arm]['params'] / 1e6:.1f}M ({models[s][arm]['delta_pct']:+.2f}%)"
            for s in sizes))
        entry = {
            "aux.arm": arm,
            "aux.ffn_by_size": {s: models[s][arm]["ffn"] for s in sizes},
            "aux.params_by_size": {s: models[s][arm]["params"] for s in sizes},
            "aux.tok_s": {s: {f"m{k}": r for k, r in models[s][arm]["tok_s"].items()}
                          for s in sizes},
        }
        entry.update({f"backend.megatron.{k}": val for k, val in meg.items()})
        emit_mapping(lines, indent + 2, entry, first_prefix=" " * indent + "- ")


def emit_chain(lines, indent, spec, ladder, c):
    p = " " * indent
    budgets = "/".join(f"{x['bt']:g}" for x in c["cells"])
    lines.append(f"{p}# {ladder} | {c['size']} gbs{c['gbs']} lr{c['lr']} | {budgets}BT | "
                 f"{c['nodes']}n mbs{c['mbs']} | trunk <= {c['worst_stable']:.1f}h, "
                 f"cooldown <= {c['worst_cooldown']:.1f}h | {c['gpu_h']:,.0f} GPU-h (all arms)"
                 + ("" if c["fits"] else "  !! exceeds sizing targets"))
    lines.append(f"{p}- type: list")
    lines.append(f"{p}  defaults:")
    g = spec["sizes"][c["size"]]
    emit_mapping(lines, indent + 4, {
        "aux.ladder": ladder,
        "aux.size": c["size"],
        "aux.hidden": g["d_model"],
        "aux.layers": g["layers"],
        "aux.heads": g["heads"],
        "aux.gbs": c["gbs"],
        "aux.mbs": c["mbs"],
        "aux.mbs_key": f"m{c['mbs']}",
        "backend.megatron.lr": c["lr"],
        "slurm.sbatch.nodes": c["nodes"],
    })
    lines.append(f"{p}  configs:")
    # ---- stable trunk ----
    branches = [x["start"] for x in c["cells"]]
    emit_mapping(lines, indent + 6, {
        "stage": "stable",
        "aux.tokens": c["cells"][-1]["tokens"],
        "aux.extra_steps": branches,
        "backend.megatron.train_iters": c["stable_iters"],
        "backend.megatron.lr_wsd_decay_iters": 0,
        "backend.megatron.save_interval": c["save_every"],
    }, first_prefix=p + "    - ")
    # ---- cooldowns ----
    lines.append(f"{p}    - type: list")
    lines.append(f"{p}      configs:")
    q = p + "        "
    lines.append(f"{q}- type: list")
    lines.append(f"{q}  defaults:")
    emit_mapping(lines, indent + 12, {
        "stage": f"firstcd_decay{E}{{aux.budget_tag}}BT",
        "backend.megatron.load": f"{E}{{sibling.stable.job.base_output_dir}}",
        "backend.megatron.override_opt_param_scheduler": True,
        "backend.megatron.exit_on_missing_checkpoint": True,
        "backend.megatron.lr_wsd_decay_iters": 1,
        "backend.megatron.save_interval": 1,
        "backend.megatron.eval_iters": 1,
    })
    lines.append(f"{q}  configs:")
    for x in c["cells"]:
        emit_mapping(lines, indent + 14, {
            "aux.budget_tag": x["tag"],
            "aux.tokens": x["tokens"],
            "aux.start_iter": x["start"],
            "backend.megatron.ckpt_step": x["start"],
            "backend.megatron.train_iters": x["start"] + 1,
            "job.start_condition": tracker_ge(f"{E}{{sibling.stable.job.base_output_dir}}", x["start"]),
        }, first_prefix=q + "    - ")
    lines.append(f"{q}- type: list")
    lines.append(f"{q}  defaults:")
    emit_mapping(lines, indent + 12, {
        "stage": f"contcd_decay{E}{{aux.budget_tag}}BT",
        "backend.megatron.load": firstcd_dir(),
        "backend.megatron.save": firstcd_dir(),
        "backend.megatron.override_opt_param_scheduler": True,
        "backend.megatron.exit_on_missing_checkpoint": True,
        "backend.megatron.eval_iters": VALID_SEQS // c["gbs"],
    })
    lines.append(f"{q}  configs:")
    for x in c["cells"]:
        span = x["iters"] - x["start"]
        emit_mapping(lines, indent + 14, {
            "aux.budget_tag": x["tag"],
            "aux.tokens": x["tokens"],
            "aux.start_iter": x["start"],
            "backend.megatron.train_iters": x["iters"],
            "backend.megatron.lr_wsd_decay_iters": span,
            # ~3 restart points inside the decay; the endpoint is the final save
            "backend.megatron.save_interval": max(1, span // 3),
            "job.start_condition": tracker_ge(firstcd_dir(), x["start"] + 1),
        }, first_prefix=q + "    - ")


HEADER = """\
# @package _global_
# GENERATED by scripts/korbi/gen_ckda_sweep.py from sweep_spec.yaml -- do not
# edit by hand; change the spec and re-run the generator.
#
# {title}
#
# MODEL (arm; its MLP width per size matched to the dense arm's parameter
# count) x RECIPE (size | data | gbs | lr chain: stable trunk to 0.8 x max
# budget with a persistent save at every branch, then per budget
# firstcd -> contcd).
# Sizes/budgets/params/hours: MODELS.md and BUDGET.md next to this file.
#
# Points: {points} ({stables} stable, {cds} firstcd, {cds} contcd).
# Estimated compute: {gpu_h:,.0f} GPU-h (measured torchtitan rates x Megatron
# factor, see sweep_spec.yaml; firstcd seeds are ~0).
#
# Submit (or --dry-run first; --array-subset for a slice):
#   scripts/run_autoexp.py --submit-and-exit --config-name {config}
defaults:
  - ckda_base
  - _self_

sweep:
  type: product
  store_sweep_json: true
  groups:
    - type: list
      configs:
"""


def emit_sweep(spec, models, chains_by_ladder, ladders, name, title):
    by_size = {}
    for lad in ladders:
        for c in chains_by_ladder[lad]:
            by_size.setdefault(c["size"], []).append((lad, c))
    n_arms = len(spec["run_arms"])
    stables = n_arms * sum(len(v) for v in by_size.values())
    cds = n_arms * sum(len(c["cells"]) for v in by_size.values() for _l, c in v)
    gpu_h = sum(c["gpu_h"] for v in by_size.values() for _l, c in v)
    lines = HEADER.format(title=title, points=stables + 2 * cds, stables=stables, cds=cds,
                          gpu_h=gpu_h, config=f"{CONFIG_PREFIX}/{name}").splitlines()
    sizes = [s for s in spec["sizes"] if s in by_size]
    lines.append("        # ================= MODEL: arm (x size via aux.size) =================")
    emit_arms(lines, 8, spec, models, sizes)
    lines.append("    # ================= RECIPE: size | data | gbs | lr chains =================")
    lines.append("    - type: list")
    lines.append("      configs:")
    for size in sizes:
        g = spec["sizes"][size]
        lines.append(f"        # ---------- {size}: d {g['d_model']}, {g['layers']} layers, "
                     f"{g['heads']} heads ----------")
        for lad, c in by_size[size]:
            emit_chain(lines, 8, spec, lad, c)
    (OUT_DIR / f"{name}.yaml").write_text("\n".join(lines) + "\n")
    return stables + 2 * cds, gpu_h


# --------------------------------------------------------------------------
# Reports.
# --------------------------------------------------------------------------
def write_models_md(spec, models):
    out = ["# Complex-KDA sweep: models (generated)", "",
           "MLP width matched to the dense `attn` arm at each size (grid "
           f"{spec['ffn_multiple']}). Params are total, tied embedding counted once; "
           "`aux.params_by_size` in the sweep is the same number, to compare against "
           "Megatron's startup print.", ""]
    for size, g in spec["sizes"].items():
        out.append(f"## {size}: d {g['d_model']}, {g['layers']} layers, {g['heads']} heads, "
                   f"dense ffn {g['ffn']}")
        out.append("")
        out.append("| arm | params | delta vs dense | ffn | ffn/d | tok/s/GPU @ mbs cap |")
        out.append("|---|---:|---:|---:|---:|---:|")
        for arm, m in models[size].items():
            run = "" if arm in spec["run_arms"] else " (not run)"
            out.append(f"| {arm}{run} | {m['params']:,} | {m['delta_pct']:+.2f}% | {m['ffn']} | "
                       f"{m['ffn'] / g['d_model']:.2f} | {m['tok_s'][max(m['tok_s'])]:,} |")
        out.append("")
    (OUT_DIR / "MODELS.md").write_text("\n".join(out))


def write_budget_md(spec, chains_by_ladder, totals):
    out = ["# Complex-KDA sweep: recipes and compute (generated)", "",
           "Per chain: one stable trunk (to 0.8 x largest budget) and one cooldown "
           "(20% linear decay) per budget. Hours are wall-clock of the slowest arm at "
           "the chosen width; GPU-h summed over all run arms (" + ", ".join(spec["run_arms"]) + ").",
           ""]
    for lad, chains in chains_by_ladder.items():
        n, gh = totals[lad]
        out.append(f"## {lad}: {n} points, {gh:,.0f} GPU-h")
        out.append("")
        out.append("| size | gbs | lr | budgets (BT) | nodes | mbs | trunk iters | trunk h | "
                   "max cooldown h | GPU-h |")
        out.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|")
        for c in chains:
            b = ", ".join(f"{x['bt']:g}" for x in c["cells"])
            flag = "" if c["fits"] else " !!"
            out.append(f"| {c['size']} | {c['gbs']} | {c['lr']} | {b} | {c['nodes']} | {c['mbs']} | "
                       f"{c['stable_iters']:,} | {c['worst_stable']:.1f}{flag} | "
                       f"{c['worst_cooldown']:.1f} | {c['gpu_h']:,.0f} |")
        out.append("")
    out.append("GPU-h per arm, all ladders:")
    out.append("")
    per_arm = {a: 0.0 for a in spec["run_arms"]}
    for chains in chains_by_ladder.values():
        for c in chains:
            for a, h in c["hours"].items():
                per_arm[a] += h["gpu_h"]
    out.append("| arm | GPU-h |")
    out.append("|---|---:|")
    for a, h in per_arm.items():
        out.append(f"| {a} | {h:,.0f} |")
    (OUT_DIR / "BUDGET.md").write_text("\n".join(out) + "\n")


def check_ladder_matched(path, spec, models):
    """Compare against ComplexKDA's fla-instantiated table (keys: 47M, ...,
    arms with '-' names)."""
    ref = json.loads(Path(path).read_text())
    bad = 0
    for size in spec["sizes"]:
        tag = size[1:].replace("p", ".")
        for arm in models[size]:
            rarm = {"attn_qknorm": None}.get(arm, arm.replace("_", "-"))
            if rarm is None or rarm not in ref[tag]["archs"]:
                continue
            r = ref[tag]["archs"][rarm]
            m = models[size][arm]
            ok = (r["d_ffn"], r["N"]) == (m["ffn"], m["params"])
            bad += not ok
            print(f"  {'ok' if ok else '!!'} {size:6s} {arm:30s} ffn {m['ffn']:5d} vs {r['d_ffn']:5d}   "
                  f"N {m['params']:>13,} vs {r['N']:>13,}")
    print(f"{bad} mismatches against {path}")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-ladder-matched", metavar="PATH",
                    help="ComplexKDA lm_scaling/ladder_matched.json to verify the counts against")
    a = ap.parse_args()

    spec = yaml.safe_load(SPEC.read_text())
    models = build_models(spec)
    if a.check_ladder_matched and check_ladder_matched(a.check_ladder_matched, spec, models):
        raise SystemExit(1)

    chains_by_ladder = {lad: [plan_chain(c, spec, models) for c in chains]
                        for lad, chains in spec["ladders"].items()}
    totals = {}
    for lad in spec["ladders"]:
        totals[lad] = emit_sweep(spec, models, chains_by_ladder, [lad], f"ckda_{lad}",
                                 f"Complex-KDA Megatron scaling sweep: {lad}.")
    all_n, all_h = emit_sweep(spec, models, chains_by_ladder, list(spec["ladders"]),
                              "ckda_scaling_all",
                              "Complex-KDA Megatron scaling sweep: ALL ladders ("
                              + ", ".join(spec["ladders"]) + ").")
    write_models_md(spec, models)
    write_budget_md(spec, chains_by_ladder, totals)
    for lad, (n, h) in totals.items():
        print(f"  ckda_{lad}.yaml: {n} points, {h:,.0f} GPU-h")
    print(f"  ckda_scaling_all.yaml: {all_n} points, {all_h:,.0f} GPU-h")


if __name__ == "__main__":
    main()
