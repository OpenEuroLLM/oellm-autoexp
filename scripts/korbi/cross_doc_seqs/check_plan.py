#!/usr/bin/env python3
"""Production-scale check of packing plans through Megatron itself (docs/cross_doc_seqs/README.md): the run's
real training BlendedDataset (531 datasets, from the cache, read-only) wrapped by --packing-plan-path's
loader exactly as pretrain_hybrid.py does.

    check_plan.py --out <pipeline dir> --plan-name q3e256_w500 --data-cache-path <cache> \
        --data-args-path <datamix> --tokenizer <training tokenizer>

1. The BlendedDataset rebuilt from the cache has the run's description hash (= the plans' blend_md5).
2. wrap_with_packing_plans with the run's sampler geometry: geometry check, coverage from the first plan's
   start, the end of the contiguous planned range.
3. Per window: the plan's slots are exactly the window's non-solo slots, and its sequences hold exactly the
   window's owned tokens, as often as the window reads them (coverage arithmetic over (dataset, document,
   offset)).
4. Per boundary mode (doc / group / none): random planned samples served by PackingPlanDataset equal the plan's
   segments read independently from the .bin (tokens, labels, cu_seqlens, positions); random unplanned
   samples equal the BlendedDataset's.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_plan  # noqa: E402
from cds_data import BlendStream, find_cache_entry, read_blend  # noqa: E402
from cds_megatron import MegatronBlend  # noqa: E402


def owned_intervals(ds, doc, off, ln):
    """Canonical form of a multiset of token intervals: the net +1 / -1 coverage changes per (document,
    position), sorted, zeros dropped. Equal iff every token appears equally often (a document read twice
    by the window -- another epoch -- has coverage 2 and must have it in the plan too)."""
    key = build_plan.doc_key(np.asarray(ds), np.asarray(doc))
    start = np.asarray(off, np.int64)
    end = start + np.asarray(ln, np.int64)
    keep = end > start
    k = np.r_[key[keep], key[keep]]
    pos = np.r_[start[keep], end[keep]]
    delta = np.r_[np.ones(keep.sum(), np.int64), -np.ones(keep.sum(), np.int64)]
    o = np.lexsort((pos, k))
    k, pos, delta = k[o], pos[o], delta[o]
    new = np.r_[True, (k[1:] != k[:-1]) | (pos[1:] != pos[:-1])]
    idx = np.nonzero(new)[0]
    net = np.add.reduceat(delta, idx)
    nz = net != 0
    return k[idx][nz], pos[idx][nz], net[nz]


def plan_owned(z):
    """The plan's owned segments: every sequence's segments without its last (label-only) token."""
    ln = z["seg_len"].astype(np.int64).copy()
    last = z["seq_ptr"][1:] - 1
    ln[last] -= 1
    return z["seg_ds"], z["seg_doc"], z["seg_off"], ln


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plan-name", required=True)
    ap.add_argument("--data-cache-path", required=True)
    ap.add_argument("--data-args-path", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--bos-id", type=int, default=None)
    ap.add_argument("--eod-id", type=int, default=None)
    ap.add_argument("--vocab-size", type=int, default=262144)
    ap.add_argument("--samples", type=int, default=32, help="random planned / unplanned samples per mode")
    a = ap.parse_args()

    from megatron.core.datasets.packing_plan import PackingPlanDataset, wrap_with_packing_plans

    blend = json.load(open(os.path.join(a.out, "blend.json")))
    _, prefixes = read_blend(a.data_args_path)
    entry = find_cache_entry(a.data_cache_path, prefixes, blend["seq_length"], blend_md5=blend["blend_md5"])
    bos, eod, vocab = a.bos_id, a.eod_id, a.vocab_size
    if a.tokenizer:
        from transformers import AutoTokenizer

        hf = AutoTokenizer.from_pretrained(a.tokenizer)
        bos, eod, vocab = hf.bos_token_id, hf.eos_token_id, len(hf)
    S = blend["seq_length"]
    fails = []

    def check(ok, what):
        print(("ok   " if ok else "FAIL ") + what, flush=True)
        if not ok:
            fails.append(what)

    # 1. the run's BlendedDataset, through Megatron
    blended = MegatronBlend(entry, bos, eod, vocab).blended(a.data_args_path)
    check(blended.unique_description_hash == entry.blend_md5, f"BlendedDataset hash {blended.unique_description_hash}")
    plan_dirs = sorted(p for p in glob.glob(os.path.join(a.out, "plan", a.plan_name, "ck*_*")) if ".tmp" not in os.path.basename(p))
    assert plan_dirs, f"no plans under {a.out}/plan/{a.plan_name}"
    metas = [json.load(open(os.path.join(p, "meta.json"))) for p in plan_dirs]

    # 2. the loader hook with the run's geometry
    g = blend["geometry"]
    args = SimpleNamespace(
        packing_plan_path=[os.path.join(a.out, "plan", a.plan_name)], packing_plan_boundaries="group",
        packing_plan_at_end="exit", global_batch_size=g["global_batch_size"],
        data_sampler_reference_dp_size=g["ref_dp"], data_sampler_reference_micro_batch_size=g["ref_mbs"],
        data_sampler_reference_global_batch_size=g.get("ref_gbs"), train_iters=g["train_iters"],
        phase_transition_iterations=[g["phase_start"]], iteration=min(m["from_ckpt"] for m in metas))
    wrapped = wrap_with_packing_plans(blended, args)
    check(isinstance(wrapped, PackingPlanDataset), "wrap_with_packing_plans wraps the training dataset")
    print(f"     planned from {args.iteration} to {args.packing_plan_end_iteration}", flush=True)
    stray = SimpleNamespace(**{**vars(args), "global_batch_size": g["global_batch_size"] // 2})
    try:
        wrap_with_packing_plans(blended, stray)
        check(False, "a plan for another global batch size is refused")
    except AssertionError:
        check(True, "a plan for another global batch size is refused")

    # 3. per window: slots and tokens conserved
    stream = BlendStream(entry)
    for p, m in zip(plan_dirs, metas):
        z = {k: np.load(os.path.join(p, k + ".npy")) for k in ("slot_bidx", "slot_seq", "seq_ptr", "seg_ds",
                                                             "seg_doc", "seg_off", "seg_len")}
        win = build_plan.load_window(a.out, m["from_ckpt"], m["to_ckpt"])
        solo = win["nseg"] == 1
        seg_slot = np.repeat(np.arange(len(win["nseg"])), win["nseg"])
        k = ~solo[seg_slot]
        check(np.array_equal(np.sort(win["slot_bidx"][~solo]), z["slot_bidx"]),
              f"{os.path.basename(p)}: plan slots == the window's {int((~solo).sum())} non-solo slots")
        check(sorted(z["slot_seq"].tolist()) == list(range(len(z["seq_ptr"]) - 1)),
              f"{os.path.basename(p)}: every plan sequence used once")
        before = owned_intervals(win["seg_ds"][k], win["seg_doc"][k], win["seg_off"][k], win["seg_len"][k])
        after = owned_intervals(*plan_owned(z))
        same = all(np.array_equal(x, y) for x, y in zip(before, after))
        check(same, f"{os.path.basename(p)}: the plan holds exactly the window's "
                    f"{int(win['seg_len'][k].sum())} owned tokens (as often as the window reads them)")
        check(bool((np.diff(z["seq_ptr"]) > 0).all()) and
              all(int(z["seg_len"][z["seq_ptr"][s]:z["seq_ptr"][s + 1]].sum()) == S + 1
                  for s in range(0, len(z["seq_ptr"]) - 1, max(1, (len(z["seq_ptr"]) - 1) // 2000))),
              f"{os.path.basename(p)}: sequences are {S} + 1 tokens")

    # 4. what Megatron serves, per boundary mode
    rng = np.random.default_rng(0)
    planned_all = np.concatenate([np.load(os.path.join(p, "slot_bidx.npy")) for p in plan_dirs])
    for mode in ("doc", "group", "none"):
        ds = PackingPlanDataset(blended, plan_dirs, mode)
        bad = 0
        for idx in rng.choice(planned_all, size=min(a.samples, len(planned_all)), replace=False):
            plan, slot = ds.plan_slot(int(idx))
            s = int(plan.slot_seq[slot])
            lo, hi = int(plan.seq_ptr[s]), int(plan.seq_ptr[s + 1])
            segs = [(int(plan.seg_ds[j]), int(plan.seg_doc[j]), int(plan.seg_off[j]), int(plan.seg_len[j]))
                    for j in range(lo, hi)]
            text = np.concatenate([stream.read(d, c, o, n, bos) for d, c, o, n in segs]).astype(np.int64)
            got = ds[int(idx)]
            ok = np.array_equal(got["tokens"].numpy(), text[:-1]) and np.array_equal(got["labels"].numpy(), text[1:])
            ln = np.array([x[3] for x in segs], np.int64)
            if mode == "doc":
                parts = ln.copy()
            elif mode == "group":
                grp = np.asarray(plan.seg_group[lo:hi])
                parts = np.add.reduceat(ln, np.nonzero(np.r_[True, grp[1:] != grp[:-1]])[0])
            else:
                parts = np.array([ln.sum()])
            parts[-1] -= 1
            parts = parts[parts > 0]
            cu = got["cu_seqlens"].numpy()
            cu = cu[: int(np.argmax(cu == S)) + 1]
            ok &= cu.tolist() == np.r_[0, np.cumsum(parts)].tolist()
            pos = got["position_ids"].numpy()
            ok &= all(pos[x:y].tolist() == list(range(y - x)) for x, y in zip(cu[:-1], cu[1:]))
            bad += not ok
        check(bad == 0, f"mode {mode}: {a.samples} planned samples == the plan's segments (tokens, labels, "
                        f"cu_seqlens, positions)")
        planned_set = set(planned_all.tolist())
        others = [int(i) for i in rng.integers(0, len(blended), 4 * a.samples) if int(i) not in planned_set][: a.samples]
        same = all(all(np.array_equal(np.asarray(ds[i][key]), np.asarray(ref[key])) for key in ref)
                   for i, ref in ((i, blended[i]) for i in others))
        check(same, f"mode {mode}: {len(others)} unplanned samples == the BlendedDataset's")

    print(f"check_plan {a.plan_name}: {len(fails)} failures over {len(plan_dirs)} plans", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
