"""Prefill time per layer kind on the reduced v4_2ph model, extrapolated to the full 98 layers.

Each decoder layer is timed with CUDA events (forward pre / post hooks), its mixer (and the
input norm) separately, so a layer's mHC read / write is ``layer - mixer - norm``. The full-model
estimate is embed + sum over kinds (48 moe, 42 ckda, 6 attention, 2 engram) + norm + lm_head.

    CC=/usr/bin/gcc PYTHONPATH=flash-linear-attention python \
        experiment/013_hf_prefill_speed/scripts/bench_layers.py --shapes 1x2048 8x512 --out r.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiny_model import FULL_COUNTS, build  # noqa: E402

LAYERS = ["attention", "moe", "engram", "ckda", "moe", "ckda", "moe", "ckda", "moe"]


class Timer:
    def __init__(self):
        self.events = defaultdict(list)  # key -> [(start, end)]
        self.open = {}

    def pre(self, key):
        def hook(mod, args, kwargs=None):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self.open[key] = ev

        return hook

    def post(self, key):
        def hook(mod, args, out, kwargs=None):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self.events[key].append((self.open.pop(key), ev))

        return hook

    def collect(self):
        torch.cuda.synchronize()
        out = {k: [a.elapsed_time(b) for a, b in v] for k, v in self.events.items()}
        self.events.clear()
        return out


def attach(model, timer):
    hooks = []
    for i, layer in enumerate(model.model.layers):
        for key, mod in ((f"layer{i}", layer), (f"mixer{i}", layer.mixer), (f"norm{i}", getattr(layer, "input_layernorm", None))):
            if mod is None:
                continue
            hooks.append(mod.register_forward_pre_hook(timer.pre(key)))
            hooks.append(mod.register_forward_hook(timer.post(key)))
    for key, mod in (("embed", model.model.embed_tokens), ("lm_head", model.lm_head), ("final_norm", model.model.norm)):
        hooks.append(mod.register_forward_pre_hook(timer.pre(key)))
        hooks.append(mod.register_forward_hook(timer.post(key)))
    return hooks


def run(model, ids, mask, use_cache):
    return model(input_ids=ids, attention_mask=mask, use_cache=use_cache).logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", nargs="+", default=["1x2048", "8x512", "32x256", "4x4096"])
    p.add_argument("--layers", nargs="+", default=LAYERS)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--padded", action="store_true", help="right-pad: each row a random length >= s/2")
    p.add_argument("--use-cache", action="store_true")
    p.add_argument("--profile", default=None, help="write a torch profiler table for the first shape")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    t0 = time.time()
    model, cfg = build(a.layers)
    print(f"built in {time.time() - t0:.0f} s; env:", {k: v for k, v in os.environ.items() if k.startswith("OELLM")}, flush=True)
    kinds = cfg.block_types
    timer = Timer()
    results = []
    gen = torch.Generator(device="cpu").manual_seed(1)
    for shape in a.shapes:
        b, s = map(int, shape.split("x"))
        ids = torch.randint(3, cfg.vocab_size, (b, s), generator=gen).cuda()
        mask = None
        if a.padded:
            lens = torch.randint(s // 2, s + 1, (b,), generator=gen)
            lens[0] = s
            mask = (torch.arange(s)[None] < lens[:, None]).long().cuda()
        with torch.no_grad():
            for _ in range(2):  # warm-up (Triton autotune, cuBLAS)
                run(model, ids, mask, a.use_cache)
            torch.cuda.synchronize()
            hooks = attach(model, timer)
            walls = []
            for _ in range(a.reps):
                torch.cuda.synchronize()
                w0 = time.perf_counter()
                run(model, ids, mask, a.use_cache)
                torch.cuda.synchronize()
                walls.append(time.perf_counter() - w0)
            for h in hooks:
                h.remove()
            times = timer.collect()
            if a.profile and shape == a.shapes[0]:
                from torch.profiler import ProfilerActivity, profile

                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    run(model, ids, mask, a.use_cache)
                    torch.cuda.synchronize()
                Path(a.profile).write_text(prof.key_averages().table(sort_by="cuda_time_total", row_limit=60))
                prof.export_chrome_trace(str(Path(a.profile).with_suffix(".trace.json")))
        med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
        per_kind = defaultdict(lambda: defaultdict(list))
        for i, kind in enumerate(kinds):
            lay = med(times[f"layer{i}"])
            mix = med(times[f"mixer{i}"])
            norm = med(times[f"norm{i}"]) if f"norm{i}" in times else 0.0
            per_kind[kind]["layer"].append(lay)
            per_kind[kind]["mixer"].append(mix)
            per_kind[kind]["mhc"].append(lay - mix - norm)
            per_kind[kind]["norm"].append(norm)
        mean = lambda v: sum(v) / len(v)  # noqa: E731
        summary = {k: {part: mean(v) for part, v in d.items()} for k, d in per_kind.items()}
        other = {k: med(times[k]) for k in ("embed", "final_norm", "lm_head")}
        full = sum(other.values()) + sum(FULL_COUNTS[k] * summary[k]["layer"] for k in summary)
        tokens = b * s if mask is None else int(mask.sum())
        row = {
            "shape": shape, "tokens": tokens, "wall_ms_reduced": 1e3 * med(walls),
            "per_kind_ms": summary, "other_ms": other, "full_model_ms_est": full,
            "full_model_tok_s_est": tokens / (full / 1e3),
            "full_model_breakdown_ms": {
                **{f"{k}.mixer": FULL_COUNTS[k] * (summary[k]["mixer"] + summary[k]["norm"]) for k in summary},
                "mhc(all layers)": sum(FULL_COUNTS[k] * summary[k]["mhc"] for k in summary),
                **other,
            },
        }
        results.append(row)
        print(json.dumps({k: row[k] for k in ("shape", "tokens", "full_model_ms_est", "full_model_tok_s_est")}), flush=True)
        print("   ", {k: round(v, 1) for k, v in row["full_model_breakdown_ms"].items()}, flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps({"env": {k: v for k, v in os.environ.items() if k.startswith("OELLM")},
                                           "gpu": torch.cuda.get_device_name(), "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
