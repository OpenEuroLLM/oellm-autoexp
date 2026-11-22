"""Parity of an HF export with Megatron's own forward, on the run's validation data. One GPU, launched
with ``torchrun --nproc-per-node 1`` (Megatron needs torch.distributed):

    LAUNCHER="torchrun --standalone --nproc-per-node 1" GPUS=1 run_in_container.sh megatron_parity \\
        --export-dir <run>/hf/iter_<N> [--samples 64] [--layer-samples 2] [--out <json>]

1. **Megatron as trained**: the run's own command line (``job.sbatch``, minus wandb / tensorboard / save),
   ``parse_and_validate_args`` + ``initialize_megatron``, ``pretrain_hybrid``'s model provider, and
   ``load_checkpoint`` of the export's iteration (no optimizer, no RNG).
2. **The training's validation samples**: ``pretrain_hybrid.train_valid_test_datasets_provider`` with the
   training's sample counts (so the same cached indices, the same samples). These runs use no
   inter-document masking: a sample is one 4096-token sequence, documents separated by EOD, every
   label counted -- exactly what the HF model computes on it.
3. **Per layer** (``--layer-samples``): every layer's output in Megatron bf16 and HF bf16, each against
   an HF fp32 reference (TF32 off). Megatron and the HF model compute the same function if Megatron is
   as close to the fp32 reference as the HF model's own bf16 rounding is.
4. **Logits** (``--samples``): KL(Megatron || HF), top-1 agreement, the NLL of both on the labels, and
   Megatron's logged validation loss at that iteration (other samples; for scale).
5. **Engram ablations** (HF): the NLL with the Engram outputs zeroed, and with the hash multipliers of
   a builder that drops the per-layer seed offset; both must be worse than the export.
6. **Tokenizer**: complete documents of the samples decoded and re-encoded by the export's tokenizer,
   with and without ``fix_mistral_regex``: which reproduces the training data's token ids.
7. **Batched generation** (HF): greedy decoding of left-padded prompts in one batch against each
   prompt alone.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import shlex
import sys
import time
from functools import partial
from pathlib import Path

import torch

RESULTS: list[tuple[str, bool, str]] = []
_DROP_WITH_VALUE = {"--wandb-project", "--wandb-entity", "--wandb-exp-name", "--wandb-save-dir",
                    "--tensorboard-dir", "--save", "--save-interval"}
_DROP_FLAGS = {"--log-timers-to-tensorboard"}
PROMPTS = ("The capital of France is", "Die Hauptstadt von Deutschland ist eine Stadt, die",
           "def fibonacci(n):", "La Unión Europea es una unión política y económica de")


def report(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def training_argv(run_dir: Path, tokenizer: str) -> list[str]:
    """The arguments ``pretrain_hybrid.py`` was launched with (``job.sbatch``), minus logging / saving;
    the tokenizer as a local directory (offline, the hub id resolves only through a config.json)."""
    text = (run_dir / "job.sbatch").read_text()
    cmd = text.split("pretrain_hybrid.py", 1)[1].split("'", 1)[0].replace("\\\n", " ")
    argv, out, skip = shlex.split(cmd), [], False
    for tok in argv:
        if skip and not tok.startswith("--"):
            continue
        skip = False
        if tok in _DROP_WITH_VALUE:
            skip = True
        elif tok not in _DROP_FLAGS:
            out.append(tok)
    i = out.index("--tokenizer-model")
    out[i + 1] = tokenizer
    return out + ["--no-load-optim", "--no-load-rng"]


def build_megatron(run_dir: Path, iteration: int, tokenizer: str):
    from hybrid_builders import hybrid_builder
    from megatron.core.enums import ModelType
    from megatron.training import get_args
    from megatron.training.arguments import parse_and_validate_args
    from megatron.training.checkpointing import load_checkpoint
    from megatron.training.initialize import initialize_megatron
    from megatron.training.training import get_model
    from model_provider import model_provider

    sys.argv = ["pretrain_hybrid.py"] + training_argv(run_dir, tokenizer)
    parse_and_validate_args(args_defaults={"tokenizer_type": "GPT2BPETokenizer"})
    initialize_megatron()
    args = get_args()
    model = get_model(partial(model_provider, hybrid_builder), ModelType.encoder_or_decoder, wrap_with_ddp=False)
    args.ckpt_step = iteration
    loaded = load_checkpoint(model, None, None, strict=True)
    loaded_iter = loaded[0] if isinstance(loaded, tuple) else loaded
    report("Megatron: model built from the run's command line, checkpoint loaded (strict)",
           int(loaded_iter) == iteration, f"iteration {loaded_iter}")
    model = model[0].eval()
    return model, model.module, args


def validation_samples(n: int, start: int):
    import pretrain_hybrid
    from megatron.training.training import get_train_valid_test_num_samples

    _, valid, _ = pretrain_hybrid.train_valid_test_datasets_provider(get_train_valid_test_num_samples())
    return [valid[i] for i in range(start, start + n)], len(valid)


def logged_validation_loss(run_dir: Path, iteration: int) -> float | None:
    # the last evaluation of a run reads "... at iteration N on validation set | ..."
    pat = re.compile(rf"validation loss at iteration {iteration}(?: on [^|]*)? \| lm loss value: ([0-9.E+-]+)")
    for log in sorted((run_dir / "logs").glob("stdout-*.log")):
        for line in log.read_text(errors="replace").splitlines():
            if m := pat.search(line):
                return float(m.group(1))
    return None


@contextlib.contextmanager
def capture(layers, final_norm, megatron: bool):
    """Outputs of every decoder layer and of the final norm, as fp32 [b, s, h]."""
    out: list = [None] * (len(layers) + 1)

    def hook(i):
        def fn(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            out[i] = (h.transpose(0, 1) if megatron else h).float()
        return fn

    handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(layers)]
    handles.append(final_norm.register_forward_hook(hook(len(layers))))
    try:
        yield out
    finally:
        for h in handles:
            h.remove()


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


def nll_and_kl(meg_logits, hf_logits, labels, loss_mask):
    """Token-summed NLL of both, summed KL(meg || hf), top-1 agreement count, tokens (chunked)."""
    out = torch.zeros(5, dtype=torch.float64)
    for s in range(0, labels.shape[0], 512):
        lm = torch.log_softmax(meg_logits[s:s + 512].float(), -1)
        lh = torch.log_softmax(hf_logits[s:s + 512].float(), -1)
        y, w = labels[s:s + 512], loss_mask[s:s + 512].float()
        out[0] += (-(lm.gather(-1, y[:, None])[:, 0]) * w).sum().item()
        out[1] += (-(lh.gather(-1, y[:, None])[:, 0]) * w).sum().item()
        out[2] += ((lm.exp() * (lm - lh)).sum(-1) * w).sum().item()
        out[3] += ((lm.argmax(-1) == lh.argmax(-1)).float() * w).sum().item()
        out[4] += w.sum().item()
    return out


def hf_nll(model, samples, device) -> float:
    total = count = 0.0
    with torch.no_grad():
        for smp in samples:
            ids = smp["tokens"][None].to(device)
            logits = model(ids, use_cache=False).logits[0]
            lab, w = smp["labels"].to(device), smp["loss_mask"].to(device).float()
            for s in range(0, lab.shape[0], 512):
                lp = torch.log_softmax(logits[s:s + 512].float(), -1)
                total += (-(lp.gather(-1, lab[s:s + 512, None])[:, 0]) * w[s:s + 512]).sum().item()
            count += w.sum().item()
    return total / count


# --------------------------------------------------------------------------- checks


def check_layers(meg, meg_inner, hf16, hf32, samples, device, block_types) -> dict:
    layers = {"megatron": [], "hf_bf16": [], "hf_fp32": []}
    rows = []
    for smp in samples:
        ids = smp["tokens"][None].to(device)
        pos = smp["position_ids"][None].to(device)
        with torch.no_grad():
            with capture(meg_inner.decoder.layers, meg_inner.decoder.final_norm, True) as m_out:
                meg(ids, pos, None)
            with capture(hf16.model.layers, hf16.model.norm, False) as h_out:
                hf16(ids, use_cache=False)
            tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            try:
                with capture(hf32.model.layers, hf32.model.norm, False) as r_out:
                    hf32(ids, use_cache=False)
            finally:
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
        rows.append([(rel(m, r), rel(h, r), rel(m, h)) for m, h, r in zip(m_out, h_out, r_out)])
    print("      layer  type        Megatron bf16   HF bf16      Megatron vs HF   (relative L2 error"
          " against HF fp32, mean over samples)")
    worst = 0.0
    summary = []
    for i in range(len(rows[0])):
        m, h, mh = (sum(r[i][k] for r in rows) / len(rows) for k in range(3))
        kind = block_types[i] if i < len(block_types) else "final_norm"
        summary.append({"layer": i, "type": kind, "megatron_vs_fp32": m, "hf_vs_fp32": h, "megatron_vs_hf": mh})
        worst = max(worst, m / max(h, 1e-12))
        print(f"      {i:5d}  {kind:10s}  {m:.2e}        {h:.2e}     {mh:.2e}")
    report(f"per-layer outputs ({len(rows[0])} incl. final norm, {len(samples)} x {samples[0]['tokens'].numel()} tokens):"
           " Megatron as close to the fp32 reference as HF bf16", worst < 3.0,
           f"largest ratio Megatron error / HF bf16 error {worst:.2f}")
    return {"layers": summary, "worst_ratio": worst}


def check_logits(meg, hf16, samples, device, logged: float | None) -> dict:
    acc = torch.zeros(5, dtype=torch.float64)
    started = time.perf_counter()
    with torch.no_grad():
        for n, smp in enumerate(samples, 1):
            ids, pos = smp["tokens"][None].to(device), smp["position_ids"][None].to(device)
            meg_logits = meg(ids, pos, None)[0]
            hf_logits = hf16(ids, use_cache=False).logits[0]
            acc += nll_and_kl(meg_logits, hf_logits, smp["labels"].to(device), smp["loss_mask"].to(device))
            if n % 64 == 0 or n == len(samples):
                nll_m, nll_h, kl, agree = (acc[:4] / acc[4]).tolist()
                print(f"    {n}/{len(samples)} sequences, {int(acc[4]):,} tokens: "
                      f"NLL Megatron {nll_m:.4f}, HF {nll_h:.4f}, KL {kl:.2e}, "
                      f"top-1 {agree * 100:.2f}%, {(time.perf_counter() - started) / n:.2f}s/seq",
                      flush=True)
    nll_m, nll_h, kl, agree = (acc[:4] / acc[4]).tolist()
    tokens = int(acc[4])
    report(f"logits on {len(samples)} validation samples ({tokens:,} tokens): KL(Megatron || HF), top-1 agreement",
           kl < 2e-3 and agree >= 0.98, f"mean KL {kl:.2e} nats, top-1 agreement {agree * 100:.2f}%")
    report("NLL on the same tokens: Megatron vs HF", abs(nll_m - nll_h) < 1e-2,
           f"Megatron {nll_m:.4f}, HF {nll_h:.4f} (|diff| {abs(nll_m - nll_h):.4f}); "
           f"Megatron's logged validation loss at this iteration {logged} (2560 other samples)")
    return {"tokens": tokens, "nll_megatron": nll_m, "nll_hf": nll_h, "kl": kl, "top1_agreement": agree,
            "logged_validation_loss": logged}


def check_engram_ablations(hf16, samples, device, m: dict) -> dict:
    from ..export_ambitious import engram_hash_multipliers

    # by block type, not isinstance: trust_remote_code loads the export directory's copy of the classes
    mods = [(i, layer.engram) for i, layer in enumerate(hf16.model.layers) if layer.block_type == "engram"]
    if not mods:
        return {}
    base = hf_nll(hf16, samples, device)
    cls = type(mods[0][1])
    orig_forward = cls.forward

    def zero_forward(self, h, *a, **k):
        y, state = orig_forward(self, h, *a, **k)
        return torch.zeros_like(y), state

    cls.forward = zero_forward
    try:
        zeroed = hf_nll(hf16, samples, device)
    finally:
        cls.forward = orig_forward
    saved = [mod.hash_mult for _, mod in mods]
    for i, mod in mods:
        mod.hash_mult = engram_hash_multipliers({**m, "engram_hash_per_layer": False}, i + 1)
    try:
        wrong = hf_nll(hf16, samples, device)
    finally:
        for (_, mod), mult in zip(mods, saved):
            mod.hash_mult = mult
    report(f"Engram ablations on {len(samples)} samples: zeroed output and per-layer-offset-less hashes are worse",
           zeroed > base + 1e-3 and wrong > base + 1e-3,
           f"NLL export {base:.4f}, Engram zeroed {zeroed:.4f} (+{zeroed - base:.4f}), "
           f"hashes without the per-layer offset {wrong:.4f} (+{wrong - base:.4f})")
    return {"nll": base, "nll_engram_zeroed": zeroed, "nll_wrong_hash": wrong}


def check_tokenizer(export_dir: Path, samples, eod: int) -> dict:
    from transformers import AutoTokenizer

    docs = []
    for smp in samples:
        ids = smp["tokens"].tolist()
        cuts = [i for i, t in enumerate(ids) if t == eod]
        docs += [ids[a + 1:b] for a, b in zip(cuts, cuts[1:]) if b - a > 2]  # complete documents only
    out = {}
    for fix in (False, True):
        try:
            tok = AutoTokenizer.from_pretrained(export_dir, trust_remote_code=True, fix_mistral_regex=fix)
        except TypeError:
            continue
        same = sum(tok(tok.decode(d), add_special_tokens=False).input_ids == d for d in docs)
        out[f"fix_mistral_regex={fix}"] = same / max(len(docs), 1)
    best = max(out, key=out.get) if out else None
    report(f"tokenizer: {len(docs)} complete documents decoded and re-encoded == the training data's ids",
           bool(out) and out.get("fix_mistral_regex=False", 0) >= 0.99,
           ", ".join(f"{k}: {v * 100:.1f}%" for k, v in out.items()) + f" (best: {best})")
    return {"documents": len(docs), **out}


def check_batched_generation(export_dir: Path, hf16, device) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(export_dir, trust_remote_code=True)
    tok.padding_side = "left"
    pad = hf16.config.pad_token_id
    kw = dict(max_new_tokens=20, do_sample=False, pad_token_id=pad, output_logits=True, return_dict_in_generate=True)
    batch = tok(list(PROMPTS), return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = hf16.generate(**batch, **kw)
    kls, flips, worst_gap, same = [], 0, 0.0, 0
    for b, prompt in enumerate(PROMPTS):
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            single = hf16.generate(ids, **kw)
        ls = torch.log_softmax(torch.cat(single.logits).float(), -1)  # [new, vocab]
        lb = torch.log_softmax(torch.stack([step[b] for step in out.logits]).float(), -1)
        tb, ts = out.sequences[b, batch.input_ids.shape[1]:], single.sequences[0, ids.shape[1]:]
        same += int(torch.equal(tb, ts))
        diff = torch.nonzero(tb != ts).flatten()
        # compared up to the first divergence: after it the two runs condition on different tokens
        n = diff[0].item() + 1 if diff.numel() else ls.shape[0]
        kls.append((ls[:n].exp() * (ls[:n] - lb[:n])).sum(-1).mean().item())
        if diff.numel():  # the first divergence must be a near tie of the single-prompt run
            t = diff[0].item()
            flips += 1
            worst_gap = max(worst_gap, (ls[t].max() - ls[t, tb[t]]).item())
        print(f"      {prompt!r} -> {tok.decode(tb)!r}")
    report(f"batched left-padded generate ({len(PROMPTS)} prompts, lengths "
           f"{[len(tok(p).input_ids) for p in PROMPTS]}) vs each prompt alone",
           max(kls) < 5e-3 and worst_gap < 0.5,
           f"{same}/{len(PROMPTS)} identical; per-prompt mean KL (up to a divergence) at most {max(kls):.1e}; "
           f"{flips} diverging, largest log-prob gap at the divergence {worst_gap:.3f}")
    return {"identical": same, "kl": kls, "divergence_gap": worst_gap}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export-dir", type=Path, required=True)
    p.add_argument("--samples", type=int, default=64, help="validation samples for the logits / NLL comparison")
    p.add_argument("--layer-samples", type=int, default=2, help="samples for the per-layer comparison")
    p.add_argument("--ablation-samples", type=int, default=32)
    p.add_argument("--start", type=int, default=0, help="first validation sample")
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    from transformers import AutoModelForCausalLM

    from ..export_q35 import megatron_args
    from .export_hf import run_paths
    from .hf_config import tokenizer_dir

    source = json.loads((a.export_dir / "megatron_source.json").read_text())
    run_dir, iteration = Path(source["run_dir"]), int(source["iteration"])
    config_yaml, _ = run_paths(run_dir, iteration)
    m = megatron_args(config_yaml)
    print(f"== {a.export_dir.name} of {run_dir.name}, iteration {iteration}", flush=True)

    meg, meg_inner, args = build_megatron(run_dir, iteration, str(tokenizer_dir(m["tokenizer_model"])))
    device = torch.device("cuda", torch.cuda.current_device())
    samples, n_valid = validation_samples(max(a.samples, a.layer_samples, a.ablation_samples), a.start)
    print(f"  validation samples {a.start} .. {a.start + len(samples) - 1} of {n_valid:,} "
          f"(seq {samples[0]['tokens'].numel()}, all labels counted: "
          f"{bool(all(s['loss_mask'].all() for s in samples))})", flush=True)

    hf16 = AutoModelForCausalLM.from_pretrained(a.export_dir, trust_remote_code=True, dtype=torch.bfloat16).to(device).eval()
    hf32 = AutoModelForCausalLM.from_pretrained(a.export_dir, trust_remote_code=True, dtype=torch.float32).to(device).eval()
    results = {"export": str(a.export_dir), "run": run_dir.name, "iteration": iteration}
    print("== per layer", flush=True)
    results["layers"] = check_layers(meg, meg_inner, hf16, hf32, samples[:a.layer_samples], device, hf16.config.block_types)
    del hf32
    torch.cuda.empty_cache()
    print("== logits and NLL", flush=True)
    results["logits"] = check_logits(meg, hf16, samples[:a.samples], device, logged_validation_loss(run_dir, iteration))
    print("== Engram ablations", flush=True)
    results["engram"] = check_engram_ablations(hf16, samples[:a.ablation_samples], device, m)
    print("== tokenizer", flush=True)
    from megatron.training import get_tokenizer

    results["tokenizer"] = check_tokenizer(a.export_dir, samples[:a.samples], get_tokenizer().eod)
    print("== batched generation", flush=True)
    results["generation"] = check_batched_generation(a.export_dir, hf16, device)

    failed = [n for n, ok, _ in RESULTS if not ok]
    results["checks"] = [{"name": n, "ok": ok, "detail": d} for n, ok, d in RESULTS]
    if a.out:
        a.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed", flush=True)
    torch.distributed.destroy_process_group()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
