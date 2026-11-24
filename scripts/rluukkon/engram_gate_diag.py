"""Offline diagnosis of the Engram gate on an HF export (one node; the export's own model code).

Every Engram layer's forward is replaced by an instrumented copy of OellmEngramV2.forward (no cache): it records
per token the gate's raw logit x = <RMSNorm(h), RMSNorm(k)> / sqrt(d), the logit after the signed sqrt, the gate,
|RMSNorm(h)|, |RMSNorm(k)|, their cosine, and the RMS of the gated value (alpha v), the conv term and the input
read; and it takes knobs: zero the layer, scale the gate logit, fix the gate. Per document the runs are

  base            the model as it is (with the records)
  zero_L<i>       Engram layer i zeroed               -> per-token utility dNLL_t = NLL_t(zero) - NLL_t(base)
  scale<k>        gate = sigmoid(k * logit), all layers
  const           gate = its document mean, all layers
  open            gate = 1, all layers
  fixed_ls<s>     the scale-invariant query, gate = sigmoid(signed_sqrt(x * s / |h~|)), with --fixed-query

Sets: the run's fixed validation sets (BOS + document, as trained), FLORES passages (24 languages, via
Belebele), GSM8K question + reference answer. Output: <out>/<set>.npz (per-token arrays) + <out>/runs.json.

The fixed-query pass is the offline test of the proposed fix: the eps-limited query norm is replaced by a
scale-invariant one (divide by |h~|), and a logit_scale sets the size before the signed square root. The
swept layer takes each --logit-scales value; the other Engram layers keep mean(|h~|), their
function-preserving init. Read the output with scripts/rluukkon/engram_gate_rescore.py.

    python engram_gate_diag.py --model-dir <export> --out <dir> [--docs 16] [--flores 24] [--gsm8k 64]
    python engram_gate_diag.py --model-dir <export> --out <dir> --fixed-query --logit-scales 0.025,0.1,0.3,1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

VALID = Path("/gpfs/projects/ehpc533/users/korbip/data/v8ml-valid")
DOMAINS = ("web_en", "multilingual", "code", "math", "reasoning")
LANGS = ["eng_Latn", "spa_Latn", "fra_Latn", "deu_Latn", "por_Latn", "ita_Latn", "pol_Latn", "nld_Latn",
         "ces_Latn", "swe_Latn", "dan_Latn", "fin_Latn", "hun_Latn", "ron_Latn", "bul_Cyrl", "ell_Grek",
         "slk_Latn", "hrv_Latn", "lit_Latn", "lvs_Latn", "est_Latn", "slv_Latn", "mlt_Latn", "isl_Latn"]


class Ctl:
    """What the instrumented Engram layers do in the current pass."""

    def __init__(self):
        self.zero = set()
        self.scale = 1.0
        self.const = None   # per-layer constant gate
        self.open = False
        self.fixed = None   # dict(layer, scale, init) of the scale-invariant query pass
        self.record = None  # dict layer -> list of per-token arrays


def instrument(model, ctl):
    layers = []
    for i, layer in enumerate(model.model.layers):
        mod = getattr(layer, "mixer", None)
        if mod is None or not hasattr(mod, "key_proj") or not hasattr(mod, "value_proj"):
            continue
        M = sys.modules[type(mod).__module__]
        layers.append(i)

        def forward(hidden_states, input_ids, token_mask=None, state=None, _m=mod, _i=i, _M=M):
            assert state is None and token_mask is None, "diagnosis runs full forwards without cache / padding"
            b, s, h = hidden_states.shape
            ids = input_ids if _m.token_map is None else _m.token_map[input_ids]
            shifted = ids.long() + 1
            p = max(_m.orders) - 1
            shifted = torch.cat([shifted.new_zeros(b, p), shifted], dim=1)
            new_id_state = shifted[:, shifted.shape[1] - p:]
            fast = getattr(_M, "_fast", lambda t: False)(hidden_states) and hasattr(_m, "_fast_lookup")
            e = _m._fast_lookup(shifted, s, hidden_states.dtype) if fast else None
            if e is None:
                e = _m._lookup(_m._rows(shifted, s), hidden_states.dtype).to(hidden_states.device)
                if getattr(_m, "sign_hash", False):
                    e = e * _m.sign_table[_m._sign_rows(shifted, s).to(_m.sign_table.device)].to(e.device, e.dtype)
            e = e.reshape(b, s, -1)
            k, v = _m.key_proj(e), _m.value_proj(e)
            hn, kn = _m.hidden_norm(hidden_states).float(), _m.key_norm(k).float()
            x = (hn * kn).sum(-1, keepdim=True) / h ** 0.5
            if ctl.fixed is not None:
                # scale-invariant query: x * logit_scale / |h~| before the signed square root
                ls = ctl.fixed["scale"] if _i == ctl.fixed["layer"] else ctl.fixed["init"][_i]
                hn_rms = hn.norm(dim=-1, keepdim=True) / h ** 0.5
                xf = x * ls / hn_rms.clamp_min(1e-12)
                g = xf.abs().clamp_min(1e-6).sqrt() * xf.sign()
                alpha = torch.sigmoid(g)
            else:
                g = x.abs().clamp_min(1e-6).sqrt() * x.sign() if getattr(_m, "signed_sqrt", False) else x
                alpha = torch.sigmoid(ctl.scale * g)
            if ctl.const is not None:
                alpha = torch.full_like(alpha, ctl.const[_i])
            if ctl.open:
                alpha = torch.ones_like(alpha)
            vv = alpha.to(v.dtype) * v
            conv_norm = getattr(_m, "conv_norm", None)
            conv_in = vv if conv_norm is None else conv_norm(vv)
            pad = (_m.kernel_size - 1) * _m.dilation
            padded = torch.cat([conv_in.transpose(1, 2).new_zeros(b, h, pad), conv_in.transpose(1, 2)], dim=-1)
            conv = F.conv1d(padded, _m.conv1d.weight, groups=h, dilation=_m.dilation)
            new_conv_state = padded[..., padded.shape[-1] - pad:]
            cterm = F.silu(conv.transpose(1, 2))
            y = cterm + vv
            if ctl.record is not None:
                rms = lambda t: t.float().pow(2).mean(-1).sqrt()[0]  # noqa: E731
                cos = (hn * kn).sum(-1) / (hn.norm(dim=-1) * kn.norm(dim=-1)).clamp_min(1e-12)
                ctl.record.setdefault(_i, []).append({
                    "x": x[0, :, 0], "g": g[0, :, 0], "alpha": alpha[0, :, 0],
                    "hn": hn.norm(dim=-1)[0] / h ** 0.5, "kn": kn.norm(dim=-1)[0] / h ** 0.5, "cos": cos[0],
                    "rms_gated": rms(vv), "rms_conv": rms(cterm), "rms_y": rms(y), "rms_read": rms(hidden_states)})
            if _i in ctl.zero:
                y = torch.zeros_like(y)
            return y, (new_id_state, new_conv_state)

        mod.forward = forward

        def pre(_layer, args, kwargs, _i=i):
            """The mHC streams entering the Engram layer: per-token RMS of each stream (averaged over the
            streams) and of their plain mean, next to the read's RMS (rms_read)."""
            if ctl.record is None:
                return
            st = (args[0] if args else kwargs["streams"])
            if not torch.is_tensor(st) or st.dim() != 4:
                return
            st = st.float()
            ctl.record.setdefault(("streams", _i), []).append({
                "rms_stream": st.pow(2).mean(-1).sqrt()[0].mean(-1),
                "rms_stream_mean": st.mean(-2).pow(2).mean(-1).sqrt()[0]})

        layer.register_forward_pre_hook(pre, with_kwargs=True)
    return layers


def token_nll(model, ids, chunk=1024):
    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, :-1]
    head = model.lm_head
    dev = head.weight.device
    hidden, target = hidden.to(dev), ids[0, 1:].to(dev)
    out = [F.cross_entropy(head(hidden[i:i + chunk]).float(), target[i:i + chunk], reduction="none")
           for i in range(0, hidden.shape[0], chunk)]
    return torch.cat(out).float().cpu().numpy()


def valid_docs(name, n, seq_len, min_len, seed):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from oellm_autoexp.hf_export.score_on_flag import read_index

    prefix = str(VALID / f"{name}_text_document")
    dtype, lens, ptrs = read_index(prefix)
    data = np.memmap(prefix + ".bin", dtype=dtype, mode="r")
    rng = np.random.default_rng(seed)
    idx = [j for j in rng.permutation(len(lens)) if lens[j] >= min_len][:n]
    return [np.asarray(data[ptrs[j] // dtype.itemsize: ptrs[j] // dtype.itemsize + min(lens[j], seq_len - 1)],
                       dtype=np.int64) for j in idx]


def flores_docs(tok, lang, n):
    """FLORES passages of ``lang`` (via the cached Belebele test split); None if it is not cached."""
    from datasets import load_dataset

    try:
        rows = load_dataset("facebook/belebele", lang, split="test")
    except Exception as exc:  # offline cache without this language
        print(f"flores_{lang}: skipped ({type(exc).__name__})", flush=True)
        return None
    seen = []
    for r in rows:
        if r["flores_passage"] not in seen:
            seen.append(r["flores_passage"])
        if len(seen) == n:
            break
    return [np.asarray(tok(p, add_special_tokens=False)["input_ids"], dtype=np.int64) for p in seen]


def gsm8k_docs(tok, n):
    from datasets import load_dataset

    rows = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    return [np.asarray(tok(f"Question: {r['question']}\nAnswer: {r['answer']}", add_special_tokens=False)["input_ids"],
                       dtype=np.int64) for r in rows]


def run_set(model, ctl, layers, docs, bos, dev, fixed_scales=(), sweep_layer=None):
    """Per-token arrays of one set: base records, NLLs of every pass."""
    keep = {f"{k}_L{i}": [] for i in layers for k in ("x", "g", "alpha", "hn", "kn", "cos", "rms_gated", "rms_conv",
                                                       "rms_y", "rms_read", "rms_stream", "rms_stream_mean")}
    nll = {k: [] for k in ["base"] + [f"zero_L{i}" for i in layers] + ["scale0.5", "scale2", "scale4", "scale8",
                                                                       "const", "open"]
           + [f"fixed_ls{ls:g}" for ls in fixed_scales]}
    for d in docs:
        ids = torch.tensor(np.concatenate([[bos], d]), device=dev).unsqueeze(0)
        with torch.no_grad():
            ctl.record = {}
            nll["base"].append(token_nll(model, ids))
            rec = {i: {**ctl.record[i][0], **ctl.record.get(("streams", i), [{}])[0]} for i in layers}
            ctl.record = None
            for i in layers:
                for k, t in rec[i].items():
                    keep[f"{k}_L{i}"].append(t.float().cpu().numpy())
            keep = {k: v for k, v in keep.items() if v}
            for i in layers:
                ctl.zero = {i}
                nll[f"zero_L{i}"].append(token_nll(model, ids))
            ctl.zero = set()
            for k in (0.5, 2, 4, 8):
                ctl.scale = k
                nll[f"scale{k:g}"].append(token_nll(model, ids))
            ctl.scale = 1.0
            ctl.const = {i: float(rec[i]["alpha"].float().mean()) for i in layers}
            nll["const"].append(token_nll(model, ids))
            ctl.const = None
            ctl.open = True
            nll["open"].append(token_nll(model, ids))
            ctl.open = False
            if fixed_scales:
                init = {i: float(rec[i]["hn"].float().mean()) for i in layers}
                for ls in fixed_scales:
                    ctl.fixed = {"layer": sweep_layer if sweep_layer is not None else layers[0],
                                 "scale": float(ls), "init": init}
                    nll[f"fixed_ls{ls:g}"].append(token_nll(model, ids))
                ctl.fixed = None
    lengths = np.array([len(d) for d in docs])
    out = {k: np.concatenate(v).astype(np.float32) for k, v in {**keep, **nll}.items()}
    out["doc_lengths"] = lengths
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--docs", type=int, default=16, help="documents per validation domain")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--flores", type=int, default=24, help="passages per language (0: off)")
    ap.add_argument("--gsm8k", type=int, default=64)
    ap.add_argument("--gpu-mem-gib", type=float, default=54)
    ap.add_argument("--fixed-query", action="store_true",
                    help="also run the scale-invariant query (x * logit_scale / |h~|) at each --logit-scales value")
    ap.add_argument("--logit-scales", default="0.025,0.1,0.3,1.0",
                    help="comma-separated logit_scale values of the fixed query (with --fixed-query)")
    ap.add_argument("--sweep-layer", type=int, default=None,
                    help="layer whose logit_scale is swept; the others keep mean(|h~|) (default: the first)")
    a = ap.parse_args()
    fixed_scales = tuple(float(s) for s in a.logit_scales.replace(" ", "").split(",") if s) if a.fixed_query else ()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from transformers import AutoConfig, AutoTokenizer

    from oellm_autoexp.hf_export.score_on_flag import layer_device_map, load_engram_tables_cpu

    t0 = time.time()
    cfg = AutoConfig.from_pretrained(a.model_dir, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    bos = tok.bos_token_id
    model = load_engram_tables_cpu(a.model_dir, layer_device_map(cfg, a.gpu_mem_gib, True)).eval()
    dev = next(model.parameters()).device
    print(f"loaded in {time.time() - t0:.0f}s; bos {bos}", flush=True)

    # the instrumented forward reproduces the model's own (checked on one document)
    probe = torch.tensor(np.concatenate([[bos], valid_docs("all", 1, 1024, 512, 1)[0]]), device=dev).unsqueeze(0)
    with torch.no_grad():
        ref = token_nll(model, probe)
    ctl = Ctl()
    layers = instrument(model, ctl)
    M = sys.modules[type(model.model.layers[layers[0]].mixer).__module__]
    M_fast = getattr(M, "_fast", lambda t: False)
    with torch.no_grad():
        ctl.record = {}
        mine = token_nll(model, probe)
        recorded = sorted(k for k in ctl.record if isinstance(k, int))
        streams_recorded = sorted(k[1] for k in ctl.record if isinstance(k, tuple))
        ctl.record = None
        ctl.zero = set(layers)
        zeroed = token_nll(model, probe)
        ctl.zero = set()
    check = {"engram_layers": layers, "recorded_layers": recorded, "streams_recorded": streams_recorded,
             "ref_nll": float(ref.mean()),
             "instrumented_nll": float(mine.mean()), "max_abs_token_diff": float(np.abs(ref - mine).max()),
             "all_engram_zeroed_nll": float(zeroed.mean()),
             "fast_path": bool(M_fast(torch.zeros(1, 1, 1, device=dev, dtype=torch.bfloat16)))}
    print(json.dumps(check), flush=True)
    if recorded != layers or abs(check["all_engram_zeroed_nll"] - check["instrumented_nll"]) < 1e-6:
        raise SystemExit("the instrumented Engram forward is not in use")
    if abs(check["instrumented_nll"] - check["ref_nll"]) > 2e-3:
        raise SystemExit("the instrumented Engram forward does not reproduce this export's model code")
    gains = {}
    for i in layers:
        m = model.model.layers[i].mixer
        for n in ("hidden_norm", "key_norm", "conv_norm"):
            if getattr(m, n, None) is None:
                continue
            w = getattr(m, n).weight.float()
            gains[f"{n}_L{i}"] = {"mean": float(w.mean()), "rms": float(w.pow(2).mean().sqrt()),
                                  "absmin": float(w.abs().min()), "absmax": float(w.abs().max())}
        gains[f"conv1d_L{i}_rms"] = float(m.conv1d.weight.float().pow(2).mean().sqrt())
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "runs.json").write_text(json.dumps({"model": str(a.model_dir), "check": check, "gains": gains}, indent=1))

    sets = [(f"valid_{d}", lambda d=d: valid_docs(d, a.docs, a.seq_len, 512, 0)) for d in DOMAINS]
    if a.gsm8k:
        sets.append(("gsm8k", lambda: gsm8k_docs(tok, a.gsm8k)))
    if a.flores:
        sets += [(f"flores_{lang}", lambda lang=lang: flores_docs(tok, lang, a.flores)) for lang in LANGS]
    for name, make in sets:
        if (a.out / f"{name}.npz").exists():
            print(f"{name}: done before, skipped", flush=True)
            continue
        docs = make()
        if not docs:
            continue
        t1 = time.time()
        res = run_set(model, ctl, layers, docs, bos, dev, fixed_scales=fixed_scales, sweep_layer=a.sweep_layer)
        np.savez_compressed(a.out / f"{name}.npz", **res)
        d = {k: float(res[k].mean()) for k in res if not k.startswith(("x_", "g_", "alpha", "hn", "kn", "cos", "rms",
                                                                         "doc_"))}
        print(f"{name}: {len(docs)} docs, {int(res['doc_lengths'].sum())} tokens, {time.time() - t1:.0f}s; "
              + ", ".join(f"{k} {v:.4f}" for k, v in d.items()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
