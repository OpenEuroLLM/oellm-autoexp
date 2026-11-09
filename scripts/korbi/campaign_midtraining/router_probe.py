"""R0 / R2 of the midtraining campaign (docs/campaign_3b_hybrid_midtraining.md §3.1, §5) on an HF export of
oellm_ambitious (v4_2ph): how predictable the MoE routing is, and what skipping unavailable experts costs.

R0  router predictability. Every MoE layer's router input, selection and gate weights are captured in a prefill
    forward. For every target layer m (the router's own top-k S_m) predictors rank the experts from what a
    decoder knows earlier:
      x<d>       the router input of the MoE layer d below m for the same token (d = 1, 2, 4), below layer 0
                 the token embedding                                              -- next-layer prediction
      xprev      layer m's router input for the previous token
      selprev    layer m's selection for the previous token (multi-hot)          -- next-token prediction
      emb        the token's input embedding (known once the token is sampled)
      xlastprev  the last MoE layer's router input for the previous token (R0 only: a parallel forward has
                 not computed it yet when layer m runs, so R2 cannot use it)
    Free ones (no parameters): base_d<d> / base_prev = layer m's router on x<d> / xprev, sel_prev = the
    previous token's experts first, hist16 = a decayed count of the last 16 tokens' experts, oracle = layer
    m's own router (checks the metrics: recall@5 = 1).
    Learned ones (trained online here with a listwise cross-entropy on S_m, prequential: every batch is scored
    before it is trained on; each starts at its free baseline): lin_d<d>, lin_prev, tokahead = xprev + selprev
    (known before the token is sampled), start = emb + xprev + selprev (known at the START of the token's
    decode step), start_deep = start + xlastprev, combo_d<d> = start + x<d>. One set per --lrs value; R2
    takes the best lr per kind.
    Metrics per target layer: recall@K (share of S_m in the predicted top-K) and gate-weighted recall@K. At
    batch 16 (16 rows at the same position = 16 decode streams): the coverage of all assignments by the union
    of the rows' top-K, and that union's size. Cache windows (the experts used in the last w positions, per
    row or over the batch) are the parameter-free reference.
R2  skip policy. The forward runs with an available-expert set per token and layer: a prefetch (a predictor's
    top-K, per row or the batch-16 union) and / or a cache window. Assignments outside it are skipped, with the
    kept gate weights renormalized (skip) or not (skipnr), or substituted by the router's top-k within the
    available set (sub). Reported: the NLL change vs the unmodified routing, the miss rate (assignments, gate
    weight) and the available-set size. Effects propagate through the layers, as they would at inference.

R3  routing locality (moe_router_stay_bonus). The selection gets a bonus beta (selection-score units) on the
    experts of the previous token: "par" = the previous token's plain top-k (Megatron's training rule, parallel),
    "hyst" = the previous token's FINAL selection (sequential, what decode can do directly). The gate weights
    stay the router's. Reported: the NLL change, the stay rate (share of a token's experts the previous token
    used too) and the miss rate of a batch-1 window cache holding the experts of the last w tokens (w = 1, 4, 16).

    python router_probe.py --model <export> --stages r0,r2 --out <dir> \\
        --data web_en=<prefix>,code=<prefix>,...,long_64k=<prefix> [--long long_64k]
    python router_probe.py --tiny --stages r0,r2 --out /tmp/x        # local check: reduced random model
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault(
    "OELLM_HF_PIPELINE", "0"
)  # one call per MoE layer per forward (no micro-batches)

KS = (5, 8, 10, 16, 24, 32, 48, 64)
UNION_KS = (5, 8, 10, 16, 24, 32)
WINDOWS = (1, 4, 16, 64)
FREE = ["oracle", "base_d1", "base_d2", "base_d4", "base_prev", "sel_prev", "hist16"]
LEARNED = {
    "lin_d1": ["x1"],
    "lin_d2": ["x2"],
    "lin_d4": ["x4"],
    "lin_prev": ["xprev"],
    "tokahead": ["xprev", "selprev"],  # before the token is sampled: the previous token only
    "start": ["emb", "xprev", "selprev"],
    "start_deep": [
        "emb",
        "xprev",
        "selprev",
        "xlastprev",
    ],  # + the previous token's last MoE input (R0 only)
    "combo_d1": ["x1", "emb", "xprev", "selprev"],
    "combo_d2": ["x2", "emb", "xprev", "selprev"],
    "combo_d4": ["x4", "emb", "xprev", "selprev"],
}


# --------------------------------------------------------------------------- model + routing
def load_model(path: str | None, tiny: bool):
    if tiny:
        sys.path.insert(
            0,
            str(
                Path(__file__).resolve().parents[3]
                / "experiment"
                / "013_hf_prefill_speed"
                / "scripts"
            ),
        )
        from tiny_model import build  # noqa: E402

        kinds = [
            "attention",
            "moe",
            "engram",
            "ckda",
            "moe",
            "ckda",
            "moe",
            "attention",
            "moe",
            "ckda",
            "moe",
            "ckda",
            "moe",
        ]
        model, cfg = build(kinds)
        return model
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto"
    ).eval()


def moe_layers(model):
    return [m for m in model.modules() if type(m).__name__ == "OellmMoE"]


def _mod(moe):
    return sys.modules[type(moe).__module__]


def gate_f32(moe):
    return _mod(moe)._cached(moe, "_gate_f32", lambda: moe.gate.float(), moe.gate)


def selection_score(moe, logits):
    """The router's ranking: sigmoid scores + expert bias (or logits - qb_beta under quantile balancing)."""
    if moe.quantile:
        return logits - moe.qb_beta.float()
    return torch.sigmoid(logits) + moe.e_score_correction_bias.float()


def gate_weights(moe, logits, idx, keep=None, renorm=True):
    """Megatron's gate weights of the selected experts [T, k] fp32 (normalized sigmoid scores times the
    scaling)
    ``keep`` [T, k] bool drops slots (their weight 0), renormalizing the rest or not."""
    s = torch.sigmoid(logits).gather(-1, idx)
    if keep is None:
        w = s / (s.sum(-1, keepdim=True) + 1e-20) if moe.top_k > 1 else s
    elif renorm:
        s = s * keep
        w = s / (s.sum(-1, keepdim=True) + 1e-20)
    else:
        w = s / (s.sum(-1, keepdim=True) + 1e-20) * keep
    return w * moe.scaling


def apply_experts(moe, t, idx, w):
    """The routed experts for given (idx [T, k], w [T, k]) + the shared expert, as OellmMoE's fused path."""
    if not _mod(moe)._fast(t):
        out = torch.zeros_like(t)
        wt = w.to(t.dtype)
        for e in torch.unique(idx).tolist():
            tok, slot = torch.where(idx == e)
            g, u = F.linear(t[tok], moe._expert("gate_up_proj", e, t.dtype)).chunk(2, dim=-1)
            y = F.linear(
                _mod(moe)._swiglu(g, u, moe.limit, moe.expert_norm),
                moe._expert("down_proj", e, t.dtype),
            )
            out.index_add_(0, tok, (y * wt[tok, slot, None]).to(out.dtype))
        if moe.shared_gate_proj is not None:
            sw = _mod(moe)._swiglu
            out = out + moe.shared_down_proj(
                sw(moe.shared_gate_proj(t), moe.shared_up_proj(t), moe.limit, moe.shared_norm)
            )
        return out
    kern = _mod(moe)._kernels()
    idx = idx.to(torch.int32)
    sched = kern.ExpertSchedule(
        idx, moe.num_experts, kern.block_m_for(idx.numel(), moe.num_experts)
    )
    fp8 = moe.fp8
    gu = kern.grouped_gemm(
        t, sched, moe.gate_up_proj, moe.gate_up_proj_scale if fp8 else None, gather=True
    )
    act = kern.swiglu_norm(gu, moe.limit, moe.expert_norm)
    y = kern.grouped_gemm(
        act, sched, moe.down_proj, moe.down_proj_scale if fp8 else None, gather=False
    )
    shared = None
    if moe.shared_gate_proj is not None:
        g, u = moe.shared_gate_proj.weight, moe.shared_up_proj.weight
        wgu = _mod(moe)._cached(moe, "_shared_gu", lambda: torch.cat([g, u]), g, u)
        shared = moe.shared_down_proj(
            kern.swiglu_norm(F.linear(t, wgu), moe.limit, moe.shared_norm)
        )
    return kern.moe_combine(y, sched, idx, w.to(t.dtype), shared)


class Probe:
    """Replaces every OellmMoE forward: routes in torch (exactly the router's rule), records the router input /
    selection / gate weights per layer, and applies an optional skip policy (R2)."""

    def __init__(self, model):
        self.model = model
        self.moes = moe_layers(model)
        self.policy = None  # callable(probe, m, moe, x [B,S,h], logits [B,S,E], idx [B,S,k]) -> (idx, w) or None
        self.record = True
        self.x, self.idx, self.w = {}, {}, {}
        self.emb = None
        self.stats = defaultdict(float)
        self.bypass = False  # the export's own MoE forward (the parity check of the torch routing)
        for m, moe in enumerate(self.moes):
            moe._probe_orig = moe.forward
            moe.forward = types.MethodType(self._forward_factory(m), moe)
        inner = model.model
        orig = inner._embed

        def _embed(this, *a, **k):
            x = orig(*a, **k)
            self.emb = x
            return x

        inner._embed = types.MethodType(_embed, inner)

    def _forward_factory(self, m):
        probe = self

        def forward(moe, x):
            if probe.bypass:
                return moe._probe_orig(x)
            shape = x.shape
            t = x.reshape(-1, shape[-1])
            logits = F.linear(t.float(), gate_f32(moe))
            idx = selection_score(moe, logits).topk(moe.top_k, dim=-1).indices
            if probe.record:
                probe.x[m] = x
                probe.idx[m] = idx.view(*shape[:-1], -1)
            out = None
            if probe.policy is not None:
                out = probe.policy(
                    probe, m, moe, x, logits.view(*shape[:-1], -1), idx.view(*shape[:-1], -1)
                )
            if out is None:
                w = gate_weights(moe, logits, idx)
            else:
                idx, w = out[0].reshape(-1, moe.top_k), out[1].reshape(-1, moe.top_k)
            if probe.record:
                probe.w[m] = gate_weights(moe, logits, probe.idx[m].reshape(-1, moe.top_k)).view(
                    *shape[:-1], -1
                )
            return apply_experts(moe, t, idx, w).view(shape)

        return forward

    def clear(self):
        self.x, self.idx, self.w, self.emb = {}, {}, {}, None


# --------------------------------------------------------------------------- features + predictors
def shift(t):
    """[B, S, ...] -> the previous position's value (zero at position 0)."""
    out = torch.zeros_like(t)
    out[:, 1:] = t[:, :-1]
    return out


def onehot(idx, e, dtype=torch.float32):
    out = torch.zeros(*idx.shape[:-1], e, dtype=dtype, device=idx.device)
    return out.scatter_(-1, idx, 1.0)


def feature(probe, name, m, dev, num_experts):
    if name == "emb":
        return probe.emb.to(dev)
    if name == "xprev":
        return shift(probe.x[m])
    if name == "xlastprev":
        return shift(probe.x[len(probe.moes) - 1].to(dev))
    if name == "selprev":
        return shift(onehot(probe.idx[m], num_experts))
    d = int(name[1:])
    return (probe.x[m - d] if m - d >= 0 else probe.emb).to(dev)


def free_score(kind, probe, m, moe):
    e = moe.num_experts
    dev = probe.x[m].device
    g = gate_f32(moe)
    if kind == "oracle":
        return selection_score(moe, F.linear(probe.x[m].float(), g))
    if kind.startswith("base_d") or kind == "base_prev":
        f = feature(probe, "xprev" if kind == "base_prev" else "x" + kind[6:], m, dev, e)
        return selection_score(moe, F.linear(f.float(), g))
    if kind == "sel_prev":
        prev = selection_score(moe, F.linear(shift(probe.x[m]).float(), g))
        return shift(onehot(probe.idx[m], e)) + 1e-3 * prev
    if kind == "hist16":
        oh = onehot(probe.idx[m], e)
        h = torch.zeros_like(oh)
        for j in range(1, 17):  # positions t-1 .. t-16, weight 0.8^(j-1)
            h[:, j:] += 0.8 ** (j - 1) * oh[:, :-j]
        return h
    raise ValueError(kind)


class Learned(torch.nn.Module):
    """beta * (layer m's router ranking on the main source) + learned linear terms of every feature (zero at
    init), so a predictor starts at its free baseline and can only learn to improve on it."""

    def __init__(self, feats, moe):
        super().__init__()
        self.feats = list(feats)
        self.moe = [moe]  # not a submodule: the router stays the export's
        self.main = next((f for f in self.feats if f.startswith("x") and f[1:].isdigit()), None)
        self.main = self.main or ("xprev" if "xprev" in self.feats else self.feats[0])
        g = moe.gate.detach()
        e, h = g.shape
        self.w = torch.nn.ParameterDict()
        for f in self.feats:
            shape = (e, e) if f == "selprev" else (e, h)
            self.w[f] = torch.nn.Parameter(torch.zeros(shape, device=g.device))
        self.b = torch.nn.Parameter(torch.zeros(e, device=g.device))
        self.beta = torch.nn.Parameter(torch.tensor(10.0, device=g.device))

    def forward(self, feats: dict):
        moe = self.moe[0]
        with torch.no_grad():
            base = selection_score(moe, F.linear(feats[self.main].float(), gate_f32(moe)))
        out = self.beta * base + self.b
        for f in self.feats:
            w = self.w[f]
            x = feats[f]
            out = out + (F.linear(x, w) if f == "selprev" else F.linear(x, w.to(x.dtype)).float())
        return out


def listwise_loss(score, idx, valid):
    lp = torch.log_softmax(score.float(), dim=-1).gather(-1, idx).mean(-1)
    return -(lp * valid).sum() / valid.sum().clamp(min=1)


# --------------------------------------------------------------------------- metrics
class Metrics:
    """Sums per (domain, predictor, target layer)
    "_n" counts the measured tokens."""

    def __init__(self):
        self.sum = defaultdict(lambda: defaultdict(float))

    def add(self, key, name, value):
        self.sum[key][name] += float(value)

    def table(self):
        out = {}
        for key, d in self.sum.items():
            n = max(d.get("_n", 0.0), 1e-9)
            out["|".join(map(str, key))] = {k: v / n for k, v in d.items() if k != "_n"} | {
                "tokens": n
            }
        return out


@torch.no_grad()
def score_metrics(met, key, score, idx, w, valid, union=True):
    """score [B,S,E], target idx [B,S,k], gate weights w [B,S,k], valid [B,S] (positions >= 1)."""
    b, s, e = score.shape
    tgt = score.gather(-1, idx)  # [B,S,k]
    rank = (score.unsqueeze(-2) > tgt.unsqueeze(-1)).sum(
        -1
    )  # [B,S,k]: experts ranked above each target
    v = valid.float()
    met.add(key, "_n", v.sum())
    wn = w / w.sum(-1, keepdim=True).clamp(min=1e-20)
    for k in KS:
        hit = (rank < k).float()
        met.add(key, f"recall@{k}", (hit.mean(-1) * v).sum())
        met.add(key, f"wrecall@{k}", ((hit * wn).sum(-1) * v).sum())
    if union and b >= 16:
        for k in UNION_KS:
            top = score.topk(k, dim=-1).indices
            u = (
                torch.zeros(b, s, e, dtype=torch.bool, device=score.device)
                .scatter_(-1, top, True)
                .any(0)
            )  # [S,E]
            cov = u.unsqueeze(0).expand(b, s, e).gather(-1, idx).float().mean(-1)  # [B,S]
            met.add(key, f"u16cov@{k}", (cov * v).sum())
            met.add(
                key, f"u16size@{k}", (u.sum(-1).float() * v[0]).sum() * b
            )  # per position, weighted like cov


@torch.no_grad()
def window_metrics(met, dom, m, idx, valid, e):
    b, s, _ = idx.shape
    oh = onehot(idx, e, torch.int32)  # [B,S,E]
    v = valid.float()
    n = v.sum()
    for scope in ("bs1", "bs16"):
        if scope == "bs16" and b < 16:
            continue
        used = oh if scope == "bs1" else oh.amax(0, keepdim=True).expand(b, s, e)
        cum = torch.cumsum(used, dim=1)
        for wnd in WINDOWS:
            prev = torch.zeros_like(cum)
            prev[:, 1:] = cum[:, :-1]  # uses in positions < t
            old = torch.zeros_like(cum)
            if wnd + 1 <= s:
                old[:, wnd + 1 :] = cum[:, : -wnd - 1]  # uses in positions < t - wnd
            cached = (prev - old) > 0  # used in [t - wnd, t - 1]
            key = (dom, f"win{wnd}_{scope}", m)
            hit = cached.gather(-1, idx).float().mean(-1)
            met.add(key, "_n", n)
            met.add(key, "cov", (hit * v).sum())
            met.add(key, "size", (cached.sum(-1).float() * v).sum())
    if b >= 16:
        key = (dom, "true_union16", m)
        u = oh.amax(0).sum(-1).float()  # [S]
        met.add(key, "_n", n)
        met.add(key, "size", (u * v[0]).sum() * b)


# --------------------------------------------------------------------------- data
_IDX_DTYPES = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float64,
    7: np.float32,
    8: np.uint16,
}


class read_indexed:
    """A Megatron indexed dataset (``<prefix>.idx`` / ``.bin``, MMIDIDX version 1) without Megatron: item i is
    sequence i (one per document in the v8ml sets)."""

    def __init__(self, prefix):
        with open(prefix + ".idx", "rb") as f:
            if f.read(9) != b"MMIDIDX\x00\x00":
                raise ValueError(f"{prefix}.idx: not an MMIDIDX index")
            (version,) = np.frombuffer(f.read(8), np.uint64)
            (code,) = np.frombuffer(f.read(1), np.uint8)
            n_seq, _ = np.frombuffer(f.read(16), np.uint64)
            offset = f.tell()
        if version != 1:
            raise ValueError(f"{prefix}.idx: version {version}")
        idx = np.memmap(prefix + ".idx", mode="r", order="C")
        self.lengths = np.frombuffer(idx, np.int32, int(n_seq), offset)
        self.pointers = np.frombuffer(idx, np.int64, int(n_seq), offset + self.lengths.nbytes)
        self.dtype = _IDX_DTYPES[int(code)]
        self.bin = np.memmap(prefix + ".bin", mode="r", order="C")

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, i):
        return np.frombuffer(self.bin, self.dtype, int(self.lengths[i]), int(self.pointers[i]))


def batches(prefix, seq_len, rows, n_batches, bos, part, seed=0):
    """Rows of ``seq_len`` tokens: documents in a seeded order (part 'head' from the front, 'tail' from the
    back, disjoint), each preceded by BOS and concatenated (no document masking)."""
    if prefix == "random":
        g = torch.Generator().manual_seed(seed + (part == "tail"))
        for _ in range(n_batches):
            yield torch.randint(3, 50_000, (rows, seq_len), generator=g)
        return
    docs = read_indexed(prefix)
    order = np.random.default_rng(seed).permutation(len(docs))
    if part == "tail":
        order = order[::-1]
    need = rows * seq_len
    buf, it = [], iter(order)
    for _ in range(n_batches):
        while len(buf) < need:
            try:
                buf.append(bos)
                buf.extend(docs[int(next(it))].tolist())
            except StopIteration:
                return
        yield torch.tensor(buf[:need], dtype=torch.long).view(rows, seq_len)
        buf = buf[need:]


def domain_iters(spec, args, part):
    out = {}
    for item in spec.split(","):
        name, prefix = item.split("=", 1)
        long = name in args.long.split(",")
        seq, rows = (args.long_seq_len, 1) if long else (args.seq_len, args.rows)
        out[name] = batches(
            prefix, seq, rows, args.batches if part == "head" else args.r2_batches, args.bos, part
        )
    return out


# --------------------------------------------------------------------------- R0
def run_r0(model, probe, args, out: Path):
    moes = probe.moes
    e = moes[0].num_experts
    lrs = [float(x) for x in args.lrs.split(",")]
    variants = {
        f"{k}@{lr:g}": (k, lr) for k in LEARNED for lr in lrs
    }  # every learned kind at every lr
    preds = {
        (v, m): Learned(LEARNED[k], moe)
        for v, (k, _) in variants.items()
        for m, moe in enumerate(moes)
    }
    opts = []  # one per target layer (its predictors' parameters are disjoint from the others')
    for m in range(len(moes)):
        groups = []
        for v, (k, lr) in variants.items():
            p = preds[(v, m)]
            # projections of hidden states at lr; transition table, bias and beta at 30 lr
            groups.append({"params": [t for n, t in p.w.items() if n != "selprev"], "lr": lr})
            groups.append(
                {
                    "params": [t for n, t in p.w.items() if n == "selprev"] + [p.b, p.beta],
                    "lr": 30 * lr,
                }
            )
        opts.append(torch.optim.Adam(groups))
    met = Metrics()
    iters = domain_iters(args.data, args, "head")
    traces = out / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    first_dev = next(model.parameters()).device
    t0 = time.time()
    for b in range(args.batches):
        for dom, it in list(iters.items()):
            ids = next(it, None)
            if ids is None:
                iters.pop(dom)
                continue
            probe.clear()
            with torch.no_grad():
                model.model(input_ids=ids.to(first_dev))
            measure = b >= args.warm
            valid = torch.ones(ids.shape, dtype=torch.bool)
            valid[:, 0] = False
            if b < args.trace_batches:
                np.savez_compressed(
                    traces / f"{dom}_{b:03d}.npz",
                    idx=torch.stack([probe.idx[m].cpu() for m in range(len(moes))], 2)
                    .short()
                    .numpy(),
                    w=torch.stack([probe.w[m].cpu() for m in range(len(moes))], 2).half().numpy(),
                )
            for m, moe in enumerate(moes):
                dev = probe.x[m].device
                vm = valid.to(dev)
                idx, w = probe.idx[m], probe.w[m]
                if measure:
                    for kind in FREE:
                        score_metrics(
                            met, (dom, kind, m), free_score(kind, probe, m, moe), idx, w, vm
                        )
                    window_metrics(met, dom, m, idx, vm, e)
                feats = {
                    f: feature(probe, f, m, dev, e)
                    for f in ("x1", "x2", "x4", "xprev", "selprev", "emb", "xlastprev")
                }
                if measure:
                    with torch.no_grad():
                        for v in variants:
                            score_metrics(met, (dom, v, m), preds[(v, m)](feats), idx, w, vm)
                # online training on this batch (after it was scored): row chunks keep the shifted features
                nb = ids.shape[0]
                step = max(1, nb // args.train_chunks)
                with torch.enable_grad():
                    for r in range(0, nb, step):
                        sub = {f: t[r : r + step] for f, t in feats.items()}
                        loss = sum(
                            listwise_loss(
                                preds[(v, m)](sub), idx[r : r + step], vm[r : r + step].float()
                            )
                            for v in variants
                        )
                        loss.backward()
                        opts[m].step()
                        opts[m].zero_grad(set_to_none=True)
                del feats
            print(f"[r0] batch {b} {dom} {tuple(ids.shape)} {time.time() - t0:.0f}s", flush=True)
    table = met.table()
    (out / "r0_metrics.json").write_text(json.dumps(table))
    # per learned kind the lr with the best mean recall@10 (over domains and layers) goes on to R2
    score = defaultdict(list)
    for key, d in table.items():
        _, v, _ = key.split("|")
        if v in variants:
            score[v].append(d["recall@10"])
    best = {}
    for k in LEARNED:
        cands = [v for v in variants if variants[v][0] == k and score.get(v)]
        if cands:
            best[k] = max(cands, key=lambda v: sum(score[v]) / len(score[v]))
    (out / "r0_best_lr.json").write_text(json.dumps(best, indent=1))
    chosen = {(k, m): preds[(v, m)] for k, v in best.items() for m in range(len(moes))}
    torch.save(
        {
            f"{k}|{m}": {n: t.detach().to(torch.bfloat16).cpu() for n, t in p.state_dict().items()}
            for (k, m), p in chosen.items()
            if k in ("tokahead", "start", "combo_d1", "combo_d2", "lin_d1")
        },
        out / "predictors.pt",
    )
    write_r0_summary(table, FREE + list(variants), out / "r0_summary.md")
    return chosen


def write_r0_summary(table, kinds, path):
    rows = defaultdict(lambda: defaultdict(list))  # (dom, kind) -> metric -> per-layer values
    for key, d in table.items():
        dom, kind, m = key.split("|")
        for k, v in d.items():
            rows[(dom, kind)][k].append(v)
    doms = sorted({d for d, _ in rows})
    lines = [
        "# R0 router predictability (mean over target MoE layers, prequential for learned predictors)",
        "",
    ]
    for dom in doms:
        lines += [
            f"## {dom}",
            "",
            "| predictor | "
            + " | ".join(f"R@{k}" for k in KS)
            + " | wR@5 | wR@10 | "
            + " | ".join(f"u16 cov@{k} (size)" for k in (5, 10, 16))
            + " |",
            "|---|" + "---:|" * (len(KS) + 5),
        ]
        for kind in kinds:
            d = rows.get((dom, kind))
            if not d:
                continue
            mean = {k: sum(v) / len(v) for k, v in d.items()}
            u = " | ".join(
                f"{mean.get(f'u16cov@{k}', float('nan')):.3f} ({mean.get(f'u16size@{k}', float('nan')):.0f})"
                for k in (5, 10, 16)
            )
            lines.append(
                f"| {kind} | "
                + " | ".join(f"{mean[f'recall@{k}']:.3f}" for k in KS)
                + f" | {mean['wrecall@5']:.3f} | {mean['wrecall@10']:.3f} | {u} |"
            )
        lines += ["", "| cache window | coverage | size (experts) |", "|---|---:|---:|"]
        for kind in [f"win{w}_{s}" for s in ("bs1", "bs16") for w in WINDOWS] + ["true_union16"]:
            d = rows.get((dom, kind))
            if d:
                mean = {k: sum(v) / len(v) for k, v in d.items()}
                lines.append(
                    f"| {kind} | {mean.get('cov', float('nan')):.3f} | {mean['size']:.1f} |"
                )
        lines.append("")
    path.write_text("\n".join(lines))


# --------------------------------------------------------------------------- R2
def make_policy(spec, preds):
    """spec: dict(pred=None|kind, K, win, scope bs1|bs16, miss skip|skipnr|sub, drop=n)."""

    def policy(probe, m, moe, x, logits, idx):
        b, s, e = logits.shape
        k = moe.top_k
        st = probe.stats
        if spec.get("drop"):
            keep = torch.ones_like(idx, dtype=torch.bool)
            keep[..., k - spec["drop"] :] = (
                False  # topk is sorted by selection score: drop the lowest ones
            )
            w = gate_weights(
                moe, logits.reshape(-1, e), idx.reshape(-1, k), keep.reshape(-1, k)
            ).view(b, s, k)
            lw = gate_weights(moe, logits.reshape(-1, e), idx.reshape(-1, k)).view(b, s, k)
            st["miss"] += float((~keep).sum())
            st["assign"] += idx.numel()
            st["wmiss"] += float((lw * ~keep).sum() / moe.scaling)
            st["wsum"] += float(lw.sum() / moe.scaling)
            st["avail"] += float(k - spec["drop"]) * idx[..., 0].numel()
            st["avail_n"] += idx[..., 0].numel()
            return idx, w
        avail = torch.zeros(b, s, e, dtype=torch.bool, device=x.device)
        if spec.get("pred"):
            kind = spec["pred"]
            if kind in LEARNED:
                feats = {f: feature(probe, f, m, x.device, e) for f in LEARNED[kind]}
                score = preds[(kind, m)](feats)
            else:
                score = free_score(kind, probe, m, moe)
            top = score.topk(spec["K"], dim=-1).indices
            avail.scatter_(-1, top, True)
        if spec.get("win"):
            oh = onehot(idx, e, torch.int32)
            cum = torch.cumsum(oh, dim=1)
            prev = torch.zeros_like(cum)
            prev[:, 1:] = cum[:, :-1]
            old = torch.zeros_like(cum)
            wnd = spec["win"]
            if wnd + 1 <= s:
                old[:, wnd + 1 :] = cum[:, : -wnd - 1]
            avail |= (prev - old) > 0
        if spec["scope"] == "bs16":
            avail = avail.any(0, keepdim=True).expand(b, s, e)
        avail[:, 0] = True  # position 0: everything (a prefill / the first token)
        hit = avail.gather(-1, idx)
        lw = gate_weights(moe, logits.reshape(-1, e), idx.reshape(-1, k)).view(b, s, k)
        st["miss"] += float((~hit[:, 1:]).sum())
        st["assign"] += float(hit[:, 1:].numel())
        st["wmiss"] += float((lw * ~hit)[:, 1:].sum() / moe.scaling)
        st["wsum"] += float(lw[:, 1:].sum() / moe.scaling)
        st["avail"] += float(avail[:, 1:].sum(-1).float().mean()) * idx[:, 1:, 0].numel()
        st["avail_n"] += idx[:, 1:, 0].numel()
        if spec["miss"] == "sub":
            sel = selection_score(moe, logits).masked_fill(~avail, float("-inf"))
            nidx = sel.topk(k, dim=-1).indices
            return nidx, gate_weights(moe, logits.reshape(-1, e), nidx.reshape(-1, k)).view(b, s, k)
        w = gate_weights(
            moe,
            logits.reshape(-1, e),
            idx.reshape(-1, k),
            hit.reshape(-1, k),
            renorm=spec["miss"] == "skip",
        ).view(b, s, k)
        return idx, w

    return policy


def r2_specs(have_learned: bool, rows16: bool):
    specs = [dict(name="full"), dict(name="drop1", drop=1), dict(name="drop2", drop=2)]
    preds = ["base_d1", "base_prev"] + (
        ["tokahead", "start", "combo_d1", "combo_d2"] if have_learned else []
    )
    for p in preds:
        for kk in (5, 10, 16):
            for miss in ("skip", "sub"):
                specs.append(
                    dict(name=f"{p}_K{kk}_bs1_{miss}", pred=p, K=kk, scope="bs1", miss=miss)
                )
    specs.append(
        dict(name="base_d1_K10_bs1_skipnr", pred="base_d1", K=10, scope="bs1", miss="skipnr")
    )
    for wnd in (16, 64):
        for miss in ("skip", "sub"):
            specs.append(dict(name=f"win{wnd}_bs1_{miss}", win=wnd, scope="bs1", miss=miss))
    best = "combo_d1" if have_learned else "base_d1"
    for miss in ("skip", "sub"):
        specs.append(
            dict(
                name=f"win16+{best}_K10_bs1_{miss}", pred=best, K=10, win=16, scope="bs1", miss=miss
            )
        )
    if rows16:
        for kk in (5, 10):
            for miss in ("skip", "sub"):
                specs.append(
                    dict(name=f"{best}_K{kk}_bs16_{miss}", pred=best, K=kk, scope="bs16", miss=miss)
                )
                specs.append(
                    dict(
                        name=f"win4+{best}_K{kk}_bs16_{miss}",
                        pred=best,
                        K=kk,
                        win=4,
                        scope="bs16",
                        miss=miss,
                    )
                )
        for miss in ("skip", "sub"):
            specs.append(dict(name=f"win4_bs16_{miss}", win=4, scope="bs16", miss=miss))
    return specs


@torch.no_grad()
def nll(model, ids, first_dev):
    h = model.model(input_ids=ids.to(first_dev)).last_hidden_state
    tot, n = 0.0, 0
    labels = ids.to(h.device)
    for r in range(h.shape[0]):
        for a in range(0, h.shape[1] - 1, 2048):
            e_ = min(a + 2048, h.shape[1] - 1)
            lp = torch.log_softmax(model.lm_head(h[r, a:e_]).float(), dim=-1)
            tot += float(-lp.gather(-1, labels[r, a + 1 : e_ + 1, None]).sum())
            n += e_ - a
    return tot, n


def run_r2(model, probe, args, out: Path, preds):
    first_dev = next(model.parameters()).device
    data = {dom: list(it) for dom, it in domain_iters(args.data, args, "tail").items()}
    long = set(args.long.split(","))
    results = {}
    t0 = time.time()
    dom0 = next(d for d, bl in data.items() if bl)
    probe.bypass = True
    native = nll(model, data[dom0][0], first_dev)
    probe.bypass = False
    probe.clear()
    mine = nll(model, data[dom0][0], first_dev)
    results["check|native_vs_probe"] = {
        "native": native[0] / native[1],
        "probe": mine[0] / mine[1],
        "domain": dom0,
    }
    print(
        f"[r2] parity {dom0}: native {native[0] / native[1]:.6f} probe routing {mine[0] / mine[1]:.6f}",
        flush=True,
    )
    for dom, bl in data.items():
        if not bl:
            continue
        rows16 = bl[0].shape[0] >= 16 and dom not in long
        for spec in r2_specs(preds is not None, rows16):
            if spec.get("pred") in LEARNED and preds is None:
                continue
            probe.policy = None if spec["name"] == "full" else make_policy(spec, preds)
            probe.record = True
            probe.stats = defaultdict(float)
            tot = n = 0.0
            for ids in bl:
                probe.clear()
                a, c = nll(model, ids, first_dev)
                tot += a
                n += c
            st = probe.stats
            results[f"{dom}|{spec['name']}"] = {
                "nll": tot / n,
                "tokens": n,
                "miss": st["miss"] / max(st["assign"], 1),
                "wmiss": st["wmiss"] / max(st["wsum"], 1e-9),
                "avail": st["avail"] / max(st["avail_n"], 1),
                "spec": {k: v for k, v in spec.items() if k != "name"},
            }
            r = results[f"{dom}|{spec['name']}"]
            print(
                f"[r2] {dom} {spec['name']}: nll {r['nll']:.5f} miss {r['miss']:.3f} {time.time() - t0:.0f}s",
                flush=True,
            )
    probe.policy = None
    (out / "r2_results.json").write_text(json.dumps(results, indent=1))
    lines = [
        "# R2 skip policy (NLL change vs the unmodified routing. miss: share of the router's assignments "
        "outside the available set, w-miss: gate-weighted, avail: available experts per layer)",
        "",
    ]
    chk = results["check|native_vs_probe"]
    lines += [
        f"Parity ({chk['domain']}, one batch): export forward {chk['native']:.6f}, probe routing "
        f"{chk['probe']:.6f}",
        "",
    ]
    for dom in data:
        base = results.get(f"{dom}|full", {}).get("nll")
        if base is None:
            continue
        lines += [
            f"## {dom} (full NLL {base:.4f})",
            "",
            "| policy | ΔNLL | miss | w-miss | avail |",
            "|---|---:|---:|---:|---:|",
        ]
        for key, r in results.items():
            d, name = key.split("|")
            if d == dom and name != "full" and "nll" in r:
                lines.append(
                    f"| {name} | {r['nll'] - base:+.4f} | {r['miss']:.3f} | {r['wmiss']:.3f} | {r['avail']:.1f} |"
                )
        lines.append("")
    (out / "r2_summary.md").write_text("\n".join(lines))


# --------------------------------------------------------------------------- R3
R3_BETAS = (0.01, 0.03, 0.1, 0.3, 1.0)


def stay_policy(mode, beta):
    """The routing with a stay bonus: mode "par" (bonus on the previous token's plain top-k) or "hyst" (on its final
    selection, sequential over positions). Records stay / window-cache statistics of the final selection."""

    def policy(probe, m, moe, x, logits, idx):
        b, s, e = logits.shape
        k = moe.top_k
        sel = selection_score(moe, logits)
        if mode == "par":
            nidx = (sel + beta * shift(onehot(idx, e))).topk(k, dim=-1).indices
        else:
            nidx = torch.empty_like(idx)
            nidx[:, 0] = idx[:, 0]
            prev = onehot(idx[:, 0], e)
            for t in range(1, s):
                cur = (sel[:, t] + beta * prev).topk(k, dim=-1).indices
                nidx[:, t] = cur
                prev = onehot(cur, e)
        window_stats(probe.stats, nidx, e)
        return nidx, gate_weights(moe, logits.reshape(-1, e), nidx.reshape(-1, k)).view(b, s, k)

    return policy


def window_stats(st, idx, e):
    """stay = share of a token's experts in the previous token's; miss<w> = share outside the last w tokens'."""
    oh = onehot(idx, e, torch.int32)
    cum = torch.cumsum(oh, dim=1)
    s = idx.shape[1]
    n = float(idx[:, 1:].numel())
    for w in (1, 4, 16):
        prev = torch.zeros_like(cum)
        prev[:, 1:] = cum[:, :-1]
        old = torch.zeros_like(cum)
        if w + 1 <= s:
            old[:, w + 1 :] = cum[:, : -w - 1]
        hit = ((prev - old) > 0).gather(-1, idx)[:, 1:]
        st[f"miss{w}"] += float((~hit).sum())
    st["assign"] += n


def run_r3(model, probe, args, out: Path):
    first_dev = next(model.parameters()).device
    data = {dom: list(it)[: args.r3_batches] for dom, it in domain_iters(args.data, args, "tail").items()}
    results = {}
    t0 = time.time()
    specs = [("full", None)] + [(f"{mode}_b{beta:g}", (mode, beta)) for mode in ("par", "hyst") for beta in R3_BETAS]
    for dom, bl in data.items():
        if not bl:
            continue
        for name, spec in specs:
            probe.policy = stay_policy(*spec) if spec else stay_policy("par", 0.0)
            probe.record = False
            probe.stats = defaultdict(float)
            tot = n = 0.0
            for ids in bl:
                probe.clear()
                a, c = nll(model, ids, first_dev)
                tot += a
                n += c
            st = probe.stats
            r = {"nll": tot / n, "tokens": n, "spec": spec}
            for w in (1, 4, 16):
                r[f"miss{w}"] = st[f"miss{w}"] / max(st["assign"], 1)
            r["stay"] = 1 - r["miss1"]
            results[f"{dom}|{name}"] = r
            print(f"[r3] {dom} {name}: nll {r['nll']:.5f} stay {r['stay']:.3f} miss4 {r['miss4']:.3f} "
                  f"miss16 {r['miss16']:.3f} {time.time() - t0:.0f}s", flush=True)
            (out / "r3_results.json").write_text(json.dumps(results, indent=1))
    probe.policy = None
    probe.record = True
    lines = ["# R3 routing locality (stay bonus beta in selection-score units; par = bonus on the previous token's "
             "plain top-k as in training, hyst = on its final selection; miss<w> = share of assignments outside "
             "the last w tokens' experts, batch 1)", ""]
    for dom in data:
        base = results.get(f"{dom}|full")
        if base is None:
            continue
        lines += [f"## {dom} (full NLL {base['nll']:.4f}, stay {base['stay']:.3f})", "",
                  "| policy | ΔNLL | stay | miss1 | miss4 | miss16 |", "|---|---:|---:|---:|---:|---:|"]
        for key, r in results.items():
            d, name = key.split("|")
            if d == dom:
                lines.append(f"| {name} | {r['nll'] - base['nll']:+.4f} | {r['stay']:.3f} | {r['miss1']:.3f} | "
                             f"{r['miss4']:.3f} | {r['miss16']:.3f} |")
        lines.append("")
    (out / "r3_summary.md").write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model")
    p.add_argument("--tiny", action="store_true", help="reduced random-weight model (local check)")
    p.add_argument("--data", default="random=random")
    p.add_argument(
        "--long", default="long_64k", help="domain names read as one row of --long-seq-len"
    )
    p.add_argument("--stages", default="r0,r2")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--long-seq-len", type=int, default=65536)
    p.add_argument("--rows", type=int, default=16)
    p.add_argument("--batches", type=int, default=24, help="R0 batches per domain")
    p.add_argument(
        "--warm",
        type=int,
        default=8,
        help="R0: batches before measuring (learned predictors warm up)",
    )
    p.add_argument("--trace-batches", type=int, default=2)
    p.add_argument("--train-chunks", type=int, default=4)
    p.add_argument(
        "--lrs",
        default="3e-6,3e-5",
        help="Adam lr of the hidden-state projections, one predictor set each",
    )
    p.add_argument(
        "--r2-batches", type=int, default=3, help="R2 batches per domain (disjoint documents)"
    )
    p.add_argument("--r3-batches", type=int, default=2, help="R3 batches per domain")
    p.add_argument("--predictors", type=Path, help="R2 without R0: predictors.pt of an R0 run")
    p.add_argument("--bos", type=int, default=1)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps(vars(args), default=str, indent=1))
    t0 = time.time()
    model = load_model(args.model, args.tiny)
    print(f"[probe] model loaded in {time.time() - t0:.0f}s", flush=True)
    probe = Probe(model)
    stages = args.stages.split(",")
    preds = None
    if "r0" in stages:
        preds = run_r0(model, probe, args, args.out)
    elif args.predictors:
        state = torch.load(args.predictors)
        preds = {}
        for key, sd in state.items():
            kind, m = key.split("|")
            mod = Learned(LEARNED[kind], probe.moes[int(m)])
            mod.load_state_dict({k: v.float() for k, v in sd.items()})
            preds[(kind, int(m))] = mod
    if "r2" in stages:
        run_r2(model, probe, args, args.out, preds)
    if "r3" in stages:
        run_r3(model, probe, args, args.out)
    print(f"[probe] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
