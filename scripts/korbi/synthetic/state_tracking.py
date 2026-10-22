"""State tracking with one linear-mixer type (CKLSTM experiment B, user 2026-10-22).

Where signed gates / negative eigenvalues should matter regardless of LM loss (Grazzi et al. 2025,
"Unlocking state-tracking in linear RNNs through negative eigenvalues"; DeltaProduct): the running
state of a group word problem, predicted at every position. Small models of a single FLA mixer
type (pre-norm mixer + SwiGLU MLP blocks), trained at length 64, tested at 64 .. 4096 (length
extrapolation of the state update itself).

Tasks (token = group element, target = running product / sum, all positions):
  parity  Z2   (bits)                    -- needs a negative eigenvalue (or a sign) per step
  mod3    Z3   (running sum mod 3)       -- a rotation: products of reflections over >= 2 layers
  s3      S3   (6 permutations of 3)     -- non-abelian, solvable
  s5      S5   (120 permutations of 5)   -- non-solvable (NC1-complete word problem)

Mixers: kda, ckda (signed gate, beta in [0, 2]), kron (KronLSTM paper gates), and the CKLSTM
variants of screen5ckl: ckl_sgn, ckl_sgn_w2, ckl_sgn_cpl, ckl_polar_broad.

    python scripts/korbi/synthetic/state_tracking.py --mixers ckda ckl_sgn --tasks parity s3 \
        --out results.jsonl [--steps 4000 --layers 2 --dim 128]
Each (mixer, task, seed) appends one JSON line: accuracy per test length.
"""

import argparse
import itertools
import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- tasks
def _perm_table(n):
    perms = list(itertools.permutations(range(n)))
    index = {p: i for i, p in enumerate(perms)}
    # table[a, b] = index of (a then b): apply perm a, then perm b
    table = torch.tensor([[index[tuple(pb[pa[i]] for i in range(n))] for pb in perms] for pa in perms])
    return table


class Task:
    def __init__(self, name):
        self.name = name
        if name == "parity":
            self.table = torch.tensor([[0, 1], [1, 0]])
        elif name == "mod3":
            self.table = torch.tensor([[(a + b) % 3 for b in range(3)] for a in range(3)])
        elif name in ("s3", "s5"):
            self.table = _perm_table(3 if name == "s3" else 5)
        else:
            raise ValueError(name)
        self.n = self.table.shape[0]
        self.identity = 0  # the identity is element 0 for all tables above

    def sample(self, batch, length, device):
        x = torch.randint(0, self.n, (batch, length))
        y = torch.empty_like(x)
        state = torch.full((batch,), self.identity, dtype=torch.long)
        for t in range(length):
            state = self.table[state, x[:, t]]
            y[:, t] = state
        return x.to(device), y.to(device)


# ----------------------------------------------------------------------------- models
def make_mixer(kind, dim, heads, layer_idx):
    from fla.layers import ComplexKimiDeltaAttention, KimiDeltaAttention, KronLSTM

    hd = dim // heads
    if kind == "kda":
        return KimiDeltaAttention(hidden_size=dim, head_dim=hd, num_heads=heads, layer_idx=layer_idx)
    if kind == "ckda":
        return ComplexKimiDeltaAttention(hidden_size=dim, head_dim=hd, num_heads=heads, allow_neg_eigval=True,
                                         gate="signed_sigmoid2", gate_init_style="shipped", output_gate="lowrank",
                                         drop_silu=True, layer_idx=layer_idx)
    kron = dict(hidden_size=dim, k_head_dim=hd, v_head_dim=hd, num_k_heads=heads, num_v_heads=heads,
                output_gate_rank="lowrank", layer_idx=layer_idx)
    ext = dict(allow_neg_eigval=True, neg_eigval_init_bias=True)
    sgn = dict(**kron, **ext, gate="signed_sigmoid2")
    variants = {
        "kron": dict(**kron),
        "ckl_sgn": dict(**sgn, unit_write=True),
        "ckl_sgn_w2": dict(**sgn, unit_write=False),
        "ckl_sgn_cpl": dict(**sgn, unit_write=False, decouple_erase=False),
        "ckl_polar_broad": dict(**kron, **ext, unit_write=True, forget_angle=True, forget_angle_init="broad"),
    }
    if kind not in variants:
        raise ValueError(kind)
    return KronLSTM(**variants[kind])


class Block(nn.Module):
    def __init__(self, kind, dim, heads, idx):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(dim), nn.RMSNorm(dim)
        self.mixer = make_mixer(kind, dim, heads, idx)
        self.up = nn.Linear(dim, 8 * dim // 3 * 2, bias=False)
        self.down = nn.Linear(8 * dim // 3, dim, bias=False)

    def forward(self, x):
        out = self.mixer(self.n1(x))
        x = x + (out[0] if isinstance(out, tuple) else out)
        a, b = self.up(self.n2(x)).chunk(2, -1)
        return x + self.down(F.silu(a) * b)


class Model(nn.Module):
    def __init__(self, kind, vocab, dim, heads, layers):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList(Block(kind, dim, heads, i) for i in range(layers))
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))


# ----------------------------------------------------------------------------- run
@torch.no_grad()
def evaluate(model, task, lengths, device, batch=64):
    model.eval()
    res = {}
    for L in lengths:
        b = max(4, min(batch, 65536 // L))
        x, y = task.sample(b, L, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        acc = (logits.argmax(-1) == y).float()
        res[str(L)] = {"acc": acc.mean().item(), "acc_last_quarter": acc[:, -L // 4:].mean().item()}
    model.train()
    return res


def run(kind, task_name, seed, a, device):
    torch.manual_seed(seed)
    task = Task(task_name)
    model = Model(kind, task.n, a.dim, a.heads, a.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    t0, loss_ema = time.time(), None
    for step in range(a.steps):
        x, y = task.sample(a.batch, a.train_len, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        loss_ema = loss.item() if loss_ema is None else 0.98 * loss_ema + 0.02 * loss.item()
        if not math.isfinite(loss_ema):
            break
    return {"mixer": kind, "task": task_name, "seed": seed, "layers": a.layers, "dim": a.dim,
            "steps": a.steps, "train_len": a.train_len, "final_train_loss": loss_ema,
            "seconds": time.time() - t0, "eval": evaluate(model, task, a.lengths, device)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mixers", nargs="+",
                   default=["kda", "ckda", "kron", "ckl_sgn", "ckl_sgn_w2", "ckl_sgn_cpl", "ckl_polar_broad"])
    p.add_argument("--tasks", nargs="+", default=["parity", "mod3", "s3", "s5"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-len", type=int, default=64)
    p.add_argument("--lengths", type=int, nargs="+", default=[64, 256, 1024, 4096])
    p.add_argument("--shard", type=str, default="0/1", help="i/n: run every n-th job starting at i")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    i, n = map(int, a.shard.split("/"))
    device = "cuda"
    jobs = [(m, t, s) for t in a.tasks for m in a.mixers for s in a.seeds]
    for j, (m, t, s) in enumerate(jobs):
        if j % n != i:
            continue
        try:
            rec = run(m, t, s, a, device)
        except Exception as e:  # keep the sweep going; record the failure
            rec = {"mixer": m, "task": t, "seed": s, "error": f"{type(e).__name__}: {e}"[:500]}
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({k: rec.get(k) for k in ("mixer", "task", "seed", "final_train_loss", "error")}),
              flush=True)


if __name__ == "__main__":
    main()
