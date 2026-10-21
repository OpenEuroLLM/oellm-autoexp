"""KronLSTM gate statistics from torch_dist checkpoints (run inside the training container):

    kron_gate_stats.py <ckpt dir>[=label] ...

Per checkpoint, over its KronLSTM layers: the forget gate's magnitude m (softplus gate) and, for the
polar gate, its angle theta = pi sigmoid(u) with u ~ N(bias, s^2) -- s the per-channel spread of
the angle pre-activation for unit-variance inputs (the RMS-normed hidden state), s = |row of
W2 W1| -- so the fraction of negative forget values P(u > 0) and E[cos theta]; the erase gate's
r = sigmoid(-z) and angle phi likewise (fraction reflecting, E[beta*]); and the effective memory
horizon -1 / ln|f| at the median input. Parameters from the fp32 master copies where saved.
"""
import math
import sys

import torch
from torch.distributed.checkpoint import FileSystemReader, load


def read(path):
    md = FileSystemReader(path).read_metadata().state_dict_metadata
    want = {}
    for k, v in md.items():
        if ".self_attention.layer." not in k or not hasattr(v, "size"):
            continue
        if not (k.startswith("decoder.") or k.startswith("chained_") and ".fp32_param." in k):
            continue
        name = k.split("fp32_param.")[-1]
        if any(s in name for s in ("A_log", "dt_bias", "f_proj", "f_angle_proj", "b_proj", "b_angle_proj")):
            want[k] = torch.empty(tuple(v.size), dtype=torch.float32)
    load(want, checkpoint_id=path, no_dist=True)
    params = {}
    for k, t in want.items():  # fp32 master copy wins over the bf16 model copy
        name = k.split("fp32_param.")[-1]
        if ".fp32_param." in k or name not in params:
            params[name] = t
    layers = sorted({int(n.split(".")[2]) for n in params if n.endswith("A_log")})
    return params, layers


def gauss_cos_stats(bias, spread, n=4096):
    eps = torch.randn(n)
    u = bias[:, None] + spread[:, None] * eps[None, :]
    theta = math.pi * torch.sigmoid(u)
    return (u > 0).float().mean(1), torch.cos(theta).mean(1), torch.cos(theta)


def main():
    torch.manual_seed(0)
    for spec in sys.argv[1:]:
        path, _, label = spec.partition("=")
        p, layers = read(path)
        print(f"== {label or path}: {len(layers)} KronLSTM layers")
        hz_all, neg_all, cos_all, refl_all, beta_all = [], [], [], [], []
        for i in layers:
            pre = f"decoder.layers.{i}.self_attention.layer."
            A = p[pre + "A_log"].exp()
            dt = p[pre + "dt_bias"]
            H, D = A.numel(), dt.numel() // A.numel()
            # magnitude at the median input (z = 0): log m = -A softplus(dt_bias)
            log_m = -(A[:, None] * torch.nn.functional.softplus(dt.view(H, D))).flatten()
            log_f = log_m
            if pre + "f_angle_proj.1.bias" in p:
                W = p[pre + "f_angle_proj.1.weight"] @ p[pre + "f_angle_proj.0.weight"]
                b = p[pre + "f_angle_proj.1.bias"]
                neg, ecos, cos = gauss_cos_stats(b, W.norm(dim=1))
                neg_all.append(neg)
                cos_all.append(ecos)
                med_cos = torch.cos(math.pi * torch.sigmoid(b)).abs().clamp_min(math.exp(-5))
                log_f = log_m + med_cos.log()
            hz_all.append(-1.0 / log_f.clamp(max=-1e-6))
            if pre + "b_angle_proj.bias" in p:
                wb = p[pre + "b_proj.weight"].norm(dim=1)
                wa = p[pre + "b_angle_proj.weight"].norm(dim=1)
                ba = p[pre + "b_angle_proj.bias"]
                eps = torch.randn(4096)
                r = torch.sigmoid(-wb[:, None] * eps[None])
                phi = math.pi * torch.sigmoid(ba[:, None] + wa[:, None] * torch.randn(4096)[None])
                beta = 1 - r * torch.cos(phi)
                refl_all.append((beta > 1).float().mean(1))
                beta_all.append(beta.mean(1))
        hz = torch.cat(hz_all)
        q = torch.quantile(hz, torch.tensor([0.1, 0.5, 0.9]))
        print(f"  forget horizon -1/ln|f| at the median input (tokens): p10 {q[0]:.0f}  p50 {q[1]:.0f}  p90 {q[2]:.0f}  "
              f"(> 100 tokens: {(hz > 100).float().mean():.1%})")
        if neg_all:
            neg = torch.cat(neg_all)
            print(f"  forget angle: negative-f share over inputs, mean {neg.mean():.1%}; channels with > 10 % negative: "
                  f"{(neg > 0.1).float().mean():.1%}; E[cos theta] mean {torch.cat(cos_all).mean():.3f} "
                  f"(init 0.988)")
        if refl_all:
            print(f"  erase: reflecting (beta* > 1) share {torch.cat(refl_all).mean():.1%}; E[beta*] {torch.cat(beta_all).mean():.3f} "
                  f"(init ~0.5)")


if __name__ == "__main__":
    main()
