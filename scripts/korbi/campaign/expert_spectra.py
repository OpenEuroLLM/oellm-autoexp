"""Spectra of the routed experts' weight matrices across checkpoints (no forward pass): does an
expert's gain grow by alignment (sigma_max up, stable rank down at a flat Frobenius / row norm)?
For each checkpoint and MoE layer, every expert's fc1 gate (weight<i>_w), fc1 up (weight<i>_v) and
fc2 (weight<i>) [or the fused fc1 weight<i>, split into gate / up halves]: sigma_max, the Frobenius
norm, the stable rank |W|_F^2 / sigma_max^2 and the top-5 share of sum(sigma^2); and the expert's
end-to-end linear gain sigma_max(W2 W_up) against sigma_max(W2) sigma_max(W_up) (their ratio is 1
when fc2's input directions line up with fc1-up's output directions). Writes per layer the
quantiles over experts and the 5 experts with the largest sigma_max(W2 W_up).

    python expert_spectra.py <out.json> <layers, e.g. 10,30,69,71,73> <tag>=<checkpoint dir> ...
"""

import json
import re
import sys

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

KEY = re.compile(r"^(?:model\.)?(?:module\.)*decoder\.layers\.(\d+)\.mlp\.experts\.linear_fc([12])\.weight(\d+)(_w|_v)?$")
Q = (0.5, 0.9, 1.0)


def stats(w: torch.Tensor) -> dict:
    """[E, m, n] -> per-expert sigma_max, Frobenius, stable rank, top-5 share."""
    s = torch.linalg.svdvals(w.float())
    s2 = s.square()
    tot = s2.sum(-1)
    return {"smax": s[:, 0], "fro": tot.sqrt(), "srank": tot / s2[:, 0],
            "top5": s2[:, :5].sum(-1) / tot}


def quant(x: torch.Tensor) -> list:
    return [float(v) for v in torch.quantile(x.float().cpu(), torch.tensor(Q))]


def scan(path: str, layers: set) -> dict:
    md = FileSystemReader(path).read_metadata().state_dict_metadata
    found = {}
    for k in md:
        if "optimizer" in k:
            continue
        m = KEY.match(k)
        if m and int(m.group(1)) in layers:
            found.setdefault(int(m.group(1)), {})[(m.group(2), int(m.group(3)), m.group(4) or "")] = k
    out = {}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for L in sorted(found):
        keys = found[L]
        sd = {k: torch.empty(md[k].size, dtype=md[k].properties.dtype) for k in keys.values()}
        dcp.load(sd, storage_reader=FileSystemReader(path), no_dist=True)
        experts = sorted({e for (_, e, _) in keys})
        def get(fc, e, part):
            return sd[keys[(fc, e, part)]]
        if ("1", experts[0], "_w") in keys:
            gate = torch.stack([get("1", e, "_w") for e in experts]).to(dev)
            up = torch.stack([get("1", e, "_v") for e in experts]).to(dev)
        else:  # fused [2F, H]: gate rows then up rows
            fc1 = torch.stack([get("1", e, "") for e in experts]).to(dev)
            gate, up = fc1.chunk(2, dim=1)
        fc2 = torch.stack([get("2", e, "") for e in experts]).to(dev)
        res = {}
        for name, w in (("gate", gate), ("up", up), ("fc2", fc2)):
            st = stats(w)
            res[name] = {k: quant(v) for k, v in st.items()}
            res[name]["_smax"] = st["smax"]
        prod = torch.linalg.matrix_norm(torch.bmm(fc2.float(), up.float()), ord=2)  # [E]
        align = prod / (res["fc2"].pop("_smax") * res["up"]["_smax"])
        res["up"].pop("_smax"), res["gate"].pop("_smax")
        res["e2e_gain"] = quant(prod)
        res["e2e_align"] = quant(align)
        top = torch.topk(prod, min(5, prod.numel()))
        res["top_experts"] = [[experts[i], float(v), float(align[i])]
                              for v, i in zip(top.values.tolist(), top.indices.tolist())]
        out[L] = res
        print(f"{path} L{L}: gain med/max {res['e2e_gain'][0]:.3f}/{res['e2e_gain'][2]:.3f} "
              f"fc2 srank med {res['fc2']['srank'][0]:.1f}", flush=True)
        del sd, gate, up, fc2
    return out


def main():
    layers = {int(x) for x in sys.argv[2].split(",")}
    res = {}
    for spec in sys.argv[3:]:
        tag, path = spec.split("=", 1)
        res[tag] = scan(path, layers)
        json.dump(res, open(sys.argv[1], "w"))
    print("wrote", sys.argv[1])


if __name__ == "__main__":
    main()
