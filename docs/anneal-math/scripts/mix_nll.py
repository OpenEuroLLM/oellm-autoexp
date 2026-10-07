# ruff: noqa: E701, E702  -- kept verbatim as the record of the code that produced data/heldout_nll
# Per-dataset NLL of HF checkpoints (bf16, no BOS, like training) on a fixed anneal-mix sample.
import json
import sys
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM

H = "/e/scratch/e-sta-openeurollm/production_training/downstream_eval_32b/hf"
W = "/e/scratch/e-sta-openeurollm/poeppel1/v1annealC_probe"
z = np.load(f"{W}/mix_sample.npz")
offs = np.concatenate([[0], np.cumsum(z["lens"])])
for ck in sys.argv[1:]:
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(
        f"{H}/{ck}", dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    per = {}
    for i, g in enumerate(z["names"]):
        x = torch.from_numpy(z["toks"][offs[i] : offs[i + 1]]).cuda()[None]
        with torch.no_grad():
            lg = m(x).logits[0, :-1].float()
        nll = torch.nn.functional.cross_entropy(lg, x[0, 1:], reduction="none")
        s = per.setdefault(str(g), [0.0, 0])
        s[0] += nll.sum().item()
        s[1] += nll.numel()
    res = {g: s[0] / s[1] for g, s in per.items()}
    wts = dict(zip(z["groups"], z["weights"]))
    res["_weighted"] = sum(wts[g] * v for g, v in res.items() if g in wts) / sum(wts.values())
    json.dump(res, open(f"{W}/mixnll/{ck}.json", "w"), indent=1)
    print(ck, f"{time.time() - t0:.0f}s weighted={res['_weighted']:.4f}", flush=True)
    del m
    torch.cuda.empty_cache()
