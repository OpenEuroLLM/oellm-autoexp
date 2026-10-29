"""LDM health from torch_dist checkpoints: per site the learned forgetting factors
lambda = sigmoid(forget_logit), the read gains, the write / read gate biases (2 sigmoid(b): the
write weight omega and the read gate at a zero hidden state), the address networks' row norms,
and -- the gradient side -- Adam's moments of the 1-D LDM parameters: grad RMS sqrt(v) and the
signal-to-noise |m| / sqrt(v) (bias correction is negligible past a few hundred steps). The other
1-D Adam parameters of the model (norms, biases, gains) are summarized the same way as the
control group. Reads only these tensors -- no model weights beyond the small LDM ones, no forward.

    python ldm_ckpt_stats.py <out.json> <tag>=<checkpoint iter dir> ...
"""

import json
import re
import sys

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

# a transformer layer's two sites (mhc_attn_mixer after the mixer, mhc_mlp_mixer after the MLP), an
# Engram layer's (residual_scheme.mhc_mixer), and the boundary's embedding write / first query
SITE = (r"decoder\.(?:layers\.(\d+)\.(mhc_attn_mixer|mhc_mlp_mixer|residual_scheme\.mhc_mixer)|"
        r"(ldm_stream_boundary))\.(write_site|next_query|embed_write|first_query)")
MODEL = re.compile(r"^model\.(?:module\.)*" + SITE + r"\.(.+)$")
OPT = re.compile(r"^optimizer\.\d+\.state\.(?:module\.)*" + SITE + r"\.(.+)\.(exp_avg|exp_avg_sq|g)$")
# the control group: every other small Adam parameter (Engram's row-wise tables are excluded by size)
OTHER = re.compile(r"^optimizer\.\d+\.state\.(?:module\.)*(decoder\..+|embedding\..+|output_layer\..+)\.(exp_avg|exp_avg_sq)$")
MAX_OTHER = 1 << 16


KIND = {"mhc_attn_mixer": "mix", "mhc_mlp_mixer": "mlp", "residual_scheme.mhc_mixer": "eng"}


def site_name(m):
    layer, where, _, part = m.group(1), m.group(2), m.group(3), m.group(4)
    return f"boundary.{part}" if layer is None else f"L{int(layer):02d}.{KIND[where]}.{part}"


def q(t, qs=(0.0, 0.5, 1.0)):
    t = t.double().flatten()
    return [float(torch.quantile(t, x)) for x in qs]


def scan(path):
    md = FileSystemReader(path).read_metadata().state_dict_metadata
    want = {}
    for k, v in md.items():
        m = MODEL.match(k)
        if m:
            want[k] = ("model", site_name(m), m.group(5), None)
            continue
        m = OPT.match(k)
        if m:
            want[k] = ("opt", site_name(m), m.group(5), m.group(6))
            continue
        m = OTHER.match(k)
        if m and not re.search(r"mhc_(attn_|mlp_)?mixer|ldm_stream_boundary", k) \
                and v.size.numel() <= MAX_OTHER and len(v.size) == 1:
            want[k] = ("other", m.group(1), None, m.group(2))
    sd = {k: torch.empty(md[k].size, dtype=md[k].properties.dtype) for k in want}
    dcp.load(sd, storage_reader=FileSystemReader(path), no_dist=True)
    sites, other = {}, {}
    for k, (kind, name, param, moment) in want.items():
        t = sd[k].float()
        if kind == "other":
            other.setdefault(name, {})[moment] = t
            continue
        s = sites.setdefault(name, {})
        if kind == "model":
            s[param] = t
        else:
            s.setdefault("opt", {}).setdefault(param, {})[moment] = t
    out = {"sites": {}, "other": {}}
    for name, s in sorted(sites.items()):
        r = {}
        if "forget_logit" in s:
            r["lambda"] = q(torch.sigmoid(s["forget_logit"]))
        if "read_gain" in s:
            r["read_gain"] = q(s["read_gain"])
        if "net.gate.bias" in s:
            r["gate_at_bias"] = q(2 * torch.sigmoid(s["net.gate.bias"]))
            r["gate_w_norm"] = float(s["net.gate.weight"].norm())
        for lin in ("w1", "address"):
            if f"net.{lin}.weight" in s:
                r[f"{lin}_row_norm"] = q(s[f"net.{lin}.weight"].norm(dim=-1))
        if "net.norm_weight" in s:
            r["norm_weight"] = q(s["net.norm_weight"])
        for param, mom in s.get("opt", {}).items():
            if "exp_avg_sq" in mom:
                rms = mom["exp_avg_sq"].clamp_min(0).sqrt()
                r[f"grad_rms.{param}"] = float(rms.pow(2).mean().sqrt())
                if "exp_avg" in mom:
                    snr = mom["exp_avg"].abs() / (rms + 1e-30)
                    r[f"snr.{param}"] = float(snr.median())
        out["sites"][name] = r
    rms_all, snr_all, named = [], [], []
    for name, mom in other.items():
        if "exp_avg_sq" in mom and "exp_avg" in mom:
            rms = mom["exp_avg_sq"].clamp_min(0).sqrt()
            rms_all.append(float(rms.pow(2).mean().sqrt()))
            named.append((rms_all[-1], name))
            snr_all.append(float((mom["exp_avg"].abs() / (rms + 1e-30)).median()))
    if rms_all:
        out["other"] = {"count": len(rms_all), "grad_rms": q(torch.tensor(rms_all), (0.1, 0.5, 0.9, 1.0)),
                        "snr": q(torch.tensor(snr_all), (0.1, 0.5, 0.9)),
                        "top": sorted(named, reverse=True)[:25]}
    return out


def main():
    out_path, specs = sys.argv[1], sys.argv[2:]
    result = {}
    for spec in specs:
        tag, path = spec.split("=", 1)
        result[tag] = scan(path)
        print(f"{tag}: {len(result[tag]['sites'])} sites", flush=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
