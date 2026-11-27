#!/usr/bin/env python3
"""Replace the mHC-4 residual of an fsdp_dtensor checkpoint by the plain residual (collapse) or by Linear Depth
Memory (--ldm): the warm starts of experiments/pnazari/nucleus_ldm_replacement_mn5.

mHC carries n residual streams; the LM head reads their mean. A layer writes x' = H_res x + h_post^T F, and H_res is
doubly stochastic (its columns sum to 1), so on the mean

    m' = m + avg(h_post) F(Norm(sum_i h_pre_i x_i))          (mHC)
    m' = m + F(Norm(m))                                      (converted: one stream, the mean)

which is exact while the mixers are at their init (phi = 0, b_pre = logit(1/n), b_post = 0: every stream equal,
h_pre = 1/n, h_post = 1) and otherwise drops each layer's learned read mix (h_pre) and write weight (h_post). The
static part of the write weight, avg_i 2 sigmoid(b_post_i), is printed per mixer and not folded (on a MoE layer that
would rewrite every expert's fc2). Per tensor:

  mHC mixers  decoder.layers.<l>.{mhc_attn_mixer,mhc_mlp_mixer,residual_scheme.mhc_mixer}.{phi,b_*,alpha_*} stay in
              the tree unused (a strict load ignores extra keys; the first save drops them)
  V layers    v_gate.out_gate g_v is folded into self_attention.linear_proj: W' = g_v W, and its MuOwn row states
              g' = g_v g (the row norms), m_g' = m_g / g_v, v_g' = v_g / g_v^2 (d loss / d g scales by 1 / g_v: the
              same Adam step on g); m_u and v_norm are unchanged (the direction gradient is). v_gate.res_gate drops
              out: lerp(I, H_res, rho) is doubly stochastic, the mean does not see it
  --ldm       every mixer slot gets the LDM stream-form site (write_site, next_query) and the stack the boundary
              (ldm_stream_boundary: embed_write, first_query), initialized as Megatron does (address networks
              trunc-normal at init_method_std, zero gate heads, forgetting logit(lambda_init)) but every read gain 0:
              the continued model computes the converted one exactly, the memory opens through d loss / d gain.
              Fresh optimizer states (MuOwn for the matrices: g = row norms, v_norm 1, the rest 0; Adam for the
              vectors: 0). Their group metadata and step need nothing: the packed optimizer scalars hold only the
              saved parameters, and the load gives a parameter the checkpoint does not hold its current group's
              (checkpointing.pack_fsdp_dtensor_optimizer_scalars)
  args        mhc and v_block_learnable_gate off; --ldm: depth_memory on with the LDM settings

The new tree hard-links every .distcp of the source (dcp_surgery.write_surgery); only the rewritten / new tensors are
written.

    python mhc_to_ldm.py --src <ckpt>/iter_N --dst <new>/ckpt/iter_N [--ldm] [--dry-run]
    python mhc_to_ldm.py selftest      # a synthetic two-layer checkpoint, CPU
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

M = "model.module.decoder."
O = "optimizer.0.state.module.module.module.decoder."
MUOWN = ("g", "m_g", "v_g", "v_norm", "m_u")
ADAM = ("exp_avg", "exp_avg_sq")
MIXER = re.compile(r"layers\.(\d+)\.(mhc_attn_mixer|mhc_mlp_mixer|residual_scheme\.mhc_mixer)\.b_post")
V_GATE = re.compile(r"layers\.(\d+)\.v_gate\.out_gate")
PROJ = "self_attention.linear_proj.weight"


def ldm_parameters(hidden: int, width: int, groups: int, key_dim: int) -> dict[str, tuple[int, ...]]:
    """The parameters of one LDM write / read (DepthMemoryWrite / DepthMemoryRead), relative names -> shapes."""
    net = {"net.norm_weight": (hidden,), "net.w1.weight": (width, hidden), "net.w1.bias": (width,),
           "net.address.weight": (groups * key_dim, width), "net.address.bias": (groups * key_dim,),
           "net.gate.weight": (groups, width), "net.gate.bias": (groups,)}
    return {"write": net | {"forget_logit": (groups,)}, "read": net | {"read_gain": (groups,)}}


def ldm_init(name: str, shape, std: float, lambda_init: float, gen) -> torch.Tensor:
    """Megatron's init of one LDM parameter (depth_memory._AddressNetwork, DepthMemoryWrite / Read), read gain 0."""
    leaf = name.rsplit(".", 1)[-1]
    if name.endswith(("w1.weight", "address.weight")):
        return torch.nn.init.trunc_normal_(torch.empty(shape), std=std, a=-3 * std, b=3 * std, generator=gen)
    if name.endswith("norm_weight"):
        return torch.ones(shape)
    if leaf == "forget_logit":
        return torch.full(shape, math.log(lambda_init / (1.0 - lambda_init)))
    return torch.zeros(shape)  # biases, the gate heads (2 sigmoid(0) = 1), read_gain (the memory closed)


def imitation_values(a, groups: int):
    """The fitted fixed-address LDM of ldm_imitation_init.py (run ``a.imitate_run`` of ``a.imitate``) as per-site
    targets: writes [(address [G, K], gate logit, forget logit)] (the embedding's, then one per layer) and reads
    [(address [G, K], read gain)] (the first query, then each layer's following read; the last is the LM head's)."""
    entry = json.loads(Path(a.imitate).read_text())[a.imitate_run]
    p = {k: torch.tensor(v, dtype=torch.float64) for k, v in entry["params"].items()}
    lam_min = entry.get("lambda_min", 0.0)
    key = torch.nn.functional.normalize(p["key"], dim=-1)
    q = torch.nn.functional.normalize(p["q"], dim=-1)
    if a.duplicate:  # every slot twice: [k, k] / sqrt 2, function-preserving with exact keys
        key = torch.cat([key, key], -1) / math.sqrt(2)
        q = torch.cat([q, q], -1) / math.sqrt(2)
    K = key.shape[-1]
    w = 2 * torch.sigmoid(p["w"])
    lam = lam_min + (1 - lam_min) * torch.sigmoid(p["f"])
    logit = lambda x: math.log(x / (1 - x))  # noqa: E731
    gen = torch.Generator().manual_seed(a.seed)
    writes = [(key[t].expand(groups, K), logit(float(w[t]) / 2), logit(float(lam[t]))) for t in range(key.shape[0])]
    reads = []
    for t in range(q.shape[0]):  # symmetry breaking on the read side only (the read is linear)
        qq = q[t].expand(groups, K) + a.query_noise * torch.randn(groups, K, generator=gen, dtype=torch.float64) / math.sqrt(K)
        reads.append((torch.nn.functional.normalize(qq, dim=-1), float(p["s"][t])))
    return K, writes, reads


def imitated(rel: str, shape, w: torch.Tensor, value, kind: str, head_std: float, gen) -> torch.Tensor:
    """One LDM tensor of an imitated site: the address network gives the fitted constant address / gate for every
    token (w1 and the norm at Megatron's init, the heads at ``head_std`` -- tiny, nonzero for MuOwn's row
    directions -- with the fitted values in their biases); a site without a fitted value (the LM head's read when
    the fit had no final target) gets read gain 0."""
    if rel in ("net.address.weight", "net.gate.weight"):
        return torch.randn(shape, generator=gen) * head_std
    if value is None:
        return torch.zeros(shape) if rel == "read_gain" else w
    if rel == "net.address.bias":
        return value[0].reshape(-1).float()
    if rel == "net.gate.bias":
        return torch.full(shape, value[1] if kind == "write" else 0.0)
    if rel == "forget_logit":
        return torch.full(shape, value[2])
    if rel == "read_gain":
        return torch.full(shape, value[1])  # gate 2 sigmoid(0) = 1: gamma * gate = the fitted gain
    return w


def ldm_sites(smd) -> list[str]:
    """Every LDM site's module prefix (relative to decoder.): one per mHC mixer, plus the boundary's two."""
    sites = []
    for key in smd:
        if key.startswith(M):
            m = MIXER.fullmatch(key[len(M):])
            if m:
                sites.append((int(m.group(1)), key[len(M):-len(".b_post")]))
    out = []
    for _, prefix in sorted(sites):
        out += [f"{prefix}.write_site", f"{prefix}.next_query"]
    return ["ldm_stream_boundary.embed_write", "ldm_stream_boundary.first_query"] + out


def is_muown(shape) -> bool:
    return len(shape) == 2


def derive(smd, read, a, args) -> tuple[dict, dict]:
    """(replace: key -> new tensor, info)."""
    replace, info = {}, {"write_weight": {}, "v_gate": {}}
    # the static write weight of every mixer, for the record
    mixers = sorted((int(m.group(1)), k[len(M):]) for k in smd if k.startswith(M)
                    for m in [MIXER.fullmatch(k[len(M):])] if m)
    if not mixers:
        raise SystemExit("no mHC mixers (decoder.layers.<l>.*mixer.b_post) in the checkpoint")
    b_post = read([M + k for _, k in mixers])
    for (layer, k) in mixers:
        w = (2.0 * torch.sigmoid(b_post[M + k].float())).mean().item()
        info["write_weight"][k[: -len(".b_post")]] = round(w, 4)
    # V layers: g_v into linear_proj and its MuOwn row states
    gates = sorted(int(m.group(1)) for k in smd if k.startswith(M) for m in [V_GATE.fullmatch(k[len(M):])] if m)
    for layer in gates:
        gk = f"{M}layers.{layer}.v_gate.out_gate"
        wk = f"{M}layers.{layer}.{PROJ}"
        sk = {s: f"{O}layers.{layer}.{PROJ}.{s}" for s in MUOWN}
        missing = [k for k in [wk, *sk.values()] if k not in smd]
        if missing:
            raise SystemExit(f"V layer {layer}: {missing[0]} not in the checkpoint")
        t = read([gk, wk, sk["g"], sk["m_g"], sk["v_g"]])
        g_v = t[gk].float().reshape(-1)[0].item()
        if g_v == 0.0:
            raise SystemExit(f"V layer {layer}: g_v = 0 (the layer writes nothing; folding would zero its rows)")
        replace[wk] = (t[wk].float() * g_v).to(t[wk].dtype)
        replace[sk["g"]] = (t[sk["g"]].float() * g_v).to(t[sk["g"]].dtype)
        replace[sk["m_g"]] = (t[sk["m_g"]].float() / g_v).to(t[sk["m_g"]].dtype)
        replace[sk["v_g"]] = (t[sk["v_g"]].float() / g_v**2).to(t[sk["v_g"]].dtype)
        info["v_gate"][layer] = round(g_v, 6)
    if not a.ldm:
        return replace, info
    # LDM: every site's parameters and fresh optimizer states
    groups, key_dim, width = args.depth_memory_groups, args.depth_memory_key_dim, args.depth_memory_hidden_dim
    hidden, std = args.hidden_size, args.init_method_std
    shapes = ldm_parameters(hidden, width, groups, key_dim)
    dtype = smd[f"{M}{mixers[0][1]}"].properties.dtype  # the saved (main) parameters' dtype
    gen = torch.Generator().manual_seed(a.seed)
    sites = ldm_sites(smd)
    imitate = None
    if getattr(a, "imitate", None):
        K, writes, reads = imitation_values(a, groups)
        if K != key_dim:
            raise SystemExit(f"the imitation has key width {K}, the args {key_dim}")
        write_sites = [x for x in sites if x.endswith(("write_site", "embed_write"))]
        read_sites = [x for x in sites if x.endswith(("next_query", "first_query"))]
        if len(writes) != len(write_sites) or len(reads) not in (len(read_sites), len(read_sites) - 1):
            raise SystemExit(f"imitation: {len(writes)} writes / {len(reads)} reads for {len(write_sites)} / "
                             f"{len(read_sites)} sites")
        imitate = {**dict(zip(write_sites, writes)), **dict(zip(read_sites, reads))}
        info["imitation"] = {"run": a.imitate_run, "key_dim": K, "duplicate": a.duplicate,
                             "query_noise": a.query_noise, "head_std": a.head_std,
                             "final_read": len(reads) == len(read_sites)}
    for site in sites:
        kind = "write" if site.endswith(("write_site", "embed_write")) else "read"
        for rel, shape in shapes[kind].items():
            name = f"{site}.{rel}"
            if M + name in smd:
                raise SystemExit(f"{M + name} already exists: the source has LDM already")
            w = ldm_init(name, shape, std, args.depth_memory_lambda_init, gen)
            if imitate is not None:
                w = imitated(rel, shape, w, imitate.get(site), kind, a.head_std, gen)
            replace[M + name] = w.to(dtype)
            if is_muown(shape):
                rows, cols = shape
                replace[f"{O}{name}.g"] = w.norm(dim=1, keepdim=True)
                replace[f"{O}{name}.m_g"] = torch.zeros(rows, 1)
                replace[f"{O}{name}.v_g"] = torch.zeros(rows, 1)
                replace[f"{O}{name}.v_norm"] = torch.ones(rows, 1)
                replace[f"{O}{name}.m_u"] = torch.zeros(rows, cols)
            else:
                for s in ADAM:
                    replace[f"{O}{name}.{s}"] = torch.zeros(shape)
    info["ldm_sites"] = len(sites)
    return replace, info


def new_args(args, a):
    args = copy.deepcopy(args)
    args.mhc = False
    args.v_block_learnable_gate = False
    args.depth_memory = bool(a.ldm)
    if a.ldm:
        for k, v in (("depth_memory_impl", "cute"), ("depth_memory_unwind", True), ("depth_memory_groups", 16),
                     ("depth_memory_key_dim", getattr(a, "key_dim", 8)), ("depth_memory_hidden_dim", a.hidden_dim),
                     ("depth_memory_storage_dtype", "float16"), ("depth_memory_checkpoint_interval", 16),
                     ("residual_mixer_lr_mult", 1.0)):
            setattr(args, k, v)
    return args


def apply(a) -> int:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader

    from dcp_surgery import write_surgery

    t0 = time.time()
    log = lambda *x: print(f"[{time.time() - t0:7.1f}s]", *x, flush=True)  # noqa: E731
    reader = FileSystemReader(a.src)
    md = reader.read_metadata()
    smd = md.state_dict_metadata

    def read(keys):
        want = {k: torch.empty(tuple(smd[k].size), dtype=smd[k].properties.dtype) for k in keys}
        dcp.load(want, storage_reader=reader, no_dist=True)
        return want

    saved = {"args": None}
    dcp.load(saved, storage_reader=reader, no_dist=True)
    args = new_args(saved["args"], a)
    if not getattr(saved["args"], "mhc", False):
        raise SystemExit("the source was not trained with mhc")
    replace, info = derive(smd, read, a, args)
    replace["args"] = args
    log(f"write weights avg 2 sigmoid(b_post) per mixer: {json.dumps(info['write_weight'])}")
    log(f"V layers folded (g_v): {info['v_gate']}")
    log(f"{len(replace)} items new or rewritten" + (f", {info['ldm_sites']} LDM sites" if a.ldm else ""))
    if a.dry_run:
        for k in sorted(replace)[:40]:
            v = replace[k]
            log(f"  {k} {tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__}")
        log("dry run: nothing written")
        return 0
    write_surgery(a.src, a.dst, md, replace, tag="mhc_to_ldm", chunk_bytes=64 << 20, log=log)
    back = {k: torch.empty_like(v) for k, v in replace.items() if isinstance(v, torch.Tensor)}
    dcp.load(back, storage_reader=FileSystemReader(a.dst), no_dist=True)
    bad = [k for k in back if not torch.equal(back[k], replace[k])]
    if bad:
        raise SystemExit(f"read-back mismatch: {bad[:5]}")
    it = re.search(r"iter_(\d+)", os.path.basename(os.path.normpath(a.dst)))
    if it:
        Path(a.dst).parent.joinpath("latest_checkpointed_iteration.txt").write_text(f"{int(it.group(1))}\n")
    manifest = {"src": a.src, "ldm": a.ldm, "seed": a.seed, **info, "seconds": round(time.time() - t0)}
    Path(a.dst, "mhc_to_ldm_surgery.json").write_text(json.dumps(manifest, indent=1))
    log(f"read-back check passed; wrote {a.dst}")
    return 0


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if argv[:1] == ["selftest"]:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--ldm", action="store_true", help="add the LDM sites (read gain 0); else collapse only")
    ap.add_argument("--hidden-dim", type=int, default=128, help="depth_memory_hidden_dim (ambitious_v4_ldm: 128)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--imitate", default=None,
                    help="ldm_imitation_init.py json: the LDM sites imitate mHC (fitted fixed addresses) instead of "
                         "starting closed (read gain 0); needs --ldm")
    ap.add_argument("--imitate-run", default="K8_lam0.9_noise0.02", help="the run tag in the --imitate json")
    ap.add_argument("--duplicate", action="store_true", help="key width x2, every slot twice (exact keys)")
    ap.add_argument("--query-noise", type=float, default=0.0, help="per-group query noise (the read side only)")
    ap.add_argument("--head-std", type=float, default=1e-6, help="address / gate head weights of imitated sites")
    a = ap.parse_args(argv)
    if a.imitate:
        if not a.ldm:
            raise SystemExit("--imitate needs --ldm")
        entry = json.loads(Path(a.imitate).read_text())[a.imitate_run]
        a.key_dim = entry["K"] * (2 if a.duplicate else 1)
    return apply(a)


def selftest() -> int:
    """A synthetic three-layer mHC checkpoint (an attention mixer, a V layer, an Engram adapter) through both
    conversions: the V fold, the new LDM tensors and the args."""
    import tempfile

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader

    gen = torch.Generator().manual_seed(0)
    h, n, width = 8, 4, 4
    state = {"args": argparse.Namespace(mhc=True, v_block_learnable_gate=True, depth_memory=False, hidden_size=h,
                                        init_method_std=0.02, depth_memory_lambda_init=0.99)}
    mixers = ("layers.0.mhc_attn_mixer", "layers.1.mhc_attn_mixer", "layers.2.residual_scheme.mhc_mixer")
    for mixer in mixers:
        state[f"{M}{mixer}.phi"] = torch.randn(n * h, 2 * n + n * n, generator=gen)
        state[f"{M}{mixer}.b_post"] = torch.randn(n, generator=gen)
        state[f"{M}{mixer}.b_res"] = torch.randn(n, n, generator=gen)
        for st in ADAM:
            state[f"{O}{mixer}.b_post.{st}"] = torch.rand(n, generator=gen)
    w = torch.randn(h, 2 * h, generator=gen)
    state[f"{M}layers.1.{PROJ}"] = w
    states = {st: torch.rand(h, 1, generator=gen) for st in ("g", "m_g", "v_g", "v_norm")}
    states["m_u"] = torch.randn(h, 2 * h, generator=gen)
    for st, v in states.items():
        state[f"{O}layers.1.{PROJ}.{st}"] = v
    g_v = -0.37
    state[f"{M}layers.1.v_gate.out_gate"] = torch.tensor([g_v])
    state[f"{M}layers.1.v_gate.res_gate"] = torch.tensor([0.2])
    with tempfile.TemporaryDirectory() as root:
        src = os.path.join(root, "src", "iter_0000010")
        dcp.save(state, storage_writer=dcp.FileSystemWriter(src), no_dist=True)
        for ldm in (False, True):
            dst = os.path.join(root, "ldm" if ldm else "collapse", "iter_0000010")
            ns = argparse.Namespace(src=src, dst=dst, ldm=ldm, hidden_dim=width, seed=1, dry_run=False)
            assert apply(ns) == 0
            smd = FileSystemReader(dst).read_metadata().state_dict_metadata
            tensors = [k for k in smd if hasattr(smd[k], "size")]
            back = {k: torch.empty(tuple(smd[k].size), dtype=smd[k].properties.dtype) for k in tensors}
            back["args"] = None
            dcp.load(back, storage_reader=FileSystemReader(dst), no_dist=True)
            # the source's tensors stay, the V layer's projection and row states folded
            for k, v in state.items():
                if k == "args" or PROJ in k:
                    continue
                assert torch.equal(back[k], v), k
            assert torch.allclose(back[f"{M}layers.1.{PROJ}"], g_v * w)
            assert torch.allclose(back[f"{O}layers.1.{PROJ}.g"], g_v * states["g"])
            assert torch.allclose(back[f"{O}layers.1.{PROJ}.m_g"], states["m_g"] / g_v)
            assert torch.allclose(back[f"{O}layers.1.{PROJ}.v_g"], states["v_g"] / g_v**2)
            for st in ("v_norm", "m_u"):
                assert torch.equal(back[f"{O}layers.1.{PROJ}.{st}"], states[st]), st
            args = back["args"]
            assert not args.mhc and not args.v_block_learnable_gate and args.depth_memory == ldm
            new = sorted(set(tensors) - set(state))
            if not ldm:
                assert not new, new[:3]
                continue
            # 2 boundary sites + 2 per mixer, 7 network tensors + 1 each; MuOwn states for the 3 matrices (5 each),
            # Adam for the 5 vectors (2 each)
            sites = 2 + 2 * len(mixers)
            assert len(new) == sites * (8 + 3 * 5 + 5 * 2), len(new)
            assert all(k.startswith((M, O)) for k in new)
            for mixer in mixers:
                assert torch.equal(back[f"{M}{mixer}.next_query.read_gain"], torch.zeros(16))
                lam = torch.sigmoid(back[f"{M}{mixer}.write_site.forget_logit"])
                assert torch.allclose(lam, torch.full((16,), 0.99))
                w1 = back[f"{M}{mixer}.write_site.net.w1.weight"]
                assert w1.shape == (width, h) and torch.allclose(back[f"{O}{mixer}.write_site.net.w1.weight.g"],
                                                                 w1.norm(dim=1, keepdim=True))
                assert torch.equal(back[f"{M}{mixer}.write_site.net.gate.weight"], torch.zeros(16, width))
            assert torch.equal(back[f"{M}ldm_stream_boundary.first_query.read_gain"], torch.zeros(16))
            assert args.depth_memory_hidden_dim == width and args.depth_memory_checkpoint_interval == 16
        # --imitate: the fitted fixed addresses in the address heads' biases (4 writes: e + 3 mixers; reads: the
        # first query + 3 following reads, the last one the LM head's -- dropped here to check the gain-0 fallback)
        g = torch.Generator().manual_seed(5)
        fit = {"K": 8, "lambda_min": 0.9, "params": {
            "key": torch.randn(4, 8, generator=g).tolist(), "w": torch.randn(4, generator=g).tolist(),
            "f": torch.randn(4, generator=g).tolist(), "q": torch.randn(3, 8, generator=g).tolist(),
            "s": (3 * torch.randn(3, generator=g)).tolist()}}
        Path(root, "fit.json").write_text(json.dumps({"run": fit}))
        for dup in (False, True):
            dst = os.path.join(root, f"imitate{int(dup)}", "iter_0000010")
            ns = argparse.Namespace(src=src, dst=dst, ldm=True, hidden_dim=width, seed=1, dry_run=False,
                                    imitate=os.path.join(root, "fit.json"), imitate_run="run", duplicate=dup,
                                    query_noise=0.0, head_std=1e-6, key_dim=16 if dup else 8)
            assert apply(ns) == 0
            smd = FileSystemReader(dst).read_metadata().state_dict_metadata
            tensors = [k for k in smd if hasattr(smd[k], "size")]
            back = {k: torch.empty(tuple(smd[k].size), dtype=smd[k].properties.dtype) for k in tensors}
            back["args"] = None
            dcp.load(back, storage_reader=FileSystemReader(dst), no_dist=True)
            K = 16 if dup else 8
            assert back["args"].depth_memory_key_dim == K
            P = {k: torch.tensor(v, dtype=torch.float64) for k, v in fit["params"].items()}
            key = torch.nn.functional.normalize(P["key"], dim=-1)
            q = torch.nn.functional.normalize(P["q"], dim=-1)
            if dup:
                key, q = torch.cat([key, key], -1) / math.sqrt(2), torch.cat([q, q], -1) / math.sqrt(2)
            lam = 0.9 + 0.1 * torch.sigmoid(P["f"])
            writes = ["ldm_stream_boundary.embed_write"] + [f"{m}.write_site" for m in mixers]
            reads = ["ldm_stream_boundary.first_query"] + [f"{m}.next_query" for m in mixers]
            for t, site in enumerate(writes):
                assert torch.allclose(back[f"{M}{site}.net.address.bias"].double(), key[t].repeat(16)), site
                assert torch.allclose(back[f"{M}{site}.forget_logit"].double(), torch.logit(lam[t]).expand(16),
                                      atol=1e-5), site
                w = 2 * torch.sigmoid(back[f"{M}{site}.net.gate.bias"].double())
                assert torch.allclose(w, (2 * torch.sigmoid(P["w"][t])).expand(16), atol=1e-5), site
                assert back[f"{M}{site}.net.address.weight"].abs().max() < 1e-4, site
            for t, site in enumerate(reads):
                gain = back[f"{M}{site}.read_gain"].double()
                if t < 3:
                    assert torch.allclose(back[f"{M}{site}.net.address.bias"].double(), q[t].repeat(16)), site
                    assert torch.allclose(gain, P["s"][t].expand(16), atol=1e-5), site
                else:
                    assert torch.equal(gain, torch.zeros(16)), site  # no fitted final read: closed
    print("selftest passed")
    return 0



if __name__ == "__main__":
    sys.exit(main())
