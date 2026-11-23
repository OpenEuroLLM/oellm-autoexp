"""MoonEP stress at v4b shapes under extreme routing skew (2026-11-23, the v4b main run's
cudaErrorIllegalAddress hunt): S = 65536 tokens per rank, H 2048, F 768, 360 experts top-5 over
EP 4 (one node), 64 SMs, token padding 128 -- through Megatron's own MoonEP path (_MoonEPDispatch ->
_MoonEPExperts (prefetch, TE grouped GEMMs, normed SwiGLU, wgrad into the reduce pools,
reduce_grad) -> _MoonEPCombine), several MoE layers sharing the process-global Buffer and pools,
optionally under full recompute (torch.utils.checkpoint, as --recompute-granularity full).

Routing patterns go from uniform to the worst cases the planner has to absorb: every token of
every rank on the same 5 experts, all tokens on one rank's experts, each rank on its neighbour's
experts, Zipf, near-empty expert sets, alternating bursts. Per-rank random sleeps and side-stream
GEMMs desynchronize the ranks. Checks per iteration: finite outputs / grads; every --check
iterations the raw dispatch -> combine round trip (== K * hidden, routing weights back exactly).

    torchrun --nproc-per-node 4 scripts/korbi/hybrid_speed/moonep_skew_stress.py --iters 400
"""

import argparse
import json
import os
import random
import time

import torch
import torch.distributed as dist

from megatron.core.transformer.moe import moonep as M

p = argparse.ArgumentParser()
p.add_argument("--S", type=int, default=65536)
p.add_argument("--H", type=int, default=2048)
p.add_argument("--F", type=int, default=768)
p.add_argument("--E", type=int, default=360)
p.add_argument("--K", type=int, default=5)
p.add_argument("--sms", type=int, default=64)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--iters", type=int, default=400)
p.add_argument("--patterns", default="all")
p.add_argument("--recompute", type=int, default=1, help="1: checkpoint pairs of layers")
p.add_argument("--jitter", type=int, default=1)
p.add_argument("--side", type=int, default=1, help="side-stream GEMMs during the MoE layers")
p.add_argument("--check", type=int, default=10)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
dev = torch.device("cuda", torch.cuda.current_device())
dist.init_process_group("nccl", device_id=dev)
group = dist.new_group(list(range(world)))
R, S, H, F, E, K = world, args.S, args.H, args.F, args.E, args.K
epn = E // R
torch.manual_seed(args.seed * 100 + rank)
rng = random.Random(args.seed * 100 + rank)


def log(msg, all_ranks=False):
    if all_ranks or rank == 0:
        print(f"[r{rank} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Cfg:  # the v4b main run's expert activation settings
    activation_func_clamp_value = 10.0
    layernorm_epsilon = 1e-5
    moe_expert_post_act_norm = True
    post_act_norm_floor = 1.0
    post_act_norm_smooth = False


class Experts:
    """MoonEPGroupedMLP's methods on a plain object (no MegatronModule / parallel state)."""

    local_weights = M.MoonEPGroupedMLP.local_weights
    main_grads = M.MoonEPGroupedMLP.main_grads
    reduce_copies_grad = M.MoonEPGroupedMLP.reduce_copies_grad

    def __init__(self):
        self.num_local_experts = epn
        self.config = Cfg()
        init = lambda w: torch.nn.init.normal_(w, std=0.02)  # noqa: E731
        self.linear_fc1 = M._ExpertWeights(epn, (2 * F, H), init, True, torch.bfloat16, dev)
        self.linear_fc2 = M._ExpertWeights(epn, (H, F), init, True, torch.bfloat16, dev)
        self._dummy_w = torch.zeros(epn, M._DUMMY, M._DUMMY, device=dev, dtype=torch.bfloat16)
        self._dummy_g = torch.zeros(epn, M._DUMMY, M._DUMMY, device=dev, dtype=torch.float32)
        self.pool_shapes = [(2 * F, H), (H, F), (M._DUMMY, M._DUMMY)]
        self._record_act = False

    def weights(self):
        return self.linear_fc1.weights() + self.linear_fc2.weights()


state = M._MoonEPState.get(group, S, H, K, E, num_sms=args.sms, token_padding=128)
layers = [(Experts(), M._PlanHolder()) for _ in range(args.layers)]
log(f"MoonEP stress R={R} S={S} H={H} F={F} E={E} K={K} sms={args.sms} layers={args.layers} "
    f"recompute={args.recompute} multicast={os.environ.get('MOONEP_MULTICAST', 'auto')} "
    f"pdl={os.environ.get('MOONEP_PDL', '1')} blocking={os.environ.get('CUDA_LAUNCH_BLOCKING', '0')}")


# ---------------------------------------------------------------- routing patterns
def distinct_from(pool, n):
    """[n, K] rows of K distinct experts drawn uniformly from ``pool`` (|pool| >= K)."""
    pool = torch.as_tensor(pool, device=dev)
    idx = torch.rand(n, pool.numel(), device=dev).argsort(dim=1)[:, :K]
    return pool[idx]


def weighted_distinct(probs, n):
    """[n, K] distinct experts sampled without replacement by ``probs`` [E]."""
    return torch.multinomial(probs.expand(n, -1).contiguous(), K, replacement=False)


def experts_of(r):
    return list(range(r * epn, (r + 1) * epn))


def pattern(name, it):
    """topk [S, K] int32 for this rank (every rank uses the same name at the same iteration)."""
    g = torch.Generator(device="cpu").manual_seed(it)  # same random choice on all ranks
    if name == "uniform":
        return distinct_from(list(range(E)), S)
    if name == "same5":  # every token of every rank -> the same 5 experts (capacity exactly CAP)
        hot = torch.randperm(E, generator=g)[:K].tolist()
        return torch.tensor(hot, device=dev).expand(S, K).contiguous()
    if name == "same5_one_rank":  # the same 5 experts, all homed on one rank
        r = int(torch.randint(R, (1,), generator=g))
        return torch.tensor(experts_of(r)[:K], device=dev).expand(S, K).contiguous()
    if name.startswith("hotset"):  # hotsetN: all tokens on N experts (random homes)
        n = int(name[6:])
        hot = torch.randperm(E, generator=g)[:n].tolist()
        return distinct_from(hot, S)
    if name.startswith("rankset"):  # ranksetN: all tokens on N experts of one rank
        n = int(name[7:])
        r = int(torch.randint(R, (1,), generator=g))
        return distinct_from(experts_of(r)[:n], S)
    if name == "neighbour":  # rank i -> only the experts of rank i+1
        return distinct_from(experts_of((rank + 1) % R), S)
    if name == "neighbour5":  # rank i -> 5 experts of rank i+1
        return torch.tensor(experts_of((rank + 1) % R)[:K], device=dev).expand(S, K).contiguous()
    if name.startswith("zipf"):  # zipfA: Zipf(A) over a random expert order
        a = float(name[4:].replace("p", "."))
        order = torch.randperm(E, generator=g)
        w = torch.zeros(E)
        w[order] = 1.0 / torch.arange(1, E + 1, dtype=torch.float64).pow(a).float()
        return weighted_distinct(w.to(dev), S)
    if name == "rank0_zipf":  # one rank's 90 experts, Zipf(1.5) inside
        r = int(torch.randint(R, (1,), generator=g))
        w = torch.zeros(E)
        w[experts_of(r)] = 1.0 / torch.arange(1, epn + 1, dtype=torch.float64).pow(1.5).float()
        return weighted_distinct(w.to(dev), S)
    if name == "half_hot":  # half the tokens on 5 experts, the rest uniform
        hot = torch.randperm(E, generator=g)[:K].tolist()
        t = distinct_from(list(range(E)), S)
        t[: S // 2] = torch.tensor(hot, device=dev)
        return t
    if name == "per_rank_hot":  # each rank its own 5 hot experts, all on the last rank
        return torch.tensor(experts_of(R - 1)[rank * K:(rank + 1) * K], device=dev).expand(S, K).contiguous()
    if name == "near_cap":  # each rank's experts loaded just above / below the rank capacity
        w = torch.ones(E)
        r = int(torch.randint(R, (1,), generator=g))
        w[experts_of(r)] = 1.0 + 0.05 * (it % 7)
        return weighted_distinct(w.to(dev), S)
    if name == "sorted":  # uniform experts but tokens sorted by expert (bursty per-SM work)
        t = distinct_from(list(range(E)), S)
        return t[t[:, 0].argsort()]
    raise ValueError(name)


ALL = ["uniform", "same5", "same5_one_rank", "hotset5", "hotset6", "hotset8", "hotset16",
       "hotset45", "hotset90", "rankset5", "rankset8", "rankset45", "rankset90", "neighbour",
       "neighbour5", "zipf1p0", "zipf1p5", "zipf2p5", "rank0_zipf", "half_hot", "per_rank_hot",
       "near_cap", "sorted"]
patterns = ALL if args.patterns == "all" else args.patterns.split(",")


def routing(name, it):
    topk = pattern(name, it).to(torch.int32).contiguous()
    assert topk.shape == (S, K)
    tpe = torch.bincount(topk.reshape(-1).long(), minlength=E).to(torch.int32).contiguous()
    w = torch.rand(S, K, device=dev).softmax(-1).float().contiguous()
    return topk, tpe, w


# ---------------------------------------------------------------- the Megatron MoonEP path
side = torch.cuda.Stream()
side_a = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)


def jitter():
    if args.jitter and rng.random() < 0.5:
        torch.cuda._sleep(rng.randint(1000, 2_000_000))  # up to ~1 ms on this rank only
    if args.side and rng.random() < 0.5:
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(rng.randint(1, 4)):
                side_a.matmul(side_a)
            torch.empty(rng.randint(1, 256) << 20, dtype=torch.uint8, device=dev).fill_(1)


def moe_layer(h, i, topk, tpe, w):
    experts, holder = layers[i]
    jitter()
    x, wn, cu = M._MoonEPDispatch.apply(h, w, topk, tpe, state, holder)
    y = M._MoonEPExperts.apply(x, wn, cu, experts, state, holder, *experts.weights())
    return h + M._MoonEPCombine.apply(y, state, holder)


def pair(h, i, routes):
    for j in range(i, min(i + 2, args.layers)):
        h = moe_layer(h, j, *routes[j])
    return h


def train_iter(name, it):
    routes = [routing(name, it * 31 + j) for j in range(args.layers)]
    for r in routes:
        r[2].requires_grad_(True)
    h0 = (torch.randn(S, H, device=dev, dtype=torch.bfloat16) * 0.5).requires_grad_(True)
    h = h0
    for i in range(0, args.layers, 2):
        if args.recompute:
            h = torch.utils.checkpoint.checkpoint(pair, h, i, routes, use_reentrant=True)
        else:
            h = pair(h, i, routes)
    loss = h.float().pow(2).mean()
    loss.backward()
    grads = [h0.grad] + [r[2].grad for r in routes]
    for e, _ in layers:
        grads += [*e.main_grads()[0][:2], *e.main_grads()[1][:2]]
    finite = bool(torch.stack([g.float().isfinite().all() for g in grads if g is not None]).all())
    finite &= bool(loss.isfinite())
    return float(loss), finite, routes


def check_roundtrip(name, it):
    """Raw API: combine(dispatch(x)) == K * x and the routing weights come back unchanged."""
    topk, tpe, w = routing(name, it)
    x = torch.randn(S, H, device=dev, dtype=torch.bfloat16)
    xn, wn, cu, plan = state.buffer.dispatch(x, w, topk, tpe)
    out, wsk, _ = state.buffer.combine(plan=plan, hidden_nvsh=xn, route_weights_nvs=wn)
    err = (out.float() - K * x.float()).abs().max().item()
    werr = (wsk.float() - w).abs().max().item() if wsk is not None else 0.0
    used = int(cu[-1].item())
    ok = err <= 0.05 * K * x.float().abs().max().item() and werr < 1e-5
    return ok, err, werr, used


# ---------------------------------------------------------------- run
summary = {"pattern_iters": {}, "bad": []}
t0 = time.time()
for it in range(args.iters):
    name = patterns[it % len(patterns)] if (it // len(patterns)) % 2 == 0 else rng.choice(patterns)
    name = [name]
    dist.broadcast_object_list(name, src=0)
    name = name[0]
    loss, finite, _ = train_iter(name, it)
    ok = True
    if args.check and it % args.check == 0:
        ok, err, werr, used = check_roundtrip(name, 10_000 + it)
        if not ok or rank == 0:
            log(f"it {it} roundtrip {name}: ok={ok} err={err:.4g} werr={werr:.3g} rows={used}",
                all_ranks=not ok)
    torch.cuda.synchronize()
    summary["pattern_iters"][name] = summary["pattern_iters"].get(name, 0) + 1
    if not finite or not ok:
        summary["bad"].append({"it": it, "pattern": name, "finite": finite, "roundtrip": ok})
        log(f"BAD it {it} {name}: finite={finite} roundtrip={ok}", all_ranks=True)
    if rank == 0 and (it % 10 == 0 or it == args.iters - 1):
        log(f"it {it} {name} loss {loss:.5f} finite {finite} {(time.time() - t0) / (it + 1):.2f} s/it")

dist.barrier()
bad = torch.tensor([len(summary["bad"])], device=dev)
dist.all_reduce(bad)
log(f"STRESS DONE {args.iters} iters, bad={int(bad.item())} {json.dumps(summary)}", all_ranks=True)
state.buffer.destroy()
dist.destroy_process_group()
