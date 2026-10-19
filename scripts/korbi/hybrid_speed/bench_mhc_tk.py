"""mHC sublayer (coefficients + read + write) forward + backward: Megatron's current path vs the
read / write through DeepSeek's TileKernels (cute_kernels: oellm_mhc), at the production shape
(4 streams x 2048, 8k / 16k tokens per micro-batch). Correctness against the current path, then
timings (CUDA events, median of 30).

    python scripts/korbi/hybrid_speed/bench_mhc_tk.py   (one GPU; PYTHONPATH as the training runs)
"""

import statistics

import torch

from megatron.core.transformer.mhc import MHCMixer
from oellm_mhc import mhc_read, mhc_write

torch.manual_seed(0)
N, C = 4, 2048


def timed(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


for T in (8192, 16384):
    mixer = MHCMixer(C, N).cuda()
    with torch.no_grad():
        mixer.phi.normal_(0, 0.02)  # a non-trivial (input-dependent) mixer
    streams = torch.randn(T, 1, N, C, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    f = torch.randn(T, 1, C, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn(T, 1, N, C, device="cuda", dtype=torch.bfloat16)

    def current():
        x, _, h_post, h_res = mixer.coeffs_and_read(streams)
        return mixer.write(streams, x + f, h_post, h_res)

    def tilekernels():
        h_pre, h_post, h_res = mixer(streams)
        x = mhc_read(streams, h_pre)
        return mhc_write(streams, x + f, h_post, h_res)

    grads = {}
    for name, fn in (("current", current), ("tilekernels", tilekernels)):
        for t in (streams, f, mixer.phi):
            t.grad = None
        y = fn()
        y.backward(g)
        grads[name] = (y.detach(), streams.grad.clone(), f.grad.clone(), mixer.phi.grad.clone())
    errs = [rel(a, b) for a, b in zip(grads["tilekernels"], grads["current"])]
    print(f"T={T}: rel err vs current  out {errs[0]:.2e}  d_streams {errs[1]:.2e}  "
          f"d_f {errs[2]:.2e}  d_phi {errs[3]:.2e}", flush=True)

    def fwd_bwd(fn):
        def run():
            y = fn()
            y.backward(g)
        return run

    with torch.no_grad():
        h_pre, h_post, h_res = mixer(streams)
    sd = streams.detach().requires_grad_(True)
    fd = f.detach().requires_grad_(True)
    gr = torch.randn(T, 1, C, device="cuda", dtype=torch.bfloat16)
    rows = [
        ("sublayer (coeffs + read + write)", fwd_bwd(current), fwd_bwd(tilekernels)),
        ("read only", lambda: mixer.read(sd, h_pre).backward(gr),
         lambda: mhc_read(sd, h_pre).backward(gr)),
        ("write only", lambda: mixer.write(sd, fd, h_post, h_res).backward(g),
         lambda: mhc_write(sd, fd, h_post, h_res).backward(g)),
    ]
    for label, a, b in rows:
        ta, tb = timed(a), timed(b)
        print(f"T={T} {label:34s} fwd+bwd  current {ta:7.3f} ms  tilekernels {tb:7.3f} ms  "
              f"speedup {ta / tb:5.2f}x", flush=True)
