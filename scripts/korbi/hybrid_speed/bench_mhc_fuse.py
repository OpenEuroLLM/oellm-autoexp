"""mHC sublayer cost breakdown at the production shape (4 streams x 2048; 8k / 16k tokens per
micro-batch): forward and backward of coefficients + read and of the write, each against the
HBM time of the bytes it must move (copy bandwidth measured on the same GPU).

    python scripts/korbi/hybrid_speed/bench_mhc_fuse.py   (one GPU; PYTHONPATH as the training runs)
"""

import statistics

import torch

from megatron.core.transformer.mhc import MHCMixer

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


big = torch.empty(1 << 28, dtype=torch.bfloat16, device="cuda")  # 512 MB
dst = torch.empty_like(big)
bw = 2 * big.numel() * 2 / (timed(lambda: dst.copy_(big)) * 1e-3)  # bytes / s (read + write)
print(f"copy bandwidth {bw / 1e9:.0f} GB/s", flush=True)
del big, dst


def row(label, ms, nbytes):
    ideal = nbytes / bw * 1e3
    print(f"  {label:40s} {ms:7.3f} ms   ideal {ideal:6.3f} ms   {ideal / ms * 100:5.1f} % of HBM",
          flush=True)


for T in (8192, 16384):
    mixer = MHCMixer(C, N).cuda()
    with torch.no_grad():
        mixer.phi.normal_(0, 0.02)
    S = T * N * C * 2  # bytes of the bf16 stream state
    X = T * C * 2  # bytes of one [T, C] bf16 tensor
    streams = torch.randn(T, 1, N, C, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    f = torch.randn(T, 1, C, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    g_streams = torch.randn(T, 1, N, C, device="cuda", dtype=torch.bfloat16)
    g_x = torch.randn(T, 1, C, device="cuda", dtype=torch.bfloat16)
    print(f"T={T}: stream state {S / 2**20:.0f} MB", flush=True)

    def coeffs_read():
        return mixer.coeffs_and_read(streams)

    with torch.no_grad():
        row("fwd coeffs + read", timed(coeffs_read), S + X)
    x, h_pre, h_post, h_res = coeffs_read()

    def bwd_coeffs_read():
        x, _, h_post, h_res = coeffs_read()
        torch.autograd.backward([x, h_post, h_res],
                                [g_x, torch.ones_like(h_post), torch.ones_like(h_res)])

    t_fb = timed(bwd_coeffs_read)
    with torch.no_grad():
        t_f = timed(coeffs_read)
    row("bwd coeffs + read (fwd+bwd - fwd)", t_fb - t_f, 2 * S + X)  # read s, g_x; write d_s

    hp, hr = h_post.detach(), h_res.detach()
    sd = streams.detach().requires_grad_(True)
    fd = f.detach().requires_grad_(True)
    with torch.no_grad():
        t_wf = timed(lambda: mixer.write(sd, fd, hp, hr))
    row("fwd write", t_wf, 2 * S + X)

    def write_fb():
        mixer.write(sd, fd, hp, hr).backward(g_streams)

    row("bwd write (fwd+bwd - fwd)", timed(write_fb) - t_wf, 3 * S + 2 * X)  # g, s, f; d_s, d_f

    def sublayer():
        x, _, h_post, h_res = mixer.coeffs_and_read(streams)
        mixer.write(streams, x + f, h_post, h_res).backward(g_streams)

    row("sublayer fwd + bwd", timed(sublayer), 3 * S + 2 * X + 5 * S + 3 * X)

    from megatron.core.transformer import mhc_fused

    mhc_fused.ENABLED = True

    def fused_read():
        link = mhc_fused.ReadWriteLink()
        return mhc_fused.coeffs_and_read(mixer, streams, link), link

    with torch.no_grad():
        row("FUSED fwd coeffs + read", timed(fused_read), S + X)

    def fused_sublayer():
        (x, _, h_post, h_res), link = fused_read()
        mixer.write(streams, x + f, h_post, h_res, link=link).backward(g_streams)

    row("FUSED sublayer fwd + bwd", timed(fused_sublayer), 3 * S + 2 * X + 5 * S + 3 * X)
    mhc_fused.ENABLED = False
