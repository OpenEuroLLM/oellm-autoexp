"""Inference kernels of the ambitious HF port (prefill and scoring; experiment 013).

Each kernel replaces a chain of eager ops and keeps the reference's bf16 rounding points
(the reduction order inside a rounding step may differ):

- :func:`rms_norm`: ``OellmRMSNorm`` (fp32 statistics, one rounding);
- :func:`mhc_read` / :func:`mhc_write`: the mHC read (norm, coefficients, Sinkhorn, the
  weighted stream sum; optionally the layer's input norm on top) and write (``h_res`` stream
  mix + ``h_post`` x output), one launch each next to the coefficient GEMM;
- :func:`engram_rows`: the Engram n-gram hashes of a prefill (exact int64 XOR / multiply,
  Python's modulo, the row offsets);
- :class:`UvaTable`: an Engram table in pinned host memory, gathered by a GPU kernel
  (zero-copy; no host synchronization), optionally times the sign rows;
- :func:`route`: the sigmoid router (selection on scores + bias or logits - qb_beta, top-k,
  renormalized unbiased scores x scaling);
- :func:`grouped_gemm`: the routed experts' GEMMs, grouped by expert, the rows gathered in
  the kernel; FP8 (E4M3 + fp32 scale per 128 elements) weights are dequantized in the kernel
  exactly as Megatron does (``q * scale`` in fp32, rounded once to bf16), so the experts stay
  FP8 in memory;
- :func:`swiglu_norm`: SwiGLU with the clamp and the bounded post-activation norm;
- :func:`moe_combine`: the weighted expert outputs added per token in ascending expert order
  with a bf16 rounding after each add (the reference's ``index_add_`` order), + the shared
  expert.

No function synchronizes with the host, so a forward pass runs ahead of the GPU and the
layers on different GPUs of a device map overlap.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _guard(t: torch.Tensor):
    """The kernels launch on the current device; a device-mapped model has inputs elsewhere."""
    return torch.cuda.device(t.device)


# --------------------------------------------------------------------------- RMSNorm
@triton.jit
def _rms_kernel(X, W, Y, rows, H: tl.constexpr, EPS: tl.constexpr, ZC: tl.constexpr,
                BLOCK: tl.constexpr, ROWS: tl.constexpr):
    r = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    h = tl.arange(0, BLOCK)
    ok = (r[:, None] < rows) & (h[None, :] < H)
    x = tl.load(X + r[:, None] * H + h[None, :], ok, 0).to(tl.float32)
    w = tl.load(W + h, h < H, 0).to(tl.float32)
    if ZC:
        w += 1.0
    inv = tl.rsqrt(tl.sum(x * x, 1) / H + EPS)
    tl.store(Y + r[:, None] * H + h[None, :], (w[None, :] * (x * inv[:, None])).to(Y.dtype.element_ty), ok)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, zero_centered: bool = False) -> torch.Tensor:
    h = x.shape[-1]
    x2 = x.reshape(-1, h)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    out = torch.empty_like(x2)
    rows = x2.shape[0]
    if rows:
        block = triton.next_power_of_2(h)
        per = max(1, min(8, 4096 // block))
        with _guard(x):
            _rms_kernel[(triton.cdiv(rows, per),)](x2, weight, out, rows, h, eps, zero_centered, block, per,
                                                   num_warps=8 if block >= 2048 else 4)
    return out.view(x.shape)


# --------------------------------------------------------------------------- mHC
@triton.jit
def _mhc_read_kernel(S, C, APRE, APOST, ARES, BPRE, BPOST, BRES, X, POST, RES, NW,
                     H: tl.constexpr, EPS: tl.constexpr, ITERS: tl.constexpr, BLOCK: tl.constexpr,
                     NORM: tl.constexpr, NEPS: tl.constexpr, ZC: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, BLOCK)
    s = tl.arange(0, 4)
    dtype = S.dtype.element_ty
    stream = tl.load(S + t * 4 * H + s[:, None] * H + h[None, :], h[None, :] < H, 0).to(tl.float32)
    # torch.linalg.vector_norm returns the input dtype before .float()
    norm = tl.sqrt(tl.sum(tl.sum(stream * stream, 1), 0)).to(dtype).to(tl.float32)
    rms = tl.maximum(norm / ((4 * H) ** 0.5), EPS)
    cp = tl.load(C + t * 24 + s).to(tl.float32) / rms
    cq = tl.load(C + t * 24 + 4 + s).to(tl.float32) / rms
    pre = tl.sigmoid(tl.load(APRE).to(tl.float32) * cp + tl.load(BPRE + s).to(tl.float32))
    post = 2.0 * tl.sigmoid(tl.load(APOST).to(tl.float32) * cq + tl.load(BPOST + s).to(tl.float32))
    j = tl.arange(0, 4)
    offsets = s[:, None] * 4 + j[None, :]
    matrix = tl.load(C + t * 24 + 8 + offsets).to(tl.float32) / rms
    matrix = tl.load(ARES).to(tl.float32) * matrix + tl.load(BRES + offsets).to(tl.float32)
    for _ in range(ITERS):
        maximum = tl.max(matrix, 0)
        lse = maximum + tl.log(tl.sum(tl.exp(matrix - maximum[None, :]), 0))
        matrix -= lse[None, :]
        maximum = tl.max(matrix, 1)
        lse = maximum + tl.log(tl.sum(tl.exp(matrix - maximum[:, None]), 1))
        matrix -= lse[:, None]
    weighted = (pre.to(dtype).to(tl.float32)[:, None] * stream).to(dtype).to(tl.float32)
    value = tl.sum(weighted, 0)
    if NORM:  # the layer's input RMSNorm of the rounded read
        value = value.to(dtype).to(tl.float32)
        w = tl.load(NW + h, h < H, 0).to(tl.float32)
        if ZC:
            w += 1.0
        inv = tl.rsqrt(tl.sum(tl.where(h < H, value * value, 0.0), 0) / H + NEPS)
        value = w * (value * inv)
    tl.store(X + t * H + h, value.to(dtype), h < H)
    tl.store(POST + t * 4 + s, post)
    tl.store(RES + t * 16 + offsets, tl.exp(matrix))


def mhc_read(streams, phi, alpha_pre, alpha_post, alpha_res, b_pre, b_post, b_res, eps, iters, norm=None):
    """``[T, 4, h]`` streams -> (x ``[T, h]`` -- or ``norm(x)`` for ``norm = (weight, eps,
    zero_centered)`` --, (h_post ``[T, 4]``, h_res ``[T, 4, 4]``), both fp32)."""
    tokens, n, h = streams.shape
    assert n == 4 and streams.is_contiguous()
    coeffs = streams.view(tokens, n * h) @ phi.to(streams.dtype)
    x = torch.empty((tokens, h), device=streams.device, dtype=streams.dtype)
    post = torch.empty((tokens, n), device=streams.device, dtype=torch.float32)
    res = torch.empty((tokens, n, n), device=streams.device, dtype=torch.float32)
    nw, neps, zc = norm if norm is not None else (x, 0.0, False)
    if tokens:
        with _guard(streams):
            _mhc_read_kernel[(tokens,)](streams, coeffs, alpha_pre, alpha_post, alpha_res, b_pre, b_post, b_res,
                                        x, post, res, nw, h, eps, iters, triton.next_power_of_2(h),
                                        norm is not None, neps, zc, num_warps=8, enable_fp_fusion=False)
    return x, (post, res)


@triton.jit
def _mhc_write_kernel(S, OUT, POST, RES, Y, H: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    hm = h < H
    s = tl.arange(0, 4)
    dtype = S.dtype.element_ty
    base = S + t * 4 * H
    stream = tl.load(base + s[:, None] * H + h[None, :], hm[None, :], 0).to(tl.float32)
    out = tl.load(OUT + t * H + h, hm, 0).to(tl.float32)
    for j in tl.static_range(4):
        res = tl.load(RES + t * 16 + j * 4 + s).to(dtype).to(tl.float32)
        mixed = tl.sum((res[:, None] * stream).to(dtype).to(tl.float32), 0).to(dtype).to(tl.float32)
        post = tl.load(POST + t * 4 + j).to(dtype).to(tl.float32)
        weighted = (post * out).to(dtype).to(tl.float32)
        tl.store(Y + t * 4 * H + j * H + h, (mixed + weighted).to(dtype), hm)


def mhc_write(streams, out, pending):
    """``h_res @ streams + h_post (x) out`` for ``[T, 4, h]`` streams and a ``[T, h]`` output."""
    tokens, n, h = streams.shape
    assert n == 4 and streams.is_contiguous()
    out = out.contiguous()
    post, res = pending
    y = torch.empty_like(streams)
    if tokens:
        with _guard(streams):
            _mhc_write_kernel[(tokens, triton.cdiv(h, 512))](streams, out, post, res, y, h, 512,
                                                            num_warps=4, enable_fp_fusion=False)
    return y


# --------------------------------------------------------------------------- Engram
@triton.jit
def _engram_rows_kernel(SHIFTED, MULT, ORDERS, SIZES, OFFSETS, OUT, S, W, P,
                        L: tl.constexpr, MAXN: tl.constexpr, BL: tl.constexpr, BS: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    pos = tl.program_id(1) * BS + tl.arange(0, BS)
    pm = pos < S
    lk = tl.arange(0, BL)
    lok = lk < L
    n = tl.load(ORDERS + lk, mask=lok, other=0)
    hsh = tl.zeros([BS, BL], dtype=tl.int64)
    for i in tl.static_range(MAXN):
        # history[i][:, j] = shifted[:, p - i + j]
        tok = tl.load(SHIFTED + b * W + P - i + pos, mask=pm, other=0)
        m = tl.load(MULT + lk * MAXN + i, mask=lok & (i < n), other=0)
        hsh = hsh ^ (tok[:, None] * m[None, :])
    size = tl.load(SIZES + lk, mask=lok, other=1)
    r = hsh % size[None, :]
    r = tl.where(r < 0, r + size[None, :], r)  # torch.remainder: the sign of the divisor
    r = r + tl.load(OFFSETS + lk, mask=lok, other=0)[None, :]
    tl.store(OUT + (b * S + pos[:, None]) * L + lk[None, :], r, mask=pm[:, None] & lok[None, :])


class EngramHash:
    """One hash set of an Engram layer (its rows, or its sign rows) as device constants."""

    def __init__(self, orders_per_lookup, mult, sizes, offsets, device) -> None:
        self.L = len(orders_per_lookup)
        self.maxn = max(orders_per_lookup)
        m = torch.zeros(self.L, self.maxn, dtype=torch.int64)
        for j, row in enumerate(mult):
            used = [int(a) for a in row[: orders_per_lookup[j]]]
            m[j, : len(used)] = torch.tensor(used, dtype=torch.int64)
        self.mult = m.to(device)
        self.orders = torch.tensor(orders_per_lookup, dtype=torch.int32, device=device)
        self.sizes = torch.tensor(sizes, dtype=torch.int64, device=device)
        self.offsets = torch.tensor(offsets, dtype=torch.int64, device=device)

    def __call__(self, shifted: torch.Tensor, s: int) -> torch.Tensor:
        """``shifted`` [b, p + s] (p = max order - 1 previous ids in front) -> rows [b, s, L]."""
        b, w = shifted.shape
        p = w - s
        assert p >= self.maxn - 1, (w, s, self.maxn)
        shifted = shifted.contiguous()
        out = torch.empty((b, s, self.L), dtype=torch.int64, device=shifted.device)
        if b and s:
            bs = 64
            with _guard(shifted):
                _engram_rows_kernel[(b, triton.cdiv(s, bs))](shifted, self.mult, self.orders, self.sizes,
                                                             self.offsets, out, s, w, p, self.L, self.maxn,
                                                             triton.next_power_of_2(self.L), bs)
        return out


class _HostPtr:
    """A Triton pointer argument for registered host memory (UVA: the host address is the device
    address)."""

    def __init__(self, ptr: int, dtype: torch.dtype) -> None:
        self._ptr, self.dtype = ptr, dtype

    def data_ptr(self) -> int:
        return self._ptr


@triton.jit
def _gather_kernel(TABLE, ROWS, SIGN, SROWS, OUT, n, D: tl.constexpr, BLOCK: tl.constexpr, SIGNED: tl.constexpr):
    r = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    ok = r < n
    row = tl.load(ROWS + r, mask=ok, other=0).to(tl.int64)
    d = tl.arange(0, D)
    vals = tl.load(TABLE + row[:, None] * D + d[None, :], mask=ok[:, None], other=0)
    vals = vals.to(OUT.dtype.element_ty)
    if SIGNED:  # e * sign_table[sign row]: +-1, exact
        srow = tl.load(SROWS + r, mask=ok, other=0).to(tl.int64)
        sign = tl.load(SIGN + srow[:, None] * D + d[None, :], mask=ok[:, None], other=0)
        vals = vals * sign.to(OUT.dtype.element_ty)
    tl.store(OUT + r[:, None] * D + d[None, :], vals, mask=ok[:, None])


class UvaTable:
    """``rows [...] -> [..., dim]`` from a 2-D host table pinned in place (``cudaHostRegister``).

    The table keeps its storage (no copy); the registration pins its pages, so the GPU reads the
    needed rows over PCIe / C2C in one kernel, without a host round trip."""

    def __init__(self, table: torch.Tensor) -> None:
        assert table.dim() == 2 and table.device.type == "cpu" and table.is_contiguous(), table.shape
        self.table = table
        self.rows, self.dim = table.shape
        assert self.dim & (self.dim - 1) == 0, f"dim {self.dim}: not a power of 2"
        self.nbytes = table.numel() * table.element_size()
        self.registered = False
        if not table.is_pinned():
            err = torch.cuda.cudart().cudaHostRegister(table.data_ptr(), self.nbytes, 1 | 2)  # portable | mapped
            if int(err) != 0:
                raise RuntimeError(f"cudaHostRegister ({self.nbytes} bytes): error {err}")
            self.registered = True
        self._host = _HostPtr(table.data_ptr(), table.dtype)

    def release(self) -> None:
        if self.registered:
            torch.cuda.cudart().cudaHostUnregister(self.table.data_ptr())
            self.registered = False

    def __call__(self, rows: torch.Tensor, dtype: torch.dtype, sign=None) -> torch.Tensor:
        return gather(self._host, self.dim, rows, dtype, sign)


def gather(table, dim: int, rows: torch.Tensor, dtype: torch.dtype, sign=None) -> torch.Tensor:
    """``table[rows]`` (a CUDA tensor on the rows' device, or registered host memory) cast to
    ``dtype``, times ``sign`` = (sign table ``[n, dim]`` on the rows' device, sign rows like
    ``rows``) when given."""
    flat = rows.reshape(-1)
    out = torch.empty((flat.numel(), dim), dtype=dtype, device=rows.device)
    n = flat.numel()
    if n:
        block = 32
        st, sr = sign if sign is not None else (out, flat)
        with _guard(rows):
            _gather_kernel[(triton.cdiv(n, block),)](table, flat, st, sr.reshape(-1), out, n, dim, block,
                                                     sign is not None)
    return out.view(*rows.shape, dim)


# --------------------------------------------------------------------------- MoE router
@triton.jit
def _route_kernel(LOGITS, BIAS, W_OUT, I_OUT, scaling, E: tl.constexpr, K: tl.constexpr, BE: tl.constexpr,
                  BK: tl.constexpr, QUANTILE: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    e = tl.arange(0, BE)
    ok = e < E
    x = tl.load(LOGITS + t * E + e, mask=ok, other=0.0)
    s = 1.0 / (1.0 + tl.exp(-x))
    bb = tl.load(BIAS + e, mask=ok, other=0.0)
    sel = (x - bb) if QUANTILE else (s + bb)
    sel = tl.where(ok, sel, float("-inf"))
    kk = tl.arange(0, BK)
    wv = tl.zeros([BK], dtype=tl.float32)
    iv = tl.zeros([BK], dtype=tl.int32)
    for k in tl.static_range(K):
        best = tl.max(sel, axis=0)
        idx = tl.min(tl.where(sel == best, e, BE), axis=0)  # the lowest index among the maxima
        wk = tl.sum(tl.where(e == idx, s, 0.0), axis=0)
        wv = tl.where(kk == k, wk, wv)
        iv = tl.where(kk == k, idx.to(tl.int32), iv)
        sel = tl.where(e == idx, float("-inf"), sel)
    if K > 1:
        wv = wv / (tl.sum(wv, axis=0) + 1e-20)
    tl.store(W_OUT + t * K + kk, wv * scaling, mask=kk < K)
    tl.store(I_OUT + t * K + kk, iv, mask=kk < K)


def route(logits: torch.Tensor, bias: torch.Tensor, k: int, scaling: float, quantile: bool):
    """fp32 logits [T, E] -> (weights [T, k] fp32, expert ids [T, k] int32)."""
    t, e = logits.shape
    logits = logits.contiguous()
    w = torch.empty((t, k), dtype=torch.float32, device=logits.device)
    idx = torch.empty((t, k), dtype=torch.int32, device=logits.device)
    if t:
        with _guard(logits):
            _route_kernel[(t,)](logits, bias, w, idx, float(scaling), e, k, triton.next_power_of_2(e),
                                triton.next_power_of_2(k), quantile)
    return w, idx


# --------------------------------------------------------------------------- grouped expert GEMM
@triton.jit
def _decode_e4m3(b):
    """E4M3 bytes -> fp32 / 256 (exact): the 7 magnitude bits shifted into an fp16 bit pattern
    (5 exponent bits, bias 15 instead of 7) give the value times 2^-8, subnormals included."""
    v = b.to(tl.int16)
    bits = ((v & 0x80) << 8) | ((v & 0x7F) << 7)
    return bits.to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _grouped_gemm_kernel(A, A_ROWS, W, S, Y, BLOCK_EXPERT, BLOCK_ROW0, BLOCK_ROWEND,
                         N, K, stride_am, stride_we, stride_wn, stride_se, stride_sn, stride_ym,
                         GATHER: tl.constexpr, FP8: tl.constexpr, NATIVE: tl.constexpr,
                         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(BLOCK_EXPERT + pid_m)
    if e < 0:
        return
    e = e.to(tl.int64)
    r0 = tl.load(BLOCK_ROW0 + pid_m)
    rend = tl.load(BLOCK_ROWEND + pid_m)
    rows = r0 + tl.arange(0, BM)
    rmask = rows < rend
    if GATHER:
        src = tl.load(A_ROWS + rows, mask=rmask, other=0).to(tl.int64)
    else:
        src = rows.to(tl.int64)
    n = pid_n * BN + tl.arange(0, BN)
    nmask = n < N
    kk = tl.arange(0, BK)
    a_ptr = A + src[:, None] * stride_am + kk[None, :]
    w_ptr = W + e * stride_we + n[:, None].to(tl.int64) * stride_wn + kk[None, :]
    s_ptr = S + e * stride_se + n.to(tl.int64) * stride_sn
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for kb in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptr, mask=rmask[:, None], other=0.0)
        wb = tl.load(w_ptr, mask=nmask[:, None], other=0)
        if FP8:
            # Megatron's dequantization: q (fp32) * the tile's fp32 scale, rounded once to bf16
            sc = tl.load(s_ptr + kb, mask=nmask, other=0.0)
            if NATIVE:
                q = wb.to(tl.float8e4nv, bitcast=True).to(tl.float32)
                w = (q * sc[:, None]).to(tl.bfloat16)
            else:
                w = (_decode_e4m3(wb) * (sc * 256.0)[:, None]).to(tl.bfloat16)
        else:
            w = wb
        acc = tl.dot(a, tl.trans(w), acc)
        a_ptr += BK
        w_ptr += BK
    tl.store(Y + rows[:, None].to(tl.int64) * stride_ym + n[None, :], acc.to(Y.dtype.element_ty),
             mask=rmask[:, None] & nmask[None, :])


class ExpertSchedule:
    """The sorted rows of one MoE forward and its GEMM block map (device tensors, no sync).

    ``order`` sorts the flat (token, slot) pairs by expert (stable); row r of the grouped
    GEMMs is pair ``order[r]``, token ``order[r] // top_k``."""

    def __init__(self, idx: torch.Tensor, num_experts: int, block_m: int) -> None:
        t, k = idx.shape
        flat = idx.reshape(-1).to(torch.int64)
        m = flat.numel()
        dev = idx.device
        self.m, self.block_m, self.top_k = m, block_m, k
        self.order = torch.argsort(flat, stable=True)
        self.tokens = (self.order // k).to(torch.int32)
        counts = torch.zeros(num_experts, dtype=torch.int64, device=dev)
        counts.scatter_add_(0, flat, torch.ones_like(flat))
        nblk = (counts + block_m - 1) // block_m
        blk_end = torch.cumsum(nblk, 0)
        row_start = torch.cumsum(counts, 0) - counts
        grid = triton.cdiv(m, block_m) + num_experts
        b = torch.arange(grid, device=dev)
        e = torch.searchsorted(blk_end, b, right=True)
        valid = e < num_experts
        ec = e.clamp(max=num_experts - 1)
        r0 = row_start[ec] + (b - (blk_end[ec] - nblk[ec])) * block_m
        self.block_expert = torch.where(valid, ec, -1).to(torch.int32)
        self.block_row0 = r0.to(torch.int32)
        self.block_rowend = (row_start[ec] + counts[ec]).to(torch.int32)
        self.grid_m = grid
        # inverse permutation: the sorted row of pair f
        self.inv = torch.empty_like(self.order)
        self.inv[self.order] = torch.arange(m, device=dev)


def block_m_for(rows: int, num_experts: int) -> int:
    avg = rows / max(num_experts, 1)
    return 16 if avg < 24 else 32 if avg < 48 else 64 if avg < 160 else 128


def grouped_gemm(a: torch.Tensor, sched: ExpertSchedule, weight: torch.Tensor, scale: torch.Tensor | None,
                 gather: bool, out_dtype=torch.bfloat16) -> torch.Tensor:
    """Rows of ``a`` (``gather``: token rows ``sched.tokens``, else the sorted rows themselves) times
    their expert's ``weight[e]`` ``[N, K]`` (E4M3 bytes with ``scale`` ``[E, N, K / 128]``, or bf16)
    -> ``[sched.m, N]``."""
    a = a if a.stride(-1) == 1 else a.contiguous()
    e, n, k = weight.shape
    fp8 = scale is not None
    if fp8:
        assert weight.dtype == torch.uint8 and scale.shape[-1] * 128 == k, (weight.dtype, scale.shape, k)
    y = torch.empty((sched.m, n), dtype=out_dtype, device=a.device)
    if sched.m == 0:
        return y
    bm = sched.block_m
    bn = 128 if n >= 128 else 64
    bk = 128
    native = fp8 and torch.cuda.get_device_capability(a.device) >= (8, 9)
    s = scale if fp8 else weight
    with _guard(a):
        _grouped_gemm_kernel[(sched.grid_m, triton.cdiv(n, bn))](
            a, sched.tokens, weight, s, y, sched.block_expert, sched.block_row0, sched.block_rowend,
            n, k, a.stride(0), weight.stride(0), weight.stride(1),
            s.stride(0) if fp8 else 0, s.stride(1) if fp8 else 0, y.stride(0),
            gather, fp8, native, bm, bn, bk,
            num_warps=4 if bm <= 32 else 8, num_stages=3,
        )
    return y


# --------------------------------------------------------------------------- SwiGLU + post-act norm
@triton.jit
def _swiglu_kernel(GU, OUT, rows, gu_stride, o_stride, limit, eps, floor, D: tl.constexpr, BD: tl.constexpr,
                   ROWS: tl.constexpr, CLAMP: tl.constexpr, NORM: tl.constexpr, SMOOTH: tl.constexpr):
    r = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    d = tl.arange(0, BD)
    ok = (r[:, None] < rows) & (d[None, :] < D)
    g = tl.load(GU + r[:, None] * gu_stride + d[None, :], mask=ok, other=0.0)
    u = tl.load(GU + r[:, None] * gu_stride + D + d[None, :], mask=ok, other=0.0)
    if CLAMP:  # torch.clamp of the bf16 tensors: exact
        g = tl.minimum(g.to(tl.float32), limit).to(g.dtype)
        u = tl.minimum(tl.maximum(u.to(tl.float32), -limit), limit).to(u.dtype)
    gf = g.to(tl.float32)
    if NORM:
        act = gf / (1.0 + tl.exp(-gf)) * u.to(tl.float32)
        ms = tl.sum(tl.where(ok, act * act, 0.0), axis=1) / D
        if SMOOTH:
            scale = 1.0 / tl.sqrt(1.0 + ms / (floor * floor))
        else:
            scale = 1.0 / tl.sqrt(tl.maximum(ms + eps, floor * floor))
        act = act * scale[:, None]
    else:  # F.silu(gate) * up in the input dtype: two roundings
        act = (gf / (1.0 + tl.exp(-gf))).to(g.dtype).to(tl.float32) * u.to(tl.float32)
    tl.store(OUT + r[:, None] * o_stride + d[None, :], act.to(OUT.dtype.element_ty), mask=ok)


def swiglu_norm(gu: torch.Tensor, limit: float | None, norm) -> torch.Tensor:
    """``[M, 2D]`` (gate | up) -> ``[M, D]``: ``_swiglu`` of the modeling file."""
    m, d2 = gu.shape
    d = d2 // 2
    assert gu.stride(-1) == 1
    out = torch.empty((m, d), dtype=gu.dtype, device=gu.device)
    eps, floor, smooth = norm if norm is not None else (0.0, 0.0, False)
    if m:
        bd = triton.next_power_of_2(d)
        per = max(1, min(8, 2048 // bd))
        with _guard(gu):
            _swiglu_kernel[(triton.cdiv(m, per),)](gu, out, m, gu.stride(0), out.stride(0),
                                                   float(limit) if limit is not None else 0.0, float(eps),
                                                   float(floor), d, bd, per, limit is not None, norm is not None,
                                                   bool(smooth), num_warps=4)
    return out


# --------------------------------------------------------------------------- MoE combine
@triton.jit
def _combine_kernel(Y, INV, IDX, WT, SHARED, OUT, H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr,
                    BLOCK: tl.constexpr, HAS_SHARED: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    hm = h < H
    dtype = OUT.dtype.element_ty
    kk = tl.arange(0, BK)
    ids = tl.load(IDX + t * K + kk, mask=kk < K, other=2**30)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for _ in tl.static_range(K):  # slots in ascending expert order
        best = tl.min(ids, axis=0)
        slot = tl.min(tl.where(ids == best, kk, BK), axis=0)
        ids = tl.where(kk == slot, 2**30, ids)
        r = tl.load(INV + t * K + slot).to(tl.int64)
        w = tl.load(WT + t * K + slot).to(tl.float32)
        y = tl.load(Y + r * H + h, mask=hm, other=0.0).to(tl.float32)
        c = (y * w).to(dtype).to(tl.float32)
        acc = (acc + c).to(dtype).to(tl.float32)
    if HAS_SHARED:
        acc = acc + tl.load(SHARED + t * H + h, mask=hm, other=0.0).to(tl.float32)
    tl.store(OUT + t * H + h, acc.to(dtype), mask=hm)


def moe_combine(y: torch.Tensor, sched: ExpertSchedule, idx: torch.Tensor, weights: torch.Tensor,
                shared: torch.Tensor | None) -> torch.Tensor:
    """Expert outputs ``y`` [m, H] (sorted rows) -> [T, H]: per token sum_k bf16(y * w_k) in
    ascending expert order, rounded after every add, + ``shared``."""
    t, k = idx.shape
    h = y.shape[-1]
    out = torch.empty((t, h), dtype=y.dtype, device=y.device)
    if t:
        block = 1024
        sh = shared.contiguous() if shared is not None else out
        with _guard(y):
            _combine_kernel[(t, triton.cdiv(h, block))](y, sched.inv, idx.contiguous(), weights.contiguous(), sh, out,
                                                        h, k, triton.next_power_of_2(k), block, shared is not None,
                                                        num_warps=4)
    return out
