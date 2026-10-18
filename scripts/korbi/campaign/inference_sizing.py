"""Inference sizing of the campaign's hybrid MoE shapes: weights, KV / state, decode ceilings.

    python scripts/korbi/campaign/inference_sizing.py [--ctx 32768 131072 262144] [--quant q4_k_m]

Analytical counts (no Megatron run): per-layer parameters of the CKDA / attention mixers and the
MoE, summed over the 5:1 pattern; calibrated against Megatron's counts of the speed shapes
(m48a3 ~79B, A3B-30B ~30B total). Decode bytes per token = active weights at the quantization's
bits per weight + the KV read of the attention layers (dense, or top-k sparse after an NSA / DSA
stage) + the recurrent state read / write; the decode ceiling is bandwidth / bytes (single
stream, no speculative decoding). Real decode is ~55-75 % of the ceiling on GPUs and unified
memory (kernel efficiency, MoE routing, sampling); docs/inference_model_size.md has the
measured ratios.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field


@dataclass
class Shape:
    name: str
    hidden: int = 2048
    layers: int = 48  # mixer layers; each is followed by a MoE layer
    attn_every: int = 6  # 5:1 linear : attention
    q_heads: int = 16
    kv_groups: int = 2
    head_dim: int = 128
    lin_heads: int = 16  # q / k heads of the linear mixer
    lin_v_heads: int = 0  # 0: = lin_heads (GDN in Qwen3.5 / Qwen3-Next: 2x the q / k heads)
    lin_dk: int = 128
    lin_dv: int = 128
    experts: int = 128
    topk: int = 8  # routed
    shared: int = 1  # shared experts of the expert width
    expert_ffn: int = 768
    vocab: int = 262144
    tied: bool = False
    mtp: int = 0  # MTP modules (one mixer + MoE block each), not counted as active for decode
    latent: int = 0  # LatentMoE latent size (0: off)
    note: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def n_attn(self) -> int:
        return self.layers // self.attn_every

    @property
    def n_lin(self) -> int:
        return self.layers - self.n_attn

    def attn_params(self) -> int:
        h, q, kv = self.hidden, self.q_heads * self.head_dim, self.kv_groups * self.head_dim
        # q + output gate (per q channel), k, v, o
        return h * q * 2 + h * kv * 2 + q * h

    @property
    def v_heads(self) -> int:
        return self.lin_v_heads or self.lin_heads

    def lin_params(self) -> int:
        h, k, v = self.hidden, self.lin_heads * self.lin_dk, self.v_heads * self.lin_dv
        lowrank = self.lin_dv
        qkv = h * (2 * k + v)
        alpha = h * lowrank + lowrank * k  # low-rank decay gate
        gate = h * lowrank + lowrank * v  # low-rank output gate
        beta = h * self.lin_heads
        conv = 4 * (2 * k + v)
        return qkv + alpha + gate + beta + conv + v * h

    def expert_params(self) -> int:
        w = self.latent or self.hidden
        return 3 * w * self.expert_ffn

    def moe_dense_part(self) -> int:
        """Per MoE layer: router + shared expert(s) + latent projections."""
        lat = 2 * self.hidden * self.latent if self.latent else 0
        return self.hidden * self.experts + self.shared * 3 * self.hidden * self.expert_ffn + lat

    def counts(self) -> dict[str, float]:
        mix = self.n_attn * self.attn_params() + self.n_lin * self.lin_params()
        experts = self.layers * self.experts * self.expert_params()
        dense_moe = self.layers * self.moe_dense_part()
        emb = self.vocab * self.hidden
        head = 0 if self.tied else emb
        block = (mix + experts + dense_moe) / self.layers
        mtp = self.mtp * block
        total = mix + experts + dense_moe + emb + head + mtp
        active_experts = self.layers * self.topk * self.expert_params()
        # decode reads: mixers, routed top-k, router / shared / latent, LM head (input row ~0)
        active = mix + active_experts + dense_moe + (emb if self.tied else head)
        return dict(total=total, active=active, experts=experts, emb=emb + head, mtp=mtp,
                    non_expert=total - experts - mtp)

    def kv_bytes_per_token(self, kv_bytes: float = 2.0) -> float:
        return self.n_attn * 2 * self.kv_groups * self.head_dim * kv_bytes

    def state_bytes(self, state_bytes: float = 4.0) -> float:
        """Recurrent state per sequence (fp32), conv state included."""
        per = self.v_heads * self.lin_dk * self.lin_dv * state_bytes
        conv = 3 * (2 * self.lin_heads * self.lin_dk + self.v_heads * self.lin_dv) * 2
        return self.n_lin * (per + conv)


# Weight formats: effective bits per weight incl. scales (llama.cpp k-quants as reported by
# llama.cpp for typical MoE mixes; MXFP4 = 4 + 8/32; NVFP4 = 4 + 8/16).
BPW = {"bf16": 16.0, "fp8": 8.25, "q8_0": 8.5, "q6_k": 6.56, "q5_k_m": 5.7, "q4_k_m": 4.85,
       "nvfp4": 4.5, "mxfp4": 4.25, "q3_k_m": 3.9, "iq2_m": 2.7}

# Usable memory for weights + cache (GB, after OS / runtime / activations) and bandwidth (GB/s).
HW = {
    "RTX 3090/4090 24GB": (22, 936),
    "RTX 5090 32GB": (30, 1792),
    "DGX Spark 64GB": (55, 273),
    "2x RTX 5090 64GB": (60, 1792),  # TP2 / layer split: per-device bandwidth, each reads half
    "RTX PRO 6000 96GB": (92, 1792),
    "Strix Halo 128GB": (110, 212),  # measured, not the 256 nominal
    "DGX Spark 128GB": (115, 273),
    "M5 Max 128GB": (110, 614),
    "M5 Ultra 256GB": (235, 1200),
    "2x DGX Spark 256GB": (230, 273),
    "DGX Station HBM 252GB": (240, 7100),  # weights in HBM; the 496 GB LPDDR5X is extra
}

SHAPES = [
    # the Qwen-architecture baselines (GDN 3:1, gated attention 2 x 256 KV, 151k / 248k vocab)
    Shape("qwen35_35b_a3b", layers=40, attn_every=4, q_heads=16, head_dim=256, lin_heads=16,
          lin_v_heads=32, experts=256, topk=8, shared=1, expert_ffn=512, vocab=248320, mtp=1,
          note="Qwen3.5/3.6-35B-A3B arch"),
    Shape("qwen3next_80b_a3b", layers=48, attn_every=4, q_heads=16, head_dim=256, lin_heads=16,
          lin_v_heads=32, experts=512, topk=10, shared=1, expert_ffn=512, vocab=151936, mtp=1,
          note="Qwen3-Next-80B-A3B arch (= campaign A3B-80B speed shape)"),
    # ours: CKDA 5:1, attention 16 x 128 q / 2 x 128 KV, 262k untied vocab, hidden 2048 x 48
    Shape("ours_30b_a3b", experts=128, topk=8, shared=0, note="A3B-30B speed shape (128 x 768 top-8)"),
    # Qwen3.5-35B-A3B-matched: active FFN 6 x 768 = Qwen's 9 x 512 per layer, equal non-head active
    Shape("ours_e144k5", experts=144, topk=5, shared=1, mtp=1, note="equal total to Qwen3.5-35B (29x)"),
    Shape("ours_e160k5", experts=160, topk=5, shared=1, mtp=1,
          note="recommended: 39B-A2.7B, 24 GB at 4.25 bpw, 32 GB at Q5"),
    Shape("ours_e192k5", experts=192, topk=5, shared=1, mtp=1, note="47B-A2.7B: 32 GB at Q4_K_M"),
    Shape("ours_e192k4", experts=192, topk=4, shared=1, mtp=1, note="47B-A2.5B: active -9 %"),
    Shape("ours_m48a2", experts=192, topk=4, shared=1, expert_ffn=512, note="r4b m48a2"),
    Shape("ours_m32a3", experts=224, topk=7, shared=1, note="r4b m32a3"),
    Shape("ours_m48a3", experts=336, topk=7, shared=1, note="r4b m48a3"),
    Shape("ours_m64a3", experts=448, topk=7, shared=1, note="64x: fills 128 GB at 6 bit / 96 GB at 4 bit"),
    Shape("ours_a5b_120b", hidden=2560, q_heads=20, lin_heads=20, experts=384, topk=8, shared=1,
          expert_ffn=896, note="gpt-oss-120b / Mistral-Small-4 class, hidden 2560"),
    Shape("ours_a10b_230b", hidden=3072, q_heads=24, lin_heads=24, layers=60, experts=384, topk=8,
          shared=1, expert_ffn=1024, note="MiniMax-M2.5 class (2x Spark / M5 Ultra 256)"),
]


def fmt(x: float) -> str:
    return f"{x / 1e9:.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ctx", type=int, nargs="*", default=[32768, 131072, 262144])
    ap.add_argument("--sparse-topk", type=int, default=2048, help="NSA / DSA selected tokens")
    ap.add_argument("--quant", default="q4_k_m")
    args = ap.parse_args()

    print("| shape | total B | active B | non-expert B | emb+head B | KV KB/tok (bf16) | state MB/seq | note |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    for s in SHAPES:
        c = s.counts()
        print(f"| {s.name} | {fmt(c['total'])} | {fmt(c['active'])} | {fmt(c['non_expert'])} | "
              f"{fmt(c['emb'])} | {s.kv_bytes_per_token() / 1024:.0f} | {s.state_bytes() / 2**20:.0f} | {s.note} |")

    print("\nWeights in GB by format (total params; MTP included):\n")
    print("| shape | " + " | ".join(BPW) + " |")
    print("|---|" + "---:|" * len(BPW))
    for s in SHAPES:
        t = s.counts()["total"]
        print(f"| {s.name} | " + " | ".join(f"{t * b / 8 / 1e9:.0f}" for b in BPW.values()) + " |")

    print(f"\nDecode bytes / token at {args.quant} (GB): weights + KV read (dense / sparse top-{args.sparse_topk}, fp8 KV)\n")
    print("| shape | weights | " + " | ".join(f"KV dense {c // 1024}k" for c in args.ctx)
          + f" | KV sparse {args.ctx[1] // 1024}k (DSA, incl. indexer) | state r+w |")
    print("|---|---:|" + "---:|" * (len(args.ctx) + 2))
    bpw = BPW[args.quant] / 8
    for s in SHAPES:
        c = s.counts()
        # experts at the quant, attention / head at 8 bit (common practice in k-quant mixes)
        w = (c["active"] - s.counts()["emb"] / 2) * bpw + s.counts()["emb"] / 2 * 8.5 / 8
        kv = s.kv_bytes_per_token(1.0)
        # DSA-style: top-k tokens + an fp8 128-dim indexer key per token and layer (132 B with scale)
        sparse = kv * args.sparse_topk + s.n_attn * 132 * args.ctx[1]
        print(f"| {s.name} | {w / 1e9:.2f} | " + " | ".join(f"{kv * x / 1e9:.2f}" for x in args.ctx)
              + f" | {sparse / 1e9:.3f} | {2 * s.state_bytes() / 1e9:.3f} |")

    print(f"\nDecode ceiling tok/s (single stream, {args.quant}, fp8 KV, ctx {args.ctx[0] // 1024}k "
          f"/ {args.ctx[-1] // 1024}k dense; '-' = does not fit with {args.ctx[-1] // 1024}k ctx)\n")
    print("| shape | " + " | ".join(HW) + " |")
    print("|---|" + "---:|" * len(HW))
    for s in SHAPES:
        c = s.counts()
        w_total = c["total"] * bpw
        w = (c["active"] - c["emb"] / 2) * bpw + c["emb"] / 2 * 8.5 / 8
        kv = s.kv_bytes_per_token(1.0)
        row = []
        for name, (mem, bw) in HW.items():
            need = (w_total + kv * args.ctx[-1] + s.state_bytes()) / 1e9
            if need > mem:
                row.append("-")
                continue
            n_dev = 2 if name.startswith("2x") else 1  # ideal TP2 split
            st = 2 * s.state_bytes()  # read + write per step
            lo = bw * 1e9 * n_dev / (w + kv * args.ctx[-1] + st)
            hi = bw * 1e9 * n_dev / (w + kv * args.ctx[0] + st)
            row.append(f"{hi:.0f} / {lo:.0f}")
        print(f"| {s.name} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
