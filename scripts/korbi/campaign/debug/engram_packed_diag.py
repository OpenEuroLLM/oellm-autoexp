"""Packed (document-masked) vs separate documents through the hybrid with and without Engram:
where does the difference sit (per document, first tokens vs the rest), and how does it compare to
the leak without the document boundaries? On one GPU, from the repo root:

    torchrun --nproc-per-node 1 scripts/korbi/campaign/debug/engram_packed_diag.py
"""

import os
import sys

import torch

_MEGATRON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../submodules/Megatron-LM")
os.chdir(_MEGATRON)
sys.path.insert(0, _MEGATRON)
from megatron.core.packed_seq_params import PackedSeqParams  # noqa: E402
from tests.unit_tests.models import test_hybrid_packed_docs as T  # noqa: E402
from tests.unit_tests.test_utilities import Utils  # noqa: E402

ENGRAM = dict(engram_ngram_orders=[2, 3], engram_num_heads=2, engram_table_size=1000,
              engram_embed_dim=16, mhc=False, mhc_routed_topk=0)


def run_case(name, pattern, conv_std=None, fp32=False, **kw):
    Utils.initialize_model_parallel(1, 1)
    T.model_parallel_cuda_manual_seed(123)
    model = T._model(pattern=pattern, **kw)
    if fp32:
        model = model.float()
    model.train()
    if conv_std is not None:
        with torch.no_grad():
            model.decoder.layers[0].conv1d.weight.normal_(std=conv_std)
    torch.manual_seed(0)
    docs = [torch.randint(0, T.VOCAB, (1, n), device="cuda") for n in T.LENGTHS]

    def run(ids, packed=None):
        pos = torch.arange(ids.shape[1], device="cuda").unsqueeze(0)
        with torch.no_grad():
            return model(ids, position_ids=pos, attention_mask=None,
                         packed_seq_params=packed).float()

    ref = torch.cat([run(d) for d in docs], dim=1)
    cu = torch.tensor([0, *torch.tensor(T.LENGTHS).cumsum(0).tolist()], dtype=torch.int32,
                      device="cuda")
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu,
                             max_seqlen_q=max(T.LENGTHS), max_seqlen_kv=max(T.LENGTHS))
    together = run(torch.cat(docs, 1), packed)
    leaked = run(torch.cat(docs, 1))
    scale = ref.std()
    diff = (together - ref).abs().amax(-1)[0] / scale  # per position
    leak = (leaked - ref).abs().amax(-1)[0] / scale
    bounds = cu.tolist()
    parts = []
    for i in range(len(T.LENGTHS)):
        lo, hi = bounds[i], bounds[i + 1]
        parts.append(f"doc{i}: first8 {diff[lo:lo + 8].max():.3f} rest {diff[lo + 8:hi].max():.3f}"
                     f" (leak first8 {leak[lo:lo + 8].max():.3f})")
    print(f"{name:28s} max {diff.max():.3f} | " + " | ".join(parts), flush=True)
    Utils.destroy_model_parallel()


if __name__ == "__main__":
    run_case("no engram", T.PATTERN, **{k: v for k, v in ENGRAM.items() if k.startswith("mhc")})
    for tp in ("tp", "dp"):
        run_case(f"engram {tp}, conv 0", "NLELE*E", engram_table_parallel=tp, **ENGRAM)
        run_case(f"engram {tp}, conv 0.5", "NLELE*E", conv_std=0.5, engram_table_parallel=tp,
                 **ENGRAM)
    ldm = dict(mhc=False, mhc_routed_topk=0, depth_memory=True, depth_memory_groups=4,
               depth_memory_key_dim=8, depth_memory_hidden_dim=16)
    run_case("depth memory", T.PATTERN, **ldm)
    run_case("engram tp, conv 0.5, fp32", "NLELE*E", conv_std=0.5, fp32=True,
             engram_table_parallel="tp", **ENGRAM)
