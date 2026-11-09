"""Page-sparse core attention on a packed batch: one flex call per document vs one for the whole batch
(page_sparse_packed_flex), fwd + bwd, at a head dim (argv[1], default 128) and query heads (argv[2], default 10).
Run from submodules/Megatron-LM under torch.distributed.run with one process."""
import time, sys, torch
from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.experimental_attention_variant.page_sparse_attention import PageSparseCoreAttention
from tests.unit_tests.test_utilities import Utils

Utils.initialize_model_parallel(1, 1)
pg = ProcessGroupCollection(tp=parallel_state.get_tensor_model_parallel_group(), cp=parallel_state.get_context_parallel_group())
D = int(sys.argv[1]) if len(sys.argv) > 1 else 128
HQ, G = (int(sys.argv[2]) if len(sys.argv) > 2 else 10), 2
g = torch.Generator().manual_seed(0)
total = int(sys.argv[3]) if len(sys.argv) > 3 else 65536
lens = []
while sum(lens) < total:  # a mix of short and long documents, like a packed pretraining batch
    lens.append(int(min(total - sum(lens), max(64, torch.empty(1).exponential_(1 / 1500., generator=g).item()))))
lens = [l - l % 2 for l in lens if l >= 2]
t = sum(lens)
cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda")
psp = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, cu_seqlens_q_padded=cu, cu_seqlens_kv_padded=cu,
                      max_seqlen_q=max(lens), max_seqlen_kv=max(lens))
for sel in ("page_mean", "window"):
    cfg = TransformerConfig(num_layers=1, hidden_size=1280, num_attention_heads=HQ, num_query_groups=G, kv_channels=D,
                            page_sparse_selector=sel, page_sparse_page=64, page_sparse_topk=32, page_sparse_window=512,
                            page_sparse_select_block=64, page_sparse_flex_q_block=64, page_sparse_backend="flex")
    att = PageSparseCoreAttention(cfg, 1, pg_collection=pg)  # packed_flex toggled below
    q = torch.randn(t, HQ, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(t, G, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    for packed in (False, True):
        att.packed_flex = packed
        def step():
            att(q, k, v, packed_seq_params=psp).float().square().mean().backward()
        step(); step(); torch.cuda.synchronize(); t0 = time.time()
        for _ in range(3): step()
        torch.cuda.synchronize()
        print(f"{sel} docs={len(lens)} tokens={t} d={D} packed={packed}: {(time.time() - t0) / 3 * 1e3:.1f} ms fwd+bwd", flush=True)
Utils.destroy_model_parallel()
