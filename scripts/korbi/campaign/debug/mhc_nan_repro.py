"""Where does dense mHC + MoE first produce a non-finite gradient? (round 4: NaN grad norm at
step ~30 with a finite loss, s1 shapes; not reproduced at hidden 256 locally.)

    python scripts/korbi/campaign/debug/mhc_nan_repro.py moe|dense LR [HIDDEN] [SEQ] [MBS]

Run from submodules/Megatron-LM (tests.unit_tests importable). MHC_TRITON_BACKWARD=0 switches
mHC to the torch backward."""
import sys
import torch
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnBackend
from tests.unit_tests.test_utilities import Utils

moe = sys.argv[1] == "moe"
H = int(sys.argv[3]) if len(sys.argv) > 3 else 256
SEQ = int(sys.argv[4]) if len(sys.argv) > 4 else 512
MBS = int(sys.argv[5]) if len(sys.argv) > 5 else 4
heads = H // 128 if H >= 512 else 4
Utils.initialize_model_parallel(1, 1)
model_parallel_cuda_manual_seed(123)
pattern = ("LELELELELE*E" * 2) if moe else ("L-L-L-L-L-*-" * 2)
cfg = dict(
    num_layers=len(pattern), hidden_size=H, num_attention_heads=heads, num_query_groups=2,
    kv_channels=128 if H >= 512 else 64, linear_key_head_dim=128 if H >= 512 else 64,
    linear_value_head_dim=128 if H >= 512 else 64, linear_num_key_heads=heads,
    linear_num_value_heads=heads, experimental_attention_variant="complex_kda", linear_beta_max=2.0,
    linear_output_gate="lowrank", linear_gate_init_style="shipped", linear_drop_qkv_silu=True,
    ffn_hidden_size=H * 8 // 3, attention_output_gate=True, exclusive_self_attention=True,
    qk_layernorm=True, normalization="RMSNorm", add_bias_linear=False, gated_linear_unit=True,
    activation_func=torch.nn.functional.silu, hidden_dropout=0.0, attention_dropout=0.0,
    attention_backend=AttnBackend.unfused if H < 512 else AttnBackend.auto, mhc=True, mhc_expansion_rate=4, bf16=True, params_dtype=torch.bfloat16, is_hybrid_model=True,
)
if moe:
    cfg.update(num_moe_experts=144 if H >= 512 else 16, moe_router_topk=3,
               moe_ffn_hidden_size=H * 2 // 3, moe_shared_expert_intermediate_size=H * 2 // 3, moe_grouped_gemm=True,
               moe_permute_fusion=True, moe_token_dispatcher_type="alltoall",
               moe_router_load_balancing_type="aux_loss", moe_aux_loss_coeff=1e-4,
               moe_router_score_function="sigmoid", moe_router_enable_expert_bias=True,
               moe_router_dtype="fp32")
model = HybridModel(config=TransformerConfig(**cfg), hybrid_stack_spec=hybrid_stack_spec,
                    vocab_size=1024, max_sequence_length=SEQ, hybrid_layer_pattern=pattern,
                    position_embedding_type="none",
                    share_embeddings_and_output_weights=False).cuda().bfloat16()
first = {}
for name, p in model.named_parameters():
    p.register_hook(lambda g, n=name: first.setdefault(n, None) if torch.isfinite(g).all() else first.update({"BAD:" + n: g.abs().max().item()}))
opt = torch.optim.AdamW(model.parameters(), lr=float(sys.argv[2]), weight_decay=0.1)
torch.manual_seed(0)
for step in range(200):
    ids = torch.randint(0, 1024, (MBS, SEQ + 1), device="cuda")
    logits = model(ids[:, :-1], position_ids=None, attention_mask=None)
    loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, 1024), ids[:, 1:].reshape(-1))
    opt.zero_grad()
    first.clear()
    loss.backward()
    bad = [k for k in first if k.startswith("BAD:")]
    gn = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in model.parameters() if p.grad is not None))
    if step % 20 == 0 or bad or not torch.isfinite(gn):
        print(step, f"loss {loss.item():.3f} gnorm {gn.item():.3g}", bad[:6], flush=True)
    if bad or not torch.isfinite(gn):
        break
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
