import torch
import pytest
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnBackend
from .conftest import Utils
from oellm_autoexp.hf_export.convert_megatron_to_hf import build_hf_config, remap_state
from oellm_autoexp.hf_export.oellm_hybrid.configuration_oellm_hybrid import OellmHybridConfig
from oellm_autoexp.hf_export.oellm_hybrid.modeling_oellm_hybrid import OellmHybridForCausalLM

V, H = 512, 128


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_engram_model(dtype):
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(1)
    eng = dict(
        engram_ngram_orders=[2, 3], engram_num_heads=2, engram_table_size=2000, engram_embed_dim=16
    )
    c = TransformerConfig(
        num_layers=6,
        hidden_size=H,
        num_attention_heads=2,
        ffn_hidden_size=256,
        normalization="RMSNorm",
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        params_dtype=torch.float32,
        attention_backend=AttnBackend.unfused,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        is_hybrid_model=True,
        **eng,
    )
    meg = (
        HybridModel(
            config=c,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=V,
            max_sequence_length=64,
            hybrid_layer_pattern="N-*-N-",
            position_embedding_type="none",
            share_embeddings_and_output_weights=False,
        )
        .cuda()
        .eval()
    )
    with torch.no_grad():
        for lyr in meg.decoder.layers:
            if hasattr(lyr, "conv1d") and hasattr(lyr, "table"):
                lyr.conv1d.weight.normal_(0, 0.3)
    m = dict(
        hidden_size=H,
        ffn_hidden_size=256,
        num_attention_heads=2,
        seq_length=64,
        layernorm_epsilon=1e-5,
        hybrid_layer_pattern="N-*-N-",
        position_embedding_type="none",
        untie_embeddings_and_output_weights=True,
        **eng,
    )
    state = {k: v.detach().cpu() for k, v in meg.state_dict().items() if "_extra_state" not in k}
    hcfg = build_hf_config(m, V)
    tensors = remap_state(state, hcfg)
    conf = OellmHybridConfig(
        **{
            k: v
            for k, v in hcfg.items()
            if not k.startswith("_") and k not in ("architectures", "auto_map", "model_type")
        }
    )
    hf = OellmHybridForCausalLM(conf).cuda().eval()
    hf.load_state_dict(tensors, strict=False)
    meg = meg.to(dtype)
    hf = hf.to(dtype)
    ids = torch.randint(0, V, (2, 48), device="cuda")
    with torch.no_grad():
        ref = meg(ids, position_ids=None, attention_mask=None).float()
        out = hf(ids, use_cache=False).logits.float()
        # per-layer
        hs = []
        for lyr in meg.decoder.layers:
            lyr.register_forward_hook(
                lambda mod, i, o: hs.append(o[0] if isinstance(o, tuple) else o)
            )
        meg(ids, position_ids=None, attention_mask=None)
        o = hf(ids, use_cache=False, output_hidden_states=True)
        for j, mi in enumerate([1, 3, 5]):
            if j + 1 < len(o.hidden_states) - 1:
                print(
                    dtype,
                    "after hf layer",
                    j,
                    (hs[mi].transpose(0, 1).float() - o.hidden_states[j + 1].float())
                    .abs()
                    .max()
                    .item(),
                )
        kl = torch.nn.functional.kl_div(
            torch.log_softmax(out, -1),
            torch.log_softmax(ref, -1),
            log_target=True,
            reduction="none",
        ).sum(-1)
        print(dtype, "max dlogit", (out - ref).abs().max().item(), "meanKL", kl.mean().item())
    Utils.destroy_model_parallel()
