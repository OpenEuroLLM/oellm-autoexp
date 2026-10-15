import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnBackend
from .conftest import Utils
from oellm_autoexp.hf_export.convert_megatron_to_hf import build_hf_config, remap_state
from oellm_autoexp.hf_export.oellm_hybrid.configuration_oellm_hybrid import OellmHybridConfig
from oellm_autoexp.hf_export.oellm_hybrid.modeling_oellm_hybrid import OellmHybridForCausalLM


pytestmark = pytest.mark.skipif(
    __import__("importlib").util.find_spec("fla") is None, reason="needs the ComplexKDA fla"
)

V, H = 128, 128
VAR = {
    "kda": dict(linear_fla_lower_bound=-5.0),
    "signed_kda": dict(
        linear_fla_lower_bound=-5.0,
        linear_fla_gate="signed_sigmoid2",
        linear_fla_output_gate="linear",
    ),
    "complex_kda": dict(
        linear_gate_activation="signed_sigmoid2",
        linear_beta_max=2.0,
        linear_gate_init_style="spread",
    ),
}


def mcfg_for(variant, **extra):
    return dict(
        hidden_size=H,
        ffn_hidden_size=256,
        num_attention_heads=2,
        seq_length=64,
        layernorm_epsilon=1e-5,
        experimental_attention_variant=variant,
        linear_key_head_dim=64,
        linear_num_key_heads=2,
        linear_conv_kernel_dim=4,
        position_embedding_type="none",
        untie_embeddings_and_output_weights=True,
        **VAR[variant],
        **extra,
    )


def tcfg(m, **kw):
    keep = {
        k: v
        for k, v in m.items()
        if k.startswith("linear_") or k == "experimental_attention_variant"
    }
    return TransformerConfig(
        num_layers=kw.pop("num_layers"),
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
        **keep,
        **kw,
    )


def check(meg, m, ids):
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
    missing, unexpected = hf.load_state_dict(tensors, strict=False)
    assert not unexpected and not [k for k in missing if "lm_head" not in k], (missing, unexpected)
    # Through from_pretrained (meta-device init), as the eval harness loads it.
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        hf.save_pretrained(d)
        hf = OellmHybridForCausalLM.from_pretrained(d).cuda().eval()
    with torch.no_grad():
        ref = meg(ids, position_ids=None, attention_mask=None).float()
        out = hf(ids, use_cache=False).logits.float()
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
        o = hf(ids[:, :20], use_cache=True)
        cache, logits = o.past_key_values, [o.logits]
        for t in range(20, ids.shape[1]):
            o = hf(ids[:, t : t + 1], past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            logits.append(o.logits)
        torch.testing.assert_close(torch.cat(logits, 1).float(), ref, atol=1e-4, rtol=1e-4)
    return hcfg


@pytest.fixture(autouse=True)
def mp():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(1)
    yield
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("variant", list(VAR))
def test_hybrid(variant):
    m = mcfg_for(variant, hybrid_layer_pattern="L-*-L-")
    meg = (
        HybridModel(
            config=tcfg(m, num_layers=6, is_hybrid_model=True),
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=V,
            max_sequence_length=64,
            hybrid_layer_pattern="L-*-L-",
            position_embedding_type="none",
            share_embeddings_and_output_weights=False,
        )
        .cuda()
        .eval()
    )
    hcfg = check(meg, m, torch.randint(0, V, (2, 48), device="cuda"))
    assert hcfg["mixer_types"] == ["fla", "full_attention", "fla"]


@pytest.mark.parametrize("variant", list(VAR))
def test_gpt(variant):
    extra = {"linear_hybrid_attention": "gated_nope"} if variant == "complex_kda" else {}
    m = mcfg_for(variant, num_layers=4, linear_attention_freq=2, **extra)
    c = tcfg(m, num_layers=4)
    spec = get_transformer_block_with_experimental_attention_variant_spec(config=c, vp_stage=None)
    meg = (
        GPTModel(
            config=c,
            transformer_layer_spec=spec,
            vocab_size=V,
            max_sequence_length=64,
            position_embedding_type="none",
            share_embeddings_and_output_weights=False,
        )
        .cuda()
        .eval()
    )
    hcfg = check(meg, m, torch.randint(0, V, (2, 48), device="cuda"))
    soft = "fla_attention" if extra else "full_attention"
    assert hcfg["mixer_types"] == ["fla", soft, "fla", soft]
