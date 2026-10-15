import tempfile
import torch
import pytest
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnBackend
from .conftest import Utils
from oellm_autoexp.hf_export.convert_megatron_to_hf import (
    build_hf_config,
    remap_state,
    cast_for_export,
)
from oellm_autoexp.hf_export.oellm_hybrid.configuration_oellm_hybrid import OellmHybridConfig
from oellm_autoexp.hf_export.oellm_hybrid.modeling_oellm_hybrid import OellmHybridForCausalLM

V, H = 512, 128
ENG = dict(
    engram_ngram_orders=[2, 3], engram_num_heads=2, engram_table_size=2000, engram_embed_dim=16
)


@pytest.fixture(scope="module")
def exported():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(1)
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
        **ENG,
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
            if hasattr(lyr, "table"):
                lyr.conv1d.weight.normal_(0, 0.3)
                lyr.table.normal_(0, 1.0)
    m = dict(
        hidden_size=H,
        ffn_hidden_size=256,
        num_attention_heads=2,
        seq_length=64,
        layernorm_epsilon=1e-5,
        hybrid_layer_pattern="N-*-N-",
        position_embedding_type="none",
        untie_embeddings_and_output_weights=True,
        **ENG,
    )
    state = {k: v.detach().cpu() for k, v in meg.state_dict().items() if "_extra_state" not in k}
    hcfg = build_hf_config(m, V)
    tensors = remap_state(state, hcfg)
    ids = torch.randint(0, V, (2, 40), device="cuda")
    with torch.no_grad():
        ref = meg(ids, position_ids=None, attention_mask=None).float()
    Utils.destroy_model_parallel()
    return hcfg, tensors, ids, ref


@pytest.mark.parametrize(
    "table,scale,tol",
    [("fp32", "fp32", 1e-4), ("bf16", "fp32", 0.05), ("fp8", "fp32", 0.3), ("fp8", "bf16", 0.3)],
)
def test_export_table_dtypes(exported, table, scale, tol):
    hcfg, tensors, ids, ref = exported
    cfg = {
        k: v
        for k, v in hcfg.items()
        if not k.startswith("_")
        and k not in ("architectures", "auto_map", "model_type", "torch_dtype")
    }
    cfg.update(engram_table_dtype=table, engram_scale_dtype=scale)
    out = cast_for_export(tensors, "fp32", table, scale)
    tkey = [k for k in out if k.endswith(".mixer.table")][0]
    assert (
        out[tkey].dtype
        == {"fp32": torch.float32, "bf16": torch.bfloat16, "fp8": torch.uint8}[table]
    )
    model = OellmHybridForCausalLM(OellmHybridConfig(**cfg)).cuda().eval()
    missing, unexpected = model.load_state_dict(out, strict=False)
    assert not unexpected and not [k for k in missing if "lm_head" not in k], (missing, unexpected)
    with (
        tempfile.TemporaryDirectory() as d
    ):  # meta-device init + a bf16 load must not touch the table
        model.save_pretrained(d)
        bf = OellmHybridForCausalLM.from_pretrained(d, dtype=torch.bfloat16).cuda().eval()
        mix = bf.model.layers[0].mixer
        assert (
            mix.table.dtype
            == {"fp32": torch.float32, "bf16": torch.bfloat16, "fp8": torch.uint8}[table]
        )
        if table == "fp8":
            assert mix.table_scale.dtype == {"fp32": torch.float32, "bf16": torch.bfloat16}[scale]
        model = OellmHybridForCausalLM.from_pretrained(d, dtype=torch.float32).cuda().eval()
    with torch.no_grad():
        logits = model(ids, use_cache=False).logits.float()
    err = (logits - ref).abs().max().item() / ref.abs().max().item()
    print(table, scale, "rel max err", err)
    assert err < tol
