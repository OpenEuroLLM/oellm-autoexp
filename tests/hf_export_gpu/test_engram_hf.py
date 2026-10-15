import numpy as np
import torch
import pytest
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.engram import EngramLayer, engram_table_sizes, engram_hash_multipliers
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from .conftest import Utils
from oellm_autoexp.hf_export.convert_megatron_to_hf import _engram_geometry, build_hf_config
from oellm_autoexp.hf_export.oellm_hybrid.configuration_oellm_hybrid import OellmHybridConfig
from oellm_autoexp.hf_export.oellm_hybrid.modeling_oellm_hybrid import (
    OellmEngram,
    OellmHybridForCausalLM,
)

H, V = 64, 500


@pytest.fixture(autouse=True)
def mp():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(1)
    yield
    Utils.destroy_model_parallel()


def test_geometry_matches_megatron():
    m = dict(engram_ngram_orders=[2, 3, 4], engram_num_heads=3, engram_table_size=5000)
    cfg = TransformerConfig(num_layers=1, hidden_size=H, num_attention_heads=4, **m)
    sizes, mult = _engram_geometry(m)
    assert sizes == engram_table_sizes(cfg)
    assert mult == engram_hash_multipliers(cfg).tolist()


def test_layer_parity_and_decode(tmp_path):
    tmap = np.random.default_rng(0).integers(0, V, V).astype(np.int64)
    np.save(tmp_path / "m.npy", tmap)
    m = dict(
        engram_ngram_orders=[2, 3],
        engram_num_heads=2,
        engram_table_size=1000,
        engram_embed_dim=16,
        engram_token_map=str(tmp_path / "m.npy"),
    )
    cfg = TransformerConfig(
        num_layers=1, hidden_size=H, num_attention_heads=4, layernorm_epsilon=1e-5, **m
    )
    pg = ProcessGroupCollection(
        tp=parallel_state.get_tensor_model_parallel_group(),
        cp=parallel_state.get_context_parallel_group(),
    )
    meg = EngramLayer(cfg, layer_number=1, pg_collection=pg).cuda()
    with torch.no_grad():
        meg.conv1d.weight.normal_()
        meg.hidden_norm.weight.uniform_(0.5, 1.5)
    hcfg = build_hf_config(
        {
            "hybrid_layer_pattern": "N*-",
            "hidden_size": H,
            "ffn_hidden_size": 128,
            "num_attention_heads": 4,
            "seq_length": 64,
            "layernorm_epsilon": 1e-5,
            **m,
        },
        V,
    )
    hcfg = {k: v for k, v in hcfg.items() if not k.startswith("_")}
    config = OellmHybridConfig(
        **{k: v for k, v in hcfg.items() if k not in ("architectures", "auto_map", "model_type")}
    )
    hf = OellmEngram(config).cuda()
    rows = sum(config.engram_table_sizes)
    with torch.no_grad():
        hf.table.copy_(meg.table[:rows])
        for n in (
            "key_proj.weight",
            "value_proj.weight",
            "hidden_norm.weight",
            "key_norm.weight",
            "conv1d.weight",
        ):
            dict(hf.named_parameters())[n].copy_(dict(meg.named_parameters())[n])
        hf.token_map.copy_(torch.from_numpy(tmap))
    ids = torch.randint(0, V, (2, 40), device="cuda")
    h = torch.randn(40, 2, H, device="cuda")
    meg.set_input_ids(ids)
    ref = meg(h)  # [s, b, h]
    y, _ = hf(h.transpose(0, 1), ids)
    torch.testing.assert_close(h.transpose(0, 1) + y, ref.transpose(0, 1))
    # chunked / token-by-token with the cache == full
    hb = h.transpose(0, 1)
    state, outs = None, []
    for a, b in [(0, 17), *[(t, t + 1) for t in range(17, 40)]]:
        o, state = hf(hb[:, a:b], ids[:, a:b], None, state)
        outs.append(o)
    torch.testing.assert_close(torch.cat(outs, 1), y)
    # CPU-offloaded table gives the same output
    hf.offload_table("cpu")
    torch.testing.assert_close(hf(hb, ids)[0], y)


def test_full_model_cached_decode():
    m = dict(
        engram_ngram_orders=[2, 3], engram_num_heads=2, engram_table_size=1000, engram_embed_dim=16
    )
    hcfg = build_hf_config(
        {
            "hybrid_layer_pattern": "N-C-*-N*-",
            "hidden_size": H,
            "ffn_hidden_size": 128,
            "num_attention_heads": 4,
            "seq_length": 64,
            "layernorm_epsilon": 1e-5,
            **m,
        },
        V,
    )
    assert hcfg["mixer_types"] == [
        "engram",
        "short_conv",
        "full_attention",
        "engram",
        "full_attention",
    ]
    assert hcfg["mlp_layers"] == [True, True, True, False, True]
    config = OellmHybridConfig(
        **{
            k: v
            for k, v in hcfg.items()
            if not k.startswith("_") and k not in ("architectures", "auto_map", "model_type")
        }
    )
    torch.manual_seed(0)
    model = OellmHybridForCausalLM(config).cuda().float().eval()
    for lyr in model.model.layers:
        if lyr.is_engram:
            torch.nn.init.normal_(lyr.mixer.conv1d.weight)
    import tempfile

    with tempfile.TemporaryDirectory() as d:  # meta-device init, as from_pretrained loads it
        model.save_pretrained(d)
        model = OellmHybridForCausalLM.from_pretrained(d, torch_dtype=torch.float32).cuda().eval()
    ids = torch.randint(0, V, (2, 30), device="cuda")
    with torch.no_grad():
        full = model(ids, use_cache=False).logits
        out = model(ids[:, :11], use_cache=True)
        cache, logits = out.past_key_values, [out.logits]
        for t in range(11, 30):
            out = model(ids[:, t : t + 1], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            logits.append(out.logits)
    torch.testing.assert_close(torch.cat(logits, 1), full, atol=1e-4, rtol=1e-4)
