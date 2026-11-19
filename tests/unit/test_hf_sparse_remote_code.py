"""An export with sparse attention loads through transformers' remote-code path (trust_remote_code), i.e. the loader
copies sparse_attention.py / page_sparse.py / sparse_variants.py next to the model code, and the loaded model equals
the in-repo one -- in one forward and in generate() through the decode cache. 2026-11-19: ``from . import
page_sparse`` was not followed by the loader and every t16 export failed check_export with an ImportError."""
import json
import shutil
import sys

import torch

from oellm_autoexp.hf_export.oellm_hybrid import configuration_oellm_hybrid as _ch
from oellm_autoexp.hf_export.oellm_hybrid import modeling_oellm_hybrid as _mh

sys.modules.setdefault("oellm_autoexp.hf_export.oellm_ambitious.configuration_oellm_hybrid", _ch)
sys.modules.setdefault("oellm_autoexp.hf_export.oellm_ambitious.modeling_oellm_hybrid", _mh)

from oellm_autoexp.hf_export import export_ambitious as ea  # noqa: E402
from oellm_autoexp.hf_export.oellm_ambitious.configuration_oellm_ambitious import (  # noqa: E402
    OellmAmbitiousConfig,
)
from oellm_autoexp.hf_export.oellm_ambitious.modeling_oellm_ambitious import (  # noqa: E402
    OellmAmbitiousForCausalLM,
)

SPARSE = dict(page_sparse_selector="page_mean", page_sparse_per_group=True, page_sparse_page=8, page_sparse_topk=2,
              page_sparse_window=8, page_sparse_sink_pages=1, page_sparse_select_block=8)


def test_export_with_sparse_attention_loads_as_remote_code(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_MODULES_CACHE", str(tmp_path / "modules"))
    torch.manual_seed(0)
    cfg = OellmAmbitiousConfig(
        vocab_size=97, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, head_dim=8,
        num_key_value_heads=2, block_types=["attention", "attention"], residual="mhc", mhc_expansion_rate=4,
        rms_norm_eps=1e-5, qk_layernorm=True, attention_output_gate=True, exclusive_self_attention=True,
        sparse_attention=SPARSE,
        auto_map={"AutoConfig": "configuration_oellm_ambitious.OellmAmbitiousConfig",
                  "AutoModelForCausalLM": "modeling_oellm_ambitious.OellmAmbitiousForCausalLM"})
    model = OellmAmbitiousForCausalLM(cfg).eval()
    out = tmp_path / "export"
    model.save_pretrained(out)
    for f in ea.MODEL_CODE:  # as finish_export: the content, not the symlinks
        shutil.copyfile((ea._PKG / f).resolve(), out / f)
    for f in ("modeling_oellm_hybrid.py", "configuration_oellm_hybrid.py"):
        shutil.copyfile(ea._HERE / "oellm_hybrid" / f, out / f)
    assert json.loads((out / "config.json").read_text())["sparse_attention"]["page_sparse_selector"] == "page_mean"

    from transformers import AutoModelForCausalLM

    loaded = AutoModelForCausalLM.from_pretrained(out, trust_remote_code=True).eval()
    assert type(loaded).__module__.startswith("transformers_modules")
    assert loaded.model.layers[0].mixer.sparse is not None
    ids = torch.randint(0, 97, (1, 40), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        torch.testing.assert_close(loaded(ids).logits, model(ids).logits, atol=1e-5, rtol=1e-5)
        gen = loaded.generate(ids[:, :21], max_new_tokens=12, do_sample=False)
        # greedy decode through the cache = argmax of one-shot forwards of the growing sequence
        seq = ids[:, :21]
        for _ in range(12):
            seq = torch.cat([seq, model(seq).logits[:, -1:].argmax(-1)], 1)
    assert torch.equal(gen, seq)
