"""A reduced v4_2ph model for prefill profiling: the export's widths (hidden 2048, 360 FP8 experts,
GQA 16/2, CKDA, mHC-4, the sign-hashed Engram with its real multipliers), fewer layers, small Engram
tables, random weights of realistic scale.

The export's code is flattened into one package the way export_ambitious writes it (the
oellm_ambitious and oellm_hybrid files side by side), so the relative imports resolve as in a
real export directory:

    from tiny_model import build
    model, cfg = build(["attention", "moe", "engram", "ckda", "moe"])
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
HF = REPO / "oellm_autoexp" / "hf_export"
CONFIG = Path(__file__).resolve().parents[1] / "notes" / "config_v4_6000.json"
FULL_COUNTS = {"moe": 48, "ckda": 42, "attention": 6, "engram": 2}


def export_package(dest: Path) -> str:
    """Copy the export's model code into ``dest/amb013`` (fresh each call) and return the
    package name."""
    pkg = dest / "amb013"
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    for f in (HF / "oellm_ambitious").glob("*.py"):
        if f.name != "__init__.py":
            shutil.copy2(f, pkg / f.name)
    for name in ("modeling_oellm_hybrid.py", "configuration_oellm_hybrid.py"):
        shutil.copy2(HF / "oellm_hybrid" / name, pkg / name)
    if str(dest) not in sys.path:
        sys.path.insert(0, str(dest))
    for mod in [m for m in sys.modules if m == "amb013" or m.startswith("amb013.")]:
        del sys.modules[mod]
    return "amb013"


def reduced_config(block_types: list[str], table_rows: int = 200_003, **overrides):
    pkg = export_package(Path(__file__).resolve().parent / ".pkg")
    conf_mod = importlib.import_module(f"{pkg}.configuration_oellm_ambitious")
    raw = json.loads(CONFIG.read_text())
    for k in ("architectures", "auto_map", "model_type", "torch_dtype"):
        raw.pop(k, None)
    raw["block_types"] = block_types
    raw["engram_table_sizes"] = [table_rows + 2 * i for i in range(len(raw["engram_table_sizes"]))]
    # the hash multipliers of the export's first Engram layer (2), for every Engram index here
    eng = [i for i, t in enumerate(block_types) if t == "engram"]
    raw["engram_hash_multipliers"] = {str(i): raw["engram_hash_multipliers"]["2"] for i in eng}
    raw["engram_sign_multipliers"] = {str(i): raw["engram_sign_multipliers"]["2"] for i in eng}
    raw.update(overrides)
    return conf_mod.OellmAmbitiousConfig(**raw), pkg


def _fp8_quantize(w: torch.Tensor, block: int):
    """bf16 [..., n] -> (E4M3 bytes as uint8, fp32 scale per ``block`` elements)."""
    wf = w.float().view(*w.shape[:-1], -1, block)
    scale = wf.abs().amax(-1).clamp(min=1e-12) / 448.0
    q = (wf / scale[..., None]).to(torch.float8_e4m3fn).view(w.shape)
    return q.view(torch.uint8), scale


@torch.no_grad()
def randomize(model, seed: int = 0) -> None:
    g = torch.Generator(device="cpu").manual_seed(seed)

    def normal(t, std):
        t.copy_(torch.randn(t.shape, generator=g, dtype=torch.float32).mul_(std).to(t.dtype))

    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if cls == "OellmMHC":
            normal(mod.phi, 0.02)
            mod.alpha_pre.fill_(1.0), mod.alpha_post.fill_(1.0), mod.alpha_res.fill_(1.0)
            normal(mod.b_res, 1.0)
        elif cls == "OellmMoE":
            normal(mod.gate, 0.02)
            if hasattr(mod, "e_score_correction_bias"):
                normal(mod.e_score_correction_bias, 0.01)
            for wname, fan_in in (("gate_up_proj", mod.gate_up_proj.shape[-1]), ("down_proj", mod.down_proj.shape[-1])):
                if mod.fp8:
                    shape = getattr(mod, wname).shape
                    for e in range(shape[0]):
                        w = torch.randn(shape[1:], generator=g).mul_(fan_in**-0.5)
                        q, s = _fp8_quantize(w, mod.block)
                        getattr(mod, wname)[e].copy_(q)
                        getattr(mod, wname + "_scale")[e].copy_(s)
                else:
                    normal(getattr(mod, wname), fan_in**-0.5)
        elif cls == "OellmEngramV2":
            normal(mod.table, 1.0)
            mod.token_map.copy_(torch.randperm(mod.token_map.numel(), generator=g) % 200_000)
            if mod.sign_hash:
                mod.sign_table.copy_(torch.where(torch.rand(mod.sign_table.shape, generator=g) < 0.5, -1.0, 1.0))
            normal(mod.conv1d.weight, 0.3)
        elif cls == "OellmXSAAttention":
            pass
    # every remaining nn.Linear keeps its default init; the embedding and lm_head:
    normal(model.model.embed_tokens.weight, 1.0)
    normal(model.lm_head.weight, 0.02)


def build(block_types: list[str], device="cuda", dtype=torch.bfloat16, seed: int = 0, **overrides):
    cfg, pkg = reduced_config(block_types, **overrides)
    mod = importlib.import_module(f"{pkg}.modeling_oellm_ambitious")
    torch.manual_seed(seed)
    with torch.device("cpu"):
        model = mod.OellmAmbitiousForCausalLM(cfg).to(dtype)
    randomize(model, seed)
    if os.environ.get("OELLM_HF_MOE") == "grouped":  # 008's loader: expand FP8 experts before dispatch
        for m in model.modules():
            if hasattr(m, "prepare_grouped_experts"):
                m.prepare_grouped_experts()
    model = model.to(device).eval()
    return model, cfg
