"""Custom TorchTitan/Titan-OELLM model overrides.

This module is imported via torchtitan's experimental.custom_import hook.
It registers additional gpt_plus flavors without modifying submodules.
"""

from titan_oellm.models import gpt_plus
from titan_oellm.models.gpt_plus.model.args import TransformerModelArgs

_FLAVOR = "llama1_7b_qkln"

if _FLAVOR not in gpt_plus.gpt_plus_configs:
    gpt_plus.gpt_plus_configs[_FLAVOR] = TransformerModelArgs(
        dim=2048,
        n_layers=24,
        n_heads=32,
        ffn_dim_multiplier=4,
        multiple_of=256,
        rope_theta=500000,
    )
