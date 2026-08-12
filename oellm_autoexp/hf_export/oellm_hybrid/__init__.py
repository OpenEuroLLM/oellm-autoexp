"""Standalone HF architecture for the OpenEuroLLM hybrid scaling models.

The two modules in this package are copied verbatim into every exported
checkpoint directory (see ``convert_megatron_to_hf.py``) so the resulting model
dir loads with ``trust_remote_code=True`` and no dependency on this repository.
"""

from .configuration_oellm_hybrid import OellmHybridConfig
from .modeling_oellm_hybrid import (
    OellmHybridForCausalLM,
    OellmHybridModel,
    OellmHybridPreTrainedModel,
)

__all__ = [
    "OellmHybridConfig",
    "OellmHybridForCausalLM",
    "OellmHybridModel",
    "OellmHybridPreTrainedModel",
]
