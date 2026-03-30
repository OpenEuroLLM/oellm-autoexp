"""Custom TorchTitan JobConfig with extra model sizing fields."""

from dataclasses import dataclass, field

from titan_oellm.configs.oellm_job_config import JobConfig as BaseJobConfig
from titan_oellm.configs.oellm_job_config import Model as BaseModel


@dataclass
class Model(BaseModel):
    """Extend Titan-OELLM Model config.

    oellm_job_config.Model already provides all MoE fields
    (moe_num_experts, moe_top_k, moe_score_func, moe_route_norm,
    moe_route_scale, moe_score_before_experts, moe_num_shared_experts)
    and attn_gate_* fields. Only add fields not present in the base.
    """

    ffn_dim_multiplier: float | None = None
    multiple_of: int = 256


@dataclass
class JobConfig(BaseJobConfig):
    model: Model = field(default_factory=Model)
