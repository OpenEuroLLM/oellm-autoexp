"""Custom TorchTitan JobConfig with extra model sizing fields."""

from dataclasses import dataclass, field

from titan_oellm.configs.sci_job_config import JobConfig as BaseJobConfig
from titan_oellm.configs.sci_job_config import Model as BaseModel


@dataclass
class Model(BaseModel):
    """Extend Titan-OELLM Model config with explicit sizing fields."""

    num_layers: int = 24
    num_attention_heads: int = 32
    ffn_hidden_size: int = 8192
    n_kv_heads: int | None = None
    ffn_dim_multiplier: float | None = None
    multiple_of: int = 256


@dataclass
class JobConfig(BaseJobConfig):
    model: Model = field(default_factory=Model)
