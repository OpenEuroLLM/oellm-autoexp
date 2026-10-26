"""Configuration of the conservative main run: Qwen3-Next plus the router it was trained with."""

from transformers import Qwen3NextConfig


class OellmQ35Config(Qwen3NextConfig):
    """``Qwen3NextConfig`` with Megatron's router: ``router_score_function`` "sigmoid" (Qwen3-Next:
    softmax), top-k selected on ``sigmoid(logits) + e_score_correction_bias`` (the aux-loss-free
    balancing bias), weights = the unbiased scores of the selected experts normalised to sum 1,
    times ``router_scaling_factor``."""

    model_type = "oellm_q35"

    def __init__(self, router_score_function: str = "sigmoid", router_scaling_factor: float = 1.0, **kwargs):
        self.router_score_function = router_score_function
        self.router_scaling_factor = router_scaling_factor
        super().__init__(**kwargs)
