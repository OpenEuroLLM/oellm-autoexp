"""Hydra/OmegaConf resolvers - re-exported from oellm_autoexp.hydra_staged_sweep.

This module now delegates to the well-tested resolvers in hydra_staged_sweep.
"""

from oellm_autoexp.hydra_staged_sweep.config.resolvers import register_default_resolvers

__all__ = ["register_default_resolvers"]
