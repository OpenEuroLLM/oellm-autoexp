"""Hydra/OmegaConf resolvers - re-exported from hydra_staged_sweep.

This module now delegates to the well-tested resolvers in hydra_staged_sweep.
"""

from hydra_staged_sweep.config.resolvers import register_default_resolvers

__all__ = ["register_default_resolvers"]
