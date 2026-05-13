"""Hydra/OmegaConf resolvers - re-exported from oellm_autoexp.hydra_staged_sweep.

All general-purpose resolvers live in hydra_staged_sweep.config.resolvers.
This module adds the Megatron-specific oc.pad_iter resolver and ensures
all resolvers are registered on import.
"""

from omegaconf import OmegaConf

from oellm_autoexp.hydra_staged_sweep.config.resolvers import register_default_resolvers

register_default_resolvers()


def _pad_iter(step):
    """Format a checkpoint step as Megatron's zero-padded directory name."""
    return f"iter_{int(step):07d}"


OmegaConf.register_new_resolver("oc.pad_iter", _pad_iter, replace=True)


__all__ = ["register_default_resolvers"]
