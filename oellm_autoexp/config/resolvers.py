"""Hydra/OmegaConf resolvers - re-exported from oellm_autoexp.hydra_staged_sweep.

All general-purpose resolvers live in hydra_staged_sweep.config.resolvers.
This module adds the Megatron-specific oc.pad_iter / oc.train_config resolvers
and ensures all resolvers are registered on import.
"""

import glob
import os

from omegaconf import OmegaConf

from oellm_autoexp.hydra_staged_sweep.config.resolvers import register_default_resolvers

register_default_resolvers()


def _pad_iter(step):
    """Format a checkpoint step as Megatron's zero-padded directory name."""
    return f"iter_{int(step):07d}"


def _resolve_train_config(training_dir, explicit: str = ""):
    """Resolve a Megatron training-config snapshot inside ``training_dir``.

    A finished training run drops its rendered config as ``config-<jobid>.yaml``
    plus a ``current.yaml`` symlink to the latest one. We only need the model
    architecture out of it (derive_hf_arch), and every snapshot of the same run
    carries the same architecture, so any ``config-*.yaml`` works.

    Resolution order:
      1. An explicit filename (e.g. ``config-42499800.yaml``), if given.
      2. ``current.yaml`` when it points at a file that actually exists
         (``os.path.exists`` follows symlinks, so a broken/foreign symlink — a
         common case when the target lived in another user's deleted space —
         falls through).
      3. The most recently modified ``config-*.yaml`` in the dir.

    Resolution happens at config-compose time on the submitting host, which can
    see ``training_dir``; the chosen absolute path is baked into the rendered
    config the container later reads.
    """
    training_dir = str(training_dir)

    if explicit:
        path = os.path.join(training_dir, explicit)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Explicit train_config not found: {path}")
        return path

    current = os.path.join(training_dir, "current.yaml")
    if os.path.exists(current):
        return current

    candidates = glob.glob(os.path.join(training_dir, "config-*.yaml"))
    if not candidates:
        raise FileNotFoundError(
            f"No training config in {training_dir}: 'current.yaml' is missing or a "
            "broken symlink and no 'config-*.yaml' snapshot was found."
        )
    return max(candidates, key=os.path.getmtime)


OmegaConf.register_new_resolver("oc.pad_iter", _pad_iter, replace=True)
OmegaConf.register_new_resolver(
    "oc.train_config", _resolve_train_config, replace=True, use_cache=False
)


__all__ = ["register_default_resolvers"]
