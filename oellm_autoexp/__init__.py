"""oellm_autoexp package."""

from __future__ import annotations

# hydra-fast (a reimplementation of Hydra's composition core, ~100x faster on sweeps,
# no ANTLR) takes over ``import omegaconf`` / ``import hydra`` when installed. Installed
# before any submodule imports either, so every module and every registered resolver
# sees the same OmegaConf. OELLM_HYDRA_FAST=0 keeps upstream hydra-core / omegaconf.
import os as _os

HYDRA_FAST = False
if _os.environ.get("OELLM_HYDRA_FAST", "1") != "0":
    try:
        import hydra_fast.compat.omegaconf_shim as _shim
    except ImportError:
        pass
    else:
        _shim.install()
        HYDRA_FAST = True

__all__ = ["__version__"]

__version__ = "0.1.0"
