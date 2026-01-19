"""Hydra/OmegaConf resolvers - re-exported from hydra_staged_sweep.

This module now delegates to the well-tested resolvers in hydra_staged_sweep.
"""

# Import directly from module since config subpackage may not have __init__.py
import sys
from pathlib import Path

# Add hydra_staged_sweep to path if not already there
_hydra_sweep_path = Path(__file__).parent.parent.parent / "hydra_staged_sweep" / "src"
if _hydra_sweep_path.exists() and str(_hydra_sweep_path) not in sys.path:
    sys.path.insert(0, str(_hydra_sweep_path))

try:
    from hydra_staged_sweep.config.resolvers import register_default_resolvers
except (ImportError, ModuleNotFoundError):
    # Fallback: import from module file directly
    import importlib.util

    _resolver_path = _hydra_sweep_path / "hydra_staged_sweep" / "config" / "resolvers.py"
    if _resolver_path.exists():
        spec = importlib.util.spec_from_file_location("_hydra_resolvers", _resolver_path)
        _module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_module)
        register_default_resolvers = _module.register_default_resolvers
    else:
        raise ImportError("Could not import resolvers from hydra_staged_sweep")

__all__ = ["register_default_resolvers"]
