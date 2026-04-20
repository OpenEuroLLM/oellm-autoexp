from __future__ import annotations

import sys
from pathlib import Path
import types


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


try:
    import omegaconf  # noqa: F401
except ModuleNotFoundError:
    omega_module = types.ModuleType("omegaconf")

    class DictConfig(dict):
        pass

    class ListConfig(list):
        pass

    class OmegaConf:
        _resolvers: dict[str, object] = {}

        @staticmethod
        def create(value):
            if isinstance(value, dict):
                return DictConfig(value)
            return value

        @staticmethod
        def resolve(value):
            return value

        @staticmethod
        def to_container(value):
            if isinstance(value, DictConfig):
                return dict(value)
            return value

        @staticmethod
        def register_new_resolver(name, resolver):
            OmegaConf._resolvers[name] = resolver

        @staticmethod
        def to_yaml(value, *, sort_keys=True):
            if isinstance(value, DictConfig):
                data = dict(value)
            else:
                data = dict(value)
            items = sorted(data.items()) if sort_keys else data.items()
            lines = []
            for key, val in items:
                lines.append(f"{key}: {val}")
            return "\n".join(lines) + "\n"

    omega_module.OmegaConf = OmegaConf
    omega_module.DictConfig = DictConfig
    omega_module.ListConfig = ListConfig
    sys.modules["omegaconf"] = omega_module
