import importlib
import sys
import types

import pytest

import oellm_autoexp._libs  # noqa: F401


@pytest.fixture(autouse=True)
def ensure_megatron_stub(monkeypatch):
    try:
        import megatron.training.arguments  # type: ignore  # noqa: F401
    except ImportError:
        module_megatron = types.ModuleType("megatron")
        module_training = types.ModuleType("megatron.training")
        module_arguments = types.ModuleType("megatron.training.arguments")

        def add_megatron_arguments(parser):
            parser.add_argument("--lr", type=float, default=0.01, dest="lr")
            parser.add_argument("--micro-batch-size", type=int, default=1, dest="micro_batch_size")
            return parser

        module_arguments.add_megatron_arguments = add_megatron_arguments
        module_training.arguments = module_arguments
        module_megatron.training = module_training

        monkeypatch.setitem(sys.modules, "megatron", module_megatron)
        monkeypatch.setitem(sys.modules, "megatron.training", module_training)
        monkeypatch.setitem(sys.modules, "megatron.training.arguments", module_arguments)

        import oellm_autoexp.backends.megatron_args as megatron_args

        importlib.reload(megatron_args)
        yield
        importlib.reload(megatron_args)
    else:
        yield


@pytest.fixture(autouse=True)
def patch_hydra_staged_sweep(monkeypatch):
    """Patch hydra_staged_sweep to fix list override syntax bug."""
    import oellm_autoexp.hydra_staged_sweep.dag_resolver as dag_resolver
    from omegaconf import DictConfig, ListConfig
    from collections.abc import Mapping, Sequence

    def patched_config_to_cmdline(cfg_dict, override="", prefix=""):
        # Re-implementation without the buggy list index line

        def dict_to_cmdlines(dct, prefix=""):
            lines = []
            if isinstance(dct, (dict, DictConfig, Mapping)):
                for sub_cfg in dct:
                    newprefix = (prefix + "." if prefix else "") + sub_cfg
                    lines += dict_to_cmdlines(dct[sub_cfg], prefix=newprefix)
            elif isinstance(dct, (list, ListConfig, Sequence)) and not isinstance(
                dct, (str, bytes)
            ):
                # FIXED: Do not emit key[0,1] syntax
                for n, sub_cfg in enumerate(dct):
                    lines += dict_to_cmdlines(
                        sub_cfg,
                        prefix=(prefix + "." if prefix else "") + str(n),
                    )
            elif dct is None:
                lines.append(override + prefix + "=null")
            else:
                val = str(dct)
                if isinstance(dct, str) and ("{" in dct or "(" in dct):
                    val = f"'{dct}'"
                lines.append(override + prefix + "=" + val)
            return lines

        return dict_to_cmdlines(cfg_dict, prefix=prefix)

    monkeypatch.setattr(dag_resolver, "config_to_cmdline", patched_config_to_cmdline)
    yield
