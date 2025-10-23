import importlib
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
