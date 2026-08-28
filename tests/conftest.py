import importlib
import sys
import types

import pytest

import oellm_autoexp._libs  # noqa: F401


def pytest_configure(config):
    """Fail fast, and legibly, when the tests are running against the wrong
    env.

    `uv run pytest` silently falls back to whatever `pytest` is first on PATH if
    the project venv has no pytest of its own. With a conda env active that
    means a DIFFERENT interpreter and a different, older dependency set — and
    the resulting failures point nowhere near the real cause.

    Observed 2026-08-28: `.venv` had no pytest, so `uv run pytest` used
    ~/.miniconda3/envs/oellm (python 3.12, compoconf 0.1.16) while pyproject
    requires compoconf>=0.2.2. compoconf < 0.2.1 defines a ``__reduce__`` that
    rebuilds configs from ``asdict(self)``, which flattens NESTED configs into
    plain dicts, so four sweep tests failed with things like
    "'dict' object has no attribute 'base_output_dir'" — four cryptic symptoms
    of one environment mismatch. Fix: `uv pip install --python .venv/bin/python
    -e ".[dev]"`.

    Checked by BEHAVIOUR rather than version string, so it stays true if the
    same regression ever reappears under a different version number.
    """
    import compoconf

    if "__reduce__" in compoconf.ConfigInterface.__dict__:
        try:
            from importlib.metadata import version

            found = version("compoconf")
        except Exception:  # pragma: no cover - diagnostics only
            found = "unknown"
        raise pytest.UsageError(
            f"compoconf {found} at {compoconf.__file__} defines a __reduce__ that "
            f"flattens nested configs on pickle; pyproject requires >=0.2.2.\n"
            f"Interpreter: {sys.executable} (python {sys.version.split()[0]}).\n"
            f"If that is not this project's .venv, pytest was resolved from PATH "
            f'(e.g. an active conda env). Fix with:\n  uv pip install --python .venv/bin/python -e ".[dev]"'
        )


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
