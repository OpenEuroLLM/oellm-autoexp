# Adding a New Backend

A **backend** is a program that oellm-autoexp launches — usually
by way of SLURM, optionally directly/local — driven by a typed config. The backend's
only job is to turn that typed config into a **launch command string**; the
orchestrator embeds that command in the SLURM/local script and handles
submission and monitoring.
If you just want to add an untyped bash command quickly, use the "NullBackend" as it
supports arbitrary bash commands - sacrificing the "check before submission" capability
of the "type compilation".

Existing backends live in `oellm_autoexp/backends/` and are good references:

- `megatron_backend.py` — Megatron-LM training (rich typed schema)
- `titan_backend.py` — TorchTitan training (emits a TOML config file)
- `oellm_eval_backend.py` — evaluation by way of `oellm-eval schedule` (compact, good template)
- `megatron_bridge_backend.py` — checkpoint conversion
- `base.py::NullBackend` — minimal echo backend for tests

## How it fits together

- `BackendInterface` (`config/schema.py`) is a `compoconf` registrable interface.
- A backend class is registered with the `@register` decorator. The YAML
  `class_name` field selects which registered backend/config is instantiated.
  Once you register this new Implementation (Subclass) of BackendInterface, its config
  can be automatically parsed and the class can be instantiated from the config class
- The orchestrator calls `backend = job.config.backend.instantiate(BackendInterface)`
  then `backend.build_launch_command()`, and drops the result into the script.

## Steps

### 1. Implement the backend

Create `oellm_autoexp/backends/<name>_backend.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field

from compoconf import MissingValue, NonStrictDataclass, register

from oellm_autoexp.backends.base import BaseBackend, BaseBackendConfig


@dataclass(init=False)
class MyBackendConfig(NonStrictDataclass, BaseBackendConfig):
    """Typed config for MyBackend. Every field becomes a YAML-overridable option."""

    class_name: str = "MyBackend"          # Enforced to be the main class name, necessary
    env: dict[str, str] = field(default_factory=dict)  # exported into the job

    # ---- your typed options ----
    script: str = "python -m my_module.train"
    some_flag: bool = False
    extra_cli_args: list[str] = field(default_factory=list)  # escape hatch

    # Optionally assemble the command once, overridable from YAML.
    full_cmd: str = MissingValue

    def __post_init__(self) -> None:
        # optionally: completion and/or config checks
        if self.full_cmd is MissingValue:
            parts = [self.script]
            if self.some_flag:
                parts.append("--some-flag")
            parts.extend(self.extra_cli_args)
            self.full_cmd = " ".join(parts)


@register
class MyBackend(BaseBackend):
    config: MyBackendConfig

    def validate(self) -> None:
        """Raise on invalid/inconsistent config. Called before submission."""
        if not self.config.script:
            raise ValueError("MyBackend: `script` is required.")

    def build_launch_command(self) -> str:
        """Return the command the SLURM/local script will run."""
        return self.config.full_cmd


__all__ = ["MyBackend", "MyBackendConfig"]
```

Notes:
- `NonStrictDataclass` + `@dataclass(init=False)` lets Hydra/compoconf populate
  fields and tolerates extra keys; `class_name` is the discriminator.
- For a schema with many typed fields, put it in a subpackage
  `backends/<name>/config_schema.py` (see `backends/megatron/`, `backends/titan/`).
- `env` is merged into the job environment; SLURM/nnodes/GPU values are available
  by way of OmegaConf interpolation in the YAML (for example `${slurm.sbatch.nodes}`).
- It is generally recommended to define a new backend yourself, but you can also just use
  "NullBackend" with a given command. This is for simple commands where you don't need
  typed information.


### 2. Register the module (two import sites)

Registration happens here by way of import side effects of `@register`. Add your module to both:

- `oellm_autoexp/config/loader.py` → the tuple in `_ensure_registrations()`
- `oellm_autoexp/orchestrator.py` → the `import ... # noqa - register` block

```python
"oellm_autoexp.backends.my_backend",     # loader.py tuple
import oellm_autoexp.backends.my_backend  # noqa  - register   # orchestrator.py
```

### 3. Add the default YAML

Create `config/backend/<name>.yaml` (selectable by way of `backend=<name>`):

```yaml
class_name: MyBackend
script: "python -m my_module.train"
some_flag: false
env:
  MY_VAR: "value"
  NNODES: "${slurm.sbatch.nodes}"
```

If your config uses a nested schema group, add a `defaults:` list plus `_self_`
(see `config/backend/oellm_eval.yaml` for the pattern).

### 4. Use it

Select the backend from an experiment config or on the CLI by way of override:

```
backend=<name>
```

### 5. Test

Add `tests/unit/test_<name>_backend.py`:

```python
def test_mybackend_builds_command():
    from oellm_autoexp.backends.my_backend import MyBackend, MyBackendConfig

    backend = MyBackend(MyBackendConfig())
    backend.validate()
    assert "my_module.train" in backend.build_launch_command()
```

## Checklist

- [ ] `oellm_autoexp/backends/<name>_backend.py` with `@register` class + config, implementing `validate()` and `build_launch_command()`
- [ ] `class_name` field matches the class name exactly
- [ ] Added to `_ensure_registrations()` in `config/loader.py`
- [ ] Added import in `orchestrator.py`
- [ ] `config/backend/<name>.yaml` default config
- [ ] Unit test under `tests/unit/`
