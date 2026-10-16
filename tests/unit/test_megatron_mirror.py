"""The Megatron argument table is mirrored in three places here; keep them in step.

    config/backend/megatron/base_defaults.yaml        defaults, so a bare config composes
    oellm_autoexp/backends/megatron/cli_metadata.py   types, choices, actions
    oellm_autoexp/backends/megatron/config_schema.py  typed fields on MegatronConfig

A key missing from one of them, or a default that differs, is dropped or changed
before it reaches the command line -- silently. The linear-attention fields are
also checked against Megatron's own TransformerConfig (parsed from source, so
no torch is needed).
"""

from __future__ import annotations

import ast
import dataclasses
import typing
from pathlib import Path

import pytest
import yaml

from oellm_autoexp.backends.megatron.cli_metadata import (
    MEGATRON_ACTION_SPECS,
    MEGATRON_ARG_METADATA,
)
from oellm_autoexp.backends.megatron.config_schema import MegatronConfig

REPO = Path(__file__).resolve().parents[2]
REGENERATE = (
    "regenerate with `PYTHONPATH=.:submodules/Megatron-LM python "
    "scripts/generate_megatron_config.py` in an env with torch"
)
BASE_DEFAULTS = REPO / "config" / "backend" / "megatron" / "base_defaults.yaml"
TRANSFORMER_CONFIG = (
    REPO
    / "submodules"
    / "Megatron-LM"
    / "megatron"
    / "core"
    / "transformer"
    / "transformer_config.py"
)


def _schema_fields():
    return {f.name: f for f in dataclasses.fields(MegatronConfig)}


def _schema_default(field):
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        return field.default_factory()  # type: ignore[misc]
    return dataclasses.MISSING


def _norm(v):
    return list(v) if isinstance(v, tuple) else v


def test_every_argument_is_in_all_three_tables():
    fields = _schema_fields()
    defaults = yaml.safe_load(BASE_DEFAULTS.read_text())
    assert not [k for k in MEGATRON_ARG_METADATA if k not in fields], "missing from config_schema"
    assert not [k for k in MEGATRON_ARG_METADATA if k not in MEGATRON_ACTION_SPECS], (
        "missing from MEGATRON_ACTION_SPECS"
    )
    # base_defaults.yaml is GENERATED from the checked-out Megatron: regenerate
    # it rather than adding keys by hand (see REGENERATE below).
    missing_yaml = sorted(k for k in MEGATRON_ARG_METADATA if k not in defaults)
    assert not missing_yaml, f"{missing_yaml} missing from base_defaults.yaml; {REGENERATE}"
    extra_yaml = sorted(k for k in defaults if k not in MEGATRON_ARG_METADATA)
    assert not extra_yaml, f"{extra_yaml} in base_defaults.yaml but not in cli_metadata"


def test_defaults_agree():
    fields = _schema_fields()
    defaults = yaml.safe_load(BASE_DEFAULTS.read_text())
    bad = {}
    for key, meta in MEGATRON_ARG_METADATA.items():
        if key not in defaults:
            continue
        values = (
            _norm(meta.default),
            _norm(_schema_default(fields[key])),
            _norm(MEGATRON_ACTION_SPECS[key].default),
            _norm(defaults[key]),
        )
        if len({repr(v) for v in values}) > 1:
            bad[key] = values
    assert not bad, bad


def test_attention_variant_choices_agree():
    choices = set(MEGATRON_ARG_METADATA["experimental_attention_variant"].choices)
    hint = typing.get_type_hints(MegatronConfig)["experimental_attention_variant"]
    literal = {a for arg in typing.get_args(hint) for a in typing.get_args(arg)}
    assert choices == literal
    assert {"complex_kda", "signed_kda", "kda", "panini"} <= choices


def _literal_default(node):
    """A field's default if it is a literal, else MISSING (field(...), a function, ...)."""
    if node is None:
        return dataclasses.MISSING
    try:
        return ast.literal_eval(node)
    except ValueError:
        return dataclasses.MISSING


def _megatron_fields():
    """(literal default or MISSING, annotation source) of every TransformerConfig field."""
    tree = ast.parse(TRANSFORMER_CONFIG.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TransformerConfig"
    )
    out = {}
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out[node.target.id] = (_literal_default(node.value), ast.unparse(node.annotation))
    return out


@pytest.mark.skipif(not TRANSFORMER_CONFIG.exists(), reason="Megatron-LM submodule not checked out")
def test_linear_fields_match_megatron():
    """Every linear-attention knob Megatron declares is mirrored with its default."""
    megatron = {k: v for k, v in _megatron_fields().items() if k.startswith("linear_")}
    assert megatron, "no linear_* fields parsed from TransformerConfig"
    missing = sorted(k for k in megatron if k not in MEGATRON_ARG_METADATA)
    assert not missing, f"Megatron fields not mirrored here: {missing}"
    fields = _schema_fields()
    bad = {
        k: (default, _schema_default(fields[k]))
        for k, (default, _ann) in megatron.items()
        if default is not dataclasses.MISSING
        and _norm(default) != _norm(_schema_default(fields[k]))
    }
    assert not bad, bad


@pytest.mark.skipif(not TRANSFORMER_CONFIG.exists(), reason="Megatron-LM submodule not checked out")
def test_attention_variants_match_megatron():
    _default, annotation = _megatron_fields()["experimental_attention_variant"]
    literal = next(
        n
        for n in ast.walk(ast.parse(annotation, mode="eval"))
        if isinstance(n, ast.Subscript) and ast.unparse(n.value).endswith("Literal")
    )
    megatron = {ast.literal_eval(e) for e in ast.walk(literal.slice) if isinstance(e, ast.Constant)}
    assert megatron == set(MEGATRON_ARG_METADATA["experimental_attention_variant"].choices)


def _megatron_importable() -> bool:
    import importlib.util

    return TRANSFORMER_CONFIG.exists() and importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not _megatron_importable(), reason="needs torch and the Megatron-LM submodule")
def test_base_defaults_is_generated_from_checked_out_megatron(tmp_path):
    """base_defaults.yaml is exactly what the generator emits for the checked-out
    Megatron-LM -- never hand-edited, never stale."""
    import os
    import subprocess
    import sys

    out = tmp_path / "base_defaults.yaml"
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(REPO), str(REPO / "submodules" / "Megatron-LM")]),
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "generate_megatron_config.py"),
            "--output",
            str(out),
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0 and "Megatron-LM not available" in proc.stderr:
        pytest.skip("Megatron-LM does not import in this environment")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert out.read_text() == BASE_DEFAULTS.read_text(), (
        f"base_defaults.yaml is stale; {REGENERATE}"
    )


@pytest.mark.skipif(not _megatron_importable(), reason="needs torch and the Megatron-LM submodule")
def test_schema_and_cli_metadata_are_generated_from_checked_out_megatron(tmp_path):
    """config_schema.py and cli_metadata.py are exactly what
    scripts/generate_megatron_dataclass.py emits (ruff-formatted) for the checked-out
    Megatron-LM -- never hand-edited, never stale."""
    import os
    import shutil
    import subprocess
    import sys

    if shutil.which("ruff") is None and not (Path(sys.executable).parent / "ruff").exists():
        pytest.skip("ruff not installed")
    schema, cli = tmp_path / "config_schema.py", tmp_path / "cli_metadata.py"
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(REPO), str(REPO / "submodules" / "Megatron-LM")]),
        PATH=os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "")]),
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "generate_megatron_dataclass.py"),
            "--output",
            str(schema),
            "--cli-output",
            str(cli),
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0 and "Megatron-LM not available" in proc.stderr:
        pytest.skip("Megatron-LM does not import in this environment")
    assert proc.returncode == 0, proc.stderr[-2000:]
    package = REPO / "oellm_autoexp" / "backends" / "megatron"
    regenerate = (
        "regenerate with `PYTHONPATH=.:submodules/Megatron-LM python "
        "scripts/generate_megatron_dataclass.py` in an env with torch and ruff"
    )
    assert schema.read_text() == (package / "config_schema.py").read_text(), (
        f"config_schema.py is stale; {regenerate}"
    )
    assert cli.read_text() == (package / "cli_metadata.py").read_text(), (
        f"cli_metadata.py is stale; {regenerate}"
    )
