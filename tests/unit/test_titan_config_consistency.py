"""Tests for consistency between the Titan config layers.

Catches three classes of bug:
1. Broken imports in titan_custom_config (e.g. importing from a non-existent module)
2. Schema fields rejected by ConfigManager's MergedModel at runtime
3. Field-name typos in update_from_config (e.g. moe_scale_before_experts vs moe_score_before_experts)
"""

from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure submodule paths are available for import.
# Tests are skipped if the submodules are absent (e.g. shallow clone).
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TITAN_ROOT = _REPO_ROOT / "submodules" / "titan-oellm"
_TORCHTITAN_ROOT = _TITAN_ROOT / "torchtitan"

for _p in (_REPO_ROOT, _TITAN_ROOT, _TORCHTITAN_ROOT):
    if str(_p) not in sys.path and _p.is_dir():
        sys.path.insert(0, str(_p))

_SUBMODULES_AVAILABLE = (_TITAN_ROOT / "titan_oellm").is_dir()


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def schema_model_fields() -> set[str]:
    """All field names declared in config_schema.py's model dataclass.

    Uses get_type_hints to resolve string annotations (from __future__
    annotations) so we get actual types, not strings.
    """
    import typing
    import oellm_autoexp.backends.titan.config_schema as schema_mod
    from oellm_autoexp.backends.titan.config_schema import JobConfig as SchemaJobConfig

    hints = typing.get_type_hints(SchemaJobConfig, localns=vars(schema_mod))
    model_type = hints["model"]
    return {f.name for f in dataclasses.fields(model_type)}


# ---------------------------------------------------------------------------
# Test 1: titan_custom_config is importable without errors
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _SUBMODULES_AVAILABLE, reason="titan-oellm submodule not checked out")
def test_titan_custom_config_importable():
    """titan_custom_config must import cleanly.

    Previously failed because it imported from
    titan_oellm.configs.sci_job_config which does not exist (the real
    module is oellm_job_config).
    """
    import importlib

    mod = importlib.import_module("oellm_autoexp.titan_custom_config")
    assert hasattr(mod, "JobConfig"), "titan_custom_config must expose JobConfig"
    assert hasattr(mod, "Model"), "titan_custom_config must expose Model"


@pytest.mark.skipif(not _SUBMODULES_AVAILABLE, reason="titan-oellm submodule not checked out")
def test_titan_custom_config_model_has_moe_fields():
    """Custom Model must expose the MoE fields needed by qwen3_custom model
    configs."""
    from oellm_autoexp.titan_custom_config import Model

    field_names = {f.name for f in dataclasses.fields(Model)}
    required = {
        "moe_enabled",
        "moe_inter_dim",
        "moe_num_experts",
        "moe_top_k",
        "moe_score_func",
        "moe_route_norm",
        "moe_route_scale",
        "moe_score_before_experts",
        "moe_num_shared_experts",
        "attn_gate_type",
        "attn_gate_input",
        "attn_gate_activation",
        "attn_gate_lowrank_dim",
        "attn_gate_bias",
    }
    missing = required - field_names
    assert not missing, f"titan_custom_config.Model is missing fields: {missing}"


# ---------------------------------------------------------------------------
# Test 2: ConfigManager MergedModel accepts all Hydra schema Model fields
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _SUBMODULES_AVAILABLE, reason="titan-oellm submodule not checked out")
def test_merged_model_accepts_all_schema_fields(schema_model_fields):
    """Every field in config_schema.Model must exist in ConfigManager's
    MergedModel.

    If a schema field is missing from MergedModel, torchtitan will raise:
        ValueError: Invalid field names in <MergedModel> data: {...}
    at the start of every training job.
    """
    from torchtitan.config.job_config import JobConfig as TorchTitanJobConfig
    from torchtitan.config.manager import ConfigManager
    from oellm_autoexp.titan_custom_config import JobConfig as CustomJobConfig

    merged_cls = ConfigManager._merge_configs(TorchTitanJobConfig, CustomJobConfig)
    merged_model_type = next(f.type for f in dataclasses.fields(merged_cls) if f.name == "model")
    merged_fields = {f.name for f in dataclasses.fields(merged_model_type)}

    missing = schema_model_fields - merged_fields
    assert not missing, (
        f"These config_schema.Model fields are NOT in ConfigManager's MergedModel and "
        f"will be rejected at training startup: {missing}"
    )


# ---------------------------------------------------------------------------
# Test 3: Field names used in update_from_config exist in the schema
# ---------------------------------------------------------------------------

_ARGS_PY = _TITAN_ROOT / "titan_oellm" / "models" / "qwen3_custom" / "model" / "args.py"


@pytest.mark.skipif(not _ARGS_PY.is_file(), reason="qwen3_custom args.py not found")
def test_update_from_config_field_names_match_schema(schema_model_fields):
    """All job_config.model field names read in update_from_config must exist
    in the schema.

    Previously broken: update_from_config used 'moe_scale_before_experts' but the
    schema field is 'moe_score_before_experts', so the value was silently never applied.

    Reads args.py as text to avoid triggering heavy optional dependencies
    (e.g. tomli_w) pulled in by titan_oellm.models.__init__.
    """
    src = _ARGS_PY.read_text()

    # Restrict to the update_from_config method body
    match = re.search(r"def update_from_config\(.*?(?=\ndef |\Z)", src, re.DOTALL)
    assert match, "Could not find update_from_config in args.py"
    method_src = match.group(0)

    # Extract field names from hasattr(job_config.model, "field") and job_config.model.field
    hasattr_fields = set(
        re.findall(r'hasattr\s*\(\s*job_config\.model\s*,\s*["\'](\w+)["\']', method_src)
    )
    direct_fields = set(re.findall(r"job_config\.model\.(\w+)", method_src))
    all_referenced = hasattr_fields | direct_fields

    unknown = all_referenced - schema_model_fields
    assert not unknown, (
        f"update_from_config references model fields not in config_schema.Model: {unknown}\n"
        f"These are silently ignored at runtime (hasattr returns False / AttributeError)."
    )


# ---------------------------------------------------------------------------
# Test 4: Real model YAML fields are accepted by the schema
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "yaml_rel",
    [
        "config/backend/titan/model/qwen3_moe_30BA3B.yaml",
    ],
)
def test_model_yaml_fields_in_schema(yaml_rel, schema_model_fields):
    """All keys in a backend model YAML must appear in config_schema.Model.

    Prevents adding new fields to YAML configs without updating the
    schema.
    """
    import yaml

    yaml_path = _REPO_ROOT / yaml_rel
    if not yaml_path.exists():
        pytest.skip(f"{yaml_rel} not found")

    with open(yaml_path) as fh:
        data = yaml.safe_load(fh)

    # Top-level keys in a model YAML map to model config fields
    yaml_fields = set(data.keys())
    missing = yaml_fields - schema_model_fields
    assert not missing, (
        f"{yaml_rel} contains fields not in config_schema.Model: {missing}\n"
        f"Run: PYTHONPATH=.:submodules/titan-oellm:submodules/titan-oellm/torchtitan "
        f"python scripts/generate_titan_dataclass.py --custom-module oellm_autoexp.titan_custom_config:JobConfig"
    )
