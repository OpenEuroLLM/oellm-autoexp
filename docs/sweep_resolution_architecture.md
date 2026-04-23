# Sweep Resolution Architecture

## Overview

The sweep resolution system has been refactored to use a pure Hydra-based approach with DAG (Directed Acyclic Graph) ordering for multi-stage training workflows. This document describes the new architecture and key design decisions.

## Key Components

### 1. Sweep Expansion (`oellm_autoexp/sweep/expander.py`)

**Purpose**: Expand sweep configurations into individual sweep points.

**Key Concepts**:
- **SweepPoint**: Represents a single configuration point in the sweep
  - `index`: Unique integer identifier
  - `parameters`: Dictionary of parameter overrides (may contain escaped sibling references)
  - `group_path`: Tuple tracking position in sweep hierarchy (for example, `(0, 1, 0)`)

- **group_path**: Uniquely identifies each sweep point's position in the nested sweep structure
  - Used for sibling resolution based on hierarchy
  - Format: tuple of indices at each nesting level
  - Example: For a product of lr×stage, paths might be `(0, 0, 0)`, `(0, 0, 1)`, `(0, 1, 0)`, `(0, 1, 1)`

**Important Fix**: Product groups now include an index in the group_path for each parameter combination (line 173 in expander.py). This ensures that points differing in parameter values (not just stage) have unique group_paths.

### 2. DAG Resolution (`oellm_autoexp/sweep/dag_resolver.py`)

**Purpose**: Resolve sweep points using Hydra overrides and topological ordering.

**Main Function**: `resolve_sweep_with_dag(config, points, config_setup) -> list[JobPlan]`

**Resolution Process**:

1. **Build Dependency DAG**:
   - Extract sibling references from parameters (for example, `__ESCAPED_DOLLAR__{sibling.stable.output_dir}`)
   - Find sibling dependencies based on group_path hierarchy
   - Create directed edges from prerequisite jobs to dependent jobs

2. **Topological Sort**:
   - Order jobs so prerequisites are always resolved before dependents
   - Detect and reject circular dependencies

3. **Resolve Each Job** (in topological order):

   **With Hydra config** (`config_ref` provided):
   - Convert sibling metadata to Hydra cmdline overrides: `+sibling.STAGE.field=value`
   - Convert point parameters to Hydra overrides: `key=value`
   - Load full config by way of `load_config_reference(overrides=...)`
   - Extract JobPlan fields from resolved config

   **Fallback mode** (tests without Hydra):
   - Start with base config dict
   - Add sibling data for interpolation
   - Apply point parameters by way of nested dict traversal
   - Resolve sibling interpolations using OmegaConf (only for parameters with `${sibling.*}`)
   - Extract resolved parameter values for JobPlan

4. **Create JobPlan**:
   - Store parameters as list of `"key=value"` override strings (changed from dict)
   - Extract lifecycle fields (`start_conditions`, `cancel_conditions`, etc.)
   - Build output paths using resolved configuration

### 3. Sibling Resolution

**Hierarchy-Based Matching** (not value-based):
- Siblings share a common `group_path` prefix
- Differ in exactly one dimension (typically the stage dimension)
- Example:
  - Point `(0, 0, 1, 0)`: lr=0.0001, stage=stable
  - Point `(0, 0, 1, 1)`: lr=0.0001, stage=cooldown ← sibling of above
  - Point `(0, 1, 1, 0)`: lr=0.0005, stage=stable ← NOT a sibling (different lr)

**Sibling Data Structure**:
```python
sibling_data = {
    "stable": {
        "name": "job_name_stable",
        "output_dir": "/path/to/output",
        "log_path": "/path/to/log",
        "log_path_current": "/path/to/current.log",
        "stage": "stable"
    }
}
```

**Resolution**:
- Passed as Hydra overrides: `+sibling.stable.name=job_name_stable`, etc.
- Available in config as `${sibling.stable.output_dir}`, etc.
- Automatically resolved by Hydra/OmegaConf

### 4. JobPlan Structure (`oellm_autoexp/sweep/planner.py`)

**Key Change**: `parameters` field is now a **list of override strings**, not a dict.

**Format**:
```python
parameters = [
    "backend.megatron.lr=0.0001",
    "stage=cooldown",
    "backend.megatron.load=/path/to/checkpoint"
]
```

**Why**: Enables easier application as Hydra cmdline overrides in script rendering.

### 5. Validation (`oellm_autoexp/sweep/validator.py`)

**Updated** to handle both legacy dict format and new list format for parameters.

**Checks**:
- No unresolved sibling references (looks for `{sibling.` or `__ESCAPED_DOLLAR__`)
- No duplicate job names
- No circular dependencies
- Valid cancel_conditions/start_conditions structure

## Escaped Interpolation System

### Purpose
Defer resolution of certain interpolations (like sibling references) until the appropriate resolution stage.

### Marker Format
- Input (in YAML): `\\${sibling.stable.output_dir}`
- After YAML parsing: `\${sibling.stable.output_dir}`
- After escaping: `__ESCAPED_DOLLAR__{sibling.stable.output_dir}`
- Before resolution: `${sibling.stable.output_dir}` (unescaped)
- After resolution: `outputs/demo_stable` (actual value)

### Processing Flow

1. **Config Loading** (`oellm_autoexp/config/loader.py`):
   - Replace `\$` → `__ESCAPED_DOLLAR__`
   - Prevents premature OmegaConf resolution

2. **Sweep Expansion** (`expander.py`):
   - Markers preserved as literals in parameters

3. **DAG Resolution** (`dag_resolver.py`):
   - Unescape: `__ESCAPED_DOLLAR__` → `$`
   - Add sibling data to config context
   - Resolve using OmegaConf/Hydra

## Multi-Stage Workflow Example

```yaml
sweep:
  type: product
  groups:
    - type: product
      params:
        backend.megatron.lr: [1e-4, 5e-4]

    - type: list
      configs:
        - stage: stable

        - stage: cooldown
          backend.megatron.load: "\\${sibling.stable.output_dir}/checkpoint"
          job.start_conditions:
            - class_name: FileExistsCondition
              path: "\\${sibling.stable.output_dir}/checkpoint/done.txt"
```

**Resolution**:
1. Expands to 4 points: 2 lr × 2 stages
2. Builds DAG: `cooldown` jobs depend on `stable` jobs
3. Resolves in order:
   - `demo_lr0.0001_stable`
   - `demo_lr0.0005_stable`
   - `demo_lr0.0001_cooldown` (with stable sibling data)
   - `demo_lr0.0005_cooldown` (with stable sibling data)

## Design Principles

### 1. **Pure Hydra Overrides**
- All configuration by way of Hydra cmdline overrides
- No custom interpolation logic
- Enables sweeping over Hydra config groups (for example, `backend=[variant1, variant2]`)

### 2. **Hierarchy-Based Sibling Matching**
- Based on `group_path` structure, not parameter values
- More robust and predictable
- Works with any parameter names

### 3. **Avoid Config-Specific Code**
- Generic handling of parameters
- Only extract known lifecycle fields (start_conditions, etc.)
- Works with any backend/monitoring configuration

### 4. **Validation Before Parsing**
- Parse resolved config to dataclasses to ensure validity
- Catch configuration errors early

### 5. **Test Compatibility**
- Fallback mode for tests without full Hydra setup
- Same resolution logic, just manual override application

## Migration Notes

### Breaking Changes

1. **JobPlan.parameters format**:
   - **Old**: `{"backend.megatron.lr": "0.0001", "stage": "stable"}`
   - **New**: `["backend.megatron.lr=0.0001", "stage=stable"]`

2. **API changes**:
   - `build_job_plans()` removed (was in `planner.py`)
   - Use `resolve_sweep_with_dag()` instead (in `dag_resolver.py`)
   - Requires `ConfigSetup` parameter

3. **Import changes**:
   ```python
   # Old
   from oellm_autoexp.sweep.planner import build_job_plans

   # New
   from oellm_autoexp.sweep.dag_resolver import resolve_sweep_with_dag
   from oellm_autoexp.config.schema import ConfigSetup
   ```

### Backward Compatibility

- Legacy `grids` format still supported
- Validator handles both dict and list parameter formats
- Tests updated to use new API

## File Summary

| File | Purpose | Key Changes |
|------|---------|-------------|
| `sweep/expander.py` | Expand sweep configs | Fixed group_path generation for product groups |
| `sweep/dag_resolver.py` | Resolve with DAG ordering | Complete rewrite, pure Hydra approach |
| `sweep/planner.py` | JobPlan dataclass | Commented out old `build_job_plans()` |
| `sweep/validator.py` | Validate execution plans | Handle list parameter format |
| `config/schema.py` | Config dataclasses | No changes |
| `config/loader.py` | Load Hydra configs | Escape/unescape logic |

## Testing

All tests pass:
- `tests/unit/test_sweep.py`: 15 tests (basic sweep functionality)
- `tests/unit/test_multi_stage_integration.py`: 5 tests (multi-stage workflows)

## Future Enhancements

1. Support for more complex DAG structures (parallel stages, convergence)
2. Better error messages for sibling resolution failures
3. Optimization for large sweeps (lazy resolution)
4. Support for conditional sibling dependencies
