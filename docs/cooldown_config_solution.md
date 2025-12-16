# Cooldown Configuration Solution Design

This document addresses the technical challenges for implementing proper cooldown support with automatic resume and flexible configuration management.

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Current State Analysis](#current-state-analysis)
3. [Solution 1: Config Storage](#solution-1-config-storage)
4. [Solution 2: Load vs Save Path Management](#solution-2-load-vs-save-path-management)
5. [Solution 3: Implementation Plan](#solution-3-implementation-plan)

---

## Problem Statement

### Requirements

1. **Config Storage**: Need both resolved and unresolved configs stored as YAML for flexibility and reproducibility
2. **Multiple Cooldowns**: Need to support different load/save paths when triggering cooldown runs
   - Load path: based on original run's resolved output directory (stable)
   - Save path: based on new cooldown run's output directory (new timestamp)
3. **Platform Detection**: Manual override is acceptable (no automatic detection needed)

### Current Gaps

1. No unresolved YAML config is saved (only JSON with resolved values and reference metadata)
2. If we use unresolved config and override `base_output_dir`, both `load` and `save` paths update
3. Need a way to pin `load` to original location while allowing `save` to use new location

---

## Current State Analysis

### What's Currently Saved

When `run_autoexp.py` executes, it writes to each job's `{output_dir}/provenance/`:

1. **`resolved_config.json`**: Fully resolved configuration (all OmegaConf interpolations substituted)
   ```json
   {
     "project": {"name": "my_experiment", "base_output_dir": "/outputs/exp_20250115_120000"},
     "backend": {"args": {"load": "/outputs/exp_20250115_120000/checkpoints"}},
     ...
   }
   ```

2. **`config_reference.json`**: Metadata for reconstructing the config
   ```json
   {
     "config_name": "autoexp",
     "config_dir": "/path/to/config",
     "overrides": ["project=default", "backend=megatron"]
   }
   ```

3. **`run_metadata.json`**: Job-specific runtime metadata

### Current Usage in RunAutoexpAction

The `RunAutoexpAction` currently uses `config_path` pointing to `config_reference.json`, which then reconstructs the config by way of Hydra composition + overrides.

---

## Solution 1: Config Storage

### Proposed Changes

Save **three** config representations to `{output_dir}/provenance/`:

#### 1. `resolved_config.yaml` (fully resolved)
- All OmegaConf interpolations substituted
- All paths absolute and explicit
- Can be executed directly without any overrides
- **Use case**: Exact reproduction of the run

```yaml
project:
  name: my_experiment
  base_output_dir: /outputs/exp_20250115_120000
  monitoring_state_dir: /outputs/monitoring_state
backend:
  class_name: MegatronBackend
  args:
    train_iters: 100000
    save: /outputs/exp_20250115_120000/checkpoints
    load: /outputs/exp_20250115_120000/checkpoints
    learning_rate: 0.001
    ...
```

#### 2. `unresolved_config.yaml` (with OmegaConf interpolations)
- Keep all `${...}` interpolations intact
- Allows changing root variables that cascade to dependent paths
- **Use case**: Flexible re-execution with minimal overrides

```yaml
project:
  name: my_experiment
  base_output_dir: /outputs/exp_20250115_120000
  monitoring_state_dir: /outputs/monitoring_state
backend:
  class_name: MegatronBackend
  args:
    train_iters: 100000
    save: ${project.base_output_dir}/checkpoints
    load: ${project.base_output_dir}/checkpoints
    learning_rate: 0.001
    ...
```

#### 3. `config_reference.json` (Hydra reconstruction metadata) - KEEP AS IS
- Current format with `config_name`, `config_dir`, `overrides`
- **Use case**: Reconstruct config from original Hydra structure (most flexible)

### Implementation Approach

Modify `_write_job_provenance()` in `scripts/run_autoexp.py`:

```python
def _write_job_provenance(plan, *, args, config_container, ...):
    # Existing: write resolved_config.json
    resolved_config = OmegaConf.to_container(config_container, resolve=True)

    # NEW: write resolved_config.yaml (YAML version for easier editing)
    resolved_yaml_path = provenance_dir / "resolved_config.yaml"
    OmegaConf.save(config=config_container, f=resolved_yaml_path, resolve=True)

    # NEW: write unresolved_config.yaml (keep interpolations)
    unresolved_yaml_path = provenance_dir / "unresolved_config.yaml"
    OmegaConf.save(config=config_container, f=unresolved_yaml_path, resolve=False)

    # Existing: write config_reference.json (keep as is)
    ...
```

### Why Both Resolved and Unresolved?

| Scenario | Best Config | Reason |
|----------|-------------|--------|
| Exact reproduction | `resolved_config.yaml` | No ambiguity, all paths explicit |
| Change base_output_dir only | `unresolved_config.yaml` | Override one var, all paths update |
| Change platform/backend | `config_reference.json` | Reconstruct with Hydra defaults |
| Cooldown with new paths | `unresolved_config.yaml` + overrides | See Solution 2 |

---

## Solution 2: Load vs Save Path Management

### The Challenge

For a cooldown run triggered at iteration 50000:

**Original run paths:**
- Output: `/outputs/exp_20250115_120000/`
- Checkpoints saved to: `/outputs/exp_20250115_120000/checkpoints/iter_50000/`

**Cooldown run paths (desired):**
- Output: `/outputs/exp_20250115_140000_cooldown/` (NEW timestamp)
- Load FROM: `/outputs/exp_20250115_120000/checkpoints/iter_50000/` (ORIGINAL)
- Save TO: `/outputs/exp_20250115_140000_cooldown/checkpoints/` (NEW)

**Problem**: If we load `unresolved_config.yaml` and override `base_output_dir`, both `load` and `save` will point to the new directory because they're both `${project.base_output_dir}/checkpoints`.

### Solution 2A: Use Absolute Path Override for Load (RECOMMENDED)

Use `unresolved_config.yaml` but override `backend.args.load` with an **absolute path** from the event metadata.

**Event metadata** (captured when checkpoint is saved):
```yaml
checkpoint_path: /outputs/exp_20250115_120000/checkpoints/iter_50000
```

**Cooldown action**:
```yaml
action:
  class_name: RunAutoexpAction
  config_path: "{output_dir}/provenance/unresolved_config.yaml"
  overrides:
    # Override with ABSOLUTE path from event metadata (won't be affected by base_output_dir change)
    - backend.args.load={checkpoint_path}

    # Override base_output_dir to trigger new timestamp
    - project.base_output_dir=/outputs/exp_{timestamp}_cooldown

    # Save will resolve to: /outputs/exp_{timestamp}_cooldown/checkpoints
    # (because unresolved config has: save=${project.base_output_dir}/checkpoints)

    # Other overrides for cooldown behavior
    - project.name={project.name}_cooldown
    - backend.args.learning_rate=0.0001
    - monitoring=megatron_basic
```

**Key insight**: Absolute paths in overrides take precedence and don't get re-resolved, while interpolations like `${project.base_output_dir}/checkpoints` DO get resolved with the new `base_output_dir`.

### Solution 2B: Introduce Dual Variables (Alternative)

Modify config structure to have separate variables for load and save base directories.

**Modified config schema**:
```yaml
project:
  base_output_dir: /outputs/exp_20250115_120000  # For current run
  load_base_dir: ${.base_output_dir}  # Can be overridden independently

backend:
  args:
    load: ${project.load_base_dir}/checkpoints
    save: ${project.base_output_dir}/checkpoints
```

**Cooldown override**:
```yaml
overrides:
  - project.load_base_dir=/outputs/exp_20250115_120000  # Pin to original
  - project.base_output_dir=/outputs/exp_20250115_140000_cooldown  # New output
```

**Trade-offs**:
- ✅ More explicit separation of concerns
- ✅ Cleaner overrides (no absolute paths needed)
- ❌ Requires schema changes
- ❌ More complex for non-cooldown scenarios

### Solution 2C: Template Interpolation in Action Context (Not Recommended)

Store both `original_output_dir` and `checkpoint_path` in event metadata and use them in overrides.

**Why not recommended**:
- More complex event metadata management
- Hard to validate/debug
- Absolute paths (Solution 2A) are simpler and achieve the same result

### Recommendation: Use Solution 2A

**Advantages**:
1. No schema changes needed
2. Works with existing `unresolved_config.yaml`
3. Clear and explicit (absolute path for load, interpolation for save)
4. Event metadata already captures checkpoint paths

**Example in monitoring config**:
```yaml
log_events:
  - name: checkpoint_saved_cooldown
    pattern: 'successfully saved checkpoint from iteration\s+(?P<iteration>\d+) to (?P<path>\S+)'
    extract_groups:
      checkpoint_iteration: iteration
      checkpoint_path: path  # Captured as absolute path
    actions:
      - class_name: EventAction
        conditions:
          - class_name: MetadataCondition
            key: checkpoint_iteration
            equals: "50000"
        action:
          class_name: RunAutoexpAction
          config_path: "{output_dir}/provenance/unresolved_config.yaml"
          overrides:
            # Absolute path pinned to original location
            - backend.args.load={checkpoint_path}

            # New base_output_dir gets new timestamp by way of resolver
            - project.base_output_dir={project.base_output_dir}_cooldown_{timestamp}

            # These will resolve relative to NEW base_output_dir
            - project.name={project.name}_cooldown
            - backend.args.learning_rate=0.0001
```

---

## Solution 3: Implementation Plan

### Phase 1: Config Storage Enhancement

**Files to modify**:
- `scripts/run_autoexp.py`: Update `_write_job_provenance()` to save YAML configs
- `oellm_autoexp/monitor/actions.py`: Update `RunAutoexpAction` to support YAML config paths

**Changes**:
1. Save `resolved_config.yaml` (for exact reproduction)
2. Save `unresolved_config.yaml` (for flexible re-execution)
3. Keep `config_reference.json` (for Hydra reconstruction)
4. Update `RunAutoexpAction` to detect and load `.yaml` config files

**Test cases**:
- Verify all three files are written correctly
- Verify `RunAutoexpAction` can load from `unresolved_config.yaml`
- Verify path interpolations work correctly with overrides

### Phase 2: Cooldown Configuration

**Files to create**:
- `config/monitoring/megatron_cooldown_single.yaml`: Single cooldown at specific iteration
- `config/monitoring/megatron_cooldown_multi.yaml`: Multiple cooldowns at different iterations

**Configuration pattern**:
```yaml
log_events:
  - name: checkpoint_saved_cooldown
    pattern: 'successfully saved checkpoint from iteration\s+(?P<iteration>\d+) to (?P<path>\S+)'
    extract_groups:
      checkpoint_iteration: iteration
      checkpoint_path: path
    actions:
      - class_name: EventAction
        mode: queue
        conditions:
          - class_name: MetadataCondition
            key: checkpoint_iteration
            equals: "50000"
          - class_name: FileExistsCondition
            path: "{checkpoint_path}/latest_checkpointed_iteration.txt"
            blocking: true
        action:
          class_name: RunAutoexpAction
          config_path: "{output_dir}/provenance/unresolved_config.yaml"
          overrides:
            - backend.args.load={checkpoint_path}
            - project.base_output_dir={project.base_output_dir}_cooldown_$(date +%Y%m%d_%H%M%S)
            - backend.args.learning_rate=0.0001
            - monitoring=megatron_basic
```

**Test cases**:
- Original run saves checkpoint at iter 50000
- Cooldown triggers and queues `RunAutoexpAction`
- Cooldown run loads from original checkpoint
- Cooldown run saves to new output directory
- Verify paths don't collide

### Phase 3: Documentation and Examples

**Files to update**:
- `docs/advanced_features.md`: Update with final implementation details
- `README.md`: Add cooldown example to quick recipes
- `docs/monitoring_and_restart.md`: Document cooldown patterns

**Example workflow**:
```bash
# Start original training run
python scripts/run_autoexp.py \
  project=my_training \
  monitoring=megatron_cooldown_single \
  backend.args.train_iters=100000

# Monitor detects checkpoint at iteration 50000
# Automatically queues cooldown run with:
#   - Load from: {original_output}/checkpoints/iter_50000
#   - Save to: {original_output}_cooldown_{timestamp}/checkpoints
#   - Reduced LR for fine-tuning
```

---

## Alternative Approaches Considered

### Alternative 1: Use resolved_config.yaml with Manual Path Editing

**Approach**: Load `resolved_config.yaml` and override both load and save with absolute paths.

**Rejected because**:
- Requires overriding many paths (save, load, output_dir, log_dir, etc.)
- Error-prone (easy to miss a path)
- Defeats the purpose of having a config system

### Alternative 2: Generate New Config File for Each Cooldown

**Approach**: Write a new complete config file for the cooldown run.

**Rejected because**:
- Duplication of config data
- Hard to maintain consistency
- Provenance becomes unclear

### Alternative 3: Use Hydra config_reference.json

**Approach**: Use `config_reference.json` with Hydra composition for cooldown.

**Rejected because**:
- Requires Hydra config directory to be available
- May pull in different defaults if configs have changed
- Less explicit than using the exact config that was run

---

## Open Questions for Feedback

1. **Timestamp in base_output_dir override**: How should we generate a new timestamp for cooldown runs?
   - Option A: Use shell command `$(date +%Y%m%d_%H%M%S)` in override string
   - Option B: Use existing `${oc.timestring:}` resolver with `$$` escaping -> use this
   - Option C: Let `RunAutoexpAction` automatically append timestamp to base_output_dir

   **Note**: The `$$` escaping is critical - it tells OmegaConf to output a literal `${oc.timestring:}` string in the monitoring config, which then gets resolved when the cooldown run actually starts (not when the monitoring config is loaded).

2. **Monitoring config for cooldown run**: Should cooldown runs:
   - Option A: Use simplified monitoring (for example, `megatron_basic`) to avoid recursive cooldowns  -> use this
   - Option B: Support nested cooldowns by allowing `monitoring=megatron_cooldown_multi`
   - Option C: Disable monitoring entirely with `monitoring=none`

3. **Save directory naming convention**: What pattern should we use?
   - Option A: `{original}_cooldown_{iteration}_{timestamp}` (for example, `exp_20250115_120000_cooldown_20250115_140000`)  -> use this
   - Option B: `{original}/cooldown_{timestamp}` (nested under original)
   - Option C: `{original}_cooldown_{iteration}_{timestamp}` (include iteration number)

4. **Multiple cooldowns from same checkpoint**: If we want to run multiple cooldown strategies from the same checkpoint, should we:
   - Option A: Use different `project.name` suffixes for each strategy
   - Option B: Use different `base_output_dir` paths
   - Option C: Both (different names AND different output dirs)  -> use this

5. **Config file preference for RunAutoexpAction**: Should we prioritize:
   - Option A: Always use `unresolved_config.yaml` for maximum flexibility  -> use this
   - Option B: Let user specify which config file to use by way of parameter
   - Option C: Try `unresolved_config.yaml`, fallback to `config_reference.json`

---

## Summary

**Recommended Approach**:

1. **Config Storage**: Save both `resolved_config.yaml` and `unresolved_config.yaml` for different use cases
2. **Load/Save Paths**: Use `unresolved_config.yaml` with absolute path override for `load` and interpolation-based `save`
3. **Implementation**: Enhance provenance writing, update `RunAutoexpAction`, create cooldown configs

**Key Benefits**:
- ✅ No schema changes required
- ✅ Works with existing action system
- ✅ Clear separation of load (pinned) vs save (new) paths
- ✅ Flexible for different cooldown strategies
- ✅ Maintains full provenance and reproducibility

**Next Steps**:
1. Get feedback on open questions
2. Implement Phase 1 (config storage)
3. Test with simple cooldown scenario
4. Create full cooldown configurations
