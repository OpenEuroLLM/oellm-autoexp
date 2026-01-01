# Multi-Stage Training: Hybrid Push-Pull Design with Composable Sweeps

## Overview

This document describes a proposal for implementing declarative multi-stage training workflows in oellm-autoexp. The goal is to **complement** the current push-based approach (where monitoring actions trigger new stages dynamically) with a pull-based approach (where stages are planned upfront and conditions determine when each executes).

**Key Insight:** Both push and pull have their place:
- **Push-based (existing):** Dynamic responses to runtime events (errors, metrics-based decisions)
- **Pull-based (new):** Planned multi-stage workflows with predictable dependencies

**Revolutionary Simplification:** Instead of a special `derived_jobs` mechanism, we realize that **stages are just another dimension in a composable sweep**. By introducing `product` and `list` modes in sweep groups, stages become a natural `list` group (no cross-product between stages), combined with `product` groups for hyperparameter sweeps.

## Current versus Proposed Architecture

### Current System (Push-Based Only)
```
Stage 1 runs → emits event → MonitorAction evaluates conditions → RunAutoexpAction → Stage 2 starts
```

**Strengths:**
- Dynamic: Stages created in response to actual training state
- Flexible: Can make decisions based on runtime metrics
- Reactive: Handles errors, anomalies, adaptive workflows

**Weaknesses:**
- Hard to see full experiment pipeline in config (stages hidden in actions)
- Debugging requires tracing through multiple monitoring sessions
- `start_condition_cmd` is a separate mechanism from monitor conditions
- Pre-planned workflows require verbose action configurations

### Proposed Hybrid System (Push + Pull)

```
┌─────────────────────────────────────────────────────────────┐
│ PULL-BASED: Planned Stages                                 │
│ Define stages upfront → Monitor checks conditions →        │
│ Submit when ready                                           │
└─────────────────────────────────────────────────────────────┘
                              +
┌─────────────────────────────────────────────────────────────┐
│ PUSH-BASED: Dynamic Actions                                │
│ Job runs → emits event → Action responds dynamically       │
└─────────────────────────────────────────────────────────────┘
```

**Combined Benefits:**
- **Pull for planned workflows:** Multi-stage training, progressive evaluation, checkpoint sweeps
- **Push for dynamic responses:** Error recovery, metric-based branching, adaptive training
- **Unified condition system:** Same conditions used for both pull (start gating) and push (action gating)
- **Single monitoring session:** Both planned and dynamic jobs tracked together

## The Composable Sweep Revolution: Stages as List Groups

### Core Insight

**Stages are just another sweep dimension.** Instead of creating a special `derived_jobs` or `stages` mechanism, we can use a **composable sweep structure** with alternating product/list modes:

- **Product mode**: Cartesian product of parameter lists (traditional grid search)
- **List mode**: Parallel configurations without cross-product (stages, paired configs)
- **Filter**: Apply at any level to prune combinations

### Critical: Parameters as Hydra Overrides

**ALL sweep parameters are Hydra override strings**, not direct config values. This is how the system currently works and must be preserved:

**Workflow:**
1. Sweep parameters → formatted as `key=value` strings
2. Combined with base config overrides
3. Config reloaded by way of Hydra `compose(overrides=[...])`
4. Runtime re-evaluated with job-specific config

**This enables:**
- ✅ **Group selections**: `backend=megatron_fsdp` (change entire backend group)
- ✅ **Nested parameters**: `backend.megatron.lr=1e-4`
- ✅ **New parameters**: `++backend.new_param=value` (Hydra syntax)
- ✅ **Parameter deletion**: `~backend.unwanted_param`
- ✅ **Complex overrides**: Any Hydra command-line syntax

**Example from existing push-based system:**
```yaml
# In RunAutoexpAction
overrides:
  - backend.megatron.load={checkpoint_path}  # Runtime template substitution
  - monitoring=megatron_basic               # Group selection!
  - "++backend.megatron.aux.tokens=${...}"  # New param with OmegaConf expr
```

**For composable sweeps:**
```yaml
sweep:
  type: product
  groups:
    # Product group - keys become override strings
    - type: product
      params:
        backend: [megatron_torchrun, megatron_fsdp]  # Group selection
        backend.megatron.lr: [1e-4, 5e-4]             # Nested param

    # List group - each config is a set of overrides
    - type: list
      configs:
        - stage: stable
          backend.megatron.lr_wsd_decay_iters: 0
        - stage: cooldown
          backend.megatron.lr_wsd_decay_iters: 2000
          backend.megatron.load: "{sibling.stable.output_dir}/checkpoint"
          # Template resolved FIRST, then becomes override string
```

**Resolution order for sibling references:**
1. **Template resolution**: `{sibling.stable.output_dir}` → `/outputs/lr1e-4_stable`
2. **Format as override**: `backend.megatron.load=/outputs/lr1e-4_stable/checkpoint`
3. **Hydra reload**: Config reloaded with this override
4. **Runtime evaluation**: Job-specific backend created

This means sibling references work seamlessly with Hydra overrides!

**Example:**
```yaml
sweep:
  type: product  # Top-level composition mode
  groups:
    # Group 1: Hyperparameter sweep (product mode)
    - type: product
      params:
        backend.megatron.lr: [2.5e-4, 5e-4, 1e-3]
        backend.megatron.global_batch_size: [64, 128]
      # Produces: 6 combinations

    # Group 2: Training stages (list mode - NO cross-product)
    - type: list
      configs:
        - stage: stable
          backend.megatron.lr_wsd_decay_iters: 0
        - stage: cooldown
          backend.megatron.lr_wsd_decay_iters: "${oc.eval:int(${backend.megatron.train_iters}*0.2)}"
          start_conditions: [...]
      # Produces: 2 configurations (not combined with each other)

  # Total: 6 (hyperparams) × 2 (stages) = 12 jobs
```

### Why This Is Better

**Compared to `derived_jobs` approach:**
- ✅ **No special concepts** - everything is just sweep expansion
- ✅ **More flexible** - can have multiple list/product groups at any level
- ✅ **Unified mechanism** - same code path for all sweep types
- ✅ **Composable** - can nest and combine freely
- ✅ **Cleaner semantics** - stages are just parameter sets in a list

**The only missing piece:** Job reference mechanism to allow configs to reference other sweep points (for example, "load checkpoint from the stable stage").

### Sweep Composition Modes

#### Product Mode

Computes **cartesian product** of all parameter combinations.

```yaml
- type: product
  params:
    backend.megatron.lr: [1e-4, 5e-4, 1e-3]
    backend.megatron.global_batch_size: [64, 128, 256]
  filter: "backend.megatron.lr * backend.megatron.global_batch_size <= 256e-4"  # Optional pruning
  # Produces: 9 combinations → filtered → ~6 combinations
```

**Use cases:**
- Hyperparameter grid searches
- Ablation studies
- Architecture variations

#### List Mode

**No cross-product** - each config in the list is a separate sweep point.

```yaml
- type: list
  configs:
    - model: llama_1B
      backend.megatron.num_layers: 20
      backend.megatron.hidden_size: 1024

    - model: llama_3B
      backend.megatron.num_layers: 32
      backend.megatron.hidden_size: 2048

    - model: llama_7B
      backend.megatron.num_layers: 40
      backend.megatron.hidden_size: 4096
  # Produces: 3 configurations (paired, not crossed)
```

**Use cases:**
- Stages (stable, cooldown, eval)
- Paired configurations (model size + corresponding hyperparams)
- Sequential workflows
- Different datasets with specific settings

#### Composition

Groups can be combined by way of product:

```yaml
sweep:
  type: product  # Combine groups
  groups:
    - type: product
      params: {...}  # 6 points

    - type: list
      configs: [...]  # 3 points

    - type: product
      params: {...}  # 2 points

  # Total: 6 × 3 × 2 = 36 sweep points
```

#### Filters at All Levels

Filters can be applied at any level:

```yaml
sweep:
  type: product
  groups:
    - type: product
      params:
        a: [1, 2, 3, 4]
        b: [10, 20, 30]
      filter: "a * b <= 60"  # Prune this group's combinations

    - type: list
      configs: [...]

  filter: "not (a == 1 and stage == 'cooldown')"  # Prune final combinations
```

**Filter evaluation context:**
- Has access to all parameters in current sweep point
- Can reference any parameter by name
- Evaluated after expansion at that level

---

## Job Reference Mechanism: The Critical Challenge

### The Problem

With stages in a list group, we need jobs to reference each other:

```yaml
- type: list
  configs:
    - stage: stable
      # ... stable config

    - stage: cooldown
      backend.megatron.load: "???"  # How to reference stable's checkpoint path?
      start_conditions:
        - class_name: FileExistsCondition
          path: "???"  # How to reference stable's output directory?
```

**Requirements:**
1. Reference sibling jobs (same hyperparams, different stage)
2. Access their output paths, parameters, runtime metadata
3. **CRITICAL:** Don't interfere with OmegaConf's `${...}` interpolation system
4. Work during sweep expansion (before jobs are submitted)
5. Support delayed resolution (for runtime metadata like checkpoint iteration)

### OmegaConf Interpolation Conflicts

**OmegaConf uses `${...}` for interpolations:**
```yaml
backend.megatron.train_iters: "${oc.eval:(${.aux.tokens}//${.seq_length}//${.global_batch_size})}"
```

**We CANNOT use:**
- `${sibling.stable.output_dir}` - conflicts with OmegaConf syntax
- `${ref:stable:output_dir}` - OmegaConf will try to resolve this
- Any `${}` variant - OmegaConf owns this namespace

**Existing push-based approach uses string templates:**
```yaml
# In RunAutoexpAction (from repro_sweep_niccolo.yaml)
overrides:
  - backend.megatron.load={checkpoint_path}  # Runtime substitution
  - project.name={project.name}_decay{checkpoint_iteration}
```

These use `{variable}` (single braces) for **runtime** string interpolation after OmegaConf resolution.

### Proposed Solutions (Analyzed)

#### Option A: Use `{ref:...}` notation (single braces)

**Syntax:**
```yaml
- stage: cooldown
  backend.megatron.load: "{ref:stable:output_dir}/checkpoint"
  start_conditions:
    - class_name: FileExistsCondition
      path: "{ref:stable:output_dir}/checkpoint/iter_80000/done.txt"
```

**Resolution timing:**
- OmegaConf resolves first (ignores `{ref:...}` as literal strings)
- Sweep expander/job planner resolves `{ref:...}` patterns
- String manipulation after OmegaConf pass

**Advantages:**
- ✅ No conflict with OmegaConf (different syntax)
- ✅ Explicit and readable
- ✅ Consistent with existing push-based approach
- ✅ Can be extended: `{ref:stable:param_name}`, `{ref:stable:metadata.checkpoint_iteration}`

**Disadvantages:**
- ❌ Not type-safe (everything is strings)
- ❌ Requires string parsing
- ❌ Can't use in OmegaConf expressions: `${oc.eval:...}` won't see resolved values

**Reference syntax:**
```
{ref:PATTERN:ACCESSOR}

PATTERN:
  - stage_name (for example, "stable") - matches stage parameter
  - stage=stable - explicit match
  - index=0 - explicit job index
  - (future) complex patterns

ACCESSOR:
  - output_dir, log_path, name - job attributes
  - param.backend.megatron.lr - job parameters
  - metadata.checkpoint_iteration - runtime metadata (from monitoring)
```

#### Option B: Use special parameter prefix `__ref_*`

**Syntax:**
```yaml
- stage: cooldown
  __ref_stable_output: "{stable}"  # Mark as reference
  backend.megatron.load: "${.__ref_stable_output}/checkpoint"  # OmegaConf interpolation
```

**Resolution timing:**
- Sweep expander resolves `__ref_*` parameters to actual values
- OmegaConf then resolves `${.__ref_stable_output}` normally

**Advantages:**
- ✅ Works with OmegaConf type system
- ✅ Can use in expressions: `${oc.eval:int(${.__ref_stable_iters}*1.2)}`
- ✅ Type-safe

**Disadvantages:**
- ❌ Verbose (need extra parameters)
- ❌ Less intuitive
- ❌ Pollutes parameter namespace

#### Option C: Two-pass resolution with reserved namespace

**Syntax:**
```yaml
- stage: cooldown
  backend.megatron.load: "${__sweep__.sibling.stable.output_dir}/checkpoint"
```

**Resolution timing:**
- First pass: Sweep expander injects `__sweep__` dict into config
- Second pass: OmegaConf resolves normally

**Advantages:**
- ✅ Works with OmegaConf type system
- ✅ Can use in expressions
- ✅ Cleaner than Option B

**Disadvantages:**
- ❌ Complex implementation (inject special dict)
- ❌ `__sweep__` dict must be populated before OmegaConf resolution
- ❌ Requires knowing all sibling jobs during expansion

#### Option D: Delayed string template resolution (RECOMMENDED)

**Syntax:**
```yaml
- stage: cooldown
  # Use string templates (like push-based approach)
  backend.megatron.load: "{sibling.stable.output_dir}/checkpoint"

  start_conditions:
    - class_name: FileExistsCondition
      path: "{sibling.stable.output_dir}/checkpoint/iter_80000/done.txt"

  # For numeric computations, use auxiliary resolved params
  aux:
    stable_train_iters: "{sibling.stable.train_iters}"
  # Then reference in OmegaConf expressions
  backend.megatron.train_iters: "${oc.eval:int(${.aux.stable_train_iters}*1.2)}"
```

**Resolution timing:**
1. **Config load:** OmegaConf resolves `${...}` expressions (ignores `{sibling...}` as strings)
2. **Sweep expansion:** Generates sweep points, preserves `{sibling...}` templates
3. **Job planning:** Resolves `{sibling...}` templates to actual values using sibling lookup
4. **Job submission:** Fully resolved values available
5. **Monitor loop:** Can re-resolve templates for runtime metadata (for example, `{sibling.stable.metadata.checkpoint_iteration}`)

**Advantages:**
- ✅ **No OmegaConf conflicts** - different syntax space
- ✅ **Consistent with existing push approach** - same `{...}` pattern
- ✅ **Flexible** - can reference anything (paths, params, metadata)
- ✅ **Lazy resolution** - runtime metadata resolved when available
- ✅ **Simple implementation** - string substitution, no OmegaConf hacking

**Disadvantages:**
- ❌ **Can't use in OmegaConf expressions directly** - need aux params for numeric values
- ❌ **String-based** - not type-safe at config level
- ❌ **Requires careful ordering** - must know sibling jobs during planning

**Mitigation for numeric expressions:**
Use `aux` helper parameters for values needed in expressions:
```yaml
aux:
  stable_train_iters: "{sibling.stable.train_iters}"
  target_iters: "${oc.eval:int(${.stable_train_iters}*1.2)}"

backend.megatron.train_iters: "${.aux.target_iters}"
```

### Recommended Approach: Option D (Delayed String Templates)

**Why:**
1. **Proven pattern** - already works in push-based RunAutoexpAction
2. **No OmegaConf conflicts** - uses different syntax (`{...}` vs `${...}`)
3. **Simple implementation** - string substitution at job planning time
4. **Flexible** - handles both static (paths) and dynamic (metadata) references

### Reference Syntax Specification

**Template format:**
```
{sibling.PATTERN.ACCESSOR}
{sibling[PATTERN].ACCESSOR}  # Alternative syntax
```

**PATTERN matching:**
```yaml
{sibling.stable.output_dir}          # Match by stage=stable
{sibling._stage=stable.output_dir}   # Explicit
{sibling[_stage=stable].output_dir}  # Bracket syntax
{sibling.index=0.output_dir}         # Match by sweep index
```

**ACCESSOR paths:**
```yaml
{sibling.stable.output_dir}                     # Job metadata: output directory
{sibling.stable.name}                           # Job metadata: job name
{sibling.stable.log_path}                       # Job metadata: log file path
{sibling.stable.script_path}                    # Job metadata: script path

{sibling.stable.backend.megatron.lr}            # Job parameter (dotted path)
{sibling.stable.backend.megatron.train_iters}   # Job parameter

{sibling.stable.metadata.checkpoint_iteration}  # Runtime metadata (from monitoring)
{sibling.stable.metadata.checkpoint_path}       # Runtime metadata
```

**Special ACCESSORs:**
- `output_dir` → Resolved from `project.base_output_dir` + `project.name`
- `name` → Job name (resolved from `project.name` template)
- `log_path` → SLURM log path
- `metadata.*` → Runtime metadata extracted by monitoring (delayed resolution)

### Resolution Algorithm

**Phase 1: Sweep Expansion**

**CRITICAL:** Parameters are stored as-is (not evaluated), because they will become Hydra override strings.

```python
def expand_sweep(sweep_config):
    """
    Expand sweep groups into parameter sets.

    IMPORTANT: Parameters are kept as raw values (strings, numbers, etc.) because they
    will be formatted as Hydra override strings later: "key=value"
    """
    groups = sweep_config.groups
    all_points = []

    for group in groups:
        if group.type == "product":
            # Cartesian product of all parameter combinations
            # params = {"backend": ["v1", "v2"], "lr": [1e-4, 5e-4]}
            # → [{"backend": "v1", "lr": 1e-4}, {"backend": "v1", "lr": 5e-4}, ...]
            points = cartesian_product(group.params)
            if group.filter:
                points = [p for p in points if eval_filter(group.filter, p)]

        elif group.type == "list":
            # Each config is a separate point (no cross-product)
            # configs = [{"stage": "stable", "lr": 1e-4}, {"stage": "cooldown", "lr": 2e-4}]
            # → [{"stage": "stable", "lr": 1e-4}, {"stage": "cooldown", "lr": 2e-4}]
            points = group.configs
            # Keep {sibling...} templates as-is (strings) - resolved later

        all_points.append(points)

    # Combine groups by way of product
    # If groups = [[{a:1}, {a:2}], [{b:10}, {b:20}]]
    # → [{a:1, b:10}, {a:1, b:20}, {a:2, b:10}, {a:2, b:20}]
    final_points = cartesian_product_of_groups(all_points)

    # Apply top-level filter
    if sweep_config.filter:
        final_points = [p for p in final_points if eval_filter(sweep_config.filter, p)]

    return [SweepPoint(index=i, parameters=p) for i, p in enumerate(final_points)]
```

**Key observations:**
- Group selections like `backend=variant1` are stored as `{"backend": "variant1"}`
- Sibling templates like `{sibling.stable.output_dir}` are stored as literal strings
- NO config evaluation happens here - just parameter combination generation

**Phase 2: Job Planning (Template Resolution)**

**This phase resolves sibling templates BEFORE parameters become Hydra overrides.**

```python
def build_job_plans(root_config, sweep_points, config_setup):
    """
    Build job plans from sweep points.

    This phase:
    1. Resolves {sibling...} templates in parameters
    2. Creates JobPlan objects with resolved parameters
    3. Later (in orchestrator), parameters are formatted as Hydra overrides

    Note: We DON'T reload config here yet - that happens in render_scripts().
    This phase just prepares the parameters and metadata.
    """
    jobs = []

    for idx, point in enumerate(sweep_points):
        # Resolve {sibling...} templates FIRST
        # This replaces {sibling.stable.output_dir} with actual path strings
        resolved_point = resolve_sibling_templates(
            point,
            sweep_points=sweep_points,
            current_index=idx,
            jobs_meta=jobs,  # Previously built jobs (for metadata access)
            root_config=root_config,  # For resolving output paths
        )

        # Build job metadata using resolved parameters
        job_meta = {
            "index": idx,
            "parameters": resolved_point.parameters,  # Fully resolved (no more {sibling...})
            "output_dir": resolve_output_dir(root_config, resolved_point.parameters),
            "name": resolve_name(root_config, resolved_point.parameters),
            # ... other metadata
        }

        # Parse start_conditions from parameters
        start_conditions = []
        if "start_conditions" in resolved_point.parameters:
            start_conditions = parse_conditions(resolved_point.parameters["start_conditions"])
            # Remove from parameters (special field, not a Hydra override)
            del resolved_point.parameters["start_conditions"]

        jobs.append(JobPlan(
            name=job_meta["name"],
            output_dir=job_meta["output_dir"],
            parameters=resolved_point.parameters,  # Will become Hydra overrides
            start_conditions=start_conditions,
            # ...
        ))

    return jobs
```

**Key points:**
- Sibling templates resolved here: `{sibling.stable.output_dir}` → `/outputs/lr1e-4_stable`
- Parameters still just a dict: `{"backend.megatron.load": "/outputs/lr1e-4_stable/checkpoint"}`
- NOT yet Hydra overrides - that happens in Phase 3

**Phase 3: Sibling Lookup**
```python
def resolve_sibling_templates(point, sweep_points, current_index, jobs_meta):
    """Resolve {sibling.PATTERN.ACCESSOR} templates."""

    def find_sibling(pattern, sweep_points, current_index):
        """Find sibling job matching pattern."""
        current_point = sweep_points[current_index]

        # Parse pattern: "stable" or "_stage=stable" or "index=0"
        if "=" in pattern:
            key, value = pattern.split("=", 1)
            matcher = lambda p: p.parameters.get(key) == value
        else:
            # Default: match by stage parameter
            matcher = lambda p: p.parameters.get("stage") == pattern

        # Find matching sibling with same base parameters (except stage)
        for idx, candidate in enumerate(sweep_points):
            if idx == current_index:
                continue

            # Check if same "family" (same non-stage parameters)
            if is_same_family(current_point, candidate) and matcher(candidate):
                return idx, candidate

        raise ValueError(f"No sibling found matching pattern: {pattern}")

    def is_same_family(point_a, point_b):
        """Check if two points have same base parameters (ignoring stage markers)."""
        # Compare parameters except stage, _meta, etc.
        ignore_keys = {"stage", "_index", "start_conditions"}
        params_a = {k: v for k, v in point_a.parameters.items() if k not in ignore_keys}
        params_b = {k: v for k, v in point_b.parameters.items() if k not in ignore_keys}
        return params_a == params_b

    # Find all {sibling...} templates and resolve
    resolved = deep_copy(point)
    template_pattern = r"\{sibling\.([^.}]+)\.(.+?)\}"

    def replacer(match):
        pattern = match.group(1)  # for example, "stable"
        accessor = match.group(2)  # for example, "output_dir"

        sibling_idx, sibling_point = find_sibling(pattern, sweep_points, current_index)
        sibling_job = jobs_meta[sibling_idx]

        # Resolve accessor
        if accessor == "output_dir":
            return sibling_job.output_dir
        elif accessor == "name":
            return sibling_job.name
        elif accessor.startswith("metadata."):
            # Runtime metadata - preserve template for later resolution
            return f"{{runtime.{sibling_job.name}.{accessor[9:]}}}"
        elif "." in accessor:
            # Parameter access: "backend.megatron.lr"
            return get_nested(sibling_point.parameters, accessor)
        else:
            return getattr(sibling_job, accessor, f"{{unknown:{accessor}}}")

    # Apply template replacement recursively
    resolved = apply_template_replacement(resolved, template_pattern, replacer)

    return resolved
```

**Phase 3.5: Hydra Override Application (in orchestrator.render_scripts)**

**This is where parameters become Hydra overrides and config is reloaded.**

```python
def render_scripts(plan: ExecutionPlan):
    """
    Render SBATCH scripts for all jobs.

    For each job:
    1. Format parameters as Hydra override strings
    2. Reload config with these overrides
    3. Re-evaluate runtime (backend, slurm, etc.)
    4. Build launch command with job-specific runtime
    """
    for job in plan.jobs:
        if job.parameters and plan.config_setup.config_ref:
            # Format parameters as Hydra override strings
            # {"backend": "variant1", "backend.megatron.lr": 1e-4}
            # → ["backend=variant1", "backend.megatron.lr=1e-4"]
            job_overrides = [f"{key}={value}" for key, value in job.parameters.items()]

            # Combine with base overrides
            combined_overrides = list(plan.config_setup.override) + job_overrides

            # Reload config with job-specific overrides (HYDRA MAGIC HAPPENS HERE)
            # This is equivalent to: hydra --config-name=test backend=variant1 backend.megatron.lr=1e-4
            job_config = load_config_reference(
                plan.config_setup.config_ref,
                plan.config_setup.config_dir,
                overrides=combined_overrides,
            )

            # Re-evaluate to get job-specific backend/runtime
            # If override was backend=megatron_fsdp, we now have that backend loaded
            job_runtime = evaluate(job_config)

            # Build launch command with job-specific backend
            spec = BackendJobSpec(parameters={})  # Parameters already applied to config
            job_runtime.backend.validate(spec)
            launch_cmd = job_runtime.backend.build_launch_command(spec)
        else:
            # Fallback: pass parameters directly to backend (old behavior)
            spec = BackendJobSpec(parameters=dict(job.parameters))
            launch_cmd = plan.runtime.backend.build_launch_command(spec)

        # Render SBATCH script with launch command
        render_sbatch_script(job, launch_cmd)
```

**This phase is CRITICAL:**
- Parameters formatted as override strings: `"backend=variant1"`
- Hydra reloads entire config with these overrides
- Supports group selections, nested params, ANY Hydra syntax
- Each job gets a fully custom config/runtime

**Phase 4: Runtime Metadata Resolution (Monitor Loop)**
```python
def resolve_runtime_templates(job_plan, monitoring_state):
    """Resolve {runtime...} templates using monitoring state."""
    # For start_conditions that reference runtime metadata
    # for example, {runtime.job_stable.checkpoint_iteration}

    pattern = r"\{runtime\.([^.}]+)\.(.+?)\}"

    def replacer(match):
        job_name = match.group(1)
        metadata_key = match.group(2)

        job_state = monitoring_state.get_job(job_name)
        if not job_state:
            return None  # Job not yet tracked

        return job_state.extracted_metadata.get(metadata_key)

    # Apply to start_conditions
    for condition in job_plan.start_conditions:
        condition.resolve_runtime_templates(replacer)

    return all_resolved
```

### Complete Workflow Summary

**End-to-end flow showing how parameters become Hydra overrides:**

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Config Load (Initial)                                       │
│    - Hydra loads base config                                   │
│    - OmegaConf resolves ${...} expressions in base config      │
│    - Sweep definition contains {sibling...} as literal strings │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Sweep Expansion                                              │
│    - Product groups: cartesian product                         │
│    - List groups: no cross-product                             │
│    - Filter applied                                             │
│    - Result: SweepPoints with parameters dict                  │
│      Example: {"backend": "megatron_fsdp",                     │
│                "backend.megatron.lr": 1e-4,                     │
│                "stage": "cooldown",                            │
│                "backend.megatron.load": "{sibling.stable...}"} │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Sibling Template Resolution (Job Planning)                  │
│    - For each sweep point:                                      │
│      - Find sibling jobs (same family, different stage)        │
│      - Resolve {sibling.PATTERN.ACCESSOR} → actual values      │
│      - Result: Fully resolved parameters (still just dict)     │
│      Example: {"backend": "megatron_fsdp",                     │
│                "backend.megatron.lr": 1e-4,                     │
│                "stage": "cooldown",                            │
│                "backend.megatron.load": "/outputs/stable/ckpt"}│
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. Hydra Override Application (Orchestrator)                   │
│    - Format as override strings:                               │
│      ["backend=megatron_fsdp",                                 │
│       "backend.megatron.lr=1e-4",                              │
│       "stage=cooldown",                                        │
│       "backend.megatron.load=/outputs/stable/ckpt"]            │
│    - Reload config: load_config_reference(..., overrides=...) │
│    - Hydra applies group selection (backend=megatron_fsdp)    │
│    - Hydra sets nested params (backend.megatron.lr=1e-4)      │
│    - Result: Job-specific config                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. Runtime Evaluation                                           │
│    - evaluate(job_config) → job_runtime                        │
│    - Backend instantiated with job-specific config             │
│    - Launch command built                                       │
│    - SBATCH script rendered                                     │
└─────────────────────────────────────────────────────────────────┘
```

**Concrete Example:**

```yaml
# Config
sweep:
  type: product
  groups:
    - type: product
      params:
        backend: [megatron_torchrun, megatron_fsdp]  # Group selection
        backend.megatron.lr: [1e-4, 5e-4]

    - type: list
      configs:
        - stage: stable
        - stage: cooldown
          backend: [megatron_torchrun]
          backend.megatron.load: "{sibling.stable.output_dir}/checkpoint"
```

**Expansion:**
```
4 sweep points (2 backends × 2 lr) × 2 stages = 8 jobs total

Job 0: {"backend": "megatron_torchrun", "backend.megatron.lr": 1e-4, "stage": "stable"}
Job 1: {"backend": "megatron_torchrun", "backend.megatron.lr": 1e-4, "stage": "cooldown",
        "backend.megatron.load": "{sibling.stable.output_dir}/checkpoint"}
Job 2: {"backend": "megatron_torchrun", "backend.megatron.lr": 5e-4, "stage": "stable"}
Job 3: {"backend": "megatron_torchrun", "backend.megatron.lr": 5e-4, "stage": "cooldown",
        "backend.megatron.load": "{sibling.stable.output_dir}/checkpoint"}
Job 4-7: Same pattern with backend=megatron_fsdp
```

**Job 1 Resolution:**
```
1. Sibling lookup: Find Job 0 (same lr/backend, stage=stable)
2. Resolve template: {sibling.stable.output_dir} → "/outputs/lr1e-4_torchrun_stable"
3. Parameters: {"backend": "megatron_torchrun",
                "backend.megatron.lr": 1e-4,
                "stage": "cooldown",
                "backend.megatron.load": "/outputs/lr1e-4_torchrun_stable/checkpoint"}
4. Format as overrides: ["backend=megatron_torchrun",
                          "backend.megatron.lr=1e-4",
                          "stage=cooldown",
                          "backend.megatron.load=/outputs/lr1e-4_torchrun_stable/checkpoint"]
5. Hydra reload: Config loaded with backend=megatron_torchrun group
6. Launch command built with megatron_torchrun backend
```

### Conflict Avoidance Strategy

**Summary:**
1. **OmegaConf owns `${...}`** - Never use for job references
2. **Job references use `{...}`** (single braces) - Ignored by OmegaConf
3. **Parameters are Hydra override strings** - Not evaluated until Hydra reload
4. **Resolution order:**
   - OmegaConf resolves `${...}` → final config values (initial load)
   - Sweep expander sees `{sibling...}` as literal strings → preserves them
   - Job planner resolves `{sibling...}` → replaces with actual values
   - Orchestrator formats as overrides → `"key=value"` strings
   - Hydra reloads config with overrides → applies group selections, nested params
   - Monitor resolves `{runtime...}` → fills in runtime metadata

**No conflicts because:**
- Different syntax spaces: `${...}` vs `{...}`
- Different resolution phases: OmegaConf early, templates late
- Parameters treated as strings until Hydra reload

### Example: Full Multi-Stage Config with References

```yaml
# @package _global_
defaults:
  - /backend: megatron_torchrun
  - /slurm: lumi
  - _self_

project:
  name: "lr{backend.megatron.lr}_bsz{backend.megatron.global_batch_size}_{stage}"
  base_output_dir: "outputs/multi_stage"

backend:
  megatron:
    seq_length: 4096
    global_batch_size: 64
    micro_batch_size: 8
    save_interval: 2000
    lr: 5e-4

sweep:
  type: product
  groups:
    # Hyperparameter sweep
    - type: product
      params:
        backend.megatron.lr: [2.5e-4, 5e-4, 1e-3]
        backend.megatron.global_batch_size: [64, 128]

    # Training stages (list mode - no cross-product)
    - type: list
      configs:
        # Stage 1: Stable training
        - stage: stable
          backend.megatron.aux.tokens: 50_000_000_000
          backend.megatron.train_iters: "${oc.eval:(${.aux.tokens}//${.seq_length}//${.global_batch_size})}"
          backend.megatron.lr_wsd_decay_iters: 0
          # No start_conditions - submits immediately

        # Stage 2: Cooldown
        - stage: cooldown
          # Helper: get stable's train_iters for computation
          aux:
            stable_train_iters: "{sibling.stable.backend.megatron.train_iters}"
            decay_fraction: 0.2
            target_iteration: "${oc.eval:int(${.stable_train_iters}*0.8)}"
            target_iteration_round: "${oc.eval:(${.target_iteration}//${backend.megatron.save_interval})*${backend.megatron.save_interval}}"

          # Reference stable's checkpoint
          backend.megatron.load: "{sibling.stable.output_dir}/checkpoints/iter_{aux.target_iteration_round}"
          backend.megatron.aux.tokens: "{sibling.stable.backend.megatron.aux.tokens}"
          backend.megatron.train_iters: "${oc.eval:(${.aux.tokens}//${.seq_length}//${.global_batch_size})}"
          backend.megatron.lr_wsd_decay_iters: "${oc.eval:int(${.train_iters}*${.aux.decay_fraction})}"
          backend.megatron.override_opt_param_scheduler: true

          # Wait for stable to reach checkpoint
          start_conditions:
            - class_name: FileExistsCondition
              path: "{sibling.stable.output_dir}/checkpoints/iter_{aux.target_iteration_round}/latest_checkpointed_iteration.txt"
              blocking: true
              timeout_seconds: 7200

            - class_name: MetadataCondition
              # Runtime metadata - resolved by monitor
              key: "{runtime.{sibling.stable.name}.checkpoint_iteration}"
              equals: "${.aux.target_iteration_round}"

  # Total: 6 (hyperparams) × 2 (stages) = 12 jobs
```

**Resolution walkthrough:**
1. **OmegaConf pass:** Resolves `${oc.eval:...}` expressions, leaves `{sibling...}` as strings
2. **Sweep expansion:** Creates 12 sweep points (6 hyperparams × 2 stages)
3. **Job planning:** For each cooldown job:
   - Finds sibling stable job (same lr/bsz)
   - Replaces `{sibling.stable.output_dir}` with actual path
   - Replaces `{sibling.stable.backend.megatron.train_iters}` with value
   - Preserves `{runtime...}` templates for monitor
4. **Job submission:** Stable jobs submit immediately, cooldown jobs marked "waiting"
5. **Monitor loop:** Checks cooldown start_conditions:
   - Resolves `{runtime...}` templates using stable job's metadata
   - Submits cooldown when conditions satisfied

---

## Key Design Principles

### 1. Unified Condition System

**All condition types work in all contexts:**
- Monitor action conditions (existing)
- Start conditions for jobs (new)
- Both use same `BaseCondition` interface

Conditions:
- `FileExistsCondition` - wait for checkpoint files
- `MetadataCondition` - gate on iteration number or metrics
- `SlurmStateCondition` (new) - wait for parent job completion
- `CompositeCondition` - combine multiple prerequisites
- `CommandCondition` - run arbitrary check command (replaces start_condition_cmd)

### 2. Job Reference Templates

Jobs can reference other jobs in the sweep by way of templates:

```yaml
# Option 2 syntax (derived_jobs)
parameter_overrides:
  backend.megatron.load: "{base.output_dir}/checkpoint"
  project.name: "{base.name}_cooldown"

# Option 3 syntax (stages)
parameters:
  backend.megatron.load: "{parent.output_dir}/checkpoint"
  project.name: "{parent.name}_cooldown"
```

Available template variables:
- `{base.*}` or `{parent.*}` - Parent job attributes
- `{base.name}`, `{base.output_dir}`, `{base.log_path}` - Paths
- `{base.checkpoint_iteration}` - Extracted metadata from parent logs
- `{base.<param>}` - Any parameter from parent job

### 3. Monitor-Driven Submission

Instead of blocking orchestrator with `start_condition_cmd`:

```
Old (blocking):
  orchestrator.submit() → wait_for_start_condition() [BLOCKS] → slurm.submit()

New (async):
  orchestrator.submit() → register job as "waiting"
  monitor loop → check start_conditions → slurm.submit() when ready
```

**Benefits:**
- Orchestrator doesn't block on slow conditions
- Can monitor multiple pending jobs simultaneously
- Conditions checked on same cadence as job monitoring
- Failed conditions visible in monitoring state

### 4. Coexistence with Push-Based Actions

Pull-based (planned) and push-based (reactive) work together:

```yaml
sweep:
  derived_jobs:
    # PULL: Planned cooldown stage
    - name: cooldown
      for_each_base_job: true
      start_conditions: [...]
      parameter_overrides: {...}

monitoring:
  log_events:
    - name: training_diverged
      pattern: 'loss is NaN'
      actions:
        # PUSH: Dynamic response to error
        - class_name: RestartAction
          conditions:
            - class_name: MaxAttemptsCondition
              max_attempts: 3
```

Both use same condition system, tracked in same monitoring session.

## Example Configurations

All examples below use **Option 2 (derived_jobs)** syntax as the recommended approach.

### Example 1: Linear Multi-Stage Pipeline (Stable → Cooldown)

This reproduces the pattern from `dense_300M_50BT.yaml` - a stable training phase followed by a cooldown phase.

```yaml
# @package _global_
defaults:
  - /backend: megatron_torchrun
  - /slurm: lumi
  - /monitoring: megatron_basic
  - _self_

project:
  name: "dense_300M_lr{backend.megatron.lr}_gbsz{backend.megatron.global_batch_size}"
  base_output_dir: "outputs/multi_stage_300M"

backend:
  megatron:
    num_layers: 20
    hidden_size: 896
    seq_length: 4096
    micro_batch_size: 8
    save_interval: 2000

# Sweep over hyperparameters
sweep:
  grids:
    - backend.megatron.lr: [2.5e-4, 5.e-4, 1.e-3]
      backend.megatron.global_batch_size: [64, 128]
      backend.megatron.aux.tokens: 50_000_000_000
      backend.megatron.train_iters: "${oc.eval:(${.aux.tokens}//${backend.megatron.seq_length}//${backend.megatron.global_batch_size})}"
      backend.megatron.lr_wsd_decay_iters: 0

  # Derived jobs: Cooldown phase for each base sweep point
  derived_jobs:
    - name: "cooldown"
      for_each_base_job: true

      # Helper values
      aux:
        decay_fraction: 0.2
        target_iteration: "${oc.eval:int(${base.train_iters}*0.8)}"
        target_iteration_round: "${oc.eval:(${.target_iteration}//${backend.megatron.save_interval})*${backend.megatron.save_interval}}"

      # Wait conditions (checked by monitor before submission)
      start_conditions:
        - class_name: FileExistsCondition
          path: "{base.output_dir}/checkpoints/iter_{aux.target_iteration_round}/latest_checkpointed_iteration.txt"
          blocking: true
          timeout_seconds: 7200
        - class_name: MetadataCondition
          key: "base.checkpoint_iteration"
          equals: "${aux.target_iteration_round}"

      # Parameter overrides
      parameter_overrides:
        backend.megatron.load: "{base.output_dir}/checkpoints/iter_{aux.target_iteration_round}"
        backend.megatron.aux.tokens: "${oc.eval:int(${aux.target_iteration_round}/(1.-${aux.decay_fraction}))*${backend.megatron.global_batch_size}}"
        backend.megatron.train_iters: "${oc.eval:(${.aux.tokens}//${backend.megatron.seq_length}//${backend.megatron.global_batch_size})}"
        backend.megatron.lr_wsd_decay_iters: "${oc.eval:int(${.train_iters}*${aux.decay_fraction})}"
        backend.megatron.override_opt_param_scheduler: true
        project.name: "{base.name}_cooldown"

# Monitoring extracts metadata used by conditions
monitoring:
  log_events:
    - name: checkpoint_saved
      pattern: 'successfully saved checkpoint from iteration\s+(?P<iteration>\d+) to (?P<path>\S+)'
      extract_groups:
        checkpoint_iteration: iteration
        checkpoint_path: path
```

**How this works:**
1. Base sweep generates 6 jobs (3×2 grid)
2. `derived_jobs` creates 6 cooldown jobs (one per base job)
3. Base jobs submitted immediately
4. Cooldown jobs registered as "waiting" with start_conditions
5. Monitor checks conditions each cycle
6. When base job reaches iteration ~80000 and checkpoint exists → cooldown submitted
7. All 12 jobs tracked in single monitoring session

### Example 2: Branching Pipeline (Train → Multiple Evals)

Training followed by multiple evaluation branches.

```yaml
sweep:
  grids:
    - backend.megatron.lr: [5e-4]
      backend.megatron.global_batch_size: [64]
      backend.megatron.train_iters: 100000

  derived_jobs:
    # Evaluation on validation set
    - name: "eval_validation"
      for_each_base_job: true

      start_conditions:
        - class_name: SlurmStateCondition
          job_name: "{base.name}"
          state: COMPLETED
          timeout_seconds: 86400

      parameter_overrides:
        backend.megatron.load: "{base.output_dir}/checkpoints/final"
        backend.megatron.eval_only: true
        backend.megatron.eval_dataset: "validation"
        project.name: "{base.name}_eval_val"

    # Evaluation on test set (parallel with validation)
    - name: "eval_test"
      for_each_base_job: true

      start_conditions:
        - class_name: SlurmStateCondition
          job_name: "{base.name}"
          state: COMPLETED
          timeout_seconds: 86400

      parameter_overrides:
        backend.megatron.load: "{base.output_dir}/checkpoints/final"
        backend.megatron.eval_only: true
        backend.megatron.eval_dataset: "test"
        project.name: "{base.name}_eval_test"
```

**Result:** 1 base job → 3 total jobs (1 train + 2 parallel evals)

### Example 3: Progressive Cooldowns at Multiple Scales

Multiple cooldown phases at different token scales (from `dense_300M_50BT.yaml`).

```yaml
sweep:
  grids:
    - backend.megatron.lr: [5e-4, 1e-3]
      backend.megatron.global_batch_size: [64, 128]
      backend.megatron.aux.tokens: 50_000_000_000
      backend.megatron.train_iters: "${oc.eval:(${.aux.tokens}//${backend.megatron.seq_length}//${backend.megatron.global_batch_size})}"

  derived_jobs:
    # Cooldown at 6B tokens
    - name: "decay_6B"
      for_each_base_job: true
      aux:
        target_tokens: 6_000_000_000
        decay_fraction: 0.2
        target_iteration: "${oc.eval:int(${.target_tokens}//${backend.megatron.seq_length}//${backend.megatron.global_batch_size})}"
        start_iteration_round: "${oc.eval:int(int(${.target_iteration}*(1-${.decay_fraction}))//${backend.megatron.save_interval})*${backend.megatron.save_interval}}"

      start_conditions:
        - class_name: FileExistsCondition
          path: "{base.output_dir}/checkpoints/iter_{aux.start_iteration_round}/latest_checkpointed_iteration.txt"
          blocking: true
        - class_name: MetadataCondition
          key: "base.checkpoint_iteration"
          equals: "${aux.start_iteration_round}"

      parameter_overrides:
        backend.megatron.load: "{base.output_dir}/checkpoints/iter_{aux.start_iteration_round}"
        backend.megatron.aux.tokens: "${aux.target_tokens}"
        backend.megatron.train_iters: "${oc.eval:(${.aux.tokens}//${backend.megatron.seq_length}//${backend.megatron.global_batch_size})}"
        backend.megatron.lr_wsd_decay_iters: "${oc.eval:int(${.train_iters}*${aux.decay_fraction})}"
        backend.megatron.override_opt_param_scheduler: true
        project.name: "{base.name}_decay_6B"

    # Cooldown at 12B tokens (similar pattern)
    - name: "decay_12B"
      for_each_base_job: true
      aux:
        target_tokens: 12_000_000_000
        # ... (same structure as 6B)

    # Cooldown at 30B tokens
    - name: "decay_30B"
      for_each_base_job: true
      aux:
        target_tokens: 30_000_000_000
        # ... (same structure)

    # Cooldown at 50B tokens
    - name: "decay_50B"
      for_each_base_job: true
      aux:
        target_tokens: 50_000_000_000
        # ... (same structure)
```

**Result:** 4 base jobs → 20 total jobs (4 base + 16 cooldown stages, all from one config)

### Example 4: Combining Push and Pull

Using both planned stages (pull) and dynamic actions (push) together.

```yaml
sweep:
  grids:
    - backend.megatron.lr: [5e-4]
      backend.megatron.train_iters: 100000

  # PULL: Planned cooldown stage
  derived_jobs:
    - name: "cooldown"
      for_each_base_job: true

      start_conditions:
        - class_name: FileExistsCondition
          path: "{base.output_dir}/checkpoints/iter_80000/done.txt"
          blocking: true

      parameter_overrides:
        backend.megatron.load: "{base.output_dir}/checkpoints/iter_80000"
        backend.megatron.train_iters: 120000
        project.name: "{base.name}_cooldown"

# PUSH: Dynamic error recovery
monitoring:
  log_events:
    - name: checkpoint_saved
      pattern: 'saved checkpoint from iteration\s+(?P<iteration>\d+)'
      extract_groups:
        checkpoint_iteration: iteration
      # No actions - just metadata extraction for pull conditions

  state_events:
    - name: job_crashed
      state: crash
      actions:
        # Dynamic restart with error handling
        - class_name: RestartAction
          conditions:
            - class_name: MaxAttemptsCondition
              max_attempts: 3
            - class_name: CooldownCondition
              cooldown_seconds: 300

    - name: job_stalled
      state: stall
      actions:
        # Adaptive: reduce batch size if stalling
        - class_name: RunAutoexpAction
          mode: queue
          conditions:
            - class_name: MaxAttemptsCondition
              max_attempts: 1  # Try once
          action:
            script: scripts/run_autoexp_container.py
            config_path: "{output_dir}/provenance/unresolved_config.yaml"
            overrides:
              - backend.megatron.micro_batch_size=4  # Reduce from 8
              - project.name={project.name}_recovered
```

**Result:** Combines declarative pipeline (pull) with reactive error handling (push)

## Configuration Schema

### New Fields (Option 2: derived_jobs)

#### `SweepConfig.derived_jobs`

```yaml
sweep:
  grids: [...]  # Base sweep points

  derived_jobs:
    - name: string                      # Job name suffix
      for_each_base_job: bool           # Generate one per base sweep point
      aux: dict                         # Helper values for computations
      start_conditions: list[Condition] # Conditions to check before submission
      parameter_overrides: dict         # Overrides applied to base parameters
```

**Fields:**
- `name`: Suffix added to job name (for example, "cooldown" → `{base.name}_cooldown`)
- `for_each_base_job`: If true, creates one derived job per base sweep point
- `aux`: Helper values available in templates (computed values, constants)
- `start_conditions`: List of condition objects (checked by monitor before submission)
- `parameter_overrides`: Parameters to override from base job (uses Hydra syntax)

#### Extended JobPlan Fields

```python
@dataclass
class JobPlan:
    # Existing fields...
    start_condition_cmd: str | None = None  # DEPRECATED

    # NEW fields
    start_conditions: list[BaseCondition] = field(default_factory=list)
    base_job_name: str | None = None  # For derived jobs, reference to base
    is_derived: bool = False
```

#### Condition Types

All existing monitor conditions work as start_conditions:

- **FileExistsCondition**: Wait for file to appear
  ```yaml
  - class_name: FileExistsCondition
    path: "{base.output_dir}/checkpoint.pt"
    blocking: true
    timeout_seconds: 3600
  ```

- **MetadataCondition**: Gate on extracted metadata
  ```yaml
  - class_name: MetadataCondition
    key: "base.checkpoint_iteration"
    equals: "80000"
  ```

- **SlurmStateCondition** (new): Wait for job completion
  ```yaml
  - class_name: SlurmStateCondition
    job_name: "{base.name}"
    state: COMPLETED
    timeout_seconds: 86400
  ```

- **CommandCondition**: Run custom check
  ```yaml
  - class_name: CommandCondition
    command: ["bash", "-c", "test -f {base.output_dir}/done.txt"]
  ```

- **CompositeCondition**: Combine multiple conditions
  ```yaml
  - class_name: CompositeCondition
    mode: all  # or "any"
    conditions:
      - class_name: FileExistsCondition
        path: "..."
      - class_name: MetadataCondition
        key: "..."
  ```

#### Template Variables

In derived job configurations, these template variables are available:

- `{base.name}` - Base job name
- `{base.output_dir}` - Base job output directory
- `{base.log_path}` - Base job log path
- `{base.<param>}` - Any parameter from base job (for example, `{base.train_iters}`)
- `{base.checkpoint_iteration}` - Extracted metadata from base job's logs
- `{base.checkpoint_path}` - Extracted metadata
- `{aux.*}` - Values from derived job's `aux` dictionary

## Implementation Components

### High-Level Flow

```
1. Configuration Loading
   └─> RootConfig with sweep.derived_jobs defined

2. Sweep Expansion (sweep/expander.py)
   └─> Expand base grids → N base sweep points
   └─> For each derived_job definition:
       └─> For each base point:
           └─> Create derived SweepPoint with:
               - Parameters = base params + overrides
               - _base_job_index = base sweep point index
               - Index = unique (for example, base_index * 1000 + derived_index)

3. Job Planning (sweep/planner.py)
   └─> For each SweepPoint (base + derived):
       └─> Build JobPlan with:
           - Instantiated start_conditions (from config)
           - Resolved templates using base job context
           - base_job_name (for derived jobs)
   └─> Result: List of all jobs (base + derived) in dependency order

4. Job Submission (orchestrator.py)
   └─> For each job:
       ├─> If no start_conditions → submit immediately
       └─> If has start_conditions → register as "waiting"

5. Monitoring Loop (monitor/controller.py)
   └─> Each cycle:
       ├─> Observe active jobs (extract metadata to state)
       └─> Check waiting jobs:
           └─> For each waiting job:
               ├─> Build condition context (base job metadata available)
               ├─> Evaluate all start_conditions
               └─> If all pass → submit job, move to active
```

### Key Components to Modify

**Minimal changes approach:**

1. **Schema** (`config/schema.py`)
   - Add `DerivedJobDefinition` dataclass
   - Extend `SweepConfig` with `derived_jobs: list[DerivedJobDefinition]`

2. **Sweep Expansion** (`sweep/expander.py`)
   - Modify `expand_sweep()` to handle derived_jobs
   - Add `_expand_derived_jobs()` helper
   - Generate derived SweepPoints with base job references

3. **Job Planning** (`sweep/planner.py`)
   - Extend `JobPlan` with `start_conditions`, `base_job_name`, `is_derived`
   - Parse and instantiate condition objects from config
   - Resolve `{base.*}` templates during planning

4. **Monitor Controller** (`monitor/controller.py`)
   - Change: Don't call `wait_for_start_condition()` in orchestrator
   - Add `_waiting_jobs` tracking dict
   - Implement `_check_waiting_jobs()` in observe loop
   - Move to active when conditions satisfied

5. **Orchestrator** (`orchestrator.py`)
   - Change submission logic:
     - If `job.start_conditions` is empty → submit immediately
     - If `job.start_conditions` is present → register as waiting

6. **Conditions** (`monitor/conditions.py`)
   - Add `SlurmStateCondition` for checking SLURM job states
   - Extend `ConditionContext` to support `{base.*}` variable resolution

7. **Persistence** (`persistence/state_store.py`)
   - Serialize/restore waiting job state

## Migration Path

### Backward Compatibility

**Key principle:** Keep existing functionality working, add new features alongside.

1. **`start_condition_cmd` remains supported** - No breaking changes
2. **Push-based RunAutoexpAction still works** - For dynamic workflows
3. **New pull-based derived_jobs is additive** - Opt-in enhancement

### Migration from start_condition_cmd

**Old (blocking):**
```yaml
monitoring:
  start_condition_cmd: "{helpers}/check_checkpoint.sh {checkpoint_path}"
```

**New (async, unified conditions):**
```yaml
sweep:
  derived_jobs:
    - name: "stage2"
      for_each_base_job: true
      start_conditions:
        - class_name: CommandCondition
          command: ["{helpers}/check_checkpoint.sh", "{checkpoint_path}"]
```

### When to Use Pull versus Push

| Use Case | Recommended Approach |
|----------|---------------------|
| Planned multi-stage pipeline | **Pull** (derived_jobs) |
| Error recovery / restarts | **Push** (state_events + RestartAction) |
| Metric-based branching (unknown at planning) | **Push** (log_events + RunAutoexpAction) |
| Checkpoint-based continuation | **Pull** (derived_jobs with FileExistsCondition) |
| Adaptive hyperparameter adjustment | **Push** (log_events + RunAutoexpAction) |
| Progressive cooldowns at known checkpoints | **Pull** (derived_jobs) |

**General rule:** If you know the workflow structure upfront → **Pull**. If workflow depends on runtime decisions → **Push**.

## Efficiency & Implementation Considerations

### Monitor Loop Overhead

**Question:** Does checking conditions for N waiting jobs each cycle add significant overhead?

**Analysis:**
- Most conditions are cheap (file exists, metadata comparison)
- Expensive conditions (CommandCondition) should use `blocking: true` + timeout
- Condition checking is parallelizable (check multiple jobs concurrently)
- Typical scenario: 10-20 waiting jobs × 5-10 conditions each = 50-200 checks per cycle
- At 60s monitor interval, this is negligible

**Optimization opportunities:**
1. Cache file existence checks (short TTL)
2. Only re-evaluate conditions when parent job state changes
3. Skip condition checks if parent job not yet active
4. Event-driven: trigger check immediately when parent emits checkpoint event

### Template Resolution Efficiency

**Question:** How to efficiently resolve `{base.*}` templates without loading full job state?

**Proposed approach:**
1. **During planning:** Build a job lookup table: `job_name → {name, output_dir, parameters}`
2. **Template resolution:** Simple dict lookup, no expensive operations
3. **Metadata access:** Monitor already tracks extracted metadata per job
4. **Cache:** Keep job metadata in memory during monitoring session

### Condition Context Building

**Key insight:** Conditions need access to base job metadata.

**Implementation:**
```python
def _build_condition_context_for_derived_job(
    derived_job: JobPlan,
    base_job_state: JobRuntimeState,
) -> ConditionContext:
    return ConditionContext(
        event=None,  # No specific event
        job_metadata={
            "base": {
                "name": base_job_state.registration.name,
                "output_dir": base_job_state.registration.metadata.get("output_dir"),
                "checkpoint_iteration": base_job_state.extracted_metadata.get("checkpoint_iteration"),
                # ... other metadata
            },
            **derived_job.parameters,
        },
        attempts=0,
    )
```

This is **cheap** - just dict construction, no I/O.

### State Persistence

**Question:** What needs to be persisted for waiting jobs?

**Minimal state:**
```json
{
  "waiting_jobs": {
    "job_name_cooldown": {
      "job_plan": {...},
      "script_path": "...",
      "registered_at": 1234567890,
      "base_job_name": "job_name_stable"
    }
  }
}
```

This is **small** - similar size to current job registration.

## Design Decisions Summary

All pain points and open questions have been addressed. Here are the **FINAL DECISIONS** for implementation:

| # | Topic | Decision | Rationale |
|---|-------|----------|-----------|
| 1 | **Validation Phase** | ✅ Implement full validation during planning | Fail fast with clear errors, catch 99% of issues before runtime |
| 2 | **Job Indexing** | ✅ Keep sequential (natural from product expansion) | Families grouped together (0-1, 2-3, etc.) |
| 3 | **Condition Evaluation** | ✅ Simple loop (every monitor cycle) | Predictable, simple to implement and debug |
| 4 | **Failure Handling** | ✅ Timeout + cancel_conditions | Timeout for passive wait, cancel_conditions for active detection |
| 4a | **Log File Strategy** | ✅ SLURM ID + symlink (`current.log`) | Historical audit trail + stable path for conditions |
| 5 | **Multi-sibling Dependencies** | ⏸️ Keep as future extension | Naturally supported, but not needed for v1 |
| 6 | **Visualization** | ✅ ASCII → ncurses → web (phased) | v1: ASCII only, v2: ncurses interactive, v3+: web |
| 7 | **Template Syntax** | ✅ `${...}` for own config, `{...}` for siblings | Clean separation, no conflicts |
| 8 | **`_stage` Parameter** | ✅ Keep for v1 | Simple, works in filters/names, can add `_meta` later if needed |
| 9 | **Top-Level Sweep Type** | ✅ Support both `product` and `list` | Flexibility for different experiment structures |
| 10 | **Backward Compatibility** | ❌ No need (sweep not stable yet) | Can introduce breaking changes |
| 11 | **Multi-Stage Chains** | ✅ Support arbitrary length (4+ stages) | Linear chains + branching, no special nesting |
| 12 | **CLI Structure** | ✅ Separate scripts (NOT subcommands) | Preserves Hydra override compatibility: `script.py --config-ref cfg key=value` |

### Key Implementation Priorities for v1

**Must Have:**
1. ✅ Composable sweep (product + list groups)
2. ✅ Sibling template resolution `{sibling.PATTERN.ACCESSOR}`
3. ✅ Validation phase (fail fast)
4. ✅ start_conditions (pull-based staging)
5. ✅ cancel_conditions (failure detection)
6. ✅ Log file symlink strategy
7. ✅ ASCII visualization
8. ✅ Monitor-driven submission (non-blocking)

**Nice to Have (v2):**
- ncurses interactive monitor
- Multiple sibling dependencies
- `_meta` namespace for stages

**Future (v3+):**
- Web dashboard
- Event-driven condition evaluation
- Advanced dependency patterns

---

## Pain Points and Improvements

### 1. `_stage` Exposure in Parameter Namespace

**Current Design:**
Stages marked with `_stage` parameter:
```yaml
- stage: stable
- stage: cooldown
```

**Advantages:**
- ✅ Explicit and clear
- ✅ Can use in filters: `filter: 'not (_stage == "cooldown" and lr == 1e-4)'`
- ✅ Can use in job names: `name: "lr{lr}_{stage}"`
- ✅ Simple to understand

**Disadvantages:**
- ❌ Pollutes parameter namespace
- ❌ Might conflict with actual config parameters named `_stage`
- ❌ Gets passed as Hydra override (harmless but unnecessary)

**Alternative: Metadata Namespace**
```yaml
- _meta:
    stage: stable
    description: "Main training run"
- _meta:
    stage: cooldown
    parent_stage: stable
```

**Recommendation:** Keep `_stage` for v1 (simple), add `_meta` namespace in v2 if needed.

---

### 2. Outermost Sweep Type Flexibility

**Current Examples Show:**
```yaml
sweep:
  type: product  # Always product at top
  groups: [...]
```

**But Should Support:**
```yaml
sweep:
  type: list  # Or list at top!
  groups:
    - type: product
      params: {...}
```

**Use Case - Different Experiments with Different Structures:**
```yaml
sweep:
  type: list
  groups:
    - name: small_model_sweep
      type: product
      params:
        model_size: [1B, 3B]
        backend.megatron.lr: [1e-4, 5e-4]

    - name: large_model_sweep
      type: product
      params:
        model_size: [7B, 13B]
        backend.megatron.lr: [1e-5, 5e-5]  # Different lr range!
```

**Fix:** Update schema to allow `type: list | product` at top level, not just `product`.

---

### 3. Early Full Resolution and Validation (CRITICAL IMPROVEMENT)

**Key Insight:** For pull-based sibling references, we can **resolve everything before runtime** except runtime metadata.

**What Can Be Resolved Early:**
- ✅ Sibling output paths (from project.name template)
- ✅ Sibling parameters (from sweep point)
- ✅ Job names
- ✅ Static start conditions (file paths, parameter matches)

**What Must Be Delayed:**
- ❌ Runtime metadata: `{sibling.stable.metadata.checkpoint_iteration}` (from monitoring)

**Proposed: Add Validation Phase**

```python
def validate_execution_plan(plan: ExecutionPlan) -> ValidationResult:
    """
    Validate plan before submission to catch errors early.

    This runs AFTER sibling template resolution but BEFORE job submission.
    Catches errors that would otherwise surface during monitoring.
    """
    errors = []
    warnings = []

    for job in plan.jobs:
        # 1. Check sibling references resolved
        for param_key, param_value in job.parameters.items():
            if isinstance(param_value, str) and "{sibling." in param_value:
                errors.append(f"Job {job.name}: Unresolved sibling template in {param_key}: {param_value}")

        # 2. Validate start_conditions
        for condition in job.start_conditions:
            # Check that condition can be instantiated
            try:
                validate_condition(condition)
            except Exception as e:
                errors.append(f"Job {job.name}: Invalid start condition: {e}")

            # Check for runtime templates
            if has_runtime_template(condition):
                # OK - will be resolved during monitoring
                warnings.append(f"Job {job.name}: Condition uses runtime metadata - will block until available")
            else:
                # Static condition - can validate further
                if isinstance(condition, FileExistsCondition):
                    # Check path pattern is valid
                    if not is_valid_path_pattern(condition.path):
                        errors.append(f"Job {job.name}: Invalid file path pattern: {condition.path}")

        # 3. Check for circular dependencies
        if is_circular_dependency(job, plan.jobs):
            errors.append(f"Job {job.name}: Circular dependency detected in sibling chain")

        # 4. Validate Hydra overrides can be formatted
        try:
            overrides = [f"{k}={v}" for k, v in job.parameters.items()]
        except Exception as e:
            errors.append(f"Job {job.name}: Cannot format parameters as Hydra overrides: {e}")

        # 5. Check job name uniqueness
        if sum(1 for j in plan.jobs if j.name == job.name) > 1:
            errors.append(f"Duplicate job name: {job.name}")

    return ValidationResult(errors=errors, warnings=warnings)
```

**When to Run:**
```python
def build_execution_plan(config, config_setup):
    # 1. Expand sweep
    sweep_points = expand_sweep(config.sweep)

    # 2. Build jobs (resolve sibling templates)
    jobs = build_job_plans(config, sweep_points, config_setup)

    plan = ExecutionPlan(config=config, jobs=jobs, ...)

    # 3. VALIDATE before returning (FAIL FAST)
    validation = validate_execution_plan(plan)
    if validation.errors:
        raise ValueError(f"Plan validation failed:\n" + "\n".join(validation.errors))
    if validation.warnings:
        for warning in validation.warnings:
            LOGGER.warning(warning)

    return plan
```

**Benefits:**
- 🎯 **Fail fast** - Catch errors during `plan_autoexp.py`, not during monitoring
- 🎯 **Better error messages** - "Sibling 'stable' not found" instead of cryptic template errors
- 🎯 **Early detection** - Missing files, invalid conditions, circular deps
- 🎯 **Confidence** - Know plan is valid before submitting 100 jobs

**Example Errors Caught:**
```
❌ Job cooldown_lr1e-4: Sibling 'stabble' not found (typo in stage)
❌ Job eval: Circular dependency detected: eval → stable → eval
❌ Job cooldown_lr5e-4: Invalid file path pattern: /outputs/{invalid}/checkpoint
❌ Duplicate job name: lr1e-4_stable (conflict between sweep points)
```

---

### 4. Template Escaping

**Problem:** What if parameter value legitimately contains `{` or `}`?

```yaml
backend.megatron.special_config: "some{thing}weird"  # Not a template!
```

**Solution:** Require double braces for escaping:
```yaml
backend.megatron.special_config: "some{{thing}}weird"  # Literal {thing}
```

**Implementation:**
```python
def resolve_templates(text: str, context: dict) -> str:
    # Replace {{ and }} with placeholders first
    text = text.replace("{{", "\x00").replace("}}", "\x01")

    # Resolve templates
    text = resolve_sibling_templates(text, context)
    text = resolve_runtime_templates(text, context)

    # Restore escaped braces
    text = text.replace("\x00", "{").replace("\x01", "}")
    return text
```

---

### 5. Error Handling: Sibling Not Found

**Current:** Raises exception during job planning

**Better:** Collect all errors and report together
```python
def find_sibling(pattern, sweep_points, current_index):
    """Find sibling, raise descriptive error if not found."""
    try:
        return _find_sibling_impl(pattern, sweep_points, current_index)
    except ValueError:
        current = sweep_points[current_index]
        raise ValueError(
            f"Sibling not found for pattern '{pattern}' in job {current_index}.\n"
            f"Current job params: {current.parameters}\n"
            f"Looking for sibling with same base params but matching '{pattern}'\n"
            f"Available siblings: {[p.parameters.get('stage', '?') for p in sweep_points]}"
        )
```

---

### 6. Template Syntax Clarification: `${...}` vs `{...}`

**CRITICAL DISTINCTION:**

**Use `${...}` (OmegaConf) for own config references:**
```yaml
project:
  name: "lr${backend.megatron.lr}_bsz${backend.megatron.global_batch_size}_${stage}"
  #        ^^^ OmegaConf interpolation (own config)

backend:
  megatron:
    train_iters: "${oc.eval:(${.aux.tokens}//${.seq_length}//${.global_batch_size})}"
    #             ^^^ OmegaConf expression
```

**Use `{...}` (single braces) ONLY for sibling references:**
```yaml
- stage: cooldown
  backend.megatron.load: "{sibling.stable.output_dir}/checkpoint"
  #                       ^^^ Sibling template (different job)
```

**Why this works:**
1. **Sibling templates resolved FIRST** (during job planning):
   - `{sibling.stable.output_dir}` → `/outputs/lr1e-4_stable`
   - Parameters now: `{"backend.megatron.load": "/outputs/lr1e-4_stable/checkpoint"}`

2. **Format as Hydra overrides**:
   - `["backend.megatron.load=/outputs/lr1e-4_stable/checkpoint"]`

3. **Hydra reloads config** with these overrides:
   - Config contains: `backend.megatron.load: /outputs/lr1e-4_stable/checkpoint`

4. **OmegaConf `${...}` resolved** during Hydra reload:
   - `project.name: "lr${backend.megatron.lr}_..."` → `"lr1e-4_..."`

**Result:**
- `${...}` for own config - handled by OmegaConf/Hydra
- `{...}` for siblings - handled by our template engine
- No conflicts, clean separation of concerns

**Example combining both:**
```yaml
- stage: cooldown
  # Reference own config with ${...}
  aux:
    stable_train_iters: "{sibling.stable.backend.megatron.train_iters}"  # Sibling template
    target_iters: "${oc.eval:int(${.stable_train_iters}*1.2)}"  # OmegaConf expression

  # Use both in same value
  backend.megatron.load: "{sibling.stable.output_dir}/checkpoint_iter${aux.target_iters}"
  #                       ^^^^^^^^^^^^^^^^^^^^^^^^^ Sibling (resolved first)
  #                                                            ^^^^^^^^^^^^^^^^^ OmegaConf (resolved during Hydra reload)
```

**Recommendation:** Documentation should clearly state this distinction.

---

### 7. Filter Referencing Sibling Values

**Question:** Can filters reference sibling parameters?

**Example:**
```yaml
sweep:
  type: product
  groups:
    - type: list
      configs:
        - stage: stable
        - stage: cooldown
          # Only create cooldown if stable's lr > 1e-4?
          _filter: "{sibling.stable.backend.megatron.lr} > 1e-4"
```

**Analysis:**
- Filters evaluated during sweep expansion
- Sibling resolution happens during job planning (later)
- Would require two-pass expansion

**Recommendation:** Don't support in v1. Too complex. Use top-level filter instead:
```yaml
sweep:
  type: product
  groups: [...]
  filter: 'not (_stage == "cooldown" and backend.megatron.lr <= 1e-4)'
```

---

### 8. Multi-Stage Chains (4+ Stages)

**Support chains beyond 2 stages:**

```yaml
sweep:
  type: product
  groups:
    # Hyperparameter sweep
    - type: product
      params:
        backend.megatron.lr: [1e-4, 5e-4]
        backend.megatron.global_batch_size: [64, 128]

    # Multi-stage pipeline (4 stages)
    - type: list
      configs:
        # Stage 1: Pre-pre-training (foundation)
        - stage: pre_pre_training
          backend.megatron.aux.tokens: 10_000_000_000
          backend.megatron.train_iters: "${oc.eval:(${.aux.tokens}//${.seq_length}//${.global_batch_size})}"
          backend.megatron.lr_wsd_decay_iters: 0
          # No start_conditions - submits immediately

        # Stage 2: Pre-training (depends on pre-pre-training)
        - stage: pre_training
          backend.megatron.aux.tokens: 20_000_000_000
          backend.megatron.train_iters: "${oc.eval:(${.aux.tokens}//${.seq_length}//${.global_batch_size})}"
          backend.megatron.load: "{sibling.pre_pre_training.output_dir}/checkpoint/final"
          start_conditions:
            - class_name: SlurmStateCondition
              job_name: "{sibling.pre_pre_training.name}"
              state: COMPLETED

        # Stage 3: Mid-training (depends on pre-training)
        - stage: mid_training
          backend.megatron.aux.tokens: 30_000_000_000
          backend.megatron.train_iters: "${oc.eval:(${.aux.tokens}//${.seq_length}//${.global_batch_size})}"
          backend.megatron.load: "{sibling.pre_training.output_dir}/checkpoint/final"
          start_conditions:
            - class_name: SlurmStateCondition
              job_name: "{sibling.pre_training.name}"
              state: COMPLETED

        # Stage 4: Post-training (depends on mid-training)
        - stage: post_training
          backend.megatron.aux.tokens: 50_000_000_000
          backend.megatron.train_iters: "${oc.eval:(${.aux.tokens}//${.seq_length}//${.global_batch_size})}"
          backend.megatron.load: "{sibling.mid_training.output_dir}/checkpoint/final"
          backend.megatron.lr_wsd_decay_iters: "${oc.eval:int(${.train_iters}*0.2)}"  # Final cooldown
          backend.megatron.override_opt_param_scheduler: true
          start_conditions:
            - class_name: SlurmStateCondition
              job_name: "{sibling.mid_training.name}"
              state: COMPLETED
```

**Result:** 4 hyperparams × 4 stages = 16 jobs total

**Chain structure:**
```
lr1e-4_bsz64_pre_pre_training → lr1e-4_bsz64_pre_training → lr1e-4_bsz64_mid_training → lr1e-4_bsz64_post_training
lr1e-4_bsz128_pre_pre_training → ...
lr5e-4_bsz64_pre_pre_training → ...
lr5e-4_bsz128_pre_pre_training → ...
```

**Key points:**
- Each stage references only its **direct predecessor** (sibling)
- No need for "derived_jobs" concept - it's just a list group
- Chains of arbitrary length supported
- All dependencies resolved during planning phase

**Alternative: Branching after stage 2**
```yaml
- type: list
  configs:
    - stage: pre_pre_training
      # ...
    - stage: pre_training
      backend.megatron.load: "{sibling.pre_pre_training.output_dir}/checkpoint/final"
      # ...

    # Two parallel branches after pre_training
    - stage: mid_training_path_a
      backend.megatron.load: "{sibling.pre_training.output_dir}/checkpoint/final"
      backend.megatron.optimizer: adam
      # ...

    - stage: mid_training_path_b
      backend.megatron.load: "{sibling.pre_training.output_dir}/checkpoint/final"
      backend.megatron.optimizer: sgd
      # ...
```

This creates a DAG where pre_training → [mid_training_path_a, mid_training_path_b] (parallel branches).

---

## Open Questions / Discussion Points

### 1. Stage Dependency Validation

**Question:** Should we validate the dependency graph at planning time?

**Scenarios to catch:**
- ❌ Stage references sibling that doesn't exist: `{sibling.stabble.output_dir}` (typo)
- ❌ Circular dependencies: A → B → C → A
- ❌ Multiple stages with no start_conditions (ambiguous submission order)

**Recommendation:** YES - implement in validation phase (Pain Point #3). Catch these during planning, not runtime.

---

### 2. Job Indexing Strategy

**Question:** How to assign indices to jobs from composable sweeps?

**Current approach (sequential):**
```
Job 0: lr1e-4_bsz64_stable
Job 1: lr1e-4_bsz64_cooldown
Job 2: lr1e-4_bsz128_stable
Job 3: lr1e-4_bsz128_cooldown
Job 4: lr5e-4_bsz64_stable
...
```

This naturally groups "sibling families" together (indices 0-1, 2-3, 4-5, etc.).

**Alternative (separate stage blocks):**
```
Job 0: lr1e-4_bsz64_stable
Job 1: lr1e-4_bsz128_stable
Job 2: lr5e-4_bsz64_stable
Job 3: lr5e-4_bsz128_stable
Job 4: lr1e-4_bsz64_cooldown
...
```

**Recommendation:** Keep sequential (current approach). Natural result of product expansion, families grouped together.

---

### 3. Condition Evaluation Trigger

**Question:** When to evaluate start_conditions?

**Option A:** Every monitor cycle (simple, predictable)
```
while True:
    observe_active_jobs()
    check_waiting_jobs()  # Every cycle
    sleep(60)
```

**Option B:** Event-driven (efficient, complex)
```
on_job_state_change(job):
    for waiting_job in waiting_jobs:
        if waiting_job.depends_on(job):
            check_conditions(waiting_job)
```

**Recommendation:** Start with Option A (every cycle). Profile performance. Add Option B if bottleneck identified.

---

### 4. Sibling Job Failure Handling (DECISION: Timeout + Cancel Conditions)

**Question:** If a sibling job fails, what happens to jobs waiting for it?

**Scenario:**
```yaml
- stage: stable
- stage: cooldown
  backend.megatron.load: "{sibling.stable.output_dir}/checkpoint"
  start_conditions:
    - class_name: SlurmStateCondition
      job_name: "{sibling.stable.name}"
      state: COMPLETED
      timeout_seconds: 86400  # Wait max 24 hours
```

**Primary Approach: Timeout**
Use `timeout_seconds` in start_conditions. If stable fails and doesn't recover, timeout triggers and cooldown is marked as failed/skipped.

**Enhanced Approach: Cancel Conditions** ⭐

Introduce `cancel_conditions` to actively detect sibling failures:

```yaml
- stage: cooldown
  backend.megatron.load: "{sibling.stable.output_dir}/checkpoint"

  # Wait for success
  start_conditions:
    - class_name: SlurmStateCondition
      job_name: "{sibling.stable.name}"
      state: COMPLETED
      timeout_seconds: 86400

  # Cancel on failure (checked in parallel)
  cancel_conditions:
    - class_name: SlurmStateCondition
      job_name: "{sibling.stable.name}"
      state: FAILED

    - class_name: LogPatternCondition
      log_path: "{sibling.stable.log_path_current}"  # Symlink to current log
      pattern: "FATAL ERROR|OutOfMemoryError|CUDA error"
      description: "Cancel cooldown if stable hits fatal error"
```

**Implementation:**
- Monitor checks both `start_conditions` and `cancel_conditions` each cycle
- If `start_conditions` satisfied → submit job
- If `cancel_conditions` satisfied → mark job as cancelled/skipped
- Whichever triggers first wins

**Log File Naming Strategy:**

To support `cancel_conditions` that monitor sibling logs, we need stable log paths resolvable at planning time.

**Proposed Structure:**
```
/logs/
  lr1e-4_bsz64_stable/
    slurm-12345.out          # Actual log (includes SLURM job ID)
    slurm-12346.out          # Retry 1
    slurm-12347.out          # Retry 2
    current.log -> slurm-12347.out  # Symlink to most recent
```

**Benefits:**
- ✅ `current.log` has stable path, resolvable at planning time
- ✅ Historical logs preserved with SLURM IDs (audit trail)
- ✅ Re-runs create new log file + update symlink
- ✅ Cancel conditions can reference: `{sibling.stable.log_path_current}`

**Template Variables:**
```yaml
{sibling.stable.log_path_current}  # → /logs/lr1e-4_bsz64_stable/current.log (symlink)
{sibling.stable.log_dir}           # → /logs/lr1e-4_bsz64_stable/
{sibling.stable.log_path}          # → /logs/lr1e-4_bsz64_stable/slurm-%j.out (template)
```

**SLURM Script Update:**
```bash
#!/bin/bash
#SBATCH --output=/logs/lr1e-4_bsz64_stable/slurm-%j.out

# Create/update symlink to this log
ln -sf slurm-${SLURM_JOB_ID}.out /logs/lr1e-4_bsz64_stable/current.log

# Run training
srun ...
```

**Alternative: No Symlink (Simpler)**
If symlinks are problematic, use pattern matching:
```yaml
cancel_conditions:
  - class_name: LogPatternCondition
    log_pattern: "{sibling.stable.log_dir}/slurm-*.out"  # Check all logs
    pattern: "FATAL ERROR"
    check_mode: any  # Match if ANY log contains pattern
```

**DECISION: Implement symlink approach.** Cleaner, more explicit, easier to debug.

---

### 5. Multiple Sibling Dependencies (DECISION: Keep as Future Extension)

**Question:** Should a job be able to depend on multiple siblings?

**Use Case:** Ensemble evaluation depends on 3 different trained models.

**Proposed Config:**
```yaml
- type: list
  configs:
    # Three model variants train in parallel
    - stage: train_model_a
    - stage: train_model_b
    - stage: train_model_c

    # Ensemble depends on all three
    - stage: ensemble_eval
      # Reference multiple siblings
      backend.models: [
        "{sibling.train_model_a.output_dir}/checkpoint",
        "{sibling.train_model_b.output_dir}/checkpoint",
        "{sibling.train_model_c.output_dir}/checkpoint"
      ]
      start_conditions:
        - class_name: CompositeCondition
          mode: all  # All must complete
          conditions:
            - class_name: SlurmStateCondition
              job_name: "{sibling.train_model_a.name}"
              state: COMPLETED
            - class_name: SlurmStateCondition
              job_name: "{sibling.train_model_b.name}"
              state: COMPLETED
            - class_name: SlurmStateCondition
              job_name: "{sibling.train_model_c.name}"
              state: COMPLETED
```

**DECISION:** Not necessary for v1, but keep as possibility. The implementation naturally supports this:
- Template resolution handles multiple `{sibling.*}` references
- CompositeCondition already exists
- No special code needed

**For v1:** Focus on linear chains (A → B → C → D) and simple branching.

---

### 6. Visualization (DECISION: ASCII → ncurses → Web)

**Question:** How to visualize multi-stage experiments?

**DECISION: Phased Implementation**

**Phase 1 (v1): ASCII Text Visualization** ⭐

**Script:** `scripts/visualize_plan.py`

**Usage:**
```bash
# Visualize from config
python scripts/visualize_plan.py --config-ref experiments/my_experiment

# With Hydra overrides
python scripts/visualize_plan.py --config-ref experiments/my_experiment \
    backend.megatron.lr=1e-4 \
    sweep.grids.0.backend.megatron.global_batch_size=[64,128]

# Visualize from existing plan manifest
python scripts/visualize_plan.py --manifest monitoring_state/manifests/plan_20250115_abc123.json
```

**Output:**
```
Multi-Stage Experiment Plan: dense_300M_sweep
Total: 12 jobs (6 base × 2 stages)

Dependency Graph:
┌──────────────────────────────────────────────────────────────┐
│ Hyperparameter Sweep (6 combinations)                       │
│ • lr: [2.5e-4, 5e-4, 1e-3]                                  │
│ • global_batch_size: [64, 128]                              │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ Stage: stable (6 jobs)                                       │
│ • lr1e-4_bsz64_stable                                       │
│ • lr1e-4_bsz128_stable                                      │
│ • lr5e-4_bsz64_stable                                       │
│ • lr5e-4_bsz128_stable                                      │
│ • lr1e-3_bsz64_stable                                       │
│ • lr1e-3_bsz128_stable                                      │
│ Start: Immediate                                             │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ Stage: cooldown (6 jobs)                                     │
│ • lr1e-4_bsz64_cooldown                                     │
│   ├─ Waits for: lr1e-4_bsz64_stable                         │
│   └─ Condition: checkpoint at iter 80000                     │
│ • lr1e-4_bsz128_cooldown                                    │
│   ├─ Waits for: lr1e-4_bsz128_stable                        │
│   └─ Condition: checkpoint at iter 80000                     │
│ ...                                                          │
│ Start: After sibling completion                              │
└──────────────────────────────────────────────────────────────┘

Validation: ✓ All dependencies resolved
Validation: ✓ No circular dependencies
Validation: ✓ All job names unique
```

**Implementation:**
- Run during `plan_autoexp.py`
- Shows dependency tree
- Lists start/cancel conditions
- Displays validation results

**Phase 2 (v2): Interactive ncurses Interface**

**Script:** `scripts/monitor_autoexp.py --interactive`

**Usage:**
```bash
# Interactive monitoring from manifest
python scripts/monitor_autoexp.py --interactive --manifest monitoring_state/manifests/plan_abc123.json

# Or from session ID
python scripts/monitor_autoexp.py --interactive --session-id abc123
```

**Features:**
- Live updating job states
- Interactive stage inspection
- Keyboard navigation (arrow keys, vim keys)
- Condition status in real-time
- Log tail view

**Example Layout:**
```
╔══════════════════════════════════════════════════════════════╗
║ Multi-Stage Monitor: dense_300M_sweep                       ║
║ Jobs: 4/12 Running | 4/12 Waiting | 4/12 Completed         ║
╠══════════════════════════════════════════════════════════════╣
║ [↑↓] Navigate  [Enter] Inspect  [l] Logs  [q] Quit         ║
╠══════════════════════════════════════════════════════════════╣
║ Stage: stable (6 jobs)                          [COMPLETED] ║
║ ✓ lr1e-4_bsz64_stable      (23:45:12)                      ║
║ ✓ lr1e-4_bsz128_stable     (23:47:08)                      ║
║ ✓ lr5e-4_bsz64_stable      (23:46:54)                      ║
║ ...                                                          ║
║                                                              ║
║ Stage: cooldown (6 jobs)                        [RUNNING]   ║
║ ⟳ lr1e-4_bsz64_cooldown    Running (42% done)              ║
║   └─ Condition: ✓ Checkpoint found                          ║
║ ⏸ lr1e-4_bsz128_cooldown   Waiting                         ║
║   └─ Condition: ⧗ Checking checkpoint...                   ║
║ ...                                                          ║
╠══════════════════════════════════════════════════════════════╣
║ Log (lr1e-4_bsz64_cooldown):                               ║
║ [2025-01-15 10:23:45] Iteration 150 loss=2.34              ║
║ [2025-01-15 10:23:50] Iteration 151 loss=2.32              ║
╚══════════════════════════════════════════════════════════════╝
```

**Libraries:**
- `blessed` or `curses` for terminal UI
- Real-time updates without flickering

**Phase 3 (v3+): Web Dashboard**
- Graphical DAG view
- Click to inspect jobs
- Timeline view
- Mobile-friendly

**For v1:** Implement ASCII visualization only. Simple, effective, no dependencies.

---

## Script Structure and CLI Conventions

**CRITICAL:** oellm-autoexp uses **separate scripts** in the `scripts/` directory, NOT a subcommand structure.

### Why Separate Scripts?

To maintain **Hydra override compatibility**:

```bash
# ✅ CORRECT - Hydra overrides work naturally
python scripts/run_autoexp.py --config-ref experiments/my_exp \
    backend.megatron.lr=1e-4 \
    backend.megatron.global_batch_size=64

# ❌ WRONG - Would require special parsing
oellm-autoexp run --config-ref experiments/my_exp backend.megatron.lr=1e-4
```

Hydra expects overrides as positional arguments after standard argparse flags. Separate scripts preserve this interface.

### Existing Scripts (scripts/)

**Core Workflow:**
- `run_autoexp.py` - Main entry point: plan + render + submit + monitor
- `plan_autoexp.py` - Planning only: expand sweep, resolve templates, validate, render scripts
- `submit_autoexp.py` - Submission only: submit jobs from manifest
- `monitor_autoexp.py` - Monitoring only: track jobs, check conditions, trigger actions

**Container Support:**
- `run_autoexp_container.py` - Containerized version of run_autoexp.py

**Utilities:**
- `manage_monitoring.py` - Manage monitoring sessions (list, stop, restart)
- `render_config.py` - Render Hydra config to JSON/YAML
- `run_sweep_entry.py` - Execute single sweep point (for SLURM array jobs)

**Code Generation:**
- `generate_megatron_config.py` - Generate Megatron config from dataclass
- `generate_megatron_dataclass.py` - Generate dataclass from Megatron argparse

### New Scripts for Multi-Stage (v1)

**Add to scripts/:**
- `visualize_plan.py` - ASCII visualization of execution plan (Phase 1)

**Future (v2+):**
- (Maybe extend `monitor_autoexp.py --interactive` for ncurses)
- (Maybe extend `visualize_plan.py --web` for web dashboard)

### Script Naming Convention

**Pattern:** `<verb>_<noun>.py`

Examples:
- `run_autoexp.py` - Run an autoexp experiment
- `plan_autoexp.py` - Plan an autoexp experiment
- `submit_autoexp.py` - Submit jobs
- `monitor_autoexp.py` - Monitor jobs
- `visualize_plan.py` - Visualize plan
- `manage_monitoring.py` - Manage monitoring sessions

### Hydra Integration

All scripts that accept `--config-ref` should support Hydra overrides:

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-ref", default="autoexp")
    parser.add_argument("-C", "--config-dir", type=Path, default=Path("config"))
    # ... other flags ...
    parser.add_argument(
        "override", nargs="*", default=[],
        help="Hydra-style overrides (key=value)."
    )
    return parser.parse_args(argv)
```

**Usage:**
```bash
python scripts/<script>.py --config-ref my_config key1=value1 key2=value2
```

### Integration with Existing Workflow

```bash
# 1. Plan and visualize
python scripts/plan_autoexp.py --config-ref experiments/dense_300M
python scripts/visualize_plan.py --config-ref experiments/dense_300M

# 2. Review plan, then submit
python scripts/submit_autoexp.py --manifest monitoring_state/manifests/plan_abc123.json

# 3. Monitor (blocking or detached)
python scripts/monitor_autoexp.py --manifest monitoring_state/manifests/plan_abc123.json

# 4. Interactive monitoring (v2)
python scripts/monitor_autoexp.py --interactive --session-id abc123

# Or all-in-one (current default)
python scripts/run_autoexp.py --config-ref experiments/dense_300M
```

---

## Test Planning

### Testing Strategy

All tests must be runnable **locally on a laptop without a real SLURM cluster**. This requires:
1. Mock SLURM client (FakeSlurmClient)
2. Mock backend (NullBackend or simple MockBackend)
3. Temporary directories for outputs
4. No external dependencies (GPUs, containers, etc.)

### Test Categories

#### 1. Unit Tests: Composable Sweep Expansion

**Location:** `tests/unit/test_sweep_composable.py`

**Test cases:**

```python
def test_product_mode_basic():
    """Test basic product mode sweep expansion."""
    sweep = SweepConfig(
        type="product",
        groups=[
            ProductGroup(
                params={
                    "backend.megatron.lr": [1e-4, 5e-4],
                    "backend.megatron.global_batch_size": [64, 128],
                }
            )
        ]
    )
    points = expand_sweep(sweep)
    assert len(points) == 4  # 2 × 2
    # Verify all combinations present


def test_list_mode_basic():
    """Test list mode - no cross-product."""
    sweep = SweepConfig(
        type="product",
        groups=[
            ListGroup(
                configs=[
                    {"stage": "stable", "aux.tokens": 50_000_000_000},
                    {"stage": "cooldown", "aux.tokens": 60_000_000_000},
                ]
            )
        ]
    )
    points = expand_sweep(sweep)
    assert len(points) == 2  # No cross-product
    assert points[0].parameters["stage"] == "stable"
    assert points[1].parameters["stage"] == "cooldown"


def test_product_list_composition():
    """Test combining product and list groups."""
    sweep = SweepConfig(
        type="product",
        groups=[
            # Hyperparameters (product)
            ProductGroup(
                params={
                    "backend.megatron.lr": [1e-4, 5e-4],
                    "backend.megatron.global_batch_size": [64, 128],
                }
            ),
            # Stages (list)
            ListGroup(
                configs=[
                    {"stage": "stable"},
                    {"stage": "cooldown"},
                ]
            ),
        ]
    )
    points = expand_sweep(sweep)
    assert len(points) == 8  # 4 (hyperparams) × 2 (stages)

    # Verify we have both stages for each hyperparam combination
    stable_points = [p for p in points if p.parameters.get("stage") == "stable"]
    cooldown_points = [p for p in points if p.parameters.get("stage") == "cooldown"]
    assert len(stable_points) == 4
    assert len(cooldown_points) == 4


def test_nested_list_product():
    """Test multiple product groups with list group."""
    sweep = SweepConfig(
        type="product",
        groups=[
            ProductGroup(params={"a": [1, 2]}),  # 2 points
            ListGroup(configs=[{"b": 10}, {"b": 20}, {"b": 30}]),  # 3 points (no cross)
            ProductGroup(params={"c": [100, 200]}),  # 2 points
        ]
    )
    points = expand_sweep(sweep)
    assert len(points) == 12  # 2 × 3 × 2


def test_filter_at_group_level():
    """Test filter applied within a product group."""
    sweep = SweepConfig(
        type="product",
        groups=[
            ProductGroup(
                params={
                    "a": [1, 2, 3, 4],
                    "b": [10, 20, 30],
                },
                filter="a * b <= 60"  # Prune within group
            )
        ]
    )
    points = expand_sweep(sweep)
    # Should exclude: (3,30), (3,20)?, (4,20), (4,30), etc.
    assert all(p.parameters["a"] * p.parameters["b"] <= 60 for p in points)
    assert len(points) < 12  # Less than full product


def test_filter_at_top_level():
    """Test filter applied to final sweep points."""
    sweep = SweepConfig(
        type="product",
        groups=[
            ProductGroup(params={"a": [1, 2]}),
            ListGroup(configs=[{"stage": "stable"}, {"stage": "cooldown"}]),
        ],
        filter='not (a == 1 and stage == "cooldown")'  # Prune final combinations
    )
    points = expand_sweep(sweep)
    assert len(points) == 3  # Excluded: (a=1, cooldown)
    # Verify exclusion
    assert not any(
        p.parameters["a"] == 1 and p.parameters.get("stage") == "cooldown"
        for p in points
    )


def test_backward_compatibility_grids():
    """Test that old 'grids' syntax still works."""
    sweep = SweepConfig(
        grids=[
            {"backend.megatron.lr": [1e-4, 5e-4]},
            {"backend.megatron.global_batch_size": [64, 128]},
        ]
    )
    points = expand_sweep(sweep)
    # Old grids syntax: each grid is independent (not crossed with each other)
    assert len(points) == 4  # 2 + 2 points (if independent) OR 4 if crossed?
    # TODO: Clarify old grids semantics - are they crossed?
```

#### 2. Unit Tests: Sibling Reference Resolution

**Location:** `tests/unit/test_sibling_references.py`

**Test cases:**

```python
def test_find_sibling_by_stage():
    """Test finding sibling by stage parameter."""
    points = [
        SweepPoint(parameters={"backend.megatron.lr": 1e-4, "stage": "stable"}),
        SweepPoint(parameters={"backend.megatron.lr": 1e-4, "stage": "cooldown"}),
        SweepPoint(parameters={"backend.megatron.lr": 5e-4, "stage": "stable"}),
        SweepPoint(parameters={"backend.megatron.lr": 5e-4, "stage": "cooldown"}),
    ]

    # From point 1 (cooldown, lr=1e-4), find sibling with stage=stable
    sibling_idx, sibling = find_sibling("stable", points, current_index=1)
    assert sibling_idx == 0
    assert sibling.parameters["stage"] == "stable"
    assert sibling.parameters["backend.megatron.lr"] == 1e-4  # Same family


def test_find_sibling_different_families():
    """Test that siblings must be from same family (same base params)."""
    points = [
        SweepPoint(parameters={"backend.megatron.lr": 1e-4, "stage": "stable"}),
        SweepPoint(parameters={"backend.megatron.lr": 1e-4, "stage": "cooldown"}),
        SweepPoint(parameters={"backend.megatron.lr": 5e-4, "stage": "stable"}),
    ]

    # From point 2 (lr=5e-4, stable), cannot find sibling with stage=cooldown and lr=5e-4
    # because there's no such point
    with pytest.raises(ValueError, match="No sibling found"):
        find_sibling("cooldown", points, current_index=2)


def test_resolve_sibling_output_dir():
    """Test resolving {sibling.stable.output_dir}."""
    # Setup sweep points with sibling references
    config_str = """
    - stage: stable
      backend.megatron.lr: 1e-4
    - stage: cooldown
      backend.megatron.lr: 1e-4
      backend.megatron.load: "{sibling.stable.output_dir}/checkpoint"
    """

    points = [
        SweepPoint(parameters={"backend.megatron.lr": 1e-4, "stage": "stable"}),
        SweepPoint(
            parameters={
                "backend.megatron.lr": 1e-4,
                "stage": "cooldown",
                "backend.megatron.load": "{sibling.stable.output_dir}/checkpoint",
            }
        ),
    ]

    # Build jobs (which resolves templates)
    jobs_meta = [
        JobPlan(name="job_stable", output_dir="/outputs/job_stable", parameters=points[0].parameters),
    ]

    resolved = resolve_sibling_templates(points[1], points, current_index=1, jobs_meta=jobs_meta)

    assert resolved.parameters["backend.megatron.load"] == "/outputs/job_stable/checkpoint"


def test_resolve_sibling_parameter():
    """Test resolving {sibling.stable.backend.megatron.train_iters}."""
    points = [
        SweepPoint(parameters={
            "backend.megatron.lr": 1e-4,
            "stage": "stable",
            "backend.megatron.train_iters": 100000,
        }),
        SweepPoint(parameters={
            "backend.megatron.lr": 1e-4,
            "stage": "cooldown",
            "aux.stable_iters": "{sibling.stable.backend.megatron.train_iters}",
        }),
    ]

    jobs_meta = [
        JobPlan(name="job_stable", output_dir="/outputs/job_stable", parameters=points[0].parameters),
    ]

    resolved = resolve_sibling_templates(points[1], points, current_index=1, jobs_meta=jobs_meta)

    assert resolved.parameters["aux.stable_iters"] == "100000"  # Resolved to string


def test_resolve_runtime_metadata_template():
    """Test that runtime metadata templates are preserved for later resolution."""
    points = [
        SweepPoint(parameters={"stage": "stable"}),
        SweepPoint(parameters={
            "stage": "cooldown",
            "condition_key": "{sibling.stable.metadata.checkpoint_iteration}",
        }),
    ]

    jobs_meta = [
        JobPlan(name="job_stable", output_dir="/outputs/job_stable", parameters=points[0].parameters),
    ]

    resolved = resolve_sibling_templates(points[1], points, current_index=1, jobs_meta=jobs_meta)

    # Runtime metadata should be converted to {runtime...} template
    assert "{runtime.job_stable.checkpoint_iteration}" in resolved.parameters["condition_key"]


def test_nested_sibling_references():
    """Test multiple sibling references in a single parameter value."""
    points = [
        SweepPoint(parameters={"stage": "stable"}),
        SweepPoint(parameters={
            "stage": "cooldown",
            "complex_path": "{sibling.stable.output_dir}/checkpoints/iter_{sibling.stable.backend.megatron.target_iter}",
        }),
    ]

    jobs_meta = [
        JobPlan(
            name="job_stable",
            output_dir="/outputs/job_stable",
            parameters={"stage": "stable", "backend.megatron.target_iter": 80000},
        ),
    ]

    resolved = resolve_sibling_templates(points[1], points, current_index=1, jobs_meta=jobs_meta)

    assert resolved.parameters["complex_path"] == "/outputs/job_stable/checkpoints/iter_80000"
```

#### 3. Integration Tests: Full Multi-Stage Pipeline

**Location:** `tests/integration/test_multi_stage_pipeline.py`

**Test cases:**

```python
def test_two_stage_pipeline_local(tmp_path):
    """
    Test a complete two-stage pipeline (stable + cooldown) with mock backend/slurm.

    This is the comprehensive test that validates the entire flow.
    """
    # Setup config directory
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Create minimal config with composable sweep
    config_content = f"""
# @package _global_
project:
  name: "test_lr{{backend.megatron.lr}}_{{stage}}"
  base_output_dir: "{tmp_path}/outputs"

backend:
  class_name: "MockBackend"  # Simple mock that succeeds immediately
  megatron:
    lr: 1e-4
    seq_length: 4096
    global_batch_size: 64
    save_interval: 100
    train_iters: 1000

slurm:
  template_path: "dummy.sbatch"
  script_dir: "{tmp_path}/scripts"
  log_dir: "{tmp_path}/logs"
  launcher_cmd: ""
  srun_opts: ""
  client:
    class_name: "FakeSlurmClient"

monitoring:
  class_name: "MockMonitor"  # Mock that simulates checkpoint events
  state_dir: "{tmp_path}/monitoring_state"
  log_events:
    - name: checkpoint_saved
      pattern: 'saved checkpoint from iteration\\s+(?P<iteration>\\d+) to (?P<path>\\S+)'
      extract_groups:
        checkpoint_iteration: iteration
        checkpoint_path: path

sweep:
  type: product
  groups:
    # Simple hyperparam sweep
    - type: product
      params:
        backend.megatron.lr: [1e-4, 5e-4]

    # Stages (list mode)
    - type: list
      configs:
        - stage: stable
          backend.megatron.train_iters: 1000
          # No start_conditions

        - stage: cooldown
          backend.megatron.train_iters: 1200
          backend.megatron.load: "{{sibling.stable.output_dir}}/checkpoint"
          start_conditions:
            - class_name: FileExistsCondition
              path: "{{sibling.stable.output_dir}}/checkpoint/done.txt"
              blocking: true
              timeout_seconds: 10
"""

    config_file = config_dir / "test_multi_stage.yaml"
    config_file.write_text(config_content)

    # Load config and expand sweep
    root = load_config_reference("test_multi_stage", config_dir, overrides=[])
    sweep_points = expand_sweep(root.sweep)

    # Verify expansion
    assert len(sweep_points) == 4  # 2 lr × 2 stages
    stable_points = [p for p in sweep_points if p.parameters.get("stage") == "stable"]
    cooldown_points = [p for p in sweep_points if p.parameters.get("stage") == "cooldown"]
    assert len(stable_points) == 2
    assert len(cooldown_points) == 2

    # Build job plans (resolve sibling templates)
    plan = build_execution_plan(root, config_setup=ConfigSetup(...))

    # Verify sibling references resolved
    cooldown_jobs = [j for j in plan.jobs if "stage" in j.parameters and j.parameters["stage"] == "cooldown"]
    assert len(cooldown_jobs) == 2

    for cooldown_job in cooldown_jobs:
        # Check that load path references sibling's output_dir
        load_path = cooldown_job.parameters.get("backend.megatron.load")
        assert load_path is not None
        assert "/outputs/test_lr" in load_path  # Resolved to actual path
        assert "sibling" not in load_path  # Template resolved

        # Check start_conditions
        assert len(cooldown_job.start_conditions) == 1
        condition = cooldown_job.start_conditions[0]
        assert condition.class_name == "FileExistsCondition"
        assert "sibling" not in condition.path  # Template resolved

    # Simulate submission and monitoring
    orchestrator = Orchestrator(plan)

    # Submit all jobs
    orchestrator.submit_jobs()

    # Check that stable jobs submitted immediately
    fake_slurm = orchestrator.slurm_client
    submitted_jobs = fake_slurm.get_submitted_jobs()
    assert len(submitted_jobs) == 2  # Only stable jobs submitted initially

    # Check that cooldown jobs are waiting
    waiting_jobs = orchestrator.get_waiting_jobs()
    assert len(waiting_jobs) == 2

    # Simulate stable jobs completing
    for stable_job in submitted_jobs:
        # Create checkpoint file to satisfy cooldown start condition
        output_dir = Path(stable_job.output_dir)
        checkpoint_dir = output_dir / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "done.txt").write_text("done")

        # Mark job as completed
        fake_slurm.complete_job(stable_job.job_id)

    # Run monitor cycle
    orchestrator.monitor_cycle()

    # Check that cooldown jobs now submitted
    submitted_jobs_after = fake_slurm.get_submitted_jobs()
    assert len(submitted_jobs_after) == 4  # All 4 jobs submitted

    waiting_jobs_after = orchestrator.get_waiting_jobs()
    assert len(waiting_jobs_after) == 0  # No more waiting


def test_three_stage_pipeline(tmp_path):
    """Test a three-stage pipeline: train → eval_val → eval_test."""
    # Similar structure to two-stage but with branching
    # train → eval_val (parallel)
    #      → eval_test
    pass


def test_filter_prunes_stage_combinations(tmp_path):
    """Test that filters can exclude specific stage combinations."""
    # For example, filter: 'not (lr == 1e-4 and stage == "cooldown")'
    # Should create: (1e-4, stable), (5e-4, stable), (5e-4, cooldown)
    # Excludes: (1e-4, cooldown)
    pass


def test_multiple_cooldown_scales(tmp_path):
    """
    Test multiple cooldown stages at different scales (like dense_300M_50BT.yaml).

    Structure:
    - stable (50B tokens)
    - cooldown_6B
    - cooldown_12B
    - cooldown_30B
    - cooldown_50B

    All cooldowns depend on stable reaching different checkpoints.
    """
    pass


def test_cancel_conditions_on_sibling_failure(tmp_path):
    """
    Test that cancel_conditions trigger when sibling fails.

    This tests the log file symlink strategy and cancel condition detection.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    config_content = f"""
# @package _global_
project:
  name: "test_lr{{backend.megatron.lr}}_{{stage}}"
  base_output_dir: "{tmp_path}/outputs"

backend:
  class_name: "MockBackend"
  megatron:
    lr: 1e-4

slurm:
  template_path: "dummy.sbatch"
  script_dir: "{tmp_path}/scripts"
  log_dir: "{tmp_path}/logs"
  client:
    class_name: "FakeSlurmClient"

monitoring:
  class_name: "MockMonitor"
  state_dir: "{tmp_path}/monitoring_state"

sweep:
  type: product
  groups:
    - type: list
      configs:
        - stage: stable
          backend.megatron.train_iters: 1000

        - stage: cooldown
          backend.megatron.train_iters: 1200
          backend.megatron.load: "{{sibling.stable.output_dir}}/checkpoint"

          # Start when stable completes
          start_conditions:
            - class_name: SlurmStateCondition
              job_name: "{{sibling.stable.name}}"
              state: COMPLETED
              timeout_seconds: 60

          # Cancel if stable fails
          cancel_conditions:
            - class_name: SlurmStateCondition
              job_name: "{{sibling.stable.name}}"
              state: FAILED

            - class_name: LogPatternCondition
              log_path: "{{sibling.stable.log_path_current}}"
              pattern: "FATAL ERROR"
"""

    config_file = config_dir / "test_cancel.yaml"
    config_file.write_text(config_content)

    # Load and build plan
    root = load_config_reference("test_cancel", config_dir, overrides=[])
    plan = build_execution_plan(root, config_setup=ConfigSetup(...))

    assert len(plan.jobs) == 2  # stable + cooldown

    # Setup log directory with symlink
    stable_job = plan.jobs[0]
    log_dir = Path(tmp_path) / "logs" / stable_job.name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create actual log file
    actual_log = log_dir / "slurm-12345.out"
    actual_log.write_text("Starting training...")

    # Create symlink
    current_log = log_dir / "current.log"
    current_log.symlink_to(actual_log.name)

    # Submit jobs
    orchestrator = Orchestrator(plan)
    orchestrator.submit_jobs()

    # Stable submitted, cooldown waiting
    assert len(orchestrator.get_waiting_jobs()) == 1

    # Simulate stable job running
    fake_slurm = orchestrator.slurm_client
    fake_slurm.set_job_state(stable_job.job_id, "RUNNING")

    # Write FATAL ERROR to log (by way of symlink)
    actual_log.write_text("Starting training...\nFATAL ERROR: Out of memory\n")

    # Run monitor cycle
    orchestrator.monitor_cycle()

    # Check that cooldown was cancelled (not submitted)
    waiting_jobs = orchestrator.get_waiting_jobs()
    assert len(waiting_jobs) == 0

    # Check that cooldown marked as cancelled
    cooldown_job = plan.jobs[1]
    job_state = orchestrator.get_job_state(cooldown_job.name)
    assert job_state.status == "cancelled"
    assert "cancel_condition" in job_state.reason.lower()
```

#### 4. Mock Components

**Location:** `tests/mocks.py`

**Components needed:**

```python
class MockBackend(BaseBackend):
    """Simple mock backend that succeeds immediately."""

    def validate(self, spec: BackendJobSpec) -> None:
        pass  # Always valid

    def build_launch_command(self, spec: BackendJobSpec) -> list[str]:
        return ["echo", "Mock training running"]

    def check_health(self) -> bool:
        return True


class FakeSlurmClient(BaseSlurmClient):
    """In-memory fake SLURM client for testing."""

    def __init__(self):
        self.jobs: dict[str, FakeJob] = {}
        self.next_job_id = 1

    def submit(self, script_path: str) -> str:
        job_id = f"FAKE_{self.next_job_id}"
        self.next_job_id += 1
        self.jobs[job_id] = FakeJob(
            job_id=job_id,
            state="PENDING",
            script_path=script_path,
        )
        return job_id

    def get_job_state(self, job_id: str) -> str:
        return self.jobs[job_id].state

    def complete_job(self, job_id: str) -> None:
        """Helper for tests: mark job as completed."""
        self.jobs[job_id].state = "COMPLETED"

    def get_submitted_jobs(self) -> list[FakeJob]:
        return list(self.jobs.values())


class MockMonitor(BaseMonitor):
    """Mock monitor that can simulate checkpoint events."""

    def __init__(self, state_dir: str):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.job_states: dict[str, JobRuntimeState] = {}

    def observe_job(self, job: JobPlan) -> JobRuntimeState:
        if job.name not in self.job_states:
            self.job_states[job.name] = JobRuntimeState(
                registration=job,
                extracted_metadata={},
            )
        return self.job_states[job.name]

    def simulate_checkpoint_event(self, job_name: str, iteration: int, path: str):
        """Test helper: inject checkpoint metadata."""
        if job_name in self.job_states:
            self.job_states[job_name].extracted_metadata["checkpoint_iteration"] = str(iteration)
            self.job_states[job_name].extracted_metadata["checkpoint_path"] = path


@dataclass
class FakeJob:
    job_id: str
    state: str  # PENDING, RUNNING, COMPLETED, FAILED
    script_path: str
```

#### 5. Edge Case Tests

**Location:** `tests/unit/test_edge_cases.py`

**Test cases:**

```python
def test_sibling_not_found_raises_error():
    """Test that referencing non-existent sibling raises clear error."""
    points = [
        SweepPoint(parameters={"stage": "cooldown"}),  # No stable sibling
    ]

    with pytest.raises(ValueError, match="No sibling found matching pattern: stable"):
        resolve_sibling_templates(points[0], points, current_index=0, jobs_meta=[])


def test_circular_sibling_reference_detection():
    """Test detection of circular references."""
    # A → B → A (circular)
    # This shouldn't happen with stages, but test defensive programming
    pass


def test_omegaconf_expressions_with_aux_params():
    """
    Test that OmegaConf expressions can reference aux params that have sibling values.

    Example:
    aux:
      stable_train_iters: "{sibling.stable.train_iters}"  # Resolved to "100000"
      target_iters: "${oc.eval:int(${.stable_train_iters}*1.2)}"  # OmegaConf sees "100000"
    """
    pass


def test_empty_list_group():
    """Test that empty list group is handled gracefully."""
    sweep = SweepConfig(
        type="product",
        groups=[
            ListGroup(configs=[]),  # Empty
        ]
    )
    points = expand_sweep(sweep)
    assert len(points) == 0


def test_filter_excludes_all_points():
    """Test that filter excluding all points returns empty sweep."""
    sweep = SweepConfig(
        type="product",
        groups=[
            ProductGroup(params={"a": [1, 2, 3]}),
        ],
        filter="a > 10"  # Excludes all
    )
    points = expand_sweep(sweep)
    assert len(points) == 0
```

### Test Execution Plan

**Phase 1: Implement Sweep Expansion**
1. Implement `ProductGroup` and `ListGroup` schema
2. Implement `expand_sweep()` with product/list mode support
3. Add filter evaluation
4. Run unit tests: `test_sweep_composable.py`

**Phase 2: Implement Sibling References**
1. Implement `find_sibling()` logic
2. Implement `resolve_sibling_templates()`
3. Add template pattern matching
4. Run unit tests: `test_sibling_references.py`

**Phase 3: Integration Testing**
1. Implement mock components (MockBackend, FakeSlurmClient, MockMonitor)
2. Build integration test for two-stage pipeline
3. Verify end-to-end flow
4. Run integration tests: `test_multi_stage_pipeline.py`

**Phase 4: Edge Cases & Refinement**
1. Add edge case tests
2. Fix discovered issues
3. Add error handling
4. Documentation

### Success Criteria

**All tests must pass** with:
- ✅ Sweep expansion produces correct number of points
- ✅ Product/list modes work as expected
- ✅ Filters prune correctly
- ✅ Sibling references resolve correctly
- ✅ No OmegaConf conflicts
- ✅ Full pipeline runs locally without real SLURM
- ✅ Start conditions gate job submission correctly
- ✅ Monitor loop submits waiting jobs when conditions satisfied

### Test Data

**Example test config directory structure:**
```
tests/
  integration/
    test_configs/
      multi_stage_basic.yaml
      multi_stage_branching.yaml
      multi_stage_progressive_cooldown.yaml
  unit/
    test_sweep_composable.py
    test_sibling_references.py
    test_edge_cases.py
  mocks.py
```

This comprehensive test suite ensures:
1. **Correctness** - Sweep expansion and sibling resolution work
2. **No regressions** - Existing tests still pass
3. **Local development** - No cluster required
4. **Edge cases** - Error conditions handled gracefully
