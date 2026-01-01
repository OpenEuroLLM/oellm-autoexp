# Sweep Resolution Ordering: Design Options

## Problem Statement

The current sweep expansion and job planning system has a fundamental issue with resolution timing and sibling matching:

### What Works: `expand_sweep()`
✅ The sweep expansion correctly produces `SweepPoint` objects with:
- Parameter dictionaries with all sweep variables merged
- `group_path` tuples tracking position in the sweep hierarchy
- Proper handling of group-level defaults and nested structures

Example: For a sweep with 15 hyperparameter configs × 5 stages = 75 jobs:
- `expand_sweep()` correctly produces 75 SweepPoints
- Each has the right parameters (lr, batch_size, tokens, stage, etc.)
- Each has a `group_path` like `(0, 0)` or `(1, 2, 1)` tracking its position in the group hierarchy

### What Doesn't Work: `build_job_plans()` Resolution Timing

❌ The job planning tries to resolve configurations **too early**, before sibling context is available:

1. **Premature Resolution**: `build_job_plans()` tries to resolve interpolations like `${backend.megatron.aux.start_iter_round}` immediately, but these depend on:
   - Stage-specific `tokens` values (varies per stage)
   - Sibling checkpoint paths (only known after sibling is resolved)
   - Computed values like `train_iters` (depends on stage tokens)

2. **Missing Sibling Context**: When resolving a decay6B job, we need the *resolved* configuration of its stable sibling to:
   - Know the checkpoint iteration to load from
   - Reference the correct output directory
   - Set up start conditions that wait for sibling checkpoints

3. **Sibling Matching Complexity**: This is the critical issue:
   - There are **multiple jobs with the same stage name** (15 "stable" jobs, 15 "decay6B" jobs, etc.)
   - A "decay6B" job with `lr=0.001, batch_size=64` must reference the "stable" job with **the same hyperparameters**
   - Siblings must NOT cross between different branches of the cartesian product
   - Simple pattern matching `{sibling.stable.*}` is ambiguous - which of the 15 stable jobs?

4. **Current Separation is Wrong**: We have:
   - `build_job_plans()` - tries to resolve configs
   - `resolve_sibling_references()` - runs afterward, resolves `{sibling.*}` templates

   But these need to happen **together** in a DAG traversal where each job is resolved with its already-resolved siblings' context.

## Sibling Matching: The Core Challenge

### The Problem Illustrated

Consider our sweep structure:
```yaml
sweep:
  type: product
  groups:
    # GROUP 1: Hyperparameters (15 configs)
    - type: list
      configs:
        - {lr: 0.00025, batch_size: 64}  # Config 0
        - {lr: 0.0005, batch_size: 64}   # Config 1
        - {lr: 0.001, batch_size: 64}    # Config 2
        # ... 12 more configs ...

    # GROUP 2: Stages (5 stages)
    - type: list
      configs:
        - stage: stable              # Stage 0
        - stage: decay6B             # Stage 1
        - stage: decay12B            # Stage 2
        # ... more stages ...
```

This produces 75 jobs with `group_path` values like:
```
Job 0:  group_path=(0, 0)  params={lr: 0.00025, batch_size: 64, stage: stable}
Job 1:  group_path=(0, 1)  params={lr: 0.00025, batch_size: 64, stage: decay6B}
Job 2:  group_path=(0, 2)  params={lr: 0.00025, batch_size: 64, stage: decay12B}
...
Job 15: group_path=(1, 0)  params={lr: 0.0005, batch_size: 64, stage: stable}
Job 16: group_path=(1, 1)  params={lr: 0.0005, batch_size: 64, stage: decay6B}
...
```

**When Job 1 (decay6B with lr=0.00025) references `{sibling.stable.*}`, it MUST resolve to Job 0, NOT Job 15!**

### Sibling Matching Rules

**Rule 1: Same Hyperparameter Branch**
- Siblings must share the same "hyperparameter path" (first part of group_path)
- In our example: Jobs with `group_path[0] == 0` are in the same branch
- This ensures `lr=0.00025, batch_size=64` decay6B references `lr=0.00025, batch_size=64` stable

**Rule 2: Different Stage**
- Siblings must have different `stage` parameter values
- You can't reference yourself or another job with the same stage

**Rule 3: Exact Pattern Match**
- The `{sibling.PATTERN.*}` pattern should match the stage name
- Example: `{sibling.stable.output_dir}` matches jobs where `stage == "stable"`

**Rule 4: Unique Within Branch**
- Within a hyperparameter branch, each stage should appear exactly once
- If multiple matches exist (configuration error), fail with clear error message

### Implementation: `group_path`-based Matching

```python
def find_sibling(
    job: SweepPoint,
    all_points: list[SweepPoint],
    stage_pattern: str
) -> SweepPoint:
    """Find sibling job with matching hyperparameters but different stage."""

    # Extract hyperparameter portion of group_path
    # For group_path = (hp_idx, stage_idx), hp_portion is (hp_idx,)
    # This assumes outer product structure: hyperparameters × stages
    job_hp_path = job.group_path[:-1]  # All but last element

    candidates = []
    for point in all_points:
        # Must have same hyperparameter path
        if point.group_path[:-1] != job_hp_path:
            continue

        # Must match stage pattern
        if point.parameters.get('stage') != stage_pattern:
            continue

        # Must not be the same job
        if point.index == job.index:
            continue

        candidates.append(point)

    if len(candidates) == 0:
        raise ValueError(
            f"No sibling found for job {job.index} with stage={stage_pattern}"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple siblings found for job {job.index} with stage={stage_pattern}: "
            f"{[c.index for c in candidates]}"
        )

    return candidates[0]
```

**Caveat**: This assumes a specific sweep structure (hyperparameters × stages). For more complex nested groups, we need a more sophisticated algorithm.

### Alternative: Parameter-based Similarity

Instead of relying on `group_path`, we could match based on parameter values:

```python
def find_sibling_by_params(
    job: SweepPoint,
    all_points: list[SweepPoint],
    stage_pattern: str,
    hyperparameter_keys: list[str]  # for example, ['lr', 'batch_size']
) -> SweepPoint:
    """Find sibling with matching hyperparameters."""

    job_hp_values = {
        key: job.parameters.get(f'backend.megatron.{key}')
        for key in hyperparameter_keys
    }

    candidates = []
    for point in all_points:
        # Must match all hyperparameter values
        if not all(
            point.parameters.get(f'backend.megatron.{key}') == value
            for key, value in job_hp_values.items()
        ):
            continue

        # Must match stage pattern
        if point.parameters.get('stage') != stage_pattern:
            continue

        # Must not be the same job
        if point.index == job.index:
            continue

        candidates.append(point)

    if len(candidates) != 1:
        raise ValueError(f"Expected exactly 1 sibling, found {len(candidates)}")

    return candidates[0]
```

**Advantages**: More flexible, doesn't assume specific sweep structure
**Disadvantages**: Requires knowing which parameters are "hyperparameters" vs "stage-specific"

### Recommendation: Hybrid Approach

1. **Primary**: Use `group_path`-based matching (fast, leverages existing structure)
2. **Fallback**: Validate match by checking parameter values
3. **Error handling**: If group_path doesn't work, try parameter-based matching with warning

## Core Principles

1. **Use OmegaConf wherever possible** - It's the industry standard for configuration management
2. **Keep things simple** - Avoid over-engineering, but handle sibling matching correctly
3. **Support resolution in sweep parameters** - Interpolations should work in sweep defaults and configs
4. **No circular dependencies** - Jobs form a DAG, not a general graph (ensures termination)
5. **Respect sweep structure** - Siblings must not cross between different hyperparameter branches

## Current Architecture

```
Config Loading (Hydra)
  ↓
Sweep Expansion (expander.py)
  ↓ produces SweepPoints with parameters dict
Job Planning (planner.py)
  ↓ produces JobPlans with resolved paths, conditions
Sibling Resolution (sibling_resolver.py)
  ↓ resolves {sibling.*} templates
Final Jobs
```

### What Works
- Hydra config composition with defaults and overrides
- Sweep expansion into parameter combinations
- Group-level defaults for shared configuration
- Template-based sibling references `{sibling.stable.output_dir}`

### What Doesn't Work
- OmegaConf can't parse complex `${oc.eval:int((...)//...)}` expressions when creating a config from a plain dict with string values
- Resolution happens too early (during planning) before sibling context is available
- Sibling matching is pattern-based but doesn't verify parameter compatibility

## Option 1: DAG-Based Unified Resolution (RECOMMENDED)

### Approach
**Merge `build_job_plans()` and `resolve_sibling_references()` into a single DAG-based resolution pass.**

1. **Extract Dependencies**: Parse SweepPoints to find sibling references in parameters and start/cancel conditions
2. **Build DAG**: Construct dependency graph using `group_path`-based sibling matching
3. **Topological Sort**: Order jobs so dependencies are resolved before dependents
4. **Unified Resolution**: In topological order, for each job:
   - Find its resolved siblings using group_path matching
   - Build full context: base config + job parameters + sibling configs
   - Use OmegaConf to merge and resolve interpolations
   - Compute derived parameters (train_iters, etc.)
   - Store resolved job for use by dependent jobs

### Implementation Sketch
```python
def resolve_sweep_with_dag(
    config: RootConfig,
    points: list[SweepPoint]
) -> list[JobPlan]:
    """Unified resolution: job planning + sibling resolution in one DAG pass."""

    # Phase 1: Build dependency graph
    dag = build_dependency_dag_from_points(points)

    # Check for cycles
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("Circular dependencies in sweep configuration")

    # Phase 2: Topological sort
    ordered_indices = list(nx.topological_sort(dag))

    # Phase 3: Resolve in order
    resolved_jobs = {}

    for point_idx in ordered_indices:
        point = points[point_idx]

        # Find sibling points using group_path matching
        sibling_points = find_siblings_by_group_path(point, points)

        # Get already-resolved sibling jobs
        sibling_jobs = {
            sib.parameters.get('stage'): resolved_jobs[sib.index]
            for sib in sibling_points
            if sib.index in resolved_jobs
        }

        # Build full context
        context = build_resolution_context(
            base_config=config,
            point_params=point.parameters,
            resolved_siblings=sibling_jobs
        )

        # Resolve using OmegaConf
        resolved_params = resolve_with_omegaconf(context)

        # Compute derived parameters (train_iters, start_iter, etc.)
        resolved_params = compute_derived_parameters(resolved_params)

        # Create resolved job plan
        job = JobPlan(
            name=format_name(config.sweep.name_template, resolved_params),
            parameters=resolved_params,
            output_dir=...,
            start_conditions=resolve_conditions(
                point.parameters.get('job.start_conditions', []),
                resolved_params,
                sibling_jobs
            ),
            # ... other fields
        )

        resolved_jobs[point_idx] = job

    return list(resolved_jobs.values())


def build_dependency_dag_from_points(points: list[SweepPoint]) -> nx.DiGraph:
    """Build DAG from sweep points by analyzing sibling references."""
    dag = nx.DiGraph()

    for point in points:
        dag.add_node(point.index)

        # Extract sibling stage patterns from parameters
        sibling_deps = extract_sibling_patterns(point.parameters)

        for stage_pattern in sibling_deps:
            # Find sibling using group_path matching
            try:
                sibling = find_sibling_by_group_path(
                    point, points, stage_pattern
                )
                # Add edge: sibling depends on this job
                dag.add_edge(sibling.index, point.index)
            except ValueError:
                # Stage not found (for example, stable stage has no dependencies)
                continue

    return dag


def find_sibling_by_group_path(
    point: SweepPoint,
    all_points: list[SweepPoint],
    stage_pattern: str
) -> SweepPoint:
    """Find sibling with matching hyperparameters using group_path."""

    # Hyperparameter path = all but last element of group_path
    # (assumes structure: hyperparams × stages)
    hp_path = point.group_path[:-1]

    candidates = []
    for p in all_points:
        # Must share hyperparameter branch
        if p.group_path[:-1] != hp_path:
            continue

        # Must match stage pattern
        if p.parameters.get('stage') != stage_pattern:
            continue

        # Must not be same job
        if p.index == point.index:
            continue

        candidates.append(p)

    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly 1 sibling with stage={stage_pattern} "
            f"for point {point.index}, found {len(candidates)}"
        )

    return candidates[0]


def build_resolution_context(
    base_config: RootConfig,
    point_params: dict,
    resolved_siblings: dict[str, JobPlan]
) -> OmegaConf:
    """Build OmegaConf context with base config + point params + siblings."""

    # Start with base config
    context = OmegaConf.create(asdict(base_config))

    # Merge point parameters
    point_cfg = OmegaConf.create(unflatten_dict(point_params))
    context = OmegaConf.merge(context, point_cfg)

    # Add sibling references
    sibling_cfg = {}
    for stage, job in resolved_siblings.items():
        sibling_cfg[f'sibling_{stage}'] = {
            'name': job.name,
            'output_dir': job.output_dir,
            'log_path': job.log_path,
            'log_path_current': job.log_path_current,
            # Include resolved parameters that siblings might reference
            'params': job.parameters
        }

    context = OmegaConf.merge(context, OmegaConf.create(sibling_cfg))

    return context
```

### Advantages
- ✅ **Single resolution pass**: No separation between job planning and sibling resolution
- ✅ **Correct sibling matching**: Uses `group_path` to avoid cross-contamination
- ✅ **Full context available**: Each job resolves with already-resolved siblings
- ✅ **OmegaConf native**: Uses OmegaConf.merge() and standard resolution
- ✅ **Cycle detection**: Catches configuration errors early
- ✅ **Clear dependencies**: DAG makes relationships explicit

### Disadvantages
- ⚠️ **Requires networkx**: External dependency for graph operations
- ⚠️ **All-or-nothing**: Must resolve entire sweep at once
- ⚠️ **Assumes structure**: group_path matching assumes hyperparams × stages structure

### Caveats
- **group_path assumptions**: Current implementation assumes last element of group_path is stage index. Need to handle more complex nested structures.
- **Error messages**: Must provide clear errors when sibling not found or multiple matches
- **Performance**: For very large sweeps (1000+ jobs), topological sort might be slow (but likely negligible)
- **Sibling data**: Need to decide what sibling information to expose (name, paths, all parameters?)

## Option 2: Two-Phase Resolution (Current + Enhanced)

### Approach
1. **Phase 1 - Template Preservation**: Keep interpolations as escaped strings during sweep expansion and initial job planning
2. **Phase 2 - Lazy Resolution**: Resolve interpolations at "submission time" when we have full context including sibling information

### Implementation Sketch
```python
def build_job_plans(config, points):
    """Phase 1: Build jobs with templates preserved."""
    jobs = []
    for point in points:
        # Don't resolve interpolations - keep them as escaped strings
        job = JobPlan(
            name=...,
            parameters=point.parameters,  # Still contains ${...} strings
            start_conditions=point.parameters.get('job.start_conditions', []),
            # Store raw templates
        )
        jobs.append(job)
    return jobs

def resolve_job_for_submission(job, all_jobs):
    """Phase 2: Resolve when actually needed."""
    # Find sibling jobs with matching hyperparameters
    siblings = find_matching_siblings(job, all_jobs)

    # Build context with sibling data
    context = build_context(job, siblings)

    # NOW use OmegaConf to resolve
    resolved_params = resolve_with_omegaconf(job.parameters, context)

    return JobPlan(
        name=job.name,
        parameters=resolved_params,
        ...
    )
```

### Advantages
- Simpler - no graph construction needed
- Lazy evaluation - only resolve what's needed when it's needed
- OmegaConf-native - use OmegaConf.merge() to combine job config + sibling context
- Flexible - can re-resolve with different contexts if needed

### Disadvantages
- Late error detection - won't know if interpolations are invalid until submission
- Two representations - jobs exist in both "template" and "resolved" forms
- Repetitive resolution - might resolve same job multiple times
- Context passing - need to thread sibling information through submission pipeline

### Caveats
- **When to resolve**: Need clear policy on when Phase 2 happens (at submission? at monitoring?)
- **Caching**: Should cache resolved jobs to avoid re-resolution
- **Validation**: Hard to validate configuration without actually resolving it

## Option 3: Hybrid - Staged Resolution with OmegaConf Integration

### Approach
Combine best of both: use OmegaConf during sweep expansion, but delay complex cross-job resolution until we have sibling context.

1. **During Sweep Expansion**: Use OmegaConf to resolve intra-job interpolations (within a single job's config)
2. **During Job Planning**: Resolve simple interpolations that don't depend on siblings
3. **After Sibling Matching**: Resolve cross-job interpolations with full sibling context

### Implementation Sketch
```python
def expand_sweep(sweep_config):
    """Use OmegaConf for intra-job resolution."""
    points = []
    for params in expand_groups(sweep_config):
        # Use OmegaConf to resolve within this parameter set
        cfg = OmegaConf.create(params)
        # Only resolve interpolations that reference params within this config
        resolved = OmegaConf.to_container(cfg, resolve=True)
        points.append(SweepPoint(parameters=resolved))
    return points

def build_job_plans(config, points):
    """Resolve non-sibling interpolations."""
    jobs = []
    for point in points:
        context = merge_dicts(config, point.parameters)

        # Resolve simple interpolations (no sibling references)
        simple_resolved = resolve_simple_interpolations(context)

        # Keep sibling-dependent values as templates
        job = JobPlan(
            name=format_name(config.sweep.name_template, simple_resolved),
            parameters=simple_resolved,
            start_conditions=point.parameters.get('job.start_conditions', []),
        )
        jobs.append(job)
    return jobs

def resolve_cross_job_dependencies(jobs):
    """Final resolution with sibling context."""
    for job in jobs:
        siblings = find_matching_siblings(job, jobs)
        sibling_context = {
            f'sibling.{sib.stage_name}': sib.parameters
            for sib in siblings
        }

        # Merge sibling context into job parameters
        full_context = OmegaConf.merge(
            OmegaConf.create(job.parameters),
            OmegaConf.create(sibling_context)
        )

        # Resolve remaining interpolations
        resolved = OmegaConf.to_container(full_context, resolve=True)
        job.parameters = resolved

    return jobs
```

### Advantages
- OmegaConf throughout - uses standard tool at each stage
- Progressive resolution - resolve what you can, when you can
- Clear separation - intra-job vs cross-job interpolations
- Early validation - catch errors in simple interpolations early

### Disadvantages
- Three stages - more complex mental model
- Classification needed - must identify which interpolations are "simple" vs "cross-job"
- Potential conflicts - what if simple resolution conflicts with later sibling context?

### Caveats
- **Interpolation classification**: Need heuristic to identify sibling-dependent interpolations (maybe anything with `{sibling.` or references to other jobs)
- **Partial resolution risks**: If we resolve too early, we lose ability to re-resolve with sibling context
- **OmegaConf merge semantics**: Need to understand how OmegaConf.merge() handles conflicts

## Option 4: Keep Configs in OmegaConf Format Throughout

### Approach
Never convert to plain dicts - keep everything as OmegaConf DictConfig objects that preserve interpolations.

1. **Sweep Expansion**: Produce SweepPoints with OmegaConf configs (not plain dicts)
2. **Job Planning**: Manipulate OmegaConf configs, merging but not resolving
3. **Sibling Resolution**: Use OmegaConf.merge() to combine job + sibling configs
4. **Final Resolution**: Call `to_container(resolve=True)` only at the end

### Implementation Sketch
```python
@dataclass(frozen=True)
class SweepPoint:
    index: int
    parameters: DictConfig  # OmegaConf object, not plain dict
    group_path: tuple[int, ...]

def expand_sweep(sweep_config):
    """Keep configs as OmegaConf objects."""
    points = []
    for params in expand_groups(sweep_config):
        # Create OmegaConf config but DON'T resolve
        cfg = OmegaConf.create(params)
        # Keep interpolations intact
        points.append(SweepPoint(
            index=len(points),
            parameters=cfg  # DictConfig with unresolved interpolations
        ))
    return points

def build_job_plans(config, points):
    """Merge configs without resolving."""
    jobs = []
    for point in points:
        # Merge base config + point parameters (both OmegaConf)
        merged = OmegaConf.merge(config, point.parameters)

        # Extract what we need but keep interpolations
        job = JobPlan(
            name=OmegaConf.to_container(
                merged.sweep.name_template, resolve=True
            ),
            parameters=merged,  # Still OmegaConf with interpolations!
            ...
        )
        jobs.append(job)
    return jobs

def resolve_with_siblings(job, siblings):
    """Merge sibling configs and resolve."""
    sibling_configs = {
        f'sibling_{sib.stage}': sib.parameters
        for sib in siblings
    }

    # Merge everything together
    full_cfg = OmegaConf.merge(
        job.parameters,
        OmegaConf.create(sibling_configs)
    )

    # Now resolve
    return OmegaConf.to_container(full_cfg, resolve=True)
```

### Advantages
- Native OmegaConf - uses tool as designed
- Interpolations preserved - no escaping/unescaping needed
- Composable - can merge configs at any stage
- Type safety - OmegaConf provides schema validation

### Disadvantages
- Dataclass compatibility - OmegaConf configs don't work well with dataclasses
- Serialization - harder to serialize/save intermediate states
- Type annotations - loses Python type hints
- Learning curve - team needs to understand OmegaConf deeply

### Caveats
- **Dataclass conversion**: Need to convert to/from dataclasses carefully
- **Escaping still needed**: The original problem (escaping `\\$` in YAML) still exists
- **Debugging**: Harder to inspect configs in debugger (DictConfig repr is verbose)

## Current Problem: Why OmegaConf.create() Fails

The specific error we're seeing:
```
ERROR: OmegaConf resolution failed: token recognition error at: '('
    full_key: backend.megatron.train_iters
```

This happens because:
1. We unescape `\\${oc.eval:int\\(...\\)}` → `${oc.eval:int(...)}`
2. We unflatten to nested dict: `{'backend': {'megatron': {'train_iters': '${oc.eval:int(...)}'}}}`
3. We call `OmegaConf.create(nested)`
4. OmegaConf tries to **parse** the interpolation string during creation
5. The parser doesn't like the `(` after `int` - it expects `${oc.eval:expression}` where expression is simpler

**Root cause**: OmegaConf validates interpolation syntax during `create()`, before `resolve()`.

**Workaround options**:
a) Don't unescape until just before resolution
b) Use different syntax that OmegaConf accepts
c) Pre-compute complex expressions and store as simple interpolations
d) Use OmegaConf's structured configs instead of dicts

## Recommended Approach: Option 3 (Hybrid) + Pre-computation

### Strategy
1. **Compute train_iters early**: Since `train_iters = f(tokens, batch_size, seq_length)` and all those values are known after parameter merging, compute it directly instead of using OmegaConf interpolation.

2. **Use OmegaConf for simple interpolations**: Things like `${backend.megatron.aux.start_iter_round}` that reference other config keys.

3. **Stage resolution**:
   - **Stage 1 (Sweep Expansion)**: Merge defaults + configs, produce parameter dicts
   - **Stage 2 (Job Planning)**: Compute derived values (train_iters), build job plans with templates
   - **Stage 3 (Sibling Resolution)**: Match siblings, resolve cross-job references

### Why This Works
- **Avoids OmegaConf parse errors**: We don't ask OmegaConf to parse complex `oc.eval` expressions from strings
- **Uses OmegaConf correctly**: For what it's designed for (merging configs, simple interpolations)
- **Keeps things simple**: Complex math is just Python code, not string templates
- **Preserves flexibility**: Can still use interpolations for cross-job references

### Implementation
```python
def compute_derived_parameters(params: dict) -> dict:
    """Compute values that involve complex expressions."""
    tokens = params.get('backend.megatron.aux.tokens')
    seq_length = params.get('backend.megatron.seq_length')
    batch_size = params.get('backend.megatron.global_batch_size')

    if all(v is not None for v in [tokens, seq_length, batch_size]):
        # Direct computation instead of OmegaConf interpolation
        train_iters = (tokens + seq_length * batch_size - 1) // (seq_length * batch_size)
        params['backend.megatron.train_iters'] = train_iters

        # Compute other derived values
        decay_fraction = params.get('backend.megatron.aux.decay_fraction', 0.0)
        start_iter = int(train_iters * (1 - decay_fraction))
        params['backend.megatron.aux.start_iter'] = start_iter

        save_interval = params.get('backend.megatron.save_interval', 2000)
        start_iter_round = (start_iter // save_interval) * save_interval
        params['backend.megatron.aux.start_iter_round'] = start_iter_round

    return params
```

### Trade-offs
- **Pro**: Simple, works with current architecture
- **Pro**: No complex OmegaConf parsing issues
- **Pro**: Values are computed when dependencies are available
- **Con**: Mixing paradigms (some values from OmegaConf, some from Python)
- **Con**: Need to manually identify which values need pre-computation
- **Con**: Loses declarative nature of config (computation is imperative)

## Decision Matrix

| Criterion | Option 1 (DAG) | Option 2 (Two-Phase) | Option 3 (Hybrid) | Option 4 (Pure OmegaConf) | Recommended |
|-----------|----------------|----------------------|-------------------|---------------------------|-------------|
| Uses OmegaConf | Partial | Partial | Yes | Yes | ✓ Hybrid |
| Simplicity | Medium | High | Medium | Low | ✓ Two-Phase |
| Early error detection | High | Low | Medium | High | Option 1 |
| Sibling resolution | Explicit | Implicit | Hybrid | Native | Option 4 |
| Performance | Medium | High | High | Medium | ✓ Two-Phase/Hybrid |
| Handles current bug | No | No | Yes* | No* | ✓ Hybrid |

*With pre-computation

## Option 5: Pure OmegaConf Resolution (SIMPLEST)

### Core Insight
**Stop mixing template systems.** Use OmegaConf `${...}` syntax for EVERYTHING, including sibling references. No custom template resolution needed.

### Approach
1. **Change template syntax**: `{sibling.stable.output_dir}` → `${sibling.stable.output_dir}`
2. **Build OmegaConf config with sibling data** in the namespace during resolution
3. **Let OmegaConf resolve everything** - no custom code
4. **Use simple ordering** instead of full DAG (since we have stable → all decay stages, not complex chains)

### Implementation
```python
def resolve_sweep_simple(
    config: RootConfig,
    points: list[SweepPoint]
) -> list[JobPlan]:
    """Simple two-pass resolution using pure OmegaConf."""

    # Pass 1: Resolve jobs with no dependencies (for example, stable stage)
    resolved_jobs = {}
    for point in points:
        sibling_deps = extract_sibling_patterns(point.parameters)
        if not sibling_deps:  # No dependencies
            job = resolve_point_with_omegaconf(config, point, sibling_data={})
            resolved_jobs[point.index] = job

    # Pass 2: Resolve jobs with dependencies
    for point in points:
        if point.index in resolved_jobs:
            continue  # Already resolved

        # Find siblings using group_path matching
        sibling_jobs = find_siblings_for_point(point, points, resolved_jobs)

        # Resolve with sibling data
        job = resolve_point_with_omegaconf(config, point, sibling_data=sibling_jobs)
        resolved_jobs[point.index] = job

    return list(resolved_jobs.values())


def resolve_point_with_omegaconf(
    config: RootConfig,
    point: SweepPoint,
    sibling_data: dict[str, JobPlan]
) -> JobPlan:
    """Resolve a single point using pure OmegaConf."""

    from omegaconf import OmegaConf

    # Build nested config structure
    cfg_dict = {
        # Base config
        'backend': asdict(config.backend),
        'slurm': asdict(config.slurm),
        'project': asdict(config.project),
        'monitoring': asdict(config.monitoring),

        # Sibling data (so ${sibling.stable.output_dir} resolves)
        'sibling': {
            stage: {
                'name': job.name,
                'output_dir': job.output_dir,
                'log_path': job.log_path,
                'log_path_current': job.log_path_current or "",
            }
            for stage, job in sibling_data.items()
        }
    }

    # Merge point parameters (unflatten first)
    point_nested = unflatten_dict(point.parameters)
    cfg_dict = deep_merge(cfg_dict, point_nested)

    # Create OmegaConf config
    cfg = OmegaConf.create(cfg_dict)

    # Compute derived parameters (train_iters, start_iter, etc.)
    # Do this BEFORE resolution so OmegaConf can use them
    cfg = compute_derived_in_omegaconf(cfg)

    # Resolve ALL interpolations
    resolved = OmegaConf.to_container(cfg, resolve=True)

    # Extract job plan fields
    return build_job_plan_from_resolved(config, point, resolved)
```

### What Changes in YAML

**Before** (mixed syntax - WRONG):
```yaml
backend.megatron.load: "{sibling.stable.output_dir}/checkpoints/iter_{backend.megatron.aux.start_iter_round}"

job.start_conditions:
  - class_name: FileExistsCondition
    path: "{sibling.stable.output_dir}/checkpoints/iter_{backend.megatron.aux.start_iter_round}/latest_checkpointed_iteration.txt"
```

**After** (pure OmegaConf - CORRECT):
```yaml
backend.megatron.load: "${sibling.stable.output_dir}/checkpoints/iter_${backend.megatron.aux.start_iter_round}"

job.start_conditions:
  - class_name: FileExistsCondition
    path: "${sibling.stable.output_dir}/checkpoints/iter_${backend.megatron.aux.start_iter_round}/latest_checkpointed_iteration.txt"
```

Now everything is OmegaConf syntax. No custom template resolution needed!

### Advantages
- ✅ **Pure OmegaConf** - Uses industry-standard tool as intended
- ✅ **Simplest** - No DAG, no topological sort, just two passes
- ✅ **Minimal custom code** - Only sibling matching and derived parameter computation
- ✅ **Easy to debug** - OmegaConf errors are clear and well-documented
- ✅ **Extensible** - Adding more sibling data is trivial (just add to dict)

### Disadvantages
- ⚠️ **Assumes two-level dependencies** - Doesn't handle stable → decay1 → decay2 chains
- ⚠️ **Derived parameters** - Still need Python for train_iters computation (OmegaConf ${oc.eval} has limitations)

### When This Works
- ✅ Current use case: stable + multiple decay stages (all depend only on stable)
- ✅ Most multi-stage training scenarios
- ❌ Complex dependency chains (need full DAG)

### How to Extend to Full DAG
If we need complex chains later, just add topological sort:
```python
def resolve_sweep_with_dag_simple(config, points):
    dag = build_dependency_dag(points)  # Extract sibling dependencies
    ordered = list(nx.topological_sort(dag))  # Get resolution order

    resolved_jobs = {}
    for point_idx in ordered:
        point = points[point_idx]
        sibling_jobs = get_resolved_siblings(point, points, resolved_jobs)
        job = resolve_point_with_omegaconf(config, point, sibling_jobs)
        resolved_jobs[point_idx] = job

    return list(resolved_jobs.values())
```

Still pure OmegaConf for resolution, just with DAG-based ordering.

## Final Recommendation

### For Current Use Case: **Option 5 (Pure OmegaConf) + DAG Ordering**

**The simplest, most maintainable solution:**

1. **Use OmegaConf for ALL interpolations** - Change `{...}` → `${...}` in YAML
2. **Add sibling data to OmegaConf namespace** - So `${sibling.stable.output_dir}` resolves naturally
3. **Use DAG for ordering** - Ensures siblings resolved before dependents
4. **Compute derived params in Python** - train_iters, start_iter (OmegaConf ${oc.eval} limitations)
5. **No custom template resolution** - Pure OmegaConf does everything

### Why This is Best

**versus Option 1 (Current Implementation)**:
- ❌ Option 1 mixes `{...}` and `${...}` template systems - confusing
- ❌ Option 1 has custom `resolve_template_strings()` function - not needed
- ✅ Option 5 is pure OmegaConf - industry standard, well-documented

**versus Two-Pass Without DAG**:
- Two-pass works for current case (stable → decay) but breaks if we add decay chains
- DAG is more robust and handles any dependency structure
- NetworkX is already a dependency - minimal cost

**Simplicity Comparison**:
```python
# Current (Option 1): Custom template resolution
start_conditions = resolve_template_strings(start_conditions, template_context)
# Also need: resolve_with_omegaconf(), compute_derived_parameters(), etc.

# Recommended (Option 5): Pure OmegaConf
cfg = OmegaConf.create(config_with_siblings)  # Sibling data in namespace
resolved = OmegaConf.to_container(cfg, resolve=True)  # Done!
# Only custom code: compute derived params (unavoidable)
```

### Implementation Summary

**YAML Changes** (One-time):
```yaml
# Before: Mixed syntax
backend.megatron.load: "{sibling.stable.output_dir}/checkpoints/iter_{backend.megatron.aux.start_iter_round}"

# After: Pure OmegaConf
backend.megatron.load: "${sibling.stable.output_dir}/checkpoints/iter_${backend.megatron.aux.start_iter_round}"
```

**Code Structure**:
```python
def resolve_sweep_with_dag(config, points):
    # 1. Build DAG (unchanged)
    dag = build_dependency_dag_from_points(points)
    ordered = list(nx.topological_sort(dag))

    # 2. Resolve in order
    resolved_jobs = {}
    for point_idx in ordered:
        point = points[point_idx]

        # Find resolved siblings
        siblings = get_resolved_siblings(point, points, resolved_jobs)

        # Create OmegaConf config with sibling data in namespace
        cfg_dict = {
            'backend': ...,
            'sibling': {  # <-- Key: sibling data in OmegaConf namespace
                'stable': {'output_dir': siblings['stable'].output_dir, ...},
                # etc.
            }
        }
        cfg = OmegaConf.create(cfg_dict)

        # Compute derived parameters (modify cfg in-place)
        compute_derived_parameters_in_cfg(cfg)

        # Resolve ALL interpolations (including ${sibling.*})
        resolved = OmegaConf.to_container(cfg, resolve=True)

        # Create JobPlan
        job = create_job_plan(resolved, point)
        resolved_jobs[point_idx] = job

    return list(resolved_jobs.values())
```

**What's Custom** (Unavoidable):
1. `build_dependency_dag_from_points()` - Extract sibling references
2. `find_sibling_by_group_path()` - Match siblings within hyperparameter branch
3. `compute_derived_parameters_in_cfg()` - Compute train_iters, start_iter (OmegaConf ${oc.eval} can't handle complex expressions)

**What's OmegaConf** (Zero custom code):
1. All interpolation resolution - `${backend.megatron.*}`, `${sibling.stable.*}`, etc.
2. Config merging - base config + point params
3. Nested structure handling - automatic

### Benefits

1. **Maintainability**: OmegaConf is industry standard - anyone familiar with Hydra understands this
2. **Debuggability**: OmegaConf errors are clear (for example, "InterpolationKeyError: ${sibling.foo.bar}")
3. **Extensibility**: Adding new sibling fields is trivial (just add to dict)
4. **Correctness**: DAG ensures proper resolution order
5. **Simplicity**: ~100 lines of custom code versus 300+ in current implementation

### Trade-offs Accepted

1. **One-time YAML update**: Change `{...}` → `${...}` (worth it for simplicity)
2. **NetworkX dependency**: Already added, negligible cost
3. **Python for derived params**: Unavoidable (OmegaConf ${oc.eval} limitations)

### Migration Path

1. Update YAML configs: `{sibling.*}` → `${sibling.*}`, `{backend.*}` → `${backend.*}`
2. Rewrite `dag_resolver.py` to use pure OmegaConf approach
3. Remove custom template resolution functions
4. Test with existing config

Based on the analysis above, **Option 5 with DAG ordering is the correct solution** because:

1. ✅ **Pure OmegaConf** - No mixing of template systems
2. ✅ **Correct sibling matching by way of group_path** - Prevents cross-contamination between hyperparameter branches
3. ✅ **Full context during resolution** - Siblings are already resolved when needed
4. ✅ **Uses OmegaConf properly** - Merge configs, then resolve (not parse interpolations from strings)
5. ✅ **Handles derived parameters** - Compute train_iters, start_iter after merging stage-specific tokens

### Implementation Plan

**Phase 1: Core DAG Resolution** (Immediate)
1. Add networkx dependency to requirements
2. Implement `build_dependency_dag_from_points()` - extract sibling dependencies from parameters
3. Implement `find_sibling_by_group_path()` - match siblings within same hyperparameter branch
4. Implement `resolve_sweep_with_dag()` - unified resolution in topological order
5. Replace current `build_job_plans() + resolve_sibling_references()` with single `resolve_sweep_with_dag()`

**Phase 2: OmegaConf Integration** (Immediate)
1. Implement `build_resolution_context()` - merge base config + point params + sibling data using OmegaConf
2. Use `OmegaConf.merge()` instead of manual dict merging
3. Call `OmegaConf.to_container(resolve=True)` after full context is built
4. This avoids the "parse complex interpolations from strings" problem

**Phase 3: Derived Parameters** (Immediate)
1. Implement `compute_derived_parameters()` - calculate train_iters, start_iter, start_iter_round
2. Call this AFTER OmegaConf resolution, when tokens/batch_size/etc. are known
3. Update YAML configs to remove complex `${oc.eval:int((...)//...)}` expressions
4. Keep simple interpolations like `${backend.megatron.save_interval}`

**Phase 4: Validation & Error Handling** (Follow-up)
1. Add cycle detection with clear error messages
2. Validate sibling matches (exactly one sibling per stage pattern)
3. Add debug logging showing resolution order
4. Handle edge cases (no dependencies, complex nested groups)

### What Changes in YAML Configs

**Before** (complex interpolations that OmegaConf can't parse):
```yaml
defaults:
  backend.megatron.train_iters: "\\${oc.eval:int\\(\\(${backend.megatron.aux.tokens}+${backend.megatron.seq_length}*${backend.megatron.global_batch_size}-1\\)//\\(${backend.megatron.seq_length}*${backend.megatron.global_batch_size}\\)\\)}"
```

**After** (let Python compute it):
```yaml
defaults:
  backend.megatron.aux.tokens: 6_000_000_000
  # train_iters will be computed by compute_derived_parameters()
  # based on tokens, seq_length, global_batch_size
```

**Simple interpolations still work**:
```yaml
job.start_conditions:
  - class_name: FileExistsCondition
    path: "{sibling.stable.output_dir}/checkpoints/iter_${backend.megatron.aux.start_iter_round}/latest_checkpointed_iteration.txt"
```

After sibling resolution, `{sibling.stable.output_dir}` is replaced with actual path, then OmegaConf resolves `${backend.megatron.aux.start_iter_round}` from the merged context.

### Why This is Better Than Other Options

**versus Option 2 (Two-Phase)**:
- No late error detection - DAG finds issues during expansion
- No template vs resolved dual representation - everything resolved once
- Sibling matching is explicit and validated

**versus Option 3 (Hybrid)**:
- Clearer architecture - one resolution pass, not three stages
- No need to classify interpolations as "simple" vs "cross-job"
- Full context always available

**versus Option 4 (Pure OmegaConf)**:
- Still uses OmegaConf extensively (for merging and simple interpolations)
- But computes complex expressions in Python (where they belong)
- Avoids OmegaConf parsing errors

### Trade-offs Accepted

1. **Networkx dependency** - Acceptable, it's a standard library for graph operations
2. **All-or-nothing resolution** - Acceptable, we need full sweep resolved anyway for submission
3. **group_path assumptions** - Acceptable for current sweep structure (hyperparams × stages), can generalize later if needed

This approach:
- ✅ Uses OmegaConf as intended (merging configs, resolving simple interpolations)
- ✅ Keeps implementation relatively simple (one main resolution function)
- ✅ Fixes the immediate bug (no complex interpolation parsing)
- ✅ Supports sweep parameter resolution (full context available)
- ✅ Respects sweep structure (group_path-based sibling matching)
- ✅ Provides clear foundation for future enhancements
