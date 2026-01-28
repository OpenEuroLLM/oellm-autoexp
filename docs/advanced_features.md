# Advanced Features and Configuration Guide

This document addresses advanced requirements for the OELLM auto-experimentation tool, including automatic cooldown, checkpoint conversion, evaluation triggers, platform-specific configurations, and consistent job naming for automatic resume.

## Table of Contents

1. [Automatic Cooldown Support](#automatic-cooldown-support)
2. [Checkpoint Conversion to HuggingFace](#checkpoint-conversion-to-huggingface)
3. [Automatic Evaluation Triggers](#automatic-evaluation-triggers)
4. [Multiple Cooldown Strategies](#multiple-cooldown-strategies)
5. [Platform Detection and Platform-Specific Configs](#platform-detection-and-platform-specific-configs)
6. [Consistent Job Naming for Automatic Resume](#consistent-job-naming-for-automatic-resume)

---

## Automatic Cooldown Support

### Overview

Cooldown refers to scheduling follow-up tasks when training reaches a specific iteration threshold. The current system supports cooldown through:

1. **Iteration-based log events** – Pattern matching on training iteration logs
2. **Queued actions** – Scheduling downstream commands or new autoexp runs
3. **Conditional execution** – Gating actions based on iteration thresholds

### Configuration Example: Single Cooldown at Specific Iteration

The recommended approach uses `RunAutoexpAction` to inherit all original configuration parameters and apply only necessary overrides. This avoids re-specifying all training arguments.

**See:** `config/monitoring/megatron_cooldown_single.yaml`

Key features:
- Triggers at iteration 50000 when checkpoint is saved
- Loads from original checkpoint using absolute path (`backend.args.load={checkpoint_path}`)
- Saves to new directory with timestamp: `{original}_cooldown_{iteration}_{timestamp}`
- Reduces learning rate for fine-tuning
- Uses `unresolved_config.yaml` for flexible path management

### Usage

```bash
# Start training with single cooldown monitoring
python scripts/run_autoexp.py \
  job=my_training \
  monitoring=megatron_cooldown_single \
  backend.args.train_iters=100000 \
  backend.args.save_interval=5000

# When training reaches iteration 50000 and saves a checkpoint,
# the cooldown run will automatically queue and execute
```

---

## Checkpoint Conversion to HuggingFace

### Current Support

The system already supports automatic checkpoint conversion through queued actions triggered by checkpoint detection events.

### Configuration Example: HuggingFace Conversion on Checkpoint Save

```yaml
# config/monitoring/megatron_checkpoint_hf_conversion.yaml
defaults:
  - megatron_production
  - _self_

log_events:
  - name: checkpoint_saved
    pattern: 'successfully saved checkpoint from iteration\s+(?P<iteration>\d+) to (?P<path>\S+)'
    pattern_type: regex
    metadata:
      kind: checkpoint
      trigger: hf_conversion
    extract_groups:
      checkpoint_iteration: iteration
      checkpoint_path: path
    actions:
      # Log the checkpoint save
      - class_name: LogAction
        message: "checkpoint saved at iteration {checkpoint_iteration}"

      # Queue HuggingFace conversion
      - class_name: EventAction
        mode: queue
        conditions:
          # Wait for checkpoint file to be fully written
          - class_name: FileExistsCondition
            path: "{checkpoint_path}/latest_checkpointed_iteration.txt"
            blocking: true
            timeout_seconds: 600
        action:
          class_name: RunCommandAction
          command:
            - python
            - scripts/convert_checkpoint_to_hf.py
            - "--megatron-checkpoint"
            - "{checkpoint_path}"
            - "--output-dir"
            - "{output_dir}/hf_checkpoints/iter_{checkpoint_iteration}"
            - "--model-type"
            - "llama"
            - "--iteration"
            - "{checkpoint_iteration}"

      # Optionally, queue evaluation on the HF checkpoint
      - class_name: EventAction
        mode: queue
        conditions:
          - class_name: FileExistsCondition
            path: "{output_dir}/hf_checkpoints/iter_{checkpoint_iteration}/config.json"
            blocking: true
            timeout_seconds: 1800  # Wait up to 30 minutes for conversion
        action:
          class_name: RunCommandAction
          command:
            - python
            - scripts/evaluate_hf_checkpoint.py
            - "--checkpoint-dir"
            - "{output_dir}/hf_checkpoints/iter_{checkpoint_iteration}"
            - "--output-file"
            - "{output_dir}/evaluations/iter_{checkpoint_iteration}_results.json"
```

### Conversion Script Template

Create `scripts/convert_checkpoint_to_hf.py`:

```python
#!/usr/bin/env python3
"""Convert Megatron checkpoint to HuggingFace format."""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--megatron-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-type", type=str, default="llama")
    parser.add_argument("--iteration", type=str, required=True)
    args = parser.parse_args()

    # Add your conversion logic here
    # This is backend-specific and may require:
    # - Megatron's checkpoint utilities
    # - HuggingFace transformers
    # - Model architecture mapping

    print(f"Converting checkpoint from {args.megatron_checkpoint}")
    print(f"Output directory: {args.output_dir}")
    # ... conversion implementation ...

if __name__ == "__main__":
    main()
```

---

## Automatic Evaluation Triggers

### Current Support

The system supports automatic evaluation through `RunAutoexpAction` and `RunCommandAction`. See `config/monitoring/megatron_checkpoint_eval.yaml` for a working example.

### Enhanced Configuration: Multi-Stage Evaluation

```yaml
# config/monitoring/megatron_multi_eval.yaml
defaults:
  - megatron_production
  - _self_

log_events:
  - name: checkpoint_saved
    pattern: 'successfully saved checkpoint from iteration\s+(?P<iteration>\d+) to (?P<path>\S+)'
    pattern_type: regex
    metadata:
      kind: checkpoint
      trigger: multi_evaluation
    extract_groups:
      checkpoint_iteration: iteration
      checkpoint_path: path
    actions:
      # Stage 1: Quick validation on small dataset
      - class_name: EventAction
        mode: queue
        conditions:
          - class_name: FileExistsCondition
            path: "{checkpoint_path}/latest_checkpointed_iteration.txt"
            blocking: true
        action:
          class_name: RunCommandAction
          command:
            - python
            - scripts/quick_eval.py
            - "--checkpoint"
            - "{checkpoint_path}"
            - "--dataset"
            - "validation_small"
            - "--output"
            - "{output_dir}/evals/quick_iter_{checkpoint_iteration}.json"

      # Stage 2: Full evaluation (conditional on iteration)
      - class_name: EventAction
        mode: queue
        conditions:
          # Only run full eval every 10k iterations
          - class_name: CommandCondition
            command: "python -c \"import sys; sys.exit(0 if int('{checkpoint_iteration}') % 10000 == 0 else 1)\""
          - class_name: FileExistsCondition
            path: "{checkpoint_path}/latest_checkpointed_iteration.txt"
            blocking: true
        action:
          class_name: RunAutoexpAction
          script: scripts/run_autoexp.py
          config_path: "{output_dir}/provenance/config_reference.json"
          overrides:
            - backend=evaluation
            - "+checkpoint_path={checkpoint_path}"
            - "+checkpoint_iteration={checkpoint_iteration}"
            - "backend.args.load={checkpoint_path}"
            - monitoring=megatron_basic

      # Stage 3: Benchmark suite (only on final checkpoint)
      - class_name: EventAction
        mode: queue
        conditions:
          - class_name: MetadataCondition
            key: checkpoint_iteration
            equals: "{backend.args.train_iters}"  # Final iteration
        action:
          class_name: RunCommandAction
          command:
            - python
            - scripts/run_benchmark_suite.py
            - "--checkpoint"
            - "{checkpoint_path}"
            - "--benchmarks"
            - "mmlu,hellaswag,arc,truthfulqa"
            - "--output"
            - "{output_dir}/benchmarks/final_results.json"
```

---

## Multiple Cooldown Strategies

### Overview

Multiple cooldowns allow scheduling different tasks at different iteration thresholds. This is useful for:
- Progressive learning rate scheduling
- Multi-stage evaluation
- Checkpoint archival at specific milestones
- Resource scaling adjustments

### Configuration Example: Multiple Cooldown Points

Use `RunAutoexpAction` to trigger different training configurations at different iteration thresholds, inheriting the original config with stage-specific overrides.

**See:** `config/monitoring/megatron_cooldown_multi.yaml`

Key features:
- Triggers at iterations 10000, 50000, and 100000
- **10k**: Quick evaluation only (no training)
- **50k**: Continued training with reduced LR
- **100k**: Final fine-tuning with very low LR
- Each cooldown gets unique output directory: `{original}_cooldown_{iteration}_{timestamp}`
- Each cooldown gets unique project name to avoid conflicts
- Uses simplified monitoring (`megatron_basic`) to prevent recursive cooldowns

### Usage

```bash
# Start training with multi-cooldown monitoring
python scripts/run_autoexp.py \
  job=scaling_study \
  monitoring=megatron_cooldown_multi \
  backend.args.train_iters=150000 \
  backend.args.save_interval=5000

# Three cooldown runs will automatically queue at iterations 10k, 50k, and 100k
```

### Key Advantages of RunAutoexpAction for Cooldown

1. **Configuration Inheritance**: All original training parameters are inherited automatically
2. **Minimal Overrides**: Only specify what changes (learning rate, iterations, save path)
3. **Reproducibility**: Uses `unresolved_config.yaml` from provenance for exact settings
4. **Flexible Path Management**: Load from original checkpoint (absolute path), save to new location
5. **No Custom Scripts**: Leverages existing `run_autoexp.py` infrastructure
6. **Escaped Interpolations**: Use `$${oc.timestring:}` (double `$$`) to defer timestamp resolution until cooldown run time

---

## Platform Detection and Platform-Specific Configs

### Current Approach

The system currently uses separate config files for each platform:
- `config/slurm/lumi.yaml`
- `config/slurm/leonardo.yaml`
- `config/slurm/juwels.yaml`
- `config/slurm/jupiter.yaml`
- `config/slurm/marenostrum.yaml`

Users select the platform explicitly: `python scripts/run_autoexp.py slurm=lumi`

### Recommended: Automatic Platform Detection

Add a platform detection script that can be sourced in configs:

Create `scripts/detect_platform.py`:

```python
#!/usr/bin/env python3
"""Detect the current HPC platform."""
import os
import socket
from pathlib import Path

def detect_platform() -> str:
    """Detect platform based on hostname, environment, or file system markers."""
    hostname = socket.gethostname().lower()

    # Check environment variables
    if os.environ.get("LUMI_PROJECT"):
        return "lumi"
    if os.environ.get("LEONARDO_PROJECT"):
        return "leonardo"
    if os.environ.get("JUWELS_PROJECT"):
        return "juwels"

    # Check hostname patterns
    if "lumi" in hostname:
        return "lumi"
    elif "leonardo" in hostname or "login" in hostname:
        # Leonardo has generic login node names, check file system
        if Path("/leonardo").exists():
            return "leonardo"
    elif "juwels" in hostname or "jrl" in hostname:
        return "juwels"
    elif "jupiter" in hostname or "jup" in hostname:
        return "jupiter"
    elif "mn" in hostname or "marenostrum" in hostname:
        return "marenostrum"

    # Check file system markers
    if Path("/appl/lumi").exists():
        return "lumi"
    elif Path("/leonardo_work").exists():
        return "leonardo"

    return "unknown"

if __name__ == "__main__":
    platform = detect_platform()
    print(platform)
```

### Platform-Specific Default Configs

Create a base config that auto-detects platform:

```yaml
# config/platform_auto.yaml
defaults:
  - slurm: ${platform_detect:}
  - container: ${platform_detect:}
  - _self_

# The ${platform_detect:} resolver would call detect_platform.py
```

Add to `oellm_autoexp/config/resolvers.py`:

```python
from omegaconf import OmegaConf
import subprocess

def platform_detect_resolver() -> str:
    """Detect platform by running detect_platform.py script."""
    try:
        result = subprocess.run(
            ["python", "scripts/detect_platform.py"],
            capture_output=True,
            text=True,
            check=True
        )
        platform = result.stdout.strip()
        if platform == "unknown":
            return "base"  # Fall back to base config
        return platform
    except Exception as e:
        print(f"Warning: Platform detection failed: {e}")
        return "base"

# Register with OmegaConf
OmegaConf.register_new_resolver("platform_detect", platform_detect_resolver)
```

### Decoupling Platform-Specific Configs

To reduce redundancy, structure configs hierarchically:

```
config/
├── slurm/
│   ├── base.yaml              # Common defaults
│   ├── lumi.yaml              # LUMI-specific (extends base)
│   ├── leonardo.yaml          # Leonardo-specific (extends base)
│   └── ...
├── backend/
│   ├── megatron/
│   │   ├── base.yaml          # Platform-agnostic settings
│   │   ├── base_llama.yaml    # Model-specific settings
│   │   └── ...
└── experiments/
    ├── base.yaml              # Common experiment structure
    └── megatron_lumi_speed_test.yaml  # Platform + experiment combo
```

Each platform config should extend base and only override platform-specific settings:

```yaml
# config/slurm/lumi.yaml
defaults:
  - base
  - _self_

sbatch:
  account: ${oc.env:LUMI_PROJECT}
  partition: standard-g
  time: "2:00:00"

sbatch_extra_directives:
  - "#SBATCH --gpus-per-node=8"
  - "#SBATCH --exclusive"
  - "#SBATCH --hint=nomultithread"

env:
  NCCL_SOCKET_IFNAME: hsn
  NCCL_NET_GDR_LEVEL: 3
  MIOPEN_USER_DB_PATH: /tmp/${oc.env:USER}-miopen-cache-${oc.env:SLURM_JOB_ID,unknown}
  MIOPEN_CUSTOM_CACHE_DIR: ${.MIOPEN_USER_DB_PATH}

srun:
  cpu_bind: mask_cpu:fe000000000000,fe00000000000000
  gpus_per_node: 8
```

---

## Consistent Job Naming for Automatic Resume

### Problem Statement

For automatic resume and cooldown to work correctly, the system needs:

1. **Stable experiment names** – Same across job restarts for checkpoint/log continuity
2. **Unique job identifiers** – Distinguish between different SLURM job submissions
3. **Consistent directory structure** – Allow automatic resume from checkpoints

### Recommended Approach

Following your scaling experiment pattern:

- **Experiment name**: `50M_lr0.1_gbsz128_stable` (stable across runs)
- **SLURM job name**: `50M_lr0.1_gbsz128_stable_{JOBID}_{TIMESTAMP}` (unique per submission)
- **Checkpoint directory**: `{output_dir}/50M_lr0.1_gbsz128_stable/checkpoints` (stable)
- **Log files**: `{output_dir}/50M_lr0.1_gbsz128_stable/logs/slurm-{JOBID}.out` (unique)

### Configuration Implementation

```yaml
# config/sweep/scaling_experiments.yaml
class_name: Sweep
axes:
  model_size:
    - 50M
    - 100M
    - 200M
  learning_rate:
    - 0.1
    - 0.01
    - 0.001
  global_batch_size:
    - 128
    - 256
    - 512


base_values:
  backend:
    args:
      # Training arguments...
      save_interval: 1000
      log_interval: 100
```

```yaml
# config/project/scaling_study.yaml
class_name: Project
name: scaling_study
base_output_dir: ${oc.env:OUTPUT_DIR}/scaling_experiments

# Monitoring state in stable location (no timestamps in path)
monitoring_state_dir: ${.base_output_dir}/monitoring_state
resume: true
```

```yaml
# config/slurm/base_with_unique_jobname.yaml
class_name: Slurm

# Template paths use stable experiment name
script_dir: ${project.base_output_dir}/{name}/scripts
log_dir: ${project.base_output_dir}/{name}/logs

# Log path includes SLURM job ID for uniqueness
log_path: ${project.log_path_current}

# SBATCH template should include unique job name
sbatch_overrides:
  # Job name: <experiment_name>_<job_id>_<timestamp>
  # SLURM variables are expanded at submission time
  job_name: "{name}_${{SLURM_JOB_ID:-pending}}_$(date +%Y%m%d_%H%M%S)"

# Output directory uses stable name (no job ID/timestamp)
# This is critical for checkpoint resume
```

### Enhanced Job Naming with Metadata

Update `oellm_autoexp/sweep/planner.py` to support enhanced naming:

```python
def build_job_plans(config: RootConfig, points: list[SweepPoint]) -> list[JobPlan]:
    base_output = Path(config.project.base_output_dir)
    project_name = config.project.name

    plans: list[JobPlan] = []
    for point in points:
        context: dict[str, str] = {
            "project_name": project_name,
            "index": str(point.index),
        }
        context.update(
            {key.replace(".", "___"): str(value) for key, value in point.parameters.items()}
        )

        # Generate stable experiment name (used for directories and checkpoints)
        experiment_name = config.sweep.name_template.format(**context)

        # Output directory uses stable name
        output_dir = str(base_output / experiment_name)

        # Log template can include SLURM variables for unique log files
        log_template = config.monitoring.log_path_template
        format_context = {
            **context,
            "output_dir": str(output_dir),
            "name": experiment_name,  # Stable name
        }
        log_path = log_template.format(**format_context)

        # ... rest of planning logic ...
```

### SLURM Template Updates

Update `templates/sbatch_template.slurm` to use stable vs unique naming:

```bash
#!/bin/bash
#SBATCH --job-name={{ job_name }}_${SLURM_JOB_ID:-$(date +%s)}
#SBATCH --output={{ log_path }}
#SBATCH --error={{ log_path }}
# ... other SBATCH directives ...

# Export stable experiment name for resume logic
export EXPERIMENT_NAME="{{ name }}"
export OUTPUT_DIR="{{ output_dir }}"
export CHECKPOINT_DIR="{{ output_dir }}/checkpoints"

# SLURM provides these automatically
# SLURM_JOB_ID - unique job ID
# SLURM_ARRAY_JOB_ID - array master ID (if array job)
# SLURM_ARRAY_TASK_ID - array task ID (if array job)

# Log job metadata for resume/tracking
echo "Job metadata:"
echo "  Experiment: ${EXPERIMENT_NAME}"
echo "  SLURM Job ID: ${SLURM_JOB_ID}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Checkpoints: ${CHECKPOINT_DIR}"
echo "  Started: $(date)"

# ... launch command ...
```

### Automatic Resume Logic

The monitoring system will automatically resume because:

1. **Stable checkpoint directory**: `{output_dir}/checkpoints` is same across job submissions
2. **Load argument consistency**: Backend config uses `backend.args.load={output_dir}/checkpoints`
3. **WandB/logging continuity**: Same experiment name means logs append to same run

Example monitoring config that supports this:

```yaml
# config/monitoring/megatron_auto_resume.yaml
class_name: SlurmLogMonitor
poll_interval_seconds: 30
inactivity_threshold_seconds: 1800

# Log path uses SLURM job ID for uniqueness
log_path: ${project.log_path_current}

# Additional output paths for inactivity detection
output_paths:
  - "{output_dir}/train.log"
  - "{output_dir}/checkpoints/latest_checkpointed_iteration.txt"

state_events:
  - name: timeout
    state:
      class_name: TimeoutState
    actions:
      # Automatic restart with same experiment directory
      - class_name: EventAction
        conditions:
          - class_name: MaxAttemptsCondition
            max_attempts: 5
        action:
          class_name: RestartAction
          reason: "auto_resume_after_timeout"
          # Backend will automatically resume from latest checkpoint
          # because load path points to stable checkpoint directory

  - name: crash
    state:
      class_name: CrashState
    actions:
      # Restart on non-OOM crashes
      - class_name: EventAction
        conditions:
          - class_name: MetadataCondition
            key: error_type
            not_equals: oom
          - class_name: MaxAttemptsCondition
            max_attempts: 3
        action:
          class_name: RestartAction
          reason: "auto_resume_after_crash"
```

### Cooldown with Stable Naming

The cooldown scheduler can read from the stable checkpoint directory:

```python
# scripts/cooldown_scheduler.py
#!/usr/bin/env python3
"""Monitor checkpoint directory and trigger cooldown at specific iterations."""
import argparse
from pathlib import Path
import time

def get_latest_iteration(checkpoint_dir: Path) -> int:
    """Read latest checkpointed iteration from stable directory."""
    latest_file = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if latest_file.exists():
        return int(latest_file.read_text().strip())
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--cooldown-iteration", type=int, required=True)
    parser.add_argument("--poll-interval", type=int, default=60)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    cooldown_triggered = False

    print(f"Monitoring {checkpoint_dir} for iteration {args.cooldown_iteration}")

    while not cooldown_triggered:
        current_iter = get_latest_iteration(checkpoint_dir)
        print(f"Current iteration: {current_iter}")

        if current_iter >= args.cooldown_iteration:
            print(f"Cooldown iteration {args.cooldown_iteration} reached!")
            # Trigger cooldown actions
            cooldown_triggered = True
        else:
            time.sleep(args.poll_interval)

if __name__ == "__main__":
    main()
```

---

## Summary Table

| Feature | Current Support | Configuration Location | Notes |
|---------|----------------|----------------------|-------|
| **Automatic Cooldown** | ✅ Supported by way of log events + queued actions | `config/monitoring/megatron_multi_cooldown.yaml` | Use iteration detection + conditional actions |
| **Checkpoint → HF Conversion** | ✅ Supported by way of `RunCommandAction` | `config/monitoring/megatron_checkpoint_hf_conversion.yaml` | Requires conversion script implementation |
| **Automatic Evaluation** | ✅ Supported by way of `RunAutoexpAction` | `config/monitoring/megatron_checkpoint_eval.yaml` | Already in codebase |
| **Multiple Cooldowns** | ✅ Supported by way of multiple conditional actions | `config/monitoring/megatron_multi_cooldown.yaml` | Use iteration-based metadata conditions |
| **Platform Detection** | ⚠️ Manual selection required | Add `scripts/detect_platform.py` | Recommendation: implement auto-detection resolver |
| **Platform-Specific Configs** | ✅ Hierarchical config system exists | `config/slurm/{platform}.yaml` extends `base.yaml` | Already well-decoupled |
| **Stable Job Naming** | ⚠️ Requires config adjustment | Update `sweep.name_template` and SBATCH template | Implement stable experiment name + unique SLURM job ID |
| **Automatic Resume** | ✅ Supported by way of stable checkpoint paths | Standard monitoring with restart actions | Works when checkpoint dir is stable |

---

## Next Steps

### Implementation Priority

1. **High Priority - Automatic Cooldown**:
   - Create `config/monitoring/megatron_multi_cooldown.yaml` ✅ (documented above)
   - Implement `scripts/cooldown_handler.py` template ✅ (documented above)
   - Test with real training runs

2. **High Priority - Stable Job Naming**:
   - Update SBATCH templates to use `{name}_{JOBID}_{TIMESTAMP}` pattern
   - Ensure checkpoint directories use stable experiment name only
   - Update documentation on naming conventions

3. **Medium Priority - Platform Auto-Detection**:
   - Implement `scripts/detect_platform.py` ✅ (documented above)
   - Add `platform_detect` OmegaConf resolver
   - Test on each platform (LUMI, Leonardo, JUWELS, etc.)

4. **Medium Priority - Checkpoint Conversion**:
   - Implement `scripts/convert_checkpoint_to_hf.py` for your specific model
   - Add HuggingFace conversion to monitoring configs
   - Test conversion + evaluation pipeline

5. **Low Priority - Documentation**:
   - Add examples to main README
   - Create platform-specific setup guides
   - Document cooldown patterns and best practices

### Testing Recommendations

1. **Cooldown Testing**:
   ```bash
   # Test with short iteration counts
   python scripts/run_autoexp.py \
     job=cooldown_test \
     monitoring=megatron_multi_cooldown \
     backend.args.train_iters=15000 \
     backend.args.save_interval=5000
   ```

2. **Platform Detection Testing**:
   ```bash
   # Run on each platform
   python scripts/detect_platform.py
   # Should output: lumi, leonardo, juwels, etc.
   ```

3. **Resume Testing**:
   ```bash
   # Start job, let it checkpoint, cancel it
   python scripts/run_autoexp.py job=resume_test monitoring=megatron_auto_resume
   # Wait for first checkpoint, then:
   scancel <job_id>
   # Monitor should automatically restart and resume from checkpoint
   ```
