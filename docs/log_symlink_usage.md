# Log Symlink Usage

## Overview

The `log_path_current` feature creates a stable symlink (`current.log`) that always points to the most recent SLURM log file for each job. This is especially useful for multi-stage training where cooldown jobs need to monitor the stable job's log file.

## How It Works

1. **Automatic Path Generation**: When jobs are resolved, each job gets a `log_path_current` field computed from its `log_path`:
   - `log_path`: `/logs/job_name/slurm-%j.out`
   - `log_path_current`: `/logs/job_name/current.log`

2. **Template Variable**: The orchestrator generates a `{log_symlink_cmd}` variable that you can add to your SLURM template:
   ```bash
   ln -sf "slurm-${SLURM_JOB_ID}.out" "/logs/job_name/current.log"
   ```

3. **Runtime Creation**: When the job starts, the symlink is created/updated to point to the actual log file with the SLURM job ID.

## Using in SLURM Templates

Add the `{log_symlink_cmd}` to your SLURM template (for example, `template.sbatch`):

```bash
#!/bin/bash
{sbatch_directives}

# Create symlink to current log file (for multi-stage monitoring)
{log_symlink_cmd}

# Your job execution
srun {srun_opts}{launcher_cmd}
```

### Complete Example

```bash
#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --output={log_path}
#SBATCH --error={log_path}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

# Create/update symlink to most recent log file
{log_symlink_cmd}

# Load environment
module load cuda/11.8

# Run training
srun python train.py {backend_cmd}
```

## Multi-Stage Training Example

When using multi-stage training with sibling references, the cooldown job can monitor the stable job's log by way of the symlink:

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
          backend.megatron.load: "{sibling.stable.output_dir}/checkpoint"
          cancel_conditions:
            - class_name: LogPatternCondition
              log_path: "{sibling.stable.log_path_current}"  # Uses symlink!
              pattern: "FATAL ERROR|OutOfMemoryError"
```

The symlink provides:
- **Stable Path**: Always `/logs/lr1e-4_stable/current.log`, regardless of SLURM job ID
- **Automatic Updates**: If the stable job is resubmitted, the symlink updates to the new log
- **Easy Monitoring**: Cancel conditions can reference a fixed path

## Benefits

1. **Stable References**: No need to know the SLURM job ID in advance
2. **Job Restarts**: Symlink updates automatically on resubmission
3. **Historical Logs**: Original `slurm-JOBID.out` files are preserved
4. **Debug Friendly**: Easy to tail the current log: `tail -f /logs/job_name/current.log`

## File Structure

After jobs run, your log directory looks like:

```
/logs/
  lr1e-4_stable/
    slurm-12345.out      # First run
    slurm-12346.out      # Second run (after restart)
    current.log → slurm-12346.out  # Symlink to most recent

  lr1e-4_cooldown/
    slurm-12347.out
    current.log → slurm-12347.out
```

## Implementation Details

- The symlink command is only generated if `log_path_current` is set (which happens automatically during sibling resolution)
- If your job doesn't use multi-stage features, `{log_symlink_cmd}` will be empty
- The symlink is created in the same directory as the log file
- Uses `-sf` flag to force overwrite if symlink exists (handles restarts)
