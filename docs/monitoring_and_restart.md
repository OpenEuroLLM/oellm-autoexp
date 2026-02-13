# Monitoring and Automatic Restart

This document explains how the monitoring stack detects failures, persists events, and reacts through the unified Event → Condition → Action primitives. The legacy `config/restart/*.yaml` policies have been removed; everything now lives inside the monitoring configuration itself.

## Overview

1. **Log + SLURM monitoring** – `SlurmLogMonitor` watches both SLURM state and log/output files. Regex or substring matches emit named `log_events`; SLURM transitions emit `state_events`.
2. **Inline bindings** – Each event carries a list of `EventAction` bindings. A binding can include conditions (`MetadataCondition`, `MaxAttemptsCondition`, `FileExistsCondition`, …) and chooses whether to run inline or enqueue work.
3. **Action queue** – Queued actions are stored as individual JSON files under `{monitoring_state_dir}/session.actions/<event_id>/<action_id>.json`. Workers can inspect/replay items without touching the monitor.
4. **Controller decisions** – `MonitorController` restarts jobs when an inline action returns `retry`, finalises success/stall/crash states, and records every action/event transition for post-mortems.

## Monitoring Configurations

All configurations live in `config/monitoring/` and compose by way of Hydra defaults. They share the same primitives but target different workflows:

| Config | Purpose | Highlights |
| --- | --- | --- |
| `megatron_basic.yaml` | Minimal Megatron monitoring | Success detection, checkpoint logging, CUDA/Python error capture |
| `megatron_nccl.yaml` | NCCL-focused runs (`NCCL_DEBUG=info`) | Extra `log_events` for NCCL ERROR/WARN, timeouts, collectives |
| `megatron_stall_detection.yaml` | Aggressive hang detection | Lower inactivity threshold, CUDA hang/deadlock patterns |
| `megatron_production.yaml` | Recommended default | Combines success, OOM, NCCL, stall, and numerical diagnostics |
| `megatron_checkpoint_eval.yaml` | Checkpoint-triggered evals | Adds queued `RunCommandAction` + `RunAutoexpAction` for each checkpoint |
| `megatron_followup.yaml` | Follow-up automation after success | Queues a cooldown command and a second `run_autoexp.py` invocation |

Each config can be selected with `python scripts/run_autoexp.py ... monitoring=<name>`.

## Event Bindings

### Anatomy of a `log_events` entry

```yaml
log_events:
  - name: cuda_oom
    pattern: "(?:CUDA out of memory).*?(?P<bytes>\\d+ GiB)"
    pattern_type: regex
    state:
      class_name: CrashState
    metadata:
      severity: critical
      error_type: oom
    extract_groups:
      allocation_bytes: bytes
    actions:
      - class_name: EventAction
        conditions:
          - class_name: MaxAttemptsCondition
            max_attempts: 2
        action:
          class_name: RestartAction
          reason: "retry oom"
```

- **Detection** – Regex match emits `cuda_oom` with crash state.
- **Metadata** – Capture groups (for example, `{allocation_bytes}`) become part of the action context.
- **Actions** – Inline binding restarts at most twice; subsequent OOMs stop the job.

### `state_events`

```yaml
state_events:
  - name: success
    actions:
      - class_name: EventAction
        mode: queue
        action:
          class_name: RunAutoexpAction
          script: scripts/run_autoexp.py
          config_path: "{output_dir}/provenance/config_reference.json"
          overrides:
            - "+mode=evaluation"
      - class_name: EventAction
        action:
          class_name: LogAction
          message: "run_finished"
```

`state_events` fire off SLURM lifecycle transitions. Here, a clean finish queues an evaluation run and logs a note.

## Action Queue Layout

- Location: `{monitoring_state_dir}/<plan_id>.actions/`
- Structure: one directory per `event_id`, one JSON file per action (`<action_id>.json`)
- Contents:

```json
{
  "queue_id": "bcf2dafa-...",
  "event_id": "123:checkpoint_saved:171225213",
  "action_class": "RunCommandAction",
  "config": {"command": ["python", "scripts/run_eval.py", "--ckpt", "/.../iter_1000.pt"]},
  "metadata": {"job": {"job_name": "demo_0", "output_dir": "/.../demo_0"}},
  "status": "pending"
}
```

Workers pop files, execute the described action, and mark the entry as `done` or `failed`. Successful entries are deleted automatically so the queue reflects the current backlog.

## Restart Flow

1. Monitor detects failure (log or SLURM). `MonitorController` persists/updates an `EventRecord`.
2. Action bindings execute. An inline `RestartAction` returns `status="retry"` which the controller interprets as a restart request.
3. Controller cancels/resubmits the job, increments the attempt counter, and records a `MonitorRecord` linking old/new job IDs.
4. If no restart action fired, the job ends in `stop` state with metadata describing the reason.

Because everything hinges on a single primitive set, you can explain any restart by inspecting:
- the event record (`monitoring_state_dir/<plan>.json`),
- the action queue,
- the controller logs (`oellm_autoexp.monitor.controller` logger).

## Usage Examples

### Basic restart on cancelled jobs

```yaml
state_events:
  - name: crash
    actions:
      - class_name: EventAction
        conditions:
          - class_name: MetadataCondition
            key: error_type
            equals: cancelled
        action:
          class_name: RestartAction
          reason: "retry cancelled job"
```

### Queue checkpoint evaluations

```yaml
log_events:
  - name: checkpoint_saved
    pattern: "saved checkpoint to (?P<ckpt>\\S+)"
    pattern_type: regex
    metadata:
      kind: checkpoint
    actions:
      - class_name: EventAction
        mode: queue
        conditions:
          - class_name: FileExistsCondition
            path: "{ckpt}"
            blocking: true
            timeout_seconds: 900
        action:
          class_name: RunCommandAction
          command:
            - python
            - scripts/run_checkpoint_eval.py
            - "--checkpoint"
            - "{ckpt}"
```

### Trigger second autoexp run after success

Use `monitoring=megatron_followup` (ships with this repository) or embed the same snippet:

```yaml
state_events:
  - name: success
    actions:
      - class_name: EventAction
        mode: queue
        action:
          class_name: RunAutoexpAction
          config_path: "{output_dir}/provenance/config_reference.json"
          overrides:
            - "+mode=evaluation"
```

Run planning/submission as usual; queued entries will appear under `monitoring_state/<session>.actions/`.
