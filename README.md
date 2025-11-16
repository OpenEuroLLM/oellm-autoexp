# OELLM Auto Experimentation Tool

Single CLI surface for planning sweeps, launching jobs (directly or by way of containers), and monitoring SLURM runs by way of declarative configs. See `SPEC.md` for platform-wide goals; this README focuses on the workflows you touch every day.

## Cluster setup: LUMI notes
- Install prerequisites outside the container (rccl-tuner, cray-python, etc.) following LUMI docs.
- Build the Megatron container from the provided defs (see `container/megatron/MegatronTrainingLumi.def.in`) so the correct ROCm + network tuning ends up inside the image.
- Export the usual SLURM/paths (at a minimum `SLURM_ACCOUNT`, `SLURM_PARTITION[_DEBUG]`, `CONTAINER_CACHE_DIR`, `OUTPUT_DIR`) in your profile—scripts read them automatically.

## Quick Recipes

### Single job / Sweep debugging
```bash
# Plan + submit + monitor in one go (manifest written to outputs/manifests/)
python scripts/run_autoexp.py project=default

# Prefer explicit plan/submit?
python scripts/plan_autoexp.py --manifest outputs/manifests/demo.json project=default
python scripts/submit_autoexp.py --manifest outputs/manifests/demo.json

# Probe a specific sweep entry
python scripts/run_sweep_entry.py --sweep outputs/sweep.json --index 0 --dry-run
```

### Monitoring sessions
- Every submission drops `<monitoring_state_dir>/<plan_id>.json`. Resuming is symmetric:
```bash
python scripts/monitor_autoexp.py --session <plan_id>
# legacy path works too
python scripts/monitor_autoexp.py --manifest outputs/manifests/<plan>.json
```
- The session file stores `resolved_log_path`, last SLURM state, per-event history, and the manifest path so you can crash/restart without guessing log names. `scripts/tests/test_monitor_resume.py` is the on-cluster smoke test that exercises plan → submit → interrupt → resume.
- The action queue lives next to the session file (`<plan_id>.actions/`). Use the CLI to keep it short and readable:
```bash
# Summaries (optionally filter by event)
python scripts/monitor_autoexp.py --session <plan_id> --cmd queue
python scripts/monitor_autoexp.py --session <plan_id> --cmd queue --queue-event checkpoint_saved

# Inspect or retry a specific entry
python scripts/monitor_autoexp.py --session <plan_id> --cmd queue \
    --queue-id <queue_uuid> --queue-show
python scripts/monitor_autoexp.py --session <plan_id> --cmd queue \
    --queue-id <queue_uuid> --queue-retry

# Run the worker that executes queued actions (for example, RunAutoexpAction)
python scripts/monitor_autoexp.py --session <plan_id> --cmd actions
```
Each entry is a single JSON file (`<event_id>/<queue_id>.json`) so manual inspection is always possible, but the CLI avoids editing files directly.

### Container workflow (recommended for Megatron)
```bash
# Build once (see container/MegatronTraining*.def for examples)
./container/build_container.sh --backend megatron --definition MegatronTraining --output ./artifacts
# On LUMI or other clusters, where --fake-root for container builds is not available
python ./container/build_container_user.py --backend megatron --definition MegatronTrainingLumi \
    --requirements container/megatron/requirements_latest.txt \
    --output ./artifacts

# Plan inside the container, submit/monitor on host
python scripts/run_autoexp_container.py \
  --config-ref experiments/megatron_jupiter_speed_test \
  container=juwels

# Inspect or reuse the manifest later
python scripts/monitor_autoexp.py --manifest outputs/manifests/<plan>.json
```
`run_autoexp_container.py` accepts `--no-submit`, `--no-monitor`, and `--monitor-only` so the same manifest feeds both automation and manual debugging.

## Monitoring configs (log/state events)
Monitoring behavior lives entirely in YAML. Keep it small, keep it explicit:

```yaml
# config/monitoring/megatron_checkpoint_eval.yaml
log_events:
  - name: checkpoint_saved
    pattern: "saved checkpoint (?P<path>\\S+)"
    pattern_type: regex
    metadata:
      kind: checkpoint
    extract_groups:
      checkpoint_path: path
    actions:
      - class_name: EventAction
        mode: queue
        conditions:
          - class_name: FileExistsCondition
            path: "{checkpoint_path}"
            blocking: true
        action:
          class_name: RunAutoexpAction
          config_path: "{output_dir}/_provenance/config_reference.json"
          overrides:
            - evaluation.checkpoint={checkpoint_path}

state_events:
  - name: stall
    state:
      class_name: StalledState
    actions:
      - class_name: EventAction
        action:
          class_name: RestartAction
          reason: "retry stall"
  - name: success
    state:
      class_name: SuccessState
    actions:
      - class_name: EventAction
        action:
          class_name: LogAction
          message: "run_finished"
```

`log_events` describe detectors (regex/substring/inactivity). `state_events` wire SLURM transitions (`stall`, `timeout`, `success`, etc.) to actions. Because nothing is hidden in code, configs like `config/monitoring/megatron_basic.yaml` act as the canonical reference: fork it, add or remove events, and the controller will follow exactly what the YAML states.

### Tips
- Use `pattern_type: inactivity` to emit events when the log or additional output paths stop changing for `inactivity_threshold_seconds`.
- Queue actions (`mode: queue`) for sidecar jobs or evaluations; inline actions are for immediate restarts/logging.
- Everything templated by way of `str.format` gets the merged metadata (`job_id`, `checkpoint_path`, etc.) so downstream automation stays simple.

For more context (provenance capture, fake SLURM client, restart CLI), refer to `docs/` and the unit/integration tests—they mirror real-world usage.
