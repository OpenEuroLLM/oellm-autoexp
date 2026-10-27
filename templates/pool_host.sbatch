#!/bin/bash
{sbatch_directives}

{env_exports}

# Pool host (oellm_autoexp/monitor/pool.py): the batch shell runs the supervisor, which runs the
# runs the monitor assigns to this allocation as srun steps. No outer srun, no container: each
# run's step script brings its own launcher. Use `#SBATCH --signal=B:USR1@<margin>` so the
# supervisor can stop the current run gracefully (checkpoint) before the wall clock.
# `exec`: the supervisor must BE the batch shell. `--signal=B:...` signals only the batch
# shell's process; as a child, the supervisor would never see it and the untrapped shell
# would die of the signal, taking the allocation (and the run) down without a save.
ulimit -c 0
echo "Launching pool host {job_name}"
exec {command}
