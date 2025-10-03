#!/bin/bash
{sbatch_directives}

{env_exports}

echo "Launching job {name}"
srun {srun_opts}{launcher_cmd}
