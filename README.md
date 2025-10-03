# OELLM Auto Experimentation Tool

Work-in-progress monorepo consolidating sweep planning, SLURM submission, and monitoring for
OpenEuroLLM projects. The Megatron backend reuses Megatron-LM's native argument parser/command builder so sweeps remain compatible with upstream flags. See `SPEC.md` for the architectural plan.

## Quick Start

```bash
pip install -e .[dev]

# Use the bundled Hydra configs (examples for JUWELS/JUPITER)
oellm-autoexp plan autoexp --config-dir config -o project=juwels -o slurm=juwels
oellm-autoexp submit autoexp --config-dir config -o project=juwels -o slurm=juwels --fake

# Or render a concrete YAML via scripts/render_config.py
python scripts/render_config.py autoexp -C config -o project=jupiter

# Container validation (apptainer/singularity)
python scripts/run_megatron_container.py --image path/to.sif --config-ref autoexp -C config -o project=juwels --fake-submit

### Customising SLURM / srun / launcher

SLURM templates now consume aggregate directives rendered from the config:

- `config/slurm/*.yaml` exposes `sbatch_overrides` (converted into `#SBATCH` lines) plus `sbatch_extra_directives` for free-form additions.
- Each SLURM config also carries `launcher_cmd` and `srun_opts` so you can prepend a wrapper or supply extra `srun` flags inline.

Templates include placeholders `{sbatch_directives}`, `{env_exports}`, `{srun_opts}` and `{launcher_cmd}`; create per-cluster variants by overriding the relevant Hydra groups.
```
