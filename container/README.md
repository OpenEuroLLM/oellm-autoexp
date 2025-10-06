# oellm-autoexp Container Assets

This directory mirrors the container tooling for `megatron-train` with a
streamlined definition tailored to the autoexp orchestrator. Definition
templates now live under backend-specific folders (e.g. `megatron/`), and the
helper script `build_container.sh` renders the selected
Singularity/Apptainer definition, substitutes the desired base image and
requirements file, and builds `MegatronTraining_<arch>.sif` by default.

Typical usage:

```bash
./container/build_container.sh --backend megatron --definition MegatronTraining \
    --requirements container/megatron/requirements_latest.txt \
    --output ./artifacts
```

The generated image carries the repo inside `/workspace/oellm-autoexp` and is
compatible with `scripts/run_megatron_container.py`.
