# oellm-autoexp Container Assets

This directory mirrors the container tooling for `megatron-train` with a
streamlined definition tailored to the autoexp orchestrator. Definition
templates now live under backend-specific folders (for example `megatron/`), and the
helper script `build_container.sh` renders the selected
Singularity/Apptainer definition, substitutes the desired base image and
requirements file, and builds `MegatronTraining_<arch>.sif` by default.

Typical usage:

```bash
./container/build_container.sh --backend megatron --definition MegatronTraining \
    --requirements container/megatron/requirements_latest.txt \
    --output ./artifacts
```

The generated image carries the repository inside `/workspace/oellm-autoexp` and is
compatible with `scripts/run_megatron_container.py`.


LUMI
```bash
python ./container/build_container_user.py --backend megatron --definition MegatronTrainingLumi \
    --requirements container/megatron/requirements_latest.txt \
    --output ./artifacts
```

LUMI + Megatron-Bridge (conversion image; adds Bridge's extra dependencies,
including `nvidia-modelopt`, which `MegatronTrainingLumi` lacks and
`from megatron.bridge import AutoBridge` needs at import time)
```bash
python ./container/build_container_user.py --backend megatron --definition MegatronTrainingLumi-bridge \
    --requirements container/megatron/requirements_latest.txt
```
No `--output`: the image lands in `$CONTAINER_CACHE_DIR` (or the current
directory if unset) as `MegatronTrainingLumi-bridge_*_x86_64.sif`, which is
where the chain config's convert-stage `container.image` looks; see
[`docs/bridge_eval_setup.md`](../docs/bridge_eval_setup.md).
