# oellm-autoexp on JUPITER (Qwen3-1.7B-Base)


## 1. Micromamba

```bash
mkdir -p /e/scratch/<proj>/<user>/oellm_env/{bin,mamba_root}
cd /e/scratch/<proj>/<user>/oellm_env/bin
curl -Ls "https://micro.mamba.pm/api/micromamba/linux-aarch64/latest" \
  | tar -xj bin/micromamba --strip-components=1
chmod +x micromamba
```

## 2. Clone + Megatron

```bash
git clone https://github.com/OpenEuroLLM/oellm-autoexp.git
cd oellm-autoexp
git submodule update --init submodules/Megatron-LM
```

## 3. Env

```bash
module --force purge
export MAMBA_ROOT_PREFIX=/e/scratch/<proj>/<user>/oellm_env/mamba_root
export PATH=/e/scratch/<proj>/<user>/oellm_env/bin:$PATH
eval "$(micromamba shell hook --shell bash)"

micromamba create -n oellm-mamba -c conda-forge \
  python=3.12 gcc gxx cmake make pkg-config -y
micromamba activate oellm-mamba

pip install -U pip && pip install -e .[dev,megatron]
python -c "import compoconf, oellm_autoexp; print(compoconf.__version__)"
```

## 4. Container (.sif)

You can use the prebuilt one:

```bash
export CONTAINER_CACHE_DIR=/e/data1/datasets/playground/mmlaion/shared/oellm_shared_evals
# image: MegatronTraining-JUPITER_aarch64_202603120039.sif (pinned in config/container/jupiter.yaml)
```

To build it yourself (ARM64, the x86 nvcr base won't work):

```bash
export CONTAINER_CACHE_DIR=/e/scratch/<proj>/<user>/cache
export APPTAINER_CACHEDIR=/e/scratch/<proj>/<user>/.apptainer/cache
export APPTAINER_TMPDIR=/e/scratch/<proj>/<user>/.apptainer/tmp
bash container/build_container.sh --definition MegatronTraining-JUPITER
```

## 5. Megatron lives on the host, not in the .sif

Loaded at runtime via `PYTHONPATH=.:submodules/Megatron-LM`. To swap Megatron: check out a different commit in the submodule, or point `backend.launcher_script` + `backend.env.PYTHONPATH` at another clone under `/e`.

## 6. Run

```bash
export OUTPUT_DIR=/e/.../experiments
export HF_HOME=/e/.../hf_home
export CONTAINER_CACHE_DIR=/e/.../   
export OELLM_DATASETS_TOKENIZED_DIR=/e/.../tokenized_data
export OELLM_CACHE_DIR=/e/.../.cache

PYTHONPATH=. python scripts/run_autoexp.py --config-name experiments/<you>/<exp>
```

---

## 7. Walkthroughs

### 7.1. Throughput sweep (N nodes × 3 trials, 5 min each)

`throughput_sweep_qwen3_1.7b_jupiter_repro.yaml` is a `product` sweep over `slurm.sbatch.nodes` × `backend.megatron.aux.trial`. 

Submit:

```bash
PYTHONPATH=. python scripts/run_autoexp.py \
  --config-name experiments/throughput_sweep_qwen3_1.7b_jupiter_repro \
  --array-subset 0-5 --submit-and-exit
```

**Verified on this PR** (qwen3-1.7B bf16 seq 4096):

| nodes | GBS | trials done | mean TFLOP/s/GPU | stdev |
|------:|----:|:-----------:|-----------------:|------:|
| 1     | 8   | 2 / 3       | **284.6**        | 4.9   |
| 2     | 16  | 3 / 3       | **269.9**        | 4.8   |
| 4     | 32  | 3 / 3       | **261.7**        | 3.7   |
| 8     | 64  | 3 / 3       | **254.8**        | 5.4   |

### 7.2. Full training + auto-eval

`qwen3_1.7b_mixvitae_jupiter_fullrun_repro.yaml` wires:
- same-job postprocess: dist -> torch, then HF conversion.
- new-job postprocess: `oellm schedule-eval` on `open-sci-0.01`

```bash
PYTHONPATH=. python scripts/run_autoexp.py \
  --config-name experiments/qwen3_1.7b_mixvitae_jupiter_fullrun_repro
```
