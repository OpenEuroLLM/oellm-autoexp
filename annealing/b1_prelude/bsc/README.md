# B1 32B annealing on MareNostrum 5

Use this package to reproduce the two BSC annealing validations for the Prelude dominant1 and dominant2 mixtures. Each starts from the full-state B1 checkpoint at iteration 416000, follows the 20B-token sample-based WSD schedule with `minus_sqrt`, resets the validation cursor at the fork, and stops after updates 416001–416003. Training uses TP4/PP4, tensorwise FP8, BF16 at the edges, sequence length 4096 and global batch 4096. Each smoke has a ten-minute allocation and one 16-sample validation batch. These short checks have an explicit no-save endpoint. See the [JUPITER package](../README.md) for the existing full-run workflow and its rolling checkpoint and recovery requirements.

## Account and launch layout

Use a MareNostrum 5 ACC login and a user authorized for `ehpc533` with read access to the shared image, data and checkpoint and a writable project directory. The published evidence was collected as `fzju647849`. The default uses 64 nodes and 256 H100s under `acc_ehpc`. A 32-node render is also supported. The launch follows the inspected BSC ambitious and Qwen 3.5 runs with one Slurm task per node, four container ranks, `singularity/4.1.5`, GPFS mounts and explicit step GPU binding.

Both original 64-node allocations failed before training when Transformer Engine attempted CUDA multicast for TP communication overlap. This package removes `--tp-comm-overlap` on BSC, matching those two BSC launch configurations. The original JUPITER recipe, FP8 settings, TP/PP sizes, sequence parallelism, and data-parallel overlap flags remain in effect for their respective launches. See [BSC results](reference/site-validation.json) for the job evidence and validation status.

Requests above 100 nodes use `acc_xehpc`. The live `ehpc533` association includes that QoS for the executing user. Slurm accepted 128 and 256 nodes with it without a reservation, and rejected both with `acc_ehpc` at the QoS CPU limit. Run the read-only admission checks below to confirm the association for your account. The 16-sample validation batch supports 32- and 64-node launches. At 128 and 256 nodes TP4/PP4 would require validation batches divisible by DP32 and DP64 respectively, plus new cache coverage and runtime validation. The admission checks use `sbatch --test-only` and consume zero allocations.

## Start from a fresh checkout

Use your approved host Python 3.11 or newer for the standard-library controller. GPU Python, PyTorch and Transformer Engine come from the existing shared container. Run these commands on the BSC ACC controller after establishing repository and project access.

```bash
run="/gpfs/projects/ehpc533/users/$USER/runtime/b1-annealing-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$(dirname "$run")"
mkdir -m 700 "$run"
git clone --branch prod/oellm_32b_dense https://github.com/OpenEuroLLM/oellm-autoexp.git "$run/checkout"
repo="$run/checkout"
git -C "$repo" submodule update --init submodules/Megatron-LM
tools="$repo/annealing/b1_prelude/bsc"
bundle="$tools/reference"
image=/gpfs/projects/ehpc533/users/korbip/hybrid_v019/container/MegatronTraining-hybrid-ep_x86_64_202610161948.sif
test -r "$image"
```

Use the pinned training revisions so the checkout that supplies the controller can advance independently. The recipe records archive hashes for the BSC validation.

```bash
autoexp=dd112d92f99d175a8ab1701b57c8dea0363b53f2
megatron=c9cf7e84cf584dee5ca456124180c5acc98e80ba
git -C "$repo" archive --format=tar.gz "$autoexp" > "$run/autoexp.tar.gz"
git -C "$repo/submodules/Megatron-LM" archive --format=tar.gz "$megatron" > "$run/megatron.tar.gz"
(cd "$run" && python3 - "$bundle/recipe.json" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

for name, expected in json.loads(Path(sys.argv[1]).read_text())["source_archives"].items():
    with Path(name).open("rb") as archive:
        actual = hashlib.file_digest(archive, "sha256").hexdigest()
    if actual != expected:
        raise SystemExit(f"Archive identity differs: {name}")
PY
)
mkdir -p "$run/source/submodules/Megatron-LM"
tar -xzf "$run/autoexp.tar.gz" -C "$run/source"
tar -xzf "$run/megatron.tar.gz" -C "$run/source/submodules/Megatron-LM"
```

Set `B1_SEED` to a readable SHA-verified copy of the full-state `iter_0416000` directory. It must include the verified transfer marker and the original `.metadata`. The 474.56 GB copy in the evidence was made from JUPITER with the existing transfer helper, full source-to-destination SHA-256, and approved BSC transfer movers. The model payload and access credentials are external inputs. [Verified seed receipt](reference/runtime-validation.json) · [Transfer workflow](https://github.com/SLAMPAI/oellm-workflows/blob/main/transfer/BSC_TO_JSC_HYBRID.md#reverse-direction-and-evaluation-handoff)

```bash
: "${B1_SEED:?Set the path to the verified iter_0416000 directory}"
test -r "$B1_SEED/.hpc-transfer-verified.json"
test -r "$B1_SEED/.metadata"
python3 "$tools/bsc_b1_annealing.py" prepare --root "$run/arms" \
  --bundle "$bundle" --inventory "$bundle/bsc-prefixes.json" \
  --source "$run/source" --seed "$B1_SEED" --image "$image" --nodes 64
```

Preparation validates the tokenizer hashes and all 294 shared dataset pairs by readability, file sizes and the first 256 index bytes. The checked-in map points to the shared `training/collection/flag` and `training-data-collection/flag` BSC trees. Run `audit "$bundle/expected.json" ROOT ...` if the available roots change, review its matches, and pass the new JSON to `prepare --inventory`. The renderer preserves the original decimal weights and ordering. [Pinned inputs](reference/recipe.json)

## Check permissions and run

The first block checks large-node access without submitting either request. The second submits a one-node FP8 runtime probe and starts the controller. Run it inside a persistent BSC tmux session so the controller can observe Slurm across logouts.

```bash
sacctmgr -nP show assoc where user="$USER" account=ehpc533 format=User,Account,Partition,QOS
python3 "$tools/bsc_b1_annealing.py" admission "$run/arms/dominant1" --nodes 128
python3 "$tools/bsc_b1_annealing.py" admission "$run/arms/dominant1" --nodes 256
```

```bash
python3 "$tools/bsc_b1_annealing.py" probe --root "$run" \
  --source "$run/source" --image "$image"
python3 "$tools/bsc_b1_annealing.py" watch "$run" --hours 168
```

The controller sequences one-native-update cache jobs through the one-submit `acc_debug` limit. Before each ten-minute smoke it requires all 294 child identities, the ordered normalized blend and exact 12,288/16 sample coverage, healthy cache arrays and the verified seed. The two caches have separate writer leases. Previously completed matching caches can be linked with `reuse-cache NEW_ARM ORIGINAL_ARM` before starting the new controller. The controller verifies the original native job receipt and matching cache settings, mix and tokenizer. Preserve the original cache and its receipts while a linked arm uses them.

Read `"$run/status.json"` for job IDs and current states. Each passed smoke writes `smoke-proof.json` under its arm with the checkpoint restore, cursor reset, three finite numerical updates and consecutive-update loss changes. Run `result "$run/arms/dominant1" smoke` to replay that check from Slurm accounting and its log. A `STOP` file in the runtime or arm prevents further submissions. An arm with a terminal failure retains its intent, receipt and log. Create a fresh arm for a retry.

The probe and both cache builds used 0.566 ACC node-hours. The planned ceiling including preparation and two 64-node smokes is 22.42 ACC node-hours. Reuse avoids duplicate cache allocations when the inputs match. [Runtime and cache evidence](reference/site-validation.json)
