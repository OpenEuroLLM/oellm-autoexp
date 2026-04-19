"""Load an MoE model from a checkpoint for inspection and optional validation.

Accepts a checkpoint path and automatically extracts the iteration step.
Pass either the iter directory or the parent checkpoints directory:
    .../checkpoints/iter_0114441   → loads step 114441
    .../checkpoints                → loads latest checkpoint

Run with torchrun (EP size must divide num_experts evenly):
    torchrun --nproc_per_node=2 scripts/debug/load_model_from_checkpoint.py \
        --checkpoint-path results/.../checkpoints/iter_0114441

    torchrun --nproc_per_node=4 scripts/debug/load_model_from_checkpoint.py \
        --checkpoint-path results/.../checkpoints/iter_0114441 --num-experts 32

Evaluate validation loss (pass Megatron data args after --):
    torchrun --nproc_per_node=2 scripts/debug/load_model_from_checkpoint.py \
        --checkpoint-path results/.../checkpoints/iter_0114441 --evaluate \
        --valid-data-path /path/to/valid/data \
        --micro-batch-size 16 --global-batch-size 256 --eval-iters 10

Override any Megatron argument after the positional args:
    torchrun --nproc_per_node=1 scripts/debug/load_model_from_checkpoint.py \
        --checkpoint-path results/.../checkpoints --num-layers 12
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import torch

# PyTorch 2.6+ flipped `torch.load` default to weights_only=True, which blocks
# unpickling arbitrary Python objects. Megatron's common.pt contains pickled
# objects (argparse.Namespace, enums like signal.Signals, etc.), so we force
# weights_only=False. Safe here: we trust our own training checkpoints.
_orig_torch_load = torch.load
def _torch_load_full(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_full

logging.getLogger("megatron.core.tensor_parallel.random").setLevel(logging.ERROR)


def detect_launcher():
    if os.environ.get("TORCHELASTIC_RUN_ID"):
        return "torchrun"
    elif os.environ.get("SLURM_PROCID") is not None:
        return "srun (bare)"
    elif os.environ.get("RANK") is not None and os.environ.get("SLURM_PROCID") is None:
        return "srun + container (env vars forwarded)"
    elif os.environ.get("MASTER_ADDR") is None and os.environ.get("SLURM_PROCID") is None:
        return "plain python (single process)"
    else:
        return "unknown"


if int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0))) == 0:
    print(f"Launcher: {detect_launcher()}")


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "submodules" / "Megatron-LM"))

NUM_GPUS = int(os.environ.get("WORLD_SIZE", 1))

# ---- Model presets (must match training config architecture) ----
PRESETS = {
    "qwen3-30b-a3b": {
        "--num-layers": "48",
        "--hidden-size": "2048",
        "--ffn-hidden-size": "6144",
        "--num-attention-heads": "32",
        "--num-query-groups": "4",
        "--kv-channels": "128",
        "--num-experts": "128",
        "--moe-ffn-hidden-size": "768",
        "--moe-router-topk": "8",
        "--max-position-embeddings": "40960",
        "--untie-embeddings-and-output-weights": True,
        "--make-vocab-size-divisible-by": "1187",
        "--add-bias-linear": False,
    },
    "small": {
        "--num-layers": "18",
        "--hidden-size": "768",
        "--ffn-hidden-size": "2048",
        "--num-attention-heads": "12",
        # num_query_groups and group_query_attention are auto-detected
        # from checkpoint via --use-checkpoint-args
        "--num-experts": "8",
        "--moe-ffn-hidden-size": "768",
        "--moe-router-topk": "2",
        "--max-position-embeddings": "4096",
        "--add-bias-linear": False,
    },
}


def parse_checkpoint_path(ckpt_path_str):
    """Parse checkpoint path into (load_dir, ckpt_step).

    Handles:
        .../checkpoints/iter_0114441  → (../checkpoints, 114441)
        .../checkpoints               → (../checkpoints, None)  [loads latest]
    """
    ckpt_path = Path(ckpt_path_str).resolve()

    iter_match = re.match(r"iter_0*(\d+)", ckpt_path.name)
    if iter_match:
        return str(ckpt_path.parent), int(iter_match.group(1))

    return str(ckpt_path), None


def infer_num_experts_from_path(ckpt_path_str):
    """Try to extract num_experts from the run directory name (nexp_XX pattern)."""
    match = re.search(r"nexp_(\d+)", ckpt_path_str)
    if match:
        return int(match.group(1))
    return None


# ---- Parse our own args before Megatron sees them ----
pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument("--checkpoint-path", type=str, required=True,
                        help="Path to checkpoint: .../checkpoints/iter_XXXXXXX or .../checkpoints")
pre_parser.add_argument("--preset", type=str, default="small",
                        choices=list(PRESETS.keys()),
                        help="Model preset (must match training architecture)")
pre_parser.add_argument("--seq-len", type=int, default=4096,
                        help="Sequence length for the model")
pre_parser.add_argument("--interactive", action="store_true", default=False,
                        help="Drop into interactive Python shell on rank 0 after loading")
pre_parser.add_argument("--evaluate", action="store_true", default=False,
                        help="Run validation loss after loading. Pass --valid-data-path, "
                             "--eval-iters, --micro-batch-size, --global-batch-size as Megatron args.")
pre_args, remaining_argv = pre_parser.parse_known_args()

preset = PRESETS[pre_args.preset]
SEQ_LEN = pre_args.seq_len

load_dir, ckpt_step = parse_checkpoint_path(pre_args.checkpoint_path)
inferred_nexp = infer_num_experts_from_path(pre_args.checkpoint_path)

if int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0))) == 0:
    print(f"Checkpoint dir:  {load_dir}")
    print(f"Checkpoint step: {ckpt_step if ckpt_step is not None else '(latest)'}")
    if inferred_nexp is not None:
        print(f"Inferred num_experts from path: {inferred_nexp}")

# Override preset num_experts if we can infer it from the path
if inferred_nexp is not None:
    preset["--num-experts"] = str(inferred_nexp)

# Set micro/global batch sizes: larger defaults when evaluating
if pre_args.evaluate:
    DEFAULT_MBS = 4
    DEFAULT_GBS = DEFAULT_MBS * NUM_GPUS
else:
    DEFAULT_MBS = 1
    DEFAULT_GBS = max(NUM_GPUS, 1)

# Build base sys.argv from preset
base_args = [
    "load_model_from_checkpoint",
    "--tensor-model-parallel-size", "1",
    "--pipeline-model-parallel-size", "1",
    "--expert-model-parallel-size", str(NUM_GPUS),
    # --use-checkpoint-args: force-overrides architecture args (GQA, normalization,
    # rotary, etc.) from the checkpoint before the model is built, so the model
    # automatically matches the checkpoint's architecture.
    "--use-checkpoint-args",
    "--auto-detect-ckpt-format",
    # "--group-query-attention",
    # "--qk-layernorm",
    "--normalization", "RMSNorm",
    "--norm-epsilon", "1e-06",
    "--swiglu",
    "--position-embedding-type", "rope",
    "--rotary-percent", "1.0",
    "--rotary-base", "10000",
    "--rotary-seq-len-interpolation-factor", "1",
    "--seq-length", str(SEQ_LEN),
    "--use-flash-attn",
    "--bf16",
    "--moe-router-load-balancing-type", "aux_loss",
    "--moe-aux-loss-coeff", "1e-3",
    "--moe-grouped-gemm",
    "--moe-token-dispatcher-type", "allgather",
    "--moe-router-dtype", "fp32",
    # Tokenizer
    "--tokenizer-type", "GPT2BPETokenizer",
    "--vocab-file", "/leonardo_work/OELLM_prod2026/models/EleutherAI/gpt-neox-20b/vocab.json",
    "--merge-file", "/leonardo_work/OELLM_prod2026/models/EleutherAI/gpt-neox-20b/merges.txt",
    "--legacy-tokenizer",
    # Required by Megatron but irrelevant for inspection
    "--micro-batch-size", str(DEFAULT_MBS),
    "--global-batch-size", str(DEFAULT_GBS),
    "--lr", "1e-4",
    "--train-iters", "1",
    "--no-bias-dropout-fusion",
    # Checkpoint loading
    "--load", load_dir,
    "--no-load-optim",
    "--no-load-rng",
    "--exit-on-missing-checkpoint",
]

if ckpt_step is not None:
    base_args.extend(["--ckpt-step", str(ckpt_step)])

if pre_args.evaluate:
    base_args.append("--skip-train")

# Apply preset values
for flag, value in preset.items():
    if isinstance(value, bool):
        if value:
            base_args.append(flag)
    else:
        base_args.extend([flag, str(value)])

# CLI overrides (passed through to Megatron)
base_args.extend(remaining_argv)

sys.argv = base_args

from functools import partial

from gpt_builders import gpt_builder
from model_provider import model_provider
from megatron.training import get_args, get_model, print_rank_0
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron


initialize_megatron(args_defaults={
    "no_load_rng": True,
    "no_load_optim": True,
    "micro_batch_size": 1,
    "enable_msc": False,
    "use_cpu_initialization": False,
    "standalone_embedding_stage": False,
})

args = get_args()

ddp_model = get_model(partial(model_provider, gpt_builder), wrap_with_ddp=False)

loaded_iteration, _ = load_checkpoint(ddp_model, None, None)
print_rank_0(f"Checkpoint loaded at iteration {loaded_iteration}")

model = ddp_model[0]
model.eval()




# ---- Inspect distributed setup ----
global_rank = torch.distributed.get_rank()
world_size  = torch.distributed.get_world_size()
local_rank  = int(os.environ.get("LOCAL_RANK", 0))
node_name   = os.environ.get("SLURMD_NODENAME")
gpu_device  = torch.cuda.current_device()

for r in range(world_size):
    if r == global_rank:
        print(f"Global rank: {global_rank}, World size: {world_size}, "
              f"Local rank: {local_rank}, Node name: {node_name}, GPU device: {gpu_device}")
    torch.distributed.barrier()


# ---- Inspect MoE expert placement ----
decoder = model.module.decoder
layer = decoder.layers[0]
moe = layer.mlp
router = moe.router
experts = moe.experts
rank = torch.distributed.get_rank()
node = os.environ.get("SLURMD_NODENAME", "unknown")

for r in range(world_size):
    if rank == r:
        for name, mod in model.named_modules():
            if hasattr(mod, "num_local_experts"):
                print(f"[Rank {rank} @ {node}] {name}")
                print(f"  EP rank:              {rank}")
                print(f"  Num local experts:    {mod.num_local_experts}")
                print(f"  Local expert indices: {mod.local_expert_indices}")
                break
    torch.distributed.barrier()


# ---- Parameter summary ----
local_params = sum(p.numel() for p in model.parameters())
local_expert_params = sum(p.numel() for n, p in model.named_parameters() if "experts" in n)
local_non_expert = local_params - local_expert_params

ep_size = args.expert_model_parallel_size
total_expert_params = local_expert_params * ep_size
total_params = local_non_expert + total_expert_params
active_params = local_non_expert + total_expert_params * (args.moe_router_topk / args.num_experts)

print_rank_0(f"\n{'='*60}")
print_rank_0(f"Model loaded from checkpoint")
print_rank_0(f"  Checkpoint: {load_dir}")
print_rank_0(f"  Step:       {loaded_iteration}")
print_rank_0(f"{'='*60}")
print_rank_0(f"  Layers:          {args.num_layers}")
print_rank_0(f"  Hidden size:     {args.hidden_size}")
print_rank_0(f"  Attention heads: {args.num_attention_heads} (KV groups: {args.num_query_groups})")
print_rank_0(f"  Experts:         {args.num_experts} (topk={args.moe_router_topk})")
print_rank_0(f"  Expert parallel: {ep_size} GPUs ({args.num_experts // ep_size} experts/GPU)")
print_rank_0(f"  MoE FFN hidden:  {args.moe_ffn_hidden_size}")
print_rank_0(f"  Vocab size:      {args.padded_vocab_size}")
print_rank_0(f"  Seq length:      {SEQ_LEN}")
print_rank_0(f"{'='*60}")
print_rank_0(f"  Total params:    {total_params:>14,}  ({total_params/1e9:.2f}B)")
print_rank_0(f"  Expert (total):  {total_expert_params:>14,}  ({total_expert_params/1e9:.2f}B)")
print_rank_0(f"  Non-expert:      {local_non_expert:>14,}  ({local_non_expert/1e9:.2f}B)")
print_rank_0(f"  Active params:   {active_params:>14,.0f}  ({active_params/1e9:.2f}B)")
print_rank_0(f"  Per-GPU params:  {local_params:>14,}  ({local_params/1e9:.2f}B)")
print_rank_0(f"{'='*60}")

if torch.cuda.is_available():
    alloc = torch.cuda.memory_allocated() / 1e9
    free, total_mem = torch.cuda.mem_get_info()
    print(f"  [Rank {rank}] GPU allocated: {alloc:.2f} GB, free: {free/1e9:.2f} / {total_mem/1e9:.2f} GB")

torch.distributed.barrier()
print_rank_0(f"{'='*60}")
print_rank_0(f"\nModel architecture:")
print_rank_0(str(model))
decoder = model.module.decoder
layer = decoder.layers[0]
moe = layer.mlp
router = moe.router
experts = moe.experts
w1 = experts.linear_fc1.weight0
print_rank_0(f"w1: {w1.shape}")
print_rank_0(f"w1: {w1[:5, :5]}")
w2 = experts.linear_fc2.weight0
print_rank_0(f"w2: {w2.shape}")
print_rank_0(f"w2: {w2[:5, :5]}")

# ---- Interactive mode ----
if pre_args.interactive:
    import code
    rank = torch.distributed.get_rank()
    if rank == 0:
        code.interact(local=dict(globals(), **locals()))
    torch.distributed.barrier()


# ---- Optional validation loss evaluation ----
if pre_args.evaluate:
    import math

    from pretrain_gpt import forward_step, train_valid_test_datasets_provider
    from megatron.training.training import build_train_valid_test_data_iterators, evaluate
    from megatron.core.utils import get_model_config

    train_valid_test_datasets_provider.is_distributed = True

    print_rank_0(f"\n{'='*60}")
    print_rank_0("Building validation data iterator ...")
    _, valid_data_iterator, _ = build_train_valid_test_data_iterators(
        train_valid_test_datasets_provider
    )

    if valid_data_iterator is None:
        print_rank_0("ERROR: no validation data iterator created. "
                     "Did you pass --valid-data-path?")
    else:
        config = get_model_config(ddp_model[0])
        total_loss_dict, _, timelimit = evaluate(
            forward_step, valid_data_iterator, ddp_model,
            None, config, verbose=True,
        )

        print_rank_0(f"\n{'='*60}")
        print_rank_0(f"Validation loss  (step {loaded_iteration}, "
                     f"{args.eval_iters} iters × gbs {args.global_batch_size} "
                     f"= {args.eval_iters * args.global_batch_size} samples):")
        for key in total_loss_dict:
            loss_val = total_loss_dict[key].item()
            ppl = math.exp(min(20, loss_val))
            print_rank_0(f"  {key}: {loss_val:.6f}  (PPL: {ppl:.2f})")
        print_rank_0(f"{'='*60}")

torch.distributed.barrier()
torch.distributed.destroy_process_group()
