"""Megatron args of the mhc_to_ldm end-to-end check on one 4-GPU node of the MPI cluster.

Base: Korbinian's rendered nucleus_engram_bg_e2e_mn5 (v4b's recipe on a small stack, mock data), with
what the MPI cluster lacks swapped out (MoonEP -> alltoall, the HF tokenizer -> NullTokenizer, no FP8
param gather, no async / persistent-worker saves, no wandb), and a stack that has every layer kind
v4b has: attention, MoE, Engram, CKDA and one V layer (sharing layer 0's KV).
    python gen_args.py rendered_args.txt <out dir> <e2e root on the cluster>
Then, from a copy at the e2e root with mhc_to_ldm.py and dcp_surgery.py next to these scripts (HTCondor, H100 bid
35): phaseA (mHC trained 20 steps and saved; the LDM config's checkpoint layout), phaseB (the conversions, the
layout check, six resumed arms), phaseC (the negative control). Their output: results_phaseB.txt, results_phaseC.txt.
"""
import shlex
import sys
from pathlib import Path

DROP = (
    "--async-save", "--ckda-branch-", "--data-cache-path", "--dataloader-add-bos", "--engram-token-map",
    "--exit-interval", "--exit-on-missing-checkpoint", "--exit-signal", "--fsdp-fp8-param-gather",
    "--hybrid-layer-pattern", "--kv-share-", "--moe-flex-", "--moe-token-dispatcher-type",
    "--non-persistent-", "--num-layers", "--profile-", "--save", "--tensorboard-dir", "--tokenizer-",
    "--train-iters", "--use-persistent-ckpt-worker", "--wandb-", "--mhc", "--v-block-learnable-gate",
    "--residual-mixer-lr-mult", "--depth-memory",
)
COMMON = [
    "--hybrid-layer-pattern", "*ENLEVE*ELE", "--num-layers", "11",
    "--kv-share-layers", "5", "--kv-share-source-layers", "0",
    "--tokenizer-type", "NullTokenizer", "--vocab-size", "262143",
    "--moe-token-dispatcher-type", "alltoall", "--train-iters", "30",
]
MHC = ["--mhc", "--v-block-learnable-gate", "--residual-mixer-lr-mult", "0.25"]
LDM = ["--depth-memory", "--depth-memory-impl", "cute", "--depth-memory-unwind",
       "--depth-memory-groups", "16", "--depth-memory-key-dim", "8", "--depth-memory-hidden-dim", "128",
       "--depth-memory-storage-dtype", "float16", "--depth-memory-checkpoint-interval", "4",
       "--residual-mixer-lr-mult", "1.0"]


def parse(path):
    args = []
    for line in Path(path).read_text().splitlines():
        tokens = shlex.split(line)
        if not tokens:
            continue
        args.append(tokens)
    return args


def main():
    rendered, out, root = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    kept = [t for t in parse(rendered) if not t[0].startswith(DROP)]
    base = [x for t in kept for x in t] + COMMON
    arms = {
        # trains the mHC model from scratch and saves iter 20
        "ref": MHC + ["--save", f"{root}/runs/ref/ckpt", "--save-interval", "20", "--exit-interval", "20"],
        # the LDM model's checkpoint layout (keys, optimizer states, dtypes): 2 steps, saved
        "ldm_tpl": LDM + ["--save", f"{root}/runs/ldm_tpl/ckpt", "--save-interval", "2", "--exit-interval", "2"],
        # resumed arms, 5 steps each
        "ctrl": MHC + ["--load", f"{root}/runs/ref/ckpt", "--exit-interval", "25"],
        "collapse": ["--load", f"{root}/surgery/ref_collapse/ckpt", "--exit-interval", "25"],
        "ldm": LDM + ["--load", f"{root}/surgery/ref/ckpt", "--exit-interval", "25"],
        # the same from the identity-mixer checkpoint: mHC there is exactly a plain residual, so
        # ctrl_id, collapse_id and ldm_id must agree at iteration 21
        "ctrl_id": MHC + ["--load", f"{root}/identity/ckpt", "--exit-interval", "25"],
        "collapse_id": ["--load", f"{root}/surgery/identity_collapse/ckpt", "--exit-interval", "25"],
        "ldm_id": LDM + ["--load", f"{root}/surgery/identity/ckpt", "--exit-interval", "25"],
    }
    out.mkdir(parents=True, exist_ok=True)
    for arm, extra in arms.items():
        tb = ["--tensorboard-dir", f"{root}/runs/{arm}/tb", "--data-cache-path", f"{root}/data_cache"]
        (out / f"{arm}.args").write_text(shlex.join(base + extra + tb) + "\n")
    print(f"{len(arms)} arms, {len(base)} common tokens -> {out}")


if __name__ == "__main__":
    main()
