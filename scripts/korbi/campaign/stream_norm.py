#!/usr/bin/env python3
"""Residual-stream norm across depth of a trained HybridModel checkpoint: the model is rebuilt
from the training job's own Megatron command (its rendered task script / sbatch), the final
checkpoint loaded (model only), and no-grad forwards run over fixed training-mix documents (the
first N documents of seed 0, as score_on_flag.py's sampler). Per decoder layer: the stream RMS at
its input, and the RMS of the layer's update relative to it. The stream is x (plain / nGPT), all n
streams (mHC), or slot 0 of LDM's [x, records...] stream. Run under torchrun on one node:

    torchrun --nproc-per-node 4 stream_norm.py --train-script <task.sh> --load <run>/ckpt \\
        --datamix <datamix.txt> --out <out.json> [--ep 4] [--num-docs 32] [--seq-len 4096] \\
        [--ckpt-step N]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shlex
import struct
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

_LAUNCHERS = ("pretrain_hybrid.py", "pretrain_gpt.py")
# data / logging / saving / schedule flags (the recompute flags stay: LDM's unwind validates them;
# recompute itself is inactive in eval mode)
_DROP_WITH_VALUE = {
    "--data-path", "--train-data-path", "--valid-data-path", "--test-data-path",
    "--data-args-path", "--per-split-data-args-path", "--data-cache-path", "--split",
    "--num-workers", "--dataloader-prefetch-factor", "--dataloader-type",
    "--num-dataset-builder-threads", "--save", "--save-interval", "--save-retain-interval",
    "--non-persistent-save-interval", "--non-persistent-ckpt-type",
    "--non-persistent-global-ckpt-dir", "--load", "--ckpt-step", "--tensorboard-dir",
    "--wandb-project", "--wandb-exp-name", "--wandb-save-dir", "--wandb-entity", "--log-interval",
    "--arch-metrics-interval", "--eval-interval", "--eval-iters", "--micro-batch-size",
    "--global-batch-size", "--profile-ranks", "--profile-step-start", "--profile-step-end",
    "--exit-signal", "--exit-duration-in-mins", "--exit-interval", "--manual-gc-interval",
    "--cuda-graph-impl", "--rerun-mode", "--expert-model-parallel-size", "--train-iters",
}
_DROP_FLAGS = {
    "--log-throughput", "--log-timers-to-tensorboard", "--log-progress", "--async-save",
    "--use-persistent-ckpt-worker", "--exit-signal-handler", "--manual-gc", "--no-mmap-bin-files",
    "--profile", "--use-pytorch-profiler", "--skip-train",
}
_DTYPES = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: np.float64,
           7: np.float32, 8: np.uint16}


def megatron_args(script: Path):
    text = script.read_text()
    launcher = next(n for n in _LAUNCHERS if n in text)
    tail = text[text.find(launcher) + len(launcher):].replace("\\\n", " ").split("\n", 1)[0]
    tokens = shlex.split(re.sub(r"'\s*$", "", tail))
    out, i = [], 0
    while i < len(tokens):
        if not tokens[i].startswith("--"):
            i += 1
            continue
        j = i + 1
        while j < len(tokens) and not tokens[j].startswith("--"):
            j += 1
        if tokens[i].split("=", 1)[0] not in _DROP_WITH_VALUE | _DROP_FLAGS:
            out.extend(tokens[i:j])
        i = j
    return launcher, out


def read_index(prefix):
    with open(prefix + ".idx", "rb") as f:
        assert f.read(9) == b"MMIDIDX\x00\x00", prefix
        struct.unpack("<Q", f.read(8))
        (code,) = struct.unpack("<B", f.read(1))
        (n,) = struct.unpack("<Q", f.read(8))
        struct.unpack("<Q", f.read(8))
        offset = f.tell()
    lens = np.memmap(prefix + ".idx", dtype=np.int32, mode="r", offset=offset, shape=(n,))
    ptrs = np.memmap(prefix + ".idx", dtype=np.int64, mode="r", offset=offset + 4 * n, shape=(n,))
    return np.dtype(_DTYPES[code]), lens, ptrs


def sample_docs(datamix, num_docs, min_len, seed):
    toks = Path(datamix).read_text().split()
    pairs = [(float(toks[i]), toks[i + 1]) for i in range(0, len(toks) - 1, 2)]
    rng = random.Random(seed)
    docs = []
    while len(docs) < num_docs:
        _, prefix = rng.choices(pairs, weights=[w for w, _ in pairs])[0]
        dtype, lens, ptrs = read_index(prefix)
        j = rng.randrange(len(lens))
        if lens[j] < min_len:
            continue
        data = np.memmap(prefix + ".bin", dtype=dtype, mode="r")
        start = ptrs[j] // dtype.itemsize
        docs.append(np.asarray(data[start: start + lens[j]], dtype=np.int64))
    return docs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-script", type=Path, required=True)
    p.add_argument("--load", required=True)
    p.add_argument("--datamix", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ep", type=int, default=4)
    p.add_argument("--num-docs", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--ckpt-step", type=int, default=None, help="iteration to load (default: latest)")
    a = p.parse_args()
    launcher, margs = megatron_args(a.train_script)
    world = int(os.environ["WORLD_SIZE"])
    margs += ["--load", a.load, "--no-load-optim", "--no-load-rng", "--exit-on-missing-checkpoint",
              "--micro-batch-size", "1", "--global-batch-size", str(world), "--eval-iters", "0",
              "--expert-model-parallel-size", str(a.ep), "--train-iters", "1"]
    if a.ckpt_step is not None:
        margs += ["--ckpt-step", str(a.ckpt_step)]
    sys.argv = [launcher] + margs
    from megatron.core.enums import ModelType
    from megatron.training import get_tokenizer
    from megatron.training.arguments import parse_and_validate_args
    from megatron.training.checkpointing import load_checkpoint
    from megatron.training.initialize import initialize_megatron
    from megatron.training.training import get_model

    from hybrid_builders import hybrid_builder as builder
    from pretrain_hybrid import model_provider

    parse_and_validate_args(args_defaults={"tokenizer_type": "GPT2BPETokenizer"})
    initialize_megatron(allow_no_cuda=False)
    model = get_model(partial(model_provider, builder), ModelType.encoder_or_decoder,
                      wrap_with_ddp=False)
    it = load_checkpoint(model, None, None)
    net = model[0]
    net.eval()
    hybrid = net.module if hasattr(net, "module") else net
    cfg = hybrid.config
    ldm = bool(getattr(cfg, "depth_memory", False))
    # depth_memory_unwind with the CuTe kernels: the record stream is slot first, [slots, s, b, h]
    # (depth_memory_state.StreamX); the stream form otherwise is [s, b, 2, h] (x, read)
    slot_first = ldm and bool(getattr(cfg, "depth_memory_unwind", False))
    kinds = list(hybrid.decoder.layer_type_list)
    n = len(kinds)
    dev = torch.cuda.current_device()
    acc = {k: torch.zeros(n, dtype=torch.float64, device=dev) for k in ("in_sq", "upd_sq", "cnt")}
    inputs = {}

    def stream(t):
        if t.dim() == 4:
            if slot_first:
                return t[0]
            return t[..., 0, :] if ldm else t
        return t

    def pre(i):
        def fn(_m, args, kwargs):
            x = kwargs.get("hidden_states", args[0] if args else None)
            if torch.is_tensor(x):
                inputs[i] = stream(x).detach().float().clone()
        return fn

    def post(i):
        def fn(_m, _inp, out):
            y = out[0] if isinstance(out, tuple) else out
            x = inputs.pop(i, None)
            if x is None or not torch.is_tensor(y):
                return None
            y = stream(y).detach().float()
            if y.shape != x.shape:
                return None
            acc["in_sq"][i] += x.square().mean(dim=-1).sum()
            acc["upd_sq"][i] += (y - x).square().mean(dim=-1).sum()
            acc["cnt"][i] += x.numel() // x.shape[-1]
            return None
        return fn

    for i, layer in enumerate(hybrid.decoder.layers):
        layer.register_forward_pre_hook(pre(i), with_kwargs=True)
        layer.register_forward_hook(post(i))
    bos = get_tokenizer().bos
    docs = sample_docs(a.datamix, a.num_docs, 256, 0)
    with torch.no_grad():
        for d in docs:
            ids = torch.tensor(np.concatenate([[bos], d[: a.seq_len - 1]]), device=dev)[None]
            pos = torch.arange(ids.shape[1], device=dev)[None]
            net(ids, pos, None)
    if dist.get_rank() == 0:
        cnt = acc["cnt"].clamp_min(1)
        in_rms = (acc["in_sq"] / cnt).sqrt().cpu().tolist()
        upd = (acc["upd_sq"] / acc["in_sq"].clamp_min(1e-30)).sqrt().cpu().tolist()
        a.out.write_text(json.dumps({"load": a.load, "iteration": it, "kinds": kinds,
                                     "stream_rms_in": in_rms, "update_over_stream": upd,
                                     "ldm": ldm, "mhc": bool(getattr(cfg, "mhc", False)),
                                     "ngpt": bool(getattr(cfg, "ngpt", False))}, indent=1))
        print("wrote", a.out, flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
