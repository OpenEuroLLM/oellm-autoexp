"""Score an exported HF model on documents of the training mix (a sanity check of a conversion).

Documents are drawn from the datamix (shards by weight, documents uniformly), each one a separate
sequence with BOS prepended and cut at ``--seq-len`` -- what the training run saw with document
masking and ``dataloader_add_bos``. A correct conversion lands near the run's training loss at the
checkpoint's step; a broken one far above it (~10 for a scrambled layout). The model is spread over
all visible GPUs (``device_map="auto"``), so a 35-91B model fits one node.

    python -m oellm_autoexp.hf_export.score_on_flag --model-dir <hf> \\
        --datamix $HYBRID_ENV/data/flag_decon_v1_datamix_mn5.txt --num-docs 64 --seq-len 4096
"""

from __future__ import annotations

import argparse
import json
import random
import struct
import time
from pathlib import Path

import numpy as np
import torch

_DTYPES = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: np.float64, 7: np.float32, 8: np.uint16}


def read_index(prefix: str):
    with open(prefix + ".idx", "rb") as f:
        assert f.read(9) == b"MMIDIDX\x00\x00", prefix
        struct.unpack("<Q", f.read(8))
        (code,) = struct.unpack("<B", f.read(1))
        (n,) = struct.unpack("<Q", f.read(8))
        struct.unpack("<Q", f.read(8))
        lens = np.frombuffer(f.read(4 * n), dtype=np.int32)
        ptrs = np.frombuffer(f.read(8 * n), dtype=np.int64)
    return np.dtype(_DTYPES[code]), lens, ptrs


def sample_docs(datamix: Path, num_docs: int, min_len: int, seed: int) -> list[np.ndarray]:
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
        docs.append(np.asarray(data[start : start + lens[j]], dtype=np.int64))
    return docs


def layer_device_map(cfg, gib: float) -> dict[str, int]:
    """Whole decoder layers filled onto the GPUs in order, ``gib`` of weights each (the last GPU 12
    GiB less: it holds the logits). accelerate's own map reserves the largest layer (an Engram
    table, ~41 GiB) on every GPU and offloads the rest to the CPU; offloaded weights only arrive
    through a module's forward, which the mHC read / write (methods) never call."""
    from transformers import AutoModelForCausalLM

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, dtype=torch.bfloat16)
    size = lambda m: sum(t.numel() * t.element_size() for t in [*m.parameters(), *m.buffers()])  # noqa: E731
    n = torch.cuda.device_count()
    cap = [gib * 2**30 - (12 * 2**30 if i == n - 1 else 0) for i in range(n)]
    used, gpu, dmap = [0.0] * n, 0, {}
    tail = size(model.model.norm) + (0 if cfg.tie_word_embeddings else size(model.lm_head))
    cap[n - 1] -= tail
    dmap["model.embed_tokens"] = 0
    used[0] += size(model.model.embed_tokens)
    for i, layer in enumerate(model.model.layers):
        b = size(layer)
        while used[gpu] + b > cap[gpu]:
            gpu += 1
            if gpu == n:
                raise RuntimeError(f"layer {i}: the model does not fit {n} x {gib} GiB")
        dmap[f"model.layers.{i}"] = gpu
        used[gpu] += b
    dmap["model.norm"] = n - 1
    if not cfg.tie_word_embeddings:
        dmap["lm_head"] = n - 1
    print("weights per GPU (GiB):", [round(u / 2**30, 1) for u in used], flush=True)
    return dmap


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--datamix", type=Path, required=True)
    p.add_argument("--num-docs", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--min-len", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bos-id", type=int, default=None, help="default: the tokenizer's / config's bos id")
    p.add_argument("--gen-check", type=int, default=16, help="greedy tokens to compare cached decoding against a full forward (0: off)")
    p.add_argument("--gpu-mem-gib", type=float, default=None,
                   help="weights per GPU, whole layers in order (else device_map auto); the last GPU 12 GiB less")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(a.model_dir, trust_remote_code=True)
    bos = a.bos_id if a.bos_id is not None else getattr(cfg, "bos_token_id", None)
    if bos is None:
        from transformers import AutoTokenizer

        bos = AutoTokenizer.from_pretrained(a.model_dir).bos_token_id
    t0 = time.time()
    dmap = layer_device_map(cfg, a.gpu_mem_gib) if a.gpu_mem_gib else "auto"
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, trust_remote_code=True, dtype=torch.bfloat16,
                                                 device_map=dmap).eval()
    print(f"loaded {a.model_dir.name} in {time.time() - t0:.0f}s; bos {bos}", flush=True)
    docs = sample_docs(a.datamix, a.num_docs, a.min_len, a.seed)
    dev = next(model.parameters()).device
    nll = cnt = 0.0
    per_doc = []
    for d in docs:
        ids = torch.tensor(np.concatenate([[bos], d[: a.seq_len - 1]]), device=dev).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_ids=ids).logits.float()
        lp = torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:].to(logits.device), reduction="sum").item()
        n = ids.shape[1] - 1
        nll += lp
        cnt += n
        per_doc.append(lp / n)
    res = {"model": str(a.model_dir), "docs": len(docs), "tokens": int(cnt), "mean_nll": nll / cnt,
           "doc_nll_median": float(np.median(per_doc)), "seq_len": a.seq_len}
    if a.gen_check:  # the cache path (KV / recurrent state / mHC streams) must reproduce the forward
        prompt = torch.tensor(np.concatenate([[bos], docs[0][:255]]), device=dev).unsqueeze(0)
        with torch.no_grad():
            out = model.generate(input_ids=prompt, max_new_tokens=a.gen_check, do_sample=False,
                                 output_logits=True, return_dict_in_generate=True)
            gen = out.sequences
            cached = torch.stack([x[0].float().cpu() for x in out.logits]).log_softmax(-1)
            full = model(input_ids=gen).logits[0, prompt.shape[1] - 1 : -1].float().cpu().log_softmax(-1)
        new = gen[0, prompt.shape[1]:].cpu()
        top2 = full.topk(2, dim=-1).values
        res["gen_check_match"] = float((new == full.argmax(-1)).float().mean())
        # bf16 near-ties flip the argmax; a broken cache shows up as large log-prob differences
        res["gen_check_max_logprob_diff"] = float((cached - full).abs().max())
        res["gen_check_full_top1_top2_gap_at_mismatch"] = [
            float(top2[i, 0] - top2[i, 1]) for i in range(len(new)) if new[i] != full[i].argmax()]
    print(json.dumps(res, indent=2))
    if a.out:
        a.out.write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
