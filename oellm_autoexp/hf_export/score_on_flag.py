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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--datamix", type=Path, required=True)
    p.add_argument("--num-docs", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--min-len", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bos-id", type=int, default=None, help="default: the tokenizer's / config's bos id")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(a.model_dir, trust_remote_code=True)
    bos = a.bos_id if a.bos_id is not None else getattr(cfg, "bos_token_id", None)
    if bos is None:
        from transformers import AutoTokenizer

        bos = AutoTokenizer.from_pretrained(a.model_dir).bos_token_id
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto").eval()
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
    print(json.dumps(res, indent=2))
    if a.out:
        a.out.write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
