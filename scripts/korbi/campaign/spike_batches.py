#!/usr/bin/env python3
"""What was in the batches of grad-norm spike steps? Offline, CPU only -- no model, no training change.

Reconstructs the global batch of given training steps exactly as the run's dataloader built it, from the
shared data cache: Megatron's own MegatronPretrainingReferenceLayoutSampler (the run's reference layout) for
the step's sample indices, the BlendedDataset's dataset_index / dataset_sample_index for the component, the
component GPTDataset's shuffle / sample / document indices and its IndexedDataset (with the add_bos virtual
BOS, as gpt_dataset._get_document) for the tokens. Then per sample: source, documents, and repetition
features (longest run of one token, unique-token fraction, most frequent token's share, fraction of
4-grams / 16-grams that already occurred earlier in the sample), contrasted between spike and calm steps.

Step indexing: a logged iteration N trained on the samples from consumed (N - 1) x global batch
(the log line of iteration N reports consumed N x global batch).

    python scripts/korbi/campaign/spike_batches.py --blend-hash b2ac5f22bb793c504e870e109893983d \
        --steps-csv steps.csv --out <dir> [--workers 64] [--decode 12]

steps.csv: columns step, label (spike / calm), grad_norm. Writes samples.parquet (one row per sample),
steps.csv (per-step aggregates), contrast.txt (spike vs calm), outliers.txt (decoded extreme samples).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

CACHE = "/gpfs/scratch/ehpc533/hybrid_nucleus/data_cache_shared"

_G = {}  # per-worker state: blend description, indices, lazily opened components


def _init(cache, blend_hash, bos):
    from megatron.core.datasets.indexed_dataset import IndexedDataset

    desc = json.load(open(f"{cache}/{blend_hash}-BlendedDataset-train-description.txt"))
    _G.update(
        cache=cache, bos=bos, desc=desc, comps={}, IndexedDataset=IndexedDataset,
        dsi=np.load(f"{cache}/{blend_hash}-BlendedDataset-train-dataset_index.npy", mmap_mode="r"),
        dssi=np.load(f"{cache}/{blend_hash}-BlendedDataset-train-dataset_sample_index.npy", mmap_mode="r"),
    )


def _comp(c):
    if c not in _G["comps"]:
        d = _G["desc"]["datasets"][c]
        h = hashlib.md5(json.dumps(d, indent=4).encode("utf-8")).hexdigest()
        p = f"{_G['cache']}/{h}-GPTDataset-train-"
        _G["comps"][c] = dict(
            shuffle=np.load(p + "shuffle_index.npy", mmap_mode="r"),
            sample=np.load(p + "sample_index.npy", mmap_mode="r"),
            doc=np.load(p + "document_index.npy", mmap_mode="r"),
            ds=_G["IndexedDataset"](d["dataset_path"], mmap=True),
            add_bos=bool(d.get("add_bos", False)), seq=int(d["sequence_length"]),
        )
    return _G["comps"][c]


def _get_document(cp, doc, offset=0, length=None):
    """gpt_dataset._get_document: with add_bos the virtual document is [BOS] + tokens."""
    ds, bos = cp["ds"], _G["bos"]
    if not cp["add_bos"]:
        return ds.get(doc, offset=offset, length=length)
    if offset > 0:
        return ds.get(doc, offset=offset - 1, length=length)
    if length is not None and length <= 1:
        return np.array([bos], dtype=np.int64)[: max(length, 0)]
    inner = ds.get(doc, offset=0, length=None if length is None else length - 1)
    return np.concatenate([np.array([bos], dtype=inner.dtype), inner])


def sample_tokens(s):
    """Tokens (seq + 1), component and document ids of blended sample s (gpt_dataset._query_...)."""
    c, k = int(_G["dsi"][s]), int(_G["dssi"][s])
    cp = _comp(c)
    i = int(cp["shuffle"][k])
    b, bo = (int(x) for x in cp["sample"][i])
    e, eo = (int(x) for x in cp["sample"][i + 1])
    parts, docs = [], []
    if b == e:
        docs.append(int(cp["doc"][b]))
        parts.append(_get_document(cp, docs[-1], offset=bo, length=eo - bo + 1))
    else:
        for j in range(b, e + 1):
            docs.append(int(cp["doc"][j]))
            parts.append(_get_document(cp, docs[-1], offset=0 if j > b else bo,
                                       length=None if j < e else eo + 1))
    return np.concatenate(parts).astype(np.int64), c, docs, [len(p) for p in parts]


def _repeat_frac(t, n):
    """Fraction of n-gram positions whose n-gram already occurred earlier in t."""
    if len(t) < n + 1:
        return 0.0
    h = np.zeros(len(t) - n + 1, dtype=np.uint64)
    for j in range(n):
        h = h * np.uint64(1000003) + t[j : len(t) - n + 1 + j].astype(np.uint64)
    return 1.0 - len(np.unique(h)) / len(h)


def features(args):
    step, s = args
    t, c, docs, lens = sample_tokens(s)
    change = np.flatnonzero(np.diff(t) != 0)
    runs = np.diff(np.concatenate([[-1], change, [len(t) - 1]]))
    _, counts = np.unique(t, return_counts=True)
    return dict(step=step, sample=int(s), comp=c, n_docs=len(docs), max_doc_len=max(lens),
                max_run=int(runs.max()), uniq_frac=len(counts) / len(t), top1_frac=counts.max() / len(t),
                rep4=_repeat_frac(t, 4), rep16=_repeat_frac(t, 16), first_doc=docs[0])


def step_samples(step, total, gbs, ref_dp, ref_mbs):
    from megatron.training.datasets.data_samplers import MegatronPretrainingReferenceLayoutSampler

    sampler = MegatronPretrainingReferenceLayoutSampler(
        None, total_samples=total, consumed_samples=(step - 1) * gbs, micro_batch_size=gbs,
        global_batch_size=gbs, data_parallel_rank=0, data_parallel_size=1, data_sharding=True,
        reference_dp_size=ref_dp, reference_micro_batch_size=ref_mbs)
    return next(iter(sampler))


def source_of(desc, c):
    p = desc["datasets"][c]["dataset_path"]
    rel = p.split("/s1/")[-1] if "/s1/" in p else p
    return "/".join(rel.split("/")[:2])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blend-hash", required=True)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--steps-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gbs", type=int, default=8192)
    ap.add_argument("--ref-dp", type=int, default=512)
    ap.add_argument("--ref-mbs", type=int, default=8)
    ap.add_argument("--tokenizer", default="/gpfs/projects/ehpc533/users/korbip/data/tokenizers/tokenizer-256k-agent")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--decode", type=int, default=12, help="decoded outlier samples")
    ap.add_argument("--max-samples", type=int, default=0, help="smoke test: first N samples per step")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    bos = tok.bos_token_id
    desc = json.load(open(f"{a.cache}/{a.blend_hash}-BlendedDataset-train-description.txt"))
    total = int(desc["size"])
    steps = pd.read_csv(a.steps_csv)
    print(f"blend {a.blend_hash}: {total} samples, {len(desc['datasets'])} datasets; bos {bos}; "
          f"{len(steps)} steps ({(steps.label == 'spike').sum()} spike)", flush=True)

    jobs = []
    for st in steps.step:
        ids = step_samples(int(st), total, a.gbs, a.ref_dp, a.ref_mbs)
        assert len(ids) == a.gbs, (st, len(ids))
        jobs += [(int(st), s) for s in (ids[: a.max_samples] if a.max_samples else ids)]
    print(f"{len(jobs)} samples to read", flush=True)
    with Pool(a.workers, initializer=_init, initargs=(a.cache, a.blend_hash, bos)) as pool:
        rows = pool.map(features, jobs, chunksize=64)
    df = pd.DataFrame(rows).merge(steps, on="step")
    df["source"] = [source_of(desc, c) for c in df.comp]
    df.to_parquet(os.path.join(a.out, "samples.parquet"))

    feats = ["n_docs", "max_doc_len", "max_run", "uniq_frac", "top1_frac", "rep4", "rep16"]
    agg = df.groupby(["step", "label", "grad_norm"]).agg(
        **{f"{f}_mean": (f, "mean") for f in feats}, **{f"{f}_max": (f, "max") for f in feats},
        n_run512=("max_run", lambda x: int((x >= 512).sum())), n_rep16_90=("rep16", lambda x: int((x >= 0.9).sum())),
        n_uniq02=("uniq_frac", lambda x: int((x <= 0.02).sum()))).reset_index()
    agg.to_csv(os.path.join(a.out, "steps.csv"), index=False)

    with open(os.path.join(a.out, "contrast.txt"), "w") as fh:
        def w(x=""):
            print(x, flush=True)
            fh.write(x + "\n")

        w("per-step aggregates, mean over steps (spike vs calm):")
        cols = [c for c in agg.columns if c not in ("step", "label", "grad_norm")]
        m = agg.groupby("label")[cols].mean().T
        w(m.to_string(float_format=lambda v: f"{v:.4g}"))
        w("\nSpearman correlation of per-step aggregates with grad_norm (all steps):")
        w(agg[cols + ["grad_norm"]].corr(method="spearman")["grad_norm"].drop("grad_norm")
          .sort_values().to_string(float_format=lambda v: f"{v:+.3f}"))
        w("\nsource share (% of samples), spike vs calm, largest differences:")
        sh = (df.groupby(["label", "source"]).size() / df.groupby("label").size() * 100).unstack(0).fillna(0)
        sh["diff"] = sh.get("spike", 0) - sh.get("calm", 0)
        w(sh.reindex(sh["diff"].abs().sort_values(ascending=False).index).head(15)
          .to_string(float_format=lambda v: f"{v:.2f}"))

    # decode the most repetitive samples of spike steps and of calm steps for comparison
    with open(os.path.join(a.out, "outliers.txt"), "w") as fh:
        _init(a.cache, a.blend_hash, bos)
        for label in ("spike", "calm"):
            sub = df[df.label == label].sort_values(["rep16", "max_run"], ascending=False).head(a.decode)
            for _, r in sub.iterrows():
                t, c, docs, lens = sample_tokens(int(r["sample"]))
                fh.write(f"=== {label} step {r.step} (grad-norm {r.grad_norm:.2f}) sample {r['sample']} "
                         f"{r.source} docs {docs[:4]} rep16 {r.rep16:.3f} max_run {r.max_run} "
                         f"uniq {r.uniq_frac:.4f}\n")
                fh.write(tok.decode(t[:600].tolist()).replace("\n", "\\n")[:1500] + "\n\n")
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
