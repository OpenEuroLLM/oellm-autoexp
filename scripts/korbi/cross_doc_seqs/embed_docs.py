#!/usr/bin/env python3
"""Step 2 of the cross-document packing pipeline (docs/cross_doc_seqs/README.md): one embedding per
document that the extracted steps read (the documents of non-solo slots; a solo slot is one long
document filling a whole sequence and is never repacked).

    srun -n <gpus> embed_docs.py --out <pipeline dir> --name q3e256 \
        --model /gpfs/projects/ehpc533/users/korbip/models/Qwen3-Embedding-0.6B --pooling last \
        --tokenizer /gpfs/projects/ehpc533/users/korbip/data/tokenizers/tokenizer-256k-agent \
        --dim 256 --max-src-tokens 640 --max-len 512

The text of a document is its first --max-src-tokens training tokens (the .bin, no virtual BOS),
decoded with the training tokenizer and re-tokenized for the embedding model (truncated at
--max-len). One process per GPU (RANK / WORLD_SIZE / LOCAL_RANK, else SLURM_PROCID / ...); process r embeds keys
r, r + world, ... of every stream chunk and writes <out>/emb/<name>/ck<a>_<b>/part_<r>_of_<world>.npz
{keys: int64 dataset << 40 | document, vec: float16 [n, dim]} atomically. Existing parts are skipped,
so the job can be restarted / extended (new stream chunks) at any time.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cds_data import IndexedTokens, load_stream  # noqa: E402

KEY_SHIFT = 40


def chunk_complete(emb_chunk_dir: str) -> bool:
    """A chunk is embedded when every part of SOME world size exists (part_<r>_of_<W>, r < W): a job restarted
    with another number of GPUs keeps the chunks an earlier job finished."""
    worlds: dict = {}
    for f in glob.glob(os.path.join(emb_chunk_dir, "part_*_of_*.npz")):
        if f.endswith(".tmp.npz"):
            continue
        r, w = os.path.basename(f)[5:-4].split("_of_")
        worlds.setdefault(int(w), set()).add(int(r))
    return any(len(rs) == w for w, rs in worlds.items())


def claim(dst: str, timeout_s: float) -> bool:
    """Take the work unit whose output is dst: an atomic mkdir of dst.claim (a claim nobody touched for
    timeout_s -- a killed job's -- is taken over). False if done or held by another process."""
    import shutil
    import socket

    lock = dst + ".claim"
    for _ in range(2):
        if os.path.exists(dst):
            return False
        try:
            os.mkdir(lock)
            open(os.path.join(lock, "owner"), "w").write(f"{socket.gethostname()} {os.getpid()} {time.ctime()}\n")
            return True
        except FileExistsError:
            try:
                age = time.time() - os.stat(lock).st_mtime
            except FileNotFoundError:
                continue
            if age < timeout_s:
                return False
            shutil.rmtree(lock, ignore_errors=True)  # stale: retry once
    return False


def release(dst: str) -> None:
    import shutil

    shutil.rmtree(dst + ".claim", ignore_errors=True)


def emb_status(out: str, name: str) -> str:
    """Which stream chunks (step ranges) are embedded: the contiguous complete ranges, partial chunks, totals."""
    root = os.path.join(out, "emb", name)
    chunks = []
    for p in sorted(glob.glob(os.path.join(out, "stream", "ck*_*"))):
        b = os.path.basename(p)
        if ".tmp" in b:
            continue
        c0, c1 = (int(x) for x in b[2:].split("_"))
        d = os.path.join(root, b)
        done = chunk_complete(d)
        parts = len(glob.glob(os.path.join(d, "part_*_of_*.npz"))) if os.path.isdir(d) else 0
        chunks.append((c0, c1, done, parts))
    ranges, cur = [], None
    for c0, c1, done, _ in chunks:
        if done and cur and cur[1] == c0:
            cur[1] = c1
        elif done:
            cur = [c0, c1]
            ranges.append(cur)
    n_done = sum(1 for c in chunks if c[2])
    partial = [f"ck{c0}_{c1} ({parts} parts)" for c0, c1, done, parts in chunks if not done and parts]
    lines = [f"{time.strftime('%Y-%m-%d %H:%M:%S')} embeddings {name}: {n_done} of {len(chunks)} stream chunks complete",
             "embedded step ranges (steps from checkpoints): " + (", ".join(f"{a_}-{b_}" for a_, b_ in ranges) or "none"),
             f"in progress: {len(partial)} chunks" + (": " + ", ".join(partial[:12]) + (" ..." if len(partial) > 12 else "")
                                                     if partial else "")]
    return "\n".join(lines) + "\n"


def write_status(out: str, name: str) -> None:
    text = emb_status(out, name)
    path = os.path.join(out, "emb", name, "STATUS.txt")
    tmp = f"{path}.{os.getpid()}"
    open(tmp, "w").write(text)
    os.replace(tmp, path)


def chunk_keys(stream_path: str) -> np.ndarray:
    """Sorted unique document keys of the chunk's non-solo slots."""
    arr, _ = load_stream(stream_path)
    nseg = np.diff(np.asarray(arr["seg_ptr"]))
    seg_slot = np.repeat(np.arange(len(nseg)), nseg)
    keep = nseg[seg_slot] > 1
    ds = np.asarray(arr["seg_ds"])[keep].astype(np.int64)
    doc = np.asarray(arr["seg_doc"])[keep].astype(np.int64)
    return np.unique((ds << KEY_SHIFT) | doc)


class DocTexts:
    """Batches of (keys, model inputs), built in DataLoader workers. A batch arrives with every
    document's (dataset, byte offset, token count) resolved in the parent, so a worker does one read
    per document, all of a batch's reads in flight at once (GPFS random reads are latency-bound)."""

    def __init__(self, prefixes, dtypes, a):
        self.prefixes, self.dtypes, self.a = prefixes, dtypes, a
        self._tok = None
        self._fds: dict = {}
        self._pool = None

    def _init(self):
        # (re)initialize per process: a DataLoader worker forked from a process that already used this
        # object inherits a thread pool without threads (its reads would wait forever) and shared fds
        if getattr(self, "_pid", None) != os.getpid():
            self._tok, self._fds, self._pool, self._pid = None, {}, None, os.getpid()
        if self._tok is None:
            from concurrent.futures import ThreadPoolExecutor

            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.a.tokenizer)
            self._model_tok = AutoTokenizer.from_pretrained(self.a.model)
            # does the tokenizer append its own end token (Qwen3-Embedding: <|endoftext|>, the state
            # last-token pooling reads)? Then nothing is appended here.
            mt = self._model_tok
            self._adds_end = len(mt("a")["input_ids"]) > len(mt("a", add_special_tokens=False)["input_ids"])
            self._pool = ThreadPoolExecutor(self.a.read_threads)

    def _read(self, ds: int, off: int, n: int) -> list:
        fd = self._fds.get(ds)
        if fd is None:
            fd = self._fds[ds] = os.open(self.prefixes[ds] + ".bin", os.O_RDONLY)
        dt = self.dtypes[ds]
        return np.frombuffer(os.pread(fd, n * dt.itemsize, off), dtype=dt).tolist()

    def batch(self, item):
        self._init()
        keys, ds, off, ntok = item
        a = self.a
        toks = list(self._pool.map(self._read, ds.tolist(), off.tolist(), ntok.tolist()))
        texts = self._tok.batch_decode(toks, skip_special_tokens=True)
        texts = [a.prefix + t for t in texts]
        mt = self._model_tok
        enc = mt(texts, add_special_tokens=True, truncation=True, max_length=a.max_len - 1)["input_ids"]
        eos = mt.eos_token_id
        if a.pooling == "last" and not self._adds_end and eos is not None:  # pool on an end token
            enc = [e + [eos] for e in enc]
        if a.pack:  # padding-free: documents concatenated, varlen flash attention, positions restart
            lens = np.array([len(e) for e in enc], np.int64)
            ids = np.fromiter((t for e in enc for t in e), dtype=np.int64, count=int(lens.sum()))
            pos = np.concatenate([np.arange(n) for n in lens])
            cu = np.r_[0, np.cumsum(lens)].astype(np.int32)
            return {"keys": keys, "ids": ids, "pos": pos, "cu": cu, "maxlen": int(lens.max())}
        L = max(1, max(len(e) for e in enc))
        pad = mt.pad_token_id if mt.pad_token_id is not None else (eos or 0)
        ids = np.full((len(enc), L), pad, dtype=np.int64)
        mask = np.zeros((len(enc), L), dtype=np.int64)
        for i, e in enumerate(enc):
            if a.pooling == "last":  # left padding: the last position is every row's last token
                ids[i, L - len(e):] = e
                mask[i, L - len(e):] = 1
            else:
                ids[i, :len(e)] = e
                mask[i, :len(e)] = 1
        return {"keys": keys, "ids": ids, "mask": mask}


def embed_batch(model, b, a, dev):
    """(vectors [n, dim] float32 normalized, model tokens) of one batch from DocTexts.batch."""
    import torch

    if a.engine == "vllm":  # model = vllm.LLM; the loader's token ids as prompts (same inputs as HF)
        from vllm.inputs import TokensPrompt

        cu = b["cu"]
        ids = b["ids"].tolist()
        prompts = [TokensPrompt(prompt_token_ids=ids[cu[i]:cu[i + 1]]) for i in range(len(cu) - 1)]
        outs = model.embed(prompts, use_tqdm=False)
        v = torch.tensor([o.outputs.embedding[: a.dim] for o in outs], dtype=torch.float32)
        return torch.nn.functional.normalize(v, dim=-1), int(cu[-1])
    if "cu" in b:
        ids = torch.as_tensor(b["ids"], device=dev)[None]
        pos = torch.as_tensor(b["pos"], device=dev)[None]
        cu = torch.as_tensor(b["cu"], device=dev, dtype=torch.int32)
        m = int(b["maxlen"])
        h = model(input_ids=ids, position_ids=pos, cu_seq_lens_q=cu, cu_seq_lens_k=cu,
                  max_length_q=m, max_length_k=m).last_hidden_state[0]
        if a.pooling == "last":
            v = h[cu[1:].long() - 1]
        else:
            lens = (cu[1:] - cu[:-1]).long()
            seg = torch.repeat_interleave(torch.arange(len(lens), device=dev), lens)
            v = torch.zeros(len(lens), h.shape[-1], device=dev, dtype=torch.float32).index_add_(0, seg, h.float())
            v = v / lens.clamp(min=1)[:, None]
        n_tok = int(b["cu"][-1])
    else:
        ids = torch.as_tensor(b["ids"], device=dev)
        mask = torch.as_tensor(b["mask"], device=dev)
        h = model(input_ids=ids, attention_mask=mask).last_hidden_state
        if a.pooling == "last":
            v = h[:, -1]
        else:
            mk = mask.unsqueeze(-1).to(h.dtype)
            v = (h * mk).sum(1) / mk.sum(1).clamp(min=1)
        n_tok = int(b["mask"].sum())
    return torch.nn.functional.normalize(v.float()[:, : a.dim], dim=-1), n_tok


class _Batches:
    """torch Dataset over precomputed batches (sorted by length, so padding stays small)."""

    def __init__(self, texts: DocTexts, batches):
        self.texts, self.batches = texts, batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        return self.texts.batch(self.batches[i])


def resolve(prefixes, keys: np.ndarray, max_src: int):
    """Per key: dataset, byte offset of the document in its .bin, tokens to read (sorted vectorized
    lookups of the .idx lengths / pointers: near-sequential page access)."""
    ds = (keys >> KEY_SHIFT).astype(np.int64)
    doc = keys & ((1 << KEY_SHIFT) - 1)
    off = np.empty(len(keys), np.int64)
    ntok = np.empty(len(keys), np.int64)
    dtypes = {}
    for d in np.unique(ds):
        m = np.nonzero(ds == d)[0]
        it = IndexedTokens(prefixes[int(d)])
        dtypes[int(d)] = it.dtype
        o = np.argsort(doc[m], kind="stable")
        idx = doc[m][o]
        ntok[m[o]] = np.minimum(np.asarray(it.lengths[idx]), max_src)
        off[m[o]] = np.asarray(it.pointers[idx])
    return ds, off, ntok, dtypes


def make_batches(keys, ds, off, ntok, a) -> list:
    """Batches of about --batch-tokens model tokens, by estimated length (source tokens)."""
    est = np.minimum(ntok * a.len_ratio + 8, a.max_len).astype(np.int64)
    order = np.argsort(est, kind="stable")
    batches, cur, cur_max, sum_est = [], [], 0, 0
    for i in order:
        m = max(cur_max, int(est[i]))
        # padded: the batch costs its longest document x its size; packed: the sum of the lengths
        cost = (sum_est + int(est[i])) if a.pack else m * (len(cur) + 1)
        if cur and cost > a.batch_tokens:
            c = np.asarray(cur)
            batches.append((keys[c], ds[c], off[c], ntok[c]))
            cur, m, sum_est = [], int(est[i]), 0
        cur.append(i)
        cur_max = m
        sum_est += int(est[i])
    if cur:
        c = np.asarray(cur)
        batches.append((keys[c], ds[c], off[c], ntok[c]))
    rng = np.random.default_rng(0)
    rng.shuffle(batches)  # mix lengths over time (steady throughput, honest progress estimates)
    return batches


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True, help="embedding run name (<out>/emb/<name>)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--pooling", choices=["last", "mean"], required=True,
                    help="last: Qwen3-Embedding (EOS state, left padding); mean: e5 / BERT-style")
    ap.add_argument("--prefix", default="", help='text prefix, e.g. "passage: " for e5')
    ap.add_argument("--tokenizer", required=True, help="the training tokenizer (to decode the .bin)")
    ap.add_argument("--dim", type=int, default=256, help="Matryoshka truncation (then re-normalized)")
    ap.add_argument("--max-src-tokens", type=int, default=640)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--len-ratio", type=float, default=1.0, help="model tokens per source token (batching estimate)")
    ap.add_argument("--batch-tokens", type=int, default=None,
                    help="model tokens per batch (default: 262144 packed, 65536 padded)")
    ap.add_argument("--pack", choices=["auto", "on", "off"], default="auto",
                    help="padding-free batches (varlen flash attention); auto = on with flash_attention_2")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (dynamic shapes)")
    ap.add_argument("--liger", action="store_true", help="Liger fused RMSNorm / RoPE / SwiGLU kernels (HF engine)")
    ap.add_argument("--check", type=int, default=0,
                    help="first embed this many docs both packed and padded and print their cosine")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--read-threads", type=int, default=32, help="concurrent .bin reads per loader worker")
    ap.add_argument("--attn", default="flash_attention_2",
                    help="attn_implementation (flash_attention_2: unpadded, no O(L^2) mask; falls back to sdpa)")
    ap.add_argument("--chunks", default=None, help="comma list of stream chunk names (default: all)")
    ap.add_argument("--profile", type=int, default=0,
                    help="bench: profile this many batches after 3 warm-up batches, print the top CUDA kernels")
    ap.add_argument("--engine", choices=["hf", "vllm"], default="hf",
                    help="hf: transformers forward; vllm: vLLM's pooling runner (run under setup_vllm_shared.sh)")
    ap.add_argument("--vllm-mem", type=float, default=0.85, help="vLLM gpu_memory_utilization")
    ap.add_argument("--vllm-batch-tokens", type=int, default=131072, help="vLLM max_num_batched_tokens")
    ap.add_argument("--vllm-max-seqs", type=int, default=1024, help="vLLM max_num_seqs")
    ap.add_argument("--check-against", default=None,
                    help="bench: cosine of the bench vectors vs an existing embedding run's for the same documents")
    ap.add_argument("--claim-slices", type=int, default=0,
                    help="work queue: every chunk in this many slices, each taken by whichever GPU claims it "
                         "first (any number of jobs can share a range); 0 = the classic rank / world split")
    ap.add_argument("--claim-timeout", type=float, default=3600.0, help="seconds after which a silent claim is stale")
    ap.add_argument("--chunk-range", default=None, help="only chunks inside checkpoints A:B")
    ap.add_argument("--chunk", type=int, default=100, help="steps per stream chunk (to know the range's chunks)")
    ap.add_argument("--status", action="store_true", help="only print (and write) the embedding status")
    ap.add_argument("--done", default=None, help="only check: exit 0 if every listed chunk (comma list) is embedded")
    ap.add_argument("--bench", type=int, default=0, help="embed only this many docs of the first chunk, print speed")
    a = ap.parse_args()
    if a.status:
        write_status(a.out, a.name)
        print(open(os.path.join(a.out, "emb", a.name, "STATUS.txt")).read(), end="")
        sys.exit(0)
    if a.done is not None:
        root = os.path.join(a.out, "emb", a.name)
        sys.exit(0 if all(chunk_complete(os.path.join(root, c)) for c in a.done.split(",") if c) else 1)

    import torch

    if a.engine == "hf":
        from transformers import AutoModel

    # RANK / WORLD_SIZE / LOCAL_RANK (run_pipeline.sh: one process per GPU), else one task per GPU
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    local = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    dev = torch.device("cuda", local % max(torch.cuda.device_count(), 1)) if torch.cuda.is_available() else torch.device("cpu")
    blend = json.load(open(os.path.join(a.out, "blend.json")))
    prefixes = blend["prefixes"]
    root = os.path.join(a.out, "emb", a.name)
    os.makedirs(root, exist_ok=True)
    if rank == 0:
        json.dump(vars(a), open(os.path.join(root, "config.json"), "w"), indent=1)
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    if a.engine == "vllm":
        from vllm import LLM

        # one GPU per process (run_pipeline.sh sets CUDA_VISIBLE_DEVICES); packed batches from the loader
        model = LLM(model=a.model, runner="pooling", dtype="bfloat16", max_model_len=a.max_len,
                    gpu_memory_utilization=a.vllm_mem, enable_prefix_caching=False, seed=0,
                    max_num_batched_tokens=a.vllm_batch_tokens, max_num_seqs=a.vllm_max_seqs)
        a.pack = True  # the batches carry per-document token lists (cu)
        detail = f"vLLM pooling runner, {a.vllm_batch_tokens} tokens per scheduler step"
    else:
        if a.liger:  # fused Triton RMSNorm / RoPE / SwiGLU: the unfused fp32 elementwise ops were ~75 % of GPU time
            from transformers import AutoConfig

            mtype = AutoConfig.from_pretrained(a.model).model_type
            import liger_kernel.transformers as lk

            patch = getattr(lk, f"apply_liger_kernel_to_{mtype}")
            patch(rope=True, rms_norm=True, swiglu=True, cross_entropy=False, fused_linear_cross_entropy=False)
        try:
            model = AutoModel.from_pretrained(a.model, dtype=dtype, attn_implementation=a.attn)
        except (ImportError, ValueError) as e:  # e.g. flash_attention_2 unavailable: masked SDPA
            print(f"rank {rank}: attn_implementation={a.attn} unavailable ({e}); using sdpa", flush=True)
            model = AutoModel.from_pretrained(a.model, dtype=dtype, attn_implementation="sdpa")
        model = model.to(dev).eval()
        flash = model.config._attn_implementation == "flash_attention_2"
        a.pack = (a.pack == "on") or (a.pack == "auto" and flash)
        assert not a.pack or flash, "--pack on needs flash_attention_2 (varlen kernels)"
        if a.compile:
            model = torch.compile(model, dynamic=True)
        detail = (f"attention {model.config._attn_implementation}, packed {a.pack}, compile {a.compile}, "
                  f"liger {a.liger}")
    if a.batch_tokens is None:
        a.batch_tokens = 262144 if a.pack else 65536
    if rank == 0:
        print(f"model {a.model}: {detail}, batch {a.batch_tokens} tokens", flush=True)

    while True:  # claim mode: rescan until every chunk of the range is embedded (by any job)
        remaining = embed_pass(a, model, dev, rank, world, prefixes, root)
        if not (a.claim_slices and a.chunk_range and not a.bench) or remaining == 0:
            break
        print(f"rank {rank}: {remaining} chunks of the range not complete yet (other jobs / extraction): waiting",
              flush=True)
        time.sleep(60)


def embed_pass(a, model, dev, rank, world, prefixes, root) -> int:
    """One pass over the work items; returns how many chunks of --chunk-range are not complete yet."""
    import torch

    chunks = sorted(p for p in glob.glob(os.path.join(a.out, "stream", "ck*_*")) if ".tmp" not in os.path.basename(p))
    if a.chunks:
        want = set(a.chunks.split(","))
        chunks = [c for c in chunks if os.path.basename(c) in want]
    if a.chunk_range:
        lo, hi = (int(x) for x in a.chunk_range.split(":"))
        chunks = [c for c in chunks if int(os.path.basename(c)[2:9]) >= lo and int(os.path.basename(c)[10:17]) <= hi]
    # work items: (chunk, first key, key stride, output part); classic: this rank's share of every chunk; claim:
    # every slice of every chunk, in chunk order (so the embedded step range grows from the start)
    if a.claim_slices and not a.bench:
        S = a.claim_slices
        items = [(c, sl, S, f"part_{sl:04d}_of_{S:04d}.npz") for c in chunks for sl in range(S)]
    else:
        items = [(c, rank, world, f"part_{rank:04d}_of_{world:04d}.npz") for c in chunks]
    for cpath, k0, kstep, part in items:
        name = os.path.basename(cpath)
        dst = os.path.join(root, name, part)
        if not a.bench and (os.path.exists(dst) or chunk_complete(os.path.dirname(dst))):
            continue
        if a.claim_slices and not a.bench:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not claim(dst, a.claim_timeout):
                continue
        keys = chunk_keys(cpath)[k0::kstep]
        if a.bench:
            keys = keys[: a.bench]
        t0 = time.time()
        t_res = time.time()
        ds_, off_, ntok_, dtypes = resolve(prefixes, keys, a.max_src_tokens)
        t_res = time.time() - t_res
        texts = DocTexts(prefixes, dtypes, a)
        batches = make_batches(keys, ds_, off_, ntok_, a)
        if a.check and a.pack and a.engine == "hf":
            n = min(a.check, len(keys))
            item = (keys[:n], ds_[:n], off_[:n], ntok_[:n])
            with torch.inference_mode():
                vp, _ = embed_batch(model, texts.batch(item), a, dev)
                a.pack = False
                pad_texts = DocTexts(prefixes, dtypes, a)
                vq = torch.cat([embed_batch(model, pad_texts.batch(tuple(x[j:j + 64] for x in item)), a, dev)[0]
                                for j in range(0, n, 64)])
                a.pack = True
            cos = (vp * vq).sum(-1)
            print(f"CHECK rank {rank}: packed vs padded over {n} docs: cosine min {float(cos.min()):.5f}, "
                  f"mean {float(cos.mean()):.5f}", flush=True)
        loader = torch.utils.data.DataLoader(_Batches(texts, batches), batch_size=None, shuffle=False,
                                             num_workers=a.workers, prefetch_factor=4 if a.workers else None,
                                             persistent_workers=False)
        out_k, out_v, ntok, done = [], [], 0, 0
        t_wait = t_gpu = 0.0
        tw = time.time()
        with torch.inference_mode():
            prof = None
            for i, b in enumerate(loader):
                t_wait += time.time() - tw
                tg = time.time()
                if a.profile and i == 3:
                    torch.cuda.synchronize()
                    prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                              torch.profiler.ProfilerActivity.CUDA])
                    prof.__enter__()
                    t_prof, tok_prof = time.time(), 0
                v, n_tok = embed_batch(model, b, a, dev)
                if prof is not None:
                    tok_prof += n_tok
                    if i == 3 + a.profile - 1:
                        torch.cuda.synchronize()
                        dt_p = time.time() - t_prof
                        prof.__exit__(None, None, None)
                        if rank == 0:
                            print(f"PROFILE {a.profile} batches, {tok_prof} tokens in {dt_p:.2f} s "
                                  f"({tok_prof / dt_p / 1e3:.0f}k tok/s)", flush=True)
                            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25,
                                                            max_name_column_width=70), flush=True)
                        return
                out_k.append(np.asarray(b["keys"]))
                out_v.append(v.to(torch.float16).cpu().numpy())
                ntok += n_tok
                done += len(b["keys"])
                t_gpu += time.time() - tg
                tw = time.time()
                if a.claim_slices and i % 50 == 0 and os.path.isdir(dst + ".claim"):
                    os.utime(dst + ".claim")  # heartbeat: the claim stays ours
                if rank == 0 and (i % 200 == 0 or i == len(batches) - 1):
                    dt = time.time() - t0
                    print(f"[{name}] rank 0: {done}/{len(keys)} docs, {done / dt:.0f} docs/s, "
                          f"{ntok / dt / 1e3:.0f}k tok/s", flush=True)
        k = np.concatenate(out_k) if out_k else np.zeros(0, np.int64)
        v = np.concatenate(out_v) if out_v else np.zeros((0, a.dim), np.float16)
        dt = time.time() - t0
        if a.bench and a.check_against:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from build_plan import load_embeddings

            rk, rv = load_embeddings(os.path.join(a.out, "emb", a.check_against), np.unique(k))
            r = np.searchsorted(rk, k)
            ok = (r < len(rk)) & (rk[np.minimum(r, max(len(rk) - 1, 0))] == k)
            if ok.any():
                cos = (v[ok].astype(np.float32) * rv[r[ok]].astype(np.float32)).sum(-1)
                print(f"CHECK rank {rank}: vs {a.check_against} over {int(ok.sum())} docs: cosine min "
                      f"{cos.min():.5f}, mean {cos.mean():.5f}", flush=True)
            else:
                print(f"CHECK rank {rank}: no document of the bench in {a.check_against}", flush=True)
        if a.bench:
            print(f"BENCH rank {rank}: {len(k)} docs in {dt:.1f} s = {len(k) / dt:.0f} docs/s, "
                  f"{ntok / dt / 1e3:.0f}k model tok/s, mean {ntok / max(len(k), 1):.0f} tok/doc; "
                  f"resolve {t_res:.1f} s, loader wait {t_wait:.1f} s, GPU {t_gpu:.1f} s", flush=True)
            return
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = f"{dst[:-4]}.{os.getpid()}.tmp.npz"
        np.savez(tmp, keys=k, vec=v)
        os.replace(tmp, dst)
        print(f"[{name}] {part} rank {rank}: {len(k)} docs, {dt:.0f} s ({len(k) / max(dt, 1e-9):.0f} docs/s)", flush=True)
        if a.claim_slices:
            release(dst)
            if chunk_complete(os.path.dirname(dst)):  # the last slice of a chunk: log it, refresh the status
                with open(os.path.join(root, "progress.log"), "a") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {name} complete ({os.uname().nodename})\n")
                write_status(a.out, a.name)
    if not a.chunk_range:
        return 0
    lo, hi = (int(x) for x in a.chunk_range.split(":"))
    expect = [f"ck{c0:07d}_{min(c0 + a.chunk, hi):07d}" for c0 in range(lo, hi, a.chunk)]
    return sum(1 for e in expect if not chunk_complete(os.path.join(root, e)))


if __name__ == "__main__":
    main()
    # every output is written (atomically) by now: skip the interpreter teardown, where vLLM's engine
    # threads abort the process ("terminate called without an active exception", exit 134) and a
    # finished GPU would count as failed; errors above still exit non-zero with their traceback
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
