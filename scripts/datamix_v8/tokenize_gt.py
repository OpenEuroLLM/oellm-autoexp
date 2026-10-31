"""datamix v8: tokenize raw sources (HF downloads, flag sample stages) into Megatron datasets with GigaToken.

GigaToken (github.com/marcelroed/gigatoken) gives exactly SentencePiece's ids for openeurollm/tokenizer-256k
(checked: 0 mismatches on ~7k real documents + edge cases) at ~10x the throughput of HF tokenizers.

Output convention = flag/: one document per sequence, ids + <eos> (2), no <bos>, int32; MMIDIDX .idx;
shards `<out>/<source>/megatron-lm/shard_NNNN_text_document.{bin,idx}` + `shard_NNNN.info.json`.

    # 1. encode: units (parquet row groups / jsonl byte ranges / whole compressed files) split over nodes;
    #    per unit <work>/<source>/<unit>.bin + .len.npy (atomic, resumable)
    python tokenize_gt.py encode --source NAME --part $SLURM_NODEID --nparts $SLURM_NNODES
    # 2. assemble: the units in order -> shards of <= --shard-tokens tokens (shards split over nodes)
    python tokenize_gt.py assemble --source NAME --part ... --nparts ...

Chat formatting (ChatML, as the LUMI vitiugin/data_tok chat sets): `<|im_start|>{role}\\n{content}<|im_end|>`
joined by "\\n"; assistant `reasoning_content` -> `<think>\\n...\\n</think>\\n\\n` before the content; tool calls
`<tool_call>\\n{"name", "arguments"}\\n</tool_call>`; tool results `<tool_response>\\n...\\n</tool_response>` in a
`tool` turn; tool definitions as a `<tools>` block in the system turn (Qwen style).
"""

from __future__ import annotations

import argparse
import glob
import gzip
import io
import json
import os
import struct
import sys
import time
from multiprocessing import Pool

import numpy as np

# JUPITER defaults; on MareNostrum 5: TOKGT_ROOT=<data>/v8_sources (hf/... below it), TOKGT_TOKENIZER=<tokenizer.model>
ROOT = os.environ.get("TOKGT_ROOT", "/e/scratch/e-ext-2025e02-108/poeppel1/v8")
TOKENIZER = os.environ.get("TOKGT_TOKENIZER", f"{ROOT}/tokenizer-256k/tokenizer.model")
EOS = 2
_HEADER = b"MMIDIDX\x00\x00"

# name -> (input glob(s) relative to ROOT, format)
SOURCES = {
    "Nemotron-Cascade-2-SFT-Data": (["hf/nvidia/Nemotron-Cascade-2-SFT-Data/*/*.jsonl"], "chat"),
    "Nemotron-SFT-Science-v2": (["hf/nvidia/Nemotron-SFT-Science-v2/*.jsonl"], "chat"),
    "Nemotron-SFT-Math-v4": (["hf/nvidia/Nemotron-SFT-Math-v4/data/*.parquet"], "chat"),
    "marin-open-thoughts-4-code-qwen3-32b-annotated": (
        ["hf/marin-community/open-thoughts-4-code-qwen3-32b-annotated/*.parquet"], "seed_gen"),
    "marin-open-thoughts-4-math-qwen3-32b-annotated": (
        ["hf/marin-community/open-thoughts-4-math-qwen3-32b-annotated/*.parquet"], "seed_gen"),
    "marin-open-thoughts-4-science-qwen3-32b-annotated": (
        ["hf/marin-community/open-thoughts-4-science-qwen3-32b-annotated/*.parquet"], "seed_gen"),
    "Nemotron-Pretraining-Specialized-v1.2": (["hf/nvidia/Nemotron-Pretraining-Specialized-v1.2/*/*.parquet"], "text"),
    # flag's propella-filtered sample stage (2,839B tokens); the flag release is a random 25 % of it
    "common-pile-stackv2-0.1-full": (["flag_src/common-pile-stackv2-0.1/sample/documents/*.jsonl.zst"], "text"),
    # v8.1: full HF versions of sets whose LUMI data_tok copies are partial (2026-10-31)
    "hermes-agent-reasoning-traces": (["hf/lambda/hermes-agent-reasoning-traces/data/kimi/*.parquet",   # the two configs;
                                       "hf/lambda/hermes-agent-reasoning-traces/data/glm-5.1/*.parquet"], "sharegpt"),  # data/train-* = old kimi copy
    "Themis-CodePreference": (["hf/project-themis/Themis-CodePreference/data/*.parquet"], ("pair", "input", "chosen")),
    "openthoughts-4-code-qwen3-32b-32k-annotated": (
        ["hf/laion/openthoughts-4-code-qwen3-32b-32k-annotated/data/*.parquet"], ("pair", "instruction_seed", "output")),
    "Nemotron-Cascade-SFT-Stage-1": (["hf/nvidia/Nemotron-Cascade-SFT-Stage-1/*.jsonl",
                                      "hf/nvidia/Nemotron-Cascade-SFT-Stage-1/*/*.jsonl"], "chat"),
    "Nemotron-Math-v2": (["hf/nvidia/Nemotron-Math-v2/data/*.parquet"], "chat"),   # the .jsonl files hold the same rows
}
SHAREGPT_ROLE = {"human": "user", "gpt": "assistant", "system": "system", "tool": "tool", "function_call": "assistant"}
JSONL_RANGE = 1 << 30            # byte range per unit of an uncompressed .jsonl
BATCH_CHARS = 32 << 20           # characters per encode call


# ---------------------------------------------------------------- formatting

def _text(c) -> str:
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):          # content parts
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(c)


def _json(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except ValueError:
            return x
    return x


def chatml(messages, tools=None) -> str:
    messages = _json(messages) or []
    tools = _json(tools) or []
    turns = []
    if tools:
        block = "<tools>\n" + "\n".join(json.dumps(t, ensure_ascii=False) for t in tools) + "\n</tools>"
        if messages and messages[0].get("role") == "system":
            messages = [dict(messages[0], content=_text(messages[0].get("content")) + "\n\n" + block)] + list(messages[1:])
        else:
            messages = [{"role": "system", "content": block}] + list(messages)
    for m in messages:
        role = m.get("role") or "user"
        content = _text(m.get("content"))
        if role == "assistant":
            rc = m.get("reasoning_content")
            if rc:
                content = f"<think>\n{rc.strip()}\n</think>\n\n{content.lstrip()}"
            for tc in _json(m.get("tool_calls")) or []:
                fn = tc.get("function", tc)
                call = {"name": fn.get("name"), "arguments": _json(fn.get("arguments"))}
                content += ("\n" if content else "") + "<tool_call>\n" + json.dumps(call, ensure_ascii=False) + "\n</tool_call>"
        elif role == "tool":
            content = f"<tool_response>\n{content}\n</tool_response>"
        turns.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(turns)


def sharegpt(conversations, tools=None) -> str:
    """ShareGPT turns ({from, value}); the values already carry <think> / <tool_call> / <tool_response> as text.
    Tool schemas go into the system turn only when it does not list them already."""
    msgs = [{"role": SHAREGPT_ROLE.get(m.get("from"), m.get("from") or "user"), "content": _text(m.get("value"))}
            for m in (_json(conversations) or [])]
    tools = _json(tools) or []
    if tools and not any(m["role"] == "system" and "<tools>" in m["content"] and "{" in m["content"] for m in msgs):
        block = "<tools>\n" + "\n".join(json.dumps(t, ensure_ascii=False) for t in tools) + "\n</tools>"
        if msgs and msgs[0]["role"] == "system":
            msgs[0] = dict(msgs[0], content=msgs[0]["content"] + "\n\n" + block)
        else:
            msgs = [{"role": "system", "content": block}] + msgs
    return "\n".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in msgs)


def fmt_doc(fmt, row: dict) -> str | None:
    if isinstance(fmt, tuple) and fmt[0] == "pair":   # ("pair", user column, assistant column)
        q, a = row.get(fmt[1]), row.get(fmt[2])
        if not q or not a:
            return None
        return chatml([{"role": "user", "content": _text(q)}, {"role": "assistant", "content": _text(a)}])
    if fmt == "sharegpt":
        return sharegpt(row.get("conversations"), row.get("tools")) or None
    if fmt == "text":
        return row.get("text")
    if fmt == "chat":
        return chatml(row.get("messages"), row.get("tools"))
    if fmt == "seed_gen":            # marin open-thoughts-4: Qwen3-32B output (starts with <think>)
        q, a = row.get("instruction_seed"), row.get("generated_text")
        if not q or not a:
            return None
        return chatml([{"role": "user", "content": q}, {"role": "assistant", "content": a}])
    raise ValueError(fmt)


# ---------------------------------------------------------------- units

def list_units(src: str):
    globs, _ = SOURCES[src]
    files = sorted(f for g in globs for f in glob.glob(os.path.join(ROOT, g)))
    assert files, f"no input files for {src}"
    units = []
    for f in files:
        if f.endswith(".parquet"):
            import pyarrow.parquet as pq
            n = pq.ParquetFile(f).num_row_groups
            units += [(f, "rg", i, i + 1) for i in range(n)]
        elif f.endswith(".jsonl"):
            size = os.path.getsize(f)
            units += [(f, "bytes", s, min(s + JSONL_RANGE, size)) for s in range(0, size, JSONL_RANGE)]
        else:                        # .jsonl.gz / .jsonl.zst: whole file
            units.append((f, "file", 0, 0))
    return units


def unit_name(i: int) -> str:
    return f"u{i:07d}"


def iter_rows(unit):
    f, kind, a, b = unit
    if kind == "rg":
        import pyarrow.parquet as pq
        t = pq.ParquetFile(f).read_row_group(a)
        yield from t.to_pylist()
    elif kind == "bytes":
        with open(f, "rb") as fh:
            if a > 0:                # a line starting before a belongs to the previous unit
                fh.seek(a - 1)
                if fh.read(1) != b"\n":
                    fh.readline()
            while fh.tell() < b:
                line = fh.readline()
                if not line:
                    break
                if line.strip():
                    yield json.loads(line)
    else:
        if f.endswith(".zst"):
            import zstandard
            fh = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(f, "rb"), read_size=1 << 24),
                                  encoding="utf-8")
        else:
            fh = gzip.open(f, "rt", encoding="utf-8")
        with fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


_TOK = None


def _tok():
    global _TOK
    if _TOK is None:
        import gigatoken as gt
        _TOK = gt.Tokenizer.from_sentencepiece(TOKENIZER)
    return _TOK


def encode_unit(args):
    src, i, unit, work = args
    out = os.path.join(work, src, unit_name(i))
    if os.path.exists(out + ".len.npy"):
        return i, 0, 0, 0.0, True
    fmt = SOURCES[src][1]
    t0 = time.time()
    tok = _tok()
    lengths, skipped, nchar = [], 0, 0
    with open(out + ".bin.tmp", "wb") as fb:
        batch, chars = [], 0

        def flush():
            import awkward as ak
            r = tok.encode_batch(batch, parallel=False)
            n = ak.to_numpy(ak.num(r)).astype(np.int64)
            flat = ak.to_numpy(ak.flatten(r)).astype(np.int32, copy=False)
            out = np.empty(len(flat) + len(n), np.int32)
            ends = np.cumsum(n + 1)                    # one <eos> after each document
            out[ends - 1] = EOS
            mask = np.ones(len(out), bool)
            mask[ends - 1] = False
            out[mask] = flat
            fb.write(out.tobytes())
            lengths.extend((n + 1).tolist())
        for row in iter_rows(unit):
            d = fmt_doc(fmt, row)
            if not d:
                skipped += 1
                continue
            batch.append(d)
            chars += len(d)
            if chars >= BATCH_CHARS:
                flush()
                nchar += chars
                batch, chars = [], 0
        if batch:
            flush()
            nchar += chars
    os.replace(out + ".bin.tmp", out + ".bin")
    np.save(out + ".len.tmp.npy", np.asarray(lengths, np.int32))
    os.replace(out + ".len.tmp.npy", out + ".len.npy")
    if skipped:
        json.dump({"skipped": skipped}, open(out + ".skipped.json", "w"))
    return i, len(lengths), nchar, time.time() - t0, False


def my_units(units, part, nparts, sizes):
    """Greedy size-balanced assignment of units to nodes (deterministic)."""
    order = sorted(range(len(units)), key=lambda k: -sizes[k])
    load = [0] * nparts
    owner = [0] * len(units)
    for k in order:
        p = min(range(nparts), key=lambda q: load[q])
        owner[k] = p
        load[p] += sizes[k]
    return [k for k in range(len(units)) if owner[k] == part]


def unit_size(u):
    f, kind, a, b = u
    if kind == "bytes":
        return b - a
    if kind == "rg":
        import pyarrow.parquet as pq
        md = pq.ParquetFile(f).metadata
        return md.row_group(a).total_byte_size
    return os.path.getsize(f)


def cmd_encode(a):
    units = list_units(a.source)
    sizes = [unit_size(u) for u in units]
    mine = my_units(units, a.part, a.nparts, sizes)
    os.makedirs(os.path.join(a.work, a.source), exist_ok=True)
    t0, done_docs, done_chars = time.time(), 0, 0
    print(f"[{a.source}] part {a.part}/{a.nparts}: {len(mine)} of {len(units)} units, "
          f"{sum(sizes[k] for k in mine) / 1e9:.1f} GB, {a.procs} procs", flush=True)
    with Pool(a.procs, maxtasksperchild=64) as pool:
        for i, nd, nc, dt, skip in pool.imap_unordered(encode_unit, [(a.source, k, units[k], a.work) for k in mine]):
            done_docs += nd
            done_chars += nc
            if not skip:
                el = time.time() - t0
                print(f"  unit {i} docs {nd} {dt:.0f}s | total {done_docs} docs {done_chars / 1e9:.1f} GChar "
                      f"{done_chars / 1e6 / el:.0f} MChar/s", flush=True)
    print(f"[{a.source}] part {a.part} encoded in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- assemble

def write_index(path, lengths):
    lengths = np.asarray(lengths, np.int32)
    pointers = np.zeros(len(lengths), np.int64)
    if len(lengths) > 1:
        np.cumsum(lengths[:-1].astype(np.int64) * 4, out=pointers[1:])
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(_HEADER)
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<B", 4))                  # int32
        f.write(struct.pack("<Q", len(lengths)))
        f.write(struct.pack("<Q", len(lengths) + 1))
        f.write(lengths.tobytes())
        f.write(pointers.tobytes())
        f.write(np.arange(len(lengths) + 1, dtype=np.int64).tobytes())
    os.replace(tmp, path)


def plan_shards(a):
    units = list_units(a.source)
    d = os.path.join(a.work, a.source)
    missing = [i for i in range(len(units)) if not os.path.exists(os.path.join(d, unit_name(i) + ".len.npy"))]
    assert not missing, f"{len(missing)} units not encoded yet, e.g. {missing[:5]}"
    toks = [int(np.load(os.path.join(d, unit_name(i) + ".len.npy")).astype(np.int64).sum()) for i in range(len(units))]
    n_shards = max(1, -(-sum(toks) // a.shard_tokens))
    target = sum(toks) / n_shards
    shards, cur, acc = [], [], 0
    for i, t in enumerate(toks):
        cur.append(i)
        acc += t
        if acc >= target * (len(shards) + 1) and len(shards) < n_shards - 1:
            shards.append(cur)
            cur = []
    if cur:
        shards.append(cur)
    return units, toks, shards


def cmd_assemble(a):
    units, toks, shards = plan_shards(a)
    d = os.path.join(a.work, a.source)
    out = os.path.join(a.out, a.source, "megatron-lm")
    os.makedirs(out, exist_ok=True)
    print(f"[{a.source}] {sum(toks) / 1e9:.2f}B tokens, {len(units)} units -> {len(shards)} shards", flush=True)
    for s, ids in enumerate(shards):
        if s % a.nparts != a.part:
            continue
        prefix = os.path.join(out, f"shard_{s:04d}_text_document")
        info_path = os.path.join(out, f"shard_{s:04d}.info.json")
        if os.path.exists(info_path):
            print(f"  shard {s}: done", flush=True)
            continue
        t0 = time.time()
        lengths = np.concatenate([np.load(os.path.join(d, unit_name(i) + ".len.npy")) for i in ids])
        with open(prefix + ".bin.tmp", "wb") as fo:
            for i in ids:
                with open(os.path.join(d, unit_name(i) + ".bin"), "rb") as fi:
                    while True:
                        buf = fi.read(1 << 28)
                        if not buf:
                            break
                        fo.write(buf)
        size = os.path.getsize(prefix + ".bin.tmp")
        assert size == 4 * int(lengths.astype(np.int64).sum()), (size, lengths.sum())
        os.replace(prefix + ".bin.tmp", prefix + ".bin")
        write_index(prefix + ".idx", lengths)
        skipped = sum(json.load(open(p))["skipped"] for i in ids
                      for p in [os.path.join(d, unit_name(i) + ".skipped.json")] if os.path.exists(p))
        info = {"idx": {"size": os.path.getsize(prefix + ".idx"), "dtype": "int32", "sequence_count": len(lengths),
                        "document_count": len(lengths) + 1, "total_tokens": int(lengths.astype(np.int64).sum())},
                "bin": {"size": size},
                "tokenization": {"tool": "gigatoken", "tokenizer": "openeurollm/tokenizer-256k", "eos": True,
                                 "bos": False, "format": SOURCES[a.source][1], "inputs": SOURCES[a.source][0],
                                 "units": [ids[0], ids[-1]], "skipped_documents": skipped,
                                 "date": time.strftime("%Y-%m-%d")}}
        json.dump(info, open(info_path, "w"), indent=1)
        print(f"  shard {s}: {len(lengths)} docs {info['idx']['total_tokens'] / 1e9:.2f}B tokens "
              f"{time.time() - t0:.0f}s", flush=True)


def cmd_units(a):
    units = list_units(a.source)
    print(a.source, len(units), "units", f"{sum(unit_size(u) for u in units) / 1e9:.1f} GB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["units", "encode", "assemble"])
    ap.add_argument("--source", required=True, help="a SOURCES name, or several comma-separated (run one after another)")
    ap.add_argument("--part", type=int, default=int(os.environ.get("SLURM_NODEID", 0)))
    ap.add_argument("--nparts", type=int, default=int(os.environ.get("SLURM_NNODES", 1)))
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 8))
    ap.add_argument("--work", default=f"{ROOT}/tok_work")
    ap.add_argument("--out", default=f"{ROOT}/tok")
    ap.add_argument("--shard-tokens", type=int, default=50_000_000_000)
    a = ap.parse_args()
    names = [x for x in a.source.split(",") if x]
    bad = [x for x in names if x not in SOURCES]
    assert not bad, f"unknown sources {bad}; known: {sorted(SOURCES)}"
    for x in names:
        a.source = x
        {"units": cmd_units, "encode": cmd_encode, "assemble": cmd_assemble}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
