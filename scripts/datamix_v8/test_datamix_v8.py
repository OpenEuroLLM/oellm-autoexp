"""Tests of the datamix v8 pipeline on synthetic data (CPU).

    python scripts/datamix_v8/test_datamix_v8.py

- tokenize_gt (needs gigatoken, sentencepiece, pyarrow, zstandard, awkward; e.g. the JUPITER gtenv):
  a tiny SentencePiece model with the base tokenizer's reserved layout; three sources (chat jsonl split into
  small byte ranges, seed_gen parquet with several row groups and a row without output, text .jsonl.zst);
  encode over 2 parts -> assemble into several shards; every document == sp.encode(formatted) + [<eos>], in
  input order, each exactly once; .idx / info.json consistent; a rerun encodes nothing.
- build_raw_tree (needs torch for flag_decontam.common): links, <bos> stripping of a LUMI-style shard, datamix.txt.
Parts whose dependencies are missing are skipped (reported).
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "flag_retok"))
sys.path.insert(0, os.path.join(HERE, "..", "flag_decontam"))


def read_idx(path):
    import struct
    with open(path, "rb") as f:
        assert f.read(9) == b"MMIDIDX\x00\x00"
        assert struct.unpack("<Q", f.read(8))[0] == 1
        assert struct.unpack("<B", f.read(1))[0] == 4
        n = struct.unpack("<Q", f.read(8))[0]
        nd = struct.unpack("<Q", f.read(8))[0]
        lengths = np.frombuffer(f.read(4 * n), np.int32)
        pointers = np.frombuffer(f.read(8 * n), np.int64)
        doc_idx = np.frombuffer(f.read(8 * nd), np.int64)
    return lengths, pointers, doc_idx


def read_docs(prefix, dense=True):
    lengths, pointers, doc_idx = read_idx(prefix + ".idx")
    assert (doc_idx == np.arange(len(lengths) + 1)).all()
    mm = np.fromfile(prefix + ".bin", np.int32)
    if dense:                                            # .bin holds exactly the indexed tokens
        assert len(mm) == lengths.astype(np.int64).sum()
    return [mm[p // 4:p // 4 + n].tolist() for p, n in zip(pointers, lengths)]


def tiny_sp(d):
    import sentencepiece as spm
    named = ["<start_of_turn>", "<end_of_turn>", "<start_of_image>", "<end_of_image>", "<image_soft_token>",
             "<fim_prefix>", "<fim_middle>", "<fim_suffix>", "<tool_call>", "</tool_call>"]
    words = "the quick brown fox user assistant think tool answer code def return print hello world".split()
    rng = np.random.default_rng(0)
    corpus = "\n".join(" ".join(rng.choice(words, 12)) for _ in range(2000))
    buf = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(corpus.splitlines()), model_writer=buf, model_type="bpe", vocab_size=400, minloglevel=2,
        unk_id=0, bos_id=1, eos_id=2, pad_id=-1, byte_fallback=True, split_digits=True, add_dummy_prefix=False,
        remove_extra_whitespaces=False, normalization_rule_name="identity",
        user_defined_symbols=named + [f"<unused_{i}>" for i in range(100)], character_coverage=1.0)
    p = os.path.join(d, "tok.model")
    open(p, "wb").write(buf.getvalue())
    return p


def test_chatml():
    import tokenize_gt as T
    s = T.chatml([{"role": "system", "content": "sys"}, {"role": "user", "content": "q"},
                  {"role": "assistant", "content": None, "reasoning_content": " r ",
                   "tool_calls": [{"function": {"name": "f", "arguments": "{\"x\": 1}"}}]},
                  {"role": "tool", "content": "out"}, {"role": "assistant", "content": "a"}],
                 tools=[{"name": "f"}])
    assert s == ('<|im_start|>system\nsys\n\n<tools>\n{"name": "f"}\n</tools><|im_end|>\n'
                 '<|im_start|>user\nq<|im_end|>\n'
                 '<|im_start|>assistant\n<think>\nr\n</think>\n\n\n<tool_call>\n{"name": "f", "arguments": {"x": 1}}\n'
                 '</tool_call><|im_end|>\n'
                 '<|im_start|>tool\n<tool_response>\nout\n</tool_response><|im_end|>\n'
                 '<|im_start|>assistant\na<|im_end|>'), s
    assert T.fmt_doc("seed_gen", {"instruction_seed": "q", "generated_text": None}) is None
    assert T.chatml("[]") == ""
    print("chatml ok")


def test_tokenize():
    try:
        import gigatoken  # noqa: F401
        import pyarrow as pa
        import pyarrow.parquet as pq
        import sentencepiece as spm
        import zstandard
    except ImportError as e:
        print(f"tokenize_gt: SKIPPED ({e})")
        return
    import tokenize_gt as T
    with tempfile.TemporaryDirectory() as d:
        T.ROOT, T.TOKENIZER = d, tiny_sp(d)
        sp = spm.SentencePieceProcessor(model_file=T.TOKENIZER)
        rng = np.random.default_rng(1)
        words = ["hello", "world", "the", "fox", "Größe", "日本", "🤖", "x=1;", "\n", "  "]
        txt = lambda n: " ".join(rng.choice(words, n))
        os.makedirs(f"{d}/chat")
        chat_rows = [{"messages": [{"role": "user", "content": txt(5 + i % 7)},
                                   {"role": "assistant", "content": txt(9), "reasoning_content": txt(3) if i % 2 else None}],
                      "tools": [] if i % 3 else [{"name": "f"}]} for i in range(60)]
        with open(f"{d}/chat/a.jsonl", "w") as f:
            f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chat_rows))
        seed_rows = [{"instruction_seed": txt(4), "generated_text": (None if i == 7 else "<think>\n" + txt(6))}
                     for i in range(50)]
        pq.write_table(pa.Table.from_pylist(seed_rows), f"{d}/seed.parquet", row_group_size=8)
        text_rows = [{"text": txt(3 + i % 11), "id": i} for i in range(80)]
        with open(f"{d}/t.jsonl.zst", "wb") as f:
            f.write(zstandard.ZstdCompressor().compress("".join(json.dumps(r) + "\n" for r in text_rows).encode()))
        T.SOURCES = {"chat": (["chat/*.jsonl"], "chat"), "seed": (["seed.parquet"], "seed_gen"),
                     "text": (["t.jsonl.zst"], "text")}
        T.JSONL_RANGE = 333                              # many byte ranges, boundaries inside lines
        T.BATCH_CHARS = 200                              # several encode calls per unit
        expect = {"chat": [T.fmt_doc("chat", r) for r in chat_rows],
                  "seed": [T.fmt_doc("seed_gen", r) for r in seed_rows if r["generated_text"]],
                  "text": [r["text"] for r in text_rows]}
        assert len(T.list_units("chat")) > 10 and len(T.list_units("seed")) == 7
        for src in T.SOURCES:
            for part in range(2):
                a = type("A", (), dict(source=src, part=part, nparts=2, procs=2, work=f"{d}/work", out=f"{d}/out",
                                       shard_tokens=0))()
                T.cmd_encode(a)
            a.shard_tokens = -(-sum(len(sp.encode(x)) + 1 for x in expect[src]) // 3)   # -> exactly 3 shards
            for part in range(2):
                a.part = part
                T.cmd_assemble(a)
            shards = sorted(f[:-4] for f in os.listdir(f"{d}/out/{src}/megatron-lm") if f.endswith(".idx"))
            assert len(shards) == min(3, len(T.list_units(src))), (src, shards)   # units are never split
            docs = [x for s in shards for x in read_docs(f"{d}/out/{src}/megatron-lm/{s}")]
            want = [sp.encode(x) + [2] for x in expect[src]]
            assert docs == want, (src, len(docs), len(want))
            info = [json.load(open(f"{d}/out/{src}/megatron-lm/{s.removesuffix('_text_document')}.info.json"))
                    for s in shards]
            assert sum(i["idx"]["total_tokens"] for i in info) == sum(map(len, want))
            assert all(i["tokenization"]["bos"] is False for i in info)
            if src == "seed":
                assert sum(i["tokenization"]["skipped_documents"] for i in info) == 1
            r = T.encode_unit((src, 0, T.list_units(src)[0], f"{d}/work"))
            assert r[4] is True                          # resume: finished units are skipped
            print(f"tokenize_gt {src}: {len(docs)} docs in {len(shards)} shards ok")


def test_build_raw_tree():
    try:
        import build_raw_tree as B
        from common import write_index
    except ImportError as e:
        print(f"build_raw_tree: SKIPPED ({e})")
        return
    with tempfile.TemporaryDirectory() as d:
        B.SRC = f"{d}/src"

        def shard(prefix, docs):
            os.makedirs(os.path.dirname(prefix), exist_ok=True)
            lengths = np.array([len(x) for x in docs], np.int32)
            pointers = np.concatenate([[0], np.cumsum(lengths[:-1].astype(np.int64) * 4)])
            np.concatenate([np.array(x, np.int32) for x in docs]).tofile(prefix + ".bin")
            write_index(prefix + ".idx", 4, lengths, pointers, np.arange(len(docs) + 1))
        shard(f"{d}/src/jsc_mv1_subsets/math/megatron-lm/shard_0000_text_document", [[5, 2], [6, 7, 2]])
        json.dump({"idx": {}}, open(f"{d}/src/jsc_mv1_subsets/math/megatron-lm/shard_0000.info.json", "w"))
        shard(f"{d}/src/lumi_vtok/omni", [[1, 5, 6, 2], [1, 2], [1, 7, 2]])
        shard(f"{d}/src/jsc_tok/Hf-Set/megatron-lm/shard_0001_text_document", [[8, 2]])
        sys.argv = ["x", "--out", f"{d}/raw"]
        B.main()
        dm = open(f"{d}/raw/datamix.txt").read().split("\n")
        assert dm[:3] == ["1.0 hftok/Hf-Set/megatron-lm/shard_0001_text_document",
                          "1.0 mv1sub/math/megatron-lm/shard_0000_text_document",
                          "1.0 vtok/omni/megatron-lm/shard_0000_text_document"], dm
        assert read_docs(f"{d}/raw/vtok/omni/megatron-lm/shard_0000_text_document", dense=False) == [[5, 6, 2], [2], [7, 2]]
        assert os.path.islink(f"{d}/raw/vtok/omni/megatron-lm/shard_0000_text_document.bin")
        assert read_docs(f"{d}/raw/mv1sub/math/megatron-lm/shard_0000_text_document") == [[5, 2], [6, 7, 2]]
        assert os.path.islink(f"{d}/raw/mv1sub/math/megatron-lm/shard_0000.info.json")
        s = json.load(open(f"{d}/raw/vtok/omni/megatron-lm/shard_0000.info.json"))["bos_strip"]
        assert s["docs_with_bos"] == 3 and s["docs_without_bos"] == 0 and s["docs_ending_eos"] == 3
        sys.argv = ["x", "--out", f"{d}/raw", "--only", "vtok"]   # rerun of one part keeps the others listed
        B.main()
        assert len(open(f"{d}/raw/datamix.txt").read().split()) == 6
        print("build_raw_tree ok")


if __name__ == "__main__":
    test_chatml()
    test_tokenize()
    test_build_raw_tree()
    print("all run")
