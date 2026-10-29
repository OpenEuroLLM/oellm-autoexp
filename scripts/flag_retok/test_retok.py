"""End-to-end test of the delimiter re-tokenization on a synthetic Megatron dataset (CPU).

    python scripts/flag_retok/test_retok.py        # needs sentencepiece, numpy, torch (flag_decontam.common)

A tiny SentencePiece BPE model with the base tokenizer's reserved layout (<unk> <bos> <eos>, ids 3..12 named,
<unused_0..99> USER_DEFINED at 13..112, byte fallback) and its agent variant (make_tokenizer.rename_sp);
find -> write -> decon -> verify on two shards (one with delimiters, one without).
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "flag_decontam"))
from common import Index, write_index  # noqa: E402
from delimiters import AGENT_TOKENS  # noqa: E402
from make_tokenizer import rename_sp  # noqa: E402

NAMED = ["<start_of_turn>", "<end_of_turn>", "<start_of_image>", "<end_of_image>", "<image_soft_token>",
         "<fim_prefix>", "<fim_middle>", "<fim_suffix>", "<tool_call>", "</tool_call>"]
ID = {t: i for i, t in AGENT_TOKENS}


def tiny_models(d):
    import sentencepiece as spm
    words = ("the quick brown fox jumps over lazy dog user assistant think tool call answer code def return x "
             "print hello world im start end system message channel").split()
    rng = np.random.default_rng(0)
    corpus = "\n".join(" ".join(rng.choice(words, 12)) + " <b> a|b [x] (y)" for _ in range(3000))
    buf = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(corpus.splitlines()), model_writer=buf, model_type="bpe", vocab_size=600, minloglevel=2,
        unk_id=0, bos_id=1, eos_id=2, pad_id=-1, byte_fallback=True, split_digits=True,
        add_dummy_prefix=False, remove_extra_whitespaces=False, normalization_rule_name="identity",
        user_defined_symbols=NAMED + [f"<unused_{i}>" for i in range(100)], character_coverage=1.0)
    base = os.path.join(d, "base.model")
    open(base, "wb").write(buf.getvalue())
    agent = os.path.join(d, "agent.model")
    open(agent, "wb").write(rename_sp(buf.getvalue(), dict(AGENT_TOKENS)))
    return base, agent


def write_dataset(prefix, docs):
    lengths = np.array([len(x) for x in docs], np.int32)
    pointers = np.concatenate([[0], np.cumsum(lengths.astype(np.int64) * 4)[:-1]])
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    np.concatenate(docs).astype(np.int32).tofile(prefix + ".bin")
    write_index(prefix + ".idx", 4, lengths, pointers, np.arange(len(docs) + 1))
    json.dump({"idx": {"size": os.path.getsize(prefix + ".idx")}, "bin": {"size": os.path.getsize(prefix + ".bin")}},
              open(prefix.removesuffix("_text_document") + ".info.json", "w"))


def main():
    import sentencepiece as spm
    with tempfile.TemporaryDirectory() as d:
        base, agent = tiny_models(d)
        sp, sa = spm.SentencePieceProcessor(model_file=base), spm.SentencePieceProcessor(model_file=agent)
        texts_a = ["hello world the fox", "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>x</think> ok",
                   "code <b> a|b [INST] print [/INST] end", "plain <start_of_turn>user think", "no delimiters here",
                   "unknown token <|not_a_delim|> stays"]
        texts_b = ["the quick brown fox", "lazy dog jumps"]
        enc = lambda t: np.array(sp.encode(t) + [2], np.int64)            # documents end with <eos>
        root = os.path.join(d, "flag")
        write_dataset(os.path.join(root, "a/shard_0000_text_document"), [enc(t) for t in texts_a])
        write_dataset(os.path.join(root, "b/shard_0000_text_document"), [enc(t) for t in texts_b])
        open(os.path.join(root, "datamix.txt"), "w").write("0.5 a/shard_0000_text_document\n0.5 b/shard_0000_text_document\n")
        open(os.path.join(d, "shards.txt"), "w").write(f"{root}/a/shard_0000_text_document\n{root}/b/shard_0000_text_document\n")
        # v2-like decontamination tree: drop doc 1 of shard a
        dec = os.path.join(d, "flag-decon-v2", "a/shard_0000_text_document")
        os.makedirs(os.path.dirname(dec), exist_ok=True)
        np.savez(dec + ".removed.npz", doc=np.array([1]))
        idx = Index(os.path.join(root, "a/shard_0000_text_document.idx"))
        keep = np.ones(idx.doc_count - 1, bool); keep[1] = False
        from common import filter_index
        write_index(dec + ".idx", 4, *filter_index(idx, keep))
        json.dump({"idx": {}, "decontamination": {"dropped_documents": 1}},
                  open(dec.removesuffix("_text_document") + ".info.json", "w"))
        decb = os.path.join(d, "flag-decon-v2", "b/shard_0000_text_document")
        os.makedirs(os.path.dirname(decb), exist_ok=True)
        os.symlink(os.path.join(root, "b/shard_0000_text_document.idx"), decb + ".idx")

        run = [sys.executable, os.path.join(HERE, "retok.py")]
        common = ["--root", root, "--work", os.path.join(d, "work"), "--base-tok", base, "--agent-tok", agent,
                  "--part", "0", "--nparts", "1", "--procs", "2"]
        out = os.path.join(d, "flag-agenttok")
        os.environ["RETOK_UNIT_TOKENS"] = "20"     # several units per shard: re-encoded and copied ones interleave
        subprocess.run(run + ["find", "--shards", os.path.join(d, "shards.txt")] + common, check=True)
        subprocess.run(run + ["write", "--out", out, "--sources", "a"] + common, check=True)
        subprocess.run(run + ["decon", "--out", out, "--decon", os.path.join(d, "flag-decon-v2"),
                              "--decon-out", os.path.join(d, "flag-decon-v2-agenttok")] + common, check=True)
        subprocess.run(run + ["verify", "--out", out, "--sources", "a"] + common, check=True)
        # a source outside --sources is linked even with delimiters
        out2 = os.path.join(d, "flag-agenttok-b")
        subprocess.run(run + ["write", "--out", out2, "--sources", "b"] + common, check=True)
        assert os.path.islink(os.path.join(out2, "a/shard_0000_text_document.bin"))

        a = os.path.join(out, "a/shard_0000_text_document")
        assert not os.path.islink(a + ".bin") and os.path.islink(os.path.join(out, "b/shard_0000_text_document.bin"))
        ix = Index(a + ".idx")
        docs = [np.fromfile(a + ".bin", np.int32, count=int(ix.lengths[i]), offset=int(ix.pointers[i])) for i in range(ix.seq_count)]
        assert docs[1][0] == ID["<|im_start|>"] and ID["<think>"] in docs[1] and ID["</think>"] in docs[1]
        assert ID["[INST]"] in docs[2] and ID["[/INST]"] in docs[2]
        assert docs[3].tolist() == enc(texts_a[3]).tolist()                 # no agent delimiter: untouched
        assert docs[0].tolist() == enc(texts_a[0]).tolist()
        for i, t in enumerate(texts_a):
            assert sa.decode(docs[i][:-1].tolist()) == t and docs[i][-1] == 2
        assert len(docs[1]) < len(enc(texts_a[1]))
        info = json.load(open(a.removesuffix("_text_document") + ".info.json"))
        assert info["retokenization"]["sequences_reencoded"] == 2
        assert info["retokenization"]["delimiter_counts"]["<|im_start|>"] == 2
        dv = Index(os.path.join(d, "flag-decon-v2-agenttok", "a/shard_0000_text_document.idx"))
        assert dv.doc_count - 1 == 5 and dv.lengths[1] == ix.lengths[2]       # doc 1 dropped, the rest shifted
        assert os.path.realpath(os.path.join(d, "flag-decon-v2-agenttok", "a/shard_0000_text_document.bin")) == os.path.realpath(a + ".bin")
        # nothing under flag/ changed
        assert Index(os.path.join(root, "a/shard_0000_text_document.idx")).lengths.tolist() == [len(enc(t)) for t in texts_a]
    print("all passed")


if __name__ == "__main__":
    main()
