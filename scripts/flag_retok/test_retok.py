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
        PAD = 262144                                                       # an id outside the vocabulary
        enc = lambda t: np.array(sp.encode(t) + [2], np.int64)            # documents end with <eos>
        root = os.path.join(d, "flag")
        docs_a = [enc(t) for t in texts_a]
        docs_a.append(np.concatenate([enc("<|im_start|>user with pad")[:-1], [PAD, 2]]))   # skipped, kept as is
        write_dataset(os.path.join(root, "a/shard_0000_text_document"), docs_a)
        write_dataset(os.path.join(root, "b/shard_0000_text_document"), [enc(t) for t in texts_b])
        texts_d = ["<|im_start|>user\nq<|im_end|>", "plain d", "<think>t</think> ok", "more plain"]
        docs_d = [enc(t) for t in texts_d]
        dp = os.path.join(root, "d/x/shard_0000_text_document")
        os.makedirs(os.path.dirname(dp), exist_ok=True)
        np.concatenate([np.concatenate([[1], x]) for x in docs_d]).astype(np.int32).tofile(dp + ".bin")
        Ld = np.array([len(x) for x in docs_d], np.int32)
        Pd = np.concatenate([[0], np.cumsum((Ld[:-1].astype(np.int64) + 1) * 4)]) + 4
        write_index(dp + ".idx", 4, Ld, Pd, np.arange(len(docs_d) + 1))
        ep = os.path.join(root, "e/x/shard_0000_text_document")   # gap-indexed, outside --sources: linked as is
        os.makedirs(os.path.dirname(ep), exist_ok=True)
        docs_e = [enc("plain e"), enc("<think>mention</think> as text")]
        np.concatenate([np.concatenate([[1], x]) for x in docs_e]).astype(np.int32).tofile(ep + ".bin")
        Le = np.array([len(x) for x in docs_e], np.int32)
        write_index(ep + ".idx", 4, Le, np.concatenate([[0], np.cumsum((Le[:-1].astype(np.int64) + 1) * 4)]) + 4,
                    np.arange(len(docs_e) + 1))
        open(os.path.join(root, "datamix.txt"), "w").write("0.5 a/shard_0000_text_document\n0.5 b/shard_0000_text_document\n"
                                                           "0.1 d/x/shard_0000_text_document\n0.1 e/x/shard_0000_text_document\n")
        # c: a source outside the mix (openthoughts-3 style), added with --extra-rels
        write_dataset(os.path.join(root, "c/shard_0000_text_document"),
                      [enc("<|im_start|>user\nq<|im_end|>"), enc("plain c"), enc("<think>t</think> a")])
        extra = os.path.join(d, "extra.txt")
        open(extra, "w").write("0.25 c/shard_0000_text_document\n")
        # their removals (compare_pure docs.npz): docs 1 (also v2's) and 2 of shard a, doc 1 of c
        th_a = np.zeros(len(texts_a) + 1, bool); th_a[[1, 2]] = True
        np.savez(os.path.join(d, "theirs_a.npz"), theirs=th_a)
        th_c = np.zeros(3, bool); th_c[1] = True
        np.savez(os.path.join(d, "theirs_c.npz"), theirs=th_c)
        open(os.path.join(d, "shards.txt"), "w").write(f"{root}/a/shard_0000_text_document\n{root}/b/shard_0000_text_document\n"
                                                       f"{root}/c/shard_0000_text_document\n{root}/d/x/shard_0000_text_document\n"
                                                       f"{root}/e/x/shard_0000_text_document\n")
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
        decd = os.path.join(d, "flag-decon-v2", "d/x/shard_0000_text_document")
        os.makedirs(os.path.dirname(decd), exist_ok=True)
        os.symlink(dp + ".idx", decd + ".idx")
        dece = os.path.join(d, "flag-decon-v2", "e/x/shard_0000_text_document")
        os.makedirs(os.path.dirname(dece), exist_ok=True)
        os.symlink(ep + ".idx", dece + ".idx")
        decc = os.path.join(d, "flag-decon-v2", "c/shard_0000_text_document")    # v2 dropped nothing in c
        os.makedirs(os.path.dirname(decc), exist_ok=True)
        os.symlink(os.path.join(root, "c/shard_0000_text_document.idx"), decc + ".idx")

        run = [sys.executable, os.path.join(HERE, "retok.py")]
        common = ["--root", root, "--work", os.path.join(d, "work"), "--base-tok", base, "--agent-tok", agent,
                  "--part", "0", "--nparts", "1", "--procs", "2"]
        out = os.path.join(d, "flag-agenttok")
        os.environ["RETOK_UNIT_TOKENS"] = "20"     # several units per shard: re-encoded and copied ones interleave
        subprocess.run(run + ["find", "--shards", os.path.join(d, "shards.txt")] + common, check=True)
        X = ["--extra-rels", extra]
        subprocess.run(run + ["write", "--out", out, "--sources", "a,c,d/x"] + X + common, check=True)
        subprocess.run(run + ["write", "--out", out, "--sources", "a,c,d/x"] + X + common, check=True)  # resume: nothing left
        subprocess.run(run + ["assemble", "--out", out, "--sources", "a,c,d/x"] + X + common, check=True)
        # a crash between the parts' removal and info.json, then a rerun: nothing redone, info.json restored
        ia = os.path.join(out, "a/shard_0000.info.json")
        os.remove(ia)
        r = subprocess.run(run + ["assemble", "--out", out, "--sources", "a,c,d/x"] + X + common, check=True,
                           capture_output=True, text=True)
        assert "already assembled" in r.stdout and "retokenization" in json.load(open(ia))
        subprocess.run(run + ["write", "--out", out, "--sources", "a,c,d/x"] + X + common, check=True)   # no parts recreated
        assert not os.path.exists(os.path.join(out, "a/shard_0000_text_document.parts"))
        subprocess.run(run + ["decon", "--out", out, "--decon", os.path.join(d, "flag-decon-v2"),
                              "--decon-out", os.path.join(d, "flag-decon-v2-agenttok"),
                              "--extra-drop", f"a/shard_0000_text_document={d}/theirs_a.npz",
                              "--extra-drop", f"c/shard_0000_text_document={d}/theirs_c.npz"] + X + common, check=True)
        subprocess.run(run + ["verify", "--out", out, "--sources", "a,c,d/x"] + X + common, check=True)
        # a source outside --sources is linked even with delimiters
        out2 = os.path.join(d, "flag-agenttok-b")
        subprocess.run(run + ["write", "--out", out2, "--sources", "b"] + common, check=True)
        subprocess.run(run + ["assemble", "--out", out2, "--sources", "b"] + common, check=True)
        assert os.path.islink(os.path.join(out2, "a/shard_0000_text_document.bin"))

        a = os.path.join(out, "a/shard_0000_text_document")
        assert not os.path.islink(a + ".bin") and os.path.islink(os.path.join(out, "b/shard_0000_text_document.bin"))
        ix = Index(a + ".idx")
        docs = [np.fromfile(a + ".bin", np.int32, count=int(ix.lengths[i]), offset=int(ix.pointers[i])) for i in range(ix.seq_count)]
        assert docs[1][0] == ID["<|im_start|>"] and ID["<think>"] in docs[1] and ID["</think>"] in docs[1]
        assert ID["[INST]"] in docs[2] and ID["[/INST]"] in docs[2]
        assert docs[3].tolist() == enc(texts_a[3]).tolist()                 # no agent delimiter: untouched
        assert docs[0].tolist() == enc(texts_a[0]).tolist()
        for i, t in enumerate(texts_a):          # (doc 6: the <pad> one, checked above)
            assert sa.decode(docs[i][:-1].tolist()) == t and docs[i][-1] == 2
        assert len(docs[1]) < len(enc(texts_a[1]))
        info = json.load(open(a.removesuffix("_text_document") + ".info.json"))
        assert info["retokenization"]["sequences_changed"] == 2             # the <pad> sequence is not one
        assert info["retokenization"]["delimiter_counts"]["<|im_start|>"] == 3   # find counts the <pad> one too
        dv = Index(os.path.join(d, "flag-decon-v2-agenttok", "a/shard_0000_text_document.idx"))
        assert dv.doc_count - 1 == 5 and dv.lengths[1] == ix.lengths[3]       # docs 1 (v2 + theirs), 2 (theirs) dropped
        assert docs[6].tolist() == docs_a[6].tolist()                       # the <pad> sequence: unchanged
        assert os.path.realpath(os.path.join(d, "flag-decon-v2-agenttok", "a/shard_0000_text_document.bin")) == os.path.realpath(a + ".bin")
        ia_info = json.load(open(os.path.join(d, "flag-decon-v2-agenttok", "a/shard_0000.info.json")))
        assert ia_info["external_decontamination"]["removed_only_by_them"] == 1
        dc = Index(os.path.join(d, "flag-decon-v2-agenttok", "c/shard_0000_text_document.idx"))
        assert dc.doc_count - 1 == 2                                          # c: their doc 1 dropped
        c_new = os.path.join(out, "c/shard_0000_text_document")
        assert not os.path.islink(c_new + ".bin")                             # c re-tokenized
        for t in (out, os.path.join(d, "flag-decon-v2-agenttok")):
            assert "0.25 c/shard_0000_text_document" in open(os.path.join(t, "datamix.txt")).read()
        # d: gap-indexed source -> dense new .bin, re-encoded and copied documents right, the <bos> gaps dropped
        dn = os.path.join(out, "d/x/shard_0000_text_document")
        ixd = Index(dn + ".idx")
        assert not os.path.islink(dn + ".bin") and os.path.getsize(dn + ".bin") == 4 * int(ixd.lengths.sum())
        got = [np.fromfile(dn + ".bin", np.int32, count=int(ixd.lengths[i]), offset=int(ixd.pointers[i])) for i in range(4)]
        assert got[1].tolist() == docs_d[1].tolist() and got[3].tolist() == docs_d[3].tolist()
        assert got[0][0] == ID["<|im_start|>"] and ID["<think>"] in got[2]
        assert all(sa.decode(g[:-1].tolist()) == t and g[-1] == 2 for g, t in zip(got, texts_d))
        assert os.path.islink(os.path.join(out, "e/x/shard_0000_text_document.bin"))   # e: linked, verify passed
        # nothing under flag/ changed
        assert Index(os.path.join(root, "a/shard_0000_text_document.idx")).lengths.tolist() == [len(x) for x in docs_a]
    print("all passed")


if __name__ == "__main__":
    main()
