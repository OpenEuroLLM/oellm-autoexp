"""Tests of retok_v2.py on synthetic shards (tokenizer dirs from TOK2_OLD / TOK2_NEW; needs gigatoken +
sentencepiece): every document of a decontaminated (gap-indexed) view re-encoded 1:1 -- the same documents,
sequences and doc_idx, a single-sequence document exactly SentencePiece-v2's encoding of its text, a multi-sequence
document split mid-character decoded identically, <eos> where it was, empty document entries kept.

    TOK2_OLD=<tokenizer-256k-agent> TOK2_NEW=<tokenizer-256k-v2-agent> python scripts/tok_v2/test_retok_v2.py
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import retok_v2 as R  # noqa: E402

TEXTS = [
    "Plain English text with numbers 1234.5678 and punctuation!",
    "Deutsch: Größenwahn, Übermäßig — Straße. 東京 and emoji 😀 at the end",
    "<|im_start|>system\n<tools>[]</tools><|im_end|>\n<|im_start|>assistant\n<think>\nplan</think>"
    "<tool_call>{\"a\": 1}</tool_call><tool_response>ok</tool_response><|im_end|>",
    "def f(x):\n\tif x:\n\t\treturn x  # tabs\n\n\n    spaces",
    " leading space and trailing  ",
    "Ελληνικά και Українська, ქართული ენა",
    "[INST] mistral [/INST] <｜begin▁of▁sentence｜> deepseek <|start_header_id|>llama<|end_header_id|>",
    "x" * 3000 + " long " + "ü" * 2000,
]


def build_shards(d, sp_old):
    """(view prefix, full doc texts in view order, sequences per doc): a dense shard, then a gap view over it."""
    rng = np.random.default_rng(0)
    docs, seqs_per_doc = [], []
    lengths, toks, doc_idx = [], [], [0]
    for k in range(60):
        t = TEXTS[k % len(TEXTS)] + f" #{k}"
        ids = sp_old.encode(t) + [R.EOS]
        n_seq = 1 if k % 3 else int(rng.integers(2, 4))          # every third document over 2-3 sequences
        if k % 17 == 5:                                           # an empty document entry (no sequences)
            doc_idx.append(doc_idx[-1])
            docs.append(None)
            seqs_per_doc.append(0)
        cuts = sorted(rng.choice(np.arange(1, len(ids) - 1), n_seq - 1, replace=False)) if n_seq > 1 else []
        for a, b in zip([0] + list(cuts), list(cuts) + [len(ids)]):
            lengths.append(b - a)
            toks.extend(ids[a:b])
        doc_idx.append(doc_idx[-1] + n_seq)
        docs.append(t)
        seqs_per_doc.append(n_seq)
    full = os.path.join(d, "src", "full_text_document")
    os.makedirs(os.path.dirname(full))
    np.asarray(toks, np.int32).tofile(full + ".bin")
    R.write_index(full + ".idx", np.asarray(lengths), np.asarray(doc_idx))
    # gap view (as the decontaminated trees): every 7th document removed, pointers into the full .bin
    idx = R.Index(full + ".idx")
    keep = np.array([k % 7 != 3 for k in range(idx.doc_count - 1)])
    n_seq = idx.doc_idx[1:] - idx.doc_idx[:-1]
    keep_seq = np.repeat(keep, n_seq)
    view = os.path.join(d, "data", "srcA", "megatron-lm", "shard_0000_text_document")
    os.makedirs(os.path.dirname(view))
    os.link(full + ".bin", view + ".bin")
    with open(view + ".idx", "wb") as f:
        lv, pv = idx.lengths[keep_seq], idx.pointers[keep_seq]
        dv = np.concatenate([[0], np.cumsum(n_seq[keep])]).astype(np.int64)
        f.write(R._HEADER + R.struct.pack("<Q", 1) + R.struct.pack("<B", 4) + R.struct.pack("<Q", len(lv))
                + R.struct.pack("<Q", len(dv)))
        f.write(lv.astype(np.int32).tobytes() + pv.astype(np.int64).tobytes() + dv.tobytes())
    vdocs = [t for t, k in zip(docs, keep) if k]
    vseq = [s for s, k in zip(seqs_per_doc, keep) if k]
    return view, vdocs, vseq


def main():
    import sentencepiece as spm
    sp_old = spm.SentencePieceProcessor(model_file=os.path.join(R.OLD_TOK, "tokenizer.model"))
    sp_new = spm.SentencePieceProcessor(model_file=os.path.join(R.NEW_TOK, "tokenizer.model"))
    # the decoder reproduces SentencePiece's decode
    dec = R.Decoder(R.OLD_TOK)
    for t in TEXTS:
        ids = np.asarray(sp_old.encode(t))
        assert dec(ids).decode().replace("▁", " ") == sp_old.decode(ids.tolist()), t   # delimiters verbatim
    assert R.split_incomplete("aü".encode()[:-1]) == (b"a", "ü".encode()[:1])
    assert R.split_incomplete("a😀".encode()[:-2]) == (b"a", "😀".encode()[:2])
    assert R.split_incomplete("aü".encode()) == ("aü".encode(), b"")
    enc = R.Encoder(R.NEW_TOK)                 # the split DeepSeek delimiters: SentencePiece's ids exactly
    ds = "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
    hard = TEXTS + [ds, ds + ds, " " + ds + " x", "a" + ds + "b", ds + "<|im_start|>" + ds, "\uff5c plain bar \uff5c",
                    "<\uff5cUser\uff5c>hi<\uff5cAssistant\uff5c><think>x</think>", "", "  ", ds + " " * 3]
    flat, cnt = enc(hard)
    off = np.concatenate([[0], np.cumsum(cnt)])
    for k, t in enumerate(hard):
        assert flat[off[k]:off[k + 1]].tolist() == sp_new.encode(t), (t, flat[off[k]:off[k + 1]].tolist(), sp_new.encode(t))
    with tempfile.TemporaryDirectory() as d:
        view, docs, nseq = build_shards(d, sp_old)
        open(f"{d}/list.txt", "w").write(view + "\n")
        run = [sys.executable, os.path.join(HERE, "retok_v2.py")]
        plan = subprocess.run(run + ["plan", "--list", f"{d}/list.txt", "--data-root", f"{d}/data", "--out",
                                     f"{d}/out", "--parts", "1", "--unit-tokens", "200"],
                              check=True, capture_output=True, text=True).stdout
        open(f"{d}/plan.json", "w").write(plan)
        assert len(json.loads(plan)["shards"][0]["units"]) > 3          # several units per shard
        subprocess.run(run + ["run", "--plan", f"{d}/plan.json", "--part", "0", "--workers", "2"], check=True,
                       capture_output=True)
        out = f"{d}/out/srcA/megatron-lm/shard_0000_text_document"
        o, n = R.Index(view + ".idx"), R.Index(out + ".idx")
        assert o.seq_count == n.seq_count and np.array_equal(o.doc_idx, n.doc_idx)
        nb = np.fromfile(out + ".bin", np.int32)
        assert len(nb) == n.lengths.sum() and np.array_equal(n.pointers[1:], np.cumsum(n.lengths[:-1]) * 4)
        for k, (t, s) in enumerate(zip(docs, nseq)):
            seqs = [nb[n.pointers[q] // 4:n.pointers[q] // 4 + n.lengths[q]] for q in
                    range(n.doc_idx[k], n.doc_idx[k + 1])]
            if t is None:
                assert not seqs
                continue
            assert len(seqs) == s and seqs[-1][-1] == R.EOS
            joined = np.concatenate(seqs)
            if s == 1:                                     # exactly v2's encoding of the text, then <eos>
                assert joined.tolist() == sp_new.encode(t) + [R.EOS], (t, joined.tolist(), sp_new.encode(t))
            else:                                          # decodes to the same text
                assert sp_new.decode([int(x) for x in joined if x != R.EOS]).replace("\u2581", " ") == \
                    t.replace("\u2581", " "), t
        rec = json.load(open(out + ".tok2.json"))
        assert rec["sequences"] == o.seq_count and rec["tokens_new"] == len(nb)
        r = subprocess.run(run + ["check", "--plan", f"{d}/plan.json", "--sample", "40"], capture_output=True,
                           text=True)
        assert r.returncode == 0 and "0 bad" in r.stdout, r.stdout + r.stderr
        r = subprocess.run(run + ["run", "--plan", f"{d}/plan.json", "--part", "0", "--workers", "2"], check=True,
                           capture_output=True, text=True)                 # rerun: finished shard skipped
        assert "0 shards" in r.stdout, r.stdout
        # claim mode: the same bytes as run; a claimed shard is left to its claimer
        subprocess.run(run + ["claim", "--list", f"{d}/list.txt", "--data-root", f"{d}/data", "--out", f"{d}/out2",
                              "--unit-tokens", "200", "--workers", "2"], check=True, capture_output=True)
        out2 = f"{d}/out2/srcA/megatron-lm/shard_0000_text_document"
        assert open(out2 + ".bin", "rb").read() == open(out + ".bin", "rb").read()
        assert open(out2 + ".idx", "rb").read() == open(out + ".idx", "rb").read()
        os.remove(out2 + ".tok2.json")
        r = subprocess.run(run + ["claim", "--list", f"{d}/list.txt", "--data-root", f"{d}/data", "--out", f"{d}/out2",
                                  "--workers", "2"], check=True, capture_output=True, text=True)
        assert "done: 0 shards" in r.stdout, r.stdout                      # claimed (by the first run): skipped
        r = subprocess.run(run + ["unclaim", "--out", f"{d}/out2"], check=True, capture_output=True, text=True)
        assert "1 claims removed" in r.stdout, r.stdout
    print("retok_v2 ok")


if __name__ == "__main__":
    main()
