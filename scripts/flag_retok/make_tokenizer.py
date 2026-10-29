"""Build tokenizer-256k-agent from openeurollm/tokenizer-256k: free <unused_N> slots renamed to the delimiter
tokens of delimiters.py, in the SentencePiece model and the Hugging Face files alike.

    python scripts/flag_retok/make_tokenizer.py --base <dir with tokenizer.model, tokenizer.json,
        tokenizer_config.json, special_tokens_map.json[, added_tokens.json]> --out <dir>

Checks: every delimiter encodes to its single id (SentencePiece and HF fast tokenizer), the base and agent
tokenizers agree on text without delimiters, and a text with delimiters splits around them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delimiters import AGENT_TOKENS, check  # noqa: E402

SAMPLE = ("Plain text, code `x = a<b and b>c`, a table | a | b |, [link](u) and Ünïcödé: 東京, ‖ 1234.5678\n"
          "    indented <html><body>tags</body></html> ")


def rename_sp(model_bytes: bytes, mapping: dict) -> bytes:
    """The SentencePiece model with the given <unused_N> slots renamed (USER_DEFINED pieces: matched in text)."""
    from sentencepiece import sentencepiece_model_pb2 as pb
    m = pb.ModelProto()
    m.ParseFromString(model_bytes)
    UD = pb.ModelProto.SentencePiece.USER_DEFINED
    for i, t in mapping.items():
        p = m.pieces[i]
        assert p.type == UD and p.piece == f"<unused_{i - 13}>", (i, p.piece, p.type)
        p.piece = t
    return m.SerializeToString()


def build(base, out):
    import sentencepiece as spm
    mapping = check()
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "tokenizer.model"), "wb").write(
        rename_sp(open(os.path.join(base, "tokenizer.model"), "rb").read(), mapping))

    j = json.load(open(os.path.join(base, "tokenizer.json")))
    by_id = {a["id"]: a for a in j["added_tokens"]}
    vocab = j["model"]["vocab"]
    for i, t in mapping.items():
        old = by_id[i]["content"]
        by_id[i]["content"] = t
        assert vocab.pop(old) == i
        vocab[t] = i
    json.dump(j, open(os.path.join(out, "tokenizer.json"), "w"), ensure_ascii=False, indent=2)

    c = json.load(open(os.path.join(base, "tokenizer_config.json")))
    for i, t in mapping.items():
        c["added_tokens_decoder"][str(i)]["content"] = t
    extra = [t for _, t in AGENT_TOKENS]
    c["additional_special_tokens"] = [x for x in c.get("additional_special_tokens", [])
                                      if not str(x).startswith("<unused_")] + extra
    json.dump(c, open(os.path.join(out, "tokenizer_config.json"), "w"), ensure_ascii=False, indent=2)

    s = json.load(open(os.path.join(base, "special_tokens_map.json")))
    keep = [x for x in s.get("additional_special_tokens", [])
            if not (x["content"] if isinstance(x, dict) else x).startswith("<unused_")]
    s["additional_special_tokens"] = keep + [{"content": t, "lstrip": False, "normalized": False, "rstrip": False,
                                              "single_word": False} for t in extra]
    json.dump(s, open(os.path.join(out, "special_tokens_map.json"), "w"), ensure_ascii=False, indent=2)
    if os.path.exists(os.path.join(base, "added_tokens.json")):
        shutil.copy(os.path.join(base, "added_tokens.json"), os.path.join(out, "added_tokens.json"))
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write("# tokenizer-256k-agent\n\nopeneurollm/tokenizer-256k with free `<unused_N>` slots renamed to chat / "
                "agent / FIM delimiters (scripts/flag_retok/delimiters.py, docs/flag_retokenization.md). Vocabulary "
                "(262,144), ordinary pieces and merges unchanged; ids 13-18 as in openeurollm/tokenizer-256k-reason.\n\n"
                "| id | token |\n|---:|---|\n" + "".join(f"| {i} | `{t}` |\n" for i, t in AGENT_TOKENS))

    # checks
    base_sp = spm.SentencePieceProcessor(model_file=os.path.join(base, "tokenizer.model"))
    sp = spm.SentencePieceProcessor(model_file=os.path.join(out, "tokenizer.model"))
    assert sp.get_piece_size() == base_sp.get_piece_size()
    for i, t in mapping.items():
        assert sp.encode(t) == [i], (t, sp.encode(t))
    assert sp.encode(SAMPLE) == base_sp.encode(SAMPLE)
    mixed = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\nok</think> [INST] x [/INST]"
    ids = sp.encode(mixed)
    assert ids[0] == mapping_id("<|im_start|>") and mapping_id("<think>") in ids and mapping_id("[/INST]") in ids
    assert sp.decode(ids) == mixed
    try:
        from tokenizers import Tokenizer
        hf = Tokenizer.from_file(os.path.join(out, "tokenizer.json"))
        for i, t in mapping.items():
            assert hf.encode(t, add_special_tokens=False).ids == [i], (t, hf.encode(t).ids)
        assert hf.encode(SAMPLE, add_special_tokens=False).ids == sp.encode(SAMPLE)
        assert hf.encode(mixed, add_special_tokens=False).ids == ids, (hf.encode(mixed).ids, ids)
    except ImportError:
        print("tokenizers not installed: HF checks skipped")
    print(f"{out}: {len(mapping)} delimiter tokens renamed; checks passed")


def mapping_id(t):
    return {v: k for k, v in AGENT_TOKENS}[t]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.base, a.out)
