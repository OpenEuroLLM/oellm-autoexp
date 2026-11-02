"""Build tokenizer-256k-v2-agent from openeurollm/tokenizer-256k-v2: the agent / chat / FIM delimiters of
scripts/flag_retok/delimiters.py that v2 lacks, in free <unused_N> slots (ids 80..207, USER_DEFINED pieces:
SentencePiece matches a renamed slot in raw text as one piece), in the SentencePiece model and the Hugging Face
files alike. v2's own delimiters keep their ids (<think> 75, </think> 76, <tool_call> 73, </tool_call> 74,
<|im_start|> 69, <|im_end|> 70); vocabulary size (262,144), ordinary pieces and merges unchanged.

    python scripts/tok_v2/make_tokenizer_v2.py --base <openeurollm/tokenizer-256k-v2 dir> --out <dir>

Checks: every delimiter encodes to its single id (SentencePiece and HF), base and new tokenizer agree on text
without delimiters, a text with delimiters splits around them and decodes back.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "flag_retok"))
from delimiters import AGENT_TOKENS  # noqa: E402

FIRST_FREE, N_FREE = 80, 128            # v2: <unused_0> .. <unused_127> = ids 80 .. 207
SAMPLE = ("Plain text, code `x = a<b and b>c`, a table | a | b |, [link](u) and Ünïcödé: 東京, ‖ 1234.5678\n"
          "    indented <html><body>tags</body></html>\t\tx\n\n")


def plan(base_specials: dict) -> dict:
    """id -> delimiter for the delimiters v2 lacks, in delimiters.py's order, from <unused_0> on."""
    missing = [t for _, t in AGENT_TOKENS if t not in base_specials]
    assert len(missing) <= N_FREE
    return {FIRST_FREE + k: t for k, t in enumerate(missing)}


def rename_sp(model_bytes: bytes, mapping: dict) -> bytes:
    from sentencepiece import sentencepiece_model_pb2 as pb
    m = pb.ModelProto()
    m.ParseFromString(model_bytes)
    UD = pb.ModelProto.SentencePiece.USER_DEFINED
    for i, t in mapping.items():
        p = m.pieces[i]
        assert p.type == UD and p.piece == f"<unused_{i - FIRST_FREE}>", (i, p.piece, p.type)
        p.piece = t
    return m.SerializeToString()


def build(base, out):
    import sentencepiece as spm
    j = json.load(open(os.path.join(base, "tokenizer.json")))
    specials = {a["content"]: a["id"] for a in j["added_tokens"]}
    mapping = plan(specials)
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "tokenizer.model"), "wb").write(
        rename_sp(open(os.path.join(base, "tokenizer.model"), "rb").read(), mapping))

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
    c["additional_special_tokens"] = [x for x in c.get("additional_special_tokens", [])
                                      if not str(x).startswith("<unused_")] + list(mapping.values())
    json.dump(c, open(os.path.join(out, "tokenizer_config.json"), "w"), ensure_ascii=False, indent=2)

    s = json.load(open(os.path.join(base, "special_tokens_map.json")))
    keep = [x for x in s.get("additional_special_tokens", [])
            if not (x["content"] if isinstance(x, dict) else x).startswith("<unused_")]
    s["additional_special_tokens"] = keep + [{"content": t, "lstrip": False, "normalized": False, "rstrip": False,
                                              "single_word": False} for t in mapping.values()]
    json.dump(s, open(os.path.join(out, "special_tokens_map.json"), "w"), ensure_ascii=False, indent=2)
    present = [(specials[t], t) for _, t in AGENT_TOKENS if t in specials]
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write("# tokenizer-256k-v2-agent\n\nopeneurollm/tokenizer-256k-v2 with the chat / agent / FIM delimiters "
                "of tokenizer-256k-agent (scripts/flag_retok/delimiters.py) that v2 lacks, in free `<unused_N>` "
                "slots (scripts/tok_v2/make_tokenizer_v2.py). Vocabulary (262,144), ordinary pieces and merges "
                "unchanged.\n\nAlready in v2 (ids kept):\n\n| id | token |\n|---:|---|\n"
                + "".join(f"| {i} | `{t}` |\n" for i, t in sorted(present))
                + "\nAdded (former `<unused_N>`):\n\n| id | token |\n|---:|---|\n"
                + "".join(f"| {i} | `{t}` |\n" for i, t in mapping.items()))

    base_sp = spm.SentencePieceProcessor(model_file=os.path.join(base, "tokenizer.model"))
    sp = spm.SentencePieceProcessor(model_file=os.path.join(out, "tokenizer.model"))
    assert sp.get_piece_size() == base_sp.get_piece_size() == 262144
    allid = {**{t: i for i, t in present}, **{t: i for i, t in mapping.items()}}
    for t, i in allid.items():
        assert sp.encode(t) == [i], (t, sp.encode(t))
    assert sp.encode(SAMPLE) == base_sp.encode(SAMPLE)
    mixed = ("<|im_start|>system\n<tools>[]</tools><|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
             "<think>\nok</think><tool_call>{}</tool_call><tool_response>r</tool_response> [INST] x [/INST]"
             "<｜begin▁of▁sentence｜>")
    ids = sp.encode(mixed)
    for t in ["<|im_start|>", "<tools>", "</tools>", "<think>", "<tool_response>", "[/INST]", "<｜begin▁of▁sentence｜>"]:
        assert allid[t] in ids, t
    assert sp.decode(ids) == mixed.replace("\u2581", " ")       # SentencePiece decodes the ▁ inside DeepSeek delimiters as spaces
    assert sp.encode(sp.decode(ids)) == ids                     # ... and encodes either form to the delimiter
    from tokenizers import Tokenizer
    hf = Tokenizer.from_file(os.path.join(out, "tokenizer.json"))
    for t, i in allid.items():
        assert hf.encode(t, add_special_tokens=False).ids == [i], (t, hf.encode(t).ids)
    assert hf.encode(SAMPLE, add_special_tokens=False).ids == sp.encode(SAMPLE)
    assert hf.encode(mixed, add_special_tokens=False).ids == ids, (hf.encode(mixed).ids, ids)
    print(f"{out}: {len(mapping)} delimiters added (ids {min(mapping)}..{max(mapping)}), {len(present)} kept; "
          "checks passed")
    return allid


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.base, a.out)
