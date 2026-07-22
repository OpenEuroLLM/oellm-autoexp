#!/usr/bin/env python3
"""
Build a Megatron --data-args-path datamix over the READABLE long-ctx-sample data.

WHY THIS EXISTS
---------------
The real multilingual baseline reads
    /scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/...
which laingsam cannot read (not a member of project_462000963; project root is
2770 so it can't even be traversed).

But /scratch/project_465002530/preprocessed/oellm-v1-256k/long-ctx-sample/ IS
readable, is tokenized with the SAME openeurollm 256k tokenizer, and (per the
.stats.txt files) was sampled FROM that very data. It covers the same dataset
families under flattened names:
    catalogue/hplt3/hplt3_sampled/fra_Latn   ->  long-ctx-sample/hplt3-fra
    catalogue/allenai/OLMO-mix-.../arxiv-... ->  long-ctx-sample/arxiv
    tower9b/<lang>                           ->  long-ctx-sample/multisynth-9b-<lang>

IMPORTANT CAVEAT -- READ THIS BEFORE PUBLISHING ANY NUMBER
----------------------------------------------------------
long-ctx-sample is a LONG-CONTEXT-BIASED SAMPLE, not the pretraining
distribution. For hplt3-fra: the source has 3.5% of tokens in long docs; the
sample has 71%. It oversamples long documents ~20x, and holds only 14k docs
per source (~447M tokens for hplt3-fra; ~32B tokens total across 153 files).

=> Valid for: pipeline validation, and a CONTROLLED Muon-vs-Adam comparison
   (both optimizers see identical data, so the contrast is internally valid).
=> NOT valid for: comparing against the existing 0.1B_ne Adam baselines, which
   ran on the real mix. Those numbers are not comparable to these.

WEIGHTS
-------
OFFICIAL_MIX below is copied verbatim from MareNostrum5:
  $WORK/data_utils/datamix-tools/datamix-tools/output/1TT-option-4-apps.sh
with the /apps/... path prefix stripped. It is the same "option 4" mix the LUMI
runs use (LUMI's file is datamix4-lumi.txt -- same mix, different path prefix).
The script self-checks that these weights sum to ~1.0; if the transcription were
wrong, that check fails loudly rather than silently producing a bad blend.

Entries whose data is absent from long-ctx-sample (e.g. kat_Geor) are DROPPED and
the remaining weights RENORMALIZED to sum to 1.0. Dropped entries are reported.

USAGE (read-only except for the single output file):
    python3 build_longctx_datamix.py            # dry run: report only, writes nothing
    python3 build_longctx_datamix.py --write    # writes the datamix file
"""

import os
import struct
import sys

# Megatron MMapIndexedDataset .idx header (see indexed_dataset.py _IndexReader):
#   9 bytes  magic  b"MMIDIDX\x00\x00"
#   8 bytes  version (uint64)
#   1 byte   dtype code (uint8)   <- dtype of the .bin tokens, not the index arrays
#   8 bytes  sequence_count (uint64)
#   8 bytes  document_count (uint64)
# then: sequence_lengths[int32 * sc], sequence_pointers[int64 * sc],
#       document_indices[int64 * dc].
# Megatron asserts sequence_lengths.shape[0] == document_indices[-1], i.e.
#   sequence_count == document_indices[dc-1].
# We replicate that check by reading only the header + the last 8-byte int,
# so a malformed/half-written index is dropped here instead of crashing training.
_INDEX_MAGIC = b"MMIDIDX\x00\x00"
_HEADER_LEN = 9 + 8 + 1 + 8 + 8  # = 34


def validate_prefix(prefix):
    """Return None if (prefix.bin, prefix.idx) form a usable, self-consistent
    Megatron dataset; else a short reason string."""
    if not os.path.exists(prefix + ".bin"):
        return "no .bin"
    idx = prefix + ".idx"
    if not os.path.exists(idx):
        return "no .idx (unindexed)"
    try:
        with open(idx, "rb") as f:
            if f.read(9) != _INDEX_MAGIC:
                return "bad .idx magic"
            f.read(8)                                       # version
            f.read(1)                                       # dtype code
            sc = struct.unpack("<Q", f.read(8))[0]          # sequence_count
            dc = struct.unpack("<Q", f.read(8))[0]          # document_count
            if sc == 0 or dc == 0:
                return f"empty index (sc={sc}, dc={dc})"
            # document_indices[-1] offset: header + 4*sc + 8*sc + 8*(dc-1)
            f.seek(_HEADER_LEN + 12 * sc + 8 * (dc - 1))
            last = f.read(8)
            if len(last) != 8:
                return "index truncated"
            last_doc = struct.unpack("<q", last)[0]
            if last_doc != sc:
                return f"inconsistent index (document_indices[-1]={last_doc} != sequence_count={sc})"
    except Exception as e:
        return f"idx read error: {e}"
    return None

DATA_ROOT = "/scratch/project_465002530/preprocessed/oellm-v1-256k/long-ctx-sample"
OUT_PATH = "/scratch/project_465002530/users/laingsam/oellm-muon/muon_smoke/datamix4-longctx-lumi.txt"

# Verbatim from MN5 1TT-option-4-apps.sh, prefix stripped. "<weight> <family>/<lang>".
OFFICIAL_MIX = """
0.000980 opus-mt-10p-sample/est_Latn
0.000994 opus-mt-10p-sample/bos_Latn
0.000997 hplt3-10p-sampled/nno_Latn
0.000020 finepdfs-edu-10p-sample/nno_Latn
0.000678 tower9b/nno_Latn
0.000030 finepdfs-edu-10p-sample/lit_Latn
0.002216 tower9b/pol_Latn
0.002164 hplt3-10p-sampled/ces_Latn
0.001208 hplt3-10p-sampled/gle_Latn
0.000048 finepdfs-edu-10p-sample/hun_Latn
0.001176 hplt3-10p-sampled/eus_Latn
0.000140 finepdfs-edu-10p-sample/fra_Latn
0.010192 hplt3-10p-sampled/spa_Latn
0.002234 hplt3-10p-sampled/ukr_Cyrl
0.005054 hplt3-10p-sampled/ita_Latn
0.000231 finepdfs-edu-10p-sample/kat_Geor
0.001134 opus-mt-10p-sample/srp_Cyrl
0.000204 finepdfs-edu-10p-sample/spa_Latn
0.000829 finepdfs-10p-sample/tur_Latn
0.001629 tower9b/hun_Latn
0.000030 finepdfs-edu-10p-sample/mlt_Latn
0.000029 finepdfs-edu-10p-sample/ekk_Latn
0.000049 finepdfs-edu-10p-sample/ron_Latn
0.000488 finepdfs-10p-sample/hrv_Latn
0.000456 finepdfs-10p-sample/dan_Latn
0.000932 tower9b/nob_Latn
0.001399 tower72b/swe_Latn
0.001232 hplt3-10p-sampled/glg_Latn
0.002763 hplt3-10p-sampled/tur_Latn
0.001018 opus-mt-10p-sample/lit_Latn
0.000353 finepdfs-10p-sample/eus_Latn
0.000063 finepdfs-edu-10p-sample/ell_Grek
0.000033 finepdfs-edu-10p-sample/slk_Latn
0.001462 hplt3-10p-sampled/bos_Latn
0.007849 opus-mt-10p-sample/kat_Geor
0.001497 hplt3-10p-sampled/lit_Latn
0.001173 opus-mt-10p-sample/cat_Latn
0.000030 finepdfs-edu-10p-sample/slv_Latn
0.003259 hplt3-10p-sampled/pol_Latn
0.000335 finepdfs-10p-sample/isl_Latn
0.001106 opus-mt-10p-sample/hrv_Latn
0.001116 hplt3-10p-sampled/isl_Latn
0.001879 opus-mt-10p-sample/tur_Latn
0.001400 opus-mt-10p-sample/sqi_Latn
0.006995 hplt3-10p-sampled/fra_Latn
0.000045 finepdfs-edu-10p-sample/ukr_Cyrl
0.003463 finepdfs-10p-sample/kat_Geor
0.000608 finepdfs-10p-sample/lvs_Latn
0.000446 finepdfs-10p-sample/slv_Latn
0.000759 tower9b/isl_Latn
0.002441 hplt3-10p-sampled/ron_Latn
0.001660 tower9b/ron_Latn
0.000025 finepdfs-edu-10p-sample/glg_Latn
0.000421 finepdfs-10p-sample/mkd_Cyrl
0.001204 tower72b/fin_Latn
0.003437 tower72b/ita_Latn
0.000838 opus-mt-10p-sample/glg_Latn
0.000978 finepdfs-10p-sample/pol_Latn
0.000800 opus-mt-10p-sample/eus_Latn
0.001725 hplt3-10p-sampled/cat_Latn
0.000719 finepdfs-10p-sample/hun_Latn
0.002026 hplt3-10p-sampled/lvs_Latn
0.001486 hplt3-10p-sampled/slv_Latn
0.003058 finepdfs-10p-sample/spa_Latn
0.000578 finepdfs-10p-sample/bul_Cyrl
0.000846 opus-mt-10p-sample/gle_Latn
0.000370 finepdfs-10p-sample/glg_Latn
0.000439 finepdfs-10p-sample/bos_Latn
0.000033 finepdfs-edu-10p-sample/hrv_Latn
0.003138 hplt3-10p-sampled/ell_Grek
0.000024 finepdfs-edu-10p-sample/eus_Latn
0.002133 opus-mt-10p-sample/ell_Grek
0.000040 finepdfs-edu-10p-sample/lvs_Latn
0.000617 finepdfs-10p-sample/als_Latn
0.001699 finepdfs-10p-sample/por_Latn
0.000038 finepdfs-edu-10p-sample/bul_Cyrl
0.000034 finepdfs-edu-10p-sample/cat_Latn
0.006930 tower72b/spa_Latn
0.005300 hplt3-10p-sampled/deu_Latn
0.001471 opus-mt-10p-sample/ces_Latn
0.000953 opus-mt-10p-sample/mkd_Cyrl
0.000059 finepdfs-edu-10p-sample/nld_Latn
0.001667 hplt3-10p-sampled/srp_Cyrl
0.000027 finepdfs-edu-10p-sample/nob_Latn
0.002098 finepdfs-10p-sample/fra_Latn
0.000035 finepdfs-edu-10p-sample/fin_Latn
0.001377 opus-mt-10p-sample/lav_Latn
0.000030 finepdfs-edu-10p-sample/dan_Latn
0.004756 tower9b/fra_Latn
0.000447 finepdfs-10p-sample/mlt_Latn
0.000517 finepdfs-10p-sample/cat_Latn
0.000362 finepdfs-10p-sample/gle_Latn
0.001113 opus-mt-10p-sample/slk_Latn
0.002058 hplt3-10p-sampled/als_Latn
0.000432 finepdfs-10p-sample/ekk_Latn
0.000033 finepdfs-edu-10p-sample/srp_Cyrl
0.002395 hplt3-10p-sampled/hun_Latn
0.001519 hplt3-10p-sampled/dan_Latn
0.002020 tower9b/nld_Latn
0.003604 tower72b/deu_Latn
0.000891 finepdfs-10p-sample/nld_Latn
0.000022 finepdfs-edu-10p-sample/isl_Latn
0.001637 hplt3-10p-sampled/slk_Latn
0.000113 finepdfs-edu-10p-sample/por_Latn
0.001370 hplt3-10p-sampled/nob_Latn
0.000043 finepdfs-edu-10p-sample/ces_Latn
0.001491 hplt3-10p-sampled/mlt_Latn
0.000941 finepdfs-10p-sample/ell_Grek
0.000055 finepdfs-edu-10p-sample/tur_Latn
0.011542 hplt3-10p-sampled/kat_Geor
0.000029 finepdfs-edu-10p-sample/bos_Latn
0.000500 finepdfs-10p-sample/srp_Cyrl
0.003852 tower9b/por_Latn
0.001010 opus-mt-10p-sample/slv_Latn
0.000732 finepdfs-10p-sample/ron_Latn
0.000491 finepdfs-10p-sample/slk_Latn
0.000649 finepdfs-10p-sample/ces_Latn
0.000065 finepdfs-edu-10p-sample/pol_Latn
0.001441 hplt3-10p-sampled/ekk_Latn
0.000041 finepdfs-edu-10p-sample/als_Latn
0.000670 finepdfs-10p-sample/ukr_Cyrl
0.001516 finepdfs-10p-sample/ita_Latn
0.001033 tower9b/dan_Latn
0.001590 finepdfs-10p-sample/deu_Latn
0.000041 finepdfs-edu-10p-sample/swe_Latn
0.001925 hplt3-10p-sampled/bul_Cyrl
0.001402 hplt3-10p-sampled/mkd_Cyrl
0.000617 finepdfs-10p-sample/swe_Latn
0.001309 opus-mt-10p-sample/bul_Cyrl
0.000299 finepdfs-10p-sample/nno_Latn
0.000411 finepdfs-10p-sample/nob_Latn
0.000101 finepdfs-edu-10p-sample/ita_Latn
0.002971 hplt3-10p-sampled/nld_Latn
0.001014 opus-mt-10p-sample/mlt_Latn
0.001770 hplt3-10p-sampled/fin_Latn
0.000028 finepdfs-edu-10p-sample/mkd_Cyrl
0.002057 hplt3-10p-sampled/swe_Latn
0.001519 tower9b/ukr_Cyrl
0.005665 hplt3-10p-sampled/por_Latn
0.000449 finepdfs-10p-sample/lit_Latn
0.001626 hplt3-10p-sampled/hrv_Latn
0.000531 finepdfs-10p-sample/fin_Latn
0.000106 finepdfs-edu-10p-sample/deu_Latn
0.324000 dclm-baseline-1.0-10p-sample/dclm-10p-sample
0.108000 finepdfs-10p-sample/eng_Latn
0.054000 nemotron-cc/nemotron-ha-10p-sample
0.054000 nemotron-cc/nemotron-ma-10p-sample
0.131440 nemotron-cc/nemotron-mha-10p-sample
0.012560 finepdfs-edu-10p-sample/eng_Latn
0.004000 finemath-10p-sample/finemath4plus-10p-sample
0.002000 LLM360/megamath-text-code-block-10p-sample
0.002000 LLM360/megamath-web-pro-10p-sample
0.009000 OLMO-mix-10p-sample/arxiv-10p-sample
0.009000 OLMO-mix-10p-sample/pes2o-10p-sample
0.018000 OLMO-mix-10p-sample/Wiki-10p-sample
0.072000 starcoder-10p-sample/starcoder-10p-sample
"""

# MN5 mix path -> long-ctx-sample basename, for the non-per-language entries.
SINGLETON_MAP = {
    "dclm-baseline-1.0-10p-sample/dclm-10p-sample": "dclm",
    "nemotron-cc/nemotron-ha-10p-sample": "nemotron-ha",
    "nemotron-cc/nemotron-ma-10p-sample": "nemotron-ma",
    "nemotron-cc/nemotron-mha-10p-sample": "nemotron-mha",
    "finemath-10p-sample/finemath4plus-10p-sample": "finemath",
    "LLM360/megamath-text-code-block-10p-sample": "megamath-code",
    "LLM360/megamath-web-pro-10p-sample": "megamath-web",
    "OLMO-mix-10p-sample/arxiv-10p-sample": "arxiv",
    "OLMO-mix-10p-sample/pes2o-10p-sample": "pes2o",
    "OLMO-mix-10p-sample/Wiki-10p-sample": "wiki",
    "starcoder-10p-sample/starcoder-10p-sample": "starcoder",
}

# MN5 family -> long-ctx-sample family prefix, for "<family>/<lang>_<Script>" entries.
# tower9b/tower72b are MN5's names for what LUMI calls multisynth-9b/multisynth-72b.
FAMILY_MAP = {
    "hplt3-10p-sampled": "hplt3",
    "opus-mt-10p-sample": "opus-mt",
    "finepdfs-10p-sample": "finepdfs",
    "finepdfs-edu-10p-sample": "finepdfs-edu",
    "tower9b": "multisynth-9b",
    "tower72b": "multisynth-72b",
}


def map_entry(mn5_path):
    """MN5 mix path -> long-ctx-sample basename, or None if unmappable."""
    if mn5_path in SINGLETON_MAP:
        return SINGLETON_MAP[mn5_path]
    if "/" not in mn5_path:
        return None
    family, lang = mn5_path.split("/", 1)
    if family not in FAMILY_MAP:
        return None
    # "fra_Latn" -> "fra";  "kat_Geor" -> "kat"
    lang3 = lang.split("_")[0]
    return f"{FAMILY_MAP[family]}-{lang3}"


def main():
    write = "--write" in sys.argv

    entries = []
    for line in OFFICIAL_MIX.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        w, p = line.split()
        entries.append((float(w), p))

    # --- self-check: did the transcription survive? ---
    total = sum(w for w, _ in entries)
    print(f"official mix: {len(entries)} entries, weights sum to {total:.6f}")
    if abs(total - 1.0) > 0.01:
        print(f"FATAL: official weights sum to {total:.6f}, expected ~1.0.")
        print("The OFFICIAL_MIX transcription is wrong. Refusing to emit a bad blend.")
        return 1

    kept, dropped = [], []
    for w, p in entries:
        base = map_entry(p)
        if base is None:
            dropped.append((w, p, "unmapped family"))
            continue
        prefix = os.path.join(DATA_ROOT, base)
        reason = validate_prefix(prefix)   # checks .bin + .idx + index consistency
        if reason is not None:
            dropped.append((w, p, f"{base}: {reason}"))
            continue
        kept.append((w, prefix))

    print(f"\nkept:    {len(kept)} entries ({sum(w for w, _ in kept):.6f} of original weight)")
    print(f"dropped: {len(dropped)} entries ({sum(w for w, _, _ in dropped):.6f} of original weight)")
    for w, p, why in dropped:
        print(f"    DROP {w:.6f}  {p:<48} ({why})")

    if not kept:
        print("FATAL: nothing mapped. Check DATA_ROOT.")
        return 1

    # --- renormalize so the kept weights sum to 1.0 ---
    scale = 1.0 / sum(w for w, _ in kept)
    kept = [(w * scale, p) for w, p in kept]
    print(f"\nrenormalized: {len(kept)} entries now sum to {sum(w for w, _ in kept):.6f}")

    # --- emit "weight path weight path ..." (see get_blend_from_list) ---
    tokens = []
    for w, p in kept:
        tokens.append(f"{w:.8f}")
        tokens.append(p)

    # Megatron: ODD token count => the whole list is treated as prefixes with NO
    # weights. That failure is silent and would quietly give a uniform blend.
    if len(tokens) % 2 != 0:
        print(f"FATAL: emitted {len(tokens)} tokens (odd). Megatron would ignore all weights.")
        return 1
    print(f"emitting {len(tokens)} tokens (even -> weights will be honoured)")

    text = " ".join(tokens) + "\n"

    if not write:
        print(f"\nDRY RUN -- nothing written. Re-run with --write to create:\n    {OUT_PATH}")
        print(f"\nfirst 200 chars of what would be written:\n{text[:200]}...")
        return 0

    with open(OUT_PATH, "w") as f:
        f.write(text)
    print(f"\nWROTE {OUT_PATH}  ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
