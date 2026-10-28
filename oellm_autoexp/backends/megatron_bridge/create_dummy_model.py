"""Build a dummy HF model from a config + tokenizer.

``AutoBridge.export_ckpt()`` in Megatron-Bridge requires a full HF model
(initialised via ``from_hf_pretrained``), not just a config. This module
creates an empty model from a local config + tokenizer pair so that
conversion works on air-gapped clusters without hitting the HF Hub.

Ported from ``OpenEuroLLM/Megatron-Bridge-utils::create_dummy_model.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_dummy_model(config_path: Path, tokenizer_path: Path, outdir: Path) -> None:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"{outdir} is not empty; refusing to clobber")
    outdir.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(str(config_path))
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    model = AutoModelForCausalLM.from_config(config)

    model.save_pretrained(str(outdir))
    config.save_pretrained(str(outdir))
    tokenizer.save_pretrained(str(outdir))


def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path, help="HF config dir or file")
    ap.add_argument("tokenizer", type=Path, help="HF tokenizer dir")
    ap.add_argument("outdir", type=Path, help="Empty output dir to write dummy model")
    return ap.parse_args()


def main() -> int:
    args = _parse()
    build_dummy_model(args.config, args.tokenizer, args.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
