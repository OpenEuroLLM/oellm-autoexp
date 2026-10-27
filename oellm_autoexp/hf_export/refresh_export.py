"""Refresh an existing HF export without re-reading the checkpoint: the model code from this
checkout, and the bos / eos / pad ids (config.json + generation_config.json) from its tokenizer.

    python -m oellm_autoexp.hf_export.refresh_export <export dir> [<export dir> ...]
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from .export_q35 import model_token_ids, special_token_ids

_HERE = Path(__file__).resolve().parent


def refresh(d: Path) -> None:
    for pkg in ("oellm_ambitious", "oellm_hybrid", "oellm_q35"):
        for f in (_HERE / pkg).glob("*.py"):
            if (d / f.name).exists():
                shutil.copy2(f, d / f.name)
                print(f"{d.name}: {f.name}")
    cfg = json.loads((d / "config.json").read_text())
    ids = model_token_ids(special_token_ids(d), cfg["vocab_size"])
    (d / "config.json").write_text(json.dumps({**cfg, **ids}, indent=2) + "\n")
    (d / "generation_config.json").write_text(json.dumps(ids, indent=2) + "\n")
    print(f"{d.name}: {ids}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        refresh(Path(arg))
