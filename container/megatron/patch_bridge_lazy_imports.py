#!/usr/bin/env python3
"""Make ``megatron.bridge.models.__init__`` tolerant of missing imports.

NVIDIA-NeMo/Megatron-Bridge's ``models/__init__.py`` eagerly imports every
bundled model bridge (bailing, mamba, gemma_vl, …). Some of those touch
symbols that don't exist in the OpenEuroLLM Megatron-LM fork we use for
training (e.g. ``mamba_inference_stack_spec``). Wrapping each top-level
``from megatron.bridge.models.<X> import (…)`` block in a try/except lets
Bridge load whichever model bridges *can* be resolved against the fork
(Qwen3, Llama, …) while quietly skipping the rest at import time.

We rewrite the file in place; idempotent (sentinel marker bails out).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SENTINEL = "# === oellm-autoexp: patched for tolerant model imports ===\n"

# Pattern: matches any `from megatron.bridge.<dotted> import (...)` or
# `from megatron.bridge.<dotted> import Y[, Z, ...]` block (potentially
# multiline parenthesised). We deliberately skip `megatron.bridge.models.conversion.*`
# because the conversion machinery is what AutoBridge itself needs to load.
BLOCK_RE = re.compile(
    r"^from megatron\.bridge\.([A-Za-z0-9_.]+) import \([\s\S]*?\)\s*$"
    r"|^from megatron\.bridge\.([A-Za-z0-9_.]+) import [A-Za-z0-9_,  ]+(?:  #[^\n]*)?\s*$",
    re.MULTILINE,
)


def _wrap_block(block: str) -> str:
    indented = "\n".join("    " + line for line in block.rstrip().splitlines())
    return (
        "try:\n"
        f"{indented}\n"
        "except ImportError as _e:  # added by patch_bridge_lazy_imports.py\n"
        "    import warnings as _w\n"
        "    _w.warn(f'megatron.bridge: skipped model import ({_e})', stacklevel=2)\n"
    )


def patch_file(path: Path) -> int:
    text = path.read_text()
    if SENTINEL in text:
        return 0

    out_chunks: list[str] = [SENTINEL]
    last_end = 0
    n_patched = 0
    for match in BLOCK_RE.finditer(text):
        modpath = match.group(1) or match.group(2) or ""
        # Conversion machinery must succeed for AutoBridge to be defined.
        if modpath.startswith("models.conversion") or modpath.startswith("conversion"):
            continue
        block = match.group(0)
        out_chunks.append(text[last_end : match.start()])
        out_chunks.append(_wrap_block(block))
        last_end = match.end()
        n_patched += 1
    if n_patched == 0:
        return 0
    out_chunks.append(text[last_end:])
    path.write_text("".join(out_chunks))
    return n_patched


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "Usage: patch_bridge_lazy_imports.py <path/to/megatron/bridge/models>",
            file=sys.stderr,
        )
        return 1
    target = Path(sys.argv[1])
    # Accept either the directory or the top-level __init__.py (for backwards compat).
    if target.is_file():
        target = target.parent
    total = 0
    files = 0
    for init in sorted(target.rglob("__init__.py")):
        n = patch_file(init)
        if n:
            files += 1
            total += n
            print(f"patched {n} import block(s) in {init}")
    print(f"done: {total} block(s) across {files} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
