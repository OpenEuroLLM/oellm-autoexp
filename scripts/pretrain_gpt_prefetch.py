#!/usr/bin/env python3
"""Entrypoint wrapper: apply the node-local mirror read-through monkeypatch, then
run Megatron's pretrain_gpt.py unchanged.

Use as the training launcher_script when running with the async local-mirror
prefetcher. When OELLM_MIRROR_DIR is unset, the patch is a no-op, so this is a
safe drop-in for pretrain_gpt.py.
"""

import os
import runpy
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MEGATRON = os.path.join(_REPO, "submodules", "Megatron-LM")
# pretrain_gpt.py is normally run from its own dir (so `import megatron` resolves);
# replicate that, plus the repo root for the oellm_autoexp package.
sys.path.insert(0, _MEGATRON)
sys.path.insert(0, _REPO)

import oellm_autoexp.data_staging.megatron_patch  # noqa: F401,E402  (applies patch on import)

if __name__ == "__main__":
    runpy.run_path(os.path.join(_MEGATRON, "pretrain_gpt.py"), run_name="__main__")
