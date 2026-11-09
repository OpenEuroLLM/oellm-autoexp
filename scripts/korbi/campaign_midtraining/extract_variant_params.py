"""The trained sparse-attention parameters of a Megatron fsdp_dtensor checkpoint (NSA's compressors / gates, CSA2's
compression weights / lightning indexer, a page indexer: ``...self_attention.core_attention.{variant,indexer}.*``)
for long_attention_probe.py's ``--variant-params``: ``{ordinal of the attention layer: {param name: tensor}}``.
The HF export keeps the attention layers in the same order, so the k-th attention layer with parameters in Megatron
is the probe's k-th attention mixer.

    python extract_variant_params.py <ckpt>/iter_0001500 <out>/variants.pt
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from oellm_autoexp.hf_export.fsdp_dtensor import load, model_keys  # noqa: E402

_PARAM = re.compile(r"decoder\.layers\.(\d+)\.self_attention\.core_attention\.(variant|indexer)\.(.+)$")


def main(ckpt: str, out: str) -> None:
    reader, keys = model_keys(Path(ckpt))
    picked = {}
    for name, entry in keys.items():
        m = _PARAM.search(name)
        if m:
            picked[name] = (int(m.group(1)), m.group(3), entry)
    if not picked:
        raise SystemExit(f"{ckpt}: no core_attention.variant / indexer parameters (a page-selector or dense arm)")
    tensors = load(reader, {name: entry for name, (_, _, entry) in picked.items()})
    layers = sorted({layer for layer, _, _ in picked.values()})
    state = {k: {} for k in range(len(layers))}
    for name, (layer, param, _) in picked.items():
        state[layers.index(layer)][param] = tensors[name].clone()
    torch.save({"layers": layers, "state": state}, out)
    print(f"{len(picked)} tensors of {len(layers)} attention layers (Megatron layers {layers}) -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
