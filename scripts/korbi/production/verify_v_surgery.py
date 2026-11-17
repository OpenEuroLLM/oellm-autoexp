#!/usr/bin/env python3
"""Verify a V-layer checkpoint conversion against its 98-layer source.

Checks, in order:

1. Every source metadata FQN survives the 98 -> 104 layer renumbering (1:1 metadata).
2. Every source tensor chunk still points at the same hard-linked file/offset/size.
3. The six V layers have exactly the model and optimizer keys the two surgery scripts write.
4. The parameter count increases by exactly the six V-layer parameter counts.
5. New V output projections are zero, norms are one, and the mHC mixer is the identity init.
6. A sampled existing source tensor is value-identical before and after.

Usage:
  python verify_v_surgery.py --src <original>/iter_0035900 --dst <v4>/iter_0035900
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submodules" / "Megatron-LM"))
from add_v_mhc import mixer_tensors  # noqa: E402
from insert_v_layers import ORIG_PATTERN, INSERT_AFTER, build_pattern, v_tensors  # noqa: E402


def old_to_new(old: int) -> int:
    return old + sum(1 for p in INSERT_AFTER if p < old)


def rename_key(key: str) -> str:
    marker = "decoder.layers."
    if marker not in key:
        return key
    head, rest = key.split(marker, 1)
    digits = []
    for ch in rest:
        if not ch.isdigit():
            break
        digits.append(ch)
    if not digits:
        return key
    old = int("".join(digits))
    return f"{head}{marker}{old_to_new(old)}{rest[len(digits):]}"


def model_param_count(md) -> int:
    total = 0
    for key, meta in md.state_dict_metadata.items():
        if key.startswith("model.module.") and isinstance(meta, TensorStorageMetadata):
            total += meta.size.numel()
    return total


def expected_v_keys(layer: int) -> set[str]:
    v_all = v_tensors(layer, 1234)
    vm = {k: v for k, v in v_all.items() if k.startswith("model.module.")}
    vo = {k: v for k, v in v_all.items() if k.startswith("optimizer.")}
    mm, mo = mixer_tensors(layer)
    assert set(vm).isdisjoint(mm) and set(vo).isdisjoint(mo)
    return set(vm) | set(mm) | set(vo) | set(mo)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--sample-values", action="store_true", help="also load one existing tensor pair")
    a = ap.parse_args()

    smd = FileSystemReader(a.src).read_metadata()
    dmd = FileSystemReader(a.dst).read_metadata()
    s, d = smd.state_dict_metadata, dmd.state_dict_metadata
    pattern, v_layers = build_pattern()
    assert pattern.replace("V", "") == ORIG_PATTERN
    assert len(pattern) == 104
    assert len(v_layers) == 6

    # 1. Every source key has its shifted target; no key is lost.
    missing = []
    for key in s:
        if rename_key(key) not in d:
            missing.append((key, rename_key(key)))
    if missing:
        print(f"FAIL missing {len(missing)} source keys; first: {missing[:5]}")
        return 1
    print(f"PASS metadata: {len(s)} source keys -> all present after renumbering")

    # 1b. The pickled args object must describe the 104-layer model, not the 98-layer source.
    args_sd = {"args": None}
    dcp.load(args_sd, storage_reader=FileSystemReader(a.dst))
    ckpt_args = args_sd["args"]
    if getattr(ckpt_args, "num_layers", None) != len(pattern):
        print(f"FAIL args.num_layers: {getattr(ckpt_args, 'num_layers', None)} != {len(pattern)}")
        return 1
    print(f"PASS args: num_layers={ckpt_args.num_layers}, pattern={ckpt_args.hybrid_layer_pattern}")

    # 2. Every source chunk keeps its exact file/offset/size.
    bad = []
    target_chunks = {(idx.fqn, idx.offset, idx.index): info for idx, info in dmd.storage_data.items()}
    for idx, info in smd.storage_data.items():
        if idx.fqn == "args":
            continue  # deliberately replaced with the 104-layer checkpoint args
        target = rename_key(idx.fqn)
        dinfo = target_chunks.get((target, idx.offset, idx.index))
        if dinfo is None:
            bad.append((idx, "no target chunk"))
            continue
        if (dinfo.relative_path, dinfo.offset, dinfo.length) != (info.relative_path, info.offset, info.length):
            bad.append((idx, (info, dinfo)))
    if bad:
        print(f"FAIL chunk identity for {len(bad)} source chunks; first: {bad[:3]}")
        return 1
    print(f"PASS chunks: {len(smd.storage_data)} source chunks are 1:1 in file/offset/size")

    # 3. Exactly the expected new V keys, including mHC.
    expected_v = set()
    for layer in v_layers:
        expected_v |= expected_v_keys(layer)
    actual_v = {k for k in d if any(f"decoder.layers.{layer}." in k for layer in v_layers)}
    if actual_v != expected_v:
        print(f"FAIL V keys: missing={len(expected_v - actual_v)} extra={len(actual_v - expected_v)}")
        print(" first missing:", sorted(expected_v - actual_v)[:8])
        print(" first extra:", sorted(actual_v - expected_v)[:8])
        return 1
    assert not any("self_attention.self_attention" in k for k in actual_v), sorted(k for k in actual_v if "self_attention.self_attention" in k)[:3]
    print(f"PASS V keys: {len(actual_v)} model/optimizer tensors for layers {v_layers}")

    # 4. Parameter count increase is exactly the six V layers.
    sp, dp = model_param_count(smd), model_param_count(dmd)
    per_v = sum(
        meta.size.numel()
        for key, meta in d.items()
        if key.startswith("model.module.") and any(
            f"decoder.layers.{layer}." in key for layer in v_layers
        ) and isinstance(meta, TensorStorageMetadata)
    )
    # per_v is the six new layers; the source has no equivalent layers.
    if dp - sp != per_v:
        print(f"FAIL parameters: source={sp} dest={dp} delta={dp-sp} expected={per_v}")
        return 1
    print(f"PASS parameters: {sp} -> {dp} (+{dp-sp}, exactly the six V layers)")

    # 5. Identity initialization of the new layers.
    load_keys = []
    for layer in v_layers:
        load_keys += [
            f"model.module.decoder.layers.{layer}.self_attention.linear_proj.weight",
            f"model.module.decoder.layers.{layer}.self_attention.q_layernorm.weight",
            f"model.module.decoder.layers.{layer}.input_layernorm.weight",
        ]
        mm, _ = mixer_tensors(layer)
        load_keys += list(mm)
    sd = {k: torch.empty(d[k].size) for k in load_keys}
    dcp.load(sd, storage_reader=FileSystemReader(a.dst))
    for layer in v_layers:
        assert torch.count_nonzero(sd[f"model.module.decoder.layers.{layer}.self_attention.linear_proj.weight"]) == 0
        assert torch.all(sd[f"model.module.decoder.layers.{layer}.self_attention.q_layernorm.weight"] == 1)
        assert torch.all(sd[f"model.module.decoder.layers.{layer}.input_layernorm.weight"] == 1)
        assert torch.count_nonzero(sd[f"model.module.decoder.layers.{layer}.mhc_attn_mixer.phi"]) == 0
        assert torch.all(sd[f"model.module.decoder.layers.{layer}.mhc_attn_mixer.b_res"] == 5 * torch.eye(4))
    print("PASS identity: zero output projection, unit norms, zero phi, b_res = 5I")

    # 6. A real existing tensor is value-identical.
    if a.sample_values:
        key = next(k for k in s if k.startswith("model.module.decoder.layers.0.self_attention.q_layernorm.weight"))
        target = rename_key(key)
        pair = {key: torch.empty(s[key].size), target: torch.empty(d[target].size)}
        dcp.load(pair, storage_reader=FileSystemReader(a.src))
        # load target separately (the reader is tied to one checkpoint)
        got = {target: torch.empty(d[target].size)}
        dcp.load(got, storage_reader=FileSystemReader(a.dst))
        if not torch.equal(pair[key], got[target]):
            print(f"FAIL value mismatch for {key} -> {target}")
            return 1
        print(f"PASS value: {key} -> {target} identical")

    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
