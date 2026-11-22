"""Export an engram_scaling run's Megatron ``torch_dist`` checkpoint as a HF model directory.

    python -m oellm_autoexp.hf_export.engram_scaling_clariden.export_hf \\
        --run-dir output/<run> [--iter 3000] --out-dir <dir> [--engram-table-dtype bf16]

Steps:

1. the architecture from the run's resolved ``config-<job>.yaml`` (:mod:`hf_config`, with its guards);
2. the model tensors from the DCP checkpoint, one Megatron layer at a time (the optimizer state is
   never read);
3. per layer, the Megatron tensors renamed to the HF ones (the README's stage 3 table); the only
   transform is the concatenation of the GDN ``in_proj`` / ``conv1d`` sections, which the checkpoint
   stores split. Every tensor of a layer must be consumed (an unmapped one stops the export);
4. dtypes: the bf16 model weights as they are (the training forward's weights, not the optimizer's
   fp32 master copies), the router weight and expert bias in fp32, the Engram table in
   ``--engram-table-dtype`` (bf16 is lossless for inference: Megatron casts the looked-up rows to bf16);
5. the Engram token map (a ``.npy`` outside the checkpoint) into the weights, when the run used one;
6. sharded safetensors + index, ``config.json``, ``generation_config.json``, the model code
   (``oellm_engram/``), the tokenizer and ``megatron_source.json`` (provenance).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch

from ..export_q35 import megatron_args, write_config
from ..fsdp_dtensor import ShardedSafetensorsWriter
from .hf_config import hf_config, tokenizer_dir

LOGGER = logging.getLogger(__name__)
_PKG = Path(__file__).resolve().parent / "oellm_engram"
_MODEL_PREFIXES = ("embedding.", "decoder.", "output_layer.")
_LAYER = re.compile(r"^decoder\.layers\.(\d+)\.(.+)$")
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
                    "added_tokens.json")

# The order Megatron's GDN forward splits in_proj / conv1d in (gated_delta_net/gdn.py: in_proj_split_names)
GDN_IN_PROJ = ("query", "key", "value", "z", "beta", "alpha")
GDN_CONV = ("query", "key", "value")


def run_paths(run_dir: Path, iteration: int | None) -> tuple[Path, Path]:
    """(resolved config, checkpoint directory); the latest checkpointed iteration by default."""
    config = sorted(run_dir.glob("config-*.yaml"), key=lambda p: p.stat().st_mtime)[-1]
    if iteration is None:
        iteration = int((run_dir / "ckpt" / "latest_checkpointed_iteration.txt").read_text().strip())
    ckpt = run_dir / "ckpt" / f"iter_{iteration:07d}"
    if not (ckpt / ".metadata").exists():
        raise FileNotFoundError(f"{ckpt}: no DCP checkpoint")
    return config, ckpt


def iter_model_tensors(ckpt_dir: Path) -> Iterator[tuple[int | None, dict[str, torch.Tensor]]]:
    """``(None, non-layer tensors)``, then ``(i, {name within the layer: tensor})`` for every decoder
    layer in order. One DCP read per group: the metadata gives every tensor's global shape and dtype,
    so an empty template of that shape is filled with the full (TP / EP / DP-gathered) tensor."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    reader = dcp.FileSystemReader(str(ckpt_dir))
    groups: dict[int | None, dict[str, tuple[str, TensorStorageMetadata]]] = defaultdict(dict)
    for key, entry in reader.read_metadata().state_dict_metadata.items():
        if not key.startswith(_MODEL_PREFIXES) or "_extra_state" in key or not isinstance(entry, TensorStorageMetadata):
            continue
        m = _LAYER.match(key)
        groups[int(m.group(1)) if m else None][m.group(2) if m else key] = (key, entry)
    for i in [None] + sorted(k for k in groups if k is not None):
        template = {key: torch.empty(tuple(e.size), dtype=e.properties.dtype) for key, e in groups[i].values()}
        dcp.load(template, storage_reader=reader, no_dist=True)
        yield i, {name: template[key] for name, (key, _) in groups[i].items()}


def _sections(t: dict[str, torch.Tensor], name: str, parts: tuple[str, ...]) -> torch.Tensor:
    """The fused tensor ``name``: as saved, or concatenated from its named sections ``name.<part>``."""
    if name in t:
        return t.pop(name)
    missing = [p for p in parts if f"{name}.{p}" not in t]
    if missing:
        raise KeyError(f"{name}: missing sections {missing}")
    return torch.cat([t.pop(f"{name}.{p}") for p in parts], dim=0)


def convert_layer(kind: str, t: dict[str, torch.Tensor], add: Callable[[str, torch.Tensor], None],
                  table_dtype: torch.dtype, token_map: torch.Tensor | None) -> None:
    """One Megatron layer's tensors (names within ``decoder.layers.i.``) -> ``add(HF name within
    ``model.layers.i.``, tensor)``. Consumes ``t``; anything left over is an error."""
    if kind == "gdn":
        p = "self_attention."
        add("input_layernorm.weight", t.pop(p + "in_proj.layer_norm_weight"))
        add("linear_attn.in_proj.weight", _sections(t, p + "in_proj.weight", GDN_IN_PROJ))
        add("linear_attn.conv1d.weight", _sections(t, p + "conv1d.weight", GDN_CONV))
        for n in ("A_log", "dt_bias", "out_norm.weight", "out_proj.weight"):
            add(f"linear_attn.{n}", t.pop(p + n))
    elif kind == "attention":
        p = "self_attention."
        add("input_layernorm.weight", t.pop(p + "linear_qkv.layer_norm_weight"))
        add("self_attn.qkv_proj.weight", t.pop(p + "linear_qkv.weight"))
        add("self_attn.o_proj.weight", t.pop(p + "linear_proj.weight"))
        for meg, hf in (("q_layernorm", "q_norm"), ("k_layernorm", "k_norm")):
            if p + f"{meg}.weight" in t:
                add(f"self_attn.{hf}.weight", t.pop(p + f"{meg}.weight"))
    elif kind == "moe":
        add("input_layernorm.weight", t.pop("pre_mlp_layernorm.weight"))
        # fp32: Megatron routes in fp32 (moe_router_dtype); bf16 -> fp32 is exact
        add("mlp.router.weight", t.pop("mlp.router.weight").float())
        add("mlp.router.expert_bias", t.pop("mlp.router.expert_bias").float())
        add("mlp.experts_gate_up", t.pop("mlp.experts.experts.linear_fc1.weight"))
        add("mlp.experts_down", t.pop("mlp.experts.experts.linear_fc2.weight"))
        if "mlp.shared_experts.linear_fc1.weight" in t:
            add("mlp.shared_gate_up.weight", t.pop("mlp.shared_experts.linear_fc1.weight"))
            add("mlp.shared_down.weight", t.pop("mlp.shared_experts.linear_fc2.weight"))
            if "mlp.shared_experts.gate_weight" in t:
                add("mlp.shared_gate.weight", t.pop("mlp.shared_experts.gate_weight"))
    elif kind == "engram":
        add("engram.table", t.pop("table").to(table_dtype))
        for n in ("key_proj", "value_proj", "hidden_norm", "key_norm", "conv_norm", "conv1d"):
            if f"{n}.weight" in t:
                add(f"engram.{n}.weight", t.pop(f"{n}.weight"))
        if token_map is not None:  # a copy per layer: safetensors refuses tensors that share memory
            add("engram.token_map", token_map.clone())
    else:
        raise ValueError(f"unknown block type {kind!r}")
    if t:
        raise KeyError(f"{kind} layer: unmapped Megatron tensors {sorted(t)}")


def load_token_map(path: str, vocab: int) -> tuple[torch.Tensor, str]:
    """Megatron's ``engram_token_map`` (token id -> canonical id) as int64 [vocab], and its sha256."""
    raw = Path(path).read_bytes()
    tmap = np.load(path)
    if tmap.ndim != 1 or tmap.shape[0] < vocab or not np.issubdtype(tmap.dtype, np.integer):
        raise ValueError(f"{path}: expected >= {vocab} integer ids, got {tmap.dtype} {tmap.shape}")
    return torch.from_numpy(tmap[:vocab].astype(np.int64)), hashlib.sha256(raw).hexdigest()


def pad_with_eos(out_dir: Path, vocab: int) -> str | None:
    """The tokenizer's pad token -> its eos token when the pad id is outside the model's vocabulary
    (``<pad>`` is id 262144 of a 262144-row embedding: a padded batch, e.g. lm-eval's, would index past
    the embedding). Edits the copied tokenizer_config.json / special_tokens_map.json; the vocabulary is
    unchanged. Returns the replaced pad token."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(out_dir, trust_remote_code=True)
    if tok.pad_token_id is not None and tok.pad_token_id < vocab:
        return None
    old = tok.pad_token
    cfg_file, map_file = out_dir / "tokenizer_config.json", out_dir / "special_tokens_map.json"
    cfg = json.loads(cfg_file.read_text())
    cfg["pad_token"] = cfg["eos_token"]
    cfg_file.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    if map_file.exists():
        smap = json.loads(map_file.read_text())
        smap["pad_token"] = smap["eos_token"]
        map_file.write_text(json.dumps(smap, indent=2, ensure_ascii=False) + "\n")
    return old


def _git(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def export(run_dir: Path, out_dir: Path, iteration: int | None = None, engram_table_dtype: str = "bf16",
           tokenizer: str | None = None, overwrite: bool = False) -> Path:
    config_yaml, ckpt = run_paths(run_dir, iteration)
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists (--overwrite to replace it)")
        shutil.rmtree(out_dir)
    m = megatron_args(config_yaml)
    table_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[engram_table_dtype]
    LOGGER.info("%s: %s, config %s", run_dir.name, ckpt.name, config_yaml.name)

    writer = ShardedSafetensorsWriter(out_dir)
    cfg = token_map = token_map_sha = None
    for i, t in iter_model_tensors(ckpt):
        if i is None:  # embedding (tied LM head) and final norm; the architecture once the vocabulary is known
            cfg = hf_config(m, int(t["embedding.word_embeddings.weight"].shape[0]))
            cfg["engram_table_dtype"] = engram_table_dtype
            if cfg.get("engram_token_map"):
                token_map, token_map_sha = load_token_map(m["engram_token_map"], cfg["vocab_size"])
            writer.add("model.embed_tokens.weight", t.pop("embedding.word_embeddings.weight"))
            writer.add("model.norm.weight", t.pop("decoder.final_norm.weight"))
            if not cfg["tie_word_embeddings"]:
                writer.add("lm_head.weight", t.pop("output_layer.weight"))
            if t:
                raise KeyError(f"unmapped non-layer tensors {sorted(t)}")
            continue
        kind = cfg["block_types"][i]
        convert_layer(kind, t, lambda name, x: writer.add(f"model.layers.{i}.{name}", x), table_dtype, token_map)
        LOGGER.info("layer %d (%s) done", i, kind)
    if cfg is None or i != len(cfg["block_types"]) - 1:
        raise ValueError(f"{ckpt}: the checkpoint's layers do not match the pattern")
    writer.close()

    write_config(out_dir, cfg, tokenizer_dir(tokenizer or m["tokenizer_model"]))
    for f in ("configuration_oellm_engram.py", "modeling_oellm_engram.py"):
        shutil.copy2(_PKG / f, out_dir / f)
    tok = tokenizer_dir(tokenizer or m["tokenizer_model"])
    for f in _TOKENIZER_FILES:
        if (tok / f).exists():
            shutil.copy2(tok / f, out_dir / f)
    replaced_pad = pad_with_eos(out_dir, cfg["vocab_size"])
    root = Path(__file__).resolve().parents[3]
    (out_dir / "megatron_source.json").write_text(json.dumps({
        "run_dir": str(run_dir.resolve()),
        "checkpoint": str(ckpt.resolve()),
        "iteration": int(ckpt.name.split("_")[1]),
        "config": str(config_yaml.resolve()),
        "engram_token_map": {"path": m.get("engram_token_map"), "sha256": token_map_sha} if token_map_sha else None,
        "engram_table_dtype": engram_table_dtype,
        "tokenizer_pad_token_replaced_by_eos": replaced_pad,
        "oellm_autoexp_commit": _git(root),
        "megatron_commit": _git(root / "submodules" / "Megatron-LM"),
        "exported": datetime.datetime.now().isoformat(timespec="seconds"),
    }, indent=2) + "\n")
    # safetensors writes its files owner-only; the export is for the group to read
    umask = os.umask(0)
    os.umask(umask)
    for f in out_dir.iterdir():
        f.chmod(0o666 & ~umask)
    LOGGER.info("wrote %s", out_dir)
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True, help="output/<run> (config-*.yaml and ckpt/)")
    p.add_argument("--iter", type=int, help="iteration to export (default: the latest checkpointed one)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--engram-table-dtype", default="bf16", choices=["bf16", "fp32"],
                   help="bf16: inference (lossless, see the module doc); fp32: as trained, for continued training")
    p.add_argument("--tokenizer", help="tokenizer dir or hub id (default: the run's tokenizer_model)")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    export(a.run_dir, a.out_dir, a.iter, a.engram_table_dtype, a.tokenizer, a.overwrite)


if __name__ == "__main__":
    main()
