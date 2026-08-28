#!/usr/bin/env python3
"""Turn a Megatron command line back into this repo's YAML config, and diff two
of them.

The forward direction (config -> argv) lives in oellm_autoexp/backends/megatron_args.py and is
driven by MEGATRON_ARG_METADATA / MEGATRON_ACTION_SPECS. This script INVERTS that mapping, so a
command line recovered from a foreign run (a log, a `ps` dump, a config-*.yaml argv block) can be
compared against a config in the repo rather than eyeballed.

Two input shapes are accepted and auto-detected:
  * a YAML-ish list of argv entries, one per line, `- --flag` / `- 'value'` (what a dumped
    config's argv block looks like)
  * a raw command line, possibly backslash-continued (what a rendered .sbatch contains)

Usage:
    python3 scripts/korbi/megatron_args_to_yaml.py args.txt
    python3 scripts/korbi/megatron_args_to_yaml.py args.txt --against run.sbatch
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from oellm_autoexp.backends.megatron.cli_metadata import (  # noqa: E402
    MEGATRON_ACTION_SPECS,
    MEGATRON_ARG_METADATA,
)

# Keys that say nothing about the model or its speed — paths, run identity, logging sinks.
# Diffing them is pure noise when the question is "did these two runs compute the same thing".
NOISE = {
    "tensorboard_dir",
    "wandb_save_dir",
    "wandb_exp_name",
    "wandb_project",
    "wandb_entity",
    "save",
    "load",
    "data_cache_path",
    "non_persistent_global_ckpt_dir",
    "tokenizer_model",
    "data_args_path",
    "config_logger_dir",
}


def _flag_index() -> dict[str, tuple[str, object]]:
    """--option-string -> (config_key, ActionSpec)."""
    idx = {}
    for key, spec in MEGATRON_ACTION_SPECS.items():
        for opt in spec.option_strings:
            idx[opt] = (key, spec)
    return idx


def tokenize(text: str) -> list[str]:
    """Recover argv from either a YAML list block or a raw (backslash-
    continued) command line."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if sum(ln.startswith("- ") for ln in lines) > len(lines) / 2:
        out = []
        for ln in lines:
            v = ln[2:].strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                v = v[1:-1]
            out.append(v)
        return out
    # raw command line: drop continuations, then split on whitespace
    flat = text.replace("\\\n", " ")
    flat = re.sub(r"\\$", "", flat, flags=re.M)
    toks = flat.split()
    if "pretrain_gpt.py" in flat:  # keep only what follows the launcher script
        for i, t in enumerate(toks):
            if t.endswith("pretrain_gpt.py"):
                toks = toks[i + 1 :]
                break
    # a rendered sbatch wraps the whole srun payload in single quotes, so the last token
    # carries a stray closing quote -- strip shell quoting from both ends.
    return [t.rstrip("\\").strip("'\"") for t in toks if t.strip("\\'\"")]


def cast(key: str, raw: str):
    meta = MEGATRON_ARG_METADATA.get(key)
    t = getattr(meta, "arg_type", None) if meta else None
    if t is bool:
        return raw.lower() in ("1", "true", "yes")
    try:
        if t is int:
            return int(raw)
        if t is float:
            return float(raw)
    except ValueError:
        pass
    return raw


def parse(text: str) -> dict:
    idx = _flag_index()
    toks = tokenize(text)
    cfg, unknown, i = {}, [], 0
    while i < len(toks):
        tok = toks[i]
        if not tok.startswith("--"):
            i += 1
            continue
        hit = idx.get(tok)
        if hit is None:
            unknown.append(tok)
            i += 1
            continue
        key, spec = hit
        if spec.action_type in ("store_true", "store_false"):
            cfg[key] = spec.const
            i += 1
        else:
            vals = []
            j = i + 1
            while j < len(toks) and not toks[j].startswith("--"):
                vals.append(toks[j].replace("\\", ""))
                j += 1
            cfg[key] = cast(key, vals[0]) if len(vals) == 1 else [cast(key, v) for v in vals]
            i = j
    if unknown:
        print(
            f"# WARNING: {len(unknown)} unrecognised flag(s): {', '.join(unknown)}", file=sys.stderr
        )
    return cfg


def to_yaml(cfg: dict) -> str:
    lines = ["# @package _global_", "backend:", "  megatron:"]
    for k in sorted(cfg):
        v = cfg[k]
        if isinstance(v, bool):
            s = "true" if v else "false"
        elif isinstance(v, str):
            s = (
                '"' + v.replace("\\", "\\\\").replace("*", "\\\\*").replace("|", "\\\\|") + '"'
                if k == "pipeline_model_parallel_layout"
                else f'"{v}"'
            )
        else:
            s = str(v)
        lines.append(f"    {k}: {s}")
    return "\n".join(lines)


def derived(cfg: dict, nodes: int = 1024, gpus_per_node: int = 4) -> dict:
    """The quantities that actually decide throughput but never appear on the
    command line."""
    tp = cfg.get("tensor_model_parallel_size", 1)
    pp = cfg.get("pipeline_model_parallel_size", 1)
    cp = cfg.get("context_parallel_size", 1)
    mbs = cfg.get("micro_batch_size", 1)
    gbs = cfg.get("global_batch_size", 0)
    layout = cfg.get("pipeline_model_parallel_layout")
    groups = layout.count("|") + 1 if isinstance(layout, str) else None
    world = nodes * gpus_per_node
    dp = world // (tp * pp * cp) if tp * pp * cp else 0
    m = gbs // (mbs * dp) if dp and mbs else 0
    vpp = groups // pp if groups and pp and groups % pp == 0 else None
    out = {"world": world, "DP": dp, "M": m, "groups": groups, "VPP": vpp}
    if vpp and m:
        out["bubble"] = f"{(pp - 1) / (vpp * m) * 100:.1f}%"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("args_file")
    ap.add_argument("--against", help="second command line / sbatch to diff against")
    ap.add_argument("--nodes", type=int, default=1024)
    ap.add_argument("--all", action="store_true", help="include path/identity keys in the diff")
    a = ap.parse_args()

    cfg = parse(Path(a.args_file).read_text())
    print(to_yaml(cfg))
    print("\n# derived: " + ", ".join(f"{k}={v}" for k, v in derived(cfg, a.nodes).items()))

    if not a.against:
        return

    other = parse(Path(a.against).read_text())
    skip = set() if a.all else NOISE
    keys = sorted((set(cfg) | set(other)) - skip)
    rows = [
        (k, cfg.get(k, "<unset>"), other.get(k, "<unset>"))
        for k in keys
        if cfg.get(k, "<unset>") != other.get(k, "<unset>")
    ]

    print(f"\n# ==== diff: {Path(a.args_file).name}  vs  {Path(a.against).name} ====")
    if not rows:
        print("# identical on every non-path key")
    w = max((len(r[0]) for r in rows), default=10)
    for k, mine, theirs in rows:
        ms, ts = str(mine), str(theirs)
        if k == "pipeline_model_parallel_layout":
            ms, ts = f"{ms.count('|') + 1} groups", f"{ts.count('|') + 1} groups"
        print(f"  {k:<{w}}  {ms:<28}  {ts}")

    d1, d2 = derived(cfg, a.nodes), derived(other, a.nodes)
    print("\n# derived deltas")
    for k in d1:
        if d1[k] != d2[k]:
            print(f"  {k:<{w}}  {str(d1[k]):<28}  {d2[k]}")


if __name__ == "__main__":
    main()
