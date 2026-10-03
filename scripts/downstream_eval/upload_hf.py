#!/usr/bin/env python3
"""Publish converted exports to the Hub, packaged like openeurollm/prelude.

    SITE=jupiter MODEL=32b_dense python3 upload_hf.py --repo openeurollm/<name> b1_174k
    ... --execute           actually create the branch and upload (default is a dry run)

One branch per checkpoint, named after the export (prelude uses
`anneal300b_iter_0955200`); `main` is left alone. Staging uses hard links, so a
64 GB export is staged in milliseconds and costs no extra space.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

EXPORT_ROOT = Path(os.environ.get("EVAL_EXPORT_ROOT", ""))
WORK_ROOT = Path(os.environ.get("EVAL_WORK_ROOT", ""))

# The eval container (transformers 4.53) cannot parse a 5.x tokenizer.json, so the exports ship
# without it. Anyone on a current transformers expects it, so it goes back in for publication.
TOKENIZER_JSON = WORK_ROOT / "tokenizer" / "tokenizer.json"

README = """---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
  - openeurollm
---

# {name}

Base checkpoint from the OpenEuroLLM 32B dense run (`{run}`, iteration {iter}).
Converted from Megatron `torch_dist` with Megatron-Bridge (Qwen3 layout).

| | |
|---|---|
| parameters | 32B dense, 64 layers, 64 heads (8 KV) |
| context | 4096 |
| vocabulary | 262144 (`oellm_tokenizer_256k`) |
| precision | bfloat16 |

## Notes

* Research checkpoint: no instruction tuning, no alignment, no safety filtering.
* The training data contains no `<bos>`, so that embedding is barely trained even though
  `add_bos_token` is true. Few-shot scores are unaffected; 0-shot generation can be sensitive.
* `pad_token` is `<eos>`: the embedding has exactly 262144 rows, so the tokenizer's `<pad>`
  (id 262144) does not exist in this model and is omitted.
"""


def stage(name: str, out: Path) -> Path:
    src = EXPORT_ROOT / name
    if not (src / "config.json").is_file():
        sys.exit(f"error: no export {src}")
    dst = out / name
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in sorted(src.iterdir()):
        if f.name.startswith("chat_template"):  # Qwen3's template, kept out of the exports
            continue
        try:  # hard link the weights; copy the small files
            os.link(f, dst / f.name)
        except OSError:
            shutil.copy2(f, dst / f.name)
    if TOKENIZER_JSON.is_file():
        shutil.copy2(TOKENIZER_JSON, dst / "tokenizer.json")
    else:
        print(f"warning: {TOKENIZER_JSON} is missing; publishing without tokenizer.json")
    run, _, it = name.rpartition("_")
    (dst / "README.md").write_text(README.format(name=name, run=run, iter=it))
    validation = src.with_name(src.name + ".validation.json")  # written by validate_hf.py
    if validation.is_file():
        shutil.copy2(validation, dst / "validation.json")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", help="export names under EXPORT_ROOT")
    ap.add_argument("--repo", required=True, help="e.g. openeurollm/oellm-32b-dense")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--execute", action="store_true", help="without this, only stage and report")
    ap.add_argument("--staging", default=str(WORK_ROOT / "hf_publish"))
    a = ap.parse_args()

    out = Path(a.staging)
    staged = [stage(n, out) for n in a.names]
    for d in staged:
        size = sum(f.stat().st_size for f in d.iterdir())
        print(
            f"{d.name:14s} {size / 1e9:6.1f} GB  {len(list(d.iterdir()))} files -> "
            f"{a.repo} @ branch {d.name}"
        )
    if not a.execute:
        print("\ndry run; re-run with --execute to create the branches and upload")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(a.repo, repo_type="model", private=a.private, exist_ok=True)
    for d in staged:
        api.create_branch(a.repo, branch=d.name, exist_ok=True)
        api.upload_folder(
            repo_id=a.repo, folder_path=str(d), revision=d.name, commit_message=f"Add {d.name}"
        )
        print(f"uploaded https://huggingface.co/{a.repo}/tree/{d.name}")


if __name__ == "__main__":
    main()
