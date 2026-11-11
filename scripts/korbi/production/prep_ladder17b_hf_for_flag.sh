#!/bin/bash
# Make the copied jupiter 1.7B ladder exports loadable on MareNostrum 5 (goal 2026-11-11, docs/
# recall_surgery_ambitious.md §3/§6).
#
# The ladder exports carry `model_type: signed_kda` + `architectures: [SignedKDAForCausalLM]` and no code,
# because on jupiter the classes are registered by the FLA fork. `lm_scaling/hf_release/README.md` documents
# the release rename (signed_kda -> complex_kda, SignedKDAForCausalLM -> ComplexKDAForCausalLM, weights byte
# for byte), and `modeling_complex_kda.py` mirrors the exporter's module tree name for name -- so the same
# rename, plus the two code files next to the weights, makes the export loadable with `trust_remote_code`.
#
#   prep_ladder17b_hf_for_flag.sh [<root>]      # default $R/hf_models/ladder17b
#
# Idempotent: rewrites config.json from config.json.orig (created once) and relinks the code files.
set -euo pipefail
R=${R:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}
ROOT=${1:-$R/hf_models/ladder17b}
CODE=$ROOT/_code
[ -d "$CODE" ] || { echo "no $CODE (copy modeling_complex_kda.py + configuration_complex_kda.py there)" >&2; exit 1; }

python3 - "$ROOT" "$CODE" <<'PY'
import json, os, shutil, sys

root, code = sys.argv[1], sys.argv[2]
for name in sorted(os.listdir(root)):
    d = os.path.join(root, name)
    cfg = os.path.join(d, "config.json")
    if not os.path.isfile(cfg):
        continue
    orig = os.path.join(d, "config.json.orig")
    if not os.path.exists(orig):
        shutil.copy2(cfg, orig)
    with open(orig) as fh:
        c = json.load(fh)
    c["model_type"] = "complex_kda"
    c["architectures"] = ["ComplexKDAForCausalLM"]
    c["auto_map"] = {
        "AutoConfig": "configuration_complex_kda.ComplexKDAConfig",
        "AutoModelForCausalLM": "modeling_complex_kda.ComplexKDAForCausalLM",
    }
    with open(cfg, "w") as fh:
        json.dump(c, fh, indent=2)
    for f in ("modeling_complex_kda.py", "configuration_complex_kda.py"):
        src, dst = os.path.join(code, f), os.path.join(d, f)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.relpath(src, d), dst)
    print(f"{name}: model_type complex_kda, attn.layers {c.get('attn', {}).get('layers')}, "
          f"{c.get('num_hidden_layers')} layers, hidden {c.get('hidden_size')}")
PY

# the tokenizer: gpt-neox-20b, copied beside the exports by copy_ladder17b_to_mn5.sh
TOK=$ROOT/tokenizer_gpt_neox_20b
if [ -d "$TOK" ]; then
  for d in "$ROOT"/*/; do
    [ -f "$d/config.json" ] || continue
    for f in "$TOK"/*; do ln -sf "$f" "$d/$(basename "$f")"; done
  done
  echo "tokenizer linked into every export from $TOK"
else
  echo "warning: no tokenizer at $TOK" >&2
fi
