#!/bin/bash
# Build a patched copy of the native signed_kda codebase (experiment 018's MN5 copy) for the recall-surgery
# interventions: <dst>/{lm_scaling,fla,native_megatron} with megatron_ext/interventions.py added and the hybrid
# branch of megatron_ext/specs.py routed through it (SKDA_INTERVENTION; unset = the original stack, byte for
# byte the same specs). Run on MN5 from a synced snapshot:
#   bash scripts/korbi/recall_surgery/native_patch/make_code_dir.sh /gpfs/scratch/ehpc533/hybrid_nucleus/recall_surgery/code/v1
set -euo pipefail
SRC=${SRC:-/gpfs/projects/ehpc533/users/korbip/ckda_retrieval_018/source}
DST=${1:?destination code dir}
HERE=$(cd "$(dirname "$0")" && pwd)
[ -e "$DST" ] && { echo "$DST exists (code dirs are immutable: pick a new name)" >&2; exit 1; }
mkdir -p "$DST"
cp -a "$SRC/lm_scaling" "$SRC/fla" "$DST/"
[ -d "$SRC/native_megatron" ] && cp -a "$SRC/native_megatron" "$DST/"
cp "$HERE/interventions.py" "$HERE/surgery.py" "$DST/lm_scaling/megatron_ext/"
python3 - "$DST/lm_scaling/megatron_ext/specs.py" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = """        _per_layer_sharded_keys()
        return TransformerBlockSubmodules(layer_specs=specs,"""
new = """        # recall surgery (scripts/korbi/recall_surgery/native_patch): SKDA_INTERVENTION arms
        from .interventions import apply as _recall_surgery_apply
        specs = _recall_surgery_apply(specs, config, arm, HYBRID_EVERY)
        _per_layer_sharded_keys()
        return TransformerBlockSubmodules(layer_specs=specs,"""
assert s.count(old) == 1, "specs.py hook point not found"
s = s.replace(old, new)
old2 = """    if arm in LINEAR_ARMS:
        return _mixer_layer_spec(config, arm)"""
new2 = """    if arm in LINEAR_ARMS:
        # recall surgery: a pure stack with SKDA_INTERVENTION becomes per-layer specs (same parameter keys)
        from .interventions import active as _recall_surgery_active, apply as _recall_surgery_apply
        if _recall_surgery_active():
            sibling = {v: k for k, v in HYBRID_OF.items()}[arm]
            specs = _recall_surgery_apply([_mixer_layer_spec(config, arm)] * config.num_layers, config,
                                          sibling, None)
            _per_layer_sharded_keys()
            return TransformerBlockSubmodules(layer_specs=specs, layer_norm=LayerNormImpl)
        return _mixer_layer_spec(config, arm)"""
assert s.count(old2) == 1, "specs.py linear hook point not found"
open(p, "w").write(s.replace(old2, new2))
EOF
find "$DST" -name __pycache__ -prune -exec rm -rf {} +
( cd "$DST" && find . -type f -name "*.py" | sort | xargs sha256sum ) > "$DST/CODE_SHA256"
echo "code dir $DST ready ($(wc -l < "$DST/CODE_SHA256") python files)"
