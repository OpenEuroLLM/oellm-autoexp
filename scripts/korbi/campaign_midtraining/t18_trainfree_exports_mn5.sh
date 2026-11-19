#!/bin/bash
# Training-free sparse attention: the dense t16 arm's step-3000 export with a sparse arm's attention settings, as
# hf_exports/t16_s3_dense_as_<arm>_3000 (the weights / code / tokenizer symlinked, config.json's sparse_attention
# taken from the sparse arm's export). Parameter-free selectors only (page_mean / page_maxpool / LSA). Login node:
#
#   t18_trainfree_exports_mn5.sh [arm ...]     (default: page_maxpool_pg page_mean_pg lsa_fix)
set -euo pipefail
E=${HYBRID_OUTPUT_ROOT:-/gpfs/scratch/ehpc533/hybrid_nucleus}/hf_exports
dense=$E/t16_s3_dense_3000
[ -f "$dense/.check_done" ] || { echo "dense export not checked: $dense" >&2; exit 1; }
for arm in "${@:-page_maxpool_pg page_mean_pg lsa_fix}"; do
  for a in $arm; do
    src=$E/t16_s3_${a}_3000 dst=$E/t16_s3_dense_as_${a}_3000
    mkdir -p "$dst"
    for f in "$dense"/*; do
      [ "$(basename "$f")" = config.json ] || ln -sfn "$f" "$dst/$(basename "$f")"
    done
    python3 - "$dense/config.json" "$src/config.json" "$dst/config.json" <<'EOF'
import json, sys
cfg = json.load(open(sys.argv[1]))
sa = json.load(open(sys.argv[2]))["sparse_attention"]
if any(k in sa.get("page_sparse_selector", "") for k in ("nsa", "csa2", "indexer")) or sa.get("page_sparse_lsa_latent"):
    raise SystemExit(f"{sys.argv[2]}: a selector with trained parameters")
cfg["sparse_attention"] = sa
json.dump(cfg, open(sys.argv[3], "w"), indent=2)
print(sys.argv[3], "<-", sa["page_sparse_selector"])
EOF
    touch "$dst/.check_done"
  done
done
