#!/bin/bash
# Paired loss check of a recall-surgery HF export: the NATIVE Megatron model (native_loss.py through ladder17b_train.sh
# -- the training CLI, the arm's code dir and SKDA_INTERVENTION) and the HF export (export_native.py + hf_branch) score
# the SAME fixed token batch (the first N x 4097 validation tokens). FAILS (exit 1) when the means differ by more than
# TOL, or when the export drops / misplaces any tensor (export_native.py's strict accounting and strict load).
# Environment: SKDA_ARM, CKPT (a run's ckpt dir, with latest_checkpointed_iteration.txt), CODE_DIR, SKDA_INTERVENTION,
#   OUT, N (default 8), TOL (default 0.01), CKPT_STEP (the iteration to check; default: the latest -- pin it while a run
#   is still saving into CKPT)
set -uo pipefail
: "${SKDA_ARM:?}" "${CKPT:?}" "${OUT:?}"
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
BASE_CODE_DIR=${BASE_CODE_DIR:-/gpfs/projects/ehpc533/users/korbip/ckda_retrieval_018/source}
CODE_DIR=${CODE_DIR:-$BASE_CODE_DIR}; N=${N:-8}; TOL=${TOL:-0.01}
HYBRID_ENV=${HYBRID_ENV:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}
mkdir -p "$OUT"
STEP=${CKPT_STEP:-$(cat "$CKPT/latest_checkpointed_iteration.txt")}
ITER_DIR=$CKPT/iter_$(printf %07d "$STEP")
[ -d "$ITER_DIR" ] || { echo "[paired] FAIL: no $ITER_DIR"; exit 1; }
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TOKENIZERS_PARALLELISM=false
fla=$BASE_CODE_DIR; [ -d "$CODE_DIR/fla" ] && fla=$CODE_DIR
EVAL_PP="$HERE/notitan:$CODE_DIR:$CODE_DIR/lm_scaling:$fla:$HERE/compat:$HERE:$HYBRID_ENV/pylibs_flag:$HYBRID_ENV/pylibs_hfx:$HYBRID_ENV/pylibs_ep"

echo "[paired] 1/3 export $ITER_DIR (arm $SKDA_ARM, intervention '${SKDA_INTERVENTION:-}')"
rm -rf "$OUT/export"
PYTHONPATH=$EVAL_PP python "$HERE/export_native.py" --run "$ITER_DIR" --arm "$SKDA_ARM" --out "$OUT/export" \
  --intervention "${SKDA_INTERVENTION:-}" || { echo "[paired] FAIL: export"; exit 1; }

echo "[paired] 2/3 native Megatron loss"
rm -f "$OUT/native.json"
FFN=$(python3 -c "import json;print(json.load(open('$OUT/export/config.json'))['intermediate_size'])")
SRC_CKPT=$CKPT SAVE_DIR=$OUT/nosave LOAD_MODE=model_only STEPS=1 GBS=128 MBS=1 FFN=$FFN NPROC=1 \
  ENTRY=$HERE/native_loss.py EXTRA_ARGS="--skip-train --ckpt-step $STEP" RECALL_TOKENS=$OUT/tokens.npy RECALL_N=$N \
  RECALL_NATIVE_OUT=$OUT/native.json SLURM_NNODES=1 SLURM_NODEID=0 \
  bash "$HERE/ladder17b_train.sh" || { echo "[paired] FAIL: native"; exit 1; }
[ -f "$OUT/native.json" ] || { echo "[paired] FAIL: no native result"; exit 1; }

echo "[paired] 3/3 HF loss"
PYTHONPATH=$EVAL_PP python - "$OUT" "$TOL" <<'PY'
import json, sys
import numpy as np, torch
out, tol = sys.argv[1], float(sys.argv[2])
import export_hf, hf_branch
hf_branch.install()
model, cfg = export_hf.load_exported(f"{out}/export")
model = model.cuda()
tokens = torch.from_numpy(np.load(f"{out}/tokens.npy")).cuda()
losses = []
with torch.no_grad():
    for row in tokens:
        logits = model(input_ids=row[None, :-1]).logits.float()
        losses.append(float(torch.nn.functional.cross_entropy(logits[0], row[1:])))
nat = json.load(open(f"{out}/native.json"))
d = [h - n for h, n in zip(losses, nat["per_sequence"])]
res = {"hf": losses, "native": nat["per_sequence"], "hf_mean": float(np.mean(losses)), "native_mean": nat["mean"],
       "delta_mean": float(np.mean(losses) - nat["mean"]), "max_abs_delta": float(np.max(np.abs(d))),
       "branches": (getattr(cfg, "recall_branches", None) or {}).get("layers"), "tol": tol}
json.dump(res, open(f"{out}/paired.json", "w"), indent=2)
print("[paired]", json.dumps({k: v for k, v in res.items() if k not in ("hf", "native")}))
if abs(res["delta_mean"]) > tol or res["max_abs_delta"] > 5 * tol:
    print("[paired] FAIL: HF and native disagree"); sys.exit(1)
print("[paired] PASS")
PY
