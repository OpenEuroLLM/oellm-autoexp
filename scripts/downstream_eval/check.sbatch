#!/bin/bash
# Sanity check of one converted export: NLL on a short text (~1.5-2.5 expected; ~12 = broken
# conversion) and a greedy continuation. Submitted by eval_checkpoints.sh.
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=check
set -euo pipefail
apptainer exec --nv --bind "$CONTAINER_BIND" "$HF_EVAL_SIF" python - "$EXPORT_ROOT/$CHECK_NAME" <<'PY'
import sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
p = sys.argv[1]
tok = AutoTokenizer.from_pretrained(p)
m = AutoModelForCausalLM.from_pretrained(p, torch_dtype="auto", device_map="cuda")
text = ("The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
        "It is named after the engineer Gustave Eiffel, whose company designed and built the tower. "
        "Berlin ist die Hauptstadt der Bundesrepublik Deutschland und zugleich eines ihrer Länder.")
ids = tok(text, return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    print("dtype", m.dtype, "nll", round(m(ids, labels=ids).loss.item(), 3))
    q = tok("The capital of France is", return_tensors="pt").input_ids.cuda()
    print(repr(tok.decode(m.generate(q, max_new_tokens=20, do_sample=False)[0])))
PY
