"""The converted tree against the layout of a model trained with that config: every tensor the template has, with
its shape / dtype, and nothing extra but the old mHC / V-gate tensors.   compare_keys.py <template> <converted>"""
import re
import sys


def load(path):
    out = {}
    for line in open(path):
        key, kind, *rest = line.split(" ", 2)
        if kind == "tensor":
            out[key] = rest[0].strip()
    return out


tpl, conv = load(sys.argv[1]), load(sys.argv[2])
missing = [k for k in tpl if k not in conv]
wrong = [(k, tpl[k], conv[k]) for k in tpl if k in conv and tpl[k] != conv[k]]
old = re.compile(r".*(mhc_attn_mixer|mhc_mlp_mixer|mhc_mixer)\.(phi|b_pre|b_post|b_res|alpha_pre|alpha_post|alpha_res)\b.*"
                 r"|.*\.v_gate\.(out_gate|res_gate)\b.*|rng_state.*")
extra = [k for k in conv if k not in tpl and not old.fullmatch(k)]
print(f"{len(tpl)} template tensors: {len(missing)} missing, {len(wrong)} shape / dtype differ, "
      f"{len(extra)} unexpected extra")
for x in (missing[:10], wrong[:10], extra[:10]):
    for item in x:
        print("  ", item)
sys.exit(1 if missing or wrong or extra else 0)
