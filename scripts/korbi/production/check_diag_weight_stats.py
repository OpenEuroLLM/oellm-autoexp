"""Offline check of Megatron's --diag-weight-stats / --diag-norm-gains (diagnostics.py, 706c88c9e).

Recomputes the per-layer weight-family and norm-gain statistics from the FULL tensors of a
checkpoint (torch DCP, no_dist) and compares them with what the run logged to TensorBoard at
iteration CKPT_ITER + 1 (the diagnostics of iteration k are collected before step k's optimizer
update, i.e. on the weights after k - 1 updates = checkpoint iter k - 1).

    singularity exec --cleanenv --bind /gpfs <sif> python3 check_diag_weight_stats.py \
        <ckpt dir>/iter_0000019 <tensorboard dir> [--megatron <Megatron-LM checkout>]

Prints one line per compared value and a summary; exit code 1 on any mismatch beyond tolerance
(sums: 1e-4 relative, min / max: exact up to the bf16 -> fp32 cast).
"""

import argparse
import re
import sys
from collections import defaultdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("tb")
    ap.add_argument("--megatron", default=None, help="Megatron-LM checkout to import diagnostics.py from")
    ap.add_argument("--rtol", type=float, default=1e-4)
    a = ap.parse_args()
    if a.megatron:
        sys.path.insert(0, a.megatron)

    import torch
    import torch.distributed.checkpoint as dcp
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from megatron.training import diagnostics as D

    it = int(re.search(r"iter_(\d+)", a.ckpt).group(1)) + 1
    reader = dcp.FileSystemReader(a.ckpt)
    md = reader.read_metadata()
    keys = [
        k
        for k, v in md.state_dict_metadata.items()
        if k.startswith("model.") and hasattr(v, "size") and "_extra_state" not in k
    ]
    sd = {k: torch.empty(md.state_dict_metadata[k].size, dtype=md.state_dict_metadata[k].properties.dtype) for k in keys}
    dcp.load(sd, storage_reader=reader, no_dist=True)

    fam_re = D._WEIGHT_FAMILY_RE
    w = defaultdict(list)  # (family, layer) -> tensors
    g = defaultdict(list)  # (gain family, layer) -> tensors
    for k, t in sd.items():
        name = k[len("model."):]
        m = re.search(r"decoder\.layers\.(\d+)\.", name)
        layer = int(m.group(1)) if m else None
        if layer is not None:
            for fam, rx in fam_re:
                if rx.search(name):
                    w[(fam, layer)].append(t)
                    break
        if t.dim() == 1:
            gf = D._classify_gain(name)
            if gf is not None:
                g[(gf, layer)].append(t)

    ea = EventAccumulator(a.tb, size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags()["scalars"])

    def logged(tag):
        if tag not in tags:
            return None
        vals = {e.step: e.value for e in ea.Scalars(tag)}
        return vals.get(it)

    bad, n_cmp, missing = 0, 0, 0

    def cmp(tag, want, exact=False):
        nonlocal bad, n_cmp, missing
        have = logged(tag)
        if have is None:
            missing += 1
            print(f"MISSING {tag} (want {want:.6g})")
            return
        n_cmp += 1
        tol = 1e-6 * max(1.0, abs(want)) if exact else a.rtol * max(1.0, abs(want))
        ok = abs(have - want) <= tol
        bad += 0 if ok else 1
        if not ok:
            print(f"BAD  {tag}: logged {have:.8g}, full tensors {want:.8g}")

    for (fam, layer), ts in sorted(w.items()):
        flat = torch.cat([t.reshape(-1).double() for t in ts])
        p = f"diag/weight/{fam}/layer_{layer:02d}"
        mean = float(flat.mean())
        cmp(f"{p}/mean", mean)
        cmp(f"{p}/std", float(flat.std(unbiased=False)))
        cmp(f"{p}/norm", float(flat.pow(2).sum().sqrt()))
        cmp(f"{p}/min", float(flat.min()), exact=True)
        cmp(f"{p}/max", float(flat.max()), exact=True)
        rows = [t.double().pow(2).sum(1).sqrt() for t in ts if t.dim() == 2]
        if rows:
            rn = torch.cat(rows)
            cmp(f"{p}/rownorm_max", float(rn.max()))
            cmp(f"{p}/rownorm_mean", float(rn.mean()))
    for (gf, layer), ts in sorted(g.items(), key=lambda kv: (kv[0][0], -1 if kv[0][1] is None else kv[0][1])):
        flat = torch.cat([t.reshape(-1).double() for t in ts])
        p = "diag/gain/final_norm" if layer is None else f"diag/gain/{gf}/layer_{layer:02d}"
        cmp(f"{p}/mean", float(flat.mean()))
        cmp(f"{p}/std", float(flat.std(unbiased=False)))
        cmp(f"{p}/min", float(flat.min()), exact=True)
        cmp(f"{p}/max", float(flat.max()), exact=True)

    print(f"iteration {it}: compared {n_cmp} values, {bad} mismatches, {missing} missing "
          f"({len(w)} weight-family slots, {len(g)} gain slots from {len(keys)} checkpoint tensors)")
    return 1 if bad or missing else 0


if __name__ == "__main__":
    sys.exit(main())
