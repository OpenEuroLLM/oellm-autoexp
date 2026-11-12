#!/usr/bin/env python3
"""Readouts of the v4 broad session (docs/recall_surgery_ambitious.md §10): HF export + ckdab_fast (CoQA /
Winogrande / SQuAD v2) + ckdab_belebele (24 languages) for every arm x step, and the table of the results.

Arms (run dirs <root>/<arm>, saves <root>/<arm>/{ckpt_eval,ckpt,ckpt_rolling}/iter_N; decay copies <root>/dec_<arm>):
  ctrl, kv8_lr4, post4_lr4, kv8_post4_lr4, conv_lr4, kv16_lr8, k6_L49

    # what would be launched (DRY by default: prints the commands, writes / submits nothing)
    python scripts/korbi/recall_surgery/broad_readout.py launch --it 33200 --steps 200,400,800 \\
        --pool /gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p264 --snapshot autoexp_v4broad1
    # ... the same with --execute (on the MN5 login node; runs scripts/evals/launch.py per export)
    # the saves present instead of a step list:      --discover
    # the decay copies (dec_<arm>, their last save):  --dec --discover
    python scripts/korbi/recall_surgery/broad_readout.py table --it 33200 --steps 200,400,800
    python scripts/korbi/recall_surgery/broad_readout.py table --dirs <result dir name> ... --ref <name>
    python scripts/korbi/recall_surgery/broad_readout.py selftest

One launch per export (spec config/evals/ckdab_fast_mn5.yaml with rows.suite_file =
config/eval_tasks/ckdab_fast_belebele.yaml): the export job is made once -- two launches of an export that is
not ready yet would each add an export job writing the same directory. The export job runs
scripts/korbi/hf_export/export_run.sh <run dir> <step> (ckpt/iter_N, the run's newest current_*.yaml).

Per arm:
  * kv8 / kv16: num_key_value_heads from the run's num_query_groups (export_ambitious).
  * post4_lr4 / kv8_post4_lr4: additive branches -> ckda_attn (export_ambitious.branch_add; the branch's KV
    groups must equal the stack's: 2 == 2 / 8 == 8).
  * conv_lr4: crossfade branches export only past the hand-over (branch_fold: step - start >= hold + ramp,
    complex_kda weighted 0); earlier saves are skipped with a note.
  * k6_L49: num_experts_per_tok from the run's top-k; the training-time Engram output scale (Megatron
    engram_output_scale(_layers)) is NOT in the HF export, so the eval sets the vLLM switches
    OELLM_ENGRAM_OUT_SCALE=0.0 / OELLM_ENGRAM_SCALE_LAYERS=49 (vLLM 63b5fcc8fe: the Engram block's layer_idx =
    the Megatron hybrid-pattern index, as the §3.3 inference run ..._23500_eng0_L49) through runtime.env.

Priorities: the pool starts the highest pool_priority first (pool.py: queue sorted by -priority). Main run 1000,
the broad arms above the evals; the lm-eval units take --priority (default 60), the export job is fixed at 50
by scripts/evals/launch.py. Low-priority 1-node evals can still backfill nodes a crashed arm frees.

Table: per arm x step CoQA EM / F1, Winogrande, SQuAD v2 F1 / HasAns F1 / NoAns F1, belebele mean, and the
delta to ctrl at the same step (belebele: mean per-language difference on the shared languages, up / down).
Parsing as /gpfs/scratch/ehpc533/users/korbip/rows3.py and bele.py (the newest results*.json per row, merged
first; percentages; belebele acc_norm else acc).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
import statistics as st
import subprocess
import sys
from pathlib import Path

ARMS = ("ctrl", "kv8_lr4", "post4_lr4", "kv8_post4_lr4", "conv_lr4", "kv16_lr8", "k6_L49")
ROOT = "/gpfs/scratch/ehpc533/hybrid_nucleus/production/v4_broad"
HYBRID_ENV = "/gpfs/projects/ehpc533/users/korbip/hybrid_v019"   # $R
OUT = "/gpfs/scratch/ehpc533/hybrid_nucleus/flag_evals_oellm"
EXPORTS = "/gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports"
POOL = "/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p264"
SPEC = "config/evals/ckdab_fast_mn5.yaml"
SUITES = {
    "all": "config/eval_tasks/ckdab_fast_belebele.yaml",
    "fast": "config/eval_tasks/ckdab_fast.yaml",
    "belebele": "config/eval_tasks/ckdab_belebele.yaml",
}
N_ROWS = {"all": 27, "fast": 3, "belebele": 24}
# runtime env per arm (the eval-time switches a training change needs because the export does not carry it)
ARM_ENV = {"k6_L49": {"OELLM_ENGRAM_OUT_SCALE": "0.0", "OELLM_ENGRAM_SCALE_LAYERS": "49"}}
# crossfade hand-over of conv_lr4 when the run's config cannot be read (the suite's schedule)
CONV_DEFAULT = {"hold": 0, "ramp": 400}

FAST_ROWS = (("coqa_0", "coqa", ("em,none", "f1,none")),
             ("winogrande_0", "winogrande", ("acc,none",)),
             ("squadv2_10", "squadv2", ("exact,none", "f1,none", "HasAns_f1,none", "NoAns_f1,none")))


# ---------------------------------------------------------------- names / steps


def run_name(arm: str, dec: bool) -> str:
    return f"dec_{arm}" if dec else arm


def export_name(prefix: str, arm: str, step: int, dec: bool) -> str:
    return f"{prefix}_{run_name(arm, dec)}_fp8_{step}"


# where a save can sit, in export_run.sh's search order: the arms write rolling saves (+200 / +600, latest only,
# ckpt_rolling) and persistent ones (+400 / +800, ckpt); a rolling save is hard-linked into ckpt_eval before its
# export (protect_rolling), so the next rolling save cannot delete it under the export
SAVE_DIRS = ("ckpt_eval", "ckpt", "ckpt_rolling")


def find_save(run_dir: str, step: int) -> str | None:
    for sub in SAVE_DIRS:
        d = os.path.join(run_dir, sub, f"iter_{step:07d}")
        if os.path.isfile(os.path.join(d, ".metadata")):
            return d
    return None


def saved_steps(run_dir: str) -> list[int]:
    out = set()
    for sub in SAVE_DIRS:
        for p in glob.glob(os.path.join(run_dir, sub, "iter_*")):
            m = re.fullmatch(r"iter_(\d+)", os.path.basename(p))
            if m and os.path.isfile(os.path.join(p, ".metadata")):
                out.add(int(m.group(1)))
    return sorted(out)


def protect_rolling(run_dir: str, step: int, execute: bool) -> None:
    """A save found only in ckpt_rolling: hard-link it into ckpt_eval/iter_N (no data copied; free it after
    the export with rm -r <run>/ckpt_eval/iter_N)."""
    src = find_save(run_dir, step)
    if src is None or os.path.basename(os.path.dirname(src)) != "ckpt_rolling":
        return
    dst = os.path.join(run_dir, "ckpt_eval", f"iter_{step:07d}")
    print(f"mkdir -p {shlex.quote(os.path.dirname(dst))} && cp -al {shlex.quote(src)} {shlex.quote(dst)}")
    if execute:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        subprocess.check_call(["cp", "-al", src, dst])


def resolve_steps(a, run_dir: str) -> list[int]:
    """Absolute steps: --steps are relative to --it (IT + 200 ...), --abs-steps absolute, --discover the saves
    present after the start (the seeded iter_IT itself excluded)."""
    if a.discover:
        lo = a.it if a.it is not None else -1
        return [s for s in saved_steps(run_dir) if s > lo]
    steps = [int(x) for x in a.abs_steps.split(",")] if a.abs_steps else []
    if a.steps:
        if a.it is None:
            sys.exit("--steps are relative to the start: give --it (or --abs-steps)")
        steps += [a.it + int(x) for x in a.steps.split(",")]
    return sorted(set(steps))


def _megatron_cfg(run_dir: str) -> dict | None:
    """backend.megatron of the run's newest current_*.yaml (what export_run.sh exports with)."""
    try:
        import yaml
    except ImportError:
        return None
    files = sorted(glob.glob(os.path.join(run_dir, "current_*.yaml")), key=os.path.getmtime)
    if not files:
        return None
    try:
        cfg = yaml.safe_load(open(files[-1]))
    except Exception:
        return None
    m = (cfg.get("backend") or {}).get("megatron") if isinstance(cfg, dict) else None
    return m if isinstance(m, dict) else None


def conv_exportable(run_dir: str, step: int, start: int | None) -> tuple[bool, str]:
    """A crossfade branch exports (branch_fold) only once complex_kda is weighted 0."""
    m = _megatron_cfg(run_dir)
    if m and m.get("ckda_branch_crossfade"):
        s0 = int(m.get("ckda_branch_start_iteration", 0))
        hold, ramp = int(m.get("ckda_branch_alpha_hold", 25)), int(m.get("ckda_branch_alpha_ramp", 175))
        src = "run config"
    else:
        if start is None:
            return True, "no run config and no --it: not checked"
        s0, hold, ramp, src = start, CONV_DEFAULT["hold"], CONV_DEFAULT["ramp"], "suite default"
    ok = step - s0 >= hold + ramp
    return ok, (f"crossfade start {s0} hold {hold} ramp {ramp} ({src}): "
                + ("past the hand-over" if ok else f"inside the hand-over until {s0 + hold + ramp}"))


# ---------------------------------------------------------------- launch


def rows_done(name: str) -> int:
    d = Path(OUT, name, "done")
    if not d.is_dir():
        return 0
    return sum(1 for p in d.glob("*.ok") if "@" not in p.name)


def launch_cmd(a, arm: str, step: int) -> list[str]:
    name = export_name(a.export_prefix, arm, step, a.dec)
    cmd = [f"{HYBRID_ENV}/autoexp_env/bin/python", "scripts/evals/launch.py", SPEC,
           "--set", f"rows.suite_file={SUITES[a.rounds]}",
           "--set", f"export.step={step}",
           "--set", f"export.name={name}",
           "--set", f"export.run_dir={a.root}/{run_name(arm, a.dec)}",
           "--set", f"paths.pool={a.pool}",
           "--set", f"lm_eval.job.priority={a.priority}"]
    env = ARM_ENV.get(arm)
    if env:
        cmd += ["--set", "runtime.env={" + ", ".join(f'{k}: "{v}"' for k, v in env.items()) + "}"]
    if a.rounds != "all":
        cmd += ["--set", f"tag={a.rounds}"]
    if not a.execute:
        cmd += ["--dry-run"] if a.plan else []
    return cmd


def cmd_launch(a) -> int:
    snap = f"{HYBRID_ENV}/{a.snapshot}"
    on_cluster = os.path.isdir(a.root)
    rc = 0
    for arm in a.arms:
        run_dir = f"{a.root}/{run_name(arm, a.dec)}"
        for step in resolve_steps(a, run_dir):
            name = export_name(a.export_prefix, arm, step, a.dec)
            tag = f"[{run_name(arm, a.dec)} {step}]"
            if on_cluster and not a.no_check:
                if find_save(run_dir, step) is None:
                    print(f"# {tag} skipped: iter_{step:07d} not saved (yet) in {run_dir}/{{{','.join(SAVE_DIRS)}}}")
                    continue
            if arm == "conv_lr4" and not a.dec:  # a decay copy starts past the hand-over
                ok, why = conv_exportable(run_dir, step, a.it)
                if not ok:
                    print(f"# {tag} skipped: {why} -- branch_fold cannot export a mid-crossfade checkpoint")
                    continue
            if a.skip_existing:
                ready = Path(EXPORTS, name, ".downstream_ready").exists()
                done = rows_done(name)
                if ready and done >= N_ROWS[a.rounds]:
                    print(f"# {tag} skipped: export ready, {done} rows done")
                    continue
            if on_cluster:
                protect_rolling(run_dir, step, a.execute)
            cmd = launch_cmd(a, arm, step)
            print(f"cd {snap} && " + " ".join(shlex.quote(c) for c in cmd))
            if a.execute:
                r = subprocess.call(cmd, cwd=snap)
                print(f"# {tag} launch.py rc {r}")
                rc = rc or r
    if not a.execute:
        print("# DRY: nothing launched (add --execute on the MN5 login node; --plan appends launch.py --dry-run)")
    return rc


# ---------------------------------------------------------------- results


def _pct(task: str, v: float) -> float:
    """Percent: lm-eval reports coqa / winogrande / belebele as fractions, squadv2 already in percent (its
    NoAns_f1 is often < 1.5 %, so rows3.py's `v > 1.5` rule would scale it by 100)."""
    return v if task == "squadv2" else v * 100


def _newest(files: list[str]) -> str | None:
    files = sorted(files)
    merged = [f for f in files if "merged" in f]
    return (merged or files)[-1] if files else None


def fast_rows(res_dir: str) -> dict[str, float]:
    """{coqa_em, coqa_f1, wino, squad_em, squad_f1, hasans_f1, noans_f1}: percent, missing rows left out."""
    out: dict[str, float] = {}
    names = {("coqa", "em,none"): "coqa_em", ("coqa", "f1,none"): "coqa_f1",
             ("winogrande", "acc,none"): "wino", ("squadv2", "exact,none"): "squad_em",
             ("squadv2", "f1,none"): "squad_f1", ("squadv2", "HasAns_f1,none"): "hasans_f1",
             ("squadv2", "NoAns_f1,none"): "noans_f1"}
    for row, task, metrics in FAST_ROWS:
        f = _newest(glob.glob(f"{res_dir}/results/{row}/**/results*.json", recursive=True))
        if not f:
            continue
        try:
            r = json.load(open(f))["results"].get(task, {})
        except Exception:
            continue
        for m in metrics:
            if isinstance(r.get(m), (int, float)):
                out[names[(task, m)]] = _pct(task, r[m])
    return out


def belebele_rows(res_dir: str) -> dict[str, float]:
    """{belebele_<lang>_5: accuracy (0-1)}: acc_norm, else acc."""
    out = {}
    for f in glob.glob(f"{res_dir}/results/belebele_*_5/**/results*.json", recursive=True):
        row = f.split("/results/")[1].split("/")[0]
        try:
            r = json.load(open(f))["results"]
        except Exception:
            continue
        m = r.get(row.rsplit("_", 1)[0], {})
        v = m.get("acc_norm,none", m.get("acc,none"))
        if isinstance(v, (int, float)):
            out[row] = v
    return out


COLS = (("coqa_em", "CoQA EM"), ("coqa_f1", "CoQA F1"), ("wino", "Wino"), ("squad_f1", "SQuAD F1"),
        ("hasans_f1", "HasAns F1"), ("noans_f1", "NoAns F1"))


def _fmt(v: float | None, signed: bool = False) -> str:
    if v is None:
        return "-"
    return f"{v:+.2f}" if signed else f"{v:.2f}"


def table(entries: list[tuple[str, str]], ref_of: dict[str, str | None], out_root: str) -> list[str]:
    """entries: (label, result dir name); ref_of[label]: the reference's dir name (or None)."""
    cache: dict[str, tuple[dict, dict]] = {}

    def load(name):
        if name not in cache:
            d = os.path.join(out_root, name)
            cache[name] = (fast_rows(d), belebele_rows(d)) if os.path.isdir(d) else ({}, {})
        return cache[name]

    w = max([34] + [len(lab) for lab, _ in entries])
    head = f"{'arm / step':{w}s} | " + " | ".join(f"{c:>15s}" for _, c in COLS) + f" | {'belebele':>8s} | belebele vs ref"
    lines = [head, "-" * len(head)]
    for label, name in entries:
        fast, bele = load(name)
        ref = ref_of.get(label)
        rfast, rbele = load(ref) if ref and ref != name else ({}, {})
        cells = []
        for k, _ in COLS:
            v = fast.get(k)
            d = (v - rfast[k]) if (v is not None and k in rfast) else None
            cells.append(f"{_fmt(v):>7s} ({_fmt(d, True):>6s})" if ref and ref != name else f"{_fmt(v):>15s}")
        bmean = st.mean(bele.values()) * 100 if bele else None
        shared = [r for r in bele if r in rbele]
        if shared:
            diff = [bele[r] - rbele[r] for r in shared]
            bcmp = (f"{st.mean(diff) * 100:+.2f} (n {len(shared)}, {sum(x > 0 for x in diff)} up / "
                    f"{sum(x < 0 for x in diff)} down)")
        else:
            bcmp = "-" if ref and ref != name else "(reference)"
        if not fast and not bele:
            bcmp = "no results"
        lines.append(f"{label:{w}s} | " + " | ".join(cells) + f" | {_fmt(bmean):>8s} | {bcmp}")
    lines.append("(delta) = minus the reference at the same step (ctrl); noise: Winogrande +-1.4, CoQA +-~1.5 F1, "
                 "SQuAD v2 +-~1 F1; belebele n = languages present in both")
    return lines


def cmd_table(a) -> int:
    if a.dirs:
        entries = [(d, d) for d in a.dirs]
        ref_of = {d: a.ref for d in a.dirs}
    else:
        entries, ref_of = [], {}
        steps_by_arm = {arm: resolve_steps(a, f"{a.root}/{run_name(arm, a.dec)}") for arm in a.arms}
        for arm in a.arms:
            for step in steps_by_arm[arm]:
                name = export_name(a.export_prefix, arm, step, a.dec)
                label = f"{run_name(arm, a.dec)} {step}"
                entries.append((label, name))
                ref_of[label] = export_name(a.export_prefix, a.ref_arm, step, a.dec)
    for line in table(entries, ref_of, a.out):
        print(line)
    return 0


# ---------------------------------------------------------------- self-check


def cmd_selftest(_a) -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        def put(name, row, task, metrics):
            p = Path(tmp, name, "results", row, "abc")
            p.mkdir(parents=True, exist_ok=True)
            (p / "results_2026-11-12T00-00-00.json").write_text(json.dumps({"results": {task: metrics}}))

        for name, base in (("ref", 0.0), ("arm", 1.0)):
            put(name, "coqa_0", "coqa", {"em,none": 0.50 + base / 100, "f1,none": 0.66 + base / 100})
            put(name, "winogrande_0", "winogrande", {"acc,none": 0.625})
            put(name, "squadv2_10", "squadv2", {"exact,none": 21.5, "f1,none": 26.9 + base,
                                                "HasAns_f1,none": 52.9, "NoAns_f1,none": 0.5})
            put(name, "belebele_deu_Latn_5", "belebele_deu_Latn", {"acc_norm,none": 0.80 + base / 100})
            put(name, "belebele_fra_Latn_5", "belebele_fra_Latn", {"acc,none": 0.82 - base / 100})
        f = fast_rows(os.path.join(tmp, "arm"))
        assert abs(f["coqa_f1"] - 67.0) < 1e-9 and abs(f["squad_f1"] - 27.9) < 1e-9 and f["wino"] == 62.5, f
        assert f["noans_f1"] == 0.5 and f["coqa_em"] == 51.0, f  # squadv2 is percent already
        b = belebele_rows(os.path.join(tmp, "arm"))
        assert set(b) == {"belebele_deu_Latn_5", "belebele_fra_Latn_5"}, b
        lines = table([("ref", "ref"), ("arm", "arm")], {"ref": "ref", "arm": "ref"}, tmp)
        assert "+1.00" in lines[3] and "1 up / 1 down" in lines[3], lines
        # names / steps
        assert export_name("v4broad", "k6_L49", 33400, False) == "v4broad_k6_L49_fp8_33400"
        assert export_name("v4broad", "ctrl", 34100, True) == "v4broad_dec_ctrl_fp8_34100"
        ns = argparse.Namespace(discover=False, it=33200, steps="200,400,800", abs_steps=None)
        assert resolve_steps(ns, tmp) == [33400, 33600, 34000]
        # conv hand-over (no run config: the suite default, ramp 400)
        assert conv_exportable(tmp, 33400, 33200)[0] is False
        assert conv_exportable(tmp, 33600, 33200)[0] is True
        # launch command: k6_L49 carries the Engram switches, every arm its own export
        la = argparse.Namespace(export_prefix="v4broad", dec=False, root=ROOT, pool=POOL, priority=60,
                                rounds="all", execute=False, plan=False)
        c = launch_cmd(la, "k6_L49", 33400)
        assert 'runtime.env={OELLM_ENGRAM_OUT_SCALE: "0.0", OELLM_ENGRAM_SCALE_LAYERS: "49"}' in c, c
        assert f"export.run_dir={ROOT}/k6_L49" in c and "export.name=v4broad_k6_L49_fp8_33400" in c
        assert not any(x.startswith("runtime.env") for x in launch_cmd(la, "ctrl", 33400))
        print("\n".join(lines))
    print("selftest ok")
    return 0


# ---------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--arms", default=",".join(ARMS), help="comma list (default: all seven)")
        p.add_argument("--it", type=int, help="the start iteration IT (the main run's checkpoint)")
        p.add_argument("--steps", help="comma list relative to --it, e.g. 200,400,800")
        p.add_argument("--abs-steps", help="comma list of absolute steps")
        p.add_argument("--discover", action="store_true", help="the saves present in <run>/ckpt* after --it")
        p.add_argument("--dec", action="store_true", help="the decay copies dec_<arm>")
        p.add_argument("--root", default=ROOT)
        p.add_argument("--export-prefix", default="v4broad")

    pl = sub.add_parser("launch", help="export + eval launches (DRY unless --execute)")
    common(pl)
    pl.add_argument("--pool", default=POOL)
    pl.add_argument("--snapshot", default="autoexp_v4broad1", help="$R/<snapshot>: the synced code to launch from")
    pl.add_argument("--priority", type=int, default=60, help="lm-eval units' pool_priority (export job: 50)")
    pl.add_argument("--rounds", choices=sorted(SUITES), default="all",
                    help="all = fast + belebele in one launch (default); fast / belebele alone (tagged)")
    pl.add_argument("--skip-existing", action="store_true", help="skip exports that are ready with every row done")
    pl.add_argument("--no-check", action="store_true", help="do not require the checkpoint to exist")
    pl.add_argument("--plan", action="store_true", help="append launch.py --dry-run (print its plan when run)")
    pl.add_argument("--execute", action="store_true", help="run the launches (MN5 login node)")

    pt = sub.add_parser("table", help="the results table")
    common(pt)
    pt.add_argument("--out", default=OUT, help="the eval results root")
    pt.add_argument("--ref-arm", default="ctrl")
    pt.add_argument("--dirs", nargs="*", help="explicit result dir names under --out instead of arms x steps")
    pt.add_argument("--ref", help="with --dirs: the reference dir name")

    sub.add_parser("selftest", help="parser / naming checks on synthetic results")
    a = ap.parse_args(argv)
    if hasattr(a, "arms"):
        a.arms = [x for x in a.arms.split(",") if x]
        bad = [x for x in a.arms if x not in ARMS]
        if bad:
            ap.error(f"unknown arms {bad} (known: {ARMS})")
    return {"launch": cmd_launch, "table": cmd_table, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
