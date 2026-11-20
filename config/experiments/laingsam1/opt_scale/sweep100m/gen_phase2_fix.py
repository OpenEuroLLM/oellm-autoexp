"""Generate phase2_fix_<key>.yaml: the phase 2 jobs that have to be run again.

Usage, from the repo root (after gen_sweep100m.py):
  python3 config/experiments/laingsam1/opt_scale/sweep100m/gen_phase2_fix.py

What went wrong on 2026-11-19 and what these files rerun: README.md, section "Phase 2 reruns".
Nothing existing is overwritten: the 2x reruns get the suffix _rerun; the 4x jobs listed here
never produced a checkpoint or a score, so they keep their original names.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BUDGET = {"2x": (3084, 3855, "03:00:00"), "4x": (6168, 7710, "05:00:00")}   # branch, end (steps at batch 256), limit
N_VALID = 25600

# (batch size, lr, beta or momentum) per file.
#   cool2x : cooldown2x loaded a later checkpoint than its branch step -> cooldown + score again
#   cool4x : cooldown4x never started (Slurm failure at launch)        -> cooldown + score
#   score4x: cooldown4x finished, its scoring job was never submitted  -> score only
FIX = {
    "adam": dict(
        hp="beta",
        cool2x=[(128, "1.e-3", "0.9"), (128, "1.e-3", "0.975"), (128, "2.e-3", "0.9"), (128, "2.e-3", "0.975"),
                (256, "1.e-3", "0.975"), (256, "2.e-3", "0.975"), (512, "2.e-3", "0.9")],
        cool4x=[],
        score4x=[(256, "2.e-3", "0.9"), (512, "2.e-3", "0.975")],
    ),
    "adam_untied": dict(
        hp=None,
        cool2x=[(256, "1.e-3", None), (256, "2.e-3", None)],
        cool4x=[],
        score4x=[],
    ),
    "muon": dict(
        hp="mom",
        cool2x=[(g, lr, m) for g in (128, 256, 512) for lr in ("2.e-3", "4.e-3") for m in ("0.9", "0.975")],
        cool4x=[(256, "2.e-3", "0.975"), (256, "4.e-3", "0.975"), (512, "2.e-3", "0.975")],
        score4x=[],
    ),
}


def one(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def esc(s):
    """Interpolations inside the sweep block are resolved per job, so they are escaped."""
    return s.replace("${", "\\\\${")


def setting(hp, g, lr, v):
    s = f"backend.aux.gbs: {g}, backend.megatron.lr: {lr}"
    return s + (f", backend.aux.{hp}: {v}" if hp else "")


for key, f in FIX.items():
    t = open(os.path.join(HERE, f"phase2_{key}.yaml")).read()
    body = t[t.index("defaults:"):t.index("\nsweep:") + 1]
    name = body.split('\n  name: "', 1)[1].split('"', 1)[0]            # hyb100m_sweep_..._${stage}
    assert name.startswith("hyb100m_sweep_") and name.endswith("_${stage}"), name
    run = name[:-len("_${stage}")]
    sweep_dir = "${backend.aux.base}/experiments/opt_scale/hyb100m_sweep/"
    score_name = run.replace("hyb100m_sweep_", "hyb100m_rescore_") + "_${stage}"
    body = one(body, 'WANDB_TAGS: "jupiter,sweep,phase2,', 'WANDB_TAGS: "jupiter,sweep,phase2,phase2_fix,')
    opt = "adam" if key.startswith("adam") else key
    body = one(body, f"    opt: {opt}\n", f"    opt: {opt}\n    src: cooldown2x_rerun   # scoring jobs: the cooldown whose final checkpoint is scored\n")

    cools = [("2x", "cooldown2x_rerun", s) for s in f["cool2x"]] + [("4x", "cooldown4x", s) for s in f["cool4x"]]
    scores = ([("2x", "score2x_rerun", "cooldown2x_rerun", s) for s in f["cool2x"]] +
              [("4x", "score4x", "cooldown4x", s) for s in f["cool4x"] + f["score4x"]])

    s = f'''sweep:
  type: product
  groups:
    - type: list
      configs:
        # Cooldowns: as phase 2, but self-contained (the stable run is finished, its branch
        # checkpoint is read from disk) and without rolling checkpoints, so Megatron loads
        # exactly ckpt_step.
        - type: list
          defaults:
            backend.megatron.save_extra_steps: []
            backend.megatron.non_persistent_save_interval: null
            backend.megatron.non_persistent_ckpt_type: null
            backend.megatron.load: "{esc(sweep_dir + run)}_stable/ckpt"
            backend.megatron.ckpt_step: "\\\\${{backend.aux.start_iter}}"
            backend.megatron.override_opt_param_scheduler: true
          configs:
'''
    for x, stage, st in cools:
        br, end, lim = BUDGET[x]
        s += (f"            - {{{setting(f['hp'], *st)}, stage: {stage}, backend.aux.branch256: {br}, "
              f"backend.aux.end256: {end}, slurm.sbatch.time: \"{lim}\"}}\n")
    s += f'''        # Scoring jobs: as phase 2 (fixed validation set of {N_VALID:,} sequences). Each starts when the
        # final checkpoint of its cooldown exists.
        - type: list
          defaults:
            backend.megatron.skip_train: true
            backend.megatron.dataloader_type: single
            backend.megatron.eval_iters: "\\\\${{oc.eval:'{N_VALID} // \\\\${{backend.aux.gbs}}'}}"
            backend.megatron.save_extra_steps: []
            backend.megatron.non_persistent_save_interval: null
            backend.megatron.non_persistent_ckpt_type: null
            backend.megatron.override_opt_param_scheduler: true
            backend.megatron.ckpt_step: "\\\\${{backend.megatron.train_iters}}"
            backend.megatron.load: "{esc(sweep_dir + run)}_\\\\${{backend.aux.src}}/ckpt"
            slurm.sbatch.time: "01:00:00"
            job.name: "{esc(score_name)}"
            job.base_output_dir: "\\\\${{backend.aux.base}}/experiments/opt_scale/hyb100m_rescore/\\\\${{job.name}}"
            job.start_condition:
              class_name: FileExistsCondition
              path: "{esc(sweep_dir + run)}_\\\\${{backend.aux.src}}/ckpt/iter_\\\\${{oc.eval:'str(\\\\${{backend.megatron.train_iters}}).zfill(7)'}}/.metadata"
          configs:
'''
    for x, stage, src, st in scores:
        br, end, _ = BUDGET[x]
        s += (f"            - {{{setting(f['hp'], *st)}, stage: {stage}, backend.aux.src: {src}, "
              f"backend.aux.branch256: {br}, backend.aux.end256: {end}}}\n")

    nc, ns = len(cools), len(scores)
    head = "\n".join([
        "# @package _global_",
        "#",
        f"# sweep100m phase 2 reruns / {key}: the jobs of phase2_{key}.yaml that did not give a valid result",
        "# on 2026-11-19 (README.md, section \"Phase 2 reruns\"). Same settings as phase 2; the stable runs",
        "# are not repeated, their branch checkpoints are read from disk.",
        "#",
        f"#   {len(f['cool2x'])} x cooldown2x_rerun + score2x_rerun   (the first cooldown2x branched from a later checkpoint)",
        f"#   {len(f['cool4x'])} x cooldown4x + score4x               (cooldown4x never started)",
        f"#   {len(f['score4x'])} x score4x only                      (cooldown4x finished, scoring job never submitted)",
        f"# = {nc + ns} jobs. Job index: 0-{nc - 1} cooldowns, {nc}-{nc + ns - 1} scoring jobs, in the order of the sweep block." if nc else
        f"# = {ns} jobs.",
        "#",
        "# Submit inside tmux (the monitor starts each scoring job when its cooldown has finished):",
        f"#   python scripts/run_autoexp.py --config-name experiments/laingsam1/opt_scale/sweep100m/phase2_fix_{key} --dry-run",
        f"#   python scripts/run_autoexp.py --config-name experiments/laingsam1/opt_scale/sweep100m/phase2_fix_{key}",
        "# Generated by gen_phase2_fix.py from phase2_" + key + ".yaml.",
        "",
        "",
    ])
    path = os.path.join(HERE, f"phase2_fix_{key}.yaml")
    open(path, "w").write(head + body + s)
    print("wrote", os.path.relpath(path), f"({nc} cooldowns, {ns} scoring jobs)")
