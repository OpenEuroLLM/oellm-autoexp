"""Generate opt_scale/sweep100m/{adam,muon,muown_classical,prebuild_index}.yaml from hyb100m/11.

Usage (from the repo root): python3 config/experiments/laingsam1/opt_scale/sweep100m/gen_sweep100m.py config/experiments/laingsam1/opt_scale
Edit GBS / OPTS below to change the grid, then re-run (overwrites the 4 yaml files).
"""
import sys

root = sys.argv[1]  # .../config/experiments/laingsam1/opt_scale
src = open(f"{root}/hyb100m/11_wsd_ladder.yaml").read()
body0 = src[src.index("defaults:"):src.index("\nsweep:") + 1]

GBS = [128, 256, 512]
OPTS = {
    "adam": dict(lrs=["2.5e-4", "5.e-4", "1.e-3", "2.e-3", "4.e-3"], hp="beta", vals=["0.9", "0.95", "0.975"]),
    "muon": dict(lrs=["2.5e-4", "5.e-4", "1.e-3", "2.e-3", "4.e-3"], hp="mom", vals=["0.9", "0.95", "0.975"]),
    "muown_classical": dict(lrs=["2.5e-4", "5.e-4", "1.e-3", "2.e-3", "4.e-3"], hp="mom", vals=["0.9", "0.95", "0.975"]),
}

# token-aligned step counts: everything is defined in steps at gbs 256 and scaled by 256/gbs
SC = "256 / ${backend.aux.gbs}"


def common_body(opt, prebuild=False):
    b = body0
    hp = OPTS[opt]["hp"] if opt in OPTS else "beta"
    letter = "b" if hp == "beta" else "m"
    rep = {
        # --- names / dirs ---
        '''  # e.g. hyb100m_adam_lr0.001_b0.95_wd0.1_gbs256_decay2x  (= wandb run name = output dir name)
  name: "hyb100m_${backend.aux.opt}_lr${backend.megatron.lr}_b${backend.megatron.adam_beta1}_wd${backend.megatron.weight_decay}_gbs${backend.megatron.global_batch_size}_${stage}"
  base_output_dir: "${backend.aux.base}/experiments/opt_scale/hyb100m/${job.name}"''':
        (f'''  # e.g. hyb100m_sweep_{opt}_lr0.001_{hp}0.95_bs256_cooldown2x  (= wandb run name = output dir name)
  # model _ "sweep" _ optimizer _ learning rate _ {hp} _ batch size _ stage
  name: "hyb100m_sweep_${{backend.aux.opt}}_lr${{backend.megatron.lr}}_{hp}${{backend.aux.{hp}}}_bs${{backend.aux.gbs}}_${{stage}}"
  base_output_dir: "${{backend.aux.base}}/experiments/opt_scale/hyb100m_sweep/${{job.name}}"''' if not prebuild else
         '''  name: "hyb100m_sweep_indexprebuild_bs${backend.aux.gbs}_${stage}"
  base_output_dir: "${backend.aux.base}/experiments/opt_scale/hyb100m_sweep/${job.name}"'''),
        # --- aux: sweep knobs + derived step counts ---
        '''    opt: adam          # overridden by sweep
    start_iter: 0      # decays: stable checkpoint step they branch from
''':
        f'''    opt: {opt if opt in OPTS else "adam"}
    # --- swept (see the sweep block) ---
    gbs: 256           # global batch size (sequences of 4096 tokens)
    beta: 0.95         # AdamW: beta1 = beta2
    mom: 0.95          # Muon / muown_classical: momentum
    # Stage, in steps AT GBS 256 (scaled by 256/gbs below, so every batch size sees the same
    # tokens): branch256 = step the cooldown starts from, end256 = last step.
    # stable: 6168 / 6168 (no cooldown); 1x: 1542 -> 1928; 2x: 3084 -> 3855; 4x: 6168 -> 7710.
    branch256: 6168
    end256: 6168
    # Warmup: linear from 0 over N tokens, N = the model's active parameter count without the
    # embedding (the N of the 20N / 40N / 80N budgets), so 5 % of the 1x budget at every model
    # size (rule of Porian et al. 2024, arXiv:2406.19146), and at least 50 steps.
    # 100M model: 193 / 96 / 50 steps at batch 128 / 256 / 512.
    warmup_tokens: 101061472
    warmup_min_steps: 50
    # --- derived (do not set) ---
    start_iter: "${{oc.eval:'round(${{backend.aux.branch256}} * {SC})'}}"
''',
        # --- batch / steps / schedule ---
        "    global_batch_size: 256\n": "    global_batch_size: ${backend.aux.gbs}\n",
        '''    # Stable phase defaults (decays override in the sweep).
    train_iters: 6168
''': f'''    # all step counts scale with 256/gbs -> same tokens at every batch size (see backend.aux)
    train_iters: "${{oc.eval:'round(${{backend.aux.end256}} * {SC})'}}"
''',
        "    lr_warmup_iters: 40\n":
        "    lr_warmup_iters: \"${oc.eval:'max(${backend.aux.warmup_min_steps}, round(${backend.aux.warmup_tokens} / "
        "(${backend.aux.gbs} * ${backend.megatron.seq_length})))'}\"   # see backend.aux.warmup_tokens\n",
        "    lr_wsd_decay_iters: 0      # stable: no decay\n":
        f"    lr_wsd_decay_iters: \"${{oc.eval:'round(${{backend.aux.end256}} * {SC}) - round(${{backend.aux.branch256}} * {SC})'}}\"   # stable: 0\n",
        # --- validation: same tokens per eval and same eval points (in tokens) at every gbs ---
        '''    # Every 240 steps (~20 min, 26 points on the stable curve), 10 x 256 x 4096 = 10.5M tokens
    # (~15 s per evaluation). (6168//240+1) x 10 = 260 = the old 500 x 20 -> same cached
    # validation index. Megatron also evaluates once more at the end of every job.
    eval_interval: 240
    eval_iters: 10
''': f'''    # Every 252M tokens (240 steps at gbs 256), 2560 sequences = 10.5M tokens per eval, at every
    # batch size -> identical validation samples and eval points (in tokens) across batch sizes,
    # one validation index per budget shared by all of them. Plus a final eval in every job.
    eval_interval: "${{oc.eval:'240 * 256 // ${{backend.aux.gbs}}'}}"
    eval_iters: "${{oc.eval:'10 * 256 // ${{backend.aux.gbs}}'}}"
''',
        # --- checkpoints: persistent only at branch points + end; rolling one for crash recovery ---
        '''    save_interval: 514                     # ~45 min; 6168 = 12 x 514
    save_retain_interval: 1542             # keeps 1542 / 3084 / 4626 / 6168 (branch points), deletes the rest
    save_extra_steps: []
''': f'''    # Permanent checkpoints only at the cooldown branch points (+ the final step, saved
    # automatically); crash recovery from ONE rolling "non-persistent" checkpoint every ~40 min
    # in ckpt/non_persistent (Megatron loads whichever is newer).
    save_interval: 100000
    save_retain_interval: null
    save_extra_steps:                      # 1x / 2x branch points (4x branch = final step)
      - "${{oc.eval:'round(1542 * {SC})'}}"
      - "${{oc.eval:'round(3084 * {SC})'}}"
    non_persistent_save_interval: "${{oc.eval:'round(514 * {SC})'}}"
    non_persistent_ckpt_type: global
''',
        # --- data loading: as Swagatam's 100M reference config (see README "Data loading") ---
        "    num_workers: 2                # test 12: same speed as 4, fewer leaking worker processes\n":
        """    num_workers: 2
    # Megatron's default (true) makes every one of the 522 datasets cache a 4096 x 4096 attention
    # mask in every data-reading process (~9 GB per process and split) and ships 16 masks per
    # micro-batch to the GPU: this filled the host RAM in hyb100m/11. The model masks causally
    # itself; Swagatam's reference config sets false.
    create_attention_mask_in_dataloader: false
    mmap_bin_files: false         # plain reads instead of mmap on GPFS (team setting)
    # 262144 embedding rows, as Swagatam's and Korbinian's configs (the tokenizer file lists one
    # extra added token, so Megatron would otherwise round up to 262272 = 128 unused rows)
    padded_vocab_size: 262144
""",
        # --- shared data-index cache (read-only for colleagues; see RUNBOOK.md) ---
        "    data_cache_path: ${backend.aux.base}/cache/megatron_index\n":
        "    data_cache_path: ${oc.env:OPT_SCALE_CACHE,${backend.aux.base}/cache/megatron_index}\n",
        # --- tags ---
        '''    WANDB_TAGS: "jupiter,ladder_v2,hybrid_gdn_moe,hyb100m,wsd,${backend.aux.opt},${stage},lr=${backend.megatron.lr},beta=${backend.megatron.adam_beta1},wd=${backend.megatron.weight_decay},gbs=${backend.megatron.global_batch_size},nodes=${slurm.sbatch.nodes}"''':
        f'''    WANDB_TAGS: "jupiter,sweep,hybrid_gdn_moe,hyb100m,wsd,${{backend.aux.opt}},${{stage}},lr=${{backend.megatron.lr}},{hp}=${{backend.aux.{hp}}},wd=${{backend.megatron.weight_decay}},gbs=${{backend.aux.gbs}},nodes=${{slurm.sbatch.nodes}}"''',
    }
    for a, c in rep.items():
        assert b.count(a) == 1, a[:70]
        b = b.replace(a, c)
    return b


def opt_specific(b, opt):
    if opt == "adam":
        rep = {"    adam_beta1: 0.95\n    adam_beta2: 0.95\n":
               "    adam_beta1: ${backend.aux.beta}   # beta1 = beta2 (swept)\n    adam_beta2: ${backend.aux.beta}\n",
               "    lr: 1.e-3                  # overridden per optimizer by the sweep\n":
               "    lr: 1.e-3                  # swept\n    optimizer: adam\n"}
    elif opt == "muon":
        rep = {'    opt_args: ""       # set by the sweep\n':
               '''    # Batched Muon step: same-shape matrices (e.g. the 96 experts of a layer) are stacked and
    # orthogonalized in one batched call instead of one by one. Same per-matrix maths (up to
    # floating-point summation order), far fewer GPU calls: this is what makes muown_classical
    # as fast as AdamW (on by default there; off by default for Muon, also in the team configs).
    # Checked in smoke_test/16_muon_batched_check.yaml. Not in the autoexp schema -> raw flag.
    opt_args: "--muon-batched-step"
''',
               "    lr: 1.e-3                  # overridden per optimizer by the sweep\n":
               '''    lr: 1.e-3                  # swept; SAME grid as AdamW: with scale 0.2 x sqrt(max(m, n)) Muon's
    # update size matches AdamW's, so one lr scale serves both (Moonlight convention; the LUMI
    # lr / batch grids used the same lrs for both). hyb100m/11 ran Muon at 5e-3 (the LUMI value
    # for batch 4096).
    # Muon = LUMI settings otherwise; momentum swept. The Adam part (embeddings, router, norms)
    # keeps beta1 = beta2 = 0.95.
    optimizer: muon
    muon_momentum: ${backend.aux.mom}
    muon_nesterov: true
    muon_scale_mode: spectral
    muon_coefficient_type: polar_express
    muon_fp32_matmul_prec: medium
    muon_num_ns_steps: 5
    muon_tp_mode: blockwise
    muon_split_qkv: true
    muon_extra_scale_factor: 0.2
    muon_scalar_optimizer: adam
'''}
    elif opt == "muown_classical":
        rep = {"    lr: 1.e-3                  # overridden per optimizer by the sweep\n":
               "    lr: 1.e-3                  # swept\n    muon_split_qkv: true\n",
               "    weight_decay: 0.1\n": "    weight_decay: 0.0            # muown_classical (Swagatam's setting, as hyb100m/11)\n",
               '    opt_args: ""       # set by the sweep\n':
               '''    # muown_classical (as hyb100m/11); its momentum is --angular-muown-momentum (it reuses the
    # AngularMuown settings), not muon_momentum. The Adam part keeps beta1 = beta2 = 0.95.
    opt_args: "--optimizer muown_classical --muon-qkv-headwise --angular-muown-momentum ${backend.aux.mom}"
'''}
    for a, c in rep.items():
        assert b.count(a) == 1, (opt, a[:60])
        b = b.replace(a, c)
    return b


STAGES = '''    # stage: stable + 3 cooldown branches (1x / 2x / 4x Chinchilla), as in hyb100m/11
    - type: list
      configs:
        - stage: stable
        - type: list
          defaults:
            backend.megatron.save_extra_steps: []
            backend.megatron.non_persistent_save_interval: null   # a cooldown restarts from its branch point
            # No rolling checkpoints here. With "global", Megatron ignores ckpt_step whenever the
            # stable run already has a newer rolling checkpoint and loads that one instead, so a
            # cooldown that waits in the queue branches too late (21 phase 2 cooldowns, 2026-11-19).
            backend.megatron.non_persistent_ckpt_type: null
            backend.megatron.load: "\\\\${sibling.stable.job.base_output_dir}/ckpt"
            backend.megatron.ckpt_step: "\\\\${backend.aux.start_iter}"
            backend.megatron.override_opt_param_scheduler: true
            job.start_condition:
              class_name: FileExistsCondition
              # .metadata is written last -> the branch checkpoint is complete (as on LUMI)
              path: "\\\\${sibling.stable.job.base_output_dir}/ckpt/iter_\\\\${oc.eval:'str(\\\\${backend.aux.start_iter}).zfill(7)'}/.metadata"
            job.cancel_condition:
              class_name: LogPatternCondition
              log_path: "\\\\${sibling.stable.job.log_path_current}"
              pattern: "FATAL ERROR|OutOfMemoryError"
          configs:
            - {stage: cooldown1x, backend.aux.branch256: 1542, backend.aux.end256: 1928, slurm.sbatch.time: "02:00:00"}
            - {stage: cooldown2x, backend.aux.branch256: 3084, backend.aux.end256: 3855, slurm.sbatch.time: "03:00:00"}
            - {stage: cooldown4x, backend.aux.branch256: 6168, backend.aux.end256: 7710, slurm.sbatch.time: "05:00:00"}
'''


def header(opt, o):
    n = len(GBS) * len(o["lrs"]) * len(o["vals"])
    hpname = "beta1 = beta2" if o["hp"] == "beta" else "momentum"
    cost = {"adam": 50, "muon": 57, "muown_classical": 51}[opt]
    nl, nv = len(o["lrs"]), len(o["vals"])
    per_gbs = nl * nv * 4
    lines = [
        "# @package _global_",
        "#",
        f"# sweep100m / {opt}: WSD ladder (stable + 1x / 2x / 4x Chinchilla cooldowns) for every",
        f"# combination of batch size x lr x {hpname}. 100M-active hybrid MoE, everything else as",
        "# hyb100m/11_wsd_ladder.yaml (model, data, Liger + micro-batch 16, logging, restarts).",
        "# Generated from 11 by gen_sweep100m.py; see README.md in this folder.",
        "#",
        f"#   batch size : {', '.join(map(str, GBS))}   (same tokens per stage at every batch size)",
        f"#   lr         : {', '.join(o['lrs'])}",
        f"#   {hpname:11s}: {', '.join(o['vals'])}",
        f"# = {n} settings x 4 jobs (stable, cooldown1x, cooldown2x, cooldown4x) = {n * 4} jobs, ~{cost} GPU-h per",
        f"#   setting -> ~{n * cost:,} GPU-h for the whole file (~{per_gbs // 4 * cost:,} per batch size).",
        "#",
        "# Job index = ((gbs_i x %d + lr_i) x %d + %s_i) x 4 + stage_i  (gbs_i: %s;" % (nl, nv, o["hp"], " ".join(f"{i}={g}" for i, g in enumerate(GBS))),
        "#   stage_i: 0 stable, 1 cooldown1x, 2 cooldown2x, 3 cooldown4x). One batch size = %d consecutive jobs:" % per_gbs,
        "#   " + "   ".join(f"gbs {g}: {i * per_gbs}-{(i + 1) * per_gbs - 1}" for i, g in enumerate(GBS)),
        "#",
        "# BEFORE the first submission: prebuild_index.yaml (once per cache, see README).",
        "# Runs need the monitor (tmux), like 11:",
        f"#   python scripts/run_autoexp.py --config-name experiments/laingsam1/opt_scale/sweep100m/{opt} --dry-run",
        f"#   python scripts/run_autoexp.py --config-name experiments/laingsam1/opt_scale/sweep100m/{opt} --array-subset <range>",
        "",
        "",
    ]
    return "\n".join(lines)


def sweep_block(o):
    s = "sweep:\n  type: product\n  groups:\n"
    s += "    # batch size\n    - type: list\n      configs:\n"
    s += "".join(f"        - {{backend.aux.gbs: {g}}}\n" for g in GBS)
    s += "    # learning rate\n    - type: list\n      configs:\n"
    s += "".join(f"        - {{backend.megatron.lr: {lr}}}\n" for lr in o["lrs"])
    s += f"    # {'beta1 = beta2' if o['hp'] == 'beta' else 'momentum'}\n    - type: list\n      configs:\n"
    s += "".join(f"        - {{backend.aux.{o['hp']}: {v}}}\n" for v in o["vals"])
    return s + STAGES


for opt, o in OPTS.items():
    b = opt_specific(common_body(opt), opt)
    open(f"{root}/sweep100m/{opt}.yaml", "w").write(header(opt, o) + b + sweep_block(o))

# --- phase 1: learning rate x batch size at momentum 0.95, one file per optimizer, run whole ---
P1_GBS = [128, 256, 512]
P1_LRS = ["2.5e-4", "5.e-4", "1.e-3", "2.e-3", "4.e-3", "8.e-3"]
P1_COST = {"muon": 57, "muown_classical": 51, "adam": 50}


def phase1_header(opt):
    nl = len(P1_LRS)
    n = len(P1_GBS) * nl
    centre = P1_LRS.index("1.e-3")
    rng = lambda g: f"{g * nl * 4}-{(g + 1) * nl * 4 - 1}"
    pil = lambda g: f"{(g * nl + centre) * 4}-{(g * nl + centre) * 4 + 3}"
    return "\n".join([
        "# @package _global_",
        "#",
        f"# sweep100m phase 1 / {opt}: learning rate x batch size, one WSD ladder per setting.",
        "# Question: what is the best learning rate at each batch size and token budget?",
        "#",
        f"#   batch size : {', '.join(map(str, P1_GBS))} sequences of 4096 tokens (same tokens at every batch size)",
        f"#   lr         : {', '.join(P1_LRS)}",
        "#   momentum   : 0.95 (AdamW: beta1 = beta2 = 0.95); weight decay and everything else as README.md",
        f"# = {n} settings x 4 jobs (stable, cooldown1x, cooldown2x, cooldown4x) = {n * 4} jobs, ~{P1_COST[opt]} GPU-h per",
        f"#   setting -> ~{n * P1_COST[opt]:,} GPU-h for the whole file.",
        "#",
        f"# Job index = (batch_i x {nl} + lr_i) x 4 + stage_i   (stage_i: 0 stable, 1 cooldown1x, 2 cooldown2x, 3 cooldown4x)",
        "#   " + "   ".join(f"batch {b}: {rng(g)}" for g, b in enumerate(P1_GBS)),
        "#   lr 1e-3 ladders (run these first): " + ", ".join(f"batch {b}: {pil(g)}" for g, b in enumerate(P1_GBS)),
        "#",
        "# How to run it: RUNBOOK.md in this folder. Generated by gen_sweep100m.py from hyb100m/11.",
        "",
        "",
    ])


def phase1_sweep():
    s = "sweep:\n  type: product\n  groups:\n"
    s += "    # batch size\n    - type: list\n      configs:\n"
    s += "".join(f"        - {{backend.aux.gbs: {g}}}\n" for g in P1_GBS)
    s += "    # learning rate\n    - type: list\n      configs:\n"
    s += "".join(f"        - {{backend.megatron.lr: {lr}}}\n" for lr in P1_LRS)
    return s + STAGES


for opt in OPTS:
    b = opt_specific(common_body(opt), opt)
    open(f"{root}/sweep100m/phase1_{opt}.yaml", "w").write(phase1_header(opt) + b + phase1_sweep())

# --- phase 2: beta / momentum around the best learning rates of phase 1, one WSD ladder per setting ---
# Momentum 0.95 is phase 1, so only the new values are run here. Two learning rates per optimizer:
# the best of phase 1 at the 4x budget and its neighbour that is best at 1x (AdamW 1e-3 / 2e-3,
# Muon and muown_classical 2e-3 / 4e-3).
# The training jobs are configured exactly as in phase 1 (same reader, same evaluations), so
# the new values compare directly with the 0.95 runs. The only addition: every cooldown is
# followed by a scoring job on the fixed validation set of ../rescore.
P2_COST = 50                  # GPU-h per ladder: 48 measured in phase 1 + 3 scoring jobs
P2_UNTIED = ("0.9", "0.95")   # AdamW reference with the usual untied betas
P2 = {
    "adam": dict(gbs=[128, 256, 512], lrs=["1.e-3", "2.e-3"], vals=["0.9", "0.975"]),
    "adam_untied": dict(gbs=[256], lrs=["1.e-3", "2.e-3"], vals=[]),
    "muon": dict(gbs=[128, 256, 512], lrs=["2.e-3", "4.e-3"], vals=["0.9", "0.975"]),
    "muown_classical": dict(gbs=[128, 256, 512], lrs=["2.e-3", "4.e-3"], vals=["0.9", "0.975"]),
}
P2_NVALID = 25600             # sequences of the fixed validation set (as ../rescore)
P2_STAGES = ["stable", "cooldown1x", "cooldown2x", "cooldown4x", "score1x", "score2x", "score4x"]


def phase2_stages(score_name):
    """STAGES of phase 1 + one scoring job per cooldown."""
    s = STAGES.replace("# stage: stable + 3 cooldown branches (1x / 2x / 4x Chinchilla), as in hyb100m/11",
                       "# stage: stable + 3 cooldown branches (1x / 2x / 4x Chinchilla) + 3 scoring jobs")
    s += f'''        # Scoring only (no training): the final checkpoint of each cooldown on the fixed validation
        # set, the first {P2_NVALID:,} held-out sequences in fixed order (same set as ../rescore, so
        # phase 1 and phase 2 losses are comparable). Starts when the cooldown's checkpoint exists.
        - type: list
          defaults:
            backend.megatron.skip_train: true
            backend.megatron.dataloader_type: single   # fixed order: the same sequences for every checkpoint
            backend.megatron.eval_iters: "\\\\${{oc.eval:'{P2_NVALID} // \\\\${{backend.aux.gbs}}'}}"
            backend.megatron.save_extra_steps: []
            backend.megatron.non_persistent_save_interval: null
            backend.megatron.non_persistent_ckpt_type: null   # load exactly ckpt_step
            backend.megatron.override_opt_param_scheduler: true
            backend.megatron.ckpt_step: "\\\\${{backend.megatron.train_iters}}"
            slurm.sbatch.time: "01:00:00"
            job.name: "{score_name}"
            job.base_output_dir: "\\\\${{backend.aux.base}}/experiments/opt_scale/hyb100m_rescore/\\\\${{job.name}}"
          configs:
'''
    for x, br, end in (("1x", 1542, 1928), ("2x", 3084, 3855), ("4x", 6168, 7710)):
        s += f'''            - stage: score{x}
              backend.aux.branch256: {br}
              backend.aux.end256: {end}
              backend.megatron.load: "\\\\${{sibling.cooldown{x}.job.base_output_dir}}/ckpt"
              job.start_condition:
                class_name: FileExistsCondition
                path: "\\\\${{sibling.cooldown{x}.job.base_output_dir}}/ckpt/iter_\\\\${{oc.eval:'str(\\\\${{backend.megatron.train_iters}}).zfill(7)'}}/.metadata"
'''
    return s


def phase2_header(key, o, what, question):
    opt = "adam" if key.startswith("adam") else key
    nl, nv, ns = len(o["lrs"]), max(len(o["vals"]), 1), len(P2_STAGES)
    n = len(o["gbs"]) * nl * nv
    per = nl * nv * ns
    hp = "beta1 = beta2" if opt == "adam" else "momentum"
    index = (f"# Job index = ((batch_i x {nl} + lr_i) x {nv} + {'beta' if opt == 'adam' else 'mom'}_i) x {ns} + stage_i"
             if o["vals"] else f"# Job index = lr_i x {ns} + stage_i")
    return "\n".join([
        "# @package _global_",
        "#",
        f"# sweep100m phase 2 / {key}: {what}, one WSD ladder per setting.",
        f"# Question: {question}",
        "#",
        f"#   batch size : {', '.join(map(str, o['gbs']))} sequences of 4096 tokens (same tokens at every batch size)",
        f"#   lr         : {', '.join(o['lrs'])}",
        (f"#   {hp:11s}: {', '.join(o['vals'])}   (0.95 was run in phase 1)" if o["vals"] else
         f"#   betas      : beta1 = {P2_UNTIED[0]}, beta2 = {P2_UNTIED[1]} (fixed)"),
        "#   weight decay, data reader and everything else as phase 1",
        f"# = {n} settings x {ns} jobs (stable, 3 cooldowns, 3 scoring jobs) = {n * ns} jobs, ~{P2_COST} GPU-h per",
        f"#   setting -> ~{n * P2_COST:,} GPU-h for the whole file.",
        "#",
        index,
        "#   stage_i: " + ", ".join(f"{i} {s}" for i, s in enumerate(P2_STAGES)),
        "#   " + "   ".join(f"batch {b}: {g * per}-{(g + 1) * per - 1}" for g, b in enumerate(o["gbs"])),
        "#",
        "# The result of a ladder is the three scoring jobs (fixed validation set); collect them with",
        "#   python3 config/experiments/laingsam1/opt_scale/rescore/collect_rescore.py --logs $OPT_SCALE_BASE/logs",
        "# Submit inside tmux (the monitor must keep running to start the cooldowns and scoring jobs):",
        f"#   python scripts/run_autoexp.py --config-name experiments/laingsam1/opt_scale/sweep100m/phase2_{key} --dry-run",
        f"#   python scripts/run_autoexp.py --config-name experiments/laingsam1/opt_scale/sweep100m/phase2_{key}",
        "# Generated by gen_sweep100m.py from hyb100m/11.",
        "",
        "",
    ])


def phase2_sweep(key, o, score_name):
    opt = "adam" if key.startswith("adam") else key
    hk = "beta" if opt == "adam" else "mom"
    s = "sweep:\n  type: product\n  groups:\n"
    s += "    # batch size\n    - type: list\n      configs:\n"
    s += "".join(f"        - {{backend.aux.gbs: {g}}}\n" for g in o["gbs"])
    s += "    # learning rate\n    - type: list\n      configs:\n"
    s += "".join(f"        - {{backend.megatron.lr: {lr}}}\n" for lr in o["lrs"])
    if o["vals"]:
        s += f"    # {'beta1 = beta2' if opt == 'adam' else 'momentum'}\n    - type: list\n      configs:\n"
        s += "".join(f"        - {{backend.aux.{hk}: {v}}}\n" for v in o["vals"])
    return s + phase2_stages(score_name)


def phase2_body(key):
    opt = "adam" if key.startswith("adam") else key
    b = opt_specific(common_body(opt), opt)
    rep = {
        'WANDB_TAGS: "jupiter,sweep,': 'WANDB_TAGS: "jupiter,sweep,phase2,',
    }
    if key == "adam_untied":
        b1, b2 = P2_UNTIED
        rep.update({
            "    adam_beta1: ${backend.aux.beta}   # beta1 = beta2 (swept)\n    adam_beta2: ${backend.aux.beta}\n":
            f"    adam_beta1: {b1}   # untied reference\n    adam_beta2: {b2}\n",
            "_lr${backend.megatron.lr}_beta${backend.aux.beta}_bs${backend.aux.gbs}_${stage}\"":
            f"_lr${{backend.megatron.lr}}_b1-{b1}_b2-{b2}_bs${{backend.aux.gbs}}_${{stage}}\"",
            "lr=${backend.megatron.lr},beta=${backend.aux.beta},wd=": f"lr=${{backend.megatron.lr}},beta1={b1},beta2={b2},wd=",
            "  # e.g. hyb100m_sweep_adam_lr0.001_beta0.95_bs256_cooldown2x  (= wandb run name = output dir name)\n"
            "  # model _ \"sweep\" _ optimizer _ learning rate _ beta _ batch size _ stage\n":
            f"  # e.g. hyb100m_sweep_adam_lr0.001_b1-{b1}_b2-{b2}_bs256_cooldown2x  (= wandb run name = output dir name)\n"
            "  # model _ \"sweep\" _ optimizer _ learning rate _ beta1 _ beta2 _ batch size _ stage\n",
            "    beta: 0.95         # AdamW: beta1 = beta2\n": "    beta: 0.95         # not used in this file (betas are fixed below)\n",
        })
    for a, c in rep.items():
        assert b.count(a) == 1, (key, a[:60])
        b = b.replace(a, c)
    return b


P2_TEXT = {
    "adam": ("beta1 = beta2 x learning rate x batch size", "does the best loss, and the learning rate that gives it, change with the betas?"),
    "adam_untied": (f"AdamW with beta1 = {P2_UNTIED[0]}, beta2 = {P2_UNTIED[1]} (reference) at batch 256",
                    "how far is the usual untied AdamW setting from the tied betas of this sweep?"),
    "muon": ("momentum x learning rate x batch size", "does the best loss, and the learning rate that gives it, change with momentum?"),
    "muown_classical": ("momentum x learning rate x batch size (held back until the Muon result is in)",
                        "does the best loss, and the learning rate that gives it, change with momentum?"),
}
for key, o in P2.items():
    opt = "adam" if key.startswith("adam") else key
    hk = "beta" if opt == "adam" else "mom"
    mid = f"b1-{P2_UNTIED[0]}_b2-{P2_UNTIED[1]}" if key == "adam_untied" else f"{hk}\\\\${{backend.aux.{hk}}}"
    score_name = f"hyb100m_rescore_\\\\${{backend.aux.opt}}_lr\\\\${{backend.megatron.lr}}_{mid}_bs\\\\${{backend.aux.gbs}}_\\\\${{stage}}"
    open(f"{root}/sweep100m/phase2_{key}.yaml", "w").write(
        phase2_header(key, o, *P2_TEXT[key]) + phase2_body(key) + phase2_sweep(key, o, score_name))

# --- grid: completes learning rate x beta / momentum to a full rectangle at every batch size ---
# Four learning rates x {0.8, 0.9, 0.95, 0.975} per optimizer. Only the settings that phase 1
# (0.95) and phase 2 (0.9 and 0.975 at two learning rates; Muon and AdamW only) have not run.
# Jobs are configured exactly as phase 2 (same reader, same evaluations, one scoring job per
# cooldown), so all three phases compare directly.
GRID_VALS = ["0.8", "0.9", "0.95", "0.975"]
GRID_LRS = {
    "adam": ["5.e-4", "1.e-3", "2.e-3", "4.e-3"],
    "muon": ["1.e-3", "2.e-3", "4.e-3", "8.e-3"],
    "muown_classical": ["1.e-3", "2.e-3", "4.e-3", "8.e-3"],
}
GRID_DONE = {   # (lr, value) pairs with a finished ladder at every batch size
    "adam": {(lr, "0.95") for lr in P1_LRS} | {(lr, v) for lr in P2["adam"]["lrs"] for v in P2["adam"]["vals"]},
    "muon": {(lr, "0.95") for lr in P1_LRS} | {(lr, v) for lr in P2["muon"]["lrs"] for v in P2["muon"]["vals"]},
    "muown_classical": {(lr, "0.95") for lr in P1_LRS},   # its phase 2 file was never submitted
}


def grid_pairs(opt):
    return [(lr, v) for v in GRID_VALS for lr in GRID_LRS[opt] if (lr, v) not in GRID_DONE[opt]]


def grid_header(opt):
    pairs, ns = grid_pairs(opt), len(P2_STAGES)
    hp = "beta1 = beta2" if opt == "adam" else "momentum"
    n, per = len(GBS) * len(pairs), len(pairs) * ns
    lines = [
        "# @package _global_",
        "#",
        f"# sweep100m grid / {opt}: the settings missing from a full learning rate x {hp} grid,",
        "# one WSD ladder per setting.",
        "#",
        f"#   batch size : {', '.join(map(str, GBS))} sequences of 4096 tokens (same tokens at every batch size)",
        f"#   full grid  : lr {', '.join(GRID_LRS[opt])}  x  {hp} {', '.join(GRID_VALS)}",
        "#   run here   : the pairs below; the others were run in phase 1 and phase 2",
        "#   weight decay, data reader and everything else as phase 1 and phase 2",
        f"# = {n} settings x {ns} jobs (stable, 3 cooldowns, 3 scoring jobs) = {n * ns} jobs, ~{P2_COST} GPU-h per",
        f"#   setting -> ~{n * P2_COST:,} GPU-h for the whole file.",
        "#",
        f"# Job index = (batch_i x {len(pairs)} + pair_i) x {ns} + stage_i",
        "#   stage_i: " + ", ".join(f"{i} {s}" for i, s in enumerate(P2_STAGES)),
        "#   pair_i : " + ", ".join(f"{i} ({lr}, {v})" for i, (lr, v) in enumerate(pairs)),
        "#   " + "   ".join(f"batch {b}: {g * per}-{(g + 1) * per - 1}" for g, b in enumerate(GBS)),
        "#",
        "# Results: in-training validation of each cooldown and the scoring jobs (fixed validation set);",
        "#   python3 config/experiments/laingsam1/opt_scale/sweep100m/make_sweep_page.py --logs $OPT_SCALE_BASE/logs --out sweep_page.html",
        "# Submit inside tmux (the monitor must keep running to start the cooldowns and scoring jobs):",
        f"#   python scripts/run_autoexp.py --config-name experiments/laingsam1/opt_scale/sweep100m/grid_{opt} --dry-run",
        f"#   python scripts/run_autoexp.py --config-name experiments/laingsam1/opt_scale/sweep100m/grid_{opt}",
        "# Generated by gen_sweep100m.py from hyb100m/11.",
        "",
        "",
    ]
    return "\n".join(lines)


def grid_sweep(opt, score_name):
    hk = "beta" if opt == "adam" else "mom"
    s = "sweep:\n  type: product\n  groups:\n"
    s += "    # batch size\n    - type: list\n      configs:\n"
    s += "".join(f"        - {{backend.aux.gbs: {g}}}\n" for g in GBS)
    s += f"    # learning rate, {'beta1 = beta2' if opt == 'adam' else 'momentum'}\n    - type: list\n      configs:\n"
    s += "".join(f"        - {{backend.megatron.lr: {lr}, backend.aux.{hk}: {v}}}\n" for lr, v in grid_pairs(opt))
    return s + phase2_stages(score_name)


for opt in OPTS:
    hk = "beta" if opt == "adam" else "mom"
    score_name = f"hyb100m_rescore_\\\\${{backend.aux.opt}}_lr\\\\${{backend.megatron.lr}}_{hk}\\\\${{backend.aux.{hk}}}_bs\\\\${{backend.aux.gbs}}_\\\\${{stage}}"
    gb = phase2_body(opt)
    assert gb.count('WANDB_TAGS: "jupiter,sweep,phase2,') == 1
    gb = gb.replace('WANDB_TAGS: "jupiter,sweep,phase2,', 'WANDB_TAGS: "jupiter,sweep,grid,')
    open(f"{root}/sweep100m/grid_{opt}.yaml", "w").write(grid_header(opt) + gb + grid_sweep(opt, score_name))

# --- weight decay: other values at the two best learning rates and the best beta / momentum ---
# The grid ran AdamW and Muon at weight decay 0.1 and muown_classical at 0 (WD_BASE). This adds
# three more values per optimizer, at every batch size. Two learning rates because the decay is
# decoupled (applied as lr x weight decay), so the two interact. Jobs are configured exactly as
# the grid; the weight decay is part of the job name, so no existing folder is touched and the
# existing read-out scripts (which match names without "_wd") ignore these runs.
WD_BASE = {"adam": "0.1", "muon": "0.1", "muown_classical": "0.0"}
WD = {
    "adam": dict(val="0.95", lrs=["1.e-3", "2.e-3"], wds=["0.0", "0.03", "0.3"]),
    "muon": dict(val="0.9", lrs=["2.e-3", "4.e-3"], wds=["0.0", "0.03", "0.3"]),
    "muown_classical": dict(val="0.9", lrs=["2.e-3", "4.e-3"], wds=["0.03", "0.1", "0.3"]),
}


def wd_header(opt):
    o, ns = WD[opt], len(P2_STAGES)
    hp = "beta1 = beta2" if opt == "adam" else "momentum"
    npairs = len(o["lrs"]) * len(o["wds"])
    n, per = len(GBS) * npairs, npairs * ns
    lines = [
        "# @package _global_",
        "#",
        f"# sweep100m weight decay / {opt}: weight decay x learning rate x batch size, one WSD ladder per setting.",
        "#",
        f"#   batch size   : {', '.join(map(str, GBS))} sequences of 4096 tokens",
        f"#   learning rate: {', '.join(o['lrs'])}",
        f"#   weight decay : {', '.join(o['wds'])}   (the grid already has {WD_BASE[opt]} at these settings)",
        f"#   {hp:<13}: {o['val']} (fixed)",
        "#   data reader and everything else as the grid",
    ]
    if opt == "muown_classical":
        lines += [
            "#   Note: muown_classical keeps a magnitude and a normalised direction per weight row; its",
            "#   weight decay acts on the unnormalised variable, so it is not the same quantity as the",
            "#   weight decay of AdamW or Muon. The team default is 0.",
        ]
    lines += [
        f"# = {n} settings x {ns} jobs (stable, 3 cooldowns, 3 scoring jobs) = {n * ns} jobs, ~{P2_COST} GPU-h per",
        f"#   setting -> ~{n * P2_COST:,} GPU-h for the whole file.",
        "#",
        f"# Job index = ((batch_i x {len(o['lrs'])} + lr_i) x {len(o['wds'])} + wd_i) x {ns} + stage_i",
        "#   stage_i: " + ", ".join(f"{i} {s}" for i, s in enumerate(P2_STAGES)),
        "#   " + "   ".join(f"batch {b}: {g * per}-{(g + 1) * per - 1}" for g, b in enumerate(GBS)),
        "#",
        "# Submit inside tmux (the monitor must keep running to start the cooldowns and scoring jobs):",
        f"#   python scripts/run_autoexp.py --config-name experiments/laingsam1/opt_scale/sweep100m/wd_{opt} --dry-run",
        f"#   python scripts/run_autoexp.py --config-name experiments/laingsam1/opt_scale/sweep100m/wd_{opt}",
        "# Generated by gen_sweep100m.py from hyb100m/11.",
        "",
        "",
    ]
    return "\n".join(lines)


def wd_sweep(opt, score_name):
    o = WD[opt]
    hk = "beta" if opt == "adam" else "mom"
    s = "sweep:\n  type: product\n  groups:\n"
    s += "    # batch size\n    - type: list\n      configs:\n"
    s += "".join(f"        - {{backend.aux.gbs: {g}}}\n" for g in GBS)
    s += "    # learning rate\n    - type: list\n      configs:\n"
    s += "".join(f"        - {{backend.megatron.lr: {lr}, backend.aux.{hk}: {o['val']}}}\n" for lr in o["lrs"])
    s += "    # weight decay\n    - type: list\n      configs:\n"
    s += "".join(f"        - {{backend.megatron.weight_decay: {w}}}\n" for w in o["wds"])
    return s + phase2_stages(score_name)


for opt in OPTS:
    hk = "beta" if opt == "adam" else "mom"
    score_name = (f"hyb100m_rescore_\\\\${{backend.aux.opt}}_lr\\\\${{backend.megatron.lr}}_{hk}\\\\${{backend.aux.{hk}}}"
                  f"_wd\\\\${{backend.megatron.weight_decay}}_bs\\\\${{backend.aux.gbs}}_\\\\${{stage}}")
    wb = phase2_body(opt)
    for a, c in {
        'WANDB_TAGS: "jupiter,sweep,phase2,': 'WANDB_TAGS: "jupiter,sweep,wd,',
        '_bs${backend.aux.gbs}_${stage}"\n': '_wd${backend.megatron.weight_decay}_bs${backend.aux.gbs}_${stage}"\n',
    }.items():
        assert wb.count(a) == 1, (opt, a)
        wb = wb.replace(a, c)
    open(f"{root}/sweep100m/wd_{opt}.yaml", "w").write(wd_header(opt) + wb + wd_sweep(opt, score_name))

# --- 1x runs: the same learning rate x batch size grid as phase 1, but each setting is ONE run
# trained from scratch to the 1x budget (2.0B tokens: warmup, constant lr, cooldown over the
# last 20 %). No ladder, no branching: ~4-5x cheaper per setting, for locating the good region.
RUN1X_COST = {"muon": 13, "muown_classical": 11.5, "adam": 11.5}


def run1x_header(opt):
    nl = len(P1_LRS)
    n = len(P1_GBS) * nl
    return "\n".join([
        "# @package _global_",
        "#",
        f"# sweep100m 1x runs / {opt}: learning rate x batch size, one run per setting.",
        "# Question: where is the good learning-rate range at each batch size? (cheap first pass;",
        "# the best settings then get full ladders with phase1_*.yaml for the 2x / 4x budgets)",
        "#",
        f"#   batch size : {', '.join(map(str, P1_GBS))} sequences of 4096 tokens",
        f"#   lr         : {', '.join(P1_LRS)}",
        "#   momentum   : 0.95 (AdamW: beta1 = beta2 = 0.95)",
        "# Each run: 2.0B tokens from scratch = warmup 101M tokens (= N), constant lr, linear cooldown to 0",
        "# over the last 20 % (the same schedule as a ladder's 1x branch, without the branching).",
        f"# = {n} runs, 1 job each, ~{RUN1X_COST[opt]} GPU-h per run -> ~{round(n * RUN1X_COST[opt]):,} GPU-h for the whole file,",
        "#   ~3 h per run. Crash recovery from a rolling checkpoint; final checkpoint kept.",
        "#",
        f"# Job index = batch_i x {nl} + lr_i:   " + "   ".join(
            f"batch {b}: {g * nl}-{(g + 1) * nl - 1}" for g, b in enumerate(P1_GBS)),
        "#",
        "# Needs the monitor only for automatic restarts, so --submit-and-exit is fine too:",
        f"#   python scripts/run_autoexp.py --config-name experiments/laingsam1/opt_scale/sweep100m/run1x_{opt} --dry-run",
        f"#   python scripts/run_autoexp.py --config-name experiments/laingsam1/opt_scale/sweep100m/run1x_{opt}",
        "# Generated by gen_sweep100m.py from hyb100m/11.",
        "",
        "",
    ])


def run1x_sweep():
    s = "sweep:\n  type: product\n  groups:\n"
    s += "    # batch size\n    - type: list\n      configs:\n"
    s += "".join(f"        - {{backend.aux.gbs: {g}}}\n" for g in P1_GBS)
    s += "    # learning rate\n    - type: list\n      configs:\n"
    s += "".join(f"        - {{backend.megatron.lr: {lr}}}\n" for lr in P1_LRS)
    s += ("    # one run from scratch to the 1x budget: cooldown from step 1542 to 1928 (at batch 256)\n"
          "    - type: list\n      configs:\n"
          "        - stage: run1x\n"
          "          backend.aux.branch256: 1542\n"
          "          backend.aux.end256: 1928\n"
          "          backend.megatron.save_extra_steps: []\n"
          "          slurm.sbatch.time: \"04:30:00\"\n")
    return s


for opt in OPTS:
    b = opt_specific(common_body(opt), opt)
    open(f"{root}/sweep100m/run1x_{opt}.yaml", "w").write(run1x_header(opt) + b + run1x_sweep())

# --- prebuild: the data indexes the sweep needs that are not cached yet (1 step each) ---
b = opt_specific(common_body("prebuild", prebuild=True), "adam")
pre_rep = {
    "  - /job: auto_restart   # monitor resubmits on time limit / crash (resumes from latest ckpt)": "  - /job: default        # no auto-restart",
    "    exit_interval: null\n": "    exit_interval: 1             # stop right after the index build + 1 step\n",
    '''    save: "${job.base_output_dir}/ckpt"
    load: "${job.base_output_dir}/ckpt"''': "    save: null\n    load: null",
    '''    non_persistent_save_interval: "${oc.eval:'round(514 * 256 / ${backend.aux.gbs})'}"
    non_persistent_ckpt_type: global''': "    non_persistent_save_interval: null\n    non_persistent_ckpt_type: null",
    "    exit_duration_in_mins: 690             # clean save + exit 30 min before the 12 h limit\n": "    exit_duration_in_mins: null\n",
    '''    time: "12:00:00"           # stable (JUPITER max = 12 h); others override''': '''    time: "01:00:00"           # index build ~20-30 min + startup''',
}
for a, c in pre_rep.items():
    assert b.count(a) == 1, ("pre", a[:60])
    b = b.replace(a, c)
a = b.index("  # = config/job/auto_restart.yaml"); z = b.index("backend:\n")
b = b[:a] + "\n" + b[z:]
pre_header = '''# @package _global_
#
# Pre-builds the ONE data index the sweep100m configs need that hyb100m/11 + 11b have not
# cached already (AdamW, 1 step, no checkpoints): the train index of the 2x cooldown at batch
# size 512 (1927.5 steps round to 1928 -> 987,136 samples instead of 986,880). Every other
# stage / batch size uses the same sample counts as 11 and 11b (step counts scale with
# 256 / batch size; validation = 2560 sequences per evaluation everywhere).
# Run ONCE per $OPT_SCALE_BASE/cache before the first sweep submission and wait until COMPLETED:
#   python scripts/run_autoexp.py --config-name experiments/laingsam1/opt_scale/sweep100m/prebuild_index --submit-and-exit
# (If your cache is empty, i.e. you never ran hyb100m/11 + 11b, run those two first, or submit one
# setting of the sweep alone and wait until all 4 of its jobs have started training.)
# Cost: 1 job x ~25-35 min x 4 GPUs = ~2 GPU-h (cap 4).

'''
pre_sweep = '''sweep:
  type: product
  groups:
    - type: list
      configs:
        - {stage: cooldown2x, backend.aux.gbs: 512, backend.aux.branch256: 3084, backend.aux.end256: 3855}
'''
open(f"{root}/sweep100m/prebuild_index.yaml", "w").write(pre_header + b + pre_sweep)
print("written")
