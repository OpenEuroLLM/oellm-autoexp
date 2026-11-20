"""Generate sweep250m/taste_{adam,muon,muown_classical}.yaml from sweep100m/grid_<optimizer>.yaml.

Usage: python3 gen_sweep250m.py <path to opt_scale>

The 250M files are the 100M grid files with these changes, each applied as a checked text
replacement (the script stops if a line it expects is missing or ambiguous):
  model        896 wide, 5 blocks (100M: 640 wide, 4 blocks); same block, experts and vocabulary
  parallelism  4 nodes, experts split over the 4 GPUs of a node, HybridEP dispatcher,
               micro-batch 8 (fastest setting of smoke test 18)
  step counts  scaled by the active parameter count, so the budgets are again 20 / 40 / 80
               tokens per active parameter
  data reader  dataloader_type single for the stable run and the cooldowns, so a cooldown
               continues in the sample list where its branch checkpoint stopped and reads no
               sequence the stable run had read before that point (../data_order_check)
  names        hyb250m_* job names and output folders
"""
import re
import sys

root = sys.argv[1]
OPTS = ["adam", "muon", "muown_classical"]

H, BLOCKS = 896, 5
ACTIVE = 245_850_000        # active parameters without the embedding, (60.2 h^2 + 939 h) x blocks
# Steps at batch 256 (the files scale them by 256 / batch). 1x budget = 20 x ACTIVE tokens =
# 4688 steps of 256 x 4096 tokens; a cooldown is the last 20 %. All are multiples of 8.
BR1, END1, BR2, END2, BR4, END4 = 3752, 4688, 7504, 9376, 15008, 18752
EVAL = 536                  # evaluation every 536 steps at batch 256 (= BR1 / 7); cooldowns: half
ROLL = 1876                 # rolling checkpoint every 1876 steps at batch 256 (about 45 min at batch 512)
GBS = [512]
COST = 155                  # GPU-h per ladder at batch 512 (smoke test 18 x 1.1)

TASTE = {                   # learning rates at the centre momentum / beta: centre, half, double
    "adam": dict(val="0.95", lrs=["5.e-4", "1.e-3", "2.e-3"]),
    "muon": dict(val="0.9", lrs=["2.e-3", "4.e-3", "8.e-3"]),
    "muown_classical": dict(val="0.9", lrs=["2.e-3", "4.e-3", "8.e-3"]),
}
STAGES = ["stable", "cooldown1x", "cooldown2x", "cooldown4x", "score1x", "score2x", "score4x"]


def sub(t, a, b, n=1):
    assert t.count(a) == n, (a[:70], t.count(a), n)
    return t.replace(a, b)


def body(opt):
    t = open(f"{root}/sweep100m/grid_{opt}.yaml").read()
    t = t[t.index("defaults:\n"):t.index("sweep:\n")]
    hk = "beta" if opt == "adam" else "mom"
    # names, folders, tags
    t = t.replace("hyb100m_sweep", "hyb250m_sweep")
    t = sub(t, 'WANDB_TAGS: "jupiter,sweep,grid,hybrid_gdn_moe,hyb100m,', 'WANDB_TAGS: "jupiter,sweep250m,taste,hybrid_gdn_moe,hyb250m,')
    # model
    t = sub(t, "--hybrid-layer-pattern LELELE*ELELELE*ELELELE*ELELELE*E\n", "--hybrid-layer-pattern " + "LELELE*E" * BLOCKS + "\n")
    t = sub(t, "    # --- model: identical to smoke_test/06_hybrid100m_team_arch.yaml ---\n",
            f"    # --- model: the 100M block, {H} wide, {BLOCKS} blocks ({8 * BLOCKS} layers); about {ACTIVE / 1e6:.0f}M active parameters ---\n")
    t = sub(t, "    hidden_size: 640\n", f"    hidden_size: {H}\n")
    t = sub(t, "    ffn_hidden_size: 1920\n", f"    ffn_hidden_size: {3 * H}\n")
    t = sub(t, "    num_attention_heads: 10\n", f"    num_attention_heads: {H // 64}\n")
    t = sub(t, "    linear_num_key_heads: 10\n", f"    linear_num_key_heads: {H // 64}\n")
    t = sub(t, "    linear_num_value_heads: 20\n", f"    linear_num_value_heads: {H // 32}\n")
    t = sub(t, "    moe_ffn_hidden_size: 320\n", f"    moe_ffn_hidden_size: {H // 2}\n")
    t = sub(t, "    moe_shared_expert_intermediate_size: 320\n", f"    moe_shared_expert_intermediate_size: {H // 2}\n")
    # parallelism
    t = sub(t, "    moe_token_dispatcher_type: alltoall\n    expert_model_parallel_size: 1\n",
            "    # experts split over the 4 GPUs of a node (24 each), HybridEP dispatcher: fastest of\n"
            "    # smoke test 18 (35-49k tokens/s/GPU at 250M); the 4 nodes are data-parallel replicas\n"
            "    moe_token_dispatcher_type: flex\n    moe_flex_dispatcher_backend: hybridep\n    expert_model_parallel_size: 4\n")
    t = sub(t, "    micro_batch_size: 16         # with Liger (test 12: 91% GPU memory, 2.5x faster than 4)\n",
            "    micro_batch_size: 8          # smoke test 18: 66 GB of 95 GB; 16 runs out of memory\n")
    t = sub(t, "    nodes: 1\n", "    nodes: 4\n")
    # data reader
    t = sub(t, "    seq_length: 4096\n",
            "    # In-order reader: a cooldown continues in the sample list where its branch checkpoint\n"
            "    # stopped, so it reads nothing the stable run had read before that point. (With 'cyclic'\n"
            "    # the order depends on train_iters and a cooldown rereads data: ../data_order_check.)\n"
            "    # 'single' does not wrap around on the validation data: cooldowns evaluate at half the\n"
            "    # interval of the stable run (see the stage block), which sizes their validation set generously.\n"
            "    dataloader_type: single\n    seq_length: 4096\n")
    # step counts
    t = sub(t, "    # stable: 6168 / 6168 (no cooldown); 1x: 1542 -> 1928; 2x: 3084 -> 3855; 4x: 6168 -> 7710.\n"
               "    branch256: 6168\n    end256: 6168\n",
            f"    # stable: {BR4} / {BR4} (no cooldown); 1x: {BR1} -> {END1}; 2x: {BR2} -> {END2}; 4x: {BR4} -> {END4}.\n"
            f"    branch256: {BR4}\n    end256: {BR4}\n")
    t = sub(t, "    # 100M model: 193 / 96 / 50 steps at batch 128 / 256 / 512.\n    warmup_tokens: 101061472\n",
            f"    # 250M model: {round(ACTIVE / (512 * 4096))} steps at batch 512.\n    warmup_tokens: {ACTIVE}\n")
    t = sub(t, "    eval_interval: \"${oc.eval:'240 * 256 // ${backend.aux.gbs}'}\"\n",
            f"    eval_interval: \"${{oc.eval:'{EVAL} * 256 // ${{backend.aux.gbs}}'}}\"   # every {EVAL * 256 * 4096 / 1e6:.0f}M tokens\n")
    t = sub(t, "      - \"${oc.eval:'round(1542 * 256 / ${backend.aux.gbs})'}\"\n      - \"${oc.eval:'round(3084 * 256 / ${backend.aux.gbs})'}\"\n",
            f"      - \"${{oc.eval:'round({BR1} * 256 / ${{backend.aux.gbs}})'}}\"\n      - \"${{oc.eval:'round({BR2} * 256 / ${{backend.aux.gbs}})'}}\"\n")
    t = sub(t, "    non_persistent_save_interval: \"${oc.eval:'round(514 * 256 / ${backend.aux.gbs})'}\"\n",
            f"    non_persistent_save_interval: \"${{oc.eval:'round({ROLL} * 256 / ${{backend.aux.gbs}})'}}\"\n")
    t = sub(t, "    time: \"12:00:00\"           # stable (JUPITER max = 12 h); others override\n",
            "    time: \"12:00:00\"           # stable: about 6 h at batch 512, 9 h on a slow node; others override\n")
    assert "6168" not in t and "1542" not in t and "640" not in t.split("megatron:\n", 1)[1].split("# --- data")[0], opt
    return t, hk


def stages(opt, hk):
    """The stage block of the 100M grid file with the 250M step counts and time limits."""
    t = open(f"{root}/sweep100m/grid_{opt}.yaml").read()
    s = t[t.index("    # stage: stable + 3 cooldown branches"):]
    s = s.replace("hyb100m_rescore", "hyb250m_rescore")
    s = sub(s, "            backend.megatron.override_opt_param_scheduler: true\n            job.start_condition:\n              class_name: FileExistsCondition\n              # .metadata",
            "            backend.megatron.override_opt_param_scheduler: true\n"
            f"            backend.megatron.eval_interval: \"\\\\${{oc.eval:'{EVAL // 2} * 256 // \\\\${{backend.aux.gbs}}'}}\"   # half the stable interval (reader 'single')\n"
            "            job.start_condition:\n              class_name: FileExistsCondition\n              # .metadata")
    s = sub(s, '{stage: cooldown1x, backend.aux.branch256: 1542, backend.aux.end256: 1928, slurm.sbatch.time: "02:00:00"}',
            f'{{stage: cooldown1x, backend.aux.branch256: {BR1}, backend.aux.end256: {END1}, slurm.sbatch.time: "01:30:00"}}')
    s = sub(s, '{stage: cooldown2x, backend.aux.branch256: 3084, backend.aux.end256: 3855, slurm.sbatch.time: "03:00:00"}',
            f'{{stage: cooldown2x, backend.aux.branch256: {BR2}, backend.aux.end256: {END2}, slurm.sbatch.time: "02:30:00"}}')
    s = sub(s, '{stage: cooldown4x, backend.aux.branch256: 6168, backend.aux.end256: 7710, slurm.sbatch.time: "05:00:00"}',
            f'{{stage: cooldown4x, backend.aux.branch256: {BR4}, backend.aux.end256: {END4}, slurm.sbatch.time: "04:30:00"}}')
    for a, b, c, d in ((1542, 1928, BR1, END1), (3084, 3855, BR2, END2), (6168, 7710, BR4, END4)):
        s = sub(s, f"              backend.aux.branch256: {a}\n              backend.aux.end256: {b}\n",
                f"              backend.aux.branch256: {c}\n              backend.aux.end256: {d}\n")
    assert not re.search(r"\b(1542|1928|3084|3855|6168|7710)\b", s), opt
    return s


def header(opt):
    o, ns = TASTE[opt], len(STAGES)
    hp = "beta1 = beta2" if opt == "adam" else "momentum"
    n = len(GBS) * len(o["lrs"])
    return "\n".join([
        "# @package _global_",
        "#",
        f"# sweep250m taste / {opt}: first ladders at about 250M active parameters. Three learning",
        f"# rates around the best 100M setting, at one batch size. Purpose: check that the ladder",
        "# works at this size on 4 nodes, and get the first results.",
        "#",
        f"#   model        : {H} wide, {BLOCKS} blocks ({8 * BLOCKS} layers), about {ACTIVE / 1e6:.0f}M active / 2.5B total parameters",
        f"#   batch size   : {', '.join(map(str, GBS))} sequences of 4096 tokens",
        f"#   learning rate: {', '.join(o['lrs'])}",
        f"#   {hp:<13}: {o['val']}",
        f"#   budgets      : 1x / 2x / 4x = 20 / 40 / 80 tokens per active parameter = {END1 * 256 * 4096 / 1e9:.1f}B / {END2 * 256 * 4096 / 1e9:.1f}B / {END4 * 256 * 4096 / 1e9:.1f}B tokens",
        "#   machine      : 4 nodes (16 GPUs) per job, experts split inside each node, HybridEP",
        "#   data reader  : in order ('single'), so cooldowns do not reread data of the stable run",
        "#   everything else as sweep100m (schedule, weight decay, clipping, validation, scoring)",
        f"# = {n} settings x {ns} jobs (stable, 3 cooldowns, 3 scoring jobs) = {n * ns} jobs, about {COST} GPU-h per",
        f"#   setting -> about {n * COST:,} GPU-h for the whole file. Stable run: about 6 h (9 h on a slow node).",
        "#",
        "# Not exercised before this file: a whole ladder with the in-order reader, evaluation and",
        "# checkpointing with HybridEP on 4 nodes" + (", and muown_classical with split experts." if opt == "muown_classical" else "."),
        "# The first job of a new size builds the data index (20-30 min); jobs that start at the same",
        "# time wait for it.",
        "#",
        f"# Job index = lr_i x {ns} + stage_i",
        "#   stage_i: " + ", ".join(f"{i} {s}" for i, s in enumerate(STAGES)),
        "#   lr_i   : " + ", ".join(f"{i} {lr}" for i, lr in enumerate(o["lrs"])),
        "#",
        "# Check for every cooldown: the log line 'successfully loaded checkpoint ... at iteration N'",
        "# must show the step in ckpt_step.",
        "# Submit inside tmux (the monitor must keep running to start the cooldowns and scoring jobs):",
        f"#   python scripts/run_autoexp.py --config-name experiments/laingsam1/opt_scale/sweep250m/taste_{opt} --dry-run",
        f"#   python scripts/run_autoexp.py --config-name experiments/laingsam1/opt_scale/sweep250m/taste_{opt}",
        "# Generated by gen_sweep250m.py from ../sweep100m/grid_" + opt + ".yaml.",
        "",
        "",
    ])


def sweep(opt, hk):
    o = TASTE[opt]
    s = "sweep:\n  type: product\n  groups:\n"
    s += "    # batch size\n    - type: list\n      configs:\n"
    s += "".join(f"        - {{backend.aux.gbs: {g}}}\n" for g in GBS)
    s += f"    # learning rate, {'beta1 = beta2' if opt == 'adam' else 'momentum'}\n    - type: list\n      configs:\n"
    s += "".join(f"        - {{backend.megatron.lr: {lr}, backend.aux.{hk}: {o['val']}}}\n" for lr in o["lrs"])
    return s + stages(opt, hk)


for opt in OPTS:
    b, hk = body(opt)
    open(f"{root}/sweep250m/taste_{opt}.yaml", "w").write(header(opt) + b + sweep(opt, hk))
print("written")
