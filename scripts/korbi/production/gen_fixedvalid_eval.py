#!/usr/bin/env python3
"""Eval-only configs for the seven fixed held-out validation sets (all / web_en / multilingual / code /
math / reasoning at 4k, long_64k) on checkpoints of earlier runs (user 2026-11-01: the conservative
run at 4k steps, the first ambitious run, the latest ambitious run).

The model is the run's own: backend.megatron and backend.env are copied from the run's resolved
config (config-<job>.yaml next to its checkpoints), keys the current Megatron schema no longer has
dropped (printed). On top, eval-only patches: load the checkpoint at ckpt_step without optimizer /
RNG state, train_iters = ckpt_step (the loop does not run; the end-of-training evaluation does),
no saves, a three-shard training blend (fast index; not used), the fixed sets. A normal FSDP setup
(skip_train would build the model unwrapped; needs Megatron >= the --no-load-optim fix for
Megatron-FSDP), with bf16 main weights / grads to fit 4 nodes (the dp-sharded Engram tables must stay fp32).

    python scripts/korbi/production/gen_fixedvalid_eval.py NAME RESOLVED_CONFIG.yaml STEP=CKPT_DIR [...]

(one arm per STEP; CKPT_DIR holds iter_<STEP>: ckpt, ckpt_eval or ckpt_rolling)
"""

import dataclasses
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from oellm_autoexp.backends.megatron.config_schema import MegatronConfig  # noqa: E402

# FVE_VALID_ROOT: v8ml-valid (tokenizer-256k-agent) or v8ml-valid-basetok (the same sets in tokenizer-256k,
# scripts/datamix_v8/retok_valid_sets.py, for models trained on it); FVE_SETS: a subset (comma separated)
VALID_ROOT = os.environ.get("FVE_VALID_ROOT", "/gpfs/projects/ehpc533/users/korbip/data/v8ml-valid")
_ALL = {"all": 4096, "web_en": 4096, "multilingual": 4096, "code": 4096, "math": 4096, "reasoning": 4096,
        "long_64k": 65536}
SETS = os.environ.get("FVE_SETS", ",".join(_ALL)).split(",")
LENGTHS = [_ALL[s] for s in SETS]

EVAL_PATCH = {
    "no_load_optim": True,
    "no_load_rng": True,
    # the run's hybrid sharding (q35: 16 replicas at 128 nodes) does not divide 4 nodes; one FSDP
    # group (the checkpoint reshards)
    "num_distributed_optimizer_instances": 1,
    # memory (4 nodes; the first ambitious run OOMed at the load with fp32 buffers): the forward
    # runs on the bf16-rounded weights either way, and DCP loads fp32 into bf16 with
    # exactly that rounding -- the evaluation is unchanged
    "megatron_fsdp_main_params_dtype": "bf16",
    "megatron_fsdp_main_grads_dtype": "bf16",
    "fsdp_double_buffer": False,
    "save": None,
    "save_interval": None,
    "save_retain_interval": None,
    "non_persistent_ckpt_type": None,
    "non_persistent_global_ckpt_dir": None,
    "non_persistent_save_interval": None,
    "async_save": False,
    "use_persistent_ckpt_worker": False,
    "ckpt_assume_constant_structure": False,
    "exit_interval": None,
    "exit_signal_handler": False,
    "data_args_path": "/gpfs/projects/ehpc533/users/korbip/hybrid_v019/data/validsmoke_train_datamix.txt",
    "data_cache_path": "/gpfs/scratch/ehpc533/hybrid_nucleus/data_cache_shared",
    "fixed_valid_data_path": [f"{VALID_ROOT}/{s}_text_document" for s in SETS],
    "validation_set_names": SETS,
    "fixed_valid_seq_lengths": LENGTHS,
    "eval_interval": 1,
    "eval_iters": 1,
    "wandb_project": None,
    "wandb_exp_name": None,
    "wandb_save_dir": None,
    "tensorboard_dir": "${job.base_output_dir}/tb",
    "log_interval": 1,
    "arch_metrics_interval": 0,
}


def main():
    name, resolved, *marks = sys.argv[1:]
    marks = [m.split("=", 1) for m in marks]
    steps = [s for s, _ in marks]
    cfg = yaml.safe_load(open(resolved))["config"]
    meg = dict(cfg["backend"]["megatron"])
    meg.pop("aux", None)
    known = {f.name for f in dataclasses.fields(MegatronConfig)}
    dropped = sorted(k for k in meg if k not in known)
    for k in dropped:
        meg.pop(k)
    meg.update(EVAL_PATCH)
    env = {k: v for k, v in cfg["backend"]["env"].items() if k not in ("WANDB_MODE", "WANDB_RUN_GROUP")}
    env["MEM_BREAKDOWN"] = "1"
    out = {
        "defaults": ["../hybrid_speed/speed_base_mn5", "_self_"],
        "aux": {
            "experiment_group": "fixedvalid_eval",
            "suite": name,
            "arm": f"s{steps[0]}",
            "nodes": 4,
            "image": cfg["aux"].get("image", "ep"),
            "only": [],
        },
        "backend": {"env": env, "megatron": meg},
        "job": {
            "name": "fve_${aux.suite}_${aux.arm}",
            "base_output_dir": "/gpfs/scratch/ehpc533/hybrid_nucleus/fixedvalid_eval/${aux.suite}/${aux.arm}",
            "array_group": None,
            "metadata": {"pool_singleton": None, "pool_priority": 30},
            "est_step_time_s": 1.0,
            "est_startup_min": 40.0,
            "log_events": [],
            "state_events": [
                {"name": "give_up_failed", "transition": [None, "FAILED"],
                 "action": {"class_name": "CancelAction", "reason": "failed (eval: no retry)"}},
                {"name": "give_up_node_fail", "transition": [None, "NODE_FAIL"],
                 "action": {"class_name": "CancelAction", "reason": "lost its node (eval: no retry)"}},
            ],
        },
        "slurm": {"sbatch": {"nodes": 4, "account": "ehpc533", "qos": "acc_xehpc", "time": "02:00:00"}},
        "sweep": {
            "type": "product",
            "store_sweep_json": True,
            "filter": "\\${oc.eval:'len(\\${aux.only}) == 0 or \"\\${aux.arm}\" in \\${aux.only}'}",
            "groups": [{"type": "list", "configs": [
                {"aux.arm": f"s{s}", "backend.megatron.load": d,
                 "backend.megatron.ckpt_step": int(s), "backend.megatron.train_iters": int(s)}
                for s, d in marks]}],
        },
    }
    header = (
        "# @package _global_\n"
        f"# GENERATED by scripts/korbi/production/gen_fixedvalid_eval.py -- edit there.\n"
        f"# Fixed held-out validation sets at {'; '.join(f'{s} ({d})' for s, d in marks)}; the model and environment\n"
        f"# of the run's resolved config {os.environ.get("RESOLVED_SOURCE", resolved)} (keys not in the current schema dropped: {', '.join(dropped) or 'none'}).\n"
        "# Enqueue into a pool with spare nodes (4-node runs next to a main run):\n"
        "#   scripts/korbi/mn5_autoexp.sh scripts/run_autoexp.py --no-monitor --monitor-state-dir $R/pool_state \\\n"
        "#       --pool /gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p132 \\\n"
        f"#       --config-name experiments/korbi/production/fixedvalid_eval_{name}_mn5 --yes\n"
    )
    path = ROOT / f"config/experiments/korbi/production/fixedvalid_eval_{name}_mn5.yaml"
    path.write_text(header + yaml.safe_dump(out, sort_keys=False, width=110))
    print(f"Wrote {path} ({len(meg)} megatron keys; dropped {dropped})")


if __name__ == "__main__":
    main()
