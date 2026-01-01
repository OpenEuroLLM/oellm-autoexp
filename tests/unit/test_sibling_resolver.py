"""Tests for sibling reference resolution."""

from oellm_autoexp.sweep.planner import JobPlan
from oellm_autoexp.sweep.sibling_resolver import resolve_sibling_references


def test_resolve_sibling_references_basic():
    """Test basic sibling reference resolution."""
    jobs = [
        JobPlan(
            name="lr1e-4_stable",
            parameters={"stage": "stable", "backend.megatron.lr": "1e-4"},
            output_dir="/outputs/lr1e-4_stable",
            log_path="/logs/lr1e-4_stable/slurm-%j.out",
        ),
        JobPlan(
            name="lr1e-4_cooldown",
            parameters={
                "stage": "cooldown",
                "backend.megatron.lr": "1e-4",
                "backend.megatron.load": "{sibling.stable.output_dir}/checkpoint",
            },
            output_dir="/outputs/lr1e-4_cooldown",
            log_path="/logs/lr1e-4_cooldown/slurm-%j.out",
        ),
    ]

    resolved_jobs = resolve_sibling_references(jobs)

    assert len(resolved_jobs) == 2
    # First job should be unchanged
    assert resolved_jobs[0].name == "lr1e-4_stable"
    assert resolved_jobs[0].parameters["stage"] == "stable"

    # Second job should have sibling reference resolved
    assert resolved_jobs[1].name == "lr1e-4_cooldown"
    assert (
        resolved_jobs[1].parameters["backend.megatron.load"] == "/outputs/lr1e-4_stable/checkpoint"
    )


def test_resolve_sibling_references_log_path_current():
    """Test that log_path_current is computed correctly."""
    jobs = [
        JobPlan(
            name="lr1e-4_stable",
            parameters={"stage": "stable"},
            output_dir="/outputs/lr1e-4_stable",
            log_path="/logs/lr1e-4_stable/slurm-%j.out",
        ),
    ]

    resolved_jobs = resolve_sibling_references(jobs)

    assert resolved_jobs[0].log_path_current == "/logs/lr1e-4_stable/current.log"


def test_resolve_sibling_references_start_condition():
    """Test resolving sibling references in start_condition_cmd."""
    jobs = [
        JobPlan(
            name="lr1e-4_stable",
            parameters={"stage": "stable"},
            output_dir="/outputs/lr1e-4_stable",
            log_path="/logs/lr1e-4_stable/slurm-%j.out",
        ),
        JobPlan(
            name="lr1e-4_cooldown",
            parameters={"stage": "cooldown"},
            output_dir="/outputs/lr1e-4_cooldown",
            log_path="/logs/lr1e-4_cooldown/slurm-%j.out",
            start_condition_cmd="check_status {sibling.stable.name}",
        ),
    ]

    resolved_jobs = resolve_sibling_references(jobs)

    assert resolved_jobs[1].start_condition_cmd == "check_status lr1e-4_stable"


def test_resolve_sibling_references_cancel_conditions():
    """Test resolving sibling references in cancel_conditions."""
    jobs = [
        JobPlan(
            name="lr1e-4_stable",
            parameters={"stage": "stable"},
            output_dir="/outputs/lr1e-4_stable",
            log_path="/logs/lr1e-4_stable/slurm-%j.out",
        ),
        JobPlan(
            name="lr1e-4_cooldown",
            parameters={"stage": "cooldown"},
            output_dir="/outputs/lr1e-4_cooldown",
            log_path="/logs/lr1e-4_cooldown/slurm-%j.out",
            cancel_conditions=[
                {
                    "class_name": "LogPatternCondition",
                    "log_path": "{sibling.stable.log_path_current}",
                    "pattern": "FATAL ERROR",
                },
            ],
        ),
    ]

    resolved_jobs = resolve_sibling_references(jobs)

    assert len(resolved_jobs[1].cancel_conditions) == 1
    assert resolved_jobs[1].cancel_conditions[0]["log_path"] == "/logs/lr1e-4_stable/current.log"


def test_resolve_sibling_references_multiple_siblings():
    """Test resolving references with multiple stages."""
    jobs = [
        JobPlan(
            name="lr1e-4_pre_training",
            parameters={"stage": "pre_training"},
            output_dir="/outputs/lr1e-4_pre_training",
            log_path="/logs/lr1e-4_pre_training/slurm-%j.out",
        ),
        JobPlan(
            name="lr1e-4_mid_training",
            parameters={
                "stage": "mid_training",
                "backend.megatron.load": "{sibling.pre_training.output_dir}/checkpoint",
            },
            output_dir="/outputs/lr1e-4_mid_training",
            log_path="/logs/lr1e-4_mid_training/slurm-%j.out",
        ),
        JobPlan(
            name="lr1e-4_post_training",
            parameters={
                "stage": "post_training",
                "backend.megatron.load": "{sibling.mid_training.output_dir}/checkpoint",
            },
            output_dir="/outputs/lr1e-4_post_training",
            log_path="/logs/lr1e-4_post_training/slurm-%j.out",
        ),
    ]

    resolved_jobs = resolve_sibling_references(jobs)

    assert (
        resolved_jobs[1].parameters["backend.megatron.load"]
        == "/outputs/lr1e-4_pre_training/checkpoint"
    )
    assert (
        resolved_jobs[2].parameters["backend.megatron.load"]
        == "/outputs/lr1e-4_mid_training/checkpoint"
    )


def test_resolve_sibling_references_multiple_hyperparameter_jobs():
    """Test that sibling resolution works correctly with multiple
    hyperparameter combinations."""
    jobs = [
        # LR 1e-4
        JobPlan(
            name="lr1e-4_bsz64_stable",
            parameters={
                "stage": "stable",
                "backend.megatron.lr": "1e-4",
                "backend.megatron.global_batch_size": "64",
            },
            output_dir="/outputs/lr1e-4_bsz64_stable",
            log_path="/logs/lr1e-4_bsz64_stable/slurm-%j.out",
        ),
        JobPlan(
            name="lr1e-4_bsz64_cooldown",
            parameters={
                "stage": "cooldown",
                "backend.megatron.lr": "1e-4",
                "backend.megatron.global_batch_size": "64",
                "backend.megatron.load": "{sibling.stable.output_dir}/checkpoint",
            },
            output_dir="/outputs/lr1e-4_bsz64_cooldown",
            log_path="/logs/lr1e-4_bsz64_cooldown/slurm-%j.out",
        ),
        # LR 5e-4
        JobPlan(
            name="lr5e-4_bsz64_stable",
            parameters={
                "stage": "stable",
                "backend.megatron.lr": "5e-4",
                "backend.megatron.global_batch_size": "64",
            },
            output_dir="/outputs/lr5e-4_bsz64_stable",
            log_path="/logs/lr5e-4_bsz64_stable/slurm-%j.out",
        ),
        JobPlan(
            name="lr5e-4_bsz64_cooldown",
            parameters={
                "stage": "cooldown",
                "backend.megatron.lr": "5e-4",
                "backend.megatron.global_batch_size": "64",
                "backend.megatron.load": "{sibling.stable.output_dir}/checkpoint",
            },
            output_dir="/outputs/lr5e-4_bsz64_cooldown",
            log_path="/logs/lr5e-4_bsz64_cooldown/slurm-%j.out",
        ),
    ]

    resolved_jobs = resolve_sibling_references(jobs)

    # Each cooldown should reference its corresponding stable job
    assert (
        resolved_jobs[1].parameters["backend.megatron.load"]
        == "/outputs/lr1e-4_bsz64_stable/checkpoint"
    )
    assert (
        resolved_jobs[3].parameters["backend.megatron.load"]
        == "/outputs/lr5e-4_bsz64_stable/checkpoint"
    )
