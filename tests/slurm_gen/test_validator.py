"""Tests for script validation functionality."""

import pytest

from slurm_gen import validate_job_script, SlurmValidationError


class TestValidateJobScript:
    """Tests for validate_job_script function."""

    def test_valid_script(self):
        """Test validation of a valid script."""
        script = """#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --nodes=2

module load python/3.10
python train.py
"""
        # Should not raise
        validate_job_script(script, "test_job")

    def test_missing_job_name(self):
        """Test that missing job name directive raises."""
        script = """#!/bin/bash
#SBATCH --nodes=2

python train.py
"""
        with pytest.raises(SlurmValidationError, match="missing job name directive"):
            validate_job_script(script, "test_job")

    def test_wrong_job_name(self):
        """Test that wrong job name raises."""
        script = """#!/bin/bash
#SBATCH --job-name=other_job
"""
        with pytest.raises(SlurmValidationError, match="missing job name directive"):
            validate_job_script(script, "expected_job")

    def test_unreplaced_placeholder(self):
        """Test that unreplaced placeholders are detected."""
        script = """#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --nodes={nodes}

python train.py
"""
        with pytest.raises(SlurmValidationError, match="Unreplaced template placeholder"):
            validate_job_script(script, "test_job")

    def test_unreplaced_placeholder_variations(self):
        """Test various placeholder formats are detected."""
        scripts = [
            "#SBATCH --job-name=test\n{simple_placeholder}",
            "#SBATCH --job-name=test\npath={PATH_VAR}",
            "#SBATCH --job-name=test\n{var123}",
            "#SBATCH --job-name=test\n{Var_With_Underscore}",
        ]
        for script in scripts:
            with pytest.raises(SlurmValidationError, match="Unreplaced template placeholder"):
                validate_job_script(script, "test")

    def test_required_tokens_present(self):
        """Test validation with required tokens that are present."""
        script = """#!/bin/bash
#SBATCH --job-name=test_job

module load cuda/11.8
python train.py --config config.yaml
"""
        validate_job_script(script, "test_job", required_tokens=["module load", "python train.py"])

    def test_required_token_missing(self):
        """Test that missing required token raises."""
        script = """#!/bin/bash
#SBATCH --job-name=test_job

python train.py
"""
        with pytest.raises(SlurmValidationError, match="Required token 'module load'"):
            validate_job_script(script, "test_job", required_tokens=["module load"])

    def test_multiple_required_tokens_missing(self):
        """Test that first missing required token is reported."""
        script = """#!/bin/bash
#SBATCH --job-name=test_job

echo hello
"""
        with pytest.raises(SlurmValidationError, match="Required token 'module load'"):
            validate_job_script(
                script, "test_job", required_tokens=["module load", "python train.py"]
            )

    def test_empty_required_tokens(self):
        """Test that empty required tokens list is handled."""
        script = """#!/bin/bash
#SBATCH --job-name=test_job
"""
        validate_job_script(script, "test_job", required_tokens=[])

    def test_none_required_tokens(self):
        """Test that None required tokens is handled."""
        script = """#!/bin/bash
#SBATCH --job-name=test_job
"""
        validate_job_script(script, "test_job", required_tokens=None)

    def test_complex_valid_script(self):
        """Test validation of a complex but valid script."""
        script = """#!/bin/bash
#SBATCH --job-name=llm_training
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --account=myaccount

# Load modules
module load cuda/12.0
module load python/3.11

# Set environment
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Run training
srun python -m torch.distributed.launch train.py \\
    --config config.yaml \\
    --num-gpus 32 \\
    --output-dir /scratch/output
"""
        validate_job_script(
            script,
            "llm_training",
            required_tokens=["module load cuda", "srun python", "--config config.yaml"],
        )
