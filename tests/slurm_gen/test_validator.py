"""Tests for script validation functionality."""

import pytest

from oellm_autoexp.slurm_gen import validate_job_script, SlurmValidationError


class TestValidateJobScript:
    """Tests for validate_job_script function."""

    def test_valid_script(self, tmp_path):
        """Test validation of a valid script."""
        script = """#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --nodes=2

module load python/3.10
python train.py
"""
        script_file = tmp_path / "valid.sbatch"
        script_file.write_text(script)
        # Should not raise
        validate_job_script(str(script_file), "test_job")

    def test_missing_job_name(self, tmp_path):
        """Test that missing job name directive raises."""
        script = """#!/bin/bash
#SBATCH --nodes=2

python train.py
"""
        script_file = tmp_path / "missing_name.sbatch"
        script_file.write_text(script)
        with pytest.raises(SlurmValidationError, match="missing job name directive"):
            validate_job_script(str(script_file), "test_job")

    def test_wrong_job_name(self, tmp_path):
        """Test that wrong job name raises."""
        script = """#!/bin/bash
#SBATCH --job-name=other_job
"""
        script_file = tmp_path / "wrong_name.sbatch"
        script_file.write_text(script)
        with pytest.raises(SlurmValidationError, match="missing job name directive"):
            validate_job_script(str(script_file), "expected_job")

    def test_unreplaced_placeholder(self, tmp_path):
        """Test that unreplaced placeholders are detected."""
        script = """#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --nodes={nodes}

python train.py
"""
        script_file = tmp_path / "placeholder.sbatch"
        script_file.write_text(script)
        with pytest.raises(SlurmValidationError, match="Unreplaced template placeholder"):
            validate_job_script(str(script_file), "test_job")

    def test_unreplaced_placeholder_variations(self, tmp_path):
        """Test various placeholder formats are detected."""
        scripts = [
            "#SBATCH --job-name=test\n{simple_placeholder}",
            "#SBATCH --job-name=test\npath={PATH_VAR}",
            "#SBATCH --job-name=test\n{var123}",
            "#SBATCH --job-name=test\n{Var_With_Underscore}",
        ]
        for i, script in enumerate(scripts):
            script_file = tmp_path / f"placeholder_{i}.sbatch"
            script_file.write_text(script)
            with pytest.raises(SlurmValidationError, match="Unreplaced template placeholder"):
                validate_job_script(str(script_file), "test")

    def test_required_tokens_present(self, tmp_path):
        """Test validation with required tokens that are present."""
        script = """#!/bin/bash
#SBATCH --job-name=test_job

module load cuda/11.8
python train.py --config config.yaml
"""
        script_file = tmp_path / "tokens_present.sbatch"
        script_file.write_text(script)
        validate_job_script(
            str(script_file), "test_job", required_tokens=["module load", "python train.py"]
        )

    def test_required_token_missing(self, tmp_path):
        """Test that missing required token raises."""
        script = """#!/bin/bash
#SBATCH --job-name=test_job

python train.py
"""
        script_file = tmp_path / "token_missing.sbatch"
        script_file.write_text(script)
        with pytest.raises(SlurmValidationError, match="Required token 'module load'"):
            validate_job_script(str(script_file), "test_job", required_tokens=["module load"])

    def test_multiple_required_tokens_missing(self, tmp_path):
        """Test that first missing required token is reported."""
        script = """#!/bin/bash
#SBATCH --job-name=test_job

echo hello
"""
        script_file = tmp_path / "multi_missing.sbatch"
        script_file.write_text(script)
        with pytest.raises(SlurmValidationError, match="Required token 'module load'"):
            validate_job_script(
                str(script_file), "test_job", required_tokens=["module load", "python train.py"]
            )

    def test_empty_required_tokens(self, tmp_path):
        """Test that empty required tokens list is handled."""
        script = """#!/bin/bash
#SBATCH --job-name=test_job
"""
        script_file = tmp_path / "empty_tokens.sbatch"
        script_file.write_text(script)
        validate_job_script(str(script_file), "test_job", required_tokens=[])

    def test_none_required_tokens(self, tmp_path):
        """Test that None required tokens is handled."""
        script = """#!/bin/bash
#SBATCH --job-name=test_job
"""
        script_file = tmp_path / "none_tokens.sbatch"
        script_file.write_text(script)
        validate_job_script(str(script_file), "test_job", required_tokens=None)

    def test_complex_valid_script(self, tmp_path):
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
        script_file = tmp_path / "complex.sbatch"
        script_file.write_text(script)
        validate_job_script(
            str(script_file),
            "llm_training",
            required_tokens=["module load cuda", "srun python", "--config config.yaml"],
        )
