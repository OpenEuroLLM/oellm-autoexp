"""Validation helpers for SBATCH scripts - re-exported from slurm_gen."""

from slurm_gen.validator import validate_job_script, SlurmValidationError

__all__ = ["validate_job_script", "SlurmValidationError"]
