"""Utilities to render SBATCH scripts - re-exported from slurm_gen."""

from slurm_gen.template_renderer import (
    render_template,
    render_template_file,
    SbatchTemplateError,
)

__all__ = ["render_template", "render_template_file", "SbatchTemplateError"]
