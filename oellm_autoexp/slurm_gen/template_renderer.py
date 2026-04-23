"""Utilities to render SBATCH scripts from templates."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class SbatchTemplateError(RuntimeError):
    """Raised when template rendering fails."""

    pass


def render_template(template_text: str, replacements: Mapping[str, str]) -> str:
    """Render a template string with the given replacements.

    Args:
        template_text: Template string with {placeholder} variables.
        replacements: Mapping of placeholder names to values.

    Returns:
        Rendered string with placeholders replaced.

    Raises:
        SbatchTemplateError: If a required placeholder is missing.
    """
    try:
        return template_text.format(**replacements)
    except KeyError as exc:
        missing = exc.args[0]
        raise SbatchTemplateError(f"Missing template variable: {missing}") from exc


def render_template_file(
    template_path: str | Path,
    output_path: str | Path,
    replacements: Mapping[str, str],
) -> str:
    """Render a template file and write to output path.

    Args:
        template_path: Path to the template file.
        output_path: Path where the rendered script will be written.
        replacements: Mapping of placeholder names to values.

    Returns:
        The rendered template content.

    Raises:
        SbatchTemplateError: If a required placeholder is missing.
        FileNotFoundError: If template file doesn't exist.
    """
    template_path = Path(template_path)
    output_path = Path(output_path)

    template_text = template_path.read_text()
    rendered = render_template(template_text, replacements)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered)

    LOGGER.debug(f"Rendered template {template_path} to {output_path}")
    return rendered


__all__ = ["render_template", "render_template_file", "SbatchTemplateError"]
