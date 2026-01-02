"""Utilities to render SBATCH scripts."""

from __future__ import annotations
import logging
from pathlib import Path

from collections.abc import Mapping

LOGGER = logging.getLogger(__name__)


class SbatchTemplateError(RuntimeError):
    pass


def render_template(template_text: str, replacements: Mapping[str, str]) -> str:
    try:
        return template_text.format(**replacements)
    except KeyError as exc:  # pragma: no cover
        missing = exc.args[0]
        raise SbatchTemplateError(f"Missing template variable: {missing}") from exc


def render_template_file(
    template_path: str, output_path: str, replacements: Mapping[str, str]
) -> str:
    template_text = Path(template_path).read_text()
    rendered = render_template(template_text, replacements)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered)
    return rendered


__all__ = ["render_template", "render_template_file", "SbatchTemplateError"]
