"""Template rendering helpers shared by actions and conditions."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_PATTERN = re.compile(r"(?<!\$)\{([^\{\}\$:]+)\}")


def replace_braced_keys(s: str, values: Mapping[str, Any]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        return str(values[key]) if key in values else match.group(0)

    return _PATTERN.sub(repl, s)
