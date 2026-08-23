"""Hydra/OmegaConf resolvers used by oellm_autoexp."""

from __future__ import annotations

import logging
import re
from datetime import datetime, UTC
from functools import lru_cache
from math import sqrt as _sqrt
from pathlib import Path
from collections.abc import Mapping

from omegaconf import DictConfig, ListConfig, OmegaConf

LOGGER = logging.getLogger(__name__)


_REGISTRATION_SENTINEL = {"registered": False}
_FORBIDDEN_EVAL_TOKENS = ("import", "open(", "input(")


def _safe_mul(*args):
    result = 1.0
    for arg in args:
        try:
            result *= float(arg)
        except (TypeError, ValueError):
            continue
    return result


def _safe_muli(*args):
    result = 1
    for arg in args:
        try:
            result *= int(arg)
        except (TypeError, ValueError):
            continue
    return result


def _safe_add(*args):
    total = 0.0
    for arg in args:
        try:
            total += float(arg)
        except (TypeError, ValueError):
            continue
    return total


def _safe_addi(*args):
    total = 0
    for arg in args:
        try:
            total += int(arg)
        except (TypeError, ValueError):
            continue
    return total


def _safe_sub(lhs, rhs):
    try:
        return float(lhs) - float(rhs)
    except (TypeError, ValueError):
        return 0.0


def _safe_subi(lhs, rhs):
    try:
        return int(lhs) - int(rhs)
    except (TypeError, ValueError):
        return 0


def _safe_div(lhs, rhs):
    if isinstance(lhs, ListConfig):
        if isinstance(rhs, ListConfig):
            return ListConfig([_safe_div(a, b) for a, b in zip(lhs, rhs)])
        return ListConfig([_safe_div(a, rhs) for a in lhs])
    try:
        return float(lhs) / float(rhs)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _safe_divi(lhs, rhs):
    if isinstance(lhs, ListConfig):
        if isinstance(rhs, ListConfig):
            return ListConfig([_safe_divi(a, b) for a, b in zip(lhs, rhs)])
        return ListConfig([_safe_divi(a, rhs) for a in lhs])
    try:
        return int(int(lhs) // int(rhs))
    except (TypeError, ValueError, ZeroDivisionError):
        return 0


def _ceil_div_int(lhs, rhs):
    try:
        lhs_i = int(lhs)
        rhs_i = int(rhs)
    except (TypeError, ValueError):
        return 0
    if rhs_i == 0:
        return 0
    return -(-lhs_i // rhs_i)


def _sqrt_wrapper(value):
    return float(_sqrt(float(value)))


def _slice(value, start, end):
    return value[int(start) : int(end)]  # noqa


def _mul_round_int(a, b, multiple):
    return int(round(float(a) * float(b) / float(multiple)) * float(multiple))


def _concat(lhs, rhs):
    return lhs + rhs


def _int_cast(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(bool(value))


def _dict_merge(*mappings):
    merged = {}
    for mapping in mappings:
        if mapping is None:
            continue
        if isinstance(mapping, Mapping):
            merged.update(mapping)
    return merged


@lru_cache
def _timestring():
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")[:-3]


def oc_join(concat: str, args: list[str]) -> str:
    return concat.join(map(str, args))


def oc_split(inp: str, split: str) -> ListConfig:
    return ListConfig([*inp.split(split)])


def oc_template(templ: str, inp: str) -> str:
    return templ.replace("%", inp)


def oc_map_template(templ: str, inps: list[str]) -> list[str]:
    return [templ.replace("%", str(inp)) for inp in inps]


def oc_keymap_template(key: str, templ: str, inps: list[str]) -> list[DictConfig]:
    return ListConfig([DictConfig({key: templ.replace("%", str(inp))}) for inp in inps])


def oc_kvmap_template(templ: str, inps: dict[str, str]):
    OmegaConf.resolve(inps)
    return ListConfig(
        [templ.replace("%k", str(key)).replace("%v", str(val)) for key, val in inps.items()]
    )


def oc_valuemap_template(templ: str, inps: dict[str, str]):
    OmegaConf.resolve(inps)
    return DictConfig({key: templ.replace("%v", str(val)) for key, val in inps.items()})


def oc_map_extract_key(key: str, inps: dict[str, str]):
    OmegaConf.resolve(inps)
    return ListConfig([d[key] for d in inps])


def oc_map_cond_template(cond: str, tmpl_if: str, tmpl_else: str, inps: list[str]) -> list[str]:
    return ListConfig(
        [
            tmpl_if.replace("%", inp) if re.match(cond, inp) else tmpl_else.replace("%", inp)
            for inp in inps
        ]
    )


def oc_map_eval(inps: ListConfig) -> ListConfig:
    return ListConfig([_safe_eval(inp) for inp in inps])


def oc_if(a: str | int | bool, b: str, c: str):
    if (isinstance(a, str) and a.lower() == "false") or not a:
        return c
    else:
        return b


def oc_coalesce(*tokens, _root_):
    """First of `tokens` that resolves to a non-None value. SQL COALESCE.

        est_steps: ${oc.coalesce:backend.megatron.exit_interval,backend.megatron.train_iters}
        est_steps: ${oc.coalesce:backend.megatron.exit_interval,1000}

    WHY oc.select CANNOT DO THIS. `oc.select:path,default` falls back only when
    the path is MISSING; a path that exists and holds None returns None.
    Measured on omegaconf 2.3.0:

        exit_interval: None  ->  ${oc.select:....exit_interval,${...train_iters}}  ==  None
        exit_interval absent ->  same expression                                   ==  894000

    Every field of a structured config EXISTS — `exit_interval: int | None = None`
    is present-and-None, never absent — so oc.select never fires its default on
    this repo's schemas, which is precisely the case you want to write.

    ARGUMENTS ARE PATHS, NOT INTERPOLATIONS, and that is deliberate: it makes
    this LAZY. Written as `${oc.coalesce:${a},${b}}` OmegaConf would resolve both
    arguments before calling us, so a `${b}` that is missing or itself invalid
    would raise even when `${a}` was perfectly good. Taking bare paths lets us
    stop at the first hit and never look at the rest.

    A token that does not resolve to anything is used as a LITERAL, so a plain
    default still works as the last argument. The cost of that convenience is
    that a string literal shaped like a config path resolves as one; pass such
    values from a config key instead. Returns None if nothing resolves, which
    keeps `${oc.coalesce:a,b}` safe to assign to an Optional field.
    """
    for token in tokens:
        if token is None:
            continue
        key = str(token).strip()
        if not key:
            continue
        value = OmegaConf.select(_root_, key, default=None)
        if value is not None:
            return value
        # Not a resolvable path (or resolved to None): treat as a literal, but
        # only if it cannot be a path at all — otherwise a null config key would
        # silently become the string "backend.megatron.exit_interval".
        if not _LOOKS_LIKE_PATH.match(key):
            return _coerce_scalar(key)
    return None


# A bare config path: dotted identifiers, optionally with [i] list indexing.
_LOOKS_LIKE_PATH = re.compile(r"^[A-Za-z_][\w]*(\[\d+\]|\.[A-Za-z_][\w]*)*$")


def _coerce_scalar(text: str):
    """Turn a literal token into int/float/bool/None where it obviously is
    one."""
    lowered = text.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def oc_eq(a: str | int | bool, b: str | int | bool):
    return a == b


def oc_neq(a: str | int | bool, b: str | int | bool):
    return a != b


def oc_gt(a: str | int | bool, b: str | int | bool):
    return a > b


def oc_lt(a: str | int | bool, b: str | int | bool):
    return a < b


def oc_geq(a: str | int | bool, b: str | int | bool):
    return a >= b


def oc_leq(a: str | int | bool, b: str | int | bool):
    return a <= b


def oc_slurmtime(seconds: int | float) -> str:
    seconds = int(seconds)
    sec = seconds % 60
    minutes = (seconds // 60) % 60
    hours = (seconds // 3600) % 24
    days = seconds // (3600 * 24)
    return f"{days}-{hours}:{minutes}:{sec}"


def oc_exclude_nodes(path: str, sep: str = ",") -> str | None:
    """Read a node-exclusion list file and return a SLURM nodelist string.

    The file is expected to hold one node name per line; blank lines and lines
    starting with ``#`` are ignored, and entries may also be comma- or
    whitespace-separated within a line. Duplicates are dropped while preserving
    first-seen order. The resulting nodes are joined with ``sep`` (default
    ``","``) so the value can be dropped straight into ``#SBATCH --exclude=``.

    A missing or effectively empty file yields ``None`` so the caller (e.g. the
    sbatch directive builder) omits the ``--exclude`` directive entirely rather
    than emitting an empty one.

    This is registered with ``use_cache=False`` so each config resolution / script
    generation re-reads the current on-disk list (the monitor grows it over time).
    """
    target = Path(str(path)).expanduser()
    if not target.exists():
        # WARN, do not fail. Returning None is correct — a site with no list
        # should not have an --exclude directive — but staying silent about it
        # is not: the list now lives OUTSIDE the repo (a shared project path on
        # JUPITER), so a moved, renamed or archived file, or an unmounted
        # /e/project1, turns into a 512-node job submitted with NO exclusions at
        # all, and nothing anywhere says so. A typo in the path looks identical
        # to "we deliberately have no list".
        LOGGER.warning(
            "oc.exclude_nodes: %s does not exist — no --exclude directive will be "
            "emitted for this job. If a list was expected, check the path or "
            "$JUPITER_EXCLUDE_NODES.",
            target,
        )
        return None
    nodes: list[str] = []
    seen: set[str] = set()
    for line in target.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for token in re.split(r"[,\s]+", line):
            token = token.strip()
            if token and token not in seen:
                seen.add(token)
                nodes.append(token)
    if not nodes:
        return None
    return sep.join(nodes)


def _validate_eval_expression(expr: str) -> None:
    normalized = expr.replace(" ", "")
    if any(token in normalized for token in _FORBIDDEN_EVAL_TOKENS):
        raise ValueError("oc.eval contains blocked token (import/open/input).")


def _safe_eval(expr: str):
    expr = str(expr)
    try:
        _validate_eval_expression(expr)
    except Exception as e:
        print(expr)
        raise e
    return eval(expr)


def register_default_resolvers(force: bool = False) -> None:
    """Register the resolvers if they have not already been registered."""

    if _REGISTRATION_SENTINEL["registered"] and not force:
        return

    OmegaConf.register_new_resolver("oc.mul", _safe_mul, replace=True)
    OmegaConf.register_new_resolver("oc.muli", _safe_muli, replace=True)
    OmegaConf.register_new_resolver("oc.add", _safe_add, replace=True)
    OmegaConf.register_new_resolver("oc.addi", _safe_addi, replace=True)
    OmegaConf.register_new_resolver("oc.sub", _safe_sub, replace=True)
    OmegaConf.register_new_resolver("oc.subi", _safe_subi, replace=True)
    OmegaConf.register_new_resolver("oc.div", _safe_div, replace=True)
    OmegaConf.register_new_resolver("oc.divi", _safe_divi, replace=True)
    OmegaConf.register_new_resolver("oc.cdivi", _ceil_div_int, replace=True)
    OmegaConf.register_new_resolver("oc.sqrt", _sqrt_wrapper, replace=True)
    OmegaConf.register_new_resolver("oc.slice", _slice, replace=True)
    OmegaConf.register_new_resolver("oc.mul_round_int", _mul_round_int, replace=True)
    OmegaConf.register_new_resolver("oc.concat", _concat, replace=True)
    OmegaConf.register_new_resolver("oc.int", _int_cast, replace=True, use_cache=False)
    OmegaConf.register_new_resolver("oc.dict_merge", _dict_merge, replace=True)
    OmegaConf.register_new_resolver("oc.timestring", lambda: _timestring(), replace=True)
    OmegaConf.register_new_resolver("oc.len", len, replace=True)
    OmegaConf.register_new_resolver("oc.eval", _safe_eval, replace=True)  # noqa: S307
    # use_cache=False: the whole point is to re-read the referenced keys, which
    # sweep arms and CLI overrides change between resolutions.
    OmegaConf.register_new_resolver("oc.coalesce", oc_coalesce, replace=True, use_cache=False)
    OmegaConf.register_new_resolver("oc.eq", oc_eq, replace=True)
    OmegaConf.register_new_resolver("oc.neq", oc_neq, replace=True)
    OmegaConf.register_new_resolver("oc.lt", oc_lt, replace=True)
    OmegaConf.register_new_resolver("oc.gt", oc_gt, replace=True)
    OmegaConf.register_new_resolver("oc.leq", oc_leq, replace=True)
    OmegaConf.register_new_resolver("oc.geq", oc_geq, replace=True)
    OmegaConf.register_new_resolver("oc.if", oc_if, replace=True)
    OmegaConf.register_new_resolver("oc.join", oc_join, replace=True)
    OmegaConf.register_new_resolver("oc.split", oc_split, replace=True)
    OmegaConf.register_new_resolver("oc.tmpl", oc_template, replace=True)
    OmegaConf.register_new_resolver("oc.maptmpl", oc_map_template, replace=True)
    OmegaConf.register_new_resolver("oc.mapkeytmpl", oc_keymap_template, replace=True)
    OmegaConf.register_new_resolver("oc.mapkeyvaltmpl", oc_kvmap_template, replace=True)
    OmegaConf.register_new_resolver("oc.mapvaltmpl", oc_valuemap_template, replace=True)
    OmegaConf.register_new_resolver("oc.mapextractkey", oc_map_extract_key, replace=True)
    OmegaConf.register_new_resolver("oc.mapcondtmpl", oc_map_cond_template, replace=True)
    OmegaConf.register_new_resolver("oc.mapeval", oc_map_eval, replace=True)
    OmegaConf.register_new_resolver("oc.slurmtime", oc_slurmtime, replace=True)
    OmegaConf.register_new_resolver(
        "oc.exclude_nodes", oc_exclude_nodes, replace=True, use_cache=False
    )

    _REGISTRATION_SENTINEL["registered"] = True


__all__ = ["register_default_resolvers"]
