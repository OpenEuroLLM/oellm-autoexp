"""Hydra/OmegaConf resolvers used by oellm_autoexp."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from math import sqrt as _sqrt
from collections.abc import Mapping

from omegaconf import ListConfig, OmegaConf


_REGISTRATION_SENTINEL = {"registered": False}


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
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def oc_if(a: str, b: str, c: str):
    if a:
        return b
    else:
        return c


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
    OmegaConf.register_new_resolver("oc.eval", eval, replace=True)  # noqa: S307
    OmegaConf.register_new_resolver("oc.if", oc_if, replace=True)

    _REGISTRATION_SENTINEL["registered"] = True


__all__ = ["register_default_resolvers"]
