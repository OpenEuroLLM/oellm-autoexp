"""Hydra/OmegaConf resolvers used by oellm_autoexp."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from math import sqrt as _sqrt

from omegaconf import ListConfig, OmegaConf


_REGISTRATION_SENTINEL = {"registered": False}


def _safe_mul(*args):
    result = 1
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


def _sqrt_wrapper(value):
    return float(_sqrt(float(value)))


def _slice(value, start, end):
    return value[int(start) : int(end)]


def _floor_divide(lhs, rhs):
    if isinstance(lhs, ListConfig):
        if isinstance(rhs, ListConfig):
            return ListConfig([a // b for a, b in zip(lhs, rhs)])
        return ListConfig([a // rhs for a in lhs])
    return lhs // rhs


def _ceil_divide(lhs, rhs):
    lhs = int(lhs)
    rhs = int(rhs)
    return (lhs - 1) // rhs + 1


def _mul_round_int(a, b, multiple):
    return int(round(a * b / multiple) * multiple)


def _concat(lhs, rhs):
    return lhs + rhs


def _subi(lhs, rhs):
    return int(lhs) - int(rhs)


def _addi(lhs, rhs):
    return int(lhs) + int(rhs)


def _int_cast(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(bool(value))


@lru_cache
def _timestring():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def register_default_resolvers(force: bool = False) -> None:
    """Register the resolvers if they have not already been registered."""

    if _REGISTRATION_SENTINEL["registered"] and not force:
        return

    OmegaConf.register_new_resolver("oc.mul", _safe_mul, replace=True)
    OmegaConf.register_new_resolver("oc.muli", _safe_muli, replace=True)
    OmegaConf.register_new_resolver("oc.sqrt", _sqrt_wrapper, replace=True)
    OmegaConf.register_new_resolver("oc.slice", _slice, replace=True)
    OmegaConf.register_new_resolver("oc.floor_div", _floor_divide, replace=True)
    OmegaConf.register_new_resolver("oc.divi", _floor_divide, replace=True)
    OmegaConf.register_new_resolver("oc.ceil_div", _ceil_divide, replace=True)
    OmegaConf.register_new_resolver("oc.mul_round_int", _mul_round_int, replace=True)
    OmegaConf.register_new_resolver("oc.concat", _concat, replace=True)
    OmegaConf.register_new_resolver("oc.subi", _subi, replace=True)
    OmegaConf.register_new_resolver("oc.addi", _addi, replace=True)
    OmegaConf.register_new_resolver("oc.int", _int_cast, replace=True, use_cache=False)
    OmegaConf.register_new_resolver("oc.timestring", lambda: _timestring(), replace=True)
    OmegaConf.register_new_resolver("oc.len", len, replace=True)
    OmegaConf.register_new_resolver("eval", eval, replace=True)  # noqa: S307

    _REGISTRATION_SENTINEL["registered"] = True


__all__ = ["register_default_resolvers"]
