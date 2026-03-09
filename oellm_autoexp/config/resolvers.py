"""Hydra/OmegaConf resolvers - re-exported from oellm_autoexp.hydra_staged_sweep.

This module now delegates to the well-tested resolvers in hydra_staged_sweep.
"""

from oellm_autoexp.hydra_staged_sweep.config.resolvers import register_default_resolvers, _safe_eval
import re
from omegaconf import OmegaConf, ListConfig, DictConfig


def oc_map_template(templ: str, inps: list[str]) -> list[str]:
    return [templ.replace("%", str(inp)) for inp in inps]


def oc_kvmap_template(templ: str, inps: dict[str, str]):
    OmegaConf.resolve(inps)
    return [templ.replace("%k", str(key)).replace("%v", str(val)) for key, val in inps.items()]


def oc_valuemap_template(templ: str, inps: dict[str, str]):
    OmegaConf.resolve(inps)
    return DictConfig({key: templ.replace("%v", str(val)) for key, val in inps.items()})


def oc_template(templ: str, inp: str) -> str:
    return templ.replace("%", inp)


def oc_join(concat: str, args: list[str]) -> str:
    return concat.join(map(str, args))


def oc_split(inp: str, split: str) -> ListConfig:
    return ListConfig([*inp.split(split)])


def oc_map_cond_template(cond: str, tmpl_if: str, tmpl_else: str, inps: list[str]) -> list[str]:
    return [tmpl_if.replace("%", inp) if re.match(cond, inp) else tmpl_else.replace("%", inp) for inp in inps]


def oc_map_eval(inps: ListConfig) -> ListConfig:
    return ListConfig([_safe_eval(inp) for inp in inps])


def _pad_iter(step):
    """Format a checkpoint step as Megatron's zero-padded directory name."""
    return f"iter_{int(step):07d}"


OmegaConf.register_new_resolver("oc.join", oc_join)
OmegaConf.register_new_resolver("oc.maptmpl", oc_map_template)
OmegaConf.register_new_resolver("oc.mapkeyvaltmpl", oc_kvmap_template)
OmegaConf.register_new_resolver("oc.mapvaltmpl", oc_valuemap_template)
OmegaConf.register_new_resolver("oc.mapcondtmpl", oc_map_cond_template)
OmegaConf.register_new_resolver("oc.mapeval", oc_map_eval)
OmegaConf.register_new_resolver("oc.tmpl", oc_template)
OmegaConf.register_new_resolver("oc.split", oc_split)

OmegaConf.register_new_resolver("oc.pad_iter", _pad_iter, replace=True)


__all__ = ["register_default_resolvers"]
