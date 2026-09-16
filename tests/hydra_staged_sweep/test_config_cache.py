"""The composition caches must not change what composition produces."""

import textwrap
from dataclasses import dataclass, field
from typing import Any

import pytest
from omegaconf import OmegaConf

from oellm_autoexp.hydra_staged_sweep.config import cache
from oellm_autoexp.hydra_staged_sweep.config.loader import load_hydra_config
from oellm_autoexp.hydra_staged_sweep.config.schema import StagedSweepRoot
from oellm_autoexp.hydra_staged_sweep.dag_resolver import config_to_cmdline, drop_cmdline_invisible


@dataclass(kw_only=True)
class CacheTestConfig(StagedSweepRoot):
    name: str = ""
    depth: int = 0
    label: str = ""
    nested: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@pytest.fixture
def config_dir(tmp_path):
    conf = tmp_path / "conf"
    (conf / "group").mkdir(parents=True)
    (conf / "group" / "a.yaml").write_text("# @package _global_\ndepth: 1\n")
    (conf / "group" / "b.yaml").write_text("# @package _global_\ndepth: 2\n")
    (conf / "config.yaml").write_text(
        textwrap.dedent("""\
            defaults:
              - group: a
              - _self_

            name: base
            label: "${name}-${depth}"
            nested:
              on: yes
              off: no
              when: 2020-01-01
              num: 1.0e-4
            """)
    )
    return conf


@pytest.fixture
def fresh_cache():
    """Each test starts from an empty, freshly installed set of caches."""
    cache.disable()
    cache.enable()
    cache.reset_stats()
    yield cache
    cache.disable()


def _load(config_dir, overrides=None):
    return load_hydra_config("config", config_dir, overrides or [], config_class=CacheTestConfig)


def test_cached_composition_matches_uncached(config_dir):
    cache.disable()
    uncached = [_load(config_dir, [f"++depth={i}"]) for i in range(4)]

    cache.enable()
    try:
        cached = [_load(config_dir, [f"++depth={i}"]) for i in range(4)]
    finally:
        cache.disable()

    assert [c.label for c in cached] == [u.label for u in uncached]
    assert [c.nested for c in cached] == [u.nested for u in uncached]


def test_group_overrides_are_not_conflated(config_dir, fresh_cache):
    """Different config groups must not share a cached composition."""
    assert _load(config_dir, ["group=a"]).depth == 1
    assert _load(config_dir, ["group=b"]).depth == 2
    assert _load(config_dir, ["group=a"]).depth == 1


def test_editing_a_config_invalidates_the_cache(config_dir, fresh_cache):
    assert _load(config_dir).name == "base"

    target = config_dir / "config.yaml"
    target.write_text(target.read_text().replace("name: base", "name: edited") + "\n# pad\n")

    assert _load(config_dir).name == "edited"


def test_caches_record_hits(config_dir, fresh_cache):
    for _ in range(3):
        _load(config_dir)
    stats = fresh_cache.stats()
    assert stats["repo_hit"] > 0
    assert stats["compose_hit"] > 0


def test_enable_is_idempotent(config_dir, fresh_cache):
    before = _load(config_dir).label
    fresh_cache.enable()
    fresh_cache.enable()
    assert _load(config_dir).label == before


def test_disable_restores_hydra(config_dir):
    cache.enable()
    _load(config_dir)
    cache.disable()
    assert cache.stats() is not None
    # Composition still works with the caches removed.
    assert _load(config_dir).name == "base"


def test_fast_yaml_types_match_the_pure_python_loader(tmp_path):
    """Libyaml must type scalars exactly as OmegaConf's own loader does."""
    sample = tmp_path / "s.yaml"
    sample.write_text("a: yes\nb: no\nc: null\nd: 2020-01-01\ne: 1.0e-4\nf: '010'\ng: 010\nh: ~\n")
    cache.disable()
    plain = OmegaConf.to_container(OmegaConf.load(sample))
    cache.enable()
    try:
        fast = OmegaConf.to_container(OmegaConf.load(sample))
    finally:
        cache.disable()
    assert fast == plain


def test_repeated_interpolations_resolve_identically(config_dir, fresh_cache):
    """The parse-tree cache hands out one shared tree; resolution must be
    stable."""
    first = _load(config_dir, ["++depth=7"]).label
    second = _load(config_dir, ["++depth=8"]).label
    assert (first, second) == ("base-7", "base-8")


@pytest.mark.parametrize(
    "payload",
    [
        {"empty": {}},
        {"outer": {"inner": {}}},
        {"kept": 1, "dropped": {}},
        {"items": [1, {}, {"b": 2}]},
        {"items": []},
        {"nothing": None},
        {"text": "value"},
        {"interp": "${name}"},
        {"quoted": 'has "quotes"'},
    ],
)
def test_extra_config_matches_the_override_round_trip(config_dir, fresh_cache, payload):
    """Merging the context directly must land exactly where the ++overrides
    do."""
    value = {"nested": payload}
    via_overrides = _load(config_dir, config_to_cmdline(value, override="++"))
    via_merge = load_hydra_config(
        "config",
        config_dir,
        [],
        config_class=CacheTestConfig,
        extra_config=drop_cmdline_invisible(value),
    )
    assert via_merge.nested == via_overrides.nested
