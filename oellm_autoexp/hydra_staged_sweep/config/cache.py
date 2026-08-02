"""In-memory caches that make repeated Hydra composition cheap.

Building a sweep composes the same config tree once per sweep point. Hydra
caches nothing between ``compose()`` calls, so every point re-reads and
re-parses the same YAML files, re-walks the same defaults list, re-merges the
same configs and re-parses the same interpolation strings.

Seven independent layers, each individually switchable:

``fast_yaml``
    Back OmegaConf's YAML loader with libyaml's C scanner (~14x faster
    parsing). No-op when PyYAML was built without libyaml.
``repo_cache``
    Keep parsed configs -- and the defaults lists derived from them -- in
    memory, keyed by file identity.
``lookup_cache``
    Remember which search path a config or group name resolves to, so repeated
    "is this a config group?" questions stop hitting the filesystem.
``compose_cache``
    Memoize merging a defaults list into a config. Sweep points that differ
    only in ``++key=value`` overrides share one merge.
``defaults_cache``
    Memoize the Defaults List itself, keyed on the overrides that can actually
    select a config group.
``parse_cache``
    Memoize OmegaConf's ANTLR parse trees for interpolation strings. A sweep
    typically parses a couple of dozen distinct strings thousands of times.
``override_cache``
    Memoize parsing of command-line override strings. Staged sweeps pass the
    whole sibling config down as hundreds of ``++key=value`` overrides per
    point, and Hydra builds a fresh ANTLR parser for each one.

Cache entries are revalidated against ``stat()`` on every lookup, so editing a
config on disk invalidates exactly the entries that depend on it. Composition
results are unchanged; this module only avoids repeating work.
"""

from __future__ import annotations

import copy
import logging
import os
import sys
from typing import Any

import yaml

LOGGER = logging.getLogger(__name__)

__all__ = ["clear", "disable", "enable", "reset_stats", "stats"]

_INSTALL_MARKER = "_hydra_staged_sweep_cache_installed"

_stats: dict[str, int] = {
    "repo_hit": 0,
    "repo_miss": 0,
    "repo_skip": 0,
    "compose_hit": 0,
    "compose_miss": 0,
    "parse_hit": 0,
    "parse_miss": 0,
    "override_hit": 0,
    "override_miss": 0,
    "lookup_hit": 0,
    "lookup_miss": 0,
    "defaults_hit": 0,
    "defaults_miss": 0,
}


def stats() -> dict[str, int]:
    """Hit/miss counters for each layer."""
    return dict(_stats)


def reset_stats() -> None:
    for key in _stats:
        _stats[key] = 0


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _fingerprint(path: str) -> tuple[int, int] | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _resolved_file(source: Any, config_path: str) -> str | None:
    """Absolute path backing ``config_path`` in ``source``, if file-backed."""
    if source.scheme() != "file":
        return None
    return os.path.realpath(os.path.join(source.path, source._normalize_file_name(config_path)))


# ---------------------------------------------------------------------------
# 1. libyaml
# ---------------------------------------------------------------------------
def _install_fast_yaml() -> bool:
    if not hasattr(yaml, "CSafeLoader"):
        LOGGER.debug("PyYAML built without libyaml; keeping the pure-Python loader")
        return False

    from omegaconf import _utils

    if getattr(_utils.get_yaml_loader, "_hss_fast", False):
        return True

    slow = _utils.get_yaml_loader()

    class OmegaConfCLoader(yaml.CSafeLoader):  # type: ignore[misc,valid-type]
        pass

    # Carry over OmegaConf's constructors and implicit resolvers so values are
    # typed exactly as before (it overrides timestamp and bool handling).
    OmegaConfCLoader.yaml_constructors = dict(slow.yaml_constructors)
    OmegaConfCLoader.yaml_multi_constructors = dict(slow.yaml_multi_constructors)
    OmegaConfCLoader.yaml_implicit_resolvers = {
        k: list(v) for k, v in slow.yaml_implicit_resolvers.items()
    }

    def get_yaml_loader() -> Any:
        return OmegaConfCLoader

    get_yaml_loader._hss_fast = True  # type: ignore[attr-defined]
    _utils.get_yaml_loader = get_yaml_loader
    import omegaconf.omegaconf as _oc

    if hasattr(_oc, "get_yaml_loader"):
        _oc.get_yaml_loader = get_yaml_loader
    return True


# ---------------------------------------------------------------------------
# 2. repository-level cache
# ---------------------------------------------------------------------------
# key -> (file fingerprint, ConfigResult, must_copy)
_repo_cache: dict[Any, tuple[tuple[int, int] | None, Any, bool]] = {}
_orig_repo_load: Any = None


def _has_matching_schema(repo: Any, config_path: str) -> bool:
    """True if a ConfigStore schema shares this config's name.

    Only then does Hydra's deprecated automatic schema matching run, and that
    is the one code path that mutates a loaded config in place (it pops
    ``hydra`` out of the primary config). Everywhere else the loaded config is
    treated as read-only -- Hydra's own per-compose ``CachingConfigRepository``
    already hands the same object to several callers -- so entries can be
    shared instead of copied.
    """
    from hydra.plugins.config_source import ConfigSource

    try:
        source = repo.get_schema_source()
        return bool(source.is_config(ConfigSource._normalize_file_name(config_path)))
    except Exception:  # noqa: BLE001 - any failure here means "copy, don't share"
        return True


def _install_repo_cache() -> None:
    global _orig_repo_load
    from hydra._internal.config_repository import ConfigRepository

    if _orig_repo_load is not None:
        return
    _orig_repo_load = ConfigRepository.load_config

    def load_config(self: ConfigRepository, config_path: str) -> Any:
        from hydra.core.object_type import ObjectType

        source = self._find_object_source(config_path, ObjectType.CONFIG)
        if source is None:
            return _orig_repo_load(self, config_path)

        # Structured configs live in the ConfigStore, can be mutated at runtime
        # and are cheap to build. Leave them alone.
        if source.scheme() == "structured":
            _stats["repo_skip"] += 1
            return _orig_repo_load(self, config_path)

        path = _resolved_file(source, config_path)
        fingerprint = _fingerprint(path) if path is not None else None
        if path is not None and fingerprint is None:
            return _orig_repo_load(self, config_path)  # vanished; let Hydra report it

        key = (config_path, source.scheme(), source.provider, source.path)
        entry = _repo_cache.get(key)
        if entry is not None and entry[0] == fingerprint:
            _stats["repo_hit"] += 1
            return copy.deepcopy(entry[1]) if entry[2] else entry[1]

        _stats["repo_miss"] += 1
        result = _orig_repo_load(self, config_path)
        must_copy = _has_matching_schema(self, config_path)
        _repo_cache[key] = (fingerprint, copy.deepcopy(result) if must_copy else result, must_copy)
        return result

    ConfigRepository.load_config = load_config  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 2a. lazy repository copy
# ---------------------------------------------------------------------------
_orig_caching_init: Any = None
_orig_caching_initialize_sources: Any = None


def _install_lazy_repo_copy() -> None:
    """Stop copying the whole repository on every composition.

    ``CachingConfigRepository`` deep-copies its delegate up front so that
    ``initialize_sources()`` cannot mutate the loader's shared repository. That
    only happens when a config overrides ``hydra.searchpath``, so defer the copy
    until it is actually needed.
    """
    global _orig_caching_init, _orig_caching_initialize_sources
    from hydra._internal.config_repository import CachingConfigRepository

    if _orig_caching_init is not None:
        return
    _orig_caching_init = CachingConfigRepository.__init__
    _orig_caching_initialize_sources = CachingConfigRepository.initialize_sources
    orig_initialize = _orig_caching_initialize_sources

    def __init__(self: CachingConfigRepository, delegate: Any) -> None:
        self.delegate = delegate
        self.cache = {}
        self._hss_owns_delegate = False

    def initialize_sources(self: CachingConfigRepository, config_search_path: Any) -> None:
        if not getattr(self, "_hss_owns_delegate", False):
            self.delegate = copy.deepcopy(self.delegate)
            self._hss_owns_delegate = True
        orig_initialize(self, config_search_path)

    CachingConfigRepository.__init__ = __init__  # type: ignore[assignment]
    CachingConfigRepository.initialize_sources = initialize_sources  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 2b. config-group lookup cache
# ---------------------------------------------------------------------------
# Resolving the defaults list asks "is this a config group?" once per override
# per sweep point, and each answer costs a realpath() plus a stat() on every
# search path. The answers only change if the config tree is edited on disk
# mid-run, which a sweep build does not do.
_lookup_cache: dict[Any, Any] = {}
_orig_find_source: Any = None


def _install_lookup_cache() -> None:
    global _orig_find_source
    from hydra._internal.config_repository import ConfigRepository

    if _orig_find_source is not None:
        return
    _orig_find_source = ConfigRepository._find_object_source

    def _find_object_source(self: ConfigRepository, config_path: str, object_type: Any) -> Any:
        key = (
            config_path,
            object_type,
            tuple((s.scheme(), s.provider, s.path) for s in self.sources),
        )
        if key in _lookup_cache:
            _stats["lookup_hit"] += 1
            index = _lookup_cache[key]
            # Cache the position, not the source object: sources are rebuilt
            # for every composition.
            return None if index is None else self.sources[index]
        _stats["lookup_miss"] += 1
        source = _orig_find_source(self, config_path, object_type)
        _lookup_cache[key] = None if source is None else self.sources.index(source)
        return source

    ConfigRepository._find_object_source = _find_object_source  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 3. composed-config cache
# ---------------------------------------------------------------------------
# key -> (contributing file fingerprints, composed DictConfig)
_compose_cache: dict[Any, tuple[tuple[Any, ...], Any]] = {}
_orig_compose: Any = None


# Structured configs are not file-backed, so a composition that merges one in
# cannot be revalidated by stat(). Bump a generation counter whenever the
# ConfigStore changes and key the cache on it instead.
_store_generation = [0]
_orig_store: Any = None


def _install_store_watch() -> None:
    global _orig_store
    from hydra.core.config_store import ConfigStore

    if _orig_store is not None:
        return
    _orig_store = ConfigStore.store

    def store(self: ConfigStore, *args: Any, **kwargs: Any) -> Any:
        _store_generation[0] += 1
        return _orig_store(self, *args, **kwargs)

    ConfigStore.store = store  # type: ignore[assignment]


def _defaults_key(defaults: list[Any], repo: Any) -> Any:
    return (
        tuple((d.config_path, d.parent, d.package, d.is_self, d.primary) for d in defaults),
        tuple((s.scheme(), s.provider, s.path) for s in repo.get_sources()),
        _store_generation[0],
    )


def _contributing_files(defaults: list[Any], repo: Any) -> tuple[Any, ...]:
    """Fingerprints of every file that can feed this composition."""
    found = set()
    sources = [s for s in repo.get_sources() if s.scheme() == "file"]
    for default in defaults:
        if default.config_path is None:
            continue
        for source in sources:
            path = _resolved_file(source, default.config_path)
            if path is None:
                continue
            fingerprint = _fingerprint(path)
            if fingerprint is not None:
                found.add((path, fingerprint))
    return tuple(sorted(found))


def _install_compose_cache() -> None:
    global _orig_compose
    from hydra._internal.config_loader_impl import ConfigLoaderImpl

    if _orig_compose is not None:
        return
    _orig_compose = ConfigLoaderImpl._compose_config_from_defaults_list

    def _compose(self: ConfigLoaderImpl, defaults: list[Any], repo: Any) -> Any:
        key = _defaults_key(defaults, repo)
        files = _contributing_files(defaults, repo)
        entry = _compose_cache.get(key)
        if entry is not None and entry[0] == files:
            _stats["compose_hit"] += 1
            # The caller mutates this (struct flag, overrides, hydra bookkeeping).
            return copy.deepcopy(entry[1])
        _stats["compose_miss"] += 1
        cfg = _orig_compose(self, defaults, repo)
        _compose_cache[key] = (files, copy.deepcopy(cfg))
        return cfg

    ConfigLoaderImpl._compose_config_from_defaults_list = _compose  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 3b. defaults-list cache
# ---------------------------------------------------------------------------
# Walking the Defaults List means loading every config in the tree to read its
# own defaults and package header. Only overrides that select a config group
# can change the outcome; the ``++key=value`` overrides a sweep varies per
# point cannot, so every point in a sweep rebuilds the same list.
_defaults_list_cache: dict[Any, tuple[tuple[Any, ...], Any]] = {}
_orig_create_defaults_list: Any = None


def _install_defaults_list_cache() -> None:
    global _orig_create_defaults_list
    from hydra._internal import config_loader_impl
    from hydra._internal import defaults_list as defaults_list_module
    from hydra._internal.defaults_list import DefaultsList, Overrides

    if _orig_create_defaults_list is not None:
        return
    _orig_create_defaults_list = defaults_list_module.create_defaults_list

    def create_defaults_list(
        repo: Any,
        config_name: str | None,
        overrides_list: list[Any],
        prepend_hydra: bool,
        skip_missing: bool,
    ) -> Any:
        # Overrides() is what decides group override vs value override, so build
        # it first and key the cache on the group-affecting ones only.
        overrides = Overrides(repo=repo, overrides_list=overrides_list)
        value_overrides = {id(override) for override in overrides.config_overrides}
        selecting = tuple(
            override.input_line
            for override in overrides_list
            if id(override) not in value_overrides
        )
        key = (
            config_name,
            prepend_hydra,
            skip_missing,
            selecting,
            tuple((s.scheme(), s.provider, s.path) for s in repo.get_sources()),
            _store_generation[0],
        )

        entry = _defaults_list_cache.get(key)
        if entry is not None:
            defaults, tree, known_choices, known_per_group = entry[1]
            if entry[0] == _contributing_files(defaults, repo):
                _stats["defaults_hit"] += 1
                # Rebuilt per call: these depend on every override, not just the
                # selecting ones. known_choices is filled by the tree walk we
                # just skipped, so restore it from the cached run.
                overrides.known_choices = dict(known_choices)
                overrides.known_choices_per_group = {
                    group: set(choices) for group, choices in known_per_group.items()
                }
                return DefaultsList(
                    defaults=defaults,
                    defaults_tree=tree,
                    config_overrides=overrides.config_overrides,
                    overrides=overrides,
                )

        _stats["defaults_miss"] += 1
        # A miss re-runs the real thing, including the validation that reports
        # unused overrides -- so a hit can only happen for a set that validated.
        result = _orig_create_defaults_list(
            repo, config_name, overrides_list, prepend_hydra, skip_missing
        )
        _defaults_list_cache[key] = (
            _contributing_files(result.defaults, repo),
            (
                result.defaults,
                result.defaults_tree,
                dict(result.overrides.known_choices),
                {
                    group: set(choices)
                    for group, choices in result.overrides.known_choices_per_group.items()
                },
            ),
        )
        return result

    defaults_list_module.create_defaults_list = create_defaults_list
    config_loader_impl.create_defaults_list = create_defaults_list


# ---------------------------------------------------------------------------
# 4. interpolation parse-tree cache
# ---------------------------------------------------------------------------
_parse_cache: dict[Any, Any] = {}
_orig_parse: Any = None


def _materialize(node: Any) -> None:
    """Force every token in a parse tree to hold its own text.

    antlr's ``CommonToken.text`` lazily reads back from the lexer's input
    stream, and OmegaConf swaps that stream out on the next parse. Reading the
    text once here pins it, which makes the tree safe to keep and reuse.
    """
    token = getattr(node, "symbol", None)
    if token is not None:
        token.text = token.text
    for attr in ("start", "stop"):
        token = getattr(node, attr, None)
        if token is not None and hasattr(token, "text"):
            token.text = token.text
    for child in getattr(node, "children", None) or ():
        _materialize(child)


def _install_parse_cache(maxsize: int = 4096) -> None:
    global _orig_parse
    from omegaconf import grammar_parser

    if _orig_parse is not None:
        return
    _orig_parse = grammar_parser.parse

    def parse(
        value: str, parser_rule: str = "configValue", lexer_mode: str = "DEFAULT_MODE"
    ) -> Any:
        key = (value, parser_rule, lexer_mode)
        tree = _parse_cache.get(key)
        if tree is not None:
            _stats["parse_hit"] += 1
            return tree
        _stats["parse_miss"] += 1
        tree = _orig_parse(value, parser_rule, lexer_mode)
        _materialize(tree)
        if len(_parse_cache) < maxsize:
            _parse_cache[key] = tree
        return tree

    grammar_parser.parse = parse
    # omegaconf.base and omegaconf._utils imported the symbol directly.
    for modname in ("omegaconf.base", "omegaconf._utils"):
        module = sys.modules.get(modname)
        if module is not None and getattr(module, "parse", None) is _orig_parse:
            module.parse = parse


# ---------------------------------------------------------------------------
# 5. override-parse cache
# ---------------------------------------------------------------------------
# Hydra builds a fresh ANTLR lexer and parser for *every* command-line
# override. A staged sweep passes the whole sibling config down as a few
# hundred ``++sibling.<stage>.a.b.c=value`` overrides per point, nearly all of
# them repeated across points.
_override_cache: dict[Any, Any] = {}
_orig_parse_rule: Any = None


def _functions_key(parser: Any) -> Any:
    """Identify the grammar function set, so custom functions can't collide."""
    key = getattr(parser, "_hss_functions_key", None)
    if key is None:
        try:
            key = tuple(sorted(parser.functions.definitions))
        except (AttributeError, TypeError):
            key = id(parser.functions)
        parser._hss_functions_key = key
    return key


def _install_override_cache(maxsize: int = 16384) -> None:
    global _orig_parse_rule
    from hydra.core.override_parser.overrides_parser import OverridesParser

    if _orig_parse_rule is not None:
        return
    _orig_parse_rule = OverridesParser.parse_rule

    def parse_rule(self: OverridesParser, s: str, rule_name: str) -> Any:
        key = (s, rule_name, _functions_key(self))
        cached = _override_cache.get(key)
        if cached is not None:
            _stats["override_hit"] += 1
            return copy.deepcopy(cached)
        _stats["override_miss"] += 1
        result = _orig_parse_rule(self, s, rule_name)
        if len(_override_cache) < maxsize:
            _override_cache[key] = copy.deepcopy(result)
        return result

    OverridesParser.parse_rule = parse_rule  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
_enabled = False


def enable(
    *,
    fast_yaml: bool = True,
    repo_cache: bool = True,
    compose_cache: bool = True,
    defaults_cache: bool = True,
    parse_cache: bool = True,
    override_cache: bool = True,
    lookup_cache: bool = True,
) -> None:
    """Install the caches. Idempotent; safe to call from every entry point.

    Set ``HYDRA_STAGED_SWEEP_CACHE=0`` to turn the whole thing off without
    touching call sites.
    """
    global _enabled
    if _enabled or os.environ.get("HYDRA_STAGED_SWEEP_CACHE", "1") == "0":
        return
    # A vendored copy of this module and an installed one would otherwise each
    # wrap Hydra, stacking a redundant layer on every call. Mark the target.
    import hydra

    if getattr(hydra, _INSTALL_MARKER, False):
        _enabled = True
        return
    setattr(hydra, _INSTALL_MARKER, True)
    if fast_yaml:
        _install_fast_yaml()
    if repo_cache:
        _install_repo_cache()
        _install_lazy_repo_copy()
    if lookup_cache:
        _install_lookup_cache()
    if compose_cache:
        _install_store_watch()
        _install_compose_cache()
    if defaults_cache:
        _install_store_watch()
        _install_defaults_list_cache()
    if parse_cache:
        _install_parse_cache()
    if override_cache:
        _install_override_cache()
    _enabled = True
    LOGGER.debug("hydra config caches enabled")


def disable() -> None:
    """Restore Hydra's and OmegaConf's original behaviour."""
    global _enabled, _orig_repo_load, _orig_compose, _orig_parse, _orig_parse_rule
    global _orig_find_source, _orig_store, _orig_create_defaults_list, _orig_caching_init
    if _orig_repo_load is not None:
        from hydra._internal.config_repository import ConfigRepository

        ConfigRepository.load_config = _orig_repo_load  # type: ignore[assignment]
        _orig_repo_load = None
    if _orig_caching_init is not None:
        from hydra._internal.config_repository import CachingConfigRepository

        CachingConfigRepository.__init__ = _orig_caching_init  # type: ignore[assignment]
        CachingConfigRepository.initialize_sources = (  # type: ignore[assignment]
            _orig_caching_initialize_sources
        )
        _orig_caching_init = None
    if _orig_find_source is not None:
        from hydra._internal.config_repository import ConfigRepository

        ConfigRepository._find_object_source = _orig_find_source  # type: ignore[assignment]
        _orig_find_source = None
    if _orig_compose is not None:
        from hydra._internal.config_loader_impl import ConfigLoaderImpl

        ConfigLoaderImpl._compose_config_from_defaults_list = _orig_compose  # type: ignore[assignment]
        _orig_compose = None
    if _orig_create_defaults_list is not None:
        from hydra._internal import config_loader_impl
        from hydra._internal import defaults_list as defaults_list_module

        defaults_list_module.create_defaults_list = _orig_create_defaults_list
        config_loader_impl.create_defaults_list = _orig_create_defaults_list
        _orig_create_defaults_list = None
    if _orig_store is not None:
        from hydra.core.config_store import ConfigStore

        ConfigStore.store = _orig_store  # type: ignore[assignment]
        _orig_store = None
    if _orig_parse is not None:
        from omegaconf import grammar_parser

        grammar_parser.parse = _orig_parse
        for modname in ("omegaconf.base", "omegaconf._utils"):
            module = sys.modules.get(modname)
            if module is not None:
                module.parse = _orig_parse
        _orig_parse = None
    if _orig_parse_rule is not None:
        from hydra.core.override_parser.overrides_parser import OverridesParser

        OverridesParser.parse_rule = _orig_parse_rule  # type: ignore[assignment]
        _orig_parse_rule = None
    import hydra

    if hasattr(hydra, _INSTALL_MARKER):
        delattr(hydra, _INSTALL_MARKER)
    clear()
    _enabled = False


def clear() -> None:
    """Drop every cached entry (the caches refill on the next composition)."""
    _repo_cache.clear()
    _compose_cache.clear()
    _parse_cache.clear()
    _override_cache.clear()
    _lookup_cache.clear()
    _defaults_list_cache.clear()
