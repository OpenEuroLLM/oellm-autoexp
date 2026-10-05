"""Profiling taxonomies."""

from collections.abc import Callable, Sequence

from .generic import classify_generic_kernel
from .megatron import classify_megatron_kernel
from .moe import classify_moe_kernel


Classifier = Callable[[str], str]
OptionalClassifier = Callable[[str], str | None]
_PLUGINS: dict[str, OptionalClassifier] = {
    "megatron": classify_megatron_kernel,
    "moe": classify_moe_kernel,
}


def build_classifier(taxonomies: Sequence[str] = ("megatron", "moe")) -> Classifier:
    unknown = set(taxonomies) - set(_PLUGINS) - {"generic"}
    if unknown:
        raise ValueError(f"Unknown profiling taxonomies: {sorted(unknown)}")
    plugins = [_PLUGINS[name] for name in taxonomies if name in _PLUGINS]

    def classify(name: str) -> str:
        for plugin in plugins:
            category = plugin(name)
            if category is not None:
                return category
        return classify_generic_kernel(name)

    return classify


classify_kernel = build_classifier()

__all__ = [
    "Classifier",
    "build_classifier",
    "classify_generic_kernel",
    "classify_kernel",
    "classify_megatron_kernel",
    "classify_moe_kernel",
]
