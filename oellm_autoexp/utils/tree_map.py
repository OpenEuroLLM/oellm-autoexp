from typing import Any
from collections.abc import Callable


def tree_map(f: Callable[[Any], Any], obj: Any, is_leaf: Callable[[Any], bool] | None = None):
    if is_leaf is not None:
        if is_leaf(obj):
            return f(obj)
    if isinstance(obj, dict):
        return {key: tree_map(f, val, is_leaf=is_leaf) for key, val in obj.items()}
    if isinstance(obj, list):
        return [tree_map(f, val, is_leaf=is_leaf) for val in obj]
    if isinstance(obj, tuple):
        return tuple(tree_map(f, val, is_leaf=is_leaf) for val in obj)
    return f(obj)
