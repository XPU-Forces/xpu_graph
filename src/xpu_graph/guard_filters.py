"""
Backported guard_filter_fn from torch/compiler/__init__.py @torch >= 2.8
"""

__all__ = [
    "skip_all_guards_unsafe",
]


def skip_all_guards_unsafe(guard_entries):
    return [False for _ in guard_entries]
