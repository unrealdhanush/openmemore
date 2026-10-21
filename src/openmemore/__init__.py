"""OpenMemore: one shared memory for all of a user's AI assistants (SPEC.md).

`from openmemore import Memory` gives programs the same memory, in-process (SPEC.md M7.5).
It's loaded on first use, so `openmemore serve` and the CLI don't pay for it.
"""

__version__ = "0.1.0"

__all__ = ["Hit", "Memory", "OpenMemoreError", "__version__"]


def __getattr__(name: str):
    if name in ("Memory", "Hit", "OpenMemoreError"):
        from openmemore import library

        return getattr(library, name)
    raise AttributeError(f"module 'openmemore' has no attribute {name!r}")
